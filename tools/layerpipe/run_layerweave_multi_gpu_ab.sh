#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/sdu/.conda/envs/sllm-worker/bin/python}"
GPU_IDS="${GPU_IDS:-0,1}"
SCHEDULER_MODE="${SCHEDULER_MODE:-online}"
MAX_REQUESTS="${MAX_REQUESTS:-100}"
TRACE_TIME_SCALE="${TRACE_TIME_SCALE:-4}"
INPUT_SCALE="${INPUT_SCALE:-4}"
OUTPUT_TOKENS_OVERRIDE="${OUTPUT_TOKENS_OVERRIDE:-1}"
VMM_POOL_GIB="${VMM_POOL_GIB:-42}"
VMM_PAGE_SIZE_MIB="${VMM_PAGE_SIZE_MIB:-64}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-5}"
MAX_ASSIGNED_QUEUE_PER_GPU="${MAX_ASSIGNED_QUEUE_PER_GPU:-0}"
ROUTING_LOOKAHEAD="${ROUTING_LOOKAHEAD:-8}"
MAX_PLACEMENT_WAIT_MS="${MAX_PLACEMENT_WAIT_MS:-150}"
PLACEMENT_HYSTERESIS_MS="${PLACEMENT_HYSTERESIS_MS:-25}"
TRANSITION_WEIGHT="${TRANSITION_WEIGHT:-0.1}"
TRANSITION_CREDIT_CAP_MS="${TRANSITION_CREDIT_CAP_MS:-100}"
TRANSITION_PENALTY_CAP_MS="${TRANSITION_PENALTY_CAP_MS:-100}"
MINIMAL_COLD_TIE_RANDOM="${MINIMAL_COLD_TIE_RANDOM:-0}"
JOINT_COLD_TIE_RANDOM="${JOINT_COLD_TIE_RANDOM:-0}"
ALLOCATOR_TRIM_EVERY_REQUEST="${ALLOCATOR_TRIM_EVERY_REQUEST:-0}"
CACHE_POLICY_MODE="${CACHE_POLICY_MODE:-demand}"
BATCH_UNMAP="${BATCH_UNMAP:-0}"
PROFILE="${PROFILE:-tools/layerpipe/results/m4-m5.5-b1-scale4/m4-full-estimator-report.json}"
RUN_TAG="${RUN_TAG:-${MAX_REQUESTS}}"
RESULTS_DIR="${RESULTS_DIR:-tools/layerpipe/results/multi_gpu}"
mkdir -p "$RESULTS_DIR"
export LAYERWEAVE_BATCH_UNMAP="$BATCH_UNMAP"

COMMON=(
  "$PYTHON"
  tools/layerpipe/layerweave_multi_gpu_trace_bench.py
  --config configs/servegen_8_models_layerpipe_l40_pool42.json
  --trace evaluation/traces/servegen_tangram.trace
  --devices "$GPU_IDS"
  --profile "$PROFILE"
  --max-requests "$MAX_REQUESTS"
  --trace-time-scale "$TRACE_TIME_SCALE"
  --input-scale "$INPUT_SCALE"
  --output-tokens-override "$OUTPUT_TOKENS_OVERRIDE"
  --vmm-pool-gib "$VMM_POOL_GIB"
  --vmm-page-size-mib "$VMM_PAGE_SIZE_MIB"
  --startup-stagger-seconds "$STARTUP_STAGGER_SECONDS"
  --max-assigned-queue-per-gpu "$MAX_ASSIGNED_QUEUE_PER_GPU"
  --routing-lookahead "$ROUTING_LOOKAHEAD"
  --max-placement-wait-ms "$MAX_PLACEMENT_WAIT_MS"
  --placement-hysteresis-ms "$PLACEMENT_HYSTERESIS_MS"
  --transition-weight "$TRANSITION_WEIGHT"
  --transition-credit-cap-ms "$TRANSITION_CREDIT_CAP_MS"
  --transition-penalty-cap-ms "$TRANSITION_PENALTY_CAP_MS"
)
if [[ "$MINIMAL_COLD_TIE_RANDOM" == "1" ]]; then
  COMMON+=(--minimal-cold-tie-random)
fi
if [[ "$JOINT_COLD_TIE_RANDOM" == "1" ]]; then
  COMMON+=(--joint-cold-tie-random)
fi
if [[ "$ALLOCATOR_TRIM_EVERY_REQUEST" == "1" ]]; then
  COMMON+=(--allocator-trim-every-request)
fi

minimal_output="$RESULTS_DIR/layerweave-minimal-2gpu-${RUN_TAG}.json"
joint_output="$RESULTS_DIR/layerweave-joint-2gpu-${RUN_TAG}.json"

echo "Running minimal on GPUs ${GPU_IDS}"
"${COMMON[@]}" \
  --system-policy minimal \
  --cache-policy-mode demand \
  --scheduler-mode "$SCHEDULER_MODE" \
  --output "$minimal_output" \
  > "${minimal_output%.json}.log" 2>&1

echo "Running latest LayerWeave on GPUs ${GPU_IDS}"
"${COMMON[@]}" \
  --system-policy joint \
  --cache-policy-mode "$CACHE_POLICY_MODE" \
  --scheduler-mode "$SCHEDULER_MODE" \
  --output "$joint_output" \
  > "${joint_output%.json}.log" 2>&1

"$PYTHON" tools/layerpipe/validate_layerweave_multi_gpu_ab.py \
  --minimal "$minimal_output" \
  --joint "$joint_output" |
  tee "$RESULTS_DIR/layerweave-2gpu-${RUN_TAG}-validation.txt"
