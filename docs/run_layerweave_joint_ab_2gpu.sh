#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
MAX_REQUESTS="${MAX_REQUESTS:-100}"
SWAP_GPUS="${SWAP_GPUS:-0}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-5}"
PROFILE="${PROFILE:-docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json}"
PYTHON="${PYTHON:-/home/sdu/.conda/envs/sllm-worker/bin/python}"

run_one() {
  local gpu="$1"
  local policy="$2"
  local output="$3"
  shift 3
  env GPU_ID="$gpu" \
    CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
    TRACE_PATH=evaluation/traces/servegen_tangram.trace \
    LOAD_MODE=layerweave KV_BACKEND=odkv \
    VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
    MAX_REQUESTS="$MAX_REQUESTS" MAX_BATCH_SIZE=1 \
    INPUT_SCALE=4 OUTPUT_TOKENS_OVERRIDE=1 \
    TRACE_TIME_SCALE=150 LAYERWEAVE_PREFETCH=1 \
    LAYERWEAVE_CACHE_POLICY="$policy" \
    OUTPUT="$output" \
    "$@" \
    bash docs/1.2-layerpipe.sh > "${output%.json}.log" 2>&1
}

run_pair() {
  local round="$1"
  local fixed_gpu="$2"
  local joint_gpu="$3"
  local fixed="docs/joint-ab-r${round}-minimal-gpu${fixed_gpu}.json"
  local joint="docs/joint-ab-r${round}-joint-gpu${joint_gpu}.json"

  run_one "$fixed_gpu" fixed "$fixed" \
    LAYERWEAVE_PREFIX_LAYERS=full &
  local fixed_pid=$!
  # Concurrent 42 GiB cuMemCreate bursts can race in the NVIDIA driver even
  # when they target different GPUs. Stagger only pool initialization; the
  # trace replays still overlap for the performance comparison.
  sleep "$STARTUP_STAGGER_SECONDS"
  run_one "$joint_gpu" joint "$joint" \
    LAYERWEAVE_M4_PROFILE="$PROFILE" \
    LAYERWEAVE_DEMAND_DECAY=0.9 \
    LAYERWEAVE_UNCERTAINTY_MS=100 &
  local joint_pid=$!

  wait "$fixed_pid"
  wait "$joint_pid"
  "$PYTHON" tools/layerpipe/validate_layerweave_m5_5.py \
    --fixed "$fixed" \
    --dynamic "$joint" |
    tee "docs/joint-ab-r${round}-validation.txt"
}

run_pair 1 "$GPU_A" "$GPU_B"
if [[ "$SWAP_GPUS" == "1" ]]; then
  run_pair 2 "$GPU_B" "$GPU_A"
fi
