#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

BINARY_PATH="$REPO_ROOT/tools/mock_allocation/build/Allocateion"
CONFIG_PATH="$REPO_ROOT/configs/servegen_8_models.json"
TRACE_PATH="$REPO_ROOT/evaluation/traces/servegen_tangram.trace"

GPU_ID="${GPU_ID:-0}"
GPU_NUM="${GPU_NUM:-2}"
MAX_REQUESTS="${MAX_REQUESTS:-1000}"
MOCK_COPY="${MOCK_COPY:-1}"

# Breakdown stages:
#   1. model-level-reuse: model-granularity reuse, random GPU scheduling, pre-allocated KV
#   2. tensor-level-reuse: tensor-granularity reuse, random GPU scheduling, pre-allocated KV
#   3. odkv: tensor-granularity reuse plus on-demand KV allocation
#   4. affinity: odkv plus affinity-aware GPU scheduling
BREAKDOWN_LABELS=(${BREAKDOWN_LABELS:-model-level-reuse tensor-level-reuse odkv affinity})
ALLOCATE_STRATEGIES=(${ALLOCATE_STRATEGIES:-4 4 4 4})
FREE_STRATEGIES=(${FREE_STRATEGIES:-1 1 1 1})
REUSE_GRANULARITY=(${REUSE_GRANULARITY:-0 1 1 1})
USABLE_MEMORY=(${USABLE_MEMORY:-20 20 40 40})
SCHEDULE_POLICY=(${SCHEDULE_POLICY:-0 0 0 1})
KV_BATCH_SIZE=(${KV_BATCH_SIZE:-0 0 1 1})

if [[ ! -x "$BINARY_PATH" ]]; then
  echo "Binary not found: $BINARY_PATH"
  echo "Build it first:"
  echo "  cd $REPO_ROOT/tools/mock_allocation && cmake --build build --target Allocateion"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

if [[ ! -f "$TRACE_PATH" ]]; then
  echo "Trace not found: $TRACE_PATH"
  echo "Generate it with docs/legacy/reuse_only_tangram/0.1-trace_gen.md first."
  exit 1
fi

for ((i = 0; i < ${#BREAKDOWN_LABELS[@]}; i++)); do
  label="${BREAKDOWN_LABELS[i]}"
  allocate_strategy="${ALLOCATE_STRATEGIES[i]:-${ALLOCATE_STRATEGIES[0]}}"
  free_strategy="${FREE_STRATEGIES[i]:-${FREE_STRATEGIES[0]}}"
  reuse_granularity="${REUSE_GRANULARITY[i]:-${REUSE_GRANULARITY[0]}}"
  usable_memory="${USABLE_MEMORY[i]:-${USABLE_MEMORY[0]}}"
  schedule_policy="${SCHEDULE_POLICY[i]:-${SCHEDULE_POLICY[0]}}"
  kv_batch_size="${KV_BATCH_SIZE[i]:-${KV_BATCH_SIZE[0]}}"

  echo "====== ServeGen trace breakdown: $label ======"
  echo "trace: $TRACE_PATH"
  echo "config: $CONFIG_PATH"
  echo "gpu_id: $GPU_ID"
  echo "gpu_num: $GPU_NUM"
  echo "usable_memory_gb: $usable_memory"
  echo "allocate_strategy: $allocate_strategy"
  echo "free_strategy: $free_strategy"
  echo "reuse_granularity: $reuse_granularity"
  echo "schedule_policy: $schedule_policy"
  echo "kv_batch_size: $kv_batch_size"
  echo "max_requests: $MAX_REQUESTS"
  echo "mock_copy: $MOCK_COPY"

  cmd=(
    "$BINARY_PATH"
    -g "$usable_memory"
    -m 100
    -p "$allocate_strategy"
    -f "$free_strategy"
    --gpu "$GPU_ID"
    --gpu_num "$GPU_NUM"
    --config "$CONFIG_PATH"
    --req_file_path "$TRACE_PATH"
    --reuse_granularity "$reuse_granularity"
    --schedule_policy "$schedule_policy"
    --kv_batch_size "$kv_batch_size"
  )

  if [[ "$MOCK_COPY" != "0" ]]; then
    cmd+=(--mock_copy)
  fi

  if [[ "$MAX_REQUESTS" != "0" ]]; then
    cmd+=(--max_requests "$MAX_REQUESTS")
  fi

  "${cmd[@]}"
  echo "====== End ======"
done

echo "All done"
