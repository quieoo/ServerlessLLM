#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY_PATH="$REPO_ROOT/tools/mock_allocation/build/Allocateion"
CONFIG_PATH="$REPO_ROOT/configs/servegen_8_models.json"
TRACE_PATH="$REPO_ROOT/evaluation/traces/servegen_tangram.trace"

GPU_ID="${GPU_ID:-0}"
GPU_NUM="${GPU_NUM:-2}"
MAX_REQUESTS="${MAX_REQUESTS:-1000}"

# Strategy:
#   -p 0: w/o reuse
#   -p 1: global merge
#   -p 4: tensor reuse / Tangram policy
ALLOCATE_STRATEGIES=(${ALLOCATE_STRATEGIES:-0})
FREE_STRATEGIES=(${FREE_STRATEGIES:-1})
REUSE_GRANULARITY=(${REUSE_GRANULARITY:-1})
USABLE_MEMORY=(${USABLE_MEMORY:-40})
SCHEDULE_POLICY=(${SCHEDULE_POLICY:-1})
DISABLE_PARAMETER_REUSE="${DISABLE_PARAMETER_REUSE:-1}"

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
  echo "Generate it with docs/0.1-trace_gen.md first."
  exit 1
fi

for ((i = 0; i < ${#USABLE_MEMORY[@]}; i++)); do
  allocate_strategy="${ALLOCATE_STRATEGIES[i]:-${ALLOCATE_STRATEGIES[0]}}"
  free_strategy="${FREE_STRATEGIES[i]:-${FREE_STRATEGIES[0]}}"
  reuse_granularity="${REUSE_GRANULARITY[i]:-${REUSE_GRANULARITY[0]}}"
  schedule_policy="${SCHEDULE_POLICY[i]:-${SCHEDULE_POLICY[0]}}"

  echo "====== ServeGen trace overall ======"
  echo "trace: $TRACE_PATH"
  echo "config: $CONFIG_PATH"
  echo "gpu_id: $GPU_ID"
  echo "gpu_num: $GPU_NUM"
  echo "usable_memory_gb: ${USABLE_MEMORY[i]}"
  echo "allocate_strategy: $allocate_strategy"
  echo "free_strategy: $free_strategy"
  echo "reuse_granularity: $reuse_granularity"
  echo "schedule_policy: $schedule_policy"
  echo "max_requests: $MAX_REQUESTS"
  echo "disable_parameter_reuse: $DISABLE_PARAMETER_REUSE"

  cmd=(
    "$BINARY_PATH"
    -g "${USABLE_MEMORY[i]}"
    -m 100
    -p "$allocate_strategy"
    -f "$free_strategy"
    --gpu "$GPU_ID"
    --gpu_num "$GPU_NUM"
    --config "$CONFIG_PATH"
    --req_file_path "$TRACE_PATH"
    --reuse_granularity "$reuse_granularity"
    --schedule_policy "$schedule_policy"
  )

  if [[ "$DISABLE_PARAMETER_REUSE" != "0" ]]; then
    cmd+=(--disable_parameter_reuse)
  fi

  if [[ "$MAX_REQUESTS" != "0" ]]; then
    cmd+=(--max_requests "$MAX_REQUESTS")
  fi

  "${cmd[@]}"
  echo "====== End ======"
done

echo "All done"
