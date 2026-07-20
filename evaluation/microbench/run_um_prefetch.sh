#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BINARY_PATH="$REPO_ROOT/tools/mock_allocation/build/um_prefetch_bench"
CONFIG_PATH="$REPO_ROOT/configs/servegen_8_models.json"
TRACE_PATH="$REPO_ROOT/evaluation/traces/servegen_tangram.trace"
LOG_DIR="$REPO_ROOT/evaluation/log/um_prefetch"

GPU_ID="${GPU_ID:-0}"
GPU_NUM="${GPU_NUM:-2}"
MAX_REQUESTS="${MAX_REQUESTS:-1000}"
WARMUP_STEP="${WARMUP_STEP:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
PREPARE_THREADS="${PREPARE_THREADS:-64}"
VERBOSE="${VERBOSE:-0}"
GPU_ASSIGNMENT_MODE="${GPU_ASSIGNMENT_MODE:-fixed_model_owner}"
MODE_LIST_STR="${MODE_LIST:-um-prefetch}"
SCENARIO_LIST_STR="${SCENARIO_LIST:-warm-repeat}"
MAX_GPU_MEMORY_GB="${MAX_GPU_MEMORY_GB:-}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$BINARY_PATH" ]]; then
  echo "Binary not found: $BINARY_PATH"
  echo "Build it first:"
  echo "  cd $REPO_ROOT/tools/mock_allocation && cmake --build build --target um_prefetch_bench"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

if [[ ! -f "$TRACE_PATH" ]]; then
  echo "Trace not found: $TRACE_PATH"
  exit 1
fi

read -r -a MODE_LIST <<< "$MODE_LIST_STR"
read -r -a SCENARIO_LIST <<< "$SCENARIO_LIST_STR"

run_case() {
  local mode="$1"
  local scenario="$2"
  local label="${mode}-${scenario}"
  local cdf_path="$LOG_DIR/${label}.cdf"
  local req_csv="$LOG_DIR/${label}.requests.csv"
  local summary_csv="$LOG_DIR/${label}.summary.csv"

  echo "====== ServeGen trace UM baseline ======"
  echo "label: $label"
  echo "trace: $TRACE_PATH"
  echo "config: $CONFIG_PATH"
  echo "gpu_id: $GPU_ID"
  echo "gpu_num: $GPU_NUM"
  echo "mode: $mode"
  echo "scenario: $scenario"
  echo "max_requests: $MAX_REQUESTS"
  echo "warmup_step: $WARMUP_STEP"
  echo "progress_interval: $PROGRESS_INTERVAL"
  echo "prepare_threads: $PREPARE_THREADS"
  echo "gpu_assignment_mode: $GPU_ASSIGNMENT_MODE"
  if [[ -n "$MAX_GPU_MEMORY_GB" ]]; then
    echo "max_gpu_memory_gb: $MAX_GPU_MEMORY_GB"
  fi

  cmd=(
    "$BINARY_PATH"
    --config "$CONFIG_PATH"
    --req_file_path "$TRACE_PATH"
    --gpu "$GPU_ID"
    --gpu_num "$GPU_NUM"
    --mode "$mode"
    --scenario "$scenario"
    --warmup_step "$WARMUP_STEP"
    --progress_interval "$PROGRESS_INTERVAL"
    --prepare_threads "$PREPARE_THREADS"
    --gpu_assignment_mode "$GPU_ASSIGNMENT_MODE"
    --verbose "$VERBOSE"
    --load_cdf_path "$cdf_path"
    --request_csv "$req_csv"
    --summary_csv "$summary_csv"
  )

  if [[ "$MAX_REQUESTS" != "0" ]]; then
    cmd+=(--max_requests "$MAX_REQUESTS")
  fi
  if [[ -n "$MAX_GPU_MEMORY_GB" ]]; then
    cmd+=(--max_gpu_memory_gb "$MAX_GPU_MEMORY_GB")
  fi

  "${cmd[@]}"
  echo "====== End ======"
}

for mode in "${MODE_LIST[@]}"; do
  for scenario in "${SCENARIO_LIST[@]}"; do
    run_case "$mode" "$scenario"
  done
done
