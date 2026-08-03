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
MOCK_COPY="${MOCK_COPY:-0}"

# Allocation policy analysis:
#   1. global-merge + random-drop
#   2. global-merge + greedy-drop
#   3. partitioned-bin-packing + greedy-drop
#
# Keep all other settings fixed so the only policy variables are
# ALLOCATE_STRATEGIES and FREE_STRATEGIES.
ANALYSIS_LABELS=(${ANALYSIS_LABELS:-global-merge-random-drop global-merge-greedy-drop pbp-greedy-drop})
ALLOCATE_STRATEGIES=(${ALLOCATE_STRATEGIES:-1 1 4})
FREE_STRATEGIES=(${FREE_STRATEGIES:-0 1 1})

# ALLOCATE_STRATEGIES=(${ALLOCATE_STRATEGIES:-1 4})
# FREE_STRATEGIES=(${FREE_STRATEGIES:-1 1})

REUSE_GRANULARITY="${REUSE_GRANULARITY:-1}"
USABLE_MEMORY="${USABLE_MEMORY:-40}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-1}"
KV_BATCH_SIZE="${KV_BATCH_SIZE:-1}"

WARMUP_STEP="${WARMUP_STEP:-0}"
LOAD_CDF_PATH="${LOAD_CDF_PATH:-"./{}-cdf.log"}"
LOAD_CDF_PATHS=(${LOAD_CDF_PATHS:-})

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

if [[ ${#ALLOCATE_STRATEGIES[@]} -ne ${#FREE_STRATEGIES[@]} ]]; then
  echo "ALLOCATE_STRATEGIES and FREE_STRATEGIES must have the same length."
  exit 1
fi

if [[ ${#LOAD_CDF_PATHS[@]} -ne 0 && ${#LOAD_CDF_PATHS[@]} -ne ${#ALLOCATE_STRATEGIES[@]} ]]; then
  echo "LOAD_CDF_PATHS must be empty or have the same length as ALLOCATE_STRATEGIES."
  exit 1
fi

for ((i = 0; i < ${#ALLOCATE_STRATEGIES[@]}; i++)); do
  label="${ANALYSIS_LABELS[i]:-alloc-${ALLOCATE_STRATEGIES[i]}-free-${FREE_STRATEGIES[i]}}"
  allocate_strategy="${ALLOCATE_STRATEGIES[i]:-${ALLOCATE_STRATEGIES[0]}}"
  free_strategy="${FREE_STRATEGIES[i]:-${FREE_STRATEGIES[0]}}"
  load_cdf_path=""
  if [[ ${#LOAD_CDF_PATHS[@]} -ne 0 ]]; then
    load_cdf_path="${LOAD_CDF_PATHS[i]}"
  elif [[ -n "$LOAD_CDF_PATH" ]]; then
    if [[ "$LOAD_CDF_PATH" == *"{}"* ]]; then
      load_cdf_path="${LOAD_CDF_PATH//\{\}/$label}"
    elif [[ -d "$LOAD_CDF_PATH" || "$LOAD_CDF_PATH" != *.* ]]; then
      load_cdf_path="${LOAD_CDF_PATH%/}/${label}.cdf"
    else
      cdf_dir="$(dirname "$LOAD_CDF_PATH")"
      cdf_file="$(basename "$LOAD_CDF_PATH")"
      cdf_ext=""
      cdf_stem="$cdf_file"
      if [[ "$cdf_file" == *.* ]]; then
        cdf_ext=".${cdf_file##*.}"
        cdf_stem="${cdf_file%.*}"
      fi
      load_cdf_path="${cdf_dir}/${cdf_stem}-${label}${cdf_ext}"
    fi
  fi

  echo "====== ServeGen trace allocation analysis: $label ======"
  echo "trace: $TRACE_PATH"
  echo "config: $CONFIG_PATH"
  echo "gpu_id: $GPU_ID"
  echo "gpu_num: $GPU_NUM"
  echo "usable_memory_gb: $USABLE_MEMORY"
  echo "allocate_strategy: $allocate_strategy"
  echo "free_strategy: $free_strategy"
  echo "reuse_granularity: $REUSE_GRANULARITY"
  echo "schedule_policy: $SCHEDULE_POLICY"
  echo "kv_batch_size: $KV_BATCH_SIZE"
  echo "warmup_step: $WARMUP_STEP"
  echo "max_requests: $MAX_REQUESTS"
  echo "mock_copy: $MOCK_COPY"
  echo "load_cdf_path: ${load_cdf_path:-none}"

  cmd=(
    "$BINARY_PATH"
    -g "$USABLE_MEMORY"
    -m 100
    -p "$allocate_strategy"
    -f "$free_strategy"
    --gpu "$GPU_ID"
    --gpu_num "$GPU_NUM"
    --config "$CONFIG_PATH"
    --req_file_path "$TRACE_PATH"
    --reuse_granularity "$REUSE_GRANULARITY"
    --schedule_policy "$SCHEDULE_POLICY"
    --kv_batch_size "$KV_BATCH_SIZE"
  )

  if [[ "$MOCK_COPY" != "0" ]]; then
    cmd+=(--mock_copy)
  fi

  if [[ "$MAX_REQUESTS" != "0" ]]; then
    cmd+=(--max_requests "$MAX_REQUESTS")
  fi

  if [[ "$WARMUP_STEP" != "0" ]]; then
    cmd+=(--warmup_step "$WARMUP_STEP")
  fi

  if [[ -n "$load_cdf_path" ]]; then
    cmd+=(--load_cdf_path "$load_cdf_path")
  fi

  "${cmd[@]}"
  echo "====== End ======"
done

echo "All done"
