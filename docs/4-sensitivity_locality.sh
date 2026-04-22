#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY_PATH="$REPO_ROOT/tools/mock_allocation/build/Allocateion"
CONFIG_PATH="$REPO_ROOT/configs/servegen_8_models.json"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/evaluation/traces/forged_random}"
# TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/evaluation/traces/forged_small}"
# TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/evaluation/traces/forged_large}"

GPU_ID="${GPU_ID:-0}"
GPU_NUM="${GPU_NUM:-2}"
# 0 means use the whole trace. Override it when a quick smoke test is needed.
MAX_REQUESTS="${MAX_REQUESTS:-10000}"
MOCK_COPY="${MOCK_COPY:-1}"

# Locality sensitivity:
#   Read every .trace file in TRACE_DIR and keep Tangram settings fixed so the
#   only changing variable is the trace locality pattern.
TRACE_FILES=()

# Tangram reduced-IO settings.
ALLOCATE_STRATEGY="${ALLOCATE_STRATEGY:-4}"
FREE_STRATEGY="${FREE_STRATEGY:-1}"
REUSE_GRANULARITY="${REUSE_GRANULARITY:-1}"
USABLE_MEMORY="${USABLE_MEMORY:-40}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-1}"
KV_BATCH_SIZE="${KV_BATCH_SIZE:-1}"
WARMUP_STEP="${WARMUP_STEP:-0}"

LOAD_CDF_PATH="${LOAD_CDF_PATH:-}"

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

if [[ ! -d "$TRACE_DIR" ]]; then
  echo "Trace directory not found: $TRACE_DIR"
  exit 1
fi

while IFS= read -r trace_path; do
  TRACE_FILES+=("$trace_path")
done < <(find "$TRACE_DIR" -maxdepth 1 -type f -name "*.trace" | sort)

if [[ ${#TRACE_FILES[@]} -eq 0 ]]; then
  echo "No .trace files found in: $TRACE_DIR"
  exit 1
fi

for trace_path in "${TRACE_FILES[@]}"; do
  trace_file="$(basename "$trace_path")"
  label="${trace_file%.trace}"
  load_cdf_path=""

  if [[ -n "$LOAD_CDF_PATH" ]]; then
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

  echo "====== ServeGen locality sensitivity: $label ======"
  echo "trace_dir: $TRACE_DIR"
  echo "trace: $trace_path"
  echo "config: $CONFIG_PATH"
  echo "gpu_id: $GPU_ID"
  echo "gpu_num: $GPU_NUM"
  echo "usable_memory_gb: $USABLE_MEMORY"
  echo "allocate_strategy: $ALLOCATE_STRATEGY"
  echo "free_strategy: $FREE_STRATEGY"
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
    -p "$ALLOCATE_STRATEGY"
    -f "$FREE_STRATEGY"
    --gpu "$GPU_ID"
    --gpu_num "$GPU_NUM"
    --config "$CONFIG_PATH"
    --req_file_path "$trace_path"
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
