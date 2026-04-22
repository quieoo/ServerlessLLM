#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY_PATH="$REPO_ROOT/tools/mock_allocation/build/Allocateion"
CONFIG_PATH="$REPO_ROOT/configs/servegen_8_models.json"
TRACE_DIR="$REPO_ROOT/evaluation/traces"
TRACE_PREFIX="${TRACE_PREFIX:-servegen_tangram_alpha}"

GPU_ID="${GPU_ID:-0}"
GPU_NUM="${GPU_NUM:-2}"
# 0 means use the whole trace. Override it when a quick smoke test is needed.
MAX_REQUESTS="${MAX_REQUESTS:-0}"
MOCK_COPY="${MOCK_COPY:-1}"

# Mapping sensitivity:
#   alpha controls the ServeGen mapping locality in the input traces.
#   Keep Tangram settings fixed so the only changing variable is the trace.
ALPHAS=(${ALPHAS:-0 1 2 4 8})

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

for alpha in "${ALPHAS[@]}"; do
  trace_path="$TRACE_DIR/${TRACE_PREFIX}_${alpha}.trace"

  if [[ ! -f "$trace_path" ]]; then
    echo "Trace not found: $trace_path"
    echo "Generate it with docs/0.1-trace_gen.md first."
    exit 1
  fi
done

for alpha in "${ALPHAS[@]}"; do
  label="alpha-${alpha}"
  trace_path="$TRACE_DIR/${TRACE_PREFIX}_${alpha}.trace"
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

  echo "====== ServeGen mapping sensitivity: $label ======"
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
