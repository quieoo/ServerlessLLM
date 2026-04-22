#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MOCK_LOAD_PATH="$REPO_ROOT/evaluation/Aegaeon/mock_load.py"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/servegen_8_models.json}"
TRACE_PATH="${TRACE_PATH:-$REPO_ROOT/evaluation/traces/servegen_tangram.trace}"

GPU_NUM="${GPU_NUM:-2}"
MAX_REQUESTS="${MAX_REQUESTS:-1000}"
SEED="${SEED:-0}"
USABLE_MEMORY=(${USABLE_MEMORY:-40})
LOADER_THREADS="${LOADER_THREADS:-4}"
CHUNK_MB="${CHUNK_MB:-64}"
PREFETCH="${PREFETCH:-1}"
PREFETCH_PROB="${PREFETCH_PROB:-0.5}"

if [[ ! -f "$MOCK_LOAD_PATH" ]]; then
  echo "Python script not found: $MOCK_LOAD_PATH"
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

for usable_memory in "${USABLE_MEMORY[@]}"; do
  echo "====== Aegaeon mock load ======"
  echo "trace: $TRACE_PATH"
  echo "config: $CONFIG_PATH"
  echo "gpu_num: $GPU_NUM"
  echo "usable_memory_gb: $usable_memory"
  echo "max_requests: $MAX_REQUESTS"
  echo "loader_threads: $LOADER_THREADS"
  echo "chunk_mb: $CHUNK_MB"
  echo "prefetch: $PREFETCH"
  echo "prefetch_prob: $PREFETCH_PROB"
  echo "seed: $SEED"

  cmd=(
    "$PYTHON_BIN"
    "$MOCK_LOAD_PATH"
    --config "$CONFIG_PATH"
    --trace "$TRACE_PATH"
    --gpu-num "$GPU_NUM"
    --gpu-pool-gb "$usable_memory"
    --max-requests "$MAX_REQUESTS"
    --loader-threads "$LOADER_THREADS"
    --chunk-mb "$CHUNK_MB"
    --prefetch-prob "$PREFETCH_PROB"
    --seed "$SEED"
  )

  if [[ "$PREFETCH" != "0" ]]; then
    cmd+=(--prefetch)
  fi

  "${cmd[@]}"
  echo "====== End ======"
done

echo "All done"
