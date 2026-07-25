#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY_PATH="$REPO_ROOT/tools/mock_allocation/build/Allocateion"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/servegen_8_models.json}"
TRACE_PATH="${TRACE_PATH:-$REPO_ROOT/evaluation/traces/servegen_tangram.trace}"

GPU_ID="${GPU_ID:-0}"
GPU_NUM="${GPU_NUM:-2}"
# MAX_REQUESTS="${MAX_REQUESTS:-10}"
MAX_REQUESTS="${MAX_REQUESTS:-1000}"
USABLE_MEMORY=(${USABLE_MEMORY:-40})
SCHEDULE_POLICY=(${SCHEDULE_POLICY:-1})
# 0 uses the native minimum returned by cuMemGetAllocationGranularity.
VMM_PAGE_SIZE_MB="${VMM_PAGE_SIZE_MB:-0}"
# 1 splits checkpoint groups into tensor-sized file/copy units. The VMM backend
# still packs those units into one stable per-model VA arena, so tensors may
# share a physical page. 0 keeps checkpoint tensor-group file/copy units.
TENSOR_ONLY="${TENSOR_ONLY:-0}"

# KV blocks are derived from token lengths in the request trace. Set this to
# zero to replay only model weights. Weights and KV blocks share one VMM pool.
KV_BLOCK_FILE_PATH="${KV_BLOCK_FILE_PATH:-}"
KV_BATCH_SIZE="${KV_BATCH_SIZE:-1}"

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

if [[ -n "$KV_BLOCK_FILE_PATH" && ! -f "$KV_BLOCK_FILE_PATH" ]]; then
  echo "KV block file not found: $KV_BLOCK_FILE_PATH"
  exit 1
fi
if [[ "$TENSOR_ONLY" != "0" && "$TENSOR_ONLY" != "1" ]]; then
  echo "TENSOR_ONLY must be 0 or 1, got: $TENSOR_ONLY"
  exit 1
fi

for ((i = 0; i < ${#USABLE_MEMORY[@]}; i++)); do
  schedule_policy="${SCHEDULE_POLICY[i]:-${SCHEDULE_POLICY[0]}}"

  echo "====== ServeGen trace overall (VMM) ======"
  echo "trace: $TRACE_PATH"
  echo "config: $CONFIG_PATH"
  echo "gpu_id: $GPU_ID"
  echo "gpu_num: $GPU_NUM"
  echo "vmm_pool_size_gb: ${USABLE_MEMORY[i]}"
  echo "tensor_only: $TENSOR_ONLY"
  echo "schedule_policy: $schedule_policy"
  echo "vmm_page_size_mb: $VMM_PAGE_SIZE_MB"
  echo "max_requests: $MAX_REQUESTS"
  echo "kv_batch_size: $KV_BATCH_SIZE"
  echo "kv_block_file_path: ${KV_BLOCK_FILE_PATH:-disabled}"

  cmd=(
    "$BINARY_PATH"
    -g "${USABLE_MEMORY[i]}"
    -m 100
    --gpu "$GPU_ID"
    --gpu_num "$GPU_NUM"
    --config "$CONFIG_PATH"
    --req_file_path "$TRACE_PATH"
    --schedule_policy "$schedule_policy"
    --memory_backend vmm
    --vmm_page_size_mb "$VMM_PAGE_SIZE_MB"
    --kv_batch_size "$KV_BATCH_SIZE"
  )

  if [[ "$TENSOR_ONLY" == "1" ]]; then
    cmd+=(--tensor-only)
  fi
  if [[ "$MAX_REQUESTS" != "0" ]]; then
    cmd+=(--max_requests "$MAX_REQUESTS")
  fi
  if [[ -n "$KV_BLOCK_FILE_PATH" ]]; then
    cmd+=(--kv_block_file_path "$KV_BLOCK_FILE_PATH")
  fi

  "${cmd[@]}"
  echo "====== End ======"
done

echo "All done"
