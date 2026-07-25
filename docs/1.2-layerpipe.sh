#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/sdu/.conda/envs/sllm-worker/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/servegen_8_models.json}"
TRACE_PATH="${TRACE_PATH:-$REPO_ROOT/evaluation/traces/servegen_tangram.trace}"
MAX_REQUESTS="${MAX_REQUESTS:-100}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-0}"
OUTPUT_TOKENS_OVERRIDE="${OUTPUT_TOKENS_OVERRIDE:-0}"
TRACE_TIME_SCALE="${TRACE_TIME_SCALE:-1}"
GPU_ID="${GPU_ID:-0}"
LOAD_MODE="${LOAD_MODE:-layerpipe}"
VMM_POOL_GIB="${VMM_POOL_GIB:-40}"
VMM_PAGE_SIZE_MIB="${VMM_PAGE_SIZE_MIB:-0}"
VMM_TENSOR_ONLY="${VMM_TENSOR_ONLY:-0}"
VMM_MERGE_TENSOR_GROUPS="${VMM_MERGE_TENSOR_GROUPS:-}"
OUTPUT="${OUTPUT:-$REPO_ROOT/docs/1.2-${LOAD_MODE}-${MAX_REQUESTS}-${MAX_BATCH_SIZE}-output_tokens_override_${OUTPUT_TOKENS_OVERRIDE}-result.json}"
MODEL_PATH_OVERRIDES="${MODEL_PATH_OVERRIDES:-}"

echo "====== LayerPipe trace inference ======"
echo "gpu_id: $GPU_ID"
echo "load_mode: $LOAD_MODE"
echo "config: $CONFIG_PATH"
echo "trace: $TRACE_PATH"
echo "max_requests: $MAX_REQUESTS"
echo "max_batch_size: $MAX_BATCH_SIZE"
echo "output_tokens_override: $OUTPUT_TOKENS_OVERRIDE"
echo "trace_time_scale: $TRACE_TIME_SCALE"
echo "vmm_pool_gib: $VMM_POOL_GIB"
echo "vmm_page_size_mib: $VMM_PAGE_SIZE_MIB"
echo "vmm_tensor_only: $VMM_TENSOR_ONLY"
echo "vmm_merge_tensor_groups: ${VMM_MERGE_TENSOR_GROUPS:-disabled}"
echo "output: $OUTPUT"

cmd=(
  "$PYTHON"
  "$REPO_ROOT/tools/layerpipe/layerpipe_bench.py"
  --config "$CONFIG_PATH"
  --trace "$TRACE_PATH"
  --device "$GPU_ID"
  --load-mode "$LOAD_MODE"
  --max-requests "$MAX_REQUESTS"
  --max-batch-size "$MAX_BATCH_SIZE"
  --output-tokens-override "$OUTPUT_TOKENS_OVERRIDE"
  --trace-time-scale "$TRACE_TIME_SCALE"
  --vmm-pool-gib "$VMM_POOL_GIB"
  --vmm-page-size-mib "$VMM_PAGE_SIZE_MIB"
  --output "$OUTPUT"
)

if [[ "$VMM_TENSOR_ONLY" == "1" ]]; then
  cmd+=(--vmm-tensor-only)
elif [[ "$VMM_TENSOR_ONLY" != "0" ]]; then
  echo "VMM_TENSOR_ONLY must be 0 or 1, got: $VMM_TENSOR_ONLY"
  exit 1
fi
if [[ -n "$VMM_MERGE_TENSOR_GROUPS" ]]; then
  cmd+=(--vmm-merge-tensor-groups "$VMM_MERGE_TENSOR_GROUPS")
fi

# Space-separated MODEL_ID=/path entries, for example:
# MODEL_PATH_OVERRIDES="0=/mnt/n0/models/qwen2-3b 4=/mnt/n0/models/qwen2-14"
for override in $MODEL_PATH_OVERRIDES; do
  cmd+=(--model-path "$override")
done

"${cmd[@]}"
