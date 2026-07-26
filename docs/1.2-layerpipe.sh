#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/sdu/.conda/envs/sllm-worker/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/servegen_8_models_layerpipe.json}"
TRACE_PATH="${TRACE_PATH:-$REPO_ROOT/evaluation/traces/servegen_tangram.trace}"
MAX_REQUESTS="${MAX_REQUESTS:-100}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-0}"
OUTPUT_TOKENS_OVERRIDE="${OUTPUT_TOKENS_OVERRIDE:-0}"
TRACE_TIME_SCALE="${TRACE_TIME_SCALE:-1}"
INPUT_SCALE="${INPUT_SCALE:-1}"
GPU_ID="${GPU_ID:-0}"
LOAD_MODE="${LOAD_MODE:-layerpipe}"
if [[ -z "${KV_BACKEND+x}" ]]; then
  if [[ "$LOAD_MODE" == "vmm" || "$LOAD_MODE" == "layerweave" ]]; then
    KV_BACKEND="odkv"
  else
    KV_BACKEND="default"
  fi
fi
VMM_POOL_GIB="${VMM_POOL_GIB:-40}"
LAYERWEAVE_RUNTIME_RESERVE_GIB="${LAYERWEAVE_RUNTIME_RESERVE_GIB:-6.5}"
LAYERWEAVE_PREFETCH="${LAYERWEAVE_PREFETCH:-1}"
LAYERWEAVE_PREFIX_LAYERS="${LAYERWEAVE_PREFIX_LAYERS:-full}"
LAYERWEAVE_CACHE_POLICY="${LAYERWEAVE_CACHE_POLICY:-fixed}"
LAYERWEAVE_M4_PROFILE="${LAYERWEAVE_M4_PROFILE:-$REPO_ROOT/docs/m4-full/m4-full-estimator-report.json}"
LAYERWEAVE_DEMAND_DECAY="${LAYERWEAVE_DEMAND_DECAY:-0.9}"
LAYERWEAVE_UNCERTAINTY_MS="${LAYERWEAVE_UNCERTAINTY_MS:-100}"
LAYERWEAVE_MEMORY_DEBUG_MODEL_ID="${LAYERWEAVE_MEMORY_DEBUG_MODEL_ID:-}"
VMM_PAGE_SIZE_MIB="${VMM_PAGE_SIZE_MIB:-0}"
VMM_TENSOR_ONLY="${VMM_TENSOR_ONLY:-0}"
VMM_MERGE_TENSOR_GROUPS="${VMM_MERGE_TENSOR_GROUPS:-}"
# 0 means use each model's configured safe input limit. Set a positive value
# only when the experiment needs one uniform global context cap.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
TRUNCATE_INPUT_TO_MODEL_LIMIT="${TRUNCATE_INPUT_TO_MODEL_LIMIT:-1}"
OUTPUT="${OUTPUT:-$REPO_ROOT/docs/1.2-${LOAD_MODE}-${KV_BACKEND}-${MAX_REQUESTS}-${MAX_BATCH_SIZE}-output_tokens_override_${OUTPUT_TOKENS_OVERRIDE}-result.json}"
MODEL_PATH_OVERRIDES="${MODEL_PATH_OVERRIDES:-}"

echo "====== LayerPipe trace inference ======"
echo "gpu_id: $GPU_ID"
echo "load_mode: $LOAD_MODE"
echo "kv_backend: $KV_BACKEND"
echo "config: $CONFIG_PATH"
echo "trace: $TRACE_PATH"
echo "max_requests: $MAX_REQUESTS"
echo "max_batch_size: $MAX_BATCH_SIZE"
echo "output_tokens_override: $OUTPUT_TOKENS_OVERRIDE"
echo "trace_time_scale: $TRACE_TIME_SCALE"
echo "input_scale: $INPUT_SCALE"
echo "vmm_pool_gib: $VMM_POOL_GIB"
echo "layerweave_runtime_reserve_gib: $LAYERWEAVE_RUNTIME_RESERVE_GIB"
echo "layerweave_prefetch: $LAYERWEAVE_PREFETCH"
echo "layerweave_prefix_layers: $LAYERWEAVE_PREFIX_LAYERS"
echo "layerweave_cache_policy: $LAYERWEAVE_CACHE_POLICY"
echo "layerweave_m4_profile: $LAYERWEAVE_M4_PROFILE"
echo "layerweave_demand_decay: $LAYERWEAVE_DEMAND_DECAY"
echo "layerweave_uncertainty_ms: $LAYERWEAVE_UNCERTAINTY_MS"
echo "layerweave_memory_debug_model_id: ${LAYERWEAVE_MEMORY_DEBUG_MODEL_ID:-disabled}"
echo "vmm_page_size_mib: $VMM_PAGE_SIZE_MIB"
echo "vmm_tensor_only: $VMM_TENSOR_ONLY"
echo "vmm_merge_tensor_groups: ${VMM_MERGE_TENSOR_GROUPS:-disabled}"
echo "max_model_len: $MAX_MODEL_LEN"
echo "truncate_input_to_model_limit: $TRUNCATE_INPUT_TO_MODEL_LIMIT"
echo "output: $OUTPUT"

BENCHMARK="$REPO_ROOT/tools/layerpipe/layerpipe_bench.py"
if [[ ( "$LOAD_MODE" == "vmm" || "$LOAD_MODE" == "layerweave" ) && "$KV_BACKEND" == "odkv" ]]; then
  BENCHMARK="$REPO_ROOT/tools/layerpipe/vllm_odkv_trace_bench.py"
elif [[ "$LOAD_MODE" != "vmm" && "$LOAD_MODE" != "layerweave" && "$KV_BACKEND" != "default" ]]; then
  echo "KV_BACKEND=odkv is only supported with LOAD_MODE=vmm or layerweave"
  exit 1
elif [[ "$KV_BACKEND" != "default" ]]; then
  echo "KV_BACKEND must be default or odkv, got: $KV_BACKEND"
  exit 1
fi

cmd=(
  "$PYTHON"
  "$BENCHMARK"
  --config "$CONFIG_PATH"
  --trace "$TRACE_PATH"
  --device "$GPU_ID"
  --load-mode "$LOAD_MODE"
  --max-requests "$MAX_REQUESTS"
  --max-batch-size "$MAX_BATCH_SIZE"
  --output-tokens-override "$OUTPUT_TOKENS_OVERRIDE"
  --trace-time-scale "$TRACE_TIME_SCALE"
  --input-scale "$INPUT_SCALE"
  --vmm-pool-gib "$VMM_POOL_GIB"
  --vmm-page-size-mib "$VMM_PAGE_SIZE_MIB"
  --max-model-len "$MAX_MODEL_LEN"
  --output "$OUTPUT"
)

if [[ "$LOAD_MODE" == "layerweave" && "$KV_BACKEND" == "default" ]]; then
  cmd+=(--layerweave-runtime-reserve-gib "$LAYERWEAVE_RUNTIME_RESERVE_GIB")
fi

if [[ "$TRUNCATE_INPUT_TO_MODEL_LIMIT" == "1" ]]; then
  cmd+=(--truncate-input-to-model-limit)
elif [[ "$TRUNCATE_INPUT_TO_MODEL_LIMIT" != "0" ]]; then
  echo "TRUNCATE_INPUT_TO_MODEL_LIMIT must be 0 or 1, got: $TRUNCATE_INPUT_TO_MODEL_LIMIT"
  exit 1
fi
if [[ "$LAYERWEAVE_PREFETCH" != "0" && "$LAYERWEAVE_PREFETCH" != "1" ]]; then
  echo "LAYERWEAVE_PREFETCH must be 0 or 1, got: $LAYERWEAVE_PREFETCH"
  exit 1
fi
if [[ ! "$LAYERWEAVE_PREFIX_LAYERS" =~ ^(0|2|4|8|16|32|full)$ ]]; then
  echo "LAYERWEAVE_PREFIX_LAYERS must be one of 0,2,4,8,16,32,full"
  exit 1
fi
if [[ "$LAYERWEAVE_CACHE_POLICY" != "fixed" &&
      "$LAYERWEAVE_CACHE_POLICY" != "m4" &&
      "$LAYERWEAVE_CACHE_POLICY" != "joint" ]]; then
  echo "LAYERWEAVE_CACHE_POLICY must be fixed, m4, or joint"
  exit 1
fi
if [[ "$LAYERWEAVE_CACHE_POLICY" != "fixed" && "$LOAD_MODE" != "layerweave" ]]; then
  echo "Dynamic LayerWeave cache policies require LOAD_MODE=layerweave"
  exit 1
fi
if [[ ( "$LOAD_MODE" == "vmm" || "$LOAD_MODE" == "layerweave" ) && "$KV_BACKEND" == "odkv" ]]; then
  if [[ "$VMM_TENSOR_ONLY" != "0" || -n "$VMM_MERGE_TENSOR_GROUPS" ]]; then
    echo "VMM segmented ODKV uses the packed checkpoint grouping; tensor-only/merge flags are unsupported"
    exit 1
  fi
elif [[ "$VMM_TENSOR_ONLY" == "1" ]]; then
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

# Select the physical GPU before Python imports torch. Inside the process this
# one visible device is logical GPU 0, which is also what the VMM loader uses.
CUDA_VISIBLE_DEVICES="$GPU_ID" \
USE_GPU=0 \
LAYERWEAVE_PREFETCH="$LAYERWEAVE_PREFETCH" \
LAYERWEAVE_PREFIX_LAYERS="$LAYERWEAVE_PREFIX_LAYERS" \
LAYERWEAVE_CACHE_POLICY="$LAYERWEAVE_CACHE_POLICY" \
LAYERWEAVE_M4_PROFILE="$LAYERWEAVE_M4_PROFILE" \
LAYERWEAVE_DEMAND_DECAY="$LAYERWEAVE_DEMAND_DECAY" \
LAYERWEAVE_UNCERTAINTY_MS="$LAYERWEAVE_UNCERTAINTY_MS" \
"${cmd[@]}"
