#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BINARY_PATH="${BINARY_PATH:-$REPO_ROOT/tools/mock_allocation/build/memory_coalescing_bench}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/evaluation/log/memory_coalescing}"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-11008}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"
MEASURE_ITERS="${MEASURE_ITERS:-100}"
RANDOM_SEED="${RANDOM_SEED:-1}"
VERBOSE="${VERBOSE:-0}"

KERNEL_LIST_STR="${KERNEL_LIST:-attn_proj_gemm ffn_up_gemm decode_qk rmsnorm}"
PAGE_SIZE_KB_LIST_STR="${PAGE_SIZE_KB_LIST:-4 16 64 256 1024}"
read -r -a KERNEL_LIST <<< "$KERNEL_LIST_STR"
read -r -a PAGE_SIZE_KB_LIST <<< "$PAGE_SIZE_KB_LIST_STR"

SUMMARY_CSV="${SUMMARY_CSV:-$OUTPUT_DIR/summary.csv}"

if [[ ! -x "$BINARY_PATH" ]]; then
  echo "Binary not found: $BINARY_PATH"
  echo "Build it first:"
  echo "  cd $REPO_ROOT/tools/mock_allocation && cmake --build build --target memory_coalescing_bench"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$SUMMARY_CSV"

for kernel in "${KERNEL_LIST[@]}"; do
  tensor_label="tensor-level-${kernel}"
  echo "====== Memory Coalescing Bench ======"
  echo "label: $tensor_label"
  echo "kernel: $kernel"
  echo "layout: tensor-level"

  tensor_cmd=(
    "$BINARY_PATH"
    --gpu "$GPU_ID"
    --layout tensor-level
    --kernel "$kernel"
    --page_size_kb 64
    --batch_size "$BATCH_SIZE"
    --seq_len "$SEQ_LEN"
    --hidden_size "$HIDDEN_SIZE"
    --intermediate_size "$INTERMEDIATE_SIZE"
    --num_heads "$NUM_HEADS"
    --head_dim "$HEAD_DIM"
    --warmup_iters "$WARMUP_ITERS"
    --measure_iters "$MEASURE_ITERS"
    --random_seed "$RANDOM_SEED"
    --label "$tensor_label"
    --csv "$SUMMARY_CSV"
  )

  if [[ "$VERBOSE" == "1" ]]; then
    tensor_cmd+=(--verbose)
  fi

  "${tensor_cmd[@]}"

  for page_size_kb in "${PAGE_SIZE_KB_LIST[@]}"; do
    page_label="page-level-random-${kernel}-page${page_size_kb}kb"
    echo "label: $page_label"
    echo "kernel: $kernel"
    echo "layout: page-level-random"
    echo "page_size_kb: $page_size_kb"

    page_cmd=(
      "$BINARY_PATH"
      --gpu "$GPU_ID"
      --layout page-level-random
      --kernel "$kernel"
      --page_size_kb "$page_size_kb"
      --batch_size "$BATCH_SIZE"
      --seq_len "$SEQ_LEN"
      --hidden_size "$HIDDEN_SIZE"
      --intermediate_size "$INTERMEDIATE_SIZE"
      --num_heads "$NUM_HEADS"
      --head_dim "$HEAD_DIM"
      --warmup_iters "$WARMUP_ITERS"
      --measure_iters "$MEASURE_ITERS"
      --random_seed "$RANDOM_SEED"
      --label "$page_label"
      --csv "$SUMMARY_CSV"
    )

    if [[ "$VERBOSE" == "1" ]]; then
      page_cmd+=(--verbose)
    fi

    "${page_cmd[@]}"
  done
done

echo "Summary CSV: $SUMMARY_CSV"
echo "All done"
