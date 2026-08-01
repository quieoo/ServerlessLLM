#!/usr/bin/env bash
set -u -o pipefail

cd /mnt/n0/Tangram/Tangram || exit 1

PYTHON="${PYTHON:-/home/sdu/.conda/envs/sllm-worker/bin/python}"
REQUESTS="${REQUESTS:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/traces/sgrp-paper-r${REQUESTS}}"
SEARCH_ITERATIONS="${SEARCH_ITERATIONS:-30000}"
SEEDS=(1234 1235 1236 1237 1238)
COMMON_PREFIX=16
if ((REQUESTS >= 1000)); then
  COMMON_PREFIX=32
fi

mkdir -p "${OUTPUT_DIR}/reuse" "${OUTPUT_DIR}/input" "${OUTPUT_DIR}/size"

generate_reuse() {
  local seed="$1"
  local target="$2"
  local initialization="balanced"
  local prefix="${COMMON_PREFIX}"
  if ((target <= 3)); then
    initialization="min-distance"
    prefix=0
  fi
  "${PYTHON}" evaluation/traces/generate_target_trace.py \
    --config configs/servegen_8_models_layerpipe_l40_pool42.json \
    --source evaluation/traces/servegen_tangram.trace \
    --output "${OUTPUT_DIR}/reuse/d${target}-seed${seed}.trace" \
    --requests "${REQUESTS}" --sequence controlled-distinct \
    --controlled-count-policy balanced --controlled-objective target \
    --target-distinct-mean "${target}" \
    --common-prefix-requests "${prefix}" \
    --search-iterations "${SEARCH_ITERATIONS}" --lookahead-k 64 \
    --controlled-gap-constraint none \
    --controlled-initialization "${initialization}" \
    --fixed-input-tokens 256 --output-tokens 1 --seed "${seed}"
}

for seed in "${SEEDS[@]}"; do
  for target in 2 3 4 5; do
    generate_reuse "${seed}" "${target}" > /dev/null &
  done
  "${PYTHON}" evaluation/traces/generate_target_trace.py \
    --config configs/servegen_8_models_layerpipe_l40_pool42.json \
    --source evaluation/traces/servegen_tangram.trace \
    --output "${OUTPUT_DIR}/reuse/dmax-seed${seed}.trace" \
    --requests "${REQUESTS}" --sequence controlled-distinct \
    --controlled-count-policy balanced --controlled-objective max \
    --common-prefix-requests "${COMMON_PREFIX}" --search-iterations 20000 \
    --lookahead-k 64 --controlled-gap-constraint none \
    --controlled-initialization max-distance \
    --fixed-input-tokens 256 --output-tokens 1 --seed "${seed}" \
    > /dev/null &
  for input in 32 64 128 256; do
    "${PYTHON}" evaluation/traces/generate_target_trace.py \
      --config configs/servegen_8_models_layerpipe_l40_pool42.json \
      --source evaluation/traces/servegen_tangram.trace \
      --output "${OUTPUT_DIR}/input/i${input}-seed${seed}.trace" \
      --requests "${REQUESTS}" --sequence cyclic \
      --fixed-input-tokens "${input}" --output-tokens 1 --seed "${seed}" \
      > /dev/null &
  done
done
wait

"${PYTHON}" \
  evaluation/traces/generate_size_hotness_correlation_traces.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --source evaluation/traces/servegen_tangram.trace \
  --output-dir "${OUTPUT_DIR}/size" \
  --requests "${REQUESTS}" --source-requests 1000 \
  --fixed-input-tokens 256 --output-tokens 1 \
  --seeds 1234,1235,1236,1237,1238 --lookahead-k 64
