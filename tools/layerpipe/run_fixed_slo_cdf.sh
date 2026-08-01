#!/usr/bin/env bash
set -euo pipefail

cd /mnt/n0/Tangram/Tangram

PYTHON=/home/sdu/.conda/envs/sllm-worker/bin/python
OUTPUT_DIR=doc/results/slo/fixed-r0p20-seed2234-output1-decode1-n1000
mkdir -p "${OUTPUT_DIR}/logs"

COMMON=(
  tools/layerpipe/layerweave_tensor_pipeline_sim.py
  --tensor-layout docs/tensor-level-sim/tensor-layout.json
  --config configs/servegen_8_models_layerpipe_l40_pool42.json
  --trace evaluation/traces/servegen_tangram.trace
  --pool-gib 42
  --max-requests 5000
  --warmup-requests 100
  --gpus 4
  --gpu-busy-probability 0
  --trace-mode all
  --input-scale 4
  --input-limit-policy safe
  --output-tokens-override 1
  --memory-layout segment
  --page-size-mib 8
  --h2d-gbps 24.56
  --tensor-group-min-mib 64
  --policy-suite minimal-only
  --mckp-stall-table-input
    docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json
  --execution-mode online
  --arrival-mode poisson
  --request-rate-rps 0.20
  --arrival-seed 2234
  --decode-ms-per-token 40
  --slo-scale 2
)

run_system() {
  local system=$1
  shift
  "${PYTHON}" "${COMMON[@]}" \
    --system "${system}" \
    "$@" \
    --output "${OUTPUT_DIR}/${system}.json" \
    --csv-output "${OUTPUT_DIR}/${system}-per-model.csv" \
    --requests-csv-output "${OUTPUT_DIR}/${system}-requests.csv" \
    >"${OUTPUT_DIR}/logs/${system}.log" 2>&1
}

pids=()
run_system baseline &
pids+=("$!")
run_system pipe-only &
pids+=("$!")
run_system reuse-only \
  --replacement-policy lru \
  --routing-policy cache-bytes &
pids+=("$!")
run_system aegaeon &
pids+=("$!")
run_system tangram \
  --replacement-policy mckp-prefix \
  --routing-policy mckp-transition \
  --mckp-prediction-mode lookahead \
  --mckp-lookahead-k 32 \
  --mckp-lookahead-discount 0.9 \
  --mckp-transition-weight 0.1 &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON}" tools/layerpipe/plot_online_ttft_cdf.py \
  --input-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/ttft-cdf.svg"
