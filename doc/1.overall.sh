#!/usr/bin/env bash
set -u -o pipefail

cd /mnt/n0/Tangram/Tangram || exit 1

COMMON=(
  tools/layerpipe/layerweave_tensor_pipeline_sim.py
  --tensor-layout docs/tensor-level-sim/tensor-layout.json
  --config configs/servegen_8_models_layerpipe_l40_pool42.json
  --trace evaluation/traces/servegen_tangram.trace
  --pool-gib 42
  --max-requests 5000
  --gpus 4
  --gpu-busy-probability 0.5
  --availability-seed 1234
  --trace-mode model-switches
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
)

OUTPUT_DIR="doc/results/end2end/corrected-busyp0p5-req5000"
PYTHON="/home/sdu/.conda/envs/sllm-worker/bin/python"

mkdir -p "${OUTPUT_DIR}/logs"

declare -A PIDS

run_group() {
  local name="$1"
  shift

  echo "Starting ${name}..."

  PYTHONUNBUFFERED=1 "${PYTHON}" "${COMMON[@]}" \
    --system "${name}" \
    "$@" \
    --output "${OUTPUT_DIR}/4gpu-busyp0p5-${name}.json" \
    --csv-output "${OUTPUT_DIR}/4gpu-busyp0p5-${name}.csv" \
    >"${OUTPUT_DIR}/logs/4gpu-busyp0p5-${name}.log" 2>&1 &

  PIDS["${name}"]=$!
}

run_group baseline

run_group pipe-only

run_group reuse-only \
  --replacement-policy lru \
  --routing-policy cache-bytes

run_group aegaeon

run_group tangram \
  --replacement-policy mckp-prefix \
  --routing-policy mckp-transition \
  --mckp-prediction-mode lookahead \
  --mckp-lookahead-k 32 \
  --mckp-lookahead-discount 0.9 \
  --mckp-transition-weight 0.1

echo
echo "All groups started:"
for name in "${!PIDS[@]}"; do
  echo "  ${name}: PID ${PIDS[$name]}"
done

status=0

for name in baseline pipe-only reuse-only aegaeon tangram; do
  if wait "${PIDS[$name]}"; then
    echo "[OK]     ${name}"
  else
    exit_code=$?
    echo "[FAILED] ${name}, exit code ${exit_code}"
    echo "         log: ${OUTPUT_DIR}/logs/4gpu-busyp0p5-${name}.log"
    status=1
  fi
done

if (( status == 0 )); then
  echo
  echo "All groups completed successfully."
  echo "Results: ${OUTPUT_DIR}"
else
  echo
  echo "One or more groups failed. Check the logs above."
fi

exit "${status}"