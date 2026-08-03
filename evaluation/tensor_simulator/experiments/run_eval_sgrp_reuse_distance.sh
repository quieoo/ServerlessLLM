#!/usr/bin/env bash
set -u -o pipefail

cd /mnt/n0/Tangram/Tangram || exit 1

PYTHON="${PYTHON:-/home/sdu/.conda/envs/sllm-worker/bin/python}"
TRACE_DIR="${TRACE_DIR:-evaluation/traces/sgrp-reuse-distance-balanced}"
OUTPUT_DIR="${OUTPUT_DIR:-doc/results/sgrp-reuse-distance-balanced-busy0p5}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

COMMON=(
  evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py
  --tensor-layout evaluation/tensor_simulator/results/tensor-layout.json
  --config configs/servegen_8_models_layerpipe_l40_pool42.json
  --pool-gib 42
  --max-requests 1000
  --gpus 4
  --gpu-busy-probability 0.5
  --availability-seed 1234
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
  evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json
  --mckp-prediction-mode lookahead
  --mckp-lookahead-k 32
  --mckp-lookahead-discount 0.9
  --mckp-transition-weight 0.1
)

TRACES=(min mid-low mid-high max)
STRATEGIES=(cb-lru cb-suffix-lru cb-mckp joint-mckp)
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_NAMES=()
status=0

run_one() {
  local trace_name="$1"
  local strategy="$2"
  local trace="${TRACE_DIR}/${trace_name}.trace"
  local directory="${OUTPUT_DIR}/${trace_name}"
  local -a policy_args

  case "${strategy}" in
    cb-lru)
      policy_args=(
        --replacement-policy lru
        --routing-policy cache-bytes
      )
      ;;
    cb-suffix-lru)
      policy_args=(
        --replacement-policy lru-prefix
        --routing-policy cache-bytes
      )
      ;;
    cb-mckp)
      policy_args=(
        --replacement-policy mckp-prefix
        --routing-policy cache-bytes
      )
      ;;
    joint-mckp)
      policy_args=(
        --replacement-policy mckp-prefix
        --routing-policy mckp-transition
      )
      ;;
    *)
      echo "Unknown strategy: ${strategy}" >&2
      return 2
      ;;
  esac

  mkdir -p "${directory}/logs"
  PYTHONUNBUFFERED=1 "${PYTHON}" "${COMMON[@]}" \
    --trace "${trace}" \
    "${policy_args[@]}" \
    --output "${directory}/${strategy}.json" \
    >"${directory}/logs/${strategy}.log" 2>&1
}

reap_first() {
  local pid="${ACTIVE_PIDS[0]}"
  local name="${ACTIVE_NAMES[0]}"
  if wait "${pid}"; then
    echo "[OK] ${name}"
  else
    echo "[FAILED] ${name}" >&2
    status=1
  fi
  ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
  ACTIVE_NAMES=("${ACTIVE_NAMES[@]:1}")
}

mkdir -p "${OUTPUT_DIR}"
for trace_name in "${TRACES[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    while ((${#ACTIVE_PIDS[@]} >= MAX_PARALLEL)); do
      reap_first
    done
    echo "[START] ${trace_name}/${strategy}"
    run_one "${trace_name}" "${strategy}" &
    ACTIVE_PIDS+=("$!")
    ACTIVE_NAMES+=("${trace_name}/${strategy}")
  done
done

while ((${#ACTIVE_PIDS[@]})); do
  reap_first
done

exit "${status}"
