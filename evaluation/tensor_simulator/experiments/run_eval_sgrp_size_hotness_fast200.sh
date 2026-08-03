#!/usr/bin/env bash
set -u -o pipefail

cd /mnt/n0/Tangram/Tangram || exit 1

PYTHON="${PYTHON:-/home/sdu/.conda/envs/sllm-worker/bin/python}"
TRACE_DIR="${TRACE_DIR:-evaluation/traces/sgrp-size-hotness-fast200}"
OUTPUT_DIR="${OUTPUT_DIR:-doc/results/sgrp-size-hotness-fast200-k64}"
REQUESTS="${REQUESTS:-200}"

logical_cpus="$(nproc)"
physical_cores="$(
  lscpu -p=Core,Socket |
    awk -F, '!/^#/ {seen[$1 "," $2] = 1} END {print length(seen)}'
)"
physical_cores="${physical_cores:-${logical_cpus}}"
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
memory_slots=$((available_kib / 1024 / 1024 / 2))
auto_parallel="${physical_cores}"
if ((memory_slots < auto_parallel)); then
  auto_parallel="${memory_slots}"
fi
if ((auto_parallel < 1)); then
  auto_parallel=1
fi
MAX_PARALLEL="${MAX_PARALLEL:-${auto_parallel}}"

COMMON=(
  evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py
  --tensor-layout evaluation/tensor_simulator/results/tensor-layout.json
  --config configs/servegen_8_models_layerpipe_l40_pool42.json
  --pool-gib 42
  --max-requests "${REQUESTS}"
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
  --mckp-lookahead-k 64
  --mckp-lookahead-discount 0.9
  --mckp-transition-weight 0.1
)

LEVELS=(neg1 neg0p5 zero pos0p5 pos1)
SEEDS=(1234 1235 1236 1237 1238)
STRATEGIES=(cb-lru joint-mckp)
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_NAMES=()
status=0

run_one() {
  local level="$1"
  local seed="$2"
  local strategy="$3"
  local directory="${OUTPUT_DIR}/${level}/seed${seed}"
  local -a policy_args
  case "${strategy}" in
    cb-lru)
      policy_args=(--replacement-policy lru --routing-policy cache-bytes)
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
    --trace "${TRACE_DIR}/${level}-seed${seed}.trace" \
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
echo "logical_cpus=${logical_cpus} physical_cores=${physical_cores} "\
"memory_slots=${memory_slots} max_parallel=${MAX_PARALLEL}"
for level in "${LEVELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
      while ((${#ACTIVE_PIDS[@]} >= MAX_PARALLEL)); do
        reap_first
      done
      name="${level}/seed${seed}/${strategy}"
      echo "[START] ${name}"
      run_one "${level}" "${seed}" "${strategy}" &
      ACTIVE_PIDS+=("$!")
      ACTIVE_NAMES+=("${name}")
    done
  done
done
while ((${#ACTIVE_PIDS[@]})); do
  reap_first
done
exit "${status}"
