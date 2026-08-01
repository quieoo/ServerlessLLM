#!/usr/bin/env bash
set -u -o pipefail

cd /mnt/n0/Tangram/Tangram || exit 1

PYTHON="${PYTHON:-/home/sdu/.conda/envs/sllm-worker/bin/python}"
TRACE_DIR="${TRACE_DIR:-evaluation/traces/sgrp-reuse-distance-fast200-wide}"
OUTPUT_DIR="${OUTPUT_DIR:-doc/results/sgrp-reuse-distance-fast200-2d-k64}"
REQUESTS="${REQUESTS:-200}"

# The solver is CPU-bound and effectively single-threaded. Use one process per
# physical core, additionally capped by a conservative 2-GiB/process budget.
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
  tools/layerpipe/layerweave_tensor_pipeline_sim.py
  --tensor-layout docs/tensor-level-sim/tensor-layout.json
  --config configs/servegen_8_models_layerpipe_l40_pool42.json
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
    docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json
  --mckp-prediction-mode lookahead
  --mckp-lookahead-k 64
  --mckp-lookahead-discount 0.9
  --mckp-transition-weight 0.1
)

DISTANCES=(d2 d3 d4 d5 d6 d7)
# The largest model is 38.286 GiB, so smaller pools are infeasible.
POOLS=(39 40 42 46 50)
STRATEGIES=(cb-lru joint-mckp)
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_NAMES=()
status=0

run_one() {
  local distance="$1"
  local pool="$2"
  local strategy="$3"
  local directory="${OUTPUT_DIR}/${distance}/pool${pool}"
  local -a policy_args

  case "${strategy}" in
    cb-lru)
      policy_args=(
        --replacement-policy lru
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
    --trace "${TRACE_DIR}/${distance}.trace" \
    --pool-gib "${pool}" \
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
for distance in "${DISTANCES[@]}"; do
  for pool in "${POOLS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
      while ((${#ACTIVE_PIDS[@]} >= MAX_PARALLEL)); do
        reap_first
      done
      name="${distance}/pool${pool}/${strategy}"
      echo "[START] ${name}"
      run_one "${distance}" "${pool}" "${strategy}" &
      ACTIVE_PIDS+=("$!")
      ACTIVE_NAMES+=("${name}")
    done
  done
done

while ((${#ACTIVE_PIDS[@]})); do
  reap_first
done

exit "${status}"
