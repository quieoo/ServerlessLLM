#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/servegen_8_models.json}"
TRACE_BASE="${TRACE_BASE:-$REPO_ROOT/evaluation/traces/end2end/servegen_tangram_end2end.trace}"
TRACE_DIR="$(dirname "$TRACE_BASE")"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/evaluation/log/end2end}"
FIGURE_PATH="${FIGURE_PATH:-$REPO_ROOT/evaluation/log/end2end/end2end_comparison.pdf}"
SUMMARY_PATH="${SUMMARY_PATH:-$LOG_DIR/summary.csv}"

# RPS_LIST=(${RPS_LIST:-0.4 0.8 1.2 1.6 2.4 3.2 4.0})
# RPS_LIST=(${RPS_LIST:-0.8 1.6 2.4 3.2 4.0})
RPS_LIST=(${RPS_LIST:-0.4 0.8 1.2 1.6})

GPU_LIST=(${GPU_LIST:-1 2 4 8})
# GPU_LIST=(${GPU_LIST:-4})
SLO_SCALES="${SLO_SCALES:-1,2,3,4,5,6,7,8,9}"

USABLE_MEMORY="${USABLE_MEMORY:-40}"
MAX_REQUESTS="${MAX_REQUESTS:-0}"
WARMUP_STEP="${WARMUP_STEP:-0}"
KV_BATCH_SIZE="${KV_BATCH_SIZE:-1}"
LOAD_BANDWIDTH_GBPS="${LOAD_BANDWIDTH_GBPS:-10}"
LOAD_OVERHEAD_MS="${LOAD_OVERHEAD_MS:-0}"

BACKEND="${BACKEND:-}"
SLLM_BACKEND="${SLLM_BACKEND:-${BACKEND:-python}}"
TANGRAM_BACKEND="${TANGRAM_BACKEND:-${BACKEND:-cpp}}"

CPP_BACKEND_LIB="${CPP_BACKEND_LIB:-$REPO_ROOT/tools/mock_allocation/build/libtangram_vram_backend.so}"
CPP_LOAD_TIME_SOURCE="${CPP_LOAD_TIME_SOURCE:-estimated}"
CPP_FREE_STRATEGY="${CPP_FREE_STRATEGY:-1}"
CPP_ALLOCATE_STRATEGY="${CPP_ALLOCATE_STRATEGY:-4}"
CPP_GPU_BANDWIDTH_GBPS="${CPP_GPU_BANDWIDTH_GBPS:-400}"
CPP_CPU_BANDWIDTH_GBPS="${CPP_CPU_BANDWIDTH_GBPS:-20}"
CPP_REAL_COPY="${CPP_REAL_COPY:-0}"
SLLM_KEEP_ALIVE_MS="${SLLM_KEEP_ALIVE_MS:--1}"
ENGINE_OVERHEAD_MS="${ENGINE_OVERHEAD_MS:-500}"
PREFILL_MS_PER_TOKEN="${PREFILL_MS_PER_TOKEN:-0.02}"
DECODE_MS_PER_TOKEN="${DECODE_MS_PER_TOKEN:-0.2}"

# OCCUPY_UNTIL="${OCCUPY_UNTIL:-finish}"
OCCUPY_UNTIL="${OCCUPY_UNTIL:-ttft}"


SCHEDULER_POLICY="${SCHEDULER_POLICY:-online_ttft}"
GENERATE_TRACES="${GENERATE_TRACES:-0}"
PLOT="${PLOT:-1}"

rps_label() {
  local value="$1"
  echo "${value//./p}"
}

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

mkdir -p "$TRACE_DIR" "$LOG_DIR/per_request" "$LOG_DIR/ttft_cdf"
rm -f "$SUMMARY_PATH"

if [[ "$GENERATE_TRACES" != "0" ]]; then
  rps_csv="$(IFS=,; echo "${RPS_LIST[*]}")"
  python "$REPO_ROOT/evaluation/traces/generate_trace_tangram.py" \
    --config "$CONFIG_PATH" \
    --output "$TRACE_BASE" \
    --target-rps-list "$rps_csv" \
    --duration "${DURATION:-3600}" \
    --seed "${TRACE_SEED:-0}"
fi

for rps in "${RPS_LIST[@]}"; do
  label="$(rps_label "$rps")"
  trace_path="${TRACE_BASE%.trace}_${label}.trace"
  if [[ ! -f "$trace_path" ]]; then
    echo "Trace not found: $trace_path"
    exit 1
  fi

  for gpu_num in "${GPU_LIST[@]}"; do
    for system in sllm_cm tangram; do
      if [[ "$system" == "sllm_cm" ]]; then
        backend="$SLLM_BACKEND"
      else
        backend="$TANGRAM_BACKEND"
      fi
      request_csv="$LOG_DIR/per_request/${system}_rps${label}_gpu${gpu_num}.csv"
      ttft_cdf="$LOG_DIR/ttft_cdf/${system}_rps${label}_gpu${gpu_num}.csv"

      echo "=== simulate system=$system backend=$backend rps=$rps gpu=$gpu_num ==="
      simulate_args=(
        python "$REPO_ROOT/evaluation/end2end/simulate_end2end.py"
        --trace "$trace_path"
        --config "$CONFIG_PATH"
        --system "$system"
        --backend "$backend"
        --rps "$rps"
        --num-gpus "$gpu_num"
        --gpu-memory-gb "$USABLE_MEMORY"
        --slo-scales "$SLO_SCALES"
        --load-bandwidth-gbps "$LOAD_BANDWIDTH_GBPS"
        --load-overhead-ms "$LOAD_OVERHEAD_MS"
        --cpp-backend-lib "$CPP_BACKEND_LIB"
        --cpp-load-time-source "$CPP_LOAD_TIME_SOURCE"
        --cpp-free-strategy "$CPP_FREE_STRATEGY"
        --cpp-allocate-strategy "$CPP_ALLOCATE_STRATEGY"
        --cpp-gpu-bandwidth-gbps "$CPP_GPU_BANDWIDTH_GBPS"
        --cpp-cpu-bandwidth-gbps "$CPP_CPU_BANDWIDTH_GBPS"
        --sllm-keep-alive-ms "$SLLM_KEEP_ALIVE_MS"
        --engine-overhead-ms "$ENGINE_OVERHEAD_MS"
        --prefill-ms-per-token "$PREFILL_MS_PER_TOKEN"
        --decode-ms-per-token "$DECODE_MS_PER_TOKEN"
        --occupy-until "$OCCUPY_UNTIL"
        --scheduler-policy "$SCHEDULER_POLICY"
        --max-requests "$MAX_REQUESTS"
        --warmup-step "$WARMUP_STEP"
        --random-seed "${RANDOM_SEED:-1}"
        --output-request-csv "$request_csv"
        --output-summary-csv "$SUMMARY_PATH"
        --output-ttft-cdf "$ttft_cdf"
        --append-summary
      )
      if [[ "$CPP_REAL_COPY" != "0" ]]; then
        simulate_args+=(--cpp-real-copy)
      fi
      "${simulate_args[@]}"
    done
  done
done

if [[ "$PLOT" != "0" ]]; then
  python "$REPO_ROOT/evaluation/end2end/plot_end2end.py" \
    --summary "$SUMMARY_PATH" \
    --output "$FIGURE_PATH"
fi

echo "Summary: $SUMMARY_PATH"
echo "Figure: $FIGURE_PATH"
