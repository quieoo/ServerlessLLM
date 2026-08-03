#!/usr/bin/env python3
"""Run and validate the complete multi-model LayerWeave M4 matrix."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/sdu/.conda/envs/sllm-worker/bin/python")


def valid_result(path):
    if not path.exists():
        return False
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(document.get("batch_metrics"))


def run_logged(command, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"M4_RUN={' '.join(command)}", flush=True)
    with log_path.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if (
                "VMM_POLICY=" in line
                or "Traceback" in line
                or "Error" in line
            ):
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def summarize(report):
    print("SERVICE_TTFT_BY_RESIDENCY")
    for key, metrics in report["service_ttft_summary"].items():
        print(
            f"  {key:12s} count={metrics['count']:4d} "
            f"MAE={metrics['mae_ms']:8.2f}ms "
            f"P95={metrics['p95_absolute_error_ms']:8.2f}ms "
            f"MAPE={metrics['mape'] * 100:7.2f}%")
    print("SERVICE_TTFT_BY_MODEL")
    for key, metrics in report["service_ttft_by_model"].items():
        print(
            f"  model={key} count={metrics['count']:4d} "
            f"MAE={metrics['mae_ms']:8.2f}ms "
            f"P95={metrics['p95_absolute_error_ms']:8.2f}ms "
            f"MAPE={metrics['mape'] * 100:7.2f}%")
    print("SERVICE_TTFT_BY_BATCH_SIZE")
    for key, metrics in report["service_ttft_by_batch_size"].items():
        print(
            f"  batch={key} count={metrics['count']:4d} "
            f"MAE={metrics['mae_ms']:8.2f}ms "
            f"P95={metrics['p95_absolute_error_ms']:8.2f}ms "
            f"MAPE={metrics['mape'] * 100:7.2f}%")
    gain = report["overlap_gain_summary"]
    print(
        "PREFETCH_GAIN "
        f"count={gain['count']} MAE={gain['mae_ms']:.2f}ms "
        f"P95={gain['p95_absolute_error_ms']:.2f}ms "
        f"MAPE={gain['mape'] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "tools/layerpipe/results/m4-full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--compact-batches", action="store_true",
        help=(
            "Run the full matrix for batch 1, but for larger batches "
            "calibrate full-hit compute and evaluate one partial-hit point."
        ),
    )
    parser.add_argument(
        "--marginal-prefix-matrix", action="store_true",
        help=(
            "Measure every 0/2/4/8/16/32/full prefix tier across "
            "calibration and evaluation; intended for M5.5."),
    )
    args = parser.parse_args()
    if args.input_scale <= 0:
        raise ValueError("--input-scale must be positive")
    batch_sizes = [
        int(value) for value in args.batch_sizes.split(",") if value
    ]
    scale_tag = (
        "" if args.input_scale == 1.0
        else f"_scale{args.input_scale:g}".replace(".", "p")
    )
    config = (
        REPO_ROOT / "configs/servegen_8_models_layerpipe_l40_pool42.json"
    )
    trace_dir = REPO_ROOT / "evaluation/traces"
    subprocess.run(
        [
            str(PYTHON),
            str(REPO_ROOT / "tools/layerpipe/"
                "layerweave_m4_generate_traces.py"),
            "--config", str(config),
            "--output-dir", str(trace_dir),
            "--batch-sizes", args.batch_sizes,
            "--input-scale", str(args.input_scale),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    calibration = []
    evaluation = []
    for batch_size in batch_sizes:
        if args.marginal_prefix_matrix:
            matrix = (
                ("calibration", ("0", "4", "16", "full")),
                ("evaluation", ("2", "8", "32", "full")),
            )
        elif args.compact_batches and batch_size > 1:
            matrix = (
                ("calibration", ("full",)),
                ("evaluation", ("16",)),
            )
        else:
            matrix = (
                ("calibration", ("0", "8", "full")),
                ("evaluation", ("4", "16", "full")),
            )
        for split, prefixes in matrix:
            trace = (
                trace_dir
                / f"layerweave_m4_full_{split}_b{batch_size}"
                f"{scale_tag}.trace"
            )
            request_count = sum(1 for _ in trace.open())
            for prefetch in (0, 1):
                for prefix in prefixes:
                    stem = (
                        f"m4-{split}-b{batch_size}-p{prefix}"
                        f"-f{prefetch}"
                    )
                    output = args.output_dir / f"{stem}.json"
                    log = args.output_dir / f"{stem}.log"
                    target_list = (
                        calibration if split == "calibration"
                        else evaluation
                    )
                    target_list.append(output)
                    if args.resume and valid_result(output):
                        print(f"M4_RESUME={output}", flush=True)
                        continue
                    env = os.environ.copy()
                    env.update({
                        "GPU_ID": args.gpu,
                        "CONFIG_PATH": str(config),
                        "TRACE_PATH": str(trace),
                        "LOAD_MODE": "layerweave",
                        "KV_BACKEND": "odkv",
                        "VMM_POOL_GIB": "42.0",
                        "VMM_PAGE_SIZE_MIB": "64",
                        "MAX_REQUESTS": str(request_count),
                        "MAX_BATCH_SIZE": str(batch_size),
                        "OUTPUT_TOKENS_OVERRIDE": "1",
                        "TRACE_TIME_SCALE": "1",
                        "INPUT_SCALE": str(args.input_scale),
                        "LAYERWEAVE_PREFETCH": str(prefetch),
                        "LAYERWEAVE_PREFIX_LAYERS": prefix,
                        "OUTPUT": str(output),
                    })
                    run_logged(
                        ["bash", "tools/layerpipe/run_layerpipe_real_gpu.sh"], env, log)

    report_path = args.output_dir / "m4-full-estimator-report.json"
    estimator_command = [
        str(PYTHON),
        str(REPO_ROOT / "tools/layerpipe/layerweave_estimator.py"),
        "--calibration",
        *[str(path) for path in calibration],
        "--evaluation",
        *[str(path) for path in evaluation],
        "--output",
        str(report_path),
    ]
    subprocess.run(estimator_command, cwd=REPO_ROOT, check=True)
    report = json.loads(report_path.read_text())
    summarize(report)
    models = report["service_ttft_by_model"]
    batches = report["service_ttft_by_batch_size"]
    validation = {
        "cold_mape_le_10pct": (
            report["service_ttft_summary"]["cold"]["mape"] <= 0.10
        ),
        "partial_mape_le_10pct": (
            report["service_ttft_summary"]["partial-hit"]["mape"] <= 0.10
        ),
        "cold_p95_le_100ms": (
            report["service_ttft_summary"]["cold"]
            ["p95_absolute_error_ms"] <= 100.0
        ),
        "partial_p95_le_100ms": (
            report["service_ttft_summary"]["partial-hit"]
            ["p95_absolute_error_ms"] <= 100.0
        ),
        "gain_p95_le_100ms": (
            report["overlap_gain_summary"]
            ["p95_absolute_error_ms"] <= 100.0
        ),
        "all_8_models_covered": set(models) == {
            str(model_id) for model_id in range(8)
        },
        "requested_batch_sizes_covered": set(batches) == {
            str(batch_size) for batch_size in batch_sizes
        },
    }
    report["m4_scope"] = {
        "gpu": f"NVIDIA L40 physical GPU {args.gpu}",
        "vmm_pool_gib": 42.0,
        "vmm_page_size_mib": 64,
        "models": list(range(8)),
        "batch_sizes": batch_sizes,
        "input_scale": args.input_scale,
        "max_batch_size_policy": (
            "fixed_one" if batch_sizes == [1] else "matrix"),
        "output_tokens_override": 1,
        "pipeline_activation": "single-vLLM-prefill-wave",
        "matrix": (
            "marginal-prefix-0-2-4-8-16-32-full"
            if args.marginal_prefix_matrix else (
                "batch1-full-residency;batch2-4-8-full-calibration-"
                "partial-evaluation"
                if args.compact_batches else "full"
            )
        ),
    }
    report["validation"] = validation
    report["m4_validation"] = (
        "PASS" if all(validation.values()) else "FAIL"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    for name, passed in validation.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"M4_FULL_VALIDATION={report['m4_validation']}")
    print(f"M4_FULL_REPORT={report_path}")
    if report["m4_validation"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            f"M4_FAILED returncode={error.returncode} "
            f"command={' '.join(error.cmd)}",
            file=sys.stderr,
        )
        raise
