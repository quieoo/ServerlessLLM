#!/usr/bin/env python3
"""Sweep GPU busy probability for the five End2End systems in parallel."""

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


SYSTEM_ARGS = {
    "baseline": (),
    "pipe-only": (),
    "reuse-only": (
        "--replacement-policy", "lru",
        "--routing-policy", "cache-bytes",
    ),
    "aegaeon": (),
    "tangram": (
        "--replacement-policy", "mckp-prefix",
        "--routing-policy", "mckp-transition",
        "--mckp-prediction-mode", "lookahead",
        "--mckp-lookahead-k", "32",
        "--mckp-lookahead-discount", "0.9",
        "--mckp-transition-weight", "0.1",
    ),
}


def probability_tag(index):
    return f"0p{index}" if index < 10 else "1p0"


def build_command(args, system, probability_index):
    probability = probability_index / 10
    tag = probability_tag(probability_index)
    probability_dir = args.output_dir / f"busyp{tag}"
    prefix = f"4gpu-busyp{tag}-{system}"
    return [
        str(args.python),
        str(args.simulator),
        "--tensor-layout", str(args.tensor_layout),
        "--config", str(args.config),
        "--trace", str(args.trace),
        "--pool-gib", "42",
        "--max-requests", "5000",
        "--gpus", "4",
        "--gpu-busy-probability", f"{probability:.1f}",
        "--availability-seed", "1234",
        "--trace-mode", "model-switches",
        "--input-scale", "4",
        "--input-limit-policy", "safe",
        "--output-tokens-override", "1",
        "--memory-layout", "segment",
        "--page-size-mib", "8",
        "--h2d-gbps", "24.56",
        "--tensor-group-min-mib", "64",
        "--policy-suite", "minimal-only",
        "--mckp-stall-table-input", str(args.stall_table),
        "--system", system,
        *SYSTEM_ARGS[system],
        "--output", str(probability_dir / f"{prefix}.json"),
        "--csv-output", str(probability_dir / f"{prefix}.csv"),
    ]


def run_job(job):
    command, log_path, output_path, resume = job
    if resume and output_path.exists():
        try:
            document = json.loads(output_path.read_text())
            if document.get("policies"):
                return {
                    "status": "skipped",
                    "elapsed_s": 0.0,
                    "output": str(output_path),
                    "log": str(log_path),
                    "command": shlex.join(command),
                }
        except (OSError, ValueError):
            pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {
        "status": (
            "completed" if result.returncode == 0
            else f"failed:{result.returncode}"
        ),
        "elapsed_s": time.monotonic() - started,
        "output": str(output_path),
        "log": str(log_path),
        "command": shlex.join(command),
    }


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-workers", type=int, required=True,
        help="Maximum number of simulator processes to run concurrently.")
    parser.add_argument(
        "--python", type=Path,
        default=Path("/home/sdu/.conda/envs/sllm-worker/bin/python"))
    parser.add_argument(
        "--simulator", type=Path,
        default=repo / "evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py")
    parser.add_argument(
        "--tensor-layout", type=Path,
        default=repo / "evaluation/tensor_simulator/results/tensor-layout.json")
    parser.add_argument(
        "--config", type=Path,
        default=repo / "configs/servegen_8_models_layerpipe_l40_pool42.json")
    parser.add_argument(
        "--trace", type=Path,
        default=repo / "evaluation/traces/servegen_tangram.trace")
    parser.add_argument(
        "--stall-table", type=Path,
        default=repo / (
            "evaluation/tensor_simulator/results/"
            "offline-prefix-stall-runtime-group64.json"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=repo / (
            "doc/results/end2end/"
            "corrected-busyprob-sweep-req5000"))
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip existing JSON outputs that contain a policies object.")
    args = parser.parse_args()
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    command_lines = []
    for probability_index in range(11):
        for system in SYSTEM_ARGS:
            command = build_command(args, system, probability_index)
            output_path = Path(command[command.index("--output") + 1])
            log_path = (
                args.output_dir / "logs"
                / output_path.parent.name
                / f"{output_path.stem}.log"
            )
            jobs.append((command, log_path, output_path, args.resume))
            command_lines.append(shlex.join(command))
    (args.output_dir / "commands.txt").write_text(
        "\n".join(command_lines) + "\n")

    print(json.dumps({
        "jobs": len(jobs),
        "max_workers": args.max_workers,
        "probabilities": [index / 10 for index in range(11)],
        "systems": list(SYSTEM_ARGS),
        "output_dir": str(args.output_dir),
    }), flush=True)

    records = []
    failures = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if record["status"].startswith("failed"):
                failures.append(record)
            if completed % 5 == 0 or completed == len(jobs):
                print(
                    f"{completed}/{len(jobs)} jobs; "
                    f"failures={len(failures)}; "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True)

    records.sort(key=lambda record: record["output"])
    with (args.output_dir / "run-manifest.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "status", "elapsed_s", "output", "log", "command"))
        writer.writeheader()
        writer.writerows(records)
    if failures:
        print(json.dumps(failures[:10], indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
