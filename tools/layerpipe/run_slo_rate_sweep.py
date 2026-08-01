#!/usr/bin/env python3
"""Run the five-system online SLO/request-rate sweep in parallel."""

import argparse
import concurrent.futures
import csv
import json
import os
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


def rate_tag(rate):
    return f"{rate:.2f}".replace(".", "p")


def build_command(args, system, rate, seed, output_dir):
    prefix = f"{system}-r{rate_tag(rate)}-seed{seed}"
    return [
        str(args.python),
        str(args.simulator),
        "--tensor-layout", str(args.tensor_layout),
        "--config", str(args.config),
        "--trace", str(args.trace),
        "--pool-gib", "42",
        "--max-requests", str(args.max_requests),
        "--warmup-requests", str(args.warmup_requests),
        "--gpus", "4",
        "--gpu-busy-probability", "0",
        "--trace-mode", "all",
        "--input-scale", "4",
        "--input-limit-policy", "safe",
        "--output-tokens-override", "1",
        "--memory-layout", "segment",
        "--page-size-mib", "8",
        "--h2d-gbps", "24.56",
        "--tensor-group-min-mib", "64",
        "--policy-suite", "minimal-only",
        "--mckp-stall-table-input", str(args.stall_table),
        "--execution-mode", "online",
        "--arrival-mode", "poisson",
        "--request-rate-rps", f"{rate:.2f}",
        "--arrival-seed", str(seed),
        "--decode-ms-per-token", "40",
        "--slo-scale", "2",
        "--system", system,
        *SYSTEM_ARGS[system],
        "--output", str(output_dir / f"{prefix}.json"),
        "--requests-csv-output",
        str(output_dir / f"{prefix}-requests.csv"),
    ]


def run_one(command, log_path, output_path, resume):
    if resume and output_path.exists():
        try:
            document = json.loads(output_path.read_text())
            if document["policies"]:
                return "skipped", 0.0
        except (OSError, ValueError, KeyError):
            pass
    started = time.monotonic()
    with log_path.open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    if result.returncode:
        return f"failed:{result.returncode}", elapsed
    return "completed", elapsed


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python", type=Path,
        default=Path("/home/sdu/.conda/envs/sllm-worker/bin/python"))
    parser.add_argument(
        "--simulator", type=Path,
        default=repo / "tools/layerpipe/layerweave_tensor_pipeline_sim.py")
    parser.add_argument(
        "--tensor-layout", type=Path,
        default=repo / "docs/tensor-level-sim/tensor-layout.json")
    parser.add_argument(
        "--config", type=Path,
        default=repo / "configs/servegen_8_models_layerpipe_l40_pool42.json")
    parser.add_argument(
        "--trace", type=Path,
        default=repo / "evaluation/traces/servegen_tangram.trace")
    parser.add_argument(
        "--stall-table", type=Path,
        default=repo / (
            "docs/tensor-level-sim/"
            "offline-prefix-stall-runtime-group64.json"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=repo / "doc/results/slo/request-rate-sweep")
    parser.add_argument("--max-requests", type=int, default=5000)
    parser.add_argument("--warmup-requests", type=int, default=500)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rates = (0.08, 0.12, 0.14, 0.16, 0.17,
             0.18, 0.19, 0.20, 0.21, 0.22)
    seeds = (1234, 2234, 3234, 4234, 5234)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    cpu_count = os.cpu_count() or 1
    load_one = os.getloadavg()[0]
    auto_workers = max(
        1, min(88, int(cpu_count * 0.80 - min(load_one, cpu_count))))
    workers = args.workers or auto_workers

    jobs = []
    with (args.output_dir / "commands.txt").open("w") as command_file:
        for seed in seeds:
            for rate in rates:
                for system in SYSTEM_ARGS:
                    command = build_command(
                        args, system, rate, seed, args.output_dir)
                    output_path = Path(command[command.index("--output") + 1])
                    prefix = output_path.stem
                    log_path = log_dir / f"{prefix}.log"
                    command_file.write(shlex.join(command) + "\n")
                    jobs.append((command, log_path, output_path))

    print(json.dumps({
        "cpu_count": cpu_count,
        "load_one": load_one,
        "workers": workers,
        "jobs": len(jobs),
        "output_dir": str(args.output_dir),
    }), flush=True)
    records = []
    failures = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers) as executor:
        future_jobs = {
            executor.submit(
                run_one, command, log_path, output_path, args.resume
            ): (command, log_path, output_path)
            for command, log_path, output_path in jobs
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_jobs):
            command, log_path, output_path = future_jobs[future]
            status, elapsed = future.result()
            completed += 1
            record = {
                "status": status,
                "elapsed_s": f"{elapsed:.3f}",
                "output": str(output_path),
                "log": str(log_path),
                "command": shlex.join(command),
            }
            records.append(record)
            if status.startswith("failed"):
                failures.append(record)
            if completed % 10 == 0 or completed == len(jobs):
                print(
                    f"{completed}/{len(jobs)} jobs; failures={len(failures)}; "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True)

    with (args.output_dir / "run-manifest.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    if failures:
        print(json.dumps(failures[:10], indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
