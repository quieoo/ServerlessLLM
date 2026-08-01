#!/usr/bin/env python3
"""Sweep Tensor-page and Compact-page sizes on the ServeGen workload."""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BACKENDS = ("tensor-page", "compact-page")


def command(args, backend, page_size):
    output = args.output_dir / backend / f"page-{page_size}mib.json"
    return [
        str(args.python), "tools/layerpipe/layerweave_tensor_pipeline_sim.py",
        "--tensor-layout", "docs/tensor-level-sim/tensor-layout.json",
        "--config", "configs/servegen_8_models_layerpipe_l40_pool42.json",
        "--trace", "evaluation/traces/servegen_tangram.trace",
        "--pool-gib", "42", "--max-requests", str(args.requests),
        "--gpus", "4", "--gpu-busy-probability", "0",
        "--available-gpus-per-request", "2",
        "--availability-seed", "1236", "--trace-mode", "all",
        "--input-scale", "4", "--input-limit-policy", "safe",
        "--output-tokens-override", "1", "--memory-layout", backend,
        "--page-size-mib", str(page_size), "--h2d-gbps", "24.56",
        "--tensor-group-min-mib", "64", "--policy-suite", "minimal-only",
        "--mckp-stall-table-input",
        "docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json",
        "--mckp-prediction-mode", "lookahead", "--mckp-lookahead-k", "64",
        "--mckp-lookahead-discount", "0.9",
        "--mckp-transition-weight", "0.1",
        "--replacement-policy", "mckp-prefix",
        "--routing-policy", "mckp-transition", "--output", str(output),
    ], output


def run(item):
    name, cmd, output, force = item
    if output.exists() and not force:
        return name, "SKIP"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("w") as stream:
        result = subprocess.run(
            cmd, stdout=stream, stderr=subprocess.STDOUT, text=True)
    status = "OK" if result.returncode == 0 else f"FAILED({result.returncode})"
    return name, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--page-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--max-parallel", type=int,
                        default=min(16, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(
        "/home/sdu/.conda/envs/sllm-worker/bin/python"))
    args = parser.parse_args()
    sizes = [int(value) for value in args.page_sizes.split(",")]
    if not sizes or any(value <= 0 for value in sizes):
        parser.error("page sizes must be positive MiB values")
    tasks = []
    for backend in BACKENDS:
        for page_size in sizes:
            cmd, output = command(args, backend, page_size)
            tasks.append((f"{backend}/{page_size}MiB", cmd, output, args.force))
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            name, status = future.result()
            print(f"[{status}] {name}", flush=True)
            if status.startswith("FAILED"):
                failures.append(name)
    if failures:
        raise SystemExit(f"failed tasks: {failures}")


if __name__ == "__main__":
    main()
