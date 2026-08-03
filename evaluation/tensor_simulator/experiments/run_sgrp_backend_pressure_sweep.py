#!/usr/bin/env python3
"""Run best-config tensor memory backends over cache-pressure copies 1--7."""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BACKENDS = {
    "segment": 8,
    "tensor-page": 16,
    "compact-page": 32,
}


def command(args, backend, page_size, copies):
    catalog = args.trace_dir / "catalogs" / f"copies{copies}"
    trace = args.trace_dir / "pressure" / f"c{copies}-seed1236.trace"
    output = args.output_dir / f"c{copies}" / f"{backend}.json"
    return [
        str(args.python),
        "evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py",
        "--tensor-layout", "evaluation/tensor_simulator/results/tensor-layout.json",
        "--config", str(catalog / "config.json"), "--trace", str(trace),
        "--pool-gib", "42", "--max-requests", "0", "--gpus", "4",
        "--gpu-busy-probability", "0", "--available-gpus-per-request", "2",
        "--availability-seed", "1236", "--trace-mode", "all",
        "--input-scale", "1", "--input-limit-policy", "safe",
        "--stall-table-input-policy", "clamp",
        "--output-tokens-override", "1", "--memory-layout", backend,
        "--page-size-mib", str(page_size), "--h2d-gbps", "24.56",
        "--tensor-group-min-mib", "64", "--policy-suite", "minimal-only",
        "--mckp-stall-table-input", str(catalog / "stall-table.json"),
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
    log = output.parent / "logs" / f"{output.stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(
            cmd, stdout=stream, stderr=subprocess.STDOUT, text=True)
    return name, "OK" if result.returncode == 0 else f"FAILED({result.returncode})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copies", default="1,2,3,4,5,6,7")
    parser.add_argument("--max-parallel", type=int,
                        default=min(21, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(
        "/home/sdu/.conda/envs/sllm-worker/bin/python"))
    args = parser.parse_args()
    tasks = []
    for copies in (int(value) for value in args.copies.split(",")):
        for backend, page_size in BACKENDS.items():
            cmd, output = command(args, backend, page_size, copies)
            tasks.append((f"c{copies}/{backend}", cmd, output, args.force))
    print(f"tasks={len(tasks)} max_parallel={args.max_parallel}", flush=True)
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
