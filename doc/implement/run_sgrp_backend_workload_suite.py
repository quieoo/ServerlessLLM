#!/usr/bin/env python3
"""Run the three tensor-memory backends on the six workload-suite traces."""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BACKENDS = ("segment", "tensor-page", "compact-page")
WORKLOADS = {
    "servegen": Path("evaluation/traces/servegen_tangram.trace"),
    "cyclic": Path("evaluation/traces/sgrp-paper-r1000/input/i256-seed1236.trace"),
    "reuse-d2": Path("evaluation/traces/sgrp-paper-r1000/reuse/d2-seed1236.trace"),
    "reuse-d7": Path("evaluation/traces/sgrp-paper-r1000/reuse/dmax-seed1236.trace"),
    "small-hot": Path("evaluation/traces/sgrp-paper-r1000/size/neg1-seed1236.trace"),
    "large-hot": Path("evaluation/traces/sgrp-paper-r1000/size/pos1-seed1236.trace"),
}


def command(args, workload, trace, backend):
    output = args.output_dir / workload / f"{backend}.json"
    cmd = [
        str(args.python), "tools/layerpipe/layerweave_tensor_pipeline_sim.py",
        "--tensor-layout", "docs/tensor-level-sim/tensor-layout.json",
        "--config", "configs/servegen_8_models_layerpipe_l40_pool42.json",
        "--trace", str(trace), "--pool-gib", "42",
        "--max-requests", str(args.requests), "--gpus", "4",
        "--gpu-busy-probability", "0",
        "--available-gpus-per-request", "2",
        "--availability-seed", "1236", "--trace-mode", "all",
        "--input-scale", "4", "--input-limit-policy", "safe",
        "--output-tokens-override", "1", "--memory-layout", backend,
        "--page-size-mib", "8", "--h2d-gbps", "24.56",
        "--tensor-group-min-mib", "64", "--policy-suite", "minimal-only",
        "--mckp-stall-table-input",
        "docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json",
        "--mckp-prediction-mode", "lookahead", "--mckp-lookahead-k", "64",
        "--mckp-lookahead-discount", "0.9",
        "--mckp-transition-weight", "0.1",
        "--replacement-policy", "mckp-prefix",
        "--routing-policy", "mckp-transition",
        "--output", str(output),
    ]
    return cmd, output


def run(item):
    name, cmd, output, force = item
    if output.exists() and not force:
        return name, "SKIP"
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    with log.open("w") as stream:
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                text=True)
    return name, "OK" if result.returncode == 0 else f"FAILED({result.returncode})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--workloads")
    parser.add_argument("--max-parallel", type=int, default=18)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(
        "/home/sdu/.conda/envs/sllm-worker/bin/python"))
    args = parser.parse_args()
    selected = set(args.workloads.split(",")) if args.workloads else set(WORKLOADS)
    tasks = []
    for workload, trace in WORKLOADS.items():
        if workload not in selected:
            continue
        for backend in BACKENDS:
            cmd, output = command(args, workload, trace, backend)
            tasks.append((f"{workload}/{backend}", cmd, output, args.force))
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
