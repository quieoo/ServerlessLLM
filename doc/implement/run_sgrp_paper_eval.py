#!/usr/bin/env python3
"""Run the four-policy SGRP paper matrix with CPU-aware concurrency."""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


LEVELS = {
    "reuse": ("d2", "d3", "d4", "d5", "dmax"),
    "input": ("i32", "i64", "i128", "i256"),
    "size": ("neg1", "neg0p5", "zero", "pos0p5", "pos1"),
    "gpu": ("g2", "g3", "g4", "g6", "g8"),
    "workload": ("servegen",),
}
STRATEGIES = {
    "cb-lru": ("lru", "cache-bytes"),
    "cb-suffix-lru": ("lru-prefix", "cache-bytes"),
    "cb-mckp": ("mckp-prefix", "cache-bytes"),
    "joint-mckp": ("mckp-prefix", "mckp-transition"),
}


def physical_cores():
    allowed = os.sched_getaffinity(0)
    cores = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            core = (topology / "core_id").read_text().strip()
            package = (topology / "physical_package_id").read_text().strip()
            cores.add((package, core))
        except OSError:
            return len(allowed)
    return len(cores)


def memory_slots():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            return max(1, available_kib // (2 * 1024 * 1024))
    return 1


def trace_path(root, dimension, level, seed):
    if dimension == "workload":
        return Path("evaluation/traces/servegen_tangram.trace")
    if dimension == "gpu":
        return root / "input" / f"i256-seed{seed}.trace"
    return root / dimension / f"{level}-seed{seed}.trace"


def task_command(args, dimension, level, seed, strategy):
    replacement, routing = STRATEGIES[strategy]
    gpus = int(level[1:]) if dimension == "gpu" else 4
    command = [
        str(args.python),
        "tools/layerpipe/layerweave_tensor_pipeline_sim.py",
        "--tensor-layout", "docs/tensor-level-sim/tensor-layout.json",
        "--config", "configs/servegen_8_models_layerpipe_l40_pool42.json",
        "--trace", str(trace_path(args.trace_dir, dimension, level, seed)),
        "--pool-gib", "42",
        "--max-requests", str(args.requests),
        "--gpus", str(gpus),
        "--gpu-busy-probability", "0.5",
        "--availability-seed", str(
            seed if dimension == "workload" else args.availability_seed),
        "--trace-mode", "all",
        "--input-scale", "4",
        "--input-limit-policy", "safe",
        "--output-tokens-override", "1",
        "--memory-layout", "segment",
        "--page-size-mib", "8",
        "--h2d-gbps", "24.56",
        "--tensor-group-min-mib", "64",
        "--policy-suite", "minimal-only",
        "--mckp-stall-table-input",
        "docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json",
        "--mckp-prediction-mode", "lookahead",
        "--mckp-lookahead-k", "64",
        "--mckp-lookahead-discount", "0.9",
        "--mckp-transition-weight", "0.1",
        "--replacement-policy", replacement,
        "--routing-policy", routing,
    ]
    if dimension == "gpu":
        command.extend([
            "--available-gpus-per-request", str((gpus + 1) // 2),
        ])
    output = (
        args.output_dir / dimension / level / f"seed{seed}"
        / f"{strategy}.json")
    command.extend(["--output", str(output)])
    return command, output


def run_task(item):
    name, command, output, force = item
    if output.exists() and not force:
        return name, "SKIP"
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / "logs" / f"{output.stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT,
            text=True, check=False)
    if completed.returncode:
        return name, f"FAILED({completed.returncode})"
    return name, "OK"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238")
    parser.add_argument(
        "--dimensions", default="reuse,input,size,gpu")
    parser.add_argument(
        "--levels",
        help="Optional comma-separated level names within selected dimensions.")
    parser.add_argument("--availability-seed", type=int, default=1234)
    parser.add_argument(
        "--python", type=Path,
        default=Path("/home/sdu/.conda/envs/sllm-worker/bin/python"))
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",")]
    dimensions = args.dimensions.split(",")
    selected_levels = (
        None if args.levels is None else set(args.levels.split(",")))
    workers = args.max_parallel or min(physical_cores(), memory_slots())
    tasks = []
    for dimension in dimensions:
        for level in LEVELS[dimension]:
            if selected_levels is not None and level not in selected_levels:
                continue
            for seed in seeds:
                for strategy in STRATEGIES:
                    command, output = task_command(
                        args, dimension, level, seed, strategy)
                    if not Path(command[command.index("--trace") + 1]).exists():
                        raise FileNotFoundError(
                            command[command.index("--trace") + 1])
                    name = f"{dimension}/{level}/seed{seed}/{strategy}"
                    tasks.append((name, command, output, args.force))
    print(
        f"tasks={len(tasks)} physical_cores={physical_cores()} "
        f"memory_slots={memory_slots()} max_parallel={workers}",
        flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            name, status = future.result()
            print(f"[{status}] {name}", flush=True)
            if status.startswith("FAILED"):
                failures.append(name)
    if failures:
        raise SystemExit(f"{len(failures)} tasks failed: {failures[:5]}")


if __name__ == "__main__":
    main()
