#!/usr/bin/env python3
"""Run revised input, routing weak-scaling, and cache-pressure experiments."""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


LEVELS = {
    "input": ("i128", "i512", "i2048", "i8192", "i16384"),
    "routing": ("g1", "g2", "g3", "g4"),
    "pressure": ("c1", *tuple(f"c{value}" for value in range(2, 11, 2))),
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
            cores.add(((topology / "physical_package_id").read_text(),
                       (topology / "core_id").read_text()))
        except OSError:
            return len(allowed)
    return len(cores)


def memory_slots():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return max(1, int(line.split()[1]) // (2 * 1024 * 1024))
    return 1


def task_command(args, dimension, level, seed, strategy):
    pipe_only = strategy == "pipe-only"
    replacement, routing = (
        ("lru", "cache-bytes") if pipe_only else STRATEGIES[strategy])
    if dimension == "input":
        copies, gpus, available = None, 4, 2
        config = Path("configs/servegen_8_models_layerpipe_l40_pool42.json")
        table = Path(
            "docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json")
    elif dimension == "routing":
        copies = gpus = int(level[1:])
        available = gpus
        catalog = args.trace_dir / "catalogs" / f"copies{copies}"
        config, table = catalog / "config.json", catalog / "stall-table.json"
    else:
        copies, gpus, available = int(level[1:]), 4, 2
        catalog = args.trace_dir / "catalogs" / f"copies{copies}"
        config, table = catalog / "config.json", catalog / "stall-table.json"
    trace = args.trace_dir / dimension / f"{level}-seed{seed}.trace"
    command = [
        str(args.python), "tools/layerpipe/layerweave_tensor_pipeline_sim.py",
        "--tensor-layout", "docs/tensor-level-sim/tensor-layout.json",
        "--config", str(config), "--trace", str(trace),
        "--pool-gib", "42", "--max-requests", "0",
        "--gpus", str(gpus), "--gpu-busy-probability", "0",
        "--available-gpus-per-request", str(available),
        "--availability-seed", str(seed), "--trace-mode", "all",
        "--input-scale", "1", "--input-limit-policy",
        "none" if dimension == "input" else "safe",
        "--stall-table-input-policy",
        "linear" if dimension == "input" else "clamp",
        "--output-tokens-override", "1", "--memory-layout", "segment",
        "--page-size-mib", "8", "--h2d-gbps", "24.56",
        "--tensor-group-min-mib", "64", "--policy-suite", "minimal-only",
        "--mckp-stall-table-input", str(table),
        "--mckp-prediction-mode", "lookahead", "--mckp-lookahead-k", "64",
        "--mckp-lookahead-discount", "0.9",
        "--mckp-transition-weight", "0.1",
        "--replacement-policy", replacement, "--routing-policy", routing,
    ]
    if pipe_only:
        command.extend(["--system", "pipe-only"])
    output = (args.output_dir / dimension / level / f"seed{seed}"
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
        result = subprocess.run(command, stdout=stream,
                                stderr=subprocess.STDOUT, text=True)
    return name, "OK" if result.returncode == 0 else f"FAILED({result.returncode})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238")
    parser.add_argument("--dimensions", default="input,routing,pressure")
    parser.add_argument("--levels")
    parser.add_argument("--python", type=Path, default=Path(
        "/home/sdu/.conda/envs/sllm-worker/bin/python"))
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    selected = None if args.levels is None else set(args.levels.split(","))
    prefixes = {"input": "i", "routing": "g", "pressure": "c"}
    tasks = []
    for dimension in args.dimensions.split(","):
        levels = LEVELS[dimension]
        if selected is not None:
            levels = tuple(sorted(
                (level for level in selected
                 if level.startswith(prefixes[dimension])),
                key=lambda level: int(level[1:])))
        for level in levels:
            if selected is not None and level not in selected:
                continue
            for seed in seeds:
                for strategy in STRATEGIES:
                    command, output = task_command(
                        args, dimension, level, seed, strategy)
                    tasks.append((f"{dimension}/{level}/seed{seed}/{strategy}",
                                  command, output, args.force))
                if dimension == "input":
                    command, output = task_command(
                        args, dimension, level, seed, "pipe-only")
                    tasks.append((f"{dimension}/{level}/seed{seed}/pipe-only",
                                  command, output, args.force))
    workers = args.max_parallel or min(physical_cores(), memory_slots())
    print(f"tasks={len(tasks)} max_parallel={workers}", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            name, status = future.result()
            print(f"[{status}] {name}", flush=True)
            if status.startswith("FAILED"):
                failures.append(name)
    if failures:
        raise SystemExit(f"{len(failures)} tasks failed: {failures[:8]}")


if __name__ == "__main__":
    main()
