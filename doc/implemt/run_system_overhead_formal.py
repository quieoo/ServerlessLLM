#!/usr/bin/env python3
"""Run matched native/no-hook/full-hook warm-Prefill experiments."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
import threading
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/sdu/.conda/envs/sllm-worker/bin/python")
NATIVE = REPO / "tools/layerpipe/vllm_warm_prefill_native_bench.py"
VMM = REPO / "tools/layerpipe/vllm_odkv_trace_bench.py"
SUMMARIZER = REPO / "tools/layerpipe/summarize_warm_prefill_overhead.py"


def result_valid(path, expected):
    if not path.exists():
        return False
    try:
        document = json.loads(path.read_text())
        return len(document.get("batch_metrics", [])) == expected
    except (OSError, ValueError):
        return False


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        subprocess.run(
            [str(item) for item in command],
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def make_trace(path, model_id, input_tokens, requests):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        f"0.000000 {model_id} {input_tokens} 1\n"
        for _ in range(requests)
    ))


def run_cell(cell, args, print_lock):
    model_id = cell["model_id"]
    input_tokens = cell["input_tokens"]
    device = cell["device"]
    total = args.warmup + args.samples
    stem = f"m{model_id}-in{input_tokens}"
    trace = args.output_dir / "traces" / f"{stem}.trace"
    make_trace(trace, model_id, input_tokens, total)
    outputs = {
        "native": args.output_dir / "raw" / f"{stem}-native.json",
        "vmm_resident": args.output_dir / "raw" / f"{stem}-no-hooks.json",
        "layerweave_resident": args.output_dir / "raw" / f"{stem}-hooks.json",
    }
    common = [
        "--config", args.config,
        "--trace", trace,
        "--device", device,
    ]
    commands = {
        "native": [
            PYTHON, NATIVE, *common,
            "--max-requests", total,
            "--output-tokens", 1,
            "--output", outputs["native"],
        ],
        "vmm_resident": [
            PYTHON, VMM, *common,
            "--load-mode", "layerweave",
            "--warm-resident-mode", "no-hooks",
            "--max-requests", total,
            "--max-batch-size", 1,
            "--output-tokens-override", 1,
            "--trace-time-scale", 0,
            "--vmm-pool-gib", args.pool_gib,
            "--output", outputs["vmm_resident"],
        ],
        "layerweave_resident": [
            PYTHON, VMM, *common,
            "--load-mode", "layerweave",
            "--warm-resident-mode", "hooks",
            "--max-requests", total,
            "--max-batch-size", 1,
            "--output-tokens-override", 1,
            "--trace-time-scale", 0,
            "--vmm-pool-gib", args.pool_gib,
            "--output", outputs["layerweave_resident"],
        ],
    }
    # Rotate path order across cells to reduce fixed ordering bias while
    # preserving same-GPU execution for all three matched paths.
    orders = [
        ["native", "vmm_resident", "layerweave_resident"],
        ["vmm_resident", "layerweave_resident", "native"],
        ["layerweave_resident", "native", "vmm_resident"],
    ]
    order = orders[(model_id + input_tokens) % len(orders)]
    for name in order:
        if args.force or not result_valid(outputs[name], total):
            run_logged(
                commands[name],
                args.output_dir / "logs" / f"{stem}-{name}.log",
            )
    summary = args.output_dir / "summary" / f"{stem}.json"
    run_logged([
        PYTHON, SUMMARIZER,
        "--native", outputs["native"],
        "--vmm-resident", outputs["vmm_resident"],
        "--layerweave-resident", outputs["layerweave_resident"],
        "--warmup", args.warmup,
        "--model-id", model_id,
        "--output", summary,
    ], args.output_dir / "logs" / f"{stem}-summary.log")
    with print_lock:
        print(f"complete model={model_id} input={input_tokens} gpu={device}",
              flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/servegen_8_models_layerpipe_l40_pool42.json")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument(
        "--model-ids", default="",
        help="Optional comma-separated model IDs; empty runs all models.")
    parser.add_argument("--input-lengths", default="128,1024,4096,16384")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--pool-gib", type=float, default=42.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "doc/results/system-overhead-formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    devices = [int(value) for value in args.devices.split(",")]
    selected_models = (
        {int(value) for value in args.model_ids.split(",")}
        if args.model_ids else None
    )
    lengths = [int(value) for value in args.input_lengths.split(",")]
    configuration = json.loads(args.config.read_text())
    rows = []
    cells_by_device = {device: [] for device in devices}
    for index, model in enumerate(configuration["model_lists"]):
        model_id = int(model["id"])
        if selected_models is not None and model_id not in selected_models:
            continue
        limit = int(model["l40_safe_max_input_length"])
        device = devices[index % len(devices)]
        for length in lengths:
            feasible = length + 1 <= limit
            row = {
                "model_id": model_id,
                "pattern": model.get("pattern", ""),
                "input_tokens": length,
                "safe_input_limit": limit,
                "device": device,
                "status": "pending" if feasible else "infeasible",
            }
            rows.append(row)
            if feasible:
                cells_by_device[device].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lock = threading.Lock()

    def run_device(device):
        summaries = []
        for cell in cells_by_device[device]:
            summaries.append(run_cell(cell, args, lock))
        return summaries

    summaries = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(devices)) as executor:
        futures = [executor.submit(run_device, device) for device in devices]
        for future in concurrent.futures.as_completed(futures):
            summaries.extend(future.result())
    print(f"formal overhead complete: {len(summaries)} feasible cells")


if __name__ == "__main__":
    main()
