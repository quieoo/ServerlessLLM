#!/usr/bin/env python3
"""Compare matched warm-Prefill result files for three runtime paths."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def samples(path, warmup):
    document = json.loads(path.read_text())
    values = [
        float(item["prefill_ms"])
        for item in document.get("batch_metrics", [])[warmup:]
    ]
    if not values:
        raise ValueError(f"No post-warmup Prefill samples in {path}")
    return values


def stats(values):
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--vmm-resident", type=Path, required=True)
    parser.add_argument("--layerweave-resident", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--model-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {
        "native": samples(args.native, args.warmup),
        "vmm_resident": samples(args.vmm_resident, args.warmup),
        "layerweave_resident": samples(args.layerweave_resident, args.warmup),
    }
    lengths = {len(value) for value in groups.values()}
    if len(lengths) != 1:
        raise ValueError(f"Matched groups have different sample counts: {lengths}")
    summary = {name: stats(value) for name, value in groups.items()}
    summary["model_id"] = args.model_id
    summary["vmm_base_overhead_mean_ms"] = (
        summary["vmm_resident"]["mean_ms"] - summary["native"]["mean_ms"])
    summary["readiness_hook_overhead_mean_ms"] = (
        summary["layerweave_resident"]["mean_ms"]
        - summary["vmm_resident"]["mean_ms"])
    summary["total_tangram_overhead_mean_ms"] = (
        summary["layerweave_resident"]["mean_ms"]
        - summary["native"]["mean_ms"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
