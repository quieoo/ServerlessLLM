#!/usr/bin/env python3
"""Build a prefix-residency stall table from real group boundary profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def fingerprint(document):
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def simulate_residency_stall(
        boundaries, groups, resident, h2d_bytes_per_ms, allocation_ms):
    if not boundaries:
        return 0.0
    actual_previous = None
    baseline_previous = None
    copy_end = 0.0
    for index, (baseline, group) in enumerate(zip(boundaries, groups)):
        baseline = float(baseline)
        if actual_previous is None:
            compute_ready = baseline
        else:
            compute_ready = (
                actual_previous + max(0.0, baseline - baseline_previous))
        required = bool(group["observed"])
        missing = not resident(index) and required
        if missing:
            load_ms = (
                int(group["logical_bytes"]) / h2d_bytes_per_ms
                + allocation_ms
            )
            load_start = (
                copy_end if actual_previous is None
                else max(copy_end, actual_previous)
            )
            copy_end = load_start + load_ms
            actual = max(compute_ready, copy_end)
        else:
            actual = compute_ready
        actual_previous = actual
        baseline_previous = baseline
    return max(0.0, actual_previous - float(boundaries[-1]))


def simulate_prefix_stall(boundaries, groups, prefix, h2d_bytes_per_ms,
                          allocation_ms):
    return simulate_residency_stall(
        boundaries, groups, lambda index: index < prefix,
        h2d_bytes_per_ms, allocation_ms)


def simulate_suffix_stall(boundaries, groups, suffix_start, h2d_bytes_per_ms,
                          allocation_ms):
    return simulate_residency_stall(
        boundaries, groups, lambda index: index >= suffix_start,
        h2d_bytes_per_ms, allocation_ms)


def monotone_envelope(values):
    result = [max(0.0, float(value)) for value in values]
    if result:
        result[-1] = 0.0
    for index in range(len(result) - 2, -1, -1):
        result[index] = max(result[index], result[index + 1])
    return result


def increasing_monotone_envelope(values):
    result = [max(0.0, float(value)) for value in values]
    if result:
        result[0] = 0.0
    for index in range(1, len(result)):
        result[index] = max(result[index], result[index - 1])
    return result


def correct_event_overhead(boundaries, groups, overhead_ms):
    """Remove uniformly distributed per-boundary hook overhead.

    The profiler measures the instrumented and uninstrumented GPU envelope at
    every token length.  Runtime groups are decoder-order boundaries, so the
    measured excess is apportioned over observed boundary hooks.
    """
    overhead_ms = max(0.0, float(overhead_ms or 0.0))
    observed_total = sum(bool(group["observed"]) for group in groups)
    if not boundaries or observed_total == 0 or overhead_ms == 0:
        return [float(value) for value in boundaries]
    corrected = []
    observed_rank = 0
    previous = 0.0
    for boundary, group in zip(boundaries, groups):
        if group["observed"]:
            observed_rank += 1
        value = float(boundary) - (
            overhead_ms * observed_rank / observed_total)
        value = max(previous, value, 0.0)
        corrected.append(value)
        previous = value
    return corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h2d-gbps", type=float, default=24.56)
    parser.add_argument("--segment-allocation-ms", type=float, default=0.005)
    args = parser.parse_args()
    if args.h2d_gbps <= 0 or args.segment_allocation_ms < 0:
        parser.error("H2D bandwidth must be positive and allocation nonnegative")
    h2d_bytes_per_ms = args.h2d_gbps * 1_000_000.0
    models = {}
    group_sizes = set()
    gpu_names = set()
    for path in args.profiles:
        profile = json.loads(path.read_text())
        if profile.get("format") != (
                "layerweave-runtime-group-boundary-profile-v1"):
            raise ValueError(f"unsupported profile format: {path}")
        model_id = int(profile["model_id"])
        if str(model_id) in models:
            raise ValueError(f"duplicate model profile: {model_id}")
        groups = profile["groups"]
        group_sizes.add(int(profile["tensor_group_min_bytes"]))
        gpu_names.add(profile["gpu_name"])
        rows = []
        for token_text, item in sorted(
                profile["profiles"].items(), key=lambda pair: int(pair[0])):
            boundaries = item["median_group_boundary_ms"]
            if len(boundaries) != len(groups):
                raise ValueError(
                    f"boundary/group mismatch model={model_id} tokens={token_text}")
            corrected_boundaries = correct_event_overhead(
                boundaries, groups, item.get("event_gpu_overhead_ms"))
            stalls = [
                simulate_prefix_stall(
                    corrected_boundaries, groups, prefix, h2d_bytes_per_ms,
                    args.segment_allocation_ms)
                for prefix in range(len(groups) + 1)
            ]
            suffix_stalls = [
                simulate_suffix_stall(
                    corrected_boundaries, groups, suffix_start,
                    h2d_bytes_per_ms, args.segment_allocation_ms)
                for suffix_start in range(len(groups) + 1)
            ]
            rows.append({
                "input_tokens": int(token_text),
                "raw_group_boundary_ms": boundaries,
                "group_boundary_ms": corrected_boundaries,
                "prefix_stall_ms": monotone_envelope(stalls),
                "suffix_stall_ms":
                    increasing_monotone_envelope(suffix_stalls),
                "median_profile_wall_ms": item["median_wall_ms"],
                "median_baseline_wall_ms":
                    item.get("median_baseline_wall_ms"),
                "event_wall_overhead_ms":
                    item.get("event_wall_overhead_ms"),
                "event_gpu_overhead_ms":
                    item.get("event_gpu_overhead_ms"),
            })
        prefix_bytes = [0]
        for group in groups:
            prefix_bytes.append(
                prefix_bytes[-1] + int(group["logical_bytes"]))
        models[str(model_id)] = {
            "model_id": model_id,
            "model_path": profile["model_path"],
            "group_count": len(groups),
            "runtime_parameter_bytes": profile["runtime_parameter_bytes"],
            "observed_parameter_bytes": profile["observed_parameter_bytes"],
            "unobserved_parameter_bytes":
                profile["unobserved_parameter_bytes"],
            "prefix_bytes": prefix_bytes,
            "groups": groups,
            "rows": rows,
            "profile_elapsed_seconds": profile["elapsed_seconds"],
            "order_mismatches": len(profile["order_mismatches"]),
            "profile_path": str(path),
        }
    if len(group_sizes) != 1:
        raise ValueError(f"profiles use different group sizes: {group_sizes}")
    table = {
        "format": "layerweave-prefix-stall-v1",
        "metadata": {
            "memory_layout": "segment",
            "tensor_group_semantics":
                "runtime fused weight-use stages greedily merged to minimum",
            "tensor_group_min_bytes": next(iter(group_sizes)),
            "h2d_gbps": args.h2d_gbps,
            "segment_allocation_ms": args.segment_allocation_ms,
            "compute_profile_source":
                "real vLLM prefill CUDA-event weight-use boundaries",
            "event_overhead_correction":
                "instrumented-minus-baseline GPU envelope distributed "
                "uniformly over observed group-boundary hooks",
            "stall_semantics":
                "whole-request exposed load for cached prefix and missing suffix",
            "input_interpolation": "piecewise-linear between measured rows",
            "gpu_names": sorted(gpu_names),
            "profile_count": len(args.profiles),
        },
        "models": dict(sorted(models.items(), key=lambda pair: int(pair[0]))),
    }
    table["metadata"]["fingerprint"] = fingerprint({
        "metadata": table["metadata"],
        "models": {
            key: {
                "group_bytes": [
                    group["logical_bytes"] for group in value["groups"]
                ],
                "input_tokens": [
                    row["input_tokens"] for row in value["rows"]
                ],
            }
            for key, value in table["models"].items()
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2))
    overheads = [
        row["event_wall_overhead_ms"]
        for model in models.values() for row in model["rows"]
        if row["event_wall_overhead_ms"] is not None
    ]
    print(json.dumps({
        "output": str(args.output),
        "models": len(models),
        "groups": sum(item["group_count"] for item in models.values()),
        "rows": sum(len(item["rows"]) for item in models.values()),
        "cells": sum(
            2 * len(item["rows"]) * (item["group_count"] + 1)
            for item in models.values()),
        "median_event_wall_overhead_ms":
            statistics.median(overheads) if overheads else None,
        "fingerprint": table["metadata"]["fingerprint"],
    }, indent=2))


if __name__ == "__main__":
    main()
