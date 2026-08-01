#!/usr/bin/env python3
"""Summarize four-policy SGRP sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


LEVELS = {
    "reuse": ("d2", "d3", "d4", "d5", "dmax"),
    "input": ("i32", "i64", "i128", "i256"),
    "size": ("neg1", "neg0p5", "zero", "pos0p5", "pos1"),
    "gpu": ("g2", "g3", "g4", "g6", "g8"),
}
STRATEGIES = (
    ("cb-lru", "CB-LRU"),
    ("cb-suffix-lru", "CB-SuffixLRU"),
    ("cb-mckp", "CB-MCKP"),
    ("joint-mckp", "Joint-MCKP"),
)


def quantile(values, fraction):
    values = sorted(float(value) for value in values)
    position = fraction * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def request_metrics(rows):
    ttft = [row["critical_path_ms"] for row in rows]
    exposed = [row["exposed_load_ms"] for row in rows]
    return {
        "mean_ttft_ms": statistics.fmean(ttft),
        "p95_ttft_ms": quantile(ttft, .95),
        "p99_ttft_ms": quantile(ttft, .99),
        "mean_exposed_load_ms": statistics.fmean(exposed),
        "h2d_gib": sum(row["h2d_bytes"] for row in rows) / 1024**3,
        "compaction_gib": (
            sum(row["compaction_moved_bytes"] for row in rows) / 1024**3),
        "mean_solver_ms": statistics.fmean(
            row["mckp_solver_time_ms"] for row in rows),
    }


def x_value(trace_dir, dimension, level, seed):
    if dimension == "gpu":
        return float(level[1:])
    trace = (
        trace_dir / dimension / f"{level}-seed{seed}.trace.json")
    metadata = json.loads(trace.read_text())
    if dimension == "reuse":
        return metadata["metrics"]["distinct_mean"]
    if dimension == "input":
        return float(level[1:]) * 4
    return metadata["spearman_size_hotness"]


def ci95(values):
    if len(values) < 2:
        return 0.0
    # n=5 in the paper matrix; t(4, .975)=2.776.
    t_value = 2.776 if len(values) == 5 else 1.96
    return t_value * statistics.stdev(values) / math.sqrt(len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238")
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--aggregate-csv-output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]

    detail = []
    for dimension, levels in LEVELS.items():
        for level in levels:
            for seed in seeds:
                values = {}
                for strategy, _ in STRATEGIES:
                    path = (
                        args.result_dir / dimension / level / f"seed{seed}"
                        / f"{strategy}.json")
                    document = json.loads(path.read_text())
                    rows = document["policies"]["minimal"]["requests"]
                    values[strategy] = request_metrics(
                        rows[args.warmup_requests:])
                baseline = values["cb-lru"]["mean_ttft_ms"]
                exposed_baseline = values["cb-lru"]["mean_exposed_load_ms"]
                for strategy, label in STRATEGIES:
                    row = {
                        "dimension": dimension,
                        "level": level,
                        "seed": seed,
                        "x": x_value(
                            args.trace_dir, dimension, level, seed),
                        "strategy": strategy,
                        "label": label,
                        **values[strategy],
                        "normalized_ttft":
                            values[strategy]["mean_ttft_ms"] / baseline,
                        "improvement_vs_lru_pct": 100.0 * (
                            baseline - values[strategy]["mean_ttft_ms"]
                        ) / baseline,
                        "normalized_exposed_load": (
                            values[strategy]["mean_exposed_load_ms"]
                            / exposed_baseline),
                        "exposed_load_improvement_vs_lru_pct": 100.0 * (
                            exposed_baseline
                            - values[strategy]["mean_exposed_load_ms"]
                        ) / exposed_baseline,
                    }
                    detail.append(row)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)

    aggregate = []
    for dimension, levels in LEVELS.items():
        for level in levels:
            for strategy, label in STRATEGIES:
                group = [
                    row for row in detail
                    if row["dimension"] == dimension
                    and row["level"] == level
                    and row["strategy"] == strategy
                ]
                normalized = [row["normalized_ttft"] for row in group]
                improvements = [
                    row["improvement_vs_lru_pct"] for row in group]
                exposed_normalized = [
                    row["normalized_exposed_load"] for row in group]
                exposed_improvements = [
                    row["exposed_load_improvement_vs_lru_pct"]
                    for row in group]
                aggregate.append({
                    "dimension": dimension,
                    "level": level,
                    "x": statistics.fmean(row["x"] for row in group),
                    "strategy": strategy,
                    "label": label,
                    "seeds": len(group),
                    "mean_ttft_ms": statistics.fmean(
                        row["mean_ttft_ms"] for row in group),
                    "normalized_ttft_mean": statistics.fmean(normalized),
                    "normalized_ttft_ci95": ci95(normalized),
                    "improvement_mean_pct": statistics.fmean(improvements),
                    "improvement_ci95_pct": ci95(improvements),
                    "wins_vs_lru": sum(value > 0 for value in improvements),
                    "normalized_exposed_load_mean":
                        statistics.fmean(exposed_normalized),
                    "normalized_exposed_load_ci95":
                        ci95(exposed_normalized),
                    "exposed_load_improvement_mean_pct":
                        statistics.fmean(exposed_improvements),
                    "exposed_load_improvement_ci95_pct":
                        ci95(exposed_improvements),
                    "exposed_load_wins_vs_lru": sum(
                        value > 0 for value in exposed_improvements),
                    "mean_p95_ttft_ms": statistics.fmean(
                        row["p95_ttft_ms"] for row in group),
                    "mean_p99_ttft_ms": statistics.fmean(
                        row["p99_ttft_ms"] for row in group),
                    "mean_exposed_load_ms": statistics.fmean(
                        row["mean_exposed_load_ms"] for row in group),
                    "mean_h2d_gib": statistics.fmean(
                        row["h2d_gib"] for row in group),
                    "mean_compaction_gib": statistics.fmean(
                        row["compaction_gib"] for row in group),
                    "mean_solver_ms": statistics.fmean(
                        row["mean_solver_ms"] for row in group),
                })
    with args.aggregate_csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(args.csv_output)
    print(args.aggregate_csv_output)


if __name__ == "__main__":
    main()
