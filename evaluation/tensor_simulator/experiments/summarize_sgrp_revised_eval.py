#!/usr/bin/env python3
"""Summarize revised SGRP long-context and scaling experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


LEVELS = {
    "input": ("i128", "i512", "i2048", "i8192", "i16384"),
    "routing": ("g1", "g2", "g3", "g4"),
    "pressure": ("c1", *tuple(f"c{value}" for value in range(2, 11, 2))),
}
STRATEGIES = (
    ("pipe-only", "Pipe-only"),
    ("cb-lru", "CB-LRU"),
    ("cb-suffix-lru", "CB-SuffixLRU"),
    ("cb-mckp", "CB-MCKP"),
    ("joint-mckp", "Joint-MCKP"),
)


def ci95(values):
    if len(values) < 2:
        return 0.0
    t_value = 2.776 if len(values) == 5 else 1.96
    return t_value * statistics.stdev(values) / math.sqrt(len(values))


def metrics(rows):
    return {
        "mean_exposed_load_ms": statistics.fmean(
            row["exposed_load_ms"] for row in rows),
        "mean_ttft_ms": statistics.fmean(
            row["critical_path_ms"] for row in rows),
        "mean_h2d_gib_per_request": statistics.fmean(
            row["h2d_bytes"] for row in rows) / 1024**3,
        "mean_solver_ms": statistics.fmean(
            row.get("mckp_solver_time_ms", 0.0) for row in rows),
    }


def policy_rows(path, strategy):
    document = json.loads(path.read_text())
    policy = "pipe-only" if strategy == "pipe-only" else "minimal"
    return document["policies"][policy]["requests"]


def x_value(dimension, level, trace_dir):
    if dimension == "input":
        return float(level[1:])
    if dimension == "routing":
        return float(level[1:])
    copies = int(level[1:])
    table = json.loads((trace_dir / "catalogs" / f"copies{copies}"
                        / "stall-table.json").read_text())
    total = sum(item["prefix_bytes"][-1]
                for item in table["models"].values())
    return total / 4 / (42 * 1024**3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238")
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--dimensions", default="input,routing,pressure")
    parser.add_argument("--levels")
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--aggregate-csv-output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    selected = None if args.levels is None else set(args.levels.split(","))
    prefixes = {"input": "i", "routing": "g", "pressure": "c"}
    detail = []
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
                values = {}
                strategies = STRATEGIES if dimension == "input" else STRATEGIES[1:]
                for strategy, label in strategies:
                    path = (args.result_dir / dimension / level / f"seed{seed}"
                            / f"{strategy}.json")
                    rows = policy_rows(path, strategy)[args.warmup_requests:]
                    values[strategy] = metrics(rows)
                baseline = values["cb-lru"]["mean_exposed_load_ms"]
                mckp = values["cb-mckp"]["mean_exposed_load_ms"]
                for strategy, label in strategies:
                    value = values[strategy]
                    detail.append({
                        "dimension": dimension, "level": level, "seed": seed,
                        "x": x_value(dimension, level, args.trace_dir),
                        "strategy": strategy, "label": label, **value,
                        "improvement_vs_lru_pct":
                            100 * (baseline - value["mean_exposed_load_ms"])
                            / baseline,
                        "improvement_vs_cb_mckp_pct":
                            100 * (mckp - value["mean_exposed_load_ms"]) / mckp,
                    })
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
        writer.writeheader(); writer.writerows(detail)

    aggregate = []
    groups = {}
    for row in detail:
        groups.setdefault((row["dimension"], row["level"], row["strategy"]), []).append(row)
    for (dimension, level, strategy), rows in groups.items():
        aggregate.append({
            "dimension": dimension, "level": level, "x": rows[0]["x"],
            "strategy": strategy, "label": rows[0]["label"],
            "seeds": len(rows),
            "mean_exposed_load_ms": statistics.fmean(
                row["mean_exposed_load_ms"] for row in rows),
            "exposed_load_ci95_ms": ci95(
                [row["mean_exposed_load_ms"] for row in rows]),
            "improvement_vs_lru_pct": statistics.fmean(
                row["improvement_vs_lru_pct"] for row in rows),
            "improvement_vs_lru_ci95_pct": ci95(
                [row["improvement_vs_lru_pct"] for row in rows]),
            "improvement_vs_cb_mckp_pct": statistics.fmean(
                row["improvement_vs_cb_mckp_pct"] for row in rows),
            "improvement_vs_cb_mckp_ci95_pct": ci95(
                [row["improvement_vs_cb_mckp_pct"] for row in rows]),
            "wins_vs_lru": sum(row["improvement_vs_lru_pct"] > 0 for row in rows),
            "mean_ttft_ms": statistics.fmean(row["mean_ttft_ms"] for row in rows),
            "mean_h2d_gib_per_request": statistics.fmean(
                row["mean_h2d_gib_per_request"] for row in rows),
            "mean_solver_ms": statistics.fmean(row["mean_solver_ms"] for row in rows),
        })
    with args.aggregate_csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)


if __name__ == "__main__":
    main()
