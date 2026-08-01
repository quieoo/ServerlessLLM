#!/usr/bin/env python3
"""Summarize the matched size-hotness correlation fast scan."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


LEVELS = ("neg1", "neg0p5", "zero", "pos0p5", "pos1")
STRATEGIES = ("cb-lru", "joint-mckp")


def mean_metrics(rows):
    return {
        "mean_ttft_ms": statistics.fmean(
            row["critical_path_ms"] for row in rows),
        "mean_exposed_load_ms": statistics.fmean(
            row["exposed_load_ms"] for row in rows),
        "h2d_gib": sum(row["h2d_bytes"] for row in rows) / 1024**3,
        "mean_solver_ms": statistics.fmean(
            row["mckp_solver_time_ms"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238")
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--aggregate-csv-output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]

    rows = []
    for level in LEVELS:
        for seed in seeds:
            metadata = json.loads(
                (args.trace_dir / f"{level}-seed{seed}.trace.json")
                .read_text())
            pair = {}
            for strategy in STRATEGIES:
                document = json.loads(
                    (args.result_dir / level / f"seed{seed}"
                     / f"{strategy}.json").read_text())
                requests = document["policies"]["minimal"]["requests"]
                pair[strategy] = mean_metrics(
                    requests[args.warmup_requests:])
            gain = (
                100.0
                * (pair["cb-lru"]["mean_ttft_ms"]
                   - pair["joint-mckp"]["mean_ttft_ms"])
                / pair["cb-lru"]["mean_ttft_ms"])
            rows.append({
                "level": level,
                "seed": seed,
                "spearman_size_hotness":
                    metadata["spearman_size_hotness"],
                "pearson_bytes_share": metadata["pearson_bytes_share"],
                "cb_lru_mean_ttft_ms": pair["cb-lru"]["mean_ttft_ms"],
                "joint_mckp_mean_ttft_ms":
                    pair["joint-mckp"]["mean_ttft_ms"],
                "joint_mckp_improvement_pct": gain,
                "cb_lru_mean_exposed_load_ms":
                    pair["cb-lru"]["mean_exposed_load_ms"],
                "joint_mckp_mean_exposed_load_ms":
                    pair["joint-mckp"]["mean_exposed_load_ms"],
                "cb_lru_h2d_gib": pair["cb-lru"]["h2d_gib"],
                "joint_mckp_h2d_gib": pair["joint-mckp"]["h2d_gib"],
                "joint_mckp_solver_ms":
                    pair["joint-mckp"]["mean_solver_ms"],
            })

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for level in LEVELS:
        group = [row for row in rows if row["level"] == level]
        gains = [row["joint_mckp_improvement_pct"] for row in group]
        item = {
            "level": level,
            "spearman_size_hotness": group[0]["spearman_size_hotness"],
            "pearson_bytes_share": group[0]["pearson_bytes_share"],
            "seeds": len(group),
            "cb_lru_mean_ttft_ms": statistics.fmean(
                row["cb_lru_mean_ttft_ms"] for row in group),
            "joint_mckp_mean_ttft_ms": statistics.fmean(
                row["joint_mckp_mean_ttft_ms"] for row in group),
            "improvement_mean_pct": statistics.fmean(gains),
            "improvement_stdev_pct": statistics.stdev(gains),
            "improvement_min_pct": min(gains),
            "improvement_max_pct": max(gains),
            "cb_lru_mean_exposed_load_ms": statistics.fmean(
                row["cb_lru_mean_exposed_load_ms"] for row in group),
            "joint_mckp_mean_exposed_load_ms": statistics.fmean(
                row["joint_mckp_mean_exposed_load_ms"] for row in group),
            "cb_lru_mean_h2d_gib": statistics.fmean(
                row["cb_lru_h2d_gib"] for row in group),
            "joint_mckp_mean_h2d_gib": statistics.fmean(
                row["joint_mckp_h2d_gib"] for row in group),
            "joint_mckp_mean_solver_ms": statistics.fmean(
                row["joint_mckp_solver_ms"] for row in group),
        }
        aggregate.append(item)
    with args.aggregate_csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(args.csv_output)
    print(args.aggregate_csv_output)


if __name__ == "__main__":
    main()
