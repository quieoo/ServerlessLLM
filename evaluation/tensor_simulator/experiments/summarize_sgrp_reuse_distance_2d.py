#!/usr/bin/env python3
"""Summarize the 200-request distinct-distance x pool SGRP scan."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metrics(rows):
    ttft = [row["critical_path_ms"] for row in rows]
    exposed = [row["exposed_load_ms"] for row in rows]
    solver = [row["mckp_solver_time_ms"] for row in rows]
    return {
        "mean_ttft_ms": statistics.fmean(ttft),
        "p95_ttft_ms": quantile(ttft, .95),
        "p99_ttft_ms": quantile(ttft, .99),
        "mean_exposed_load_ms": statistics.fmean(exposed),
        "h2d_gib": sum(row["h2d_bytes"] for row in rows) / 1024**3,
        "mean_mckp_solver_time_ms": statistics.fmean(solver),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    output = []
    for trace_path in sorted(args.trace_dir.glob("d*.trace")):
        name = trace_path.stem
        metadata = json.loads(
            trace_path.with_suffix(".trace.json").read_text())
        trace_metrics = metadata["metrics"]
        for pool_dir in sorted(
                (args.result_dir / name).glob("pool*"),
                key=lambda path: int(path.name.removeprefix("pool"))):
            pool = int(pool_dir.name.removeprefix("pool"))
            pair = {}
            for strategy in ("cb-lru", "joint-mckp"):
                document = json.loads(
                    (pool_dir / f"{strategy}.json").read_text())
                rows = document["policies"]["minimal"]["requests"]
                if len(rows) != metadata["requests"]:
                    raise ValueError(
                        f"{name}/pool{pool}/{strategy}: request mismatch")
                pair[strategy] = metrics(rows[args.warmup_requests:])
            gain = (
                100.0
                * (pair["cb-lru"]["mean_ttft_ms"]
                   - pair["joint-mckp"]["mean_ttft_ms"])
                / pair["cb-lru"]["mean_ttft_ms"])
            output.append({
                "trace": name,
                "distinct_mean": trace_metrics["distinct_mean"],
                "request_gap_p50": trace_metrics["request_gap_p50"],
                "request_gap_p90": trace_metrics["request_gap_p90"],
                "gap_within_k_fraction":
                    trace_metrics["request_gap_within_k_fraction"],
                "pool_gib": pool,
                "cb_lru_mean_ttft_ms":
                    pair["cb-lru"]["mean_ttft_ms"],
                "joint_mckp_mean_ttft_ms":
                    pair["joint-mckp"]["mean_ttft_ms"],
                "joint_mckp_ttft_improvement_pct": gain,
                "cb_lru_mean_exposed_load_ms":
                    pair["cb-lru"]["mean_exposed_load_ms"],
                "joint_mckp_mean_exposed_load_ms":
                    pair["joint-mckp"]["mean_exposed_load_ms"],
                "cb_lru_h2d_gib": pair["cb-lru"]["h2d_gib"],
                "joint_mckp_h2d_gib": pair["joint-mckp"]["h2d_gib"],
                "joint_mckp_solver_time_ms":
                    pair["joint-mckp"]["mean_mckp_solver_time_ms"],
                "cb_lru_p95_ttft_ms": pair["cb-lru"]["p95_ttft_ms"],
                "joint_mckp_p95_ttft_ms":
                    pair["joint-mckp"]["p95_ttft_ms"],
                "cb_lru_p99_ttft_ms": pair["cb-lru"]["p99_ttft_ms"],
                "joint_mckp_p99_ttft_ms":
                    pair["joint-mckp"]["p99_ttft_ms"],
            })

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(args.csv_output)


if __name__ == "__main__":
    main()
