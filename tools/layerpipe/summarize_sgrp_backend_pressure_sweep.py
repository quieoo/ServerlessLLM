#!/usr/bin/env python3
"""Summarize best-config backend cache-pressure results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


BACKENDS = ("segment", "tensor-page", "compact-page")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    output = []
    for copies in range(1, 8):
        table = json.loads((args.trace_dir / "catalogs"
                            / f"copies{copies}" / "stall-table.json").read_text())
        working_set = sum(
            item["prefix_bytes"][-1] for item in table["models"].values())
        pressure = working_set / (4 * 42 * 1024**3)
        values = {}
        for backend in BACKENDS:
            document = json.loads((args.result_dir / f"c{copies}"
                                   / f"{backend}.json").read_text())
            rows = document["policies"]["minimal"]["requests"][
                args.warmup_requests:]
            assert len(rows) == 1000 - args.warmup_requests
            assert all(
                row["map_calls"] == row["mapped_pages"]
                and row["unmap_calls"] == row["unmapped_pages"]
                for row in rows)
            values[backend] = {
                "mean_exposed_load_ms": statistics.fmean(
                    row["exposed_load_ms"] for row in rows),
                "mean_ttft_ms": statistics.fmean(
                    row["critical_path_ms"] for row in rows),
                "mean_h2d_gib": statistics.fmean(
                    row["h2d_bytes"] for row in rows) / 1024**3,
                "mean_vmm_ms": statistics.fmean(
                    row["map_ms"] + row["unmap_ms"] for row in rows),
                "mean_compaction_ms": statistics.fmean(
                    row["compaction_ms"] for row in rows),
                "mean_selected_mckp_solver_cpu_ms": statistics.fmean(
                    row["mckp_solver_time_ms"] for row in rows),
                "mean_internal_fragmentation_mib": statistics.fmean(
                    row["memory"]["internal_fragmentation_bytes"]
                    for row in rows) / 1024**2,
            }
        best = min(item["mean_exposed_load_ms"] for item in values.values())
        for backend in BACKENDS:
            item = values[backend]
            output.append({
                "copies": copies, "pressure": pressure,
                "backend": backend, **item,
                "delta_vs_best_pct":
                    100 * (item["mean_exposed_load_ms"] - best) / best,
            })
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
