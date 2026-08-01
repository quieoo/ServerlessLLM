#!/usr/bin/env python3
"""Summarize the ServeGen Tensor-page/Compact-page page-size sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


BACKENDS = ("tensor-page", "compact-page")
FIELDS = (
    "exposed_load_ms", "critical_path_ms", "exposed_h2d_ms",
    "exposed_page_map_ms", "exposed_page_unmap_ms",
    "mckp_solver_time_ms", "h2d_bytes", "map_calls", "unmap_calls",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--page-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    output = []
    for backend in BACKENDS:
        for page_size in (int(value) for value in args.page_sizes.split(",")):
            path = args.result_dir / backend / f"page-{page_size}mib.json"
            row = {"backend": backend, "page_size_mib": page_size}
            if not path.exists():
                row["status"] = "infeasible"
                output.append(row)
                continue
            requests = json.loads(path.read_text())["policies"]["minimal"][
                "requests"][args.warmup_requests:]
            row["status"] = "ok"
            row["requests_after_warmup"] = len(requests)
            for field in FIELDS:
                row[f"mean_{field}"] = statistics.fmean(
                    request.get(field, 0.0) for request in requests)
            row["mean_h2d_gib"] = row.pop("mean_h2d_bytes") / 1024**3
            row["mean_vmm_ms"] = (
                row["mean_exposed_page_map_ms"]
                + row["mean_exposed_page_unmap_ms"])
            row["mean_internal_fragmentation_mib"] = statistics.fmean(
                request["memory"]["internal_fragmentation_bytes"]
                for request in requests) / 1024**2
            row["mean_mckp_overrelease_mib"] = statistics.fmean(
                request.get("mckp_overrelease_bytes", 0)
                for request in requests) / 1024**2
            assert all(
                request["map_calls"] == request["mapped_pages"]
                and request["unmap_calls"] == request["unmapped_pages"]
                for request in requests)
            output.append(row)
    fieldnames = list(dict.fromkeys(
        key for row in output for key in row))
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
