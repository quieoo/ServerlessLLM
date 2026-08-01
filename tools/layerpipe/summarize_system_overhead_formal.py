#!/usr/bin/env python3
"""Aggregate per-cell formal warm-Prefill overhead summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    manifest = list(csv.DictReader(
        (args.result_dir / "manifest.csv").open()))
    output = []
    for row in manifest:
        base = {
            "model_id": int(row["model_id"]),
            "pattern": row["pattern"],
            "input_tokens": int(row["input_tokens"]),
            "safe_input_limit": int(row["safe_input_limit"]),
            "device": int(row["device"]),
            "status": row["status"],
        }
        if row["status"] == "infeasible":
            output.append(base)
            continue
        path = (args.result_dir / "summary"
                / f"m{row['model_id']}-in{row['input_tokens']}.json")
        if not path.exists():
            base["status"] = "missing"
            output.append(base)
            continue
        document = json.loads(path.read_text())
        base.update({
            "status": "complete",
            "native_mean_ms": document["native"]["mean_ms"],
            "vmm_resident_mean_ms": document[
                "vmm_resident"]["mean_ms"],
            "layerweave_resident_mean_ms": document[
                "layerweave_resident"]["mean_ms"],
            "readiness_hook_overhead_mean_ms": document[
                "readiness_hook_overhead_mean_ms"],
            "readiness_hook_overhead_pct": 100.0 * document[
                "readiness_hook_overhead_mean_ms"] / document[
                    "vmm_resident"]["mean_ms"],
            "total_tangram_overhead_mean_ms": document[
                "total_tangram_overhead_mean_ms"],
        })
        output.append(base)
    fields = []
    for row in output:
        for key in row:
            if key not in fields:
                fields.append(key)
    csv_path = args.result_dir / "summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(csv_path)


if __name__ == "__main__":
    main()
