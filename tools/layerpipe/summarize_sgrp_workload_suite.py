#!/usr/bin/env python3
"""Build the representative-workload four-policy figure and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


STRATEGIES = (
    ("cb-lru", "CB-LRU", "#4c78a8"),
    ("cb-suffix-lru", "CB-SuffixLRU", "#f58518"),
    ("cb-mckp", "CB-MCKP", "#54a24b"),
    ("joint-mckp", "Joint-MCKP", "#e45756"),
)
SELECTED = (
    ("servegen", "ServeGen", None, None),
    ("cyclic", "Balanced cyclic", "input", "i256"),
    ("reuse-low", "Reuse D=2", "reuse", "d2"),
    ("reuse-high", "Reuse D=7", "reuse", "dmax"),
    ("small-hot", "Small models hot", "size", "neg1"),
    ("large-hot", "Large models hot", "size", "pos1"),
)


def ci95(values):
    return 2.776 * statistics.stdev(values) / math.sqrt(len(values))


def servegen_rows(result_dir, warmup):
    rows = []
    by_seed = {}
    for seed in range(1234, 1239):
        by_seed[seed] = {}
        for strategy, _, _ in STRATEGIES:
            document = json.loads(
                (result_dir / "workload" / "servegen" / f"seed{seed}"
                 / f"{strategy}.json").read_text())
            requests = document["policies"]["minimal"]["requests"][warmup:]
            by_seed[seed][strategy] = statistics.fmean(
                request["critical_path_ms"] for request in requests)
            by_seed[seed][strategy + "_exposed"] = statistics.fmean(
                request["exposed_load_ms"] for request in requests)
    for strategy, label, _ in STRATEGIES:
        normalized = [
            by_seed[seed][strategy + "_exposed"]
            / by_seed[seed]["cb-lru_exposed"]
            for seed in by_seed
        ]
        rows.append({
            "workload": "servegen",
            "workload_label": "ServeGen",
            "strategy": strategy,
            "label": label,
            "normalized_exposed_load_mean": statistics.fmean(normalized),
            "normalized_exposed_load_ci95": ci95(normalized),
            "exposed_load_improvement_mean_pct": statistics.fmean(
                100 * (1 - value) for value in normalized),
        })
    return rows


def svg(rows):
    width, height = 1200, 520
    x0, y0, chart_w, chart_h = 75, 55, 1085, 360
    ymin, ymax = .88, 1.025
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{font-size:13px}'
        '.tick{font-size:11px}.legend{font-size:13px}.frame{fill:#fff;stroke:#444}'
        '.grid{stroke:#ddd}</style><rect width="1200" height="520" fill="#fff"/>',
        f'<rect x="{x0}" y="{y0}" width="{chart_w}" height="{chart_h}" '
        'class="frame"/>',
    ]

    def sy(value):
        return y0 + chart_h * (ymax - value) / (ymax - ymin)

    for index in range(6):
        value = ymin + index * (ymax - ymin) / 5
        y = sy(value)
        chunks.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + chart_w}" '
            f'y2="{y:.1f}" class="grid"/>')
        chunks.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{value:.2f}</text>')
    group_w = chart_w / len(SELECTED)
    bar_w = group_w * .16
    for group_index, (workload, label, _, _) in enumerate(SELECTED):
        center = x0 + group_w * (group_index + .5)
        for strategy_index, (strategy, _, color) in enumerate(STRATEGIES):
            row = next(
                item for item in rows
                if item["workload"] == workload
                and item["strategy"] == strategy)
            mean = float(row["normalized_exposed_load_mean"])
            ci = float(row["normalized_exposed_load_ci95"])
            x = center + (strategy_index - 1.5) * bar_w
            y = sy(mean)
            chunks.append(
                f'<rect x="{x - bar_w * .42:.1f}" y="{y:.1f}" '
                f'width="{bar_w * .84:.1f}" height="{sy(ymin) - y:.1f}" '
                f'fill="{color}"/>')
            chunks.append(
                f'<line x1="{x:.1f}" y1="{sy(mean + ci):.1f}" '
                f'x2="{x:.1f}" y2="{sy(mean - ci):.1f}" stroke="#222"/>')
        chunks.append(
            f'<text x="{center:.1f}" y="{y0 + chart_h + 24}" '
            f'text-anchor="middle" class="tick">{label}</text>')
    chunks.append(
        f'<text transform="translate(20,{y0 + chart_h / 2}) rotate(-90)" '
        'text-anchor="middle" class="axis">Normalized exposed loading stall</text>')
    for index, (_, label, color) in enumerate(STRATEGIES):
        x = 170 + index * 245
        chunks.append(
            f'<rect x="{x}" y="475" width="20" height="12" '
            f'fill="{color}"/><text x="{x + 28}" y="486" '
            f'class="legend">{label}</text>')
    chunks.append("</svg>")
    return "\n".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = list(csv.DictReader(args.aggregate_csv.open()))
    output = servegen_rows(args.result_dir, args.warmup_requests)
    for workload, workload_label, dimension, level in SELECTED[1:]:
        for strategy, label, _ in STRATEGIES:
            source = next(
                row for row in aggregate
                if row["dimension"] == dimension
                and row["level"] == level
                and row["strategy"] == strategy)
            output.append({
                "workload": workload,
                "workload_label": workload_label,
                "strategy": strategy,
                "label": label,
                "normalized_exposed_load_mean":
                    source["normalized_exposed_load_mean"],
                "normalized_exposed_load_ci95":
                    source["normalized_exposed_load_ci95"],
                "exposed_load_improvement_mean_pct":
                    source["exposed_load_improvement_mean_pct"],
            })
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    args.svg_output.write_text(svg(output))
    print(args.csv_output)
    print(args.svg_output)


if __name__ == "__main__":
    main()
