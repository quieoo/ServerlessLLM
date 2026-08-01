#!/usr/bin/env python3
"""Create paper-ready SVGs for the four-policy SGRP evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DIMENSIONS = (
    ("reuse", "(a) Distinct-model reuse distance", "Mean distinct distance"),
    ("input", "(b) Input length", "Effective input tokens"),
    ("size", "(c) Model size-hotness correlation", "Spearman correlation"),
    ("gpu", "(d) GPU count", "GPUs"),
)
STRATEGIES = (
    ("cb-lru", "CB-LRU", "#4c78a8"),
    ("cb-suffix-lru", "CB-SuffixLRU", "#f58518"),
    ("cb-mckp", "CB-MCKP", "#54a24b"),
    ("joint-mckp", "Joint-MCKP", "#e45756"),
)


def sensitivity_svg(rows, metric):
    if metric == "exposed-load":
        mean_field = "normalized_exposed_load_mean"
        ci_field = "normalized_exposed_load_ci95"
        ylabel = "Normalized exposed loading stall (lower is better)"
    else:
        mean_field = "normalized_ttft_mean"
        ci_field = "normalized_ttft_ci95"
        ylabel = "Normalized mean TTFT (lower is better)"
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" '
        'viewBox="0 0 1200 820">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:17px;'
        'font-weight:600}.axis{font-size:13px}.tick{font-size:11px}.frame{fill:#fff;'
        'stroke:#444}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:13px}'
        '</style><rect width="1200" height="820" fill="#fff"/>',
    ]
    panel_width, panel_height = 480, 280
    positions = ((80, 80), (680, 80), (80, 470), (680, 470))
    for (dimension, title, xlabel), (x0, y0) in zip(DIMENSIONS, positions):
        panel = [row for row in rows if row["dimension"] == dimension]
        xs = sorted(set(float(row["x"]) for row in panel))
        all_y = [
            float(row[mean_field])
            + float(row[ci_field])
            for row in panel
        ] + [
            float(row[mean_field])
            - float(row[ci_field])
            for row in panel
        ]
        ymin = min(.75, min(all_y) - .02)
        ymax = max(1.05, max(all_y) + .02)

        def sx(value):
            if len(xs) == 1:
                return x0 + panel_width / 2
            return x0 + panel_width * (value - xs[0]) / (xs[-1] - xs[0])

        def sy(value):
            return y0 + panel_height * (ymax - value) / (ymax - ymin)

        chunks.append(
            f'<text x="{x0 + panel_width / 2}" y="{y0 - 28}" '
            f'text-anchor="middle" class="title">{title}</text>')
        chunks.append(
            f'<rect x="{x0}" y="{y0}" width="{panel_width}" '
            f'height="{panel_height}" class="frame"/>')
        for index in range(5):
            value = ymin + index * (ymax - ymin) / 4
            y = sy(value)
            chunks.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" '
                f'y2="{y:.1f}" class="grid"/>')
            chunks.append(
                f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" '
                f'class="tick">{value:.2f}</text>')
        for value in xs:
            x = sx(value)
            label = f"{value:g}" if dimension != "reuse" else f"{value:.2f}"
            chunks.append(
                f'<text x="{x:.1f}" y="{y0 + panel_height + 20}" '
                f'text-anchor="middle" class="tick">{label}</text>')
        for strategy, _, color in STRATEGIES:
            series = sorted(
                (row for row in panel if row["strategy"] == strategy),
                key=lambda row: float(row["x"]))
            points = " ".join(
                f'{sx(float(row["x"])):.1f},'
                f'{sy(float(row[mean_field])):.1f}'
                for row in series)
            chunks.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="2.5"/>')
            for row in series:
                x = sx(float(row["x"]))
                mean = float(row[mean_field])
                ci = float(row[ci_field])
                top, bottom = sy(mean + ci), sy(mean - ci)
                chunks.append(
                    f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" '
                    f'y2="{bottom:.1f}" stroke="{color}"/>')
                chunks.append(
                    f'<circle cx="{x:.1f}" cy="{sy(mean):.1f}" r="4" '
                    f'fill="{color}"/>')
        chunks.append(
            f'<text x="{x0 + panel_width / 2}" '
            f'y="{y0 + panel_height + 48}" text-anchor="middle" '
            f'class="axis">{xlabel}</text>')
        chunks.append(
            f'<text transform="translate({x0 - 55},'
            f'{y0 + panel_height / 2}) rotate(-90)" text-anchor="middle" '
            f'class="axis">{ylabel}</text>')
    for index, (_, label, color) in enumerate(STRATEGIES):
        x = 190 + index * 245
        chunks.append(
            f'<line x1="{x}" y1="805" x2="{x + 28}" y2="805" '
            f'stroke="{color}" stroke-width="3"/>')
        chunks.append(
            f'<text x="{x + 36}" y="809" class="legend">{label}</text>')
    chunks.append("</svg>")
    return "\n".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--sensitivity-svg", type=Path, required=True)
    parser.add_argument(
        "--metric", choices=("ttft", "exposed-load"),
        default="exposed-load")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.aggregate_csv.open()))
    args.sensitivity_svg.parent.mkdir(parents=True, exist_ok=True)
    args.sensitivity_svg.write_text(sensitivity_svg(rows, args.metric))
    print(args.sensitivity_svg)


if __name__ == "__main__":
    main()
