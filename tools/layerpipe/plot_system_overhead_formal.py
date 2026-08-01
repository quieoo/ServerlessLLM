#!/usr/bin/env python3
"""Dependency-free SVG plot of readiness overhead versus input length."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


COLORS = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756",
    "#72b7b2", "#b279a2", "#ff9da6", "#9d755d",
]


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for row in csv.DictReader(args.summary.open()):
        if row["status"] == "complete":
            grouped[int(row["model_id"])].append((
                int(row["input_tokens"]),
                float(row["readiness_hook_overhead_mean_ms"]),
                float(row["readiness_hook_overhead_pct"]),
            ))
    width, height = 1000, 390
    top, bottom = 35, 55
    panels = [(65, 465, "Readiness overhead (ms)", -12, 90, 1),
              (570, 970, "Readiness overhead (%)", -15, 155, 2)]
    x_values = [128, 1024, 4096]
    x_logs = [math.log2(value) for value in x_values]

    def sx(value, left, right):
        return left + (math.log2(value) - min(x_logs)) / (
            max(x_logs) - min(x_logs)) * (right - left)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}'
        '.title{font-size:14px;font-weight:bold}</style>',
    ]
    for left, right, label, ymin, ymax, value_index in panels:
        plot_top, plot_bottom = top, height - bottom

        def sy(value):
            return plot_bottom - (value - ymin) / (ymax - ymin) * (
                plot_bottom - plot_top)

        svg.append(f'<line x1="{left}" y1="{plot_bottom}" x2="{right}" '
                   f'y2="{plot_bottom}" stroke="#333"/>')
        svg.append(f'<line x1="{left}" y1="{plot_top}" x2="{left}" '
                   f'y2="{plot_bottom}" stroke="#333"/>')
        zero = sy(0)
        svg.append(f'<line x1="{left}" y1="{zero:.2f}" x2="{right}" '
                   f'y2="{zero:.2f}" stroke="#888" stroke-dasharray="3 3"/>')
        for value, tick in zip(x_values, ("128", "1K", "4K")):
            x = sx(value, left, right)
            svg.append(f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" '
                       f'y2="{plot_bottom}" stroke="#ddd"/>')
            svg.append(f'<text x="{x:.2f}" y="{plot_bottom + 20}" '
                       f'text-anchor="middle">{tick}</text>')
        for fraction in (0, .25, .5, .75, 1):
            value = ymin + fraction * (ymax - ymin)
            y = sy(value)
            svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" '
                       f'y2="{y:.2f}" stroke="#eee"/>')
            svg.append(f'<text x="{left - 8}" y="{y + 4:.2f}" '
                       f'text-anchor="end">{value:.0f}</text>')
        svg.append(f'<text class="title" x="{(left + right) / 2}" y="18" '
                   f'text-anchor="middle">{esc(label)}</text>')
        svg.append(f'<text x="{(left + right) / 2}" y="{height - 12}" '
                   f'text-anchor="middle">Input tokens</text>')
        for model_id, rows in sorted(grouped.items()):
            rows.sort()
            points = " ".join(
                f"{sx(row[0], left, right):.2f},{sy(row[value_index]):.2f}"
                for row in rows)
            color = COLORS[model_id % len(COLORS)]
            svg.append(f'<polyline points="{points}" fill="none" '
                       f'stroke="{color}" stroke-width="2"/>')
            for row in rows:
                x, y = sx(row[0], left, right), sy(row[value_index])
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" '
                           f'fill="{color}"/>')
    for model_id in sorted(grouped):
        x = 75 + model_id * 54
        svg.append(f'<line x1="{x}" y1="{height - 33}" x2="{x + 14}" '
                   f'y2="{height - 33}" stroke="{COLORS[model_id]}" '
                   'stroke-width="3"/>')
        svg.append(f'<text x="{x + 18}" y="{height - 29}">M{model_id}</text>')
    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
