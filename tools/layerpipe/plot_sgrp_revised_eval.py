#!/usr/bin/env python3
"""Render the revised three-panel SGRP evaluation as dependency-free SVG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = {
    "pipe-only": "#777777", "cb-lru": "#4c78a8",
    "cb-suffix-lru": "#f58518", "cb-mckp": "#54a24b",
    "joint-mckp": "#e45756",
}
LABELS = {
    "pipe-only": "Pipe-only", "cb-lru": "CB-LRU",
    "cb-suffix-lru": "CB-SuffixLRU", "cb-mckp": "CB-MCKP",
    "joint-mckp": "Joint-MCKP",
}
PANELS = (
    ("input", "(a) Long-context sensitivity", "Input tokens", True),
    ("routing", "(b) Routing weak scaling", "GPUs", False),
    ("pressure", "(c) Cache-pressure scaling",
     "Model working set / aggregate cache", False),
)


def svg(rows):
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="510" '
        'viewBox="0 0 1500 510">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:17px;'
        'font-weight:600}.axis{font-size:13px}.tick{font-size:11px}.legend{font-size:13px}'
        '.frame{fill:#fff;stroke:#444}.grid{stroke:#ddd;stroke-width:1}</style>',
        '<rect width="1500" height="510" fill="#fff"/>',
    ]
    width, height, y0 = 380, 300, 70
    for panel_index, (dimension, title, xlabel, absolute) in enumerate(PANELS):
        x0 = 80 + panel_index * 485
        panel = [row for row in rows if row["dimension"] == dimension]
        xs = sorted(set(float(row["x"]) for row in panel))
        field = "mean_exposed_load_ms" if absolute else "improvement_vs_lru_pct"
        ci_field = "exposed_load_ci95_ms" if absolute else "improvement_vs_lru_ci95_pct"
        values = [float(row[field]) for row in panel]
        cis = [float(row[ci_field]) for row in panel]
        ymin = min(0.0, min(v-c for v, c in zip(values, cis)))
        ymax = max(v+c for v, c in zip(values, cis)) * 1.08
        if ymax <= ymin: ymax = ymin + 1

        def sx(value):
            if dimension == "input":
                import math
                left, right = math.log2(xs[0]), math.log2(xs[-1])
                return x0 + width * (math.log2(value)-left)/(right-left)
            return x0 + width * (value-xs[0])/(xs[-1]-xs[0])

        def sy(value):
            return y0 + height * (ymax-value)/(ymax-ymin)

        chunks += [
            f'<text x="{x0+width/2}" y="35" text-anchor="middle" class="title">{title}</text>',
            f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" class="frame"/>',
        ]
        for index in range(5):
            value = ymin + index*(ymax-ymin)/4
            yy = sy(value)
            chunks += [
                f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+width}" y2="{yy:.1f}" class="grid"/>',
                f'<text x="{x0-7}" y="{yy+4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>',
            ]
        for value in xs:
            xx = sx(value)
            label = f"{value/1024:g}K" if dimension == "input" and value >= 1024 else f"{value:g}"
            chunks.append(f'<text x="{xx:.1f}" y="{y0+height+20}" text-anchor="middle" class="tick">{label}</text>')
        strategies = ("pipe-only", "cb-lru", "cb-suffix-lru", "cb-mckp", "joint-mckp") if absolute else ("cb-lru", "cb-suffix-lru", "cb-mckp", "joint-mckp")
        for strategy in strategies:
            series = sorted((row for row in panel if row["strategy"] == strategy), key=lambda row: float(row["x"]))
            points = " ".join(f'{sx(float(row["x"])):.1f},{sy(float(row[field])):.1f}' for row in series)
            color = COLORS[strategy]
            chunks.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for row in series:
                xx, mean, ci = sx(float(row["x"])), float(row[field]), float(row[ci_field])
                chunks += [
                    f'<line x1="{xx:.1f}" y1="{sy(mean+ci):.1f}" x2="{xx:.1f}" y2="{sy(mean-ci):.1f}" stroke="{color}"/>',
                    f'<circle cx="{xx:.1f}" cy="{sy(mean):.1f}" r="4" fill="{color}"/>',
                ]
        ylabel = "Exposed loading stall (ms)" if absolute else "Improvement vs CB-LRU (%)"
        chunks += [
            f'<text x="{x0+width/2}" y="{y0+height+47}" text-anchor="middle" class="axis">{xlabel}</text>',
            f'<text transform="translate({x0-53},{y0+height/2}) rotate(-90)" text-anchor="middle" class="axis">{ylabel}</text>',
        ]
    for index, strategy in enumerate(("pipe-only", "cb-lru", "cb-suffix-lru", "cb-mckp", "joint-mckp")):
        x = 250 + index*215
        chunks += [
            f'<line x1="{x}" y1="480" x2="{x+28}" y2="480" stroke="{COLORS[strategy]}" stroke-width="3"/>',
            f'<text x="{x+35}" y="484" class="legend">{LABELS[strategy]}</text>',
        ]
    chunks.append('</svg>')
    return "\n".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.aggregate_csv.open()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg(rows))


if __name__ == "__main__":
    main()
