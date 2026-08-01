#!/usr/bin/env python3
"""Plot profile-driven TTFT CDFs from tensor simulator JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def load_ttft(path):
    document = json.loads(path.read_text())
    rows = document["policies"]["minimal"]["requests"]
    return sorted(float(row["critical_path_ms"]) for row in rows)


def quantile(values, fraction):
    index = fraction * (len(values) - 1)
    lower = int(index)
    upper = min(len(values) - 1, lower + 1)
    weight = index - lower
    return values[lower] + weight * (values[upper] - values[lower])


def polyline(values, x0, y0, width, height, xmin, xmax, ymin):
    points = []
    for index, value in enumerate(values, 1):
        x = x0 + width * (value - xmin) / max(1e-9, xmax - xmin)
        cdf = index / len(values)
        y = y0 + height * (1.0 - (cdf - ymin) / (1.0 - ymin))
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def panel(
        chunks, values_by_label, x0, title, xmin, xmax, ymin,
        colors, width=470, height=300):
    y0 = 82
    chunks.append(
        f'<text x="{x0 + width / 2}" y="48" text-anchor="middle" '
        f'class="title">{title}</text>')
    chunks.append(
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
        'class="frame"/>')
    for tick in range(6):
        x = x0 + width * tick / 5
        value = xmin + (xmax - xmin) * tick / 5
        chunks.append(
            f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" '
            f'y2="{y0 + height}" class="grid"/>')
        chunks.append(
            f'<text x="{x:.2f}" y="{y0 + height + 22}" '
            f'text-anchor="middle" class="tick">{value:.0f}</text>')
    for tick in range(5):
        cdf = ymin + (1 - ymin) * tick / 4
        y = y0 + height * (1 - tick / 4)
        chunks.append(
            f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + width}" '
            f'y2="{y:.2f}" class="grid"/>')
        chunks.append(
            f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'class="tick">{cdf:.2f}</text>')
    for (label, values), color in zip(values_by_label, colors):
        points = polyline(
            values, x0, y0, width, height, xmin, xmax, ymin)
        chunks.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.4"/>')
    chunks.append(
        f'<text x="{x0 + width / 2}" y="{y0 + height + 52}" '
        'text-anchor="middle" class="axis">Simulated TTFT (ms)</text>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--mckp", type=Path)
    parser.add_argument(
        "--series", nargs=2, action="append", metavar=("JSON", "LABEL"),
        help="Optional repeated JSON/label pairs; replaces baseline/mckp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="TTFT CDF: tensor-level simulator, 2 GPUs")
    parser.add_argument(
        "--baseline-label", default="Ordinary TensorGroup LRU + cache-bytes")
    parser.add_argument(
        "--mckp-label", default="Prefix-MCKP K=32 + transition routing")
    args = parser.parse_args()
    if not args.series and (args.baseline is None or args.mckp is None):
        parser.error("provide --series pairs or both --baseline and --mckp")

    labels = (
        [(label, load_ttft(Path(path))) for path, label in args.series]
        if args.series else [
            (args.baseline_label, load_ttft(args.baseline)),
            (args.mckp_label, load_ttft(args.mckp)),
        ])
    colors = ("#2563eb", "#d9485f", "#16a085", "#6b7280")
    xmax = max(values[-1] for _, values in labels) * 1.01
    tail_min = min(
        quantile(values, 0.85) for _, values in labels)
    request_count = len(labels[0][1])
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" '
        'viewBox="0 0 1120 500">',
        '<style>.title{font:600 16px sans-serif}.tick{font:12px '
        'sans-serif;fill:#374151}.axis{font:14px sans-serif}.frame{fill:white;'
        'stroke:#374151}.grid{stroke:#d1d5db;stroke-width:1}.legend{font:13px '
        'sans-serif}</style>',
        '<rect width="1120" height="500" fill="white"/>',
        '<text x="560" y="24" text-anchor="middle" class="title">'
        f'{args.title}'
        '</text>',
    ]
    panel(
        chunks, labels, 70, f"Full distribution (n={request_count})",
        0, xmax, 0.0, colors)
    panel(
        chunks, labels, 630, "Tail detail (CDF 0.85-1.00)",
        tail_min, xmax, 0.85, colors)
    legend_y = 474
    for index, ((label, _), color) in enumerate(zip(labels, colors)):
        x = 95 + index * 350
        chunks.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 32}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        chunks.append(
            f'<text x="{x + 40}" y="{legend_y + 4}" '
            f'class="legend">{label}</text>')
    chunks.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(chunks))
    print(args.output)


if __name__ == "__main__":
    main()
