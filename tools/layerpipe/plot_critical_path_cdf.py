#!/usr/bin/env python3
"""Plot per-request critical-path CDFs from five system JSON files."""

import argparse
import html
import json
import math
from pathlib import Path


SYSTEMS = ("baseline", "pipe-only", "reuse-only", "aegaeon", "tangram")
LABELS = {
    "baseline": "Baseline",
    "pipe-only": "Pipe-only",
    "reuse-only": "Reuse-only",
    "aegaeon": "Aegaeon",
    "tangram": "Tangram",
}
COLORS = ("#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756")


def percentile(values, probability):
    index = min(len(values) - 1, round(probability * (len(values) - 1)))
    return values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-fragment", action="store_true")
    args = parser.parse_args()

    series = []
    for system, color in zip(SYSTEMS, COLORS):
        path = args.input_dir / f"4gpu-busyp0p5-{system}.json"
        document = json.loads(path.read_text())
        values = sorted(
            row["critical_path_ms"] / 1000.0
            for row in document["policies"][system]["requests"]
        )
        series.append((
            system, color, values,
            f"{LABELS[system]} "
            f"(mean {sum(values) / len(values):.3f}s, "
            f"P95 {percentile(values, 0.95):.3f}s)",
        ))

    width, height = 900, 560
    left, right, top, bottom = 85, 25, 25, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [value for _, _, values, _ in series for value in values]
    x_min = 10 ** math.floor(math.log10(max(min(all_values), 1e-3)))
    x_max = 10 ** math.ceil(math.log10(max(all_values)))

    def x_position(value):
        fraction = (
            (math.log10(max(value, x_min)) - math.log10(x_min))
            / (math.log10(x_max) - math.log10(x_min))
        )
        return left + fraction * plot_width

    def y_position(fraction):
        return top + (1.0 - fraction) * plot_height

    if args.html_fragment:
        colors = tuple(
            f"var(--viz-series-{index})" for index in range(1, 6))
        grid_color = "var(--border)"
        text_color = "var(--foreground)"
        svg = [
            '<div id="critical-path-cdf" style="width:100%">',
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="CDF of per-request critical-path latency for five '
            f'systems" viewBox="0 0 {width} {height}" '
            'style="display:block;width:100%;height:auto">',
            f'<g font-family="sans-serif" font-size="12" '
            f'fill="{text_color}">',
        ]
        series = [
            (system, color, values, label)
            for (system, _, values, label), color in zip(series, colors)
        ]
    else:
        grid_color = "#dddddd"
        text_color = "#222222"
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<g font-family="sans-serif" font-size="12" '
            f'fill="{text_color}">',
        ]

    for probability in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_position(probability)
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" '
            f'x2="{width-right}" y2="{y:.2f}" '
            f'stroke="{grid_color}" stroke-width="1"/>')
        svg.append(
            f'<text x="{left-10}" y="{y+4:.2f}" '
            f'text-anchor="end">{probability:.2f}</text>')
    exponent = math.floor(math.log10(x_min))
    while 10 ** exponent <= x_max:
        value = 10 ** exponent
        x = x_position(value)
        svg.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{height-bottom}" stroke="{grid_color}" stroke-width="1"/>')
        svg.append(
            f'<text x="{x:.2f}" y="{height-bottom+22}" '
            f'text-anchor="middle">{value:g}</text>')
        exponent += 1
    for _, color, values, _ in series:
        points = " ".join(
            f"{x_position(value):.2f},"
            f"{y_position((index + 1) / len(values)):.2f}"
            for index, value in enumerate(values)
        )
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>')
    svg.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="{text_color}"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{height-bottom}" stroke="{text_color}"/>',
        f'<text x="{left+plot_width/2:.2f}" y="{height-18}" '
        'text-anchor="middle">Critical-path latency (seconds, log scale)</text>',
        f'<text x="20" y="{top+plot_height/2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 20 {top+plot_height/2:.2f})">'
        'Cumulative fraction of requests</text>',
    ])
    legend_x, legend_y = left + 12, top + 12
    for index, (_, color, _, label) in enumerate(series):
        y = legend_y + index * 21
        svg.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>')
        svg.append(
            f'<text x="{legend_x+32}" y="{y+4}">'
            f'{html.escape(label)}</text>')
    svg.extend(["</g>", "</svg>"])
    if args.html_fragment:
        svg.append("</div>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
