#!/usr/bin/env python3
"""Sweep a fixed critical-path SLO scale over five system JSON files."""

import argparse
import csv
import html
import json
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


def find_input_paths(input_dir):
    baseline_suffix = "-baseline.json"
    baseline_paths = sorted(input_dir.glob(f"*{baseline_suffix}"))
    if not baseline_paths:
        raise FileNotFoundError(
            f"no *{baseline_suffix} input file found in {input_dir}")
    if len(baseline_paths) > 1:
        names = ", ".join(path.name for path in baseline_paths)
        raise ValueError(
            f"multiple input sets found in {input_dir}: {names}")

    prefix = baseline_paths[0].name[:-len(baseline_suffix)]
    paths = {
        system: input_dir / f"{prefix}-{system}.json"
        for system in SYSTEMS
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete input set in {input_dir}; missing: "
            + ", ".join(missing))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--base-slo-ms", type=float, default=1000.0)
    parser.add_argument("--scale-start", type=float, default=1.0)
    parser.add_argument("--scale-stop", type=float, default=3.0)
    parser.add_argument("--scale-step", type=float, default=0.25)
    parser.add_argument("--html-fragment", action="store_true")
    args = parser.parse_args()
    if (
        args.base_slo_ms <= 0 or args.scale_start <= 0
        or args.scale_stop < args.scale_start or args.scale_step <= 0
    ):
        parser.error("SLO and scale arguments must define a positive range")

    values = {}
    for system, path in find_input_paths(args.input_dir).items():
        document = json.loads(path.read_text())
        values[system] = [
            row["critical_path_ms"]
            for row in document["policies"][system]["requests"]
        ]
    counts = {len(system_values) for system_values in values.values()}
    if len(counts) != 1:
        raise ValueError("systems have different request counts")

    step_count = round(
        (args.scale_stop - args.scale_start) / args.scale_step)
    scales = [
        args.scale_start + index * args.scale_step
        for index in range(step_count + 1)
    ]
    rows = []
    for scale in scales:
        threshold = args.base_slo_ms * scale
        row = {
            "slo_scale": scale,
            "slo_threshold_ms": threshold,
        }
        for system in SYSTEMS:
            row[system] = sum(
                value <= threshold for value in values[system]
            ) / len(values[system])
        rows.append(row)

    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "slo_scale", "slo_threshold_ms", *SYSTEMS))
            writer.writeheader()
            writer.writerows(rows)

    width, height = 900, 540
    left, right, top, bottom = 80, 25, 35, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    def xp(scale):
        return left + (
            (scale - args.scale_start)
            / (args.scale_stop - args.scale_start)
        ) * plot_width

    def yp(attainment):
        return top + (1.0 - attainment) * plot_height

    if args.html_fragment:
        colors = tuple(
            f"var(--viz-series-{index})" for index in range(1, 6))
        grid = "var(--border)"
        foreground = "var(--foreground)"
        parts = [
            '<div id="critical-path-slo-scale" style="width:100%">',
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="SLO attainment versus critical-path SLO scale for '
            f'five systems" viewBox="0 0 {width} {height}" '
            'style="display:block;width:100%;height:auto">',
            f'<g font-family="sans-serif" font-size="12" fill="{foreground}">',
        ]
    else:
        colors = COLORS
        grid, foreground = "#dddddd", "#222222"
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<g font-family="sans-serif" font-size="12" fill="{foreground}">',
        ]

    for attainment in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = yp(attainment)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" stroke="{grid}" stroke-width="1"/>')
        parts.append(
            f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end">'
            f'{attainment:.2f}</text>')
    for scale in scales:
        x = xp(scale)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{height-bottom}" stroke="{grid}" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{height-bottom+22}" '
            f'text-anchor="middle">{scale:g}</text>')

    for system, color in zip(SYSTEMS, colors):
        points = " ".join(
            f"{xp(row['slo_scale']):.2f},{yp(row[system]):.2f}"
            for row in rows
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.2"/>')
        for row in rows:
            parts.append(
                f'<circle cx="{xp(row["slo_scale"]):.2f}" '
                f'cy="{yp(row[system]):.2f}" r="2.5" fill="{color}"/>')

    parts.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="{foreground}"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{height-bottom}" stroke="{foreground}"/>',
        f'<text x="{left+plot_width/2:.2f}" y="{height-18}" '
        'text-anchor="middle">SLO scale '
        f'(threshold = scale × {args.base_slo_ms:g} ms)</text>',
        f'<text x="18" y="{top+plot_height/2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top+plot_height/2:.2f})">'
        'SLO attainment</text>',
    ])
    legend_x, legend_y = left + 12, top + 14
    for index, (system, color) in enumerate(zip(SYSTEMS, colors)):
        y = legend_y + index * 21
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{legend_x+32}" y="{y+4}">'
            f'{html.escape(LABELS[system])}</text>')
    parts.extend(["</g>", "</svg>"])
    if args.html_fragment:
        parts.append("</div>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
