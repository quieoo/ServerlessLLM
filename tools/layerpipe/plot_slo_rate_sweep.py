#!/usr/bin/env python3
"""Plot five-system SLO attainment over offered request rate."""

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


def load_rows(directory):
    with (directory / "aggregate.csv").open(newline="") as stream:
        aggregate = list(csv.DictReader(stream))
    best = json.loads((directory / "best-seed.json").read_text())["best"]
    with (directory / "all-runs.csv").open(newline="") as stream:
        selected = [
            row for row in csv.DictReader(stream)
            if int(row["seed"]) == best["seed"]
        ]
    return aggregate, selected, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-fragment", action="store_true")
    args = parser.parse_args()
    aggregate, selected, best = load_rows(args.input_dir)

    width, height = 1040, 470
    left, right, top, bottom, gap = 68, 20, 52, 58, 58
    panel_width = (width - left - right - gap) / 2
    panel_height = height - top - bottom
    x_min, x_max = 0.08, 0.22

    def xp(rate, panel):
        return (
            left + panel * (panel_width + gap)
            + (rate - x_min) / (x_max - x_min) * panel_width
        )

    def yp(value):
        return top + (1.0 - value) * panel_height

    if args.html_fragment:
        colors = tuple(
            f"var(--viz-series-{index})" for index in range(1, 6))
        grid = "var(--border)"
        foreground = "var(--foreground)"
        muted = "var(--muted-foreground)"
        parts = [
            '<div id="slo-rate-sweep" style="width:100%">',
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="SLO attainment versus request rate for five systems" '
            f'viewBox="0 0 {width} {height}" '
            'style="display:block;width:100%;height:auto">',
            f'<g font-family="sans-serif" font-size="12" fill="{foreground}">',
        ]
    else:
        colors = COLORS
        grid, foreground, muted = "#dddddd", "#222222", "#666666"
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<g font-family="sans-serif" font-size="12" fill="{foreground}">',
        ]

    panel_titles = (
        "5-seed mean with 95% CI",
        f"Selected seed {best['seed']} (maximum Tangram−Reuse AUC)",
    )
    for panel, title in enumerate(panel_titles):
        x0 = left + panel * (panel_width + gap)
        parts.append(
            f'<text x="{x0 + panel_width / 2:.2f}" y="24" '
            f'text-anchor="middle" font-weight="500">{html.escape(title)}</text>')
        for value in (0, 0.25, 0.5, 0.75, 1.0):
            y = yp(value)
            parts.append(
                f'<line x1="{x0:.2f}" y1="{y:.2f}" '
                f'x2="{x0 + panel_width:.2f}" y2="{y:.2f}" '
                f'stroke="{grid}" stroke-width="1"/>')
            if panel == 0:
                parts.append(
                    f'<text x="{x0 - 9:.2f}" y="{y + 4:.2f}" '
                    f'text-anchor="end">{value:.2f}</text>')
        for rate in (0.08, 0.12, 0.16, 0.18, 0.20, 0.22):
            x = xp(rate, panel)
            parts.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{top + panel_height}" stroke="{grid}" stroke-width="1"/>')
            parts.append(
                f'<text x="{x:.2f}" y="{top + panel_height + 22}" '
                f'text-anchor="middle">{rate:.2f}</text>')
        parts.append(
            f'<line x1="{x0:.2f}" y1="{top + panel_height}" '
            f'x2="{x0 + panel_width:.2f}" y2="{top + panel_height}" '
            f'stroke="{foreground}"/>')
        parts.append(
            f'<line x1="{x0:.2f}" y1="{top}" x2="{x0:.2f}" '
            f'y2="{top + panel_height}" stroke="{foreground}"/>')
        parts.append(
            f'<text x="{x0 + panel_width / 2:.2f}" y="{height - 13}" '
            'text-anchor="middle">Offered request rate (requests/s)</text>')
    parts.append(
        f'<text x="17" y="{top + panel_height / 2:.2f}" '
        f'text-anchor="middle" transform="rotate(-90 17 '
        f'{top + panel_height / 2:.2f})">SLO attainment</text>')

    for system, color in zip(SYSTEMS, colors):
        group = sorted(
            (
                float(row["rate_rps"]),
                float(row["slo_attainment_mean"]),
                float(row["slo_attainment_ci95"]),
            )
            for row in aggregate if row["system"] == system
        )
        upper = [
            (xp(rate, 0), yp(min(1, mean + ci)))
            for rate, mean, ci in group
        ]
        lower = [
            (xp(rate, 0), yp(max(0, mean - ci)))
            for rate, mean, ci in reversed(group)
        ]
        polygon = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in upper + lower)
        parts.append(
            f'<polygon points="{polygon}" fill="{color}" opacity="0.10"/>')
        mean_points = " ".join(
            f"{xp(rate, 0):.2f},{yp(mean):.2f}"
            for rate, mean, _ in group
        )
        parts.append(
            f'<polyline points="{mean_points}" fill="none" '
            f'stroke="{color}" stroke-width="2.2"/>')
        selected_group = sorted(
            (float(row["rate_rps"]), float(row["slo_attainment"]))
            for row in selected if row["system"] == system
        )
        selected_points = " ".join(
            f"{xp(rate, 1):.2f},{yp(value):.2f}"
            for rate, value in selected_group
        )
        parts.append(
            f'<polyline points="{selected_points}" fill="none" '
            f'stroke="{color}" stroke-width="2.2"/>')
        for panel, values in (
                (0, [(rate, mean) for rate, mean, _ in group]),
                (1, selected_group)):
            for rate, value in values:
                parts.append(
                    f'<circle cx="{xp(rate, panel):.2f}" '
                    f'cy="{yp(value):.2f}" r="2.5" fill="{color}"/>')

    legend_y = 41
    legend_width = 122
    legend_x = width / 2 - len(SYSTEMS) * legend_width / 2
    for index, (system, color) in enumerate(zip(SYSTEMS, colors)):
        x = legend_x + index * legend_width
        parts.append(
            f'<line x1="{x:.2f}" y1="{legend_y}" '
            f'x2="{x + 22:.2f}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{x + 29:.2f}" y="{legend_y + 4}">'
            f'{html.escape(LABELS[system])}</text>')
    parts.extend(["</g>", "</svg>"])
    if args.html_fragment:
        parts.append("</div>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
