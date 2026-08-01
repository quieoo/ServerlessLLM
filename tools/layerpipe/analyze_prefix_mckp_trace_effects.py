#!/usr/bin/env python3
"""Paired trace-feature analysis for Prefix-MCKP versus a baseline.

The two simulator outputs must contain the same request stream.  Positive
``saved_*_ms`` values mean that the candidate is faster than the baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


MIB = 1024 * 1024
GIB = 1024 * MIB


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize(rows):
    critical = [row["saved_critical_ms"] for row in rows]
    exposed = [row["saved_exposed_ms"] for row in rows]
    return {
        "requests": len(rows),
        "mean_saved_critical_ms": mean(critical) if critical else 0.0,
        "median_saved_critical_ms": median(critical) if critical else 0.0,
        "p90_saved_critical_ms": percentile(critical, 0.90),
        "mean_saved_exposed_ms": mean(exposed) if exposed else 0.0,
        "median_saved_exposed_ms": median(exposed) if exposed else 0.0,
        "p90_saved_exposed_ms": percentile(exposed, 0.90),
        "candidate_win_rate": (
            sum(value > 0 for value in critical) / len(critical)
            if critical else 0.0
        ),
        "baseline_cold_rate": (
            sum(row["baseline_exposed_ms"] > 0 for row in rows) / len(rows)
            if rows else 0.0
        ),
    }


def grouped_summary(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: summarize(values)
        for name, values in sorted(grouped.items())
    }


def reuse_bin(distance):
    if distance is None:
        return "cold"
    if distance <= 2:
        return "01-02"
    if distance <= 7:
        return "03-07"
    if distance <= 15:
        return "08-15"
    if distance <= 31:
        return "16-31"
    return "32+"


def size_bin(size_gib):
    if size_gib < 8:
        return "small_<8GiB"
    if size_gib < 20:
        return "medium_8-20GiB"
    return "large_>=20GiB"


def quartile_bin(value, cuts):
    if value <= cuts[0]:
        return "Q1"
    if value <= cuts[1]:
        return "Q2"
    if value <= cuts[2]:
        return "Q3"
    return "Q4"


def input_length_bin(tokens):
    if tokens <= 512:
        return "<=512"
    if tokens <= 2048:
        return "513-2048"
    if tokens <= 4096:
        return "2049-4096"
    return "4097+"


def load_requests(path):
    document = json.loads(path.read_text())
    policies = document.get("policies", {})
    if len(policies) != 1:
        raise ValueError(f"{path}: expected exactly one policy")
    return next(iter(policies.values()))["requests"]


def load_model_sizes(path):
    document = json.loads(path.read_text())
    return {
        int(model_id): item["logical_bytes"]
        for model_id, item in document["models"].items()
    }


def svg_escape(value):
    return (
        str(value).replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_svg(path, report):
    panels = [
        ("By model", report["by_model"], "model"),
        ("By reuse distance", report["by_reuse_distance"], "reuse"),
        ("By model-size class", report["by_model_size"], "size"),
        ("By effective input length", report["by_input_length"], "input"),
    ]
    width, panel_height = 1120, 260
    height = 70 + panel_height * len(panels)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}'
        '.title{font-size:22px;font-weight:700}.label{font-size:13px}'
        '.small{font-size:11px;fill:#555}.axis{stroke:#999;stroke-width:1}'
        '.zero{stroke:#444;stroke-width:1.2}</style>',
        '<text x="35" y="38" class="title">Prefix-MCKP paired TTFT savings '
        "(positive is better)</text>",
    ]
    for panel_index, (title, values, _) in enumerate(panels):
        top = 65 + panel_index * panel_height
        labels = list(values)
        scores = [values[label]["mean_saved_critical_ms"] for label in labels]
        max_abs = max([abs(value) for value in scores] + [1.0])
        plot_left, plot_right = 260, width - 55
        zero = (plot_left + plot_right) / 2
        half_width = (plot_right - plot_left) / 2
        row_height = min(27, 180 / max(1, len(labels)))
        parts.append(
            f'<text x="35" y="{top + 22}" class="label" '
            f'font-weight="700">{svg_escape(title)}</text>')
        parts.append(
            f'<line x1="{zero}" y1="{top + 34}" x2="{zero}" '
            f'y2="{top + 47 + row_height * len(labels)}" class="zero"/>')
        for index, label in enumerate(labels):
            item = values[label]
            score = item["mean_saved_critical_ms"]
            bar_width = abs(score) / max_abs * half_width
            x = zero if score >= 0 else zero - bar_width
            y = top + 43 + index * row_height
            color = "#238636" if score >= 0 else "#cf222e"
            parts.extend([
                f'<text x="{plot_left - 12}" y="{y + row_height * .65}" '
                f'text-anchor="end" class="label">{svg_escape(label)}</text>',
                f'<rect x="{x}" y="{y}" width="{bar_width}" '
                f'height="{row_height * .68}" fill="{color}" opacity=".82"/>',
                f'<text x="{x + bar_width + (6 if score >= 0 else -6)}" '
                f'y="{y + row_height * .62}" '
                f'text-anchor="{"start" if score >= 0 else "end"}" '
                f'class="small">{score:+.1f} ms; n={item["requests"]}</text>',
            ])
        parts.append(
            f'<text x="{zero}" y="{top + 238}" text-anchor="middle" '
            f'class="small">scale: +/- {max_abs:.1f} ms within panel</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tensor-layout", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-svg", type=Path)
    parser.add_argument("--hot-window", type=int, default=32)
    args = parser.parse_args()

    baseline = load_requests(args.baseline)
    candidate = load_requests(args.candidate)
    if len(baseline) != len(candidate):
        raise ValueError("request counts differ")
    sizes = load_model_sizes(args.tensor_layout)
    counts = Counter(row["model_id"] for row in baseline)
    per_model_tokens = defaultdict(list)
    for row in baseline:
        per_model_tokens[row["model_id"]].append(row["input_tokens"])
    token_cuts = {
        model_id: (
            percentile(values, .25),
            percentile(values, .50),
            percentile(values, .75),
        )
        for model_id, values in per_model_tokens.items()
    }

    last_position = {}
    window = []
    rows = []
    for index, (base, cand) in enumerate(zip(baseline, candidate)):
        identity = ("request_id", "model_id", "input_tokens", "output_tokens")
        if any(base[key] != cand[key] for key in identity):
            raise ValueError(f"request {index} is not paired")
        model_id = base["model_id"]
        previous = last_position.get(model_id)
        distance = None if previous is None else index - previous
        distinct_distance = (
            None if previous is None
            else len({item for item in window[previous + 1:]})
        )
        recent_count = sum(item == model_id for item in window[-args.hot_window:])
        size_gib = sizes[model_id] / GIB
        row = {
            "request_index": index,
            "request_id": base["request_id"],
            "model_id": model_id,
            "model_size_gib": size_gib,
            "model_size_bin": size_bin(size_gib),
            "model_frequency": counts[model_id],
            "recent_frequency": recent_count,
            "input_tokens": base["input_tokens"],
            "input_length_bin": input_length_bin(base["input_tokens"]),
            "input_quartile": quartile_bin(
                base["input_tokens"], token_cuts[model_id]),
            "reuse_distance": distance,
            "reuse_distance_distinct_models": distinct_distance,
            "reuse_bin": reuse_bin(distance),
            "baseline_gpu": base["gpu"],
            "candidate_gpu": cand["gpu"],
            "route_changed": base["gpu"] != cand["gpu"],
            "baseline_critical_ms": base["critical_path_ms"],
            "candidate_critical_ms": cand["critical_path_ms"],
            "saved_critical_ms": (
                base["critical_path_ms"] - cand["critical_path_ms"]),
            "baseline_exposed_ms": base["exposed_load_ms"],
            "candidate_exposed_ms": cand["exposed_load_ms"],
            "saved_exposed_ms": (
                base["exposed_load_ms"] - cand["exposed_load_ms"]),
            "baseline_evicted_gib": base["evicted_bytes"] / GIB,
            "candidate_evicted_gib": cand["evicted_bytes"] / GIB,
        }
        rows.append(row)
        last_position[model_id] = index
        window.append(model_id)

    report = {
        "format": "prefix-mckp-trace-effects-v1",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "tensor_layout": str(args.tensor_layout),
        "semantics": (
            "saved metrics are baseline minus candidate; positive is better"),
        "overall": summarize(rows),
        "route_changed": grouped_summary(rows, "route_changed"),
        "by_model": grouped_summary(rows, "model_id"),
        "by_model_size": grouped_summary(rows, "model_size_bin"),
        "by_reuse_distance": grouped_summary(rows, "reuse_bin"),
        "by_input_length": grouped_summary(rows, "input_length_bin"),
        "by_input_quartile": grouped_summary(rows, "input_quartile"),
        "by_size_and_reuse": grouped_summary(
            [
                dict(row, size_reuse=(
                    f'{row["model_size_bin"]}|{row["reuse_bin"]}'))
                for row in rows
            ],
            "size_reuse",
        ),
        "model_metadata": {
            str(model_id): {
                "logical_gib": sizes[model_id] / GIB,
                "requests": counts[model_id],
                "request_share": counts[model_id] / len(rows),
                "input_p25": token_cuts[model_id][0],
                "input_p50": token_cuts[model_id][1],
                "input_p75": token_cuts[model_id][2],
            }
            for model_id in sorted(counts)
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.output_svg:
        args.output_svg.parent.mkdir(parents=True, exist_ok=True)
        write_svg(args.output_svg, report)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
