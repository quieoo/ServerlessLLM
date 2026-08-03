#!/usr/bin/env python3
"""Summarize and plot the controlled distinct-reuse SGRP experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


TRACES = ("min", "mid-low", "mid-high", "max")
STRATEGIES = (
    ("cb-lru", "CB-LRU"),
    ("cb-suffix-lru", "CB-SuffixLRU"),
    ("cb-mckp", "CB-MCKP"),
    ("joint-mckp", "Joint-MCKP"),
)


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rows(rows):
    critical = [row["critical_path_ms"] for row in rows]
    exposed = [row["exposed_load_ms"] for row in rows]
    return {
        "requests": len(rows),
        "mean_ttft_ms": statistics.fmean(critical),
        "p50_ttft_ms": quantile(critical, .50),
        "p90_ttft_ms": quantile(critical, .90),
        "p95_ttft_ms": quantile(critical, .95),
        "p99_ttft_ms": quantile(critical, .99),
        "mean_exposed_load_ms": statistics.fmean(exposed),
        "p95_exposed_load_ms": quantile(exposed, .95),
        "h2d_gib": sum(row["h2d_bytes"] for row in rows) / 1024**3,
        "compaction_moved_gib": (
            sum(row["compaction_moved_bytes"] for row in rows) / 1024**3),
        "evicted_gib": sum(row["evicted_bytes"] for row in rows) / 1024**3,
    }


def load_results(trace_dir, result_dir, warmup):
    results = []
    token_hash = None
    counts = None
    common_prefix = None
    for trace_name in TRACES:
        metadata = json.loads(
            (trace_dir / f"{trace_name}.trace.json").read_text())
        metrics = metadata["metrics"]
        trace_hash = metadata["token_multiset_sha256"]
        trace_counts = metrics["counts"]
        sequence = [
            int(line.split()[1])
            for line in (trace_dir / f"{trace_name}.trace").read_text()
            .splitlines() if line.strip()
        ]
        if token_hash is None:
            token_hash = trace_hash
            counts = trace_counts
            common_prefix = sequence[:warmup]
        if trace_hash != token_hash or trace_counts != counts:
            raise ValueError(f"{trace_name}: unmatched counts or token shapes")
        if sequence[:warmup] != common_prefix:
            raise ValueError(f"{trace_name}: unmatched common prefix")
        if any(a == b for a, b in zip(sequence, sequence[1:])):
            raise ValueError(f"{trace_name}: consecutive repeated model")
        for strategy, label in STRATEGIES:
            document = json.loads(
                (result_dir / trace_name / f"{strategy}.json").read_text())
            rows = document["policies"]["minimal"]["requests"]
            if len(rows) != len(sequence):
                raise ValueError(
                    f"{trace_name}/{strategy}: request count mismatch")
            item = {
                "trace": trace_name,
                "strategy": strategy,
                "label": label,
                "distinct_mean": metrics["distinct_mean"],
                "distinct_p50": metrics["distinct_p50"],
                "distinct_p90": metrics["distinct_p90"],
                "distinct_p99": metrics["distinct_p99"],
                "request_gap_p50": metrics["request_gap_p50"],
                "request_gap_p90": metrics["request_gap_p90"],
                "request_gap_within_k_fraction":
                    metrics["request_gap_within_k_fraction"],
                "transition_entropy_bits":
                    metrics["transition_entropy_bits"],
                "routing": dict(Counter(document["routing"][warmup:])),
            }
            item.update(summarize_rows(rows[warmup:]))
            results.append(item)
    for trace_name in TRACES:
        baseline = next(
            row for row in results
            if row["trace"] == trace_name and row["strategy"] == "cb-lru")
        for row in results:
            if row["trace"] != trace_name:
                continue
            row["mean_ttft_improvement_vs_cb_lru_pct"] = (
                100.0
                * (baseline["mean_ttft_ms"] - row["mean_ttft_ms"])
                / baseline["mean_ttft_ms"])
    return results


def write_csv(path, results):
    fields = [
        key for key in results[0] if key not in ("label", "routing")
    ] + ["routing"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            output = dict(row)
            output["routing"] = json.dumps(row["routing"], sort_keys=True)
            output.pop("label")
            writer.writerow(output)


def svg_panel(chunks, results, metric, title, x0, y0, width, height):
    colors = ("#2563eb", "#d97706", "#059669", "#dc2626")
    xs = [row["distinct_mean"] for row in results]
    ys = [row[metric] for row in results]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    padding = max(1.0, (ymax - ymin) * 0.12)
    ymin = max(0.0, ymin - padding)
    ymax += padding

    def sx(value):
        return x0 + width * (value - xmin) / (xmax - xmin)

    def sy(value):
        return y0 + height * (1.0 - (value - ymin) / (ymax - ymin))

    chunks.append(
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 24}" '
        f'text-anchor="middle" class="title">{title}</text>')
    chunks.append(
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
        'class="frame"/>')
    for index in range(5):
        value = ymin + index * (ymax - ymin) / 4
        y = sy(value)
        chunks.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" '
            f'y2="{y:.1f}" class="grid"/>')
        chunks.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{value:.0f}</text>')
    trace_xs = sorted(set(xs))
    for value in trace_xs:
        x = sx(value)
        chunks.append(
            f'<text x="{x:.1f}" y="{y0 + height + 22}" '
            f'text-anchor="middle" class="tick">{value:.2f}</text>')
    for (strategy, label), color in zip(STRATEGIES, colors):
        rows = sorted(
            (row for row in results if row["strategy"] == strategy),
            key=lambda row: row["distinct_mean"])
        points = " ".join(
            f'{sx(row["distinct_mean"]):.1f},{sy(row[metric]):.1f}'
            for row in rows)
        chunks.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.5"/>')
        for row in rows:
            x = sx(row["distinct_mean"])
            y = sy(row[metric])
            chunks.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" '
                f'fill="{color}"/>')
    chunks.append(
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 50}" '
        'text-anchor="middle" class="axis">'
        'Mean distinct-model reuse distance</text>')


def write_svg(path, results):
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="520" '
        'viewBox="0 0 1180 520">',
        '<style>.title{font:600 16px sans-serif}.tick{font:12px '
        'sans-serif;fill:#374151}.axis{font:14px sans-serif}.frame{fill:white;'
        'stroke:#374151}.grid{stroke:#d1d5db;stroke-width:1}.legend{font:13px '
        'sans-serif}</style>',
        '<rect width="1180" height="520" fill="white"/>',
    ]
    svg_panel(
        chunks, results, "mean_ttft_ms", "Mean TTFT", 70, 70, 480, 330)
    svg_panel(
        chunks, results, "mean_exposed_load_ms",
        "Mean exposed loading", 660, 70, 480, 330)
    colors = ("#2563eb", "#d97706", "#059669", "#dc2626")
    for index, ((_, label), color) in enumerate(zip(STRATEGIES, colors)):
        x = 140 + index * 250
        chunks.append(
            f'<line x1="{x}" y1="485" x2="{x + 30}" y2="485" '
            f'stroke="{color}" stroke-width="3"/>')
        chunks.append(
            f'<text x="{x + 38}" y="489" class="legend">{label}</text>')
    chunks.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    args = parser.parse_args()
    results = load_results(
        args.trace_dir, args.result_dir, args.warmup_requests)
    write_csv(args.csv_output, results)
    write_svg(args.svg_output, results)
    print(args.csv_output)
    print(args.svg_output)


if __name__ == "__main__":
    main()
