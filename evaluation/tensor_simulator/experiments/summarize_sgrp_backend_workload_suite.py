#!/usr/bin/env python3
"""Summarize three memory backends over the six workload-suite traces."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


WORKLOADS = (
    ("servegen", "ServeGen"), ("cyclic", "Balanced cyclic"),
    ("reuse-d2", "Reuse D=2"), ("reuse-d7", "Reuse D=7"),
    ("small-hot", "Small models hot"),
    ("large-hot", "Large models hot"),
)
BACKENDS = (
    ("segment", "Segment", "#4c78a8"),
    ("tensor-page", "Tensor-page", "#f58518"),
    ("compact-page", "Compact-page", "#54a24b"),
)


def quantile(values, fraction):
    values = sorted(values)
    position = fraction * (len(values) - 1)
    low = int(position); high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def metrics(rows):
    exposed = [row["exposed_load_ms"] for row in rows]
    ttft = [row["critical_path_ms"] for row in rows]
    for row in rows:
        assert row["map_calls"] == row["mapped_pages"]
        assert row["unmap_calls"] == row["unmapped_pages"]
    return {
        "mean_exposed_load_ms": statistics.fmean(exposed),
        "p95_exposed_load_ms": quantile(exposed, .95),
        "mean_ttft_ms": statistics.fmean(ttft),
        "p95_ttft_ms": quantile(ttft, .95),
        "exposed_h2d_ms_per_request": statistics.fmean(
            row["exposed_h2d_ms"] for row in rows),
        "exposed_compaction_ms_per_request": statistics.fmean(
            row["exposed_compaction_ms"] for row in rows),
        "exposed_page_map_ms_per_request": statistics.fmean(
            row["exposed_page_map_ms"] for row in rows),
        "exposed_page_unmap_ms_per_request": statistics.fmean(
            row["exposed_page_unmap_ms"] for row in rows),
        "exposed_page_map_unmap_ms_per_request": statistics.fmean(
            row["exposed_page_map_unmap_ms"] for row in rows),
        "exposed_allocation_ms_per_request": statistics.fmean(
            row["exposed_allocation_ms"] for row in rows),
        "h2d_gib_per_request": statistics.fmean(
            row["h2d_bytes"] for row in rows) / 1024**3,
        "vmm_ms_per_request": statistics.fmean(
            row["map_ms"] + row["unmap_ms"] for row in rows),
        "compaction_ms_per_request": statistics.fmean(
            row["compaction_ms"] for row in rows),
        "map_calls_per_request": statistics.fmean(
            row["map_calls"] for row in rows),
        "unmap_calls_per_request": statistics.fmean(
            row["unmap_calls"] for row in rows),
        "internal_fragmentation_mib": statistics.fmean(
            row["memory"]["internal_fragmentation_bytes"] for row in rows)
            / 1024**2,
        "external_fragmentation_ratio": statistics.fmean(
            row["memory"]["external_fragmentation_ratio"] for row in rows),
        "solver_ms_per_request": statistics.fmean(
            row["mckp_solver_time_ms"] for row in rows),
    }


def render_svg(rows, workloads):
    width, height = 1200, 510
    x0, y0, chart_w, chart_h = 75, 55, 1085, 350
    ymax = max(float(row["normalized_exposed_load"]) for row in rows) * 1.08
    ymin = min(.9, min(float(row["normalized_exposed_load"]) for row in rows) * .97)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{font-size:13px}.tick{font-size:11px}.legend{font-size:13px}.frame{fill:#fff;stroke:#444}.grid{stroke:#ddd}</style>',
        '<rect width="1200" height="510" fill="#fff"/>',
        f'<rect x="{x0}" y="{y0}" width="{chart_w}" height="{chart_h}" class="frame"/>',
    ]
    def sy(value): return y0 + chart_h * (ymax-value)/(ymax-ymin)
    for index in range(6):
        value = ymin + index*(ymax-ymin)/5; yy = sy(value)
        chunks += [f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+chart_w}" y2="{yy:.1f}" class="grid"/>',
                   f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>']
    group_w = chart_w / len(workloads); bar_w = group_w * .22
    for wi, (workload, label) in enumerate(workloads):
        center = x0 + group_w*(wi+.5)
        for bi, (backend, _, color) in enumerate(BACKENDS):
            row = next(r for r in rows if r["workload"]==workload and r["backend"]==backend)
            value = float(row["normalized_exposed_load"]); xx = center+(bi-1)*bar_w; yy=sy(value)
            chunks.append(f'<rect x="{xx-bar_w*.4:.1f}" y="{yy:.1f}" width="{bar_w*.8:.1f}" height="{sy(ymin)-yy:.1f}" fill="{color}"/>')
        chunks.append(f'<text x="{center:.1f}" y="{y0+chart_h+22}" text-anchor="middle" class="tick">{label}</text>')
    chunks.append(f'<text transform="translate(20,{y0+chart_h/2}) rotate(-90)" text-anchor="middle" class="axis">Normalized mean exposed loading stall</text>')
    for index, (_, label, color) in enumerate(BACKENDS):
        xx=280+index*240
        chunks += [f'<rect x="{xx}" y="465" width="20" height="12" fill="{color}"/>',
                   f'<text x="{xx+28}" y="476" class="legend">{label}</text>']
    chunks.append('</svg>')
    return "\n".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument(
        "--workloads",
        help="Optional comma-separated workload ids; defaults to all six.")
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    args = parser.parse_args()
    selected = (
        set(args.workloads.split(",")) if args.workloads else None)
    workloads = [
        item for item in WORKLOADS
        if selected is None or item[0] in selected
    ]
    if selected is not None and selected != {item[0] for item in workloads}:
        parser.error("unknown workload id")
    output = []
    for workload, workload_label in workloads:
        values = {}
        for backend, backend_label, _ in BACKENDS:
            document = json.loads((args.result_dir / workload / f"{backend}.json").read_text())
            rows = document["policies"]["minimal"]["requests"][args.warmup_requests:]
            values[backend] = metrics(rows)
        segment = values["segment"]["mean_exposed_load_ms"]
        for backend, backend_label, _ in BACKENDS:
            value = values[backend]
            output.append({
                "workload": workload, "workload_label": workload_label,
                "backend": backend, "backend_label": backend_label,
                **value,
                "normalized_exposed_load": value["mean_exposed_load_ms"] / segment,
                "exposed_load_delta_vs_segment_pct":
                    100 * (value["mean_exposed_load_ms"] - segment) / segment,
            })
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    args.svg_output.write_text(render_svg(output, workloads))


if __name__ == "__main__":
    main()
