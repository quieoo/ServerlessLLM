#!/usr/bin/env python3
"""Run paired LayerPipe residual-stall measurements across input scales."""

import argparse
import csv
import html
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "configs/servegen_8_models_layerpipe_l40_pool42.json"
)
DEFAULT_TRACE = REPO_ROOT / "evaluation/traces/servegen_tangram.trace"
COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#000000", "#7A4EAB",
)


def parse_scales(value):
    scales = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        scale = float(part)
        if not math.isfinite(scale) or scale <= 0:
            raise argparse.ArgumentTypeError(
                "input scales must be finite positive numbers"
            )
        scales.append(scale)
    if not scales:
        raise argparse.ArgumentTypeError("at least one input scale is required")
    return scales


def scale_tag(scale):
    return f"{scale:g}".replace(".", "p")


def count_trace_requests(path):
    count = 0
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def valid_result(path, scale):
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("backend") == "layerpipe_paired_residual"
        and float(result.get("input_scale", -1)) == scale
        and bool(result.get("pairs"))
    )


def run_logged(command, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN={' '.join(command)}", flush=True)
    with log_path.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if "Traceback" in line or "Error" in line:
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_sweep(args, scales):
    max_requests = args.max_requests
    if max_requests == 0:
        max_requests = count_trace_requests(args.trace)
    outputs = []
    for scale in scales:
        tag = scale_tag(scale)
        output = args.output_dir / f"paired-residual-input-scale-{tag}.json"
        log = args.output_dir / f"paired-residual-input-scale-{tag}.log"
        outputs.append((scale, output))
        if not args.force and valid_result(output, scale):
            print(f"RESUME={output}", flush=True)
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        command = [
            sys.executable,
            "tools/layerpipe/paired_residual_stall_bench.py",
            "--config", str(args.config),
            "--trace", str(args.trace),
            "--device", "0",
            "--max-requests", str(max_requests),
            "--max-pairs-per-model", str(args.max_pairs_per_model),
            "--input-scale", f"{scale:g}",
            "--truncate-input-to-model-limit",
            "--output", str(output),
        ]
        run_logged(command, env, log)
    return outputs


def load_model_labels(config_path):
    config = json.loads(config_path.read_text())
    return {
        int(item["id"]): item.get("pattern", f"model-{item['id']}")
        for item in config["model_lists"]
    }


def load_residual_samples(outputs):
    rows = []
    grouped = defaultdict(list)
    for scale, path in outputs:
        result = json.loads(path.read_text())
        for pair in result["pairs"]:
            value = float(pair["residual_loading_stall_ms"])
            row = dict(pair)
            rows.append(row)
            grouped[(scale, row["model_id"])].append(value)
    return rows, grouped


def write_cdf(path, rows):
    samples = defaultdict(list)
    for row in rows:
        samples[(row["input_scale"], row["model_id"])].append(row)
    fields = list(rows[0]) + ["sample_rank", "sample_count", "cdf"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(samples):
            ordered = sorted(
                samples[key],
                key=lambda item: item["residual_loading_stall_ms"],
            )
            count = len(ordered)
            for rank, row in enumerate(ordered, 1):
                writer.writerow({
                    **row,
                    "sample_rank": rank,
                    "sample_count": count,
                    "cdf": rank / count,
                })


def percentile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def write_summary(path, grouped):
    fields = [
        "input_scale", "model_id", "count", "mean_ms",
        "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (scale, model_id), values in sorted(grouped.items()):
            ordered = sorted(values)
            writer.writerow({
                "input_scale": scale,
                "model_id": model_id,
                "count": len(ordered),
                "mean_ms": statistics.mean(ordered),
                "p50_ms": percentile(ordered, 0.50),
                "p90_ms": percentile(ordered, 0.90),
                "p95_ms": percentile(ordered, 0.95),
                "p99_ms": percentile(ordered, 0.99),
                "max_ms": ordered[-1],
            })


def svg_text(x, y, value, size=12, anchor="start", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" {extra}>{html.escape(str(value))}</text>'
    )


def nice_max(value):
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    step = 1 if normalized <= 1 else 2 if normalized <= 2 else 5
    if normalized > 5:
        step = 10
    return step * magnitude


def write_svg(path, scales, grouped, labels):
    columns = min(4, len(scales))
    rows = math.ceil(len(scales) / columns)
    panel_w, panel_h = 430, 330
    left, right, top, bottom = 64, 18, 42, 52
    width, height = columns * panel_w, rows * panel_h + 66
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans,Arial,sans-serif" fill="#222">',
    ]
    for panel, scale in enumerate(scales):
        col, row = panel % columns, panel // columns
        ox, oy = col * panel_w, row * panel_h
        x0, x1 = ox + left, ox + panel_w - right
        y0, y1 = oy + top, oy + panel_h - bottom
        model_values = {
            model_id: sorted(values)
            for (item_scale, model_id), values in grouped.items()
            if item_scale == scale
        }
        maximum = nice_max(max(
            (values[-1] for values in model_values.values()), default=1.0
        ))
        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
            'fill="#fafafa" stroke="#aaa"/>'
        )
        for tick in range(6):
            fraction = tick / 5
            x = x0 + fraction * (x1 - x0)
            y = y1 - fraction * (y1 - y0)
            parts.append(
                f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" '
                'stroke="#e4e4e4"/>'
            )
            parts.append(
                f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" '
                'stroke="#e4e4e4"/>'
            )
            parts.append(svg_text(x, y1 + 18, f"{maximum*fraction:g}",
                                  10, "middle"))
            parts.append(svg_text(x0 - 8, y + 4, f"{fraction:.1f}",
                                  10, "end"))
        for model_id, values in sorted(model_values.items()):
            points = [(x0, y1)]
            count = len(values)
            for index, value in enumerate(values, 1):
                x = x0 + min(value / maximum, 1.0) * (x1 - x0)
                y_previous = y1 - ((index - 1) / count) * (y1 - y0)
                y_current = y1 - (index / count) * (y1 - y0)
                points.extend(((x, y_previous), (x, y_current)))
            path_data = " ".join(
                ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
                for index, (x, y) in enumerate(points)
            )
            color = COLORS[model_id % len(COLORS)]
            parts.append(
                f'<path d="{path_data}" fill="none" stroke="{color}" '
                'stroke-width="1.7"/>'
            )
        parts.append(svg_text(
            (x0 + x1) / 2, oy + 24, f"INPUT_SCALE={scale:g}",
            15, "middle", 'font-weight="bold"'
        ))
        parts.append(svg_text(
            (x0 + x1) / 2, y1 + 38, "Residual loading stall (ms)",
            11, "middle"
        ))
        parts.append(
            f'<text x="{ox+16}" y="{(y0+y1)/2}" font-size="11" '
            f'text-anchor="middle" transform="rotate(-90 {ox+16} '
            f'{(y0+y1)/2})">CDF</text>'
        )
    legend_y = rows * panel_h + 28
    legend_width = min(190, width / max(1, len(labels)))
    legend_start = max(12, (width - legend_width * len(labels)) / 2)
    for index, (model_id, label) in enumerate(sorted(labels.items())):
        x = legend_start + index * legend_width
        color = COLORS[model_id % len(COLORS)]
        parts.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x+24}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(svg_text(
            x + 30, legend_y + 4, f"model-{model_id}: {label}", 10
        ))
    parts.extend([
        svg_text(
            width / 2, height - 10,
            "Residual stall = max(0, matched cold service TTFT - warm service TTFT)",
            10, "middle", 'fill="#555"'
        ),
        "</g></svg>",
    ])
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--input-scales", type=parse_scales,
        default=parse_scales("1,2,4,8"),
        help="Comma-separated positive scales (default: 1,2,4,8).",
    )
    parser.add_argument("--max-requests", type=int, default=1000)
    parser.add_argument(
        "--max-pairs-per-model", type=int, default=0,
        help="Limit measured pairs per model; zero keeps all selected requests.",
    )
    parser.add_argument(
        "--trace-time-scale", type=float, default=150.0,
        help="Accepted for command compatibility; paired replay ignores arrivals.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "docs/residual-stall-input-scale-sweep",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Do not run benchmarks; analyze existing JSON files.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun scales whose valid JSON result already exists.",
    )
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be non-negative")
    if args.max_pairs_per_model < 0:
        parser.error("--max-pairs-per-model must be non-negative")
    if args.trace_time_scale < 0:
        parser.error("--trace-time-scale must be non-negative")

    args.config = args.config.resolve()
    args.trace = args.trace.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scales = args.input_scales
    outputs = [
        (
            scale,
            args.output_dir
            / f"paired-residual-input-scale-{scale_tag(scale)}.json",
        )
        for scale in scales
    ]
    if not args.analyze_only:
        outputs = run_sweep(args, scales)
    missing = [str(path) for _, path in outputs if not valid_result(path, _)]
    if missing:
        raise FileNotFoundError(
            "Missing or incompatible results:\n  " + "\n  ".join(missing)
        )

    rows, grouped = load_residual_samples(outputs)
    if not rows:
        raise ValueError("No paired residual-stall samples found")
    cdf_path = args.output_dir / "residual-stall-cdf.csv"
    summary_path = args.output_dir / "residual-stall-summary.csv"
    svg_path = args.output_dir / "residual-stall-cdf.svg"
    write_cdf(cdf_path, rows)
    write_summary(summary_path, grouped)
    write_svg(svg_path, scales, grouped, load_model_labels(args.config))
    print(f"CDF_CSV={cdf_path}")
    print(f"SUMMARY_CSV={summary_path}")
    print(f"CDF_FIGURE={svg_path}")


if __name__ == "__main__":
    main()
