#!/usr/bin/env python3
"""Build per-model service-TTFT ECDF and summary CSV files."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(ordered, percent):
    """Return a linearly interpolated percentile from sorted values."""
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def read_samples(path):
    samples = defaultdict(list)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"model_id", "service_ttft_ms"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for line_no, row in enumerate(reader, 2):
            model_id = row["model_id"].strip()
            if not model_id:
                raise ValueError(f"{path}:{line_no}: empty model_id")
            try:
                value = float(row["service_ttft_ms"])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_no}: invalid service_ttft_ms "
                    f"{row['service_ttft_ms']!r}"
                ) from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{path}:{line_no}: service_ttft_ms must be a "
                    f"finite non-negative number, got {value}"
                )
            samples[model_id].append(value)

    if not samples:
        raise ValueError(f"{path} contains no samples")
    return samples


def model_sort_key(model_id):
    try:
        return 0, int(model_id)
    except ValueError:
        return 1, model_id


def write_ecdf(path, samples):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model_id",
                "service_ttft_ms",
                # "sample_rank",
                # "sample_count",
                "cdf",
            ],
        )
        writer.writeheader()
        for model_id in sorted(samples, key=model_sort_key):
            ordered = sorted(samples[model_id])
            count = len(ordered)
            for rank, value in enumerate(ordered, 1):
                writer.writerow({
                    "model_id": model_id,
                    "service_ttft_ms": f"{value:.9f}",
                    # "sample_rank": rank,
                    # "sample_count": count,
                    "cdf": f"{rank / count:.9f}",
                })


def write_summary(path, samples):
    fields = [
        "model_id",
        "count",
        "mean_ms",
        "stddev_ms",
        "min_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_id in sorted(samples, key=model_sort_key):
            ordered = sorted(samples[model_id])
            row = {
                "model_id": model_id,
                "count": len(ordered),
                "mean_ms": statistics.mean(ordered),
                "stddev_ms": (
                    statistics.stdev(ordered) if len(ordered) > 1 else 0.0
                ),
                "min_ms": ordered[0],
                "p50_ms": percentile(ordered, 50),
                "p90_ms": percentile(ordered, 90),
                "p95_ms": percentile(ordered, 95),
                "p99_ms": percentile(ordered, 99),
                "max_ms": ordered[-1],
            }
            writer.writerow({
                key: f"{value:.9f}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Group service_ttft_ms samples by model_id and emit a "
            "plot-ready empirical CDF plus percentile summaries."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input CSV containing model_id and service_ttft_ms columns.",
    )
    parser.add_argument(
        "--cdf-output",
        type=Path,
        help="ECDF output path (default: INPUT stem + .cdf.csv).",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Summary output path (default: INPUT stem + .summary.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cdf_output = args.cdf_output or args.input.with_name(
        f"{args.input.stem}.cdf.csv"
    )
    summary_output = args.summary_output or args.input.with_name(
        f"{args.input.stem}.summary.csv"
    )
    cdf_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    samples = read_samples(args.input)
    write_ecdf(cdf_output, samples)
    write_summary(summary_output, samples)

    print(
        f"Read {sum(map(len, samples.values()))} samples "
        f"across {len(samples)} models"
    )
    for model_id in sorted(samples, key=model_sort_key):
        print(f"model_id={model_id}: {len(samples[model_id])} samples")
    print(f"CDF: {cdf_output}")
    print(f"Summary: {summary_output}")


if __name__ == "__main__":
    main()
