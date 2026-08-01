#!/usr/bin/env python3
"""Summarize paired PSE-predicted and GPU-measured exposed load stalls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return (values[lower] * (upper - position)
            + values[upper] * (position - lower))


def pearson(left, right):
    if len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def classify(metric):
    route = metric.get("routing") or {}
    chosen = route.get("chosen_device")
    estimates = route.get("candidate_estimates") or {}
    estimate = estimates.get(str(chosen), estimates.get(chosen, {}))
    if estimate:
        missing = int(estimate.get("missing_pages", 0))
        required = int(estimate.get("required_pages", 0))
        if missing == 0:
            return "full-hit"
        if required > 0 and missing >= required:
            return "full-miss"
        return "partial-hit"
    layerweave = metric.get("layerweave") or {}
    if int(layerweave.get("to_load_bytes", 0)) == 0:
        return "full-hit"
    if int(layerweave.get("cached_bytes", 0)) == 0:
        return "full-miss"
    return "partial-hit"


def paired_rows(document):
    rows = []
    for metric in document.get("batch_metrics", []):
        predicted = metric.get("predicted_exposed_load_ms")
        measured = metric.get("measured_exposed_load_ms")
        if predicted is None or measured is None:
            continue
        predicted = float(predicted)
        measured = float(measured)
        rows.append({
            "request_id": metric.get("request_id"),
            "batch_id": metric.get("batch_id"),
            "model_id": int(metric["model_id"]),
            "residency_class": classify(metric),
            "predicted_exposed_load_ms": predicted,
            "measured_exposed_load_ms": measured,
            "signed_error_ms": predicted - measured,
            "absolute_error_ms": abs(predicted - measured),
        })
    return rows


def summarize(rows, mape_floor_ms=1.0):
    predicted = [row["predicted_exposed_load_ms"] for row in rows]
    measured = [row["measured_exposed_load_ms"] for row in rows]
    signed = [row["signed_error_ms"] for row in rows]
    absolute = [row["absolute_error_ms"] for row in rows]
    relative = [
        row["absolute_error_ms"] / row["measured_exposed_load_ms"]
        for row in rows
        if row["measured_exposed_load_ms"] >= mape_floor_ms
    ]
    return {
        "count": len(rows),
        "predicted_mean_ms": statistics.fmean(predicted) if rows else 0.0,
        "measured_mean_ms": statistics.fmean(measured) if rows else 0.0,
        "bias_ms": statistics.fmean(signed) if rows else 0.0,
        "mae_ms": statistics.fmean(absolute) if rows else 0.0,
        "median_absolute_error_ms": percentile(absolute, 0.5),
        "p95_absolute_error_ms": percentile(absolute, 0.95),
        "pearson": pearson(predicted, measured),
        "mape_floor_ms": mape_floor_ms,
        "mape_count": len(relative),
        "mape": statistics.fmean(relative) if relative else 0.0,
    }


def analyze(document, mape_floor_ms=1.0):
    rows = paired_rows(document)
    by_model = defaultdict(list)
    by_class = defaultdict(list)
    for row in rows:
        by_model[row["model_id"]].append(row)
        by_class[row["residency_class"]].append(row)
    return rows, {
        "overall": summarize(rows, mape_floor_ms),
        "by_model": {
            str(key): summarize(values, mape_floor_ms)
            for key, values in sorted(by_model.items())
        },
        "by_residency_class": {
            key: summarize(values, mape_floor_ms)
            for key, values in sorted(by_class.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mape-floor-ms", type=float, default=1.0)
    args = parser.parse_args()
    document = json.loads(args.input.read_text())
    rows, summary = analyze(document, args.mape_floor_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [
            "request_id", "batch_id", "model_id", "residency_class",
            "predicted_exposed_load_ms", "measured_exposed_load_ms",
            "signed_error_ms", "absolute_error_ms",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
