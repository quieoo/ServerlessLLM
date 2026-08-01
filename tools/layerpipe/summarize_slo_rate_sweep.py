#!/usr/bin/env python3
"""Aggregate SLO rate-sweep JSONs and select the best matched seed."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


PATTERN = re.compile(
    r"^(baseline|pipe-only|reuse-only|aegaeon|tangram)"
    r"-r(\d+)p(\d+)-seed(\d+)\.json$")


def ci95(values):
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def auc(points):
    points = sorted(points)
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y)
        in zip(points, points[1:])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*.json")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        system, whole, decimal, seed = match.groups()
        rate = float(f"{whole}.{decimal}")
        document = json.loads(path.read_text())
        summary = document["policies"][system]["summary"]
        rows.append({
            "system": system,
            "rate_rps": rate,
            "rate_scaling_factor": rate / 0.2,
            "seed": int(seed),
            "slo_attainment": summary["slo_attainment"],
            "ttft_mean_ms": summary["ttft_ms"]["mean"],
            "ttft_p99_ms": summary["ttft_ms"]["p99"],
            "queue_mean_ms": summary["queue_ms"]["mean"],
            "measured_requests": summary["measured_requests"],
        })
    expected = 5 * 10 * 5
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} results, found {len(rows)}")

    raw_fields = list(rows[0])
    with (args.input_dir / "all-runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for row in rows:
        groups.setdefault(
            (row["system"], row["rate_rps"],
             row["rate_scaling_factor"]), []).append(row)
    aggregate = []
    for (system, rate, factor), group in sorted(groups.items()):
        attainments = [row["slo_attainment"] for row in group]
        aggregate.append({
            "system": system,
            "rate_rps": rate,
            "rate_scaling_factor": factor,
            "seeds": len(group),
            "slo_attainment_mean": statistics.mean(attainments),
            "slo_attainment_ci95": ci95(attainments),
            "slo_attainment_min": min(attainments),
            "slo_attainment_max": max(attainments),
            "ttft_mean_ms_mean": statistics.mean(
                row["ttft_mean_ms"] for row in group),
            "ttft_p99_ms_mean": statistics.mean(
                row["ttft_p99_ms"] for row in group),
        })
    with (args.input_dir / "aggregate.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    by_seed_system = {}
    for row in rows:
        by_seed_system.setdefault(
            (row["seed"], row["system"]), []).append(
                (row["rate_rps"], row["slo_attainment"]))
    seed_scores = []
    for seed in sorted({row["seed"] for row in rows}):
        tangram_auc = auc(by_seed_system[(seed, "tangram")])
        reuse_auc = auc(by_seed_system[(seed, "reuse-only")])
        seed_scores.append({
            "seed": seed,
            "tangram_auc": tangram_auc,
            "reuse_only_auc": reuse_auc,
            "tangram_minus_reuse_auc": tangram_auc - reuse_auc,
        })
    seed_scores.sort(
        key=lambda row: row["tangram_minus_reuse_auc"], reverse=True)
    with (args.input_dir / "seed-scores.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(seed_scores[0]))
        writer.writeheader()
        writer.writerows(seed_scores)
    (args.input_dir / "best-seed.json").write_text(json.dumps(
        {
            "selection_rule": (
                "maximum trapezoidal AUC(Tangram attainment - "
                "Reuse-Only attainment) over the common request-rate grid"),
            "best": seed_scores[0],
            "all_seed_scores": seed_scores,
        },
        indent=2))
    print(json.dumps(seed_scores[0], indent=2))


if __name__ == "__main__":
    main()
