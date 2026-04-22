#!/usr/bin/env python3
"""Analyze model reuse distances in a generated ServeGen trace.

The input trace is expected to contain:
    timestamp model_id input_tokens output_tokens

For each model request, the reuse distance is the number of distinct other
models accessed since the previous request to the same model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[2]
DEFAULT_TRACE = WORKSPACE_ROOT / "ServerlessLLM" / "evaluation" / "traces" / "servegen_tangram.trace"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "ServerlessLLM" / "evaluation" / "traces" / "servegen_trace_pattern_cdf.png"


@dataclass(frozen=True)
class TraceRequest:
    timestamp: float
    model_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print reuse-distance statistics and plot CDFs for a Tangram trace."
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-print-counts",
        type=int,
        default=20,
        help="Maximum number of interval-count entries to print per model.",
    )
    return parser.parse_args()


def has_header(parts: Sequence[str]) -> bool:
    if not parts:
        return False
    try:
        float(parts[0])
        int(parts[1])
    except (IndexError, ValueError):
        return True
    return False


def read_trace(trace_path: Path) -> List[TraceRequest]:
    requests: List[TraceRequest] = []

    with trace_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()
            if has_header(parts):
                if line_no == 1:
                    continue
                raise ValueError(f"Invalid trace row at line {line_no}: {line}")
            if len(parts) < 2:
                raise ValueError(f"Trace row has fewer than 2 columns at line {line_no}: {line}")

            requests.append(TraceRequest(timestamp=float(parts[0]), model_id=int(parts[1])))

    requests.sort(key=lambda req: (req.timestamp, req.model_id))
    return requests


def compute_reuse_distances(requests: Sequence[TraceRequest]) -> Dict[int, List[int]]:
    """Compute per-model reuse distances between repeated accesses.

    The first request for each model has no previous access, so it is excluded.
    """
    reuse_distances_by_model: Dict[int, List[int]] = {}
    seen_since_last: Dict[int, set[int]] = {}

    for req in requests:
        model_id = req.model_id
        reuse_distances_by_model.setdefault(model_id, [])

        if model_id in seen_since_last:
            reuse_distances_by_model[model_id].append(len(seen_since_last[model_id]))
            seen_since_last[model_id].clear()
        else:
            seen_since_last[model_id] = set()

        for other_model_id, seen_models in seen_since_last.items():
            if other_model_id != model_id:
                seen_models.add(model_id)

    return reuse_distances_by_model


def count_reuse_distances(values: Iterable[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def counts_to_cdf(counts: Dict[int, int]) -> Tuple[List[int], List[float]]:
    if not counts:
        return [], []

    intervals = sorted(counts.keys())
    total = sum(counts.values())
    cumulative = 0
    cdf: List[float] = []
    for interval in intervals:
        cumulative += counts[interval]
        cdf.append(cumulative / total)
    return intervals, cdf


def average(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def print_stats(reuse_distances_by_model: Dict[int, List[int]], max_print_counts: int) -> None:
    all_reuse_distances = [
        distance
        for distances in reuse_distances_by_model.values()
        for distance in distances
    ]

    print("Model reuse distance statistics:")
    print(
        "Average reuse distance: "
        f"{average(all_reuse_distances):.6f} "
        f"({len(all_reuse_distances)} repeated-access samples)"
    )

    for model_id in sorted(reuse_distances_by_model):
        reuse_distances = reuse_distances_by_model[model_id]
        counts = count_reuse_distances(reuse_distances)
        preview_items = list(counts.items())[:max_print_counts]
        preview = dict(preview_items)
        suffix = "" if len(counts) <= max_print_counts else f" ... ({len(counts)} bins total)"

        print(f"\nModel {model_id}:")
        print(f"  Samples: {len(reuse_distances)}")
        print(f"  Average reuse distance: {average(reuse_distances):.6f}")
        print(f"  Reuse distance counts: {preview}{suffix}")


def plot_reuse_distance_cdfs(
    reuse_distances_by_model: Dict[int, List[int]],
    output_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("\nmatplotlib is not installed; skipped reuse-distance CDF plot.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for model_id in sorted(reuse_distances_by_model):
        counts = count_reuse_distances(reuse_distances_by_model[model_id])
        values, cdf = counts_to_cdf(counts)
        if not values:
            continue
        plt.plot(
            values,
            cdf,
            marker="o",
            markersize=3,
            linewidth=1.8,
            alpha=0.9,
            label=f"Model {model_id}",
        )

    plt.xlabel("Reuse Distance")
    plt.ylabel("CDF")
    plt.title("Model Reuse Distance Pattern")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    requests = read_trace(args.trace)
    if not requests:
        raise ValueError(f"No requests found in trace: {args.trace}")

    reuse_distances_by_model = compute_reuse_distances(requests)
    print(f"Loaded {len(requests)} requests from {args.trace}")
    print_stats(reuse_distances_by_model, args.max_print_counts)
    if plot_reuse_distance_cdfs(reuse_distances_by_model, args.output):
        print(f"\nSaved reuse-distance CDF plot to {args.output}")


if __name__ == "__main__":
    main()
