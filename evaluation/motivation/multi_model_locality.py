#!/usr/bin/env python3
"""
Example: Generate merged workloads from multiple logical models and analyze locality.

This script demonstrates how to:
1. Organize all models in one list
2. Select the first `model_num` models to participate
3. Generate one workload per selected model and merge them by timestamp
4. Preserve the originating pool id for each request
5. Measure model access locality using access-interval counts and CDFs
"""

from dataclasses import dataclass
import csv
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from servegen import Category, ClientPool, generate_workload
from servegen.utils import get_bounded_rate_fn, get_constant_rate_fn


@dataclass
class MergedRequest:
    """A workload request annotated with its source pool."""

    request_id: int
    timestamp: float
    pool_id: int
    data: Dict[str, object]


@dataclass(frozen=True)
class ModelSpec:
    """Logical model specification used for multi-model generation."""

    pool_id: int
    model_name: str
    category: Category
    source_model: str
    rate_strategy: str
    rate_value: float | None
    seed: int
    time_offset: int


# The repository currently ships a smaller set of backing datasets than the
# production-model groups described in the paper. These logical models map
# to the available datasets so we can exercise an 8-model workflow.
ALL_MODELS: List[ModelSpec] = [
    ModelSpec(0, "lang-small-0", Category.LANGUAGE, "m-small", "bounded", None, 0, 0),
    ModelSpec(1, "lang-small-1", Category.LANGUAGE, "m-small", "bounded", None, 1, 3600),
    ModelSpec(2, "lang-mid-0", Category.LANGUAGE, "m-mid", "bounded", None, 2, 0),
    ModelSpec(3, "lang-mid-1", Category.LANGUAGE, "m-mid", "bounded", None, 3, 3600),
    ModelSpec(4, "lang-large-0", Category.LANGUAGE, "m-large", "bounded", None, 4, 0),
    ModelSpec(5, "lang-large-1", Category.LANGUAGE, "m-large", "bounded", None, 5, 3600),
    ModelSpec(6, "mm-image-0", Category.MULTIMODAL, "mm-image", "bounded", None, 6, 0),
    ModelSpec(7, "reason-r1-0", Category.REASON, "deepseek-r1", "bounded", None, 7, 0),
]

POOL_CACHE: Dict[Tuple[Category, str], ClientPool] = {}


def get_or_create_pool(category: Category, model: str) -> ClientPool:
    """Reuse loaded client pools for logical models sharing the same source."""
    cache_key = (category, model)
    if cache_key not in POOL_CACHE:
        POOL_CACHE[cache_key] = ClientPool(category, model, base_dir="../data")
    return POOL_CACHE[cache_key]


def get_peak_rate(view) -> float:
    """Compute the original peak aggregate rate for a view."""
    windows = view.get()
    totals_by_timestamp: Dict[int, float] = {}
    for window in windows:
        totals_by_timestamp.setdefault(window.timestamp, 0.0)
        if window.rate is not None:
            totals_by_timestamp[window.timestamp] += window.rate
    return max(totals_by_timestamp.values(), default=0.0)


def resolve_rate_value(view, rate_strategy: str, rate_value: float | None) -> float:
    """Use manual rate_value when provided, otherwise infer a realistic default."""
    if rate_value is not None:
        return rate_value
    if rate_strategy == "bounded":
        return get_peak_rate(view)
    raise ValueError(f"rate_value must be provided for strategy: {rate_strategy}")


def generate_pool_workload(
    pool_id: int,
    category: Category,
    model: str,
    view_start: int,
    view_end: int,
    rate_strategy: str,
    rate_value: float | None,
    seed: int,
) -> List[MergedRequest]:
    """Generate one workload for a single model pool using realistic views."""
    pool = get_or_create_pool(category, model)
    view = pool.span(view_start, view_end)
    duration = view_end - view_start
    effective_rate_value = resolve_rate_value(view, rate_strategy, rate_value)

    if rate_strategy == "constant":
        rate_fn = get_constant_rate_fn(view, effective_rate_value)
    elif rate_strategy == "bounded":
        rate_fn = get_bounded_rate_fn(view, effective_rate_value)
    else:
        raise ValueError(f"Unsupported rate strategy: {rate_strategy}")

    requests = generate_workload(view, rate_fn, duration=duration, seed=seed)

    merged_requests: List[MergedRequest] = []
    for req in requests:
        merged_requests.append(
            MergedRequest(
                request_id=-1,
                timestamp=req.timestamp,
                pool_id=pool_id,
                data=req.data,
            )
        )
    return merged_requests


def merge_workloads(workloads: List[List[MergedRequest]]) -> List[MergedRequest]:
    """Merge multiple workloads and reassign global request ids."""
    merged = [req for workload in workloads for req in workload]
    merged.sort(key=lambda req: (req.timestamp, req.pool_id))
    for request_id, req in enumerate(merged):
        req.request_id = request_id
    return merged


def save_merged_requests_to_csv(requests: List[MergedRequest], filename: str) -> None:
    """Save merged requests to CSV, including the source pool id."""
    if not requests:
        raise ValueError("No requests to save.")

    fieldnames = ["request_id", "timestamp", "pool_id"]
    data_fields = sorted({key for req in requests for key in req.data.keys()})
    fieldnames.extend(data_fields)

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for req in requests:
            row = {
                "request_id": req.request_id,
                "timestamp": req.timestamp,
                "pool_id": req.pool_id,
            }
            for field in data_fields:
                value = to_python_value(req.data.get(field))
                if isinstance(value, list):
                    row[field] = json.dumps(value)
                else:
                    row[field] = value
            writer.writerow(row)


def to_python_value(value: object) -> object:
    """Convert numpy scalars and containers into JSON-serializable Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [to_python_value(item) for item in value]
    if isinstance(value, tuple):
        return [to_python_value(item) for item in value]
    return value


def compute_access_intervals(
    requests: List[MergedRequest],
) -> Dict[int, List[int]]:
    """Compute access intervals per pool.

    For each request from pool P, the access interval is the number of distinct
    other pools requested since the previous request from P.
    """
    intervals_by_pool: Dict[int, List[int]] = {}
    last_seen_index: Dict[int, int] = {}

    for index, req in enumerate(requests):
        pool_id = req.pool_id
        intervals_by_pool.setdefault(pool_id, [])

        if pool_id in last_seen_index:
            prior_index = last_seen_index[pool_id]
            distinct_other_pools = {
                requests[i].pool_id
                for i in range(prior_index + 1, index)
                if requests[i].pool_id != pool_id
            }
            intervals_by_pool[pool_id].append(len(distinct_other_pools))

        last_seen_index[pool_id] = index

    return intervals_by_pool


def count_access_intervals(values: List[int]) -> Dict[int, int]:
    """Count how many times each access interval appears."""
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def counts_to_cdf(counts: Dict[int, int]) -> Tuple[List[int], List[float]]:
    """Convert interval counts to a discrete empirical CDF."""
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


def print_locality_stats(intervals_by_pool: Dict[int, List[int]]) -> None:
    """Print interval counts and CDF values for each pool."""
    print("\nModel locality statistics:")
    for pool_id in sorted(intervals_by_pool.keys()):
        intervals = intervals_by_pool[pool_id]
        interval_counts = count_access_intervals(intervals)
        values, cdf = counts_to_cdf(interval_counts)
        print(f"\nPool {pool_id}:")
        print(f"  Samples: {len(intervals)}")
        print(f"  Access Interval Counts: {interval_counts}")
        print(f"  CDF: {cdf}")


def plot_locality_cdfs(
    intervals_by_pool: Dict[int, List[int]],
    selected_models: List[ModelSpec],
    filename: str,
) -> None:
    """Plot all model locality CDFs on the same figure and save it locally."""
    plt.figure(figsize=(10, 6))
    model_names = {spec.pool_id: spec.model_name for spec in selected_models}

    for pool_id in sorted(intervals_by_pool.keys()):
        interval_counts = count_access_intervals(intervals_by_pool[pool_id])
        values, cdf = counts_to_cdf(interval_counts)
        if not values:
            continue
        label = model_names.get(pool_id, f"Pool {pool_id}")
        plt.plot(
            values,
            cdf,
            marker="o",
            markersize=4,
            linewidth=1.8,
            alpha=0.9,
            label=label,
        )

    plt.xlabel("Access Interval")
    plt.ylabel("CDF")
    plt.title("Model Access Locality CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main() -> None:
    model_num = 8
    shared_view_start = 39600
    duration = 3600

    if model_num <= 0 or model_num > len(ALL_MODELS):
        raise ValueError(
            f"model_num must be between 1 and {len(ALL_MODELS)}, got {model_num}"
        )

    selected_models = ALL_MODELS[:model_num]

    print("Generating per-model workloads...")
    workloads: List[List[MergedRequest]] = []
    print(f"Using {len(selected_models)} logical models out of {len(ALL_MODELS)} total")
    for spec in selected_models:
        view_start = shared_view_start + spec.time_offset
        view_end = view_start + duration
        view = get_or_create_pool(spec.category, spec.source_model).span(view_start, view_end)
        effective_rate_value = resolve_rate_value(view, spec.rate_strategy, spec.rate_value)
        requests = generate_pool_workload(
            pool_id=spec.pool_id,
            category=spec.category,
            model=spec.source_model,
            view_start=view_start,
            view_end=view_end,
            rate_strategy=spec.rate_strategy,
            rate_value=spec.rate_value,
            seed=spec.seed,
        )
        workloads.append(requests)
        print(
            f"  Pool {spec.pool_id} ({spec.model_name} -> {spec.source_model}): "
            f"{len(requests)} requests, "
            f"view={view_start}-{view_end}, "
            f"strategy={spec.rate_strategy}({effective_rate_value:.2f})"
        )

    merged_requests = merge_workloads(workloads)
    output_file = "multi_model_workload.csv"
    save_merged_requests_to_csv(merged_requests, output_file)

    print(f"\nMerged workload size: {len(merged_requests)}")
    print(f"Saved merged workload to {output_file}")

    intervals_by_pool = compute_access_intervals(merged_requests)
    print_locality_stats(intervals_by_pool)
    figure_file = "multi_model_locality_cdf.png"
    plot_locality_cdfs(intervals_by_pool, selected_models, figure_file)
    print(f"Saved locality plot to {figure_file}")


if __name__ == "__main__":
    main()
