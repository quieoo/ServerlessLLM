#!/usr/bin/env python3
"""
Estimate and plot request counts over time for all shipped ServeGen workloads.

This script is intentionally different from generate_workload(): it does not
resample arrivals or request fields. The closest recoverable signal in the
released data is each client's per-window request rate, so we reconstruct the
per-window request volume as:

    estimated_requests = sum(client_rate_requests_per_second) * window_seconds

The released data is already aggregated into trace windows, so this is a
statistical reconstruction rather than an exact replay of raw production
requests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
OUTPUT_FIGURE = Path(__file__).with_name("reqrate_over_time.png")
OUTPUT_CSV = Path(__file__).with_name("reqrate_over_time.csv")
DEFAULT_WINDOW_SIZE_SECONDS = 600
DEFAULT_DURATION_SECONDS: Optional[int] = None
DEFAULT_TOP_SHARE_THRESHOLD_PERCENT = 90.0


@dataclass(frozen=True)
class WorkloadSpec:
    label: str
    relative_dir: Path


WORKLOADS: List[WorkloadSpec] = [
    WorkloadSpec("language/m-small", Path("language/m-small")),
    WorkloadSpec("language/m-mid", Path("language/m-mid")),
    WorkloadSpec("language/m-large", Path("language/m-large")),
    WorkloadSpec("multimodal/mm-image", Path("multimodal/mm-image")),
    WorkloadSpec("reason/deepseek-r1", Path("reason/deepseek-r1")),
]


def iter_trace_rows(workload_dir: Path) -> Iterable[Tuple[int, float]]:
    """Yield timestamp and request rate from all trace files in one workload."""
    trace_files = sorted(workload_dir.glob("chunk-*-trace.csv"))
    if not trace_files:
        raise ValueError(f"No trace files found under {workload_dir}")

    for trace_file in trace_files:
        with trace_file.open(newline="") as csvfile:
            for row in csv.reader(csvfile):
                if not row:
                    continue
                timestamp = int(row[0])
                rate = float(row[1])
                yield timestamp, rate


def infer_window_seconds(timestamps: Iterable[int]) -> int:
    """Infer the smallest native trace interval."""
    sorted_timestamps = sorted(set(timestamps))
    intervals = [
        right - left
        for left, right in zip(sorted_timestamps, sorted_timestamps[1:])
        if right > left
    ]
    if not intervals:
        raise ValueError("Need at least two timestamps to infer the window size")
    return min(intervals)


def reconstruct_native_request_counts(workload_dir: Path) -> Tuple[int, Dict[int, float]]:
    """Aggregate client rates and convert them to per-window request counts."""
    total_rate_by_timestamp: Dict[int, float] = {}
    timestamps: List[int] = []

    for timestamp, rate in iter_trace_rows(workload_dir):
        timestamps.append(timestamp)
        total_rate_by_timestamp[timestamp] = (
            total_rate_by_timestamp.get(timestamp, 0.0) + rate
        )

    window_seconds = infer_window_seconds(timestamps)
    request_count_by_timestamp = {
        timestamp: total_rate * window_seconds
        for timestamp, total_rate in sorted(total_rate_by_timestamp.items())
    }
    return window_seconds, request_count_by_timestamp


def rebin_request_counts(
    native_counts: Dict[int, float],
    native_window_seconds: int,
    target_window_seconds: int,
    duration_seconds: Optional[int],
) -> Dict[int, float]:
    """Project native windows into user-selected windows by interval overlap."""
    if target_window_seconds <= 0:
        raise ValueError(f"window size must be positive, got {target_window_seconds}s")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError(f"duration must be positive, got {duration_seconds}s")

    rebinned_counts: Dict[int, float] = {}
    for timestamp, request_count in native_counts.items():
        native_start = timestamp
        native_end = timestamp + native_window_seconds
        if duration_seconds is not None:
            native_end = min(native_end, duration_seconds)
        if native_end <= native_start:
            continue

        request_rate = request_count / native_window_seconds
        bucket_start = (native_start // target_window_seconds) * target_window_seconds
        while bucket_start < native_end:
            bucket_end = bucket_start + target_window_seconds
            overlap_start = max(native_start, bucket_start)
            overlap_end = min(native_end, bucket_end)
            if overlap_end > overlap_start:
                rebinned_counts[bucket_start] = (
                    rebinned_counts.get(bucket_start, 0.0)
                    + request_rate * (overlap_end - overlap_start)
                )
            bucket_start += target_window_seconds

    return dict(sorted(rebinned_counts.items()))


def save_counts_csv(
    counts_by_workload: Dict[str, Dict[int, float]],
    output_file: Path,
) -> None:
    """Save the reconstructed time series for inspection."""
    all_timestamps = sorted(
        {
            timestamp
            for counts in counts_by_workload.values()
            for timestamp in counts.keys()
        }
    )
    labels = list(counts_by_workload.keys())

    with output_file.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", *labels])
        for timestamp in all_timestamps:
            writer.writerow(
                [
                    timestamp,
                    *[
                        counts_by_workload[label].get(timestamp, "")
                        for label in labels
                    ],
                ]
            )


def plot_counts(
    counts_by_workload: Dict[str, Dict[int, float]],
    window_seconds: int,
    output_file: Path,
) -> None:
    """Plot all workloads in one figure."""
    plt.figure(figsize=(12, 6))

    for label, counts in counts_by_workload.items():
        timestamps = np.array(sorted(counts.keys()), dtype=float)
        request_counts = np.array([counts[int(ts)] for ts in timestamps], dtype=float)
        hours = timestamps / 3600.0
        plt.plot(hours, request_counts, linewidth=1.6, alpha=0.9, label=label)

    plt.xlabel("Time since trace start (hours)")
    plt.ylabel(f"Requests per {window_seconds}s window")
    plt.title("Reconstructed Request Load Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def compute_top_k_share_stats(
    counts_by_workload: Dict[str, Dict[int, float]],
    top_k: int,
    share_threshold_percent: float,
) -> Tuple[int, int, float]:
    """Count windows where the top-k workloads exceed a request-share threshold."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not 0 <= share_threshold_percent <= 100:
        raise ValueError(
            "share threshold must be in [0, 100], "
            f"got {share_threshold_percent}"
        )

    all_timestamps = sorted(
        {
            timestamp
            for counts in counts_by_workload.values()
            for timestamp in counts.keys()
        }
    )
    labels = list(counts_by_workload.keys())
    matched_windows = 0
    active_windows = 0

    for timestamp in all_timestamps:
        values = [
            counts_by_workload[label].get(timestamp, 0.0)
            for label in labels
        ]
        total_requests = sum(values)
        if total_requests <= 0:
            continue

        active_windows += 1
        top_k_requests = sum(sorted(values, reverse=True)[:top_k])
        top_k_share_percent = top_k_requests / total_requests * 100.0
        if top_k_share_percent >= share_threshold_percent:
            matched_windows += 1

    matched_percent = (
        matched_windows / active_windows * 100.0
        if active_windows
        else 0.0
    )
    return matched_windows, active_windows, matched_percent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct request counts from ServeGen's aggregated trace rates "
            "and plot all five shipped workloads."
        )
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE_SECONDS,
        help=(
            "Aggregation window in seconds. Values smaller than the native "
            "600s trace window are estimated by assuming a uniform rate within "
            "each native window. Default: 600."
        ),
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help=(
            "Total duration in seconds from trace start. Default: use the full "
            "available duration for each workload."
        ),
    )
    parser.add_argument(
        "--top-share-threshold",
        type=float,
        default=DEFAULT_TOP_SHARE_THRESHOLD_PERCENT,
        help=(
            "Y percent threshold for reporting how often the top-2 access "
            "patterns contribute at least Y percent of requests. Default: 80."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    native_counts_by_workload: Dict[str, Dict[int, float]] = {}
    counts_by_workload: Dict[str, Dict[int, float]] = {}
    inferred_windows: Dict[str, int] = {}

    for spec in WORKLOADS:
        workload_dir = DATA_ROOT / spec.relative_dir
        window_seconds, native_counts = reconstruct_native_request_counts(workload_dir)
        inferred_windows[spec.label] = window_seconds
        native_counts_by_workload[spec.label] = native_counts

    unique_windows = set(inferred_windows.values())
    if len(unique_windows) != 1:
        raise ValueError(f"Workloads use different window sizes: {inferred_windows}")
    native_window_seconds = unique_windows.pop()

    for spec in WORKLOADS:
        counts_by_workload[spec.label] = rebin_request_counts(
            native_counts_by_workload[spec.label],
            native_window_seconds,
            args.window_size,
            args.duration,
        )

    save_counts_csv(counts_by_workload, OUTPUT_CSV)
    plot_counts(counts_by_workload, args.window_size, OUTPUT_FIGURE)

    print("Reconstructed request counts from original trace rates.")
    print(
        "This is the closest recovery possible from the released aggregated data; "
        "it is not an exact raw-request replay."
    )
    print(f"Native time window: {native_window_seconds}s")
    print(f"Output window size: {args.window_size}s")
    print(
        "Duration: "
        + (f"{args.duration}s" if args.duration is not None else "full available range")
    )
    for label, counts in counts_by_workload.items():
        timestamps = sorted(counts.keys())
        if not timestamps:
            print(f"{label}: 0 windows")
            continue
        print(
            f"{label}: {len(timestamps)} windows, "
            f"time={timestamps[0]}-{timestamps[-1]}s, "
            f"total_requests≈{sum(counts.values()):.0f}"
        )

    matched_windows, active_windows, matched_percent = compute_top_k_share_stats(
        counts_by_workload,
        top_k=2,
        share_threshold_percent=args.top_share_threshold,
    )
    print(
        f"Top-2 access patterns contribute >= {args.top_share_threshold:.2f}% "
        f"of requests in {matched_windows}/{active_windows} active windows "
        f"({matched_percent:.2f}%)."
    )
    print(f"Saved plot to {OUTPUT_FIGURE}")
    print(f"Saved reconstructed counts to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
