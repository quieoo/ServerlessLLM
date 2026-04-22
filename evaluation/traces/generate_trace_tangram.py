#!/usr/bin/env python3
"""Generate a Tangram-style multi-model trace from ServeGen client pools.

Output format:
    timestamp model_id input_tokens output_tokens
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


THIS_FILE = Path(__file__).resolve()
SERVEGEN_ROOT = THIS_FILE.parents[1]
WORKSPACE_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(SERVEGEN_ROOT))

from servegen import Category, Client, ClientPool, generate_workload  # noqa: E402
from servegen.clientpool import ClientPoolView  # noqa: E402
from servegen.construct import Request  # noqa: E402


DEFAULT_CONFIG = WORKSPACE_ROOT / "ServerlessLLM" / "configs" / "servegen_8_models.json"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "ServerlessLLM" / "traces" / "servegen_tangram.trace"
DEFAULT_DATA_DIR = SERVEGEN_ROOT / "data"


@dataclass(frozen=True)
class ModelConfig:
    model_id: int
    pattern: str


@dataclass
class ModelWorkload:
    model_id: int
    pattern: str
    target_share: float
    share: float
    pool: ClientPool
    rate_fn: Dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an experimental trace with columns: "
            "timestamp model_id input_tokens output_tokens."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument(
        "--start-time",
        type=int,
        default=0,
        help="Start offset in the ServeGen trace, in seconds.",
    )
    parser.add_argument(
        "--target-rps",
        type=float,
        default=None,
        help=(
            "Final aggregate average requests per second across all models. "
            "Defaults to None, which keeps the original ServeGen rates."
        ),
    )
    parser.add_argument(
        "--target-rps-list",
        type=str,
        default=None,
        help=(
            "Comma-separated target RPS values for batch trace generation, e.g. "
            "'0.4,0.8,1.6'. Workloads are built once, then scaled for each value. "
            "Output files are named by appending _<rps> before the suffix unless "
            "--output contains the '{rps}' placeholder."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skew-alpha",
        type=float,
        default=0.0,
        help=(
            "Power-law skew for splitting clients among models with the same pattern. "
            "0.0 preserves the current equal-share split; larger values assign more "
            "load to lower model IDs."
        ),
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Write a header line before the trace rows.",
    )
    parser.add_argument(
        "--delimiter",
        default=" ",
        help="Column delimiter. Defaults to one space.",
    )
    return parser.parse_args()


def load_model_configs(config_path: Path) -> List[ModelConfig]:
    with config_path.open() as f:
        raw = json.load(f)

    models = raw.get("model_lists")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{config_path} must contain a non-empty 'model_lists' array")

    result: List[ModelConfig] = []
    for item in models:
        if "id" not in item or "pattern" not in item:
            raise ValueError(f"Model entry is missing 'id' or 'pattern': {item}")
        result.append(ModelConfig(model_id=int(item["id"]), pattern=str(item["pattern"])))
    return result


def parse_pattern(pattern: str) -> Tuple[Category, str]:
    try:
        category_name, model_name = pattern.split("/", 1)
    except ValueError as exc:
        raise ValueError(f"Pattern must have form '<category>/<model>': {pattern}") from exc

    for category in Category:
        if category.value == category_name:
            return category, model_name
    raise ValueError(f"Unknown ServeGen category in pattern '{pattern}'")


def weighted_pdf_stats(pdf: Optional[Sequence[float]]) -> Tuple[float, float]:
    if not pdf:
        return 0.0, 0.0
    total = float(sum(pdf))
    if total <= 0:
        return 0.0, 0.0

    avg = sum(i * p for i, p in enumerate(pdf)) / total
    cdf = np.cumsum(np.asarray(pdf, dtype=float) / total)
    p95 = float(np.searchsorted(cdf, 0.95))
    return avg, p95


def power_law_target_shares(n_parts: int, alpha: float) -> np.ndarray:
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    if alpha < 0:
        raise ValueError("--skew-alpha must be non-negative")

    ranks = np.arange(1, n_parts + 1, dtype=float)
    weights = np.power(ranks, -alpha)
    return weights / weights.sum()


def client_feature(client: Client, start_time: int, duration: int) -> np.ndarray:
    end_time = start_time + duration
    trace_times = sorted(client.trace.keys())

    volume = 0.0
    cv_mass = 0.0
    input_avg_mass = 0.0
    input_p95_mass = 0.0
    output_avg_mass = 0.0
    output_p95_mass = 0.0

    dataset_times = sorted(client.dataset.keys())

    for idx, ts in enumerate(trace_times):
        next_ts = trace_times[idx + 1] if idx + 1 < len(trace_times) else float("inf")
        window_start = max(ts, start_time)
        window_end = min(next_ts, end_time)
        if window_end <= window_start:
            continue

        trace_data = client.trace[ts]
        if not trace_data or trace_data["rate"] <= 0:
            continue

        weight = float(trace_data["rate"]) * (window_end - window_start)
        volume += weight
        cv_mass += weight * float(trace_data["cv"])

        dataset_ts = None
        for candidate in reversed(dataset_times):
            if candidate <= ts:
                dataset_ts = candidate
                break
        if dataset_ts is None:
            continue

        dataset = client.dataset[dataset_ts]
        input_pdf = dataset.get("input_tokens") or dataset.get("text_tokens")
        output_pdf = dataset.get("output_tokens")
        input_avg, input_p95 = weighted_pdf_stats(input_pdf)
        output_avg, output_p95 = weighted_pdf_stats(output_pdf)
        input_avg_mass += weight * input_avg
        input_p95_mass += weight * input_p95
        output_avg_mass += weight * output_avg
        output_p95_mass += weight * output_p95

    return np.asarray(
        [
            volume,
            cv_mass,
            input_avg_mass,
            input_p95_mass,
            output_avg_mass,
            output_p95_mass,
        ],
        dtype=float,
    )


def split_clients_shape_preserving(
    clients: Sequence[Client],
    n_parts: int,
    start_time: int,
    duration: int,
    target_shares: Optional[Sequence[float]] = None,
) -> List[List[Client]]:
    """Greedily split whole clients while balancing rate, burstiness, and lengths."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    if n_parts == 1:
        return [list(clients)]
    if target_shares is None:
        target_shares_array = np.full(n_parts, 1.0 / n_parts, dtype=float)
    else:
        if len(target_shares) != n_parts:
            raise ValueError("target_shares length must match n_parts")
        target_shares_array = np.asarray(target_shares, dtype=float)
        if np.any(target_shares_array < 0):
            raise ValueError("target_shares must be non-negative")
        total_share = float(target_shares_array.sum())
        if total_share <= 0:
            raise ValueError("target_shares must sum to a positive value")
        target_shares_array = target_shares_array / total_share

    client_features = [(client, client_feature(client, start_time, duration)) for client in clients]
    client_features.sort(key=lambda item: (item[1][0], item[0].client_id), reverse=True)

    total = np.sum([feature for _, feature in client_features], axis=0)
    scale = np.where(total > 0, total, 1.0)
    targets = [share * total for share in target_shares_array]
    feature_weights = np.asarray([4.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

    buckets: List[List[Client]] = [[] for _ in range(n_parts)]
    loads = [np.zeros_like(total) for _ in range(n_parts)]

    for client, feature in client_features:
        best_idx = 0
        best_score = float("inf")
        for idx in range(n_parts):
            score = 0.0
            for load_idx in range(n_parts):
                candidate = loads[load_idx] + feature if load_idx == idx else loads[load_idx]
                normalized_error = np.abs(candidate - targets[load_idx]) / scale
                score += float(np.sum(normalized_error * feature_weights))
            if not buckets[idx]:
                score -= 0.25
            score -= 1e-6 * float(target_shares_array[idx])
            if score < best_score:
                best_idx = idx
                best_score = score
        buckets[best_idx].append(client)
        loads[best_idx] += feature

    return buckets


def make_client_pool_like(source: ClientPool, clients: Sequence[Client], suffix: str) -> ClientPool:
    if not clients:
        raise ValueError(f"Cannot build an empty client pool for {source.category.value}/{source.model}")
    return ClientPool.from_clients(source.category, f"{source.model}-{suffix}", list(clients))


def group_by_pattern(models: Iterable[ModelConfig]) -> Dict[str, List[ModelConfig]]:
    grouped: Dict[str, List[ModelConfig]] = {}
    for model in models:
        grouped.setdefault(model.pattern, []).append(model)
    return grouped


def sum_rate_fn(pool_or_view: Union[ClientPool, ClientPoolView], duration: int) -> Dict[int, float]:
    windows = pool_or_view.span(0, duration).get()
    rate_fn: Dict[int, float] = {}
    for window in windows:
        if window.rate is None:
            continue
        rate_fn[window.timestamp] = rate_fn.get(window.timestamp, 0.0) + float(window.rate)
    return dict(sorted(rate_fn.items()))


def average_rate(rate_fn: Mapping[int, float], duration: int) -> float:
    if not rate_fn:
        return 0.0
    total = 0.0
    timestamps = sorted(ts for ts in rate_fn if 0 <= ts < duration)
    for idx, ts in enumerate(timestamps):
        next_ts = timestamps[idx + 1] if idx + 1 < len(timestamps) else duration
        if next_ts > ts:
            total += float(rate_fn[ts]) * (next_ts - ts)
    return total / duration


def parse_target_rps_list(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None

    values: List[float] = []
    for item in raw.replace(",", " ").split():
        target_rps = float(item)
        if target_rps <= 0:
            raise ValueError("--target-rps-list values must be positive")
        values.append(target_rps)

    if not values:
        raise ValueError("--target-rps-list must contain at least one value")
    return values


def format_rps_for_path(target_rps: float) -> str:
    text = f"{target_rps:g}"
    return text.replace(".", "p")


def output_path_for_target_rps(output: Path, target_rps: float) -> Path:
    rps_text = format_rps_for_path(target_rps)
    output_text = str(output)
    if "{rps}" in output_text:
        return Path(output_text.format(rps=rps_text))

    return output.with_name(f"{output.stem}_{rps_text}{output.suffix}")


def build_model_workloads(
    models: Sequence[ModelConfig],
    data_dir: Path,
    start_time: int,
    duration: int,
    skew_alpha: float,
) -> List[ModelWorkload]:
    workloads: List[ModelWorkload] = []

    for pattern, pattern_models in group_by_pattern(models).items():
        category, model_name = parse_pattern(pattern)
        source_pool = ClientPool(category, model_name, base_dir=str(data_dir))
        source_view = source_pool.span(start_time, start_time + duration)
        source_rate_fn = sum_rate_fn(source_view, duration)

        ordered_models = sorted(pattern_models, key=lambda item: item.model_id)
        target_shares = power_law_target_shares(len(ordered_models), skew_alpha)
        buckets = split_clients_shape_preserving(
            list(source_pool.clients.values()),
            len(ordered_models),
            start_time,
            duration,
            target_shares=target_shares,
        )
        bucket_volumes = [
            sum(client_feature(client, start_time, duration)[0] for client in bucket)
            for bucket in buckets
        ]
        total_volume = sum(bucket_volumes)

        for model_cfg, target_share, bucket, bucket_volume in zip(
            ordered_models, target_shares, buckets, bucket_volumes
        ):
            if not bucket:
                continue
            share = bucket_volume / total_volume if total_volume > 0 else len(bucket) / len(source_pool.clients)
            sub_pool = make_client_pool_like(source_pool, bucket, f"model-{model_cfg.model_id}")
            sub_view = sub_pool.span(start_time, start_time + duration)
            sub_timestamps = set(sum_rate_fn(sub_view, duration).keys())
            rate_fn = {
                ts: rate * float(target_share)
                for ts, rate in source_rate_fn.items()
                if ts in sub_timestamps
            }
            workloads.append(
                ModelWorkload(
                    model_id=model_cfg.model_id,
                    pattern=pattern,
                    target_share=float(target_share),
                    share=share,
                    pool=sub_view,
                    rate_fn=rate_fn,
                )
            )

    return workloads


def scale_workloads_to_target_rps(workloads: Sequence[ModelWorkload], duration: int, target_rps: float) -> float:
    base_rps = sum(average_rate(workload.rate_fn, duration) for workload in workloads)
    if base_rps <= 0:
        raise ValueError("Base workload has zero average RPS; cannot scale")

    scale = target_rps / base_rps
    for workload in workloads:
        workload.rate_fn = {ts: rate * scale for ts, rate in workload.rate_fn.items()}
    return scale


def scaled_workloads_to_target_rps(
    workloads: Sequence[ModelWorkload],
    duration: int,
    target_rps: float,
) -> Tuple[List[ModelWorkload], float]:
    base_rps = sum(average_rate(workload.rate_fn, duration) for workload in workloads)
    if base_rps <= 0:
        raise ValueError("Base workload has zero average RPS; cannot scale")

    scale = target_rps / base_rps
    scaled = [
        ModelWorkload(
            model_id=workload.model_id,
            pattern=workload.pattern,
            target_share=workload.target_share,
            share=workload.share,
            pool=workload.pool,
            rate_fn={ts: rate * scale for ts, rate in workload.rate_fn.items()},
        )
        for workload in workloads
    ]
    return scaled, scale


def input_tokens(data: Mapping[str, object]) -> int:
    if "input_tokens" in data:
        return int(data["input_tokens"])

    total = int(data.get("text_tokens", 0) or 0)
    for key in ("image_tokens", "audio_tokens", "video_tokens"):
        value = data.get(key)
        if isinstance(value, list):
            total += sum(int(item) for item in value)
        elif value is not None:
            total += int(value)
    return total


def output_tokens(data: Mapping[str, object]) -> int:
    return int(data.get("output_tokens", 0) or 0)


def generate_requests(workloads: Sequence[ModelWorkload], duration: int, seed: int) -> List[Tuple[float, int, int, int]]:
    rows: List[Tuple[float, int, int, int]] = []
    for idx, workload in enumerate(workloads):
        requests: List[Request] = generate_workload(
            workload.pool,
            workload.rate_fn,
            duration=duration,
            seed=seed + idx,
        )
        for request in requests:
            rows.append(
                (
                    request.timestamp,
                    workload.model_id,
                    input_tokens(request.data),
                    output_tokens(request.data),
                )
            )

    rows.sort(key=lambda row: (row[0], row[1]))
    return rows


def write_trace(
    rows: Sequence[Tuple[float, int, int, int]],
    output_path: Path,
    delimiter: str,
    header: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        if header:
            f.write(delimiter.join(["timestamp", "model_id", "input_tokens", "output_tokens"]) + "\n")
        for timestamp, model_id, input_len, output_len in rows:
            f.write(
                delimiter.join(
                    [
                        f"{timestamp:.6f}",
                        str(model_id),
                        str(input_len),
                        str(output_len),
                    ]
                )
                + "\n"
            )


def print_summary(
    rows: Sequence[Tuple[float, int, int, int]],
    workloads: Sequence[ModelWorkload],
    duration: int,
    scale: Optional[float],
    output_path: Path,
) -> None:
    actual_rps = len(rows) / duration if duration > 0 else 0.0
    print(f"Wrote {len(rows)} requests to {output_path}")
    print(f"Actual average RPS: {actual_rps:.4f}")
    if scale is None:
        print("Global rate scale: none")
    else:
        print(f"Global rate scale: {scale:.6f}")
    print("Per-model expected average RPS:")
    for workload in sorted(workloads, key=lambda item: item.model_id):
        print(
            f"  model_id={workload.model_id} pattern={workload.pattern} "
            f"target_share={workload.target_share:.4f} share={workload.share:.4f} "
            f"expected_rps={average_rate(workload.rate_fn, duration):.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.target_rps is not None and args.target_rps <= 0:
        raise ValueError("--target-rps must be positive")
    target_rps_list = parse_target_rps_list(args.target_rps_list)
    if args.target_rps is not None and target_rps_list is not None:
        raise ValueError("--target-rps and --target-rps-list cannot be used together")
    if args.skew_alpha < 0:
        raise ValueError("--skew-alpha must be non-negative")

    models = load_model_configs(args.config)
    workloads = build_model_workloads(models, args.data_dir, args.start_time, args.duration, args.skew_alpha)
    if target_rps_list is not None:
        for target_rps in target_rps_list:
            scaled_workloads, scale = scaled_workloads_to_target_rps(
                workloads,
                args.duration,
                target_rps,
            )
            rows = generate_requests(scaled_workloads, args.duration, args.seed)
            output_path = output_path_for_target_rps(args.output, target_rps)
            write_trace(rows, output_path, args.delimiter, args.header)
            print_summary(rows, scaled_workloads, args.duration, scale, output_path)
        return

    scale = None
    if args.target_rps is not None:
        scale = scale_workloads_to_target_rps(workloads, args.duration, args.target_rps)
    rows = generate_requests(workloads, args.duration, args.seed)
    write_trace(rows, args.output, args.delimiter, args.header)
    print_summary(rows, workloads, args.duration, scale, args.output)


if __name__ == "__main__":
    main()
