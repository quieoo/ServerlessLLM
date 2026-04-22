#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

DEFAULT_CONFIG = REPO_ROOT / "configs" / "servegen_8_models.json"
DEFAULT_TRACE = REPO_ROOT / "evaluation" / "traces" / "servegen_tangram.trace"

NUM_GPU = 2
GPU_POOL_SIZE = 40 * 1024**3

# Fallback hardware limits used only when auto measurement is unavailable.
FALLBACK_H2D_BW_GBPS = 22.0
FALLBACK_REMOTE_BW_GBPS = 8.0
FALLBACK_DEVICE_COPY_BW_GBPS = 600.0

DEFAULT_BENCHMARK_BYTES_MB = 256
DEFAULT_REMOTE_BENCHMARK_BYTES_MB = 1024

DTYPE_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4,
    "fp16": 2, "bf16": 2, "fp32": 4,
}

KV_FALLBACKS = {
    "llava": {"num_hidden_layers": 32, "hidden_size": 4096, "torch_dtype": "float16"},
}


@dataclass(frozen=True)
class ModelConfig:
    model_id: int
    path: Path
    pattern: str
    model_size: int
    kv_token_size: int


@dataclass(frozen=True)
class TraceRequest:
    timestamp: float
    model_id: int
    input_tokens: int
    output_tokens: int

    @property
    def token_num(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelTraceStats:
    model_id: int
    count: int
    input_p50: float
    input_p90: float
    output_p50: float
    output_p90: float
    total_p50: float
    total_p90: float
    service_time_ms: float


@dataclass
class GPUState:
    resident_model: Optional[int] = None
    used_bytes: int = 0

    prefetched_model: Optional[int] = None
    prefetched_bytes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Aegaeon-style model loading latency on ServeGen/Tangram traces."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--gpu-num", type=int, default=int(os.getenv("GPU_NUM", NUM_GPU)))
    parser.add_argument("--gpu-pool-gb", type=float, default=float(os.getenv("USABLE_MEMORY", "40").split()[0]))
    parser.add_argument("--max-requests", type=int, default=int(os.getenv("MAX_REQUESTS", "1000")))
    parser.add_argument("--seed", type=int, default=0, help="Accepted for docs/5-Aegaeon.sh compatibility.")
    parser.add_argument(
        "--h2d-bw-gbps",
        type=float,
        default=None,
        help="Raw host-to-device bandwidth in GiB/s. Defaults to an on-startup CUDA copy benchmark.",
    )
    parser.add_argument(
        "--remote-bw-gbps",
        type=float,
        default=None,
        help="Remote/disk-to-host bandwidth in GiB/s. Defaults to an on-startup model-file read benchmark.",
    )
    parser.add_argument(
        "--device-copy-bw-gbps",
        type=float,
        default=None,
        help="GPU-side device-copy bandwidth in GiB/s. Defaults to an on-startup CUDA copy benchmark.",
    )
    parser.add_argument("--benchmark-bytes-mb", type=int, default=DEFAULT_BENCHMARK_BYTES_MB)
    parser.add_argument("--remote-benchmark-bytes-mb", type=int, default=DEFAULT_REMOTE_BENCHMARK_BYTES_MB)
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--loader-threads", type=int, default=4)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument(
        "--prefetch-prob",
        type=float,
        default=1.0,
        help="Probability of attempting prefetch after each request when --prefetch is enabled.",
    )
    parser.add_argument(
        "--service-time-ms",
        type=float,
        default=None,
        help="Optional fixed service time override. By default service time is estimated per model from the trace.",
    )
    parser.add_argument(
        "--service-context-percentile",
        type=float,
        default=50.0,
        help="Token-length percentile used for per-model service-time estimation.",
    )
    parser.add_argument(
        "--service-base-ms",
        type=float,
        default=20.0,
        help="Base per-request service time used by the trace-derived estimator.",
    )
    parser.add_argument(
        "--prefill-ms-per-input-token",
        type=float,
        default=0.02,
        help="Estimated prefill service time per input token.",
    )
    parser.add_argument(
        "--decode-ms-per-output-token",
        type=float,
        default=20,
        help="Estimated decode service time per output token.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def model_tensor_size(model_path: Path) -> int:
    tensor_files = sorted(model_path.glob("tensor.data*"))
    if not tensor_files:
        raise ValueError(f"No tensor.data* files found under model path: {model_path}")
    return sum(item.stat().st_size for item in tensor_files)


def hf_config_path(model_path: Path) -> Path:
    for candidate in [model_path / "config.json", model_path.parent / "config.json"]:
        if candidate.is_file():
            return candidate
    raise ValueError(f"No config.json found for model path: {model_path}")


def dtype_size(config: Dict) -> int:
    return DTYPE_BYTES.get(str(config.get("torch_dtype", "float16")).lower(), 2)


def kv_source_config(config: Dict) -> Dict:
    if all(k in config for k in ("num_hidden_layers", "hidden_size")):
        return config

    model_type = str(config.get("model_type", "")).lower()
    if model_type in KV_FALLBACKS:
        fallback = dict(KV_FALLBACKS[model_type])
        fallback.setdefault("torch_dtype", config.get("torch_dtype", "float16"))
        return fallback

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        merged = dict(text_config)
        merged.setdefault("torch_dtype", config.get("torch_dtype", "float16"))
        if all(k in merged for k in ("num_hidden_layers", "hidden_size")):
            return merged

    raise ValueError("Cannot derive KV size without num_hidden_layers and hidden_size")


def kv_token_size(model_path: Path) -> int:
    config = read_json(hf_config_path(model_path))
    source = kv_source_config(config)
    return 2 * int(source["num_hidden_layers"]) * int(source["hidden_size"]) * dtype_size(source)


def load_model_configs(config_path: Path) -> List[ModelConfig]:
    raw = read_json(config_path)
    models = raw.get("model_lists")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{config_path} must contain a non-empty 'model_lists' array")

    result = []
    for item in models:
        model_path = Path(item["path"])
        result.append(ModelConfig(
            model_id=int(item["id"]),
            path=model_path,
            pattern=str(item.get("pattern", "unknown")),
            model_size=model_tensor_size(model_path),
            kv_token_size=kv_token_size(model_path),
        ))
    return sorted(result, key=lambda m: m.model_id)


def iter_trace_requests(trace_path: Path) -> Iterable[TraceRequest]:
    with trace_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("timestamp"):
                continue

            parts = line.split()

            if len(parts) == 1:
                yield TraceRequest(
                    timestamp=float(line_no),
                    model_id=int(parts[0]),
                    input_tokens=0,
                    output_tokens=0,
                )
                continue

            if len(parts) < 4:
                raise ValueError(f"Invalid trace row at line {line_no}: {line}")

            timestamp = float(parts[0])
            model_id = int(parts[1])
            input_tokens = int(parts[2])
            output_tokens = int(parts[3])
            yield TraceRequest(
                timestamp=timestamp,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )


def percentile(values: List[int], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))

    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def estimate_model_trace_stats(
    requests: List[TraceRequest],
    model_configs: List[ModelConfig],
    service_context_percentile: float,
    service_base_ms: float,
    prefill_ms_per_input_token: float,
    decode_ms_per_output_token: float,
    fixed_service_time_ms: Optional[float],
) -> Dict[int, ModelTraceStats]:
    input_tokens_by_model: Dict[int, List[int]] = {model.model_id: [] for model in model_configs}
    output_tokens_by_model: Dict[int, List[int]] = {model.model_id: [] for model in model_configs}

    for req in requests:
        input_tokens_by_model.setdefault(req.model_id, []).append(req.input_tokens)
        output_tokens_by_model.setdefault(req.model_id, []).append(req.output_tokens)

    stats: Dict[int, ModelTraceStats] = {}
    for model in model_configs:
        model_id = model.model_id
        input_tokens = input_tokens_by_model.get(model_id, [])
        output_tokens = output_tokens_by_model.get(model_id, [])
        total_tokens = [inp + out for inp, out in zip(input_tokens, output_tokens)]

        service_input = percentile(input_tokens, service_context_percentile)
        service_output = percentile(output_tokens, service_context_percentile)
        if fixed_service_time_ms is None:
            service_time_ms = (
                service_base_ms
                + service_input * prefill_ms_per_input_token
                + service_output * decode_ms_per_output_token
            )
        else:
            service_time_ms = fixed_service_time_ms

        stats[model_id] = ModelTraceStats(
            model_id=model_id,
            count=len(input_tokens),
            input_p50=percentile(input_tokens, 50),
            input_p90=percentile(input_tokens, 90),
            output_p50=percentile(output_tokens, 50),
            output_p90=percentile(output_tokens, 90),
            total_p50=percentile(total_tokens, 50),
            total_p90=percentile(total_tokens, 90),
            service_time_ms=service_time_ms,
        )

    return stats


def effective_h2d_bandwidth(raw_bw: float, chunk_mb: int, loader_threads: int) -> float:
    """
    Approximate Aegaeon-style chunked, pinned, multi-threaded H2D loading.
    Tune this with measured loading time if possible.
    """
    pinned_factor = 0.90
    thread_factor = min(1.0, 0.55 + 0.12 * loader_threads)
    chunk_factor = min(1.0, 0.70 + 0.004 * chunk_mb)
    return raw_bw * pinned_factor * thread_factor * chunk_factor


def benchmark_torch_bandwidth(
    kind: str,
    benchmark_bytes: int,
    warmup_iters: int = 3,
    measure_iters: int = 10,
) -> float:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not available") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if benchmark_bytes <= 0:
        raise ValueError("benchmark size must be positive")

    numel = max(1, benchmark_bytes // 4)
    src_cpu = None
    src_gpu = None
    dst_gpu = torch.empty(numel, dtype=torch.float32, device="cuda")

    if kind == "h2d":
        try:
            src_cpu = torch.empty(numel, dtype=torch.float32, pin_memory=True)
        except RuntimeError:
            src_cpu = torch.empty(numel, dtype=torch.float32)
        src_cpu.fill_(1.0)

        def copy_once() -> None:
            dst_gpu.copy_(src_cpu, non_blocking=True)

    elif kind == "device":
        src_gpu = torch.empty(numel, dtype=torch.float32, device="cuda")
        src_gpu.fill_(1.0)

        def copy_once() -> None:
            dst_gpu.copy_(src_gpu, non_blocking=True)

    else:
        raise ValueError(f"Unsupported CUDA bandwidth benchmark kind: {kind}")

    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        copy_once()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(measure_iters):
        copy_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        raise RuntimeError("benchmark elapsed time is zero")
    return benchmark_bytes * measure_iters / elapsed


def benchmark_remote_bandwidth(model_configs: List[ModelConfig], benchmark_bytes: int) -> float:
    if benchmark_bytes <= 0:
        raise ValueError("benchmark size must be positive")

    chunk_size = 8 * 1024 * 1024
    remaining = benchmark_bytes
    read_bytes = 0
    start = time.perf_counter()

    for model in model_configs:
        for tensor_file in sorted(model.path.glob("tensor.data*")):
            with tensor_file.open("rb", buffering=0) as f:
                while remaining > 0:
                    data = f.read(min(chunk_size, remaining))
                    if not data:
                        break
                    read_len = len(data)
                    remaining -= read_len
                    read_bytes += read_len
            if remaining <= 0:
                break
        if remaining <= 0:
            break

    elapsed = time.perf_counter() - start
    if read_bytes == 0:
        raise RuntimeError("no model bytes were read")
    if elapsed <= 0:
        raise RuntimeError("benchmark elapsed time is zero")
    return read_bytes / elapsed


def resolve_bandwidth(
    name: str,
    explicit_gbps: Optional[float],
    fallback_gbps: float,
    measure,
) -> tuple[float, str]:
    if explicit_gbps is not None:
        return explicit_gbps * 1024**3, "cli"

    try:
        return measure(), "measured"
    except Exception as exc:
        print(f"Warning: failed to measure {name} bandwidth ({exc}); using {fallback_gbps:g} GiB/s")
        return fallback_gbps * 1024**3, "fallback"


def choose_gpu(model: ModelConfig, gpu_states: List[GPUState]) -> int:
    # If Aegaeon has already prefetched this model on an instance, use that instance.
    for i, gpu in enumerate(gpu_states):
        if gpu.prefetched_model == model.model_id:
            return i

    # Otherwise, choose by resource availability/load, not by historical cached model residency.
    return min(range(len(gpu_states)), key=lambda i: gpu_states[i].used_bytes)


def estimate_load_time(
    model: ModelConfig,
    gpu: GPUState,
    host_cache: set[int],
    h2d_bw: float,
    remote_bw: float,
    device_copy_bw: float,
) -> tuple[float, str]:

    if gpu.prefetched_model == model.model_id:
        missing = max(0, model.model_size - gpu.prefetched_bytes)
        # Already-prefetched bytes only need cheap GPU-side placement/copy.
        copied = min(model.model_size, gpu.prefetched_bytes)
        return missing / h2d_bw + copied / device_copy_bw, "prefetch_hit"

    if model.model_id in host_cache:
        return model.model_size / h2d_bw, "host_cache_hit"

    remote = model.model_size / remote_bw
    h2d = model.model_size / h2d_bw
    overlap = min(remote, 0.6 * h2d)
    return remote + h2d - overlap, "host_cache_miss"


def advance_prefetch(
    current_gpu: GPUState,
    next_model: Optional[ModelConfig],
    service_time_s: float,
    h2d_bw: float,
    gpu_pool_size: int,
) -> None:
    if next_model is None:
        return

    available = max(0, gpu_pool_size - current_gpu.used_bytes)
    if available <= 0:
        return

    if current_gpu.prefetched_model not in (None, next_model.model_id):
        current_gpu.prefetched_model = None
        current_gpu.prefetched_bytes = 0

    current_gpu.prefetched_model = next_model.model_id
    current_gpu.prefetched_bytes = min(
        next_model.model_size,
        current_gpu.prefetched_bytes + int(service_time_s * h2d_bw),
        available,
    )


def gpu_memory_utilization(gpu_states: List[GPUState], gpu_pool_size: int) -> float:
    if not gpu_states or gpu_pool_size <= 0:
        return 0.0

    used = sum(
        min(gpu_pool_size, gpu.used_bytes + gpu.prefetched_bytes)
        for gpu in gpu_states
    )
    return min(1.0, used / (gpu_pool_size * len(gpu_states)))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not 0.0 <= args.prefetch_prob <= 1.0:
        raise ValueError(f"--prefetch-prob must be in [0, 1], got {args.prefetch_prob}")
    if not 0.0 <= args.service_context_percentile <= 100.0:
        raise ValueError(
            f"--service-context-percentile must be in [0, 100], got {args.service_context_percentile}"
        )

    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.trace.is_file():
        raise FileNotFoundError(f"Trace not found: {args.trace}")

    model_configs = load_model_configs(args.config)
    models_by_id = {m.model_id: m for m in model_configs}
    max_model_id = max(models_by_id)

    requests = list(iter_trace_requests(args.trace))
    if args.max_requests:
        requests = requests[:args.max_requests]
    model_trace_stats = estimate_model_trace_stats(
        requests=requests,
        model_configs=model_configs,
        service_context_percentile=args.service_context_percentile,
        service_base_ms=args.service_base_ms,
        prefill_ms_per_input_token=args.prefill_ms_per_input_token,
        decode_ms_per_output_token=args.decode_ms_per_output_token,
        fixed_service_time_ms=args.service_time_ms,
    )

    gpu_pool_size = int(args.gpu_pool_gb * 1024**3)
    benchmark_bytes = args.benchmark_bytes_mb * 1024**2
    remote_benchmark_bytes = args.remote_benchmark_bytes_mb * 1024**2
    raw_h2d_bw, h2d_bw_source = resolve_bandwidth(
        name="h2d",
        explicit_gbps=args.h2d_bw_gbps,
        fallback_gbps=FALLBACK_H2D_BW_GBPS,
        measure=lambda: benchmark_torch_bandwidth("h2d", benchmark_bytes),
    )
    remote_bw, remote_bw_source = resolve_bandwidth(
        name="remote",
        explicit_gbps=args.remote_bw_gbps,
        fallback_gbps=FALLBACK_REMOTE_BW_GBPS,
        measure=lambda: benchmark_remote_bandwidth(model_configs, remote_benchmark_bytes),
    )
    device_copy_bw, device_copy_bw_source = resolve_bandwidth(
        name="device-copy",
        explicit_gbps=args.device_copy_bw_gbps,
        fallback_gbps=FALLBACK_DEVICE_COPY_BW_GBPS,
        measure=lambda: benchmark_torch_bandwidth("device", benchmark_bytes),
    )
    h2d_bw = effective_h2d_bandwidth(raw_h2d_bw, args.chunk_mb, args.loader_threads)

    gpu_states = [GPUState() for _ in range(args.gpu_num)]
    host_cache: set[int] = set()

    total_load_time = [0.0] * (max_model_id + 1)
    total_load_cnt = [0] * (max_model_id + 1)
    memory_utilizations: List[float] = []
    hit_breakdown = {
        "gpu_hit": 0,
        "prefetch_hit": 0,
        "host_cache_hit": 0,
        "host_cache_miss": 0,
    }

    print(f"trace: {args.trace}")
    print(f"config: {args.config}")
    print(f"gpu_num: {args.gpu_num}")
    print(f"gpu_pool_gb: {args.gpu_pool_gb:g}")
    print(f"max_requests: {len(requests)}")
    print(f"raw_h2d_bw: {raw_h2d_bw / 1024**3:.2f} GiB/s ({h2d_bw_source})")
    print(f"effective_h2d_bw: {h2d_bw / 1024**3:.2f} GiB/s")
    print(f"remote_bw: {remote_bw / 1024**3:.2f} GiB/s ({remote_bw_source})")
    print(f"device_copy_bw: {device_copy_bw / 1024**3:.2f} GiB/s ({device_copy_bw_source})")
    print(f"prefetch: {args.prefetch}")
    print(f"prefetch_prob: {args.prefetch_prob:g}")
    print(f"service_context_percentile: {args.service_context_percentile:g}")
    if args.service_time_ms is not None:
        print(f"service_time_mode: fixed {args.service_time_ms:g} ms")
    else:
        print(
            "service_time_mode: trace-estimated "
            f"(base={args.service_base_ms:g}ms, "
            f"prefill={args.prefill_ms_per_input_token:g}ms/input-token, "
            f"decode={args.decode_ms_per_output_token:g}ms/output-token)"
        )

    for model in model_configs:
        stats = model_trace_stats[model.model_id]
        print(
            f"model {model.model_id}: size={model.model_size / 1024**3:.3f}GB, "
            f"kv_token_size={model.kv_token_size}B, pattern={model.pattern}, "
            f"requests={stats.count}, input_p50={stats.input_p50:.1f}, input_p90={stats.input_p90:.1f}, "
            f"output_p50={stats.output_p50:.1f}, output_p90={stats.output_p90:.1f}, "
            f"total_p50={stats.total_p50:.1f}, total_p90={stats.total_p90:.1f}, "
            f"service_time={stats.service_time_ms:.3f}ms"
        )

    for i, req in enumerate(requests):
        if req.model_id not in models_by_id:
            raise ValueError(f"Trace references absent model {req.model_id}")

        model = models_by_id[req.model_id]
        kv_size = model.kv_token_size * req.token_num

        gpu_id = choose_gpu(model, gpu_states)
        gpu = gpu_states[gpu_id]

        load_time, reason = estimate_load_time(
            model=model,
            gpu=gpu,
            host_cache=host_cache,
            h2d_bw=h2d_bw,
            remote_bw=remote_bw,
            device_copy_bw=device_copy_bw,
        )

        total_load_time[model.model_id] += load_time
        total_load_cnt[model.model_id] += 1
        hit_breakdown[reason] += 1

        # After loading, the model is available in host cache and resident on this GPU.
        host_cache.add(model.model_id)
        gpu.resident_model = model.model_id
        gpu.prefetched_model = None
        gpu.prefetched_bytes = 0

        # KV affects memory occupancy, not model loading latency.
        gpu.used_bytes = min(model.model_size + kv_size, gpu_pool_size)

        if args.prefetch and random.random() < args.prefetch_prob:
            next_model = None
            if i + 1 < len(requests):
                next_req = requests[i + 1]
                next_model = models_by_id.get(next_req.model_id)

            advance_prefetch(
                current_gpu=gpu,
                next_model=next_model,
                service_time_s=model_trace_stats[model.model_id].service_time_ms / 1000.0,
                h2d_bw=h2d_bw,
                gpu_pool_size=gpu_pool_size,
            )

        memory_utilization = gpu_memory_utilization(gpu_states, gpu_pool_size)
        memory_utilizations.append(memory_utilization)
        print(
            f"memory utilization after request {i + 1}: "
            f"{memory_utilization:.3f}"
        )

    print(f"\nprocessed requests: {len(requests)}")
    print("hit breakdown:")
    for k, v in hit_breakdown.items():
        print(f"  {k}: {v}")

    avg_memory_utilization = (
        sum(memory_utilizations) / len(memory_utilizations)
        if memory_utilizations
        else 0.0
    )
    peak_memory_utilization = max(memory_utilizations) if memory_utilizations else 0.0
    print(
        f"\nmemory utilization: avg={avg_memory_utilization:.3f}, "
        f"peak={peak_memory_utilization:.3f}"
    )

    total_load_time_all = sum(total_load_time)
    total_load_cnt_all = sum(total_load_cnt)
    total_avg_ms = total_load_time_all / total_load_cnt_all * 1000 if total_load_cnt_all else 0.0
    print(
        f"\noverall load time: total={total_load_time_all:.3f}s, "
        f"cnt={total_load_cnt_all}, avg={total_avg_ms:.3f}ms"
    )

    print("\nper-model load time:")
    for model in model_configs:
        cnt = total_load_cnt[model.model_id]
        avg_ms = total_load_time[model.model_id] / cnt * 1000 if cnt else 0.0
        print(
            f"model {model.model_id}: total={total_load_time[model.model_id]:.3f}s, "
            f"cnt={cnt}, avg={avg_ms:.3f}ms"
        )


if __name__ == "__main__":
    main()
