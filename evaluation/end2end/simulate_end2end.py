#!/usr/bin/env python3
"""Online trace-driven end-to-end simulator for Tangram/SLLM-CM."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class Request:
    req_id: int
    arrival_ms: float
    model_id: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class TensorGroup:
    fingerprint: str
    size: int


@dataclass(frozen=True)
class ModelInfo:
    model_id: int
    path: Path
    tensor_groups: Sequence[TensorGroup]
    sensitivity: float = 1.0

    @property
    def total_bytes(self) -> int:
        return sum(group.size for group in self.tensor_groups)


@dataclass(frozen=True)
class LoadEstimate:
    gpu_id: int
    cached_bytes: int
    to_load_bytes: int
    total_model_bytes: int
    full_model_hit: bool
    load_ms: float


@dataclass
class SimulatedRequest:
    req_id: int
    arrival_ms: float
    model_id: int
    gpu_id: int
    queue_ms: float
    load_ms: float
    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    e2e_ms: float
    base_slo_ms: float
    to_load_bytes: int
    cached_bytes: int
    total_model_bytes: int
    full_model_hit: bool


class OnlineVRAMBackend:
    """Small VRAMManager-shaped backend for online scheduling simulation."""

    def __init__(
        self,
        models: Sequence[ModelInfo],
        num_gpus: int,
        gpu_capacity_bytes: int,
        load_bandwidth_gbps: float,
        load_overhead_ms: float,
    ) -> None:
        self.models = {model.model_id: model for model in models}
        self.num_gpus = num_gpus
        self.gpu_capacity_bytes = gpu_capacity_bytes
        self.load_bandwidth_gbps = load_bandwidth_gbps
        self.load_overhead_ms = load_overhead_ms
        self.gpu_caches: List[OrderedDict[str, int]] = [
            OrderedDict() for _ in range(num_gpus)
        ]
        self.gpu_used_bytes = [0 for _ in range(num_gpus)]

    def _load_ms(self, to_load_bytes: int) -> float:
        if to_load_bytes <= 0:
            return 0.0
        if self.load_bandwidth_gbps <= 0.0:
            return 0.0
        return (
            to_load_bytes / (self.load_bandwidth_gbps * 1000.0 * 1000.0 * 1000.0)
        ) * 1000.0 + self.load_overhead_ms

    def expire(self, now_ms: float) -> None:
        del now_ms

    def estimate(self, model_id: int, gpu_id: int, now_ms: float) -> LoadEstimate:
        del now_ms
        model = self.models[model_id]
        cache = self.gpu_caches[gpu_id]
        cached_bytes = sum(
            group.size for group in model.tensor_groups if group.fingerprint in cache
        )
        to_load_bytes = model.total_bytes - cached_bytes
        return LoadEstimate(
            gpu_id=gpu_id,
            cached_bytes=cached_bytes,
            to_load_bytes=to_load_bytes,
            total_model_bytes=model.total_bytes,
            full_model_hit=to_load_bytes == 0,
            load_ms=self._load_ms(to_load_bytes),
        )

    def commit(self, model_id: int, gpu_id: int, now_ms: float) -> LoadEstimate:
        estimate = self.estimate(model_id, gpu_id, now_ms)
        model = self.models[model_id]
        cache = self.gpu_caches[gpu_id]
        protected = {group.fingerprint for group in model.tensor_groups}

        for group in model.tensor_groups:
            if group.fingerprint in cache:
                cache.move_to_end(group.fingerprint)

        missing = [group for group in model.tensor_groups if group.fingerprint not in cache]
        missing_bytes = sum(group.size for group in missing)
        if missing_bytes > self.gpu_capacity_bytes:
            raise ValueError(
                f"Model {model_id} needs {missing_bytes} bytes, "
                f"larger than GPU capacity {self.gpu_capacity_bytes}"
            )

        while self.gpu_used_bytes[gpu_id] + missing_bytes > self.gpu_capacity_bytes:
            evicted = False
            for fingerprint, size in list(cache.items()):
                if fingerprint in protected:
                    continue
                cache.pop(fingerprint)
                self.gpu_used_bytes[gpu_id] -= size
                evicted = True
                break
            if not evicted:
                raise ValueError(
                    f"Cannot fit model {model_id} on GPU {gpu_id}; "
                    "all remaining cache entries are required by this model"
                )

        for group in missing:
            cache[group.fingerprint] = group.size
            self.gpu_used_bytes[gpu_id] += group.size
        return estimate


class _CppVRAMEstimate(ctypes.Structure):
    _fields_ = [
        ("cached_bytes", ctypes.c_uint64),
        ("to_load_bytes", ctypes.c_uint64),
        ("total_model_bytes", ctypes.c_uint64),
        ("load_ms", ctypes.c_double),
        ("full_model_hit", ctypes.c_int),
    ]


class _CppVRAMLoadResult(ctypes.Structure):
    _fields_ = [
        ("cached_bytes", ctypes.c_uint64),
        ("to_load_bytes", ctypes.c_uint64),
        ("total_model_bytes", ctypes.c_uint64),
        ("estimated_load_ms", ctypes.c_double),
        ("wall_load_ms", ctypes.c_double),
        ("full_model_hit", ctypes.c_int),
    ]


class CppVRAMBackend:
    """ctypes bridge to the real C++ VRAMManager backend."""

    def __init__(
        self,
        models: Sequence[ModelInfo],
        num_gpus: int,
        gpu_capacity_bytes: int,
        load_bandwidth_gbps: float,
        load_overhead_ms: float,
        library_path: Path,
        reuse_granularity: int,
        mock_copy: bool,
        load_time_source: str,
        free_strategy: int,
        allocate_strategy: int,
        gpu_bandwidth_gbps: float,
        cpu_bandwidth_gbps: float,
        disable_parameter_reuse: bool,
    ) -> None:
        if not library_path.exists():
            raise FileNotFoundError(
                f"C++ VRAM backend library not found: {library_path}. "
                "Build it with: cmake --build tools/mock_allocation/build "
                "--target tangram_vram_backend"
            )
        self.models = {model.model_id: model for model in models}
        self.num_gpus = num_gpus
        self.load_time_source = load_time_source
        self._lib = ctypes.CDLL(str(library_path))
        self._configure_library()
        self._handle = self._lib.tangram_vram_create(
            ctypes.c_int(num_gpus),
            ctypes.c_uint64(gpu_capacity_bytes),
            ctypes.c_double(gpu_bandwidth_gbps * 1024.0 * 1024.0 * 1024.0),
            ctypes.c_double(cpu_bandwidth_gbps * 1024.0 * 1024.0 * 1024.0),
            ctypes.c_int(1 if mock_copy else 0),
            ctypes.c_double(load_bandwidth_gbps),
            ctypes.c_double(load_overhead_ms),
            ctypes.c_int(free_strategy),
            ctypes.c_int(allocate_strategy),
            ctypes.c_int(1 if disable_parameter_reuse else 0),
        )
        if not self._handle:
            raise RuntimeError("Failed to create C++ VRAM backend")

        for model in models:
            rc = self._lib.tangram_vram_register_model(
                self._handle,
                ctypes.c_int(model.model_id),
                str(model.path).encode(),
                ctypes.c_double(model.sensitivity),
                ctypes.c_int(reuse_granularity),
            )
            if rc != 0:
                raise RuntimeError(self._last_error())

    def _configure_library(self) -> None:
        self._lib.tangram_vram_create.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.tangram_vram_create.restype = ctypes.c_void_p
        self._lib.tangram_vram_destroy.argtypes = [ctypes.c_void_p]
        self._lib.tangram_vram_last_error.argtypes = [ctypes.c_void_p]
        self._lib.tangram_vram_last_error.restype = ctypes.c_char_p
        self._lib.tangram_vram_register_model.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_int,
        ]
        self._lib.tangram_vram_register_model.restype = ctypes.c_int
        self._lib.tangram_vram_estimate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_CppVRAMEstimate),
        ]
        self._lib.tangram_vram_estimate.restype = ctypes.c_int
        self._lib.tangram_vram_load_model.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_CppVRAMLoadResult),
        ]
        self._lib.tangram_vram_load_model.restype = ctypes.c_int

    def _last_error(self) -> str:
        raw = self._lib.tangram_vram_last_error(self._handle)
        return raw.decode(errors="replace") if raw else "unknown C++ VRAM backend error"

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.tangram_vram_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def expire(self, now_ms: float) -> None:
        del now_ms

    def estimate(self, model_id: int, gpu_id: int, now_ms: float) -> LoadEstimate:
        del now_ms
        out = _CppVRAMEstimate()
        rc = self._lib.tangram_vram_estimate(
            self._handle, ctypes.c_int(model_id), ctypes.c_int(gpu_id), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(self._last_error())
        return LoadEstimate(
            gpu_id=gpu_id,
            cached_bytes=int(out.cached_bytes),
            to_load_bytes=int(out.to_load_bytes),
            total_model_bytes=int(out.total_model_bytes),
            full_model_hit=bool(out.full_model_hit),
            load_ms=float(out.load_ms),
        )

    def commit(self, model_id: int, gpu_id: int, now_ms: float) -> LoadEstimate:
        del now_ms
        out = _CppVRAMLoadResult()
        rc = self._lib.tangram_vram_load_model(
            self._handle, ctypes.c_int(model_id), ctypes.c_int(gpu_id), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(self._last_error())
        load_ms = (
            float(out.wall_load_ms)
            if self.load_time_source == "wall"
            else float(out.estimated_load_ms)
        )
        return LoadEstimate(
            gpu_id=gpu_id,
            cached_bytes=int(out.cached_bytes),
            to_load_bytes=int(out.to_load_bytes),
            total_model_bytes=int(out.total_model_bytes),
            full_model_hit=bool(out.full_model_hit),
            load_ms=load_ms,
        )


class SLLMBackend(OnlineVRAMBackend):
    """Model-level keep-alive backend for SLLM-CM."""

    def __init__(
        self,
        models: Sequence[ModelInfo],
        num_gpus: int,
        gpu_capacity_bytes: int,
        load_bandwidth_gbps: float,
        load_overhead_ms: float,
        keep_alive_ms: float,
    ) -> None:
        super().__init__(
            models=models,
            num_gpus=num_gpus,
            gpu_capacity_bytes=gpu_capacity_bytes,
            load_bandwidth_gbps=load_bandwidth_gbps,
            load_overhead_ms=load_overhead_ms,
        )
        self.keep_alive_ms = keep_alive_ms
        self.last_model_finish: Dict[tuple[int, int], float] = {}

    def estimate(self, model_id: int, gpu_id: int, now_ms: float) -> LoadEstimate:
        if self.keep_alive_ms >= 0.0:
            return super().estimate(model_id, gpu_id, now_ms)

        model = self.models[model_id]
        return LoadEstimate(
            gpu_id=gpu_id,
            cached_bytes=0,
            to_load_bytes=model.total_bytes,
            total_model_bytes=model.total_bytes,
            full_model_hit=False,
            load_ms=self._load_ms(model.total_bytes),
        )

    def commit(self, model_id: int, gpu_id: int, now_ms: float) -> LoadEstimate:
        if self.keep_alive_ms >= 0.0:
            return super().commit(model_id, gpu_id, now_ms)
        return self.estimate(model_id, gpu_id, now_ms)

    def expire(self, now_ms: float) -> None:
        if self.keep_alive_ms < 0.0:
            return
        for (gpu_id, model_id), finish_ms in list(self.last_model_finish.items()):
            if now_ms <= finish_ms + self.keep_alive_ms:
                continue
            fingerprint = self.models[model_id].tensor_groups[0].fingerprint
            size = self.gpu_caches[gpu_id].pop(fingerprint, None)
            if size is not None:
                self.gpu_used_bytes[gpu_id] -= size
            del self.last_model_finish[(gpu_id, model_id)]

    def mark_finish(self, model_id: int, gpu_id: int, finish_ms: float) -> None:
        self.last_model_finish[(gpu_id, model_id)] = finish_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--system", choices=["sllm_cm", "tangram"], required=True)
    parser.add_argument(
        "--backend",
        choices=["python", "cpp"],
        default="python",
        help="Use the lightweight Python cache model or the C++ VRAMManager binding.",
    )
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--gpu-memory-gb", type=float, default=40.0)
    parser.add_argument("--slo-scales", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--engine-overhead-ms", type=float, default=20.0)
    parser.add_argument("--prefill-ms-per-token", type=float, default=0.02)
    parser.add_argument("--decode-ms-per-token", type=float, default=2.0)
    parser.add_argument("--base-slo-ms", type=float, default=0.0)
    parser.add_argument("--load-bandwidth-gbps", type=float, default=20.0)
    parser.add_argument("--load-overhead-ms", type=float, default=0.0)
    parser.add_argument(
        "--cpp-backend-lib",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "tools/mock_allocation/build/libtangram_vram_backend.so",
    )
    parser.add_argument(
        "--cpp-real-copy",
        action="store_false",
        dest="cpp_mock_copy",
        help="Use real CUDA allocation/copy in the C++ backend instead of mock-copy mode.",
    )
    parser.set_defaults(cpp_mock_copy=True)
    parser.add_argument(
        "--cpp-load-time-source",
        choices=["estimated", "wall"],
        default="estimated",
        help="Use byte/bandwidth estimated load time or measured C++ LoadModel wall time.",
    )
    parser.add_argument("--cpp-free-strategy", type=int, default=1)
    parser.add_argument("--cpp-allocate-strategy", type=int, default=4)
    parser.add_argument("--cpp-gpu-bandwidth-gbps", type=float, default=400.0)
    parser.add_argument("--cpp-cpu-bandwidth-gbps", type=float, default=20.0)
    parser.add_argument("--disable-parameter-reuse", action="store_true")
    parser.add_argument("--sllm-keep-alive-ms", type=float, default=0.0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--warmup-step", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument(
        "--scheduler-policy",
        choices=["online_ttft", "locality", "random"],
        default="online_ttft",
        help="How the online scheduler chooses among candidate GPUs.",
    )
    parser.add_argument(
        "--occupy-until",
        choices=["load", "ttft", "finish"],
        default="finish",
        help="How long each request occupies its assigned GPU.",
    )
    parser.add_argument("--output-request-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-ttft-cdf", type=Path, default=None)
    parser.add_argument("--append-summary", action="store_true")
    return parser.parse_args()


def parse_scales(raw: str) -> List[float]:
    scales = [float(item) for item in raw.replace(",", " ").split()]
    if not scales:
        raise ValueError("--slo-scales must contain at least one value")
    return scales


def read_trace(path: Path, max_requests: int = 0) -> List[Request]:
    requests: List[Request] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "timestamp":
                continue
            if len(fields) < 4:
                raise ValueError(f"Invalid trace row at {path}:{line_no}: {line}")
            requests.append(
                Request(
                    req_id=len(requests),
                    arrival_ms=float(fields[0]) * 1000.0,
                    model_id=int(fields[1]),
                    input_tokens=int(fields[2]),
                    output_tokens=int(fields[3]),
                )
            )
            if max_requests > 0 and len(requests) >= max_requests:
                break
    return requests


def read_models(config_path: Path, reuse_granularity: int) -> List[ModelInfo]:
    with config_path.open() as f:
        config = json.load(f)

    raw_models = []
    if "model_lists" in config:
        raw_models = [
            (int(item.get("id", idx)), Path(item["path"]), float(item.get("sensitivity", 1.0)))
            for idx, item in enumerate(config["model_lists"])
        ]
    elif "model_dirs" in config:
        raw_models = [(idx, Path(path), 1.0) for idx, path in enumerate(config["model_dirs"])]
    else:
        raise ValueError(f"{config_path} must contain model_lists or model_dirs")

    models: List[ModelInfo] = []
    for model_id, model_path, sensitivity in sorted(raw_models, key=lambda item: item[0]):
        tensor_groups = read_tensor_groups(model_path)
        if reuse_granularity == 0:
            total_bytes = sum(group.size for group in tensor_groups)
            tensor_groups = [TensorGroup(f"model:{model_id}:{model_path}", total_bytes)]
        models.append(
            ModelInfo(
                model_id=model_id,
                path=model_path,
                tensor_groups=tensor_groups,
                sensitivity=sensitivity,
            )
        )
    return models


def read_tensor_groups(model_path: Path) -> List[TensorGroup]:
    index_path = model_path / "tensor_group_index.txt"
    if not index_path.exists():
        size = model_file_size(model_path)
        return [TensorGroup(f"model:{model_path}", size)]

    groups: List[TensorGroup] = []
    current_size: int | None = None
    current_fingerprint: str | None = None
    size_pattern = re.compile(r"Group Size:\s*(\d+)")
    fingerprint_pattern = re.compile(r"Fingerprint:\s*(\S+)")
    with index_path.open() as f:
        for line in f:
            size_match = size_pattern.search(line)
            if size_match:
                current_size = int(size_match.group(1))
                continue
            fingerprint_match = fingerprint_pattern.search(line)
            if fingerprint_match:
                current_fingerprint = fingerprint_match.group(1)
                if current_size is not None:
                    groups.append(TensorGroup(current_fingerprint, current_size))
                    current_size = None
                    current_fingerprint = None
    if not groups:
        groups.append(TensorGroup(f"model:{model_path}", model_file_size(model_path)))
    return groups


def model_file_size(model_path: Path) -> int:
    total = 0
    for path in sorted(model_path.glob("tensor.data_*")):
        total += path.stat().st_size
    if total == 0:
        raise ValueError(f"No tensor.data_* files found under {model_path}")
    return total


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def build_backend(args: argparse.Namespace) -> OnlineVRAMBackend:
    reuse_granularity = 0 if args.system == "sllm_cm" else 1
    models = read_models(args.config, reuse_granularity)
    capacity = int(args.gpu_memory_gb * 1024 * 1024 * 1024)
    if args.backend == "cpp":
        return CppVRAMBackend(
            models=models,
            num_gpus=args.num_gpus,
            gpu_capacity_bytes=capacity,
            load_bandwidth_gbps=args.load_bandwidth_gbps,
            load_overhead_ms=args.load_overhead_ms,
            library_path=args.cpp_backend_lib,
            reuse_granularity=reuse_granularity,
            mock_copy=args.cpp_mock_copy,
            load_time_source=args.cpp_load_time_source,
            free_strategy=args.cpp_free_strategy,
            allocate_strategy=args.cpp_allocate_strategy,
            gpu_bandwidth_gbps=args.cpp_gpu_bandwidth_gbps,
            cpu_bandwidth_gbps=args.cpp_cpu_bandwidth_gbps,
            disable_parameter_reuse=args.disable_parameter_reuse,
        )
    if args.system == "sllm_cm":
        return SLLMBackend(
            models=models,
            num_gpus=args.num_gpus,
            gpu_capacity_bytes=capacity,
            load_bandwidth_gbps=args.load_bandwidth_gbps,
            load_overhead_ms=args.load_overhead_ms,
            keep_alive_ms=args.sllm_keep_alive_ms,
        )
    return OnlineVRAMBackend(
        models=models,
        num_gpus=args.num_gpus,
        gpu_capacity_bytes=capacity,
        load_bandwidth_gbps=args.load_bandwidth_gbps,
        load_overhead_ms=args.load_overhead_ms,
    )


def choose_gpu(
    request: Request,
    estimates: Sequence[LoadEstimate],
    load_available: Sequence[float],
    infer_available: Sequence[float],
    prefill_ms: float,
    policy: str,
    rng: random.Random,
) -> LoadEstimate:
    if policy == "random":
        return estimates[rng.randrange(len(estimates))]
    if policy == "locality":
        return min(
            estimates,
            key=lambda est: (
                est.to_load_bytes,
                max(request.arrival_ms, load_available[est.gpu_id]) + est.load_ms,
                infer_available[est.gpu_id],
                est.gpu_id,
            ),
        )
    return min(
        estimates,
        key=lambda est: (
            max(
                max(request.arrival_ms, load_available[est.gpu_id]) + est.load_ms,
                infer_available[est.gpu_id],
            )
            + prefill_ms,
            est.to_load_bytes,
            est.gpu_id,
        ),
    )


def simulate(
    requests: Sequence[Request],
    backend: OnlineVRAMBackend,
    num_gpus: int,
    engine_overhead_ms: float,
    prefill_ms_per_token: float,
    decode_ms_per_token: float,
    base_slo_ms: float,
    occupy_until: str,
    scheduler_policy: str,
    rng: random.Random,
    collect_rows: bool = True,
) -> List[SimulatedRequest]:
    load_available = [0.0 for _ in range(num_gpus)]
    infer_available = [0.0 for _ in range(num_gpus)]
    rows: List[SimulatedRequest] = []

    for request in requests:
        backend.expire(request.arrival_ms)
        prefill_ms = engine_overhead_ms + request.input_tokens * prefill_ms_per_token
        decode_ms = request.output_tokens * decode_ms_per_token
        estimates = [
            backend.estimate(request.model_id, gpu_id, request.arrival_ms)
            for gpu_id in range(num_gpus)
        ]
        chosen = choose_gpu(
            request=request,
            estimates=estimates,
            load_available=load_available,
            infer_available=infer_available,
            prefill_ms=prefill_ms,
            policy=scheduler_policy,
            rng=rng,
        )
        committed = backend.commit(request.model_id, chosen.gpu_id, request.arrival_ms)

        load_start = max(request.arrival_ms, load_available[chosen.gpu_id])
        load_end = load_start + committed.load_ms
        load_available[chosen.gpu_id] = load_end

        prefill_start = max(load_end, infer_available[chosen.gpu_id])
        ttft_time = prefill_start + prefill_ms
        finish_time = ttft_time + decode_ms
        if occupy_until == "ttft":
            infer_available[chosen.gpu_id] = ttft_time
        elif occupy_until == "finish":
            infer_available[chosen.gpu_id] = finish_time

        if isinstance(backend, SLLMBackend):
            backend.mark_finish(request.model_id, chosen.gpu_id, finish_time)

        if not collect_rows:
            continue
        request_base_slo = (
            base_slo_ms
            if base_slo_ms > 0.0
            else engine_overhead_ms + request.input_tokens * prefill_ms_per_token
        )
        rows.append(
            SimulatedRequest(
                req_id=request.req_id,
                arrival_ms=request.arrival_ms,
                model_id=request.model_id,
                gpu_id=chosen.gpu_id,
                queue_ms=max(
                    0.0, prefill_start - request.arrival_ms - committed.load_ms
                ),
                load_ms=committed.load_ms,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                ttft_ms=ttft_time - request.arrival_ms,
                e2e_ms=finish_time - request.arrival_ms,
                base_slo_ms=request_base_slo,
                to_load_bytes=committed.to_load_bytes,
                cached_bytes=committed.cached_bytes,
                total_model_bytes=committed.total_model_bytes,
                full_model_hit=committed.full_model_hit,
            )
        )
    return rows


def write_requests(path: Path, rows: Iterable[SimulatedRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "req_id",
                "arrival_ms",
                "model_id",
                "gpu_id",
                "queue_ms",
                "load_ms",
                "prefill_ms",
                "decode_ms",
                "ttft_ms",
                "e2e_ms",
                "base_slo_ms",
                "to_load_bytes",
                "cached_bytes",
                "total_model_bytes",
                "full_model_hit",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.req_id,
                    f"{row.arrival_ms:.6f}",
                    row.model_id,
                    row.gpu_id,
                    f"{row.queue_ms:.6f}",
                    f"{row.load_ms:.6f}",
                    f"{row.prefill_ms:.6f}",
                    f"{row.decode_ms:.6f}",
                    f"{row.ttft_ms:.6f}",
                    f"{row.e2e_ms:.6f}",
                    f"{row.base_slo_ms:.6f}",
                    row.to_load_bytes,
                    row.cached_bytes,
                    row.total_model_bytes,
                    1 if row.full_model_hit else 0,
                ]
            )


def write_ttft_cdf(path: Path, rows: Sequence[SimulatedRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ttfts = sorted(row.ttft_ms for row in rows)
    with path.open("w") as f:
        total = len(ttfts)
        for idx, ttft_ms in enumerate(ttfts, 1):
            f.write(f"{ttft_ms:.6f} {idx / total if total else 0.0:.6f}\n")


def append_summary(
    path: Path,
    rows: Sequence[SimulatedRequest],
    system: str,
    rps: float,
    num_gpus: int,
    slo_scales: Sequence[float],
    append: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    mode = "a" if append else "w"
    ttfts = [row.ttft_ms for row in rows]
    e2es = [row.e2e_ms for row in rows]
    load_times = [row.load_ms for row in rows]
    full_model_hit_ratio = (
        sum(1 for row in rows if row.full_model_hit) / len(rows) if rows else 0.0
    )
    total_to_load = sum(row.to_load_bytes for row in rows)
    total_model = sum(row.total_model_bytes for row in rows)
    reuse_ratio = 1.0 - total_to_load / total_model if total_model > 0 else 0.0

    with path.open(mode, newline="") as f:
        writer = csv.writer(f)
        if not append or not exists:
            writer.writerow(
                [
                    "system",
                    "rps",
                    "gpu_num",
                    "slo_scale",
                    "slo_attainment",
                    "request_count",
                    "mean_ttft_ms",
                    "p50_ttft_ms",
                    "p90_ttft_ms",
                    "p95_ttft_ms",
                    "p99_ttft_ms",
                    "mean_e2e_ms",
                    "p99_e2e_ms",
                    "mean_load_ms",
                    "p99_load_ms",
                    "reuse_ratio",
                    "keep_alive_hit_ratio",
                ]
            )
        for scale in slo_scales:
            attained = sum(row.ttft_ms <= row.base_slo_ms * scale for row in rows)
            writer.writerow(
                [
                    system,
                    f"{rps:g}",
                    num_gpus,
                    f"{scale:g}",
                    f"{attained / len(rows) if rows else 0.0:.6f}",
                    len(rows),
                    f"{mean(ttfts) if ttfts else 0.0:.6f}",
                    f"{percentile(ttfts, 50):.6f}",
                    f"{percentile(ttfts, 90):.6f}",
                    f"{percentile(ttfts, 95):.6f}",
                    f"{percentile(ttfts, 99):.6f}",
                    f"{mean(e2es) if e2es else 0.0:.6f}",
                    f"{percentile(e2es, 99):.6f}",
                    f"{mean(load_times) if load_times else 0.0:.6f}",
                    f"{percentile(load_times, 99):.6f}",
                    f"{reuse_ratio:.6f}",
                    f"{full_model_hit_ratio:.6f}",
                ]
            )


def main() -> None:
    args = parse_args()
    all_requests = read_trace(args.trace, args.max_requests)
    warmup_requests = all_requests[: args.warmup_step]
    requests = all_requests[args.warmup_step :]
    rng = random.Random(args.random_seed)

    backend = build_backend(args)
    if warmup_requests:
        simulate(
            requests=warmup_requests,
            backend=backend,
            num_gpus=args.num_gpus,
            engine_overhead_ms=args.engine_overhead_ms,
            prefill_ms_per_token=args.prefill_ms_per_token,
            decode_ms_per_token=args.decode_ms_per_token,
            base_slo_ms=args.base_slo_ms,
            occupy_until=args.occupy_until,
            scheduler_policy=args.scheduler_policy,
            rng=rng,
            collect_rows=False,
        )

    rows = simulate(
        requests=requests,
        backend=backend,
        num_gpus=args.num_gpus,
        engine_overhead_ms=args.engine_overhead_ms,
        prefill_ms_per_token=args.prefill_ms_per_token,
        decode_ms_per_token=args.decode_ms_per_token,
        base_slo_ms=args.base_slo_ms,
        occupy_until=args.occupy_until,
        scheduler_policy=args.scheduler_policy,
        rng=rng,
    )
    write_requests(args.output_request_csv, rows)
    if args.output_ttft_cdf is not None:
        write_ttft_cdf(args.output_ttft_cdf, rows)
    append_summary(
        args.output_summary_csv,
        rows,
        args.system,
        args.rps,
        args.num_gpus,
        parse_scales(args.slo_scales),
        args.append_summary,
    )
    close_backend = getattr(backend, "close", None)
    if close_backend is not None:
        close_backend()
    print(
        f"Simulated {len(rows)} requests for {args.system}, "
        f"backend={args.backend}, rps={args.rps:g}, "
        f"gpu={args.num_gpus}, policy={args.scheduler_policy}"
    )


if __name__ == "__main__":
    main()
