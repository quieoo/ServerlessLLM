#!/usr/bin/env python3
"""LayerPipe: trace-driven, single-GPU layer-load/prefill pipeline.

This benchmark executes real transformer requests.  Model weights initially
reside in CPU memory.  For a new batch, embeddings and layer 0 are copied to
the GPU synchronously.  A separate CUDA stream copies layer i+1 while layer i
computes.  Once prefill completes, all layers are resident and decode runs
normally.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import statistics
import struct
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)


@dataclass
class TraceRequest:
    request_id: int
    arrival_s: float
    model_id: int
    input_tokens: int
    output_tokens: int


@dataclass
class RequestResult:
    request_id: int
    model_id: int
    arrival_s: float
    dispatch_s: float
    first_token_s: float
    finish_s: float
    input_tokens: int
    output_tokens: int
    batch_id: int
    batch_size: int

    @property
    def queue_ms(self) -> float:
        return (self.dispatch_s - self.arrival_s) * 1000.0

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_s - self.arrival_s) * 1000.0

    @property
    def service_ttft_ms(self) -> float:
        return (self.first_token_s - self.dispatch_s) * 1000.0

    @property
    def e2e_ms(self) -> float:
        return (self.finish_s - self.arrival_s) * 1000.0

    def record(self) -> dict:
        value = asdict(self)
        value.update(
            queue_ms=self.queue_ms,
            ttft_ms=self.ttft_ms,
            service_ttft_ms=self.service_ttft_ms,
            e2e_ms=self.e2e_ms,
        )
        return value


def parse_trace(path: Path, max_requests: int) -> List[TraceRequest]:
    requests: List[TraceRequest] = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            try:
                if len(fields) < 4:
                    raise ValueError(
                        "expected timestamp model_id input_tokens output_tokens"
                    )
                request = TraceRequest(
                    request_id=len(requests),
                    arrival_s=float(fields[0]),
                    model_id=int(fields[1]),
                    input_tokens=max(1, int(fields[2])),
                    output_tokens=max(1, int(fields[3])),
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            requests.append(request)
            if max_requests > 0 and len(requests) >= max_requests:
                break
    if not requests:
        raise ValueError(f"No requests found in {path}")
    base = requests[0].arrival_s
    for request in requests:
        request.arrival_s -= base
    return requests


def load_model_paths(config_path: Path,
                     overrides: Sequence[str]) -> Dict[int, str]:
    with config_path.open() as stream:
        config = json.load(stream)
    paths = {}
    for item in config["model_lists"]:
        model_id = int(item["id"])
        # LayerPipe needs a Hugging Face checkpoint, not rank_0 tensor.data_*.
        raw_path = item.get("hf_path", item["path"])
        path = Path(raw_path)
        if path.name == "rank_0":
            path = path.parent
        paths[model_id] = str(path.resolve())
    for value in overrides:
        try:
            model_id_text, path_text = value.split("=", 1)
            paths[int(model_id_text)] = str(Path(path_text).resolve())
        except ValueError as exc:
            raise ValueError(
                f"Invalid --model-path {value!r}; expected MODEL_ID=PATH"
            ) from exc
    return paths


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def summarize(values: Sequence[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else 0.0,
    }


def log_event(name: str, **fields) -> None:
    """Emit one machine-readable, immediately visible scheduling event."""
    print(
        f"LAYERPIPE_{name}=" + json.dumps(fields, sort_keys=True),
        flush=True,
    )


def find_transformer_parts(model):
    candidates = (
        ("model.layers", lambda item: item.model.layers),
        ("language_model.model.layers",
         lambda item: item.language_model.model.layers),
        ("model.decoder.layers", lambda item: item.model.decoder.layers),
        ("transformer.h", lambda item: item.transformer.h),
        ("gpt_neox.layers", lambda item: item.gpt_neox.layers),
    )
    for name, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if len(layers):
            break
    else:
        raise TypeError(
            f"Unsupported architecture {type(model).__name__}: "
            "cannot locate decoder layers"
        )

    norm_candidates = []
    for dotted in ("model.norm", "model.decoder.final_layer_norm",
                   "language_model.model.norm", "transformer.ln_f",
                   "gpt_neox.final_layer_norm"):
        current = model
        try:
            for component in dotted.split("."):
                current = getattr(current, component)
            norm_candidates.append(current)
        except AttributeError:
            pass
    return name, layers, norm_candidates


def module_bytes(module: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(
            module.buffers(recurse=True)):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


class CpuWeightStore:
    """Packed CPU masters plus reusable GPU arenas for high-bandwidth H2D."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.locations = []
        self.entries = {}
        self.cpu_arenas = {}
        self.gpu_arenas = {}
        self.gpu_resident_ids = set()
        self.pinned = True
        self._arena_target_bytes = 1024 * 1024 * 1024
        seen_locations = set()
        tensor_map = {}
        for module in model.modules():
            for kind, container in (
                    ("parameter", module._parameters),
                    ("buffer", module._buffers)):
                for name, tensor in container.items():
                    if tensor is None:
                        continue
                    location = (id(module), kind, name)
                    if location in seen_locations:
                        continue
                    seen_locations.add(location)
                    tensor_id = id(tensor)
                    if tensor_id not in tensor_map:
                        tensor_map[tensor_id] = tensor.detach().to("cpu")
                    self.locations.append(
                        (module, kind, name, tensor_id, tensor.requires_grad)
                    )
        self._pack_cpu_tensors(tensor_map)
        self.restore_cpu()

    def _new_cpu_arena(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        try:
            return torch.empty(
                numel, dtype=dtype, device="cpu", pin_memory=True
            )
        except RuntimeError:
            self.pinned = False
            return torch.empty(numel, dtype=dtype, device="cpu")

    def _pack_cpu_tensors(self, tensors) -> None:
        by_dtype = {}
        for tensor_id, tensor in tensors.items():
            by_dtype.setdefault(tensor.dtype, []).append((tensor_id, tensor))
        for dtype, items in by_dtype.items():
            element_size = torch.empty((), dtype=dtype).element_size()
            target_numel = max(1, self._arena_target_bytes // element_size)
            groups = []
            current = []
            current_numel = 0
            for tensor_id, tensor in items:
                numel = tensor.numel()
                if current and current_numel + numel > target_numel:
                    groups.append((current, current_numel))
                    current = []
                    current_numel = 0
                current.append((tensor_id, tensor, current_numel))
                current_numel += numel
            if current:
                groups.append((current, current_numel))
            arenas = []
            for arena_index, (group, numel) in enumerate(groups):
                arena = self._new_cpu_arena(numel, dtype)
                arenas.append(arena)
                for tensor_id, source, offset in group:
                    view = arena[offset:offset + source.numel()].view(
                        source.shape
                    )
                    view.copy_(source)
                    self.entries[tensor_id] = {
                        "dtype": dtype,
                        "arena_index": arena_index,
                        "offset": offset,
                        "numel": source.numel(),
                        "shape": tuple(source.shape),
                    }
            self.cpu_arenas[dtype] = arenas

    def _view(self, tensor_id: int, device: str) -> torch.Tensor:
        entry = self.entries[tensor_id]
        arenas = (
            self.cpu_arenas if device == "cpu" else self.gpu_arenas
        )
        arena = arenas[entry["dtype"]][entry["arena_index"]]
        return arena[
            entry["offset"]:entry["offset"] + entry["numel"]
        ].view(entry["shape"])

    def _bind(self, tensor_ids, device: str) -> None:
        tensor_ids = set(tensor_ids)
        parameters = {}
        for module, kind, name, tensor_id, requires_grad in self.locations:
            if tensor_id not in tensor_ids:
                continue
            view = self._view(tensor_id, device)
            if kind == "parameter":
                parameter = parameters.get(tensor_id)
                if parameter is None:
                    current = module._parameters.get(name)
                    if current is not None:
                        current.data = view
                        current.requires_grad_(requires_grad)
                        parameter = current
                    else:
                        parameter = torch.nn.Parameter(
                            view, requires_grad=requires_grad
                        )
                    parameters[tensor_id] = parameter
                module._parameters[name] = parameter
            else:
                module._buffers[name] = view

    def _ensure_gpu_arenas(self, device: torch.device) -> None:
        if self.gpu_arenas:
            return
        for dtype, cpu_arenas in self.cpu_arenas.items():
            self.gpu_arenas[dtype] = [
                torch.empty(arena.numel(), dtype=dtype, device=device)
                for arena in cpu_arenas
            ]

    def _module_tensor_ids(self, modules) -> set:
        module_ids = {id(item) for root in modules for item in root.modules()}
        return {
            tensor_id
            for module, _kind, _name, tensor_id, _requires_grad
            in self.locations
            if id(module) in module_ids
        }

    def copy_modules(self, modules, device: torch.device) -> int:
        """Copy merged arena ranges for modules on the current CUDA stream."""
        self._ensure_gpu_arenas(device)
        tensor_ids = (
            self._module_tensor_ids(modules) - self.gpu_resident_ids
        )
        intervals = {}
        for tensor_id in tensor_ids:
            entry = self.entries[tensor_id]
            key = (entry["dtype"], entry["arena_index"])
            intervals.setdefault(key, []).append(
                (entry["offset"], entry["offset"] + entry["numel"])
            )
        copied = 0
        for (dtype, arena_index), ranges in intervals.items():
            ranges.sort()
            merged = []
            for start, end in ranges:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            source = self.cpu_arenas[dtype][arena_index]
            target = self.gpu_arenas[dtype][arena_index]
            for start, end in merged:
                target[start:end].copy_(
                    source[start:end], non_blocking=True
                )
                copied += (end - start) * source.element_size()
        self._bind(tensor_ids, "gpu")
        self.gpu_resident_ids.update(tensor_ids)
        return copied

    def copy_all(self, device: torch.device) -> int:
        self._ensure_gpu_arenas(device)
        copied = 0
        for dtype, cpu_arenas in self.cpu_arenas.items():
            for source, target in zip(
                    cpu_arenas, self.gpu_arenas[dtype]):
                target.copy_(source, non_blocking=True)
                copied += source.numel() * source.element_size()
        self._bind(self.entries, "gpu")
        self.gpu_resident_ids = set(self.entries)
        return copied

    def restore_cpu(self) -> None:
        """Rebind parameters to CPU masters and drop GPU storages without D2H."""
        self._bind(self.entries, "cpu")
        self.gpu_arenas.clear()
        self.gpu_resident_ids.clear()

    @property
    def bytes(self) -> int:
        return sum(
            arena.numel() * arena.element_size()
            for arenas in self.cpu_arenas.values() for arena in arenas
        )


class LayerPipeline:
    def __init__(self, model, device: torch.device, weights: CpuWeightStore):
        self.model = model
        self.device = device
        self.weights = weights
        self.layer_path, self.layers, self.norm_modules = (
            find_transformer_parts(model)
        )
        self.embedding = model.get_input_embeddings()
        self.output_embedding = model.get_output_embeddings()
        self.copy_stream = torch.cuda.Stream(device=device)
        self.events: List[Optional[torch.cuda.Event]] = [
            None for _ in self.layers
        ]
        self.load_started: List[Optional[torch.cuda.Event]] = [
            None for _ in self.layers
        ]
        self.hooks = []
        self.tail_event: Optional[torch.cuda.Event] = None
        self.tail_started: Optional[torch.cuda.Event] = None
        self.scheduled = set()
        self.fully_resident = False
        self._install_hooks()

    def _copy_module(self, module, key) -> torch.cuda.Event:
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.copy_stream):
            started.record(self.copy_stream)
            self.weights.copy_modules([module], self.device)
            finished.record(self.copy_stream)
        if isinstance(key, int):
            self.load_started[key] = started
            self.events[key] = finished
        else:
            self.tail_started = started
            self.tail_event = finished
        self.scheduled.add(key)
        return finished

    def _schedule_layer(self, index: int) -> None:
        if index not in self.scheduled:
            self._copy_module(self.layers[index], index)

    def _schedule_tail(self) -> None:
        if "tail" in self.scheduled:
            return
        # Move only modules used after the decoder stack. Tied output weights
        # are already moved with the input embedding.
        with torch.cuda.stream(self.copy_stream):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record(self.copy_stream)
            tail_modules = list(self.norm_modules)
            if (self.output_embedding is not None
                    and self.output_embedding is not self.embedding):
                tail_modules.append(self.output_embedding)
            self.weights.copy_modules(tail_modules, self.device)
            finished.record(self.copy_stream)
        self.tail_started = started
        self.tail_event = finished
        self.scheduled.add("tail")

    def _layer_pre_hook(self, index: int):
        def hook(_module, _args):
            event = self.events[index]
            if event is not None:
                torch.cuda.current_stream(self.device).wait_event(event)
            if index + 1 < len(self.layers):
                self._schedule_layer(index + 1)
            else:
                self._schedule_tail()
        return hook

    def _tail_pre_hook(self, _module, _args):
        if self.tail_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self.tail_event)

    def _install_hooks(self) -> None:
        for index, layer in enumerate(self.layers):
            self.hooks.append(
                layer.register_forward_pre_hook(self._layer_pre_hook(index))
            )
        for module in self.norm_modules:
            self.hooks.append(
                module.register_forward_pre_hook(self._tail_pre_hook)
            )
        if (self.output_embedding is not None
                and self.output_embedding is not self.embedding):
            self.hooks.append(
                self.output_embedding.register_forward_pre_hook(
                    self._tail_pre_hook
                )
            )

    def prepare(self) -> float:
        if self.fully_resident:
            return 0.0
        start = time.perf_counter()
        self.weights.copy_modules(
            [self.embedding, self.layers[0]], self.device
        )
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        self.events[0] = ready
        self.scheduled.add(0)
        torch.cuda.synchronize(self.device)
        return (time.perf_counter() - start) * 1000.0

    def loading_metrics(self) -> dict:
        torch.cuda.synchronize(self.device)
        per_layer = []
        for index, (started, finished) in enumerate(
                zip(self.load_started, self.events)):
            elapsed = (
                started.elapsed_time(finished)
                if started is not None and finished is not None else 0.0
            )
            per_layer.append({
                "layer": index,
                "bytes": module_bytes(self.layers[index]),
                "h2d_ms": elapsed,
            })
        tail_ms = (
            self.tail_started.elapsed_time(self.tail_event)
            if self.tail_started is not None and self.tail_event is not None
            else 0.0
        )
        return {
            "mode": "layerpipe",
            "pinned_cpu_weights": self.weights.pinned,
            "cpu_arena_count": sum(
                len(items) for items in self.weights.cpu_arenas.values()
            ),
            "layer_path": self.layer_path,
            "layers": per_layer,
            "tail_h2d_ms": tail_ms,
            "sum_h2d_ms": sum(item["h2d_ms"] for item in per_layer) + tail_ms,
        }

    def mark_prefill_complete(self) -> None:
        self.fully_resident = True

    def close(self) -> float:
        for hook in self.hooks:
            hook.remove()
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self.weights.restore_cpu()
        return (time.perf_counter() - start) * 1000.0


class NativeModelLoader:
    """Synchronous whole-model CPU-to-GPU loading without pipelining."""

    def __init__(self, model, device: torch.device, weights: CpuWeightStore):
        self.model = model
        self.device = device
        self.weights = weights
        self.fully_resident = False
        self.load_ms = 0.0
        self.model_size_bytes = module_bytes(model)

    def prepare(self) -> float:
        if self.fully_resident:
            return 0.0
        start = time.perf_counter()
        self.weights.copy_all(self.device)
        torch.cuda.synchronize(self.device)
        self.load_ms = (time.perf_counter() - start) * 1000.0
        return self.load_ms

    def mark_prefill_complete(self) -> None:
        self.fully_resident = True

    def loading_metrics(self) -> dict:
        return {
            "mode": "native",
            "pinned_cpu_weights": self.weights.pinned,
            "cpu_arena_count": sum(
                len(items) for items in self.weights.cpu_arenas.values()
            ),
            "model_bytes": self.model_size_bytes,
            "whole_model_h2d_ms": self.load_ms,
            "sum_h2d_ms": self.load_ms,
        }

    def close(self) -> float:
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self.weights.restore_cpu()
        return (time.perf_counter() - start) * 1000.0


def _load_vmm_pool_class():
    repo_root = Path(__file__).resolve().parents[2]
    elastic_kv = str(repo_root / "ElasticKV")
    if elastic_kv not in sys.path:
        sys.path.insert(0, elastic_kv)
    module = importlib.import_module(
        "vllm.model_executor.model_loader.tangram_vmm"
    )
    return module.TangramVmmPool


_SAFETENSORS_DTYPES = {
    "BOOL": "torch.bool",
    "U8": "torch.uint8",
    "I8": "torch.int8",
    "I16": "torch.int16",
    "I32": "torch.int32",
    "I64": "torch.int64",
    "F16": "torch.float16",
    "BF16": "torch.bfloat16",
    "F32": "torch.float32",
    "F64": "torch.float64",
}


def _contiguous_stride(shape: Sequence[int]) -> List[int]:
    stride = [0] * len(shape)
    value = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = value
        value *= shape[index]
    return stride


class SafetensorsVmmAdapters:
    """Metadata-only tensor.data compatibility views over HF shards."""

    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="layerpipe_vmm_hf_"
        )
        self.paths = {}

    def create(self, model_id: int, model_path: str) -> str:
        model_path = str(Path(model_path).resolve())
        adapter = Path(self.temporary.name) / f"model_{model_id}"
        adapter.mkdir()
        index_path = Path(model_path) / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open() as stream:
                weight_map = json.load(stream)["weight_map"]
            shard_names = sorted(set(weight_map.values()))
        else:
            shard_names = sorted(
                item.name for item in Path(model_path).glob("*.safetensors")
            )
        if not shard_names:
            raise ValueError(
                f"VMM mode requires Hugging Face safetensors at {model_path}"
            )

        tensor_meta = {}
        groups = []
        partition_base = 0
        for partition_id, shard_name in enumerate(shard_names):
            shard_path = Path(model_path) / shard_name
            os.symlink(
                shard_path,
                adapter / f"tensor.data_{partition_id}",
            )
            with shard_path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
            payload_base = partition_base + 8 + header_size
            for name, metadata in header.items():
                if name == "__metadata__":
                    continue
                dtype = metadata["dtype"]
                if dtype not in _SAFETENSORS_DTYPES:
                    raise ValueError(
                        f"Unsupported safetensors dtype {dtype} for {name}"
                    )
                begin, end = metadata["data_offsets"]
                shape = [int(value) for value in metadata["shape"]]
                size = int(end - begin)
                tensor_meta[name] = [
                    shape,
                    _contiguous_stride(shape),
                    _SAFETENSORS_DTYPES[dtype],
                ]
                groups.append(
                    (payload_base + int(begin), size, name)
                )
            partition_base += shard_path.stat().st_size

        with (adapter / "tensor_meta_index.json").open("w") as stream:
            json.dump(tensor_meta, stream)
        with (adapter / "tensor_group_index.txt").open("w") as stream:
            stream.write(f"Number of tensor groups: {len(groups)}\n\n")
            for offset, size, name in groups:
                stream.write(
                    f"Group Offset: {offset}\n"
                    f"Group Size: {size}\n"
                    f"Fingerprint: safetensors:{model_id}:{name}\n"
                    "Tensor Group Index:\n"
                    f"  Tensor Name: {name}\n"
                    "  Offset: 0\n"
                    f"  Size: {size}\n\n"
                )
        with (adapter / "safetensors_adapter.json").open("w") as stream:
            json.dump(
                {
                    "format": "huggingface_safetensors",
                    "model_path": model_path,
                    "shards": shard_names,
                    "tensors": len(groups),
                },
                stream,
                indent=2,
            )
        self.paths[model_id] = str(adapter)
        return str(adapter)

    def close(self) -> None:
        self.temporary.cleanup()


class VmmModelLoader:
    """Whole-model synchronous VMM load with cross-model page reuse."""

    def __init__(self, model, device: torch.device, pool, vmm_path: str,
                 hf_path: str):
        self.model = model
        self.device = device
        self.pool = pool
        self.vmm_path = vmm_path
        self.hf_path = hf_path
        self.bound = False
        self.load_result = None
        self.prepare_ms = 0.0
        self.bind_ms = 0.0

    def prepare(self) -> float:
        start = time.perf_counter()
        if not self.bound:
            state_dict, self.load_result = self.pool.load_state_dict(
                self.vmm_path
            )
            bind_start = time.perf_counter()
            incompatible = self.model.load_state_dict(
                state_dict, strict=False, assign=True
            )
            allowed_missing = (
                {"lm_head.weight"}
                if getattr(self.model.config, "tie_word_embeddings", False)
                else set()
            )
            unexpected_missing = (
                set(incompatible.missing_keys) - allowed_missing
            )
            if unexpected_missing or incompatible.unexpected_keys:
                raise ValueError(
                    "VMM/HF state dict mismatch: "
                    f"missing={sorted(unexpected_missing)[:16]}, "
                    f"unexpected={incompatible.unexpected_keys[:16]}"
                )
            self.model.tie_weights()
            self.model.eval()
            self.bind_ms = (time.perf_counter() - bind_start) * 1000.0
            self.bound = True
        else:
            self.load_result = self.pool.load_model(self.vmm_path)
            self.bind_ms = 0.0
        torch.cuda.synchronize(self.device)
        self.prepare_ms = (time.perf_counter() - start) * 1000.0
        return self.prepare_ms

    def mark_prefill_complete(self) -> None:
        pass

    def loading_metrics(self) -> dict:
        result = self.load_result
        return {
            "mode": "vmm",
            "vmm_policy": self.pool.vmm_policy,
            "hf_path": self.hf_path,
            "vmm_adapter_path": self.vmm_path,
            "cached_bytes": int(result.cached_bytes),
            "to_load_bytes": int(result.to_load_bytes),
            "total_model_bytes": int(result.total_model_bytes),
            "estimated_load_ms": float(result.estimated_load_ms),
            "vmm_wall_load_ms": float(result.wall_load_ms),
            "full_model_hit": bool(result.full_model_hit),
            "bind_ms": self.bind_ms,
            "sum_h2d_ms": float(result.wall_load_ms),
        }

    def close(self) -> float:
        # Stable VMM virtual addresses remain bound to this model object.
        # Loading the next model remaps/evicts physical pages inside the pool.
        return 0.0


class CpuModelCache:
    def __init__(self, paths: Dict[int, str], dtype: torch.dtype,
                 trust_remote_code: bool, empty_models: bool = False):
        self.paths = paths
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.empty_models = empty_models
        self.models = {}
        self.weight_stores = {}

    def get(self, model_id: int):
        if model_id not in self.paths:
            raise KeyError(f"No model path configured for model id {model_id}")
        if model_id not in self.models:
            path = self.paths[model_id]
            if self.empty_models:
                config = AutoConfig.from_pretrained(
                    path, trust_remote_code=self.trust_remote_code
                )
                config.torch_dtype = self.dtype
                with init_empty_weights():
                    try:
                        model = AutoModelForCausalLM.from_config(
                            config,
                            trust_remote_code=self.trust_remote_code,
                        )
                    except ValueError as causal_error:
                        try:
                            model = AutoModelForVision2Seq.from_config(
                                config,
                                trust_remote_code=self.trust_remote_code,
                            )
                        except ValueError:
                            raise causal_error
                self.models[model_id] = model.eval()
                return self.models[model_id]
            shards = list(Path(path).glob("*.safetensors"))
            if not shards:
                raise ValueError(
                    f"LayerPipe requires Hugging Face safetensors at {path}. "
                    "The rank_0 tensor.data_* checkpoint is a packed vLLM "
                    "format; provide --model-path MODEL_ID=/hf/checkpoint."
                )
            load_options = {
                "torch_dtype": self.dtype,
                "device_map": None,
                "low_cpu_mem_usage": True,
                "trust_remote_code": self.trust_remote_code,
            }
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path, **load_options
                )
            except ValueError as causal_error:
                try:
                    model = AutoModelForVision2Seq.from_pretrained(
                        path, **load_options
                    )
                except ValueError:
                    raise causal_error
            model = model.eval()
            self.weight_stores[model_id] = CpuWeightStore(model)
            self.models[model_id] = model
        return self.models[model_id]

    def get_weight_store(self, model_id: int) -> CpuWeightStore:
        self.get(model_id)
        return self.weight_stores[model_id]

    def preload(self, model_ids: Iterable[int]) -> None:
        for model_id in sorted(set(model_ids)):
            self.get(model_id)


def build_inputs(batch: Sequence[TraceRequest], vocab_size: int,
                 pad_token_id: int, device: torch.device,
                 seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_length = max(item.input_tokens for item in batch)
    input_ids = torch.full(
        (len(batch), max_length), pad_token_id,
        dtype=torch.long, device=device,
    )
    attention_mask = torch.zeros(
        (len(batch), max_length), dtype=torch.long, device=device
    )
    for row, request in enumerate(batch):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + request.request_id)
        tokens = torch.randint(
            low=3,
            high=max(4, vocab_size),
            size=(request.input_tokens,),
            generator=generator,
            dtype=torch.long,
        ).to(device)
        input_ids[row, -request.input_tokens:] = tokens
        attention_mask[row, -request.input_tokens:] = 1
    return input_ids, attention_mask


@torch.inference_mode()
def execute_batch(model, pipeline: LayerPipeline,
                  batch: Sequence[TraceRequest], device: torch.device,
                  seed: int, dispatch_s: float, replay_start: float,
                  batch_id: int) -> Tuple[List[RequestResult], dict]:
    config = model.config
    pad_token_id = config.pad_token_id
    if pad_token_id is None:
        pad_token_id = config.eos_token_id
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0]
    if pad_token_id is None:
        pad_token_id = 0
    inputs, attention_mask = build_inputs(
        batch, config.vocab_size, int(pad_token_id), device, seed
    )

    prepare_ms = pipeline.prepare()
    prefill_start = time.perf_counter()
    output = model(
        input_ids=inputs,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    next_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    first_token_s = time.perf_counter() - replay_start
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
    pipeline.mark_prefill_complete()

    finish_times: List[Optional[float]] = [
        first_token_s if item.output_tokens == 1 else None for item in batch
    ]
    past_key_values = output.past_key_values
    max_output = max(item.output_tokens for item in batch)
    decode_start = time.perf_counter()
    for token_index in range(1, max_output):
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (len(batch), 1), dtype=attention_mask.dtype, device=device
                ),
            ),
            dim=1,
        )
        output = model(
            input_ids=next_tokens,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        next_tokens = output.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        completing = [
            index for index, request in enumerate(batch)
            if request.output_tokens == token_index + 1
        ]
        if completing:
            torch.cuda.synchronize(device)
            completed_s = time.perf_counter() - replay_start
            for index in completing:
                finish_times[index] = completed_s
    torch.cuda.synchronize(device)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    results = []
    for request, finish_s in zip(batch, finish_times):
        results.append(
            RequestResult(
                request_id=request.request_id,
                model_id=request.model_id,
                arrival_s=request.arrival_s,
                dispatch_s=dispatch_s,
                first_token_s=first_token_s,
                finish_s=finish_s if finish_s is not None else first_token_s,
                input_tokens=request.input_tokens,
                output_tokens=request.output_tokens,
                batch_id=batch_id,
                batch_size=len(batch),
            )
        )
    metrics = {
        "batch_id": batch_id,
        "model_id": batch[0].model_id,
        "batch_size": len(batch),
        "input_tokens": sum(item.input_tokens for item in batch),
        "output_tokens": sum(item.output_tokens for item in batch),
        "synchronous_load_ms": prepare_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "loading": pipeline.loading_metrics(),
    }
    del output, past_key_values, inputs, attention_mask, next_tokens
    return results, metrics


def pop_same_model(queue: Deque[TraceRequest],
                   max_batch_size: int) -> List[TraceRequest]:
    head = queue.popleft()
    batch = [head]
    retained = deque()
    while queue:
        request = queue.popleft()
        if (request.model_id == head.model_id
                and (max_batch_size <= 0 or len(batch) < max_batch_size)):
            batch.append(request)
        else:
            retained.append(request)
    queue.extend(retained)
    return batch


def run(args) -> dict:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    requests = parse_trace(args.trace, args.max_requests)
    if args.output_tokens_override > 0:
        for request in requests:
            request.output_tokens = args.output_tokens_override
    paths = load_model_paths(args.config, args.model_path)
    vmm_paths = {}
    vmm_adapters = None
    cache = CpuModelCache(
        paths, dtype, args.trust_remote_code,
        empty_models=args.load_mode == "vmm",
    )
    vmm_pool = None
    if args.load_mode == "vmm":
        if args.vmm_merge_tensor_groups is not None:
            raise ValueError(
                "HF safetensors VMM mode does not support merging tensor "
                "groups across shard/header gaps"
            )
        vmm_adapters = SafetensorsVmmAdapters()
        for model_id in sorted({item.model_id for item in requests}):
            vmm_paths[model_id] = vmm_adapters.create(
                model_id, paths[model_id]
            )
        TangramVmmPool = _load_vmm_pool_class()
        vmm_pool = TangramVmmPool(
            args.vmm_pool_gib,
            args.device,
            page_size_mib=args.vmm_page_size_mib,
        )
        for model_id in sorted({item.model_id for item in requests}):
            vmm_pool.register_model(
                vmm_paths[model_id],
                model_id,
                tensor_only=False,
            )
        print(
            f"VMM_POLICY={vmm_pool.vmm_policy} device={args.device} "
            f"pool_gib={args.vmm_pool_gib} "
            f"page_size_mib={args.vmm_page_size_mib}",
            flush=True,
        )
    if args.preload_cpu_models:
        cache.preload(item.model_id for item in requests)

    pending: Deque[TraceRequest] = deque()
    next_index = 0
    results: List[RequestResult] = []
    batches = []
    replay_start = time.perf_counter()
    batch_id = 0
    active_model_id: Optional[int] = None
    active_model = None
    active_loader = None

    while next_index < len(requests) or pending:
        now_s = time.perf_counter() - replay_start
        while (next_index < len(requests)
               and requests[next_index].arrival_s * args.trace_time_scale
               <= now_s):
            request = requests[next_index]
            request.arrival_s *= args.trace_time_scale
            pending.append(request)
            next_index += 1
        if not pending:
            next_arrival = (
                requests[next_index].arrival_s * args.trace_time_scale
            )
            time.sleep(max(0.0, next_arrival - now_s))
            continue

        batch = pop_same_model(pending, args.max_batch_size)
        dispatch_s = time.perf_counter() - replay_start
        clear_ms = 0.0
        switched_model = active_model_id != batch[0].model_id
        if switched_model:
            log_event(
                "MODEL_SWITCH",
                elapsed_s=dispatch_s,
                gpu=args.device,
                load_mode=args.load_mode,
                from_model_id=active_model_id,
                to_model_id=batch[0].model_id,
                to_model_path=paths[batch[0].model_id],
                waiting_requests=len(pending) + len(batch),
            )
            if active_loader is not None:
                clear_ms = active_loader.close()
            active_model_id = batch[0].model_id
            active_model = cache.get(active_model_id)
            if args.load_mode == "layerpipe":
                active_weights = cache.get_weight_store(active_model_id)
                active_loader = LayerPipeline(
                    active_model, device, active_weights
                )
            elif args.load_mode == "native":
                active_weights = cache.get_weight_store(active_model_id)
                active_loader = NativeModelLoader(
                    active_model, device, active_weights
                )
            else:
                active_loader = VmmModelLoader(
                    active_model,
                    device,
                    vmm_pool,
                    vmm_paths[active_model_id],
                    paths[active_model_id],
                )
        model = active_model
        loader = active_loader
        log_event(
            "DISPATCH",
            elapsed_s=dispatch_s,
            gpu=args.device,
            load_mode=args.load_mode,
            batch_id=batch_id,
            model_id=batch[0].model_id,
            concurrent_requests=len(batch),
            waiting_requests_after_dispatch=len(pending),
            request_ids=[item.request_id for item in batch],
            input_lengths=[item.input_tokens for item in batch],
            output_lengths=[item.output_tokens for item in batch],
            total_input_tokens=sum(item.input_tokens for item in batch),
            total_output_tokens=sum(item.output_tokens for item in batch),
        )
        batch_results, batch_metrics = execute_batch(
            model, loader, batch, device, args.seed, dispatch_s,
            replay_start, batch_id,
        )
        batch_metrics["previous_model_clear_ms"] = clear_ms
        batch_metrics["cold_model_load"] = switched_model
        log_event(
            "BATCH_COMPLETE",
            elapsed_s=time.perf_counter() - replay_start,
            gpu=args.device,
            load_mode=args.load_mode,
            batch_id=batch_id,
            model_id=batch[0].model_id,
            concurrent_requests=len(batch),
            waiting_requests=len(pending),
            synchronous_load_ms=batch_metrics["synchronous_load_ms"],
            prefill_ms=batch_metrics["prefill_ms"],
            decode_ms=batch_metrics["decode_ms"],
            vmm_load=(
                batch_metrics["loading"]
                if args.load_mode == "vmm" else None
            ),
            ttft_ms=[item.ttft_ms for item in batch_results],
            e2e_ms=[item.e2e_ms for item in batch_results],
        )
        results.extend(batch_results)
        batches.append(batch_metrics)
        batch_id += 1

    final_clear_ms = (
        active_loader.close() if active_loader is not None else 0.0
    )
    records = [item.record() for item in sorted(
        results, key=lambda item: item.request_id
    )]
    summary = {
        "requests": len(records),
        "batches": len(batches),
        "ttft_ms": summarize([item["ttft_ms"] for item in records]),
        "service_ttft_ms": summarize(
            [item["service_ttft_ms"] for item in records]
        ),
        "queue_ms": summarize([item["queue_ms"] for item in records]),
        "e2e_ms": summarize([item["e2e_ms"] for item in records]),
    }
    output = {
        "backend": args.load_mode,
        "load_mode": args.load_mode,
        "trace": str(args.trace.resolve()),
        "config": str(args.config.resolve()),
        "trace_time_scale": args.trace_time_scale,
        "output_tokens_override": args.output_tokens_override,
        "summary": summary,
        "requests": records,
        "batch_metrics": batches,
        "final_gpu_clear_ms": final_clear_ms,
    }
    if vmm_pool is not None:
        output["vmm"] = {
            "policy": vmm_pool.vmm_policy,
            "pool_gib": args.vmm_pool_gib,
            "page_size_mib": args.vmm_page_size_mib,
            "source_format": "huggingface_safetensors",
            "allocation_units": "tensor",
        }
        vmm_pool.close()
        vmm_adapters.close()
    return output


def write_outputs(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(result, stream, indent=2)
    csv_path = output.with_suffix(".requests.csv")
    records = result["requests"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print("LAYERPIPE_SUMMARY=" + json.dumps(result["summary"], sort_keys=True))
    print(f"LAYERPIPE_RESULT={output}")
    print(f"LAYERPIPE_REQUESTS={csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/servegen_8_models.json"),
    )
    parser.add_argument(
        "--trace", type=Path,
        default=Path("evaluation/traces/servegen_tangram.trace"),
    )
    parser.add_argument(
        "--model-path", action="append", default=[], metavar="MODEL_ID=PATH",
        help="Override a config model with a Hugging Face checkpoint.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--load-mode", choices=("layerpipe", "native", "vmm"),
        default="layerpipe",
        help=("layerpipe overlaps layer i compute with layer i+1 loading; "
              "native synchronously loads the whole model before prefill; "
              "vmm synchronously maps/loads weights with cross-model reuse."),
    )
    parser.add_argument("--vmm-pool-gib", type=float, default=40.0)
    parser.add_argument("--vmm-page-size-mib", type=int, default=0)
    parser.add_argument("--vmm-tensor-only", action="store_true")
    parser.add_argument("--vmm-merge-tensor-groups", type=int)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument(
        "--output-tokens-override", type=int, default=0,
        help=("Positive values replace every trace output length; zero "
              "preserves each request's trace output_tokens."),
    )
    parser.add_argument("--trace-time-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--preload-cpu-models", action=argparse.BooleanOptionalAction,
        default=True,
        help="Load all referenced checkpoints into CPU memory before replay.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("layerpipe_result.json")
    )
    args = parser.parse_args()
    if args.trace_time_scale < 0:
        parser.error("--trace-time-scale must be non-negative")
    if args.max_batch_size < 0:
        parser.error("--max-batch-size must be non-negative")
    if args.output_tokens_override < 0:
        parser.error("--output-tokens-override must be non-negative")
    if args.vmm_pool_gib <= 0:
        parser.error("--vmm-pool-gib must be positive")
    if args.vmm_page_size_mib < 0:
        parser.error("--vmm-page-size-mib must be non-negative")
    if (args.vmm_page_size_mib != 0
            and args.vmm_page_size_mib
            & (args.vmm_page_size_mib - 1)):
        parser.error("--vmm-page-size-mib must be zero or a power of two")
    if (args.vmm_merge_tensor_groups is not None
            and args.vmm_merge_tensor_groups <= 0):
        parser.error("--vmm-merge-tensor-groups must be positive")
    if args.vmm_tensor_only and args.vmm_merge_tensor_groups is not None:
        parser.error(
            "--vmm-tensor-only and --vmm-merge-tensor-groups "
            "are mutually exclusive"
        )
    result = run(args)
    write_outputs(result, args.output)


if __name__ == "__main__":
    main()
