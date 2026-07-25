#!/usr/bin/env python3
"""Compare native Hugging Face CUDA weights with Tangram VMM-backed weights.

Run each backend in a separate process so the VMM pool and the exclusive model
never compete for GPU memory.  The Tangram model directory must have been saved
with save_tensor_group_dict and contain tensor_group_index.txt and
tensor_meta_index.json.
"""

import argparse
import ctypes
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class _LoadResult(ctypes.Structure):
    _fields_ = [
        ("cached_bytes", ctypes.c_uint64),
        ("to_load_bytes", ctypes.c_uint64),
        ("total_model_bytes", ctypes.c_uint64),
        ("estimated_load_ms", ctypes.c_double),
        ("wall_load_ms", ctypes.c_double),
        ("full_model_hit", ctypes.c_int),
    ]


class _TensorBinding(ctypes.Structure):
    _fields_ = [
        ("group_base", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


class VmmSession:
    def __init__(self, library, model_path, pool_bytes, device):
        self.lib = ctypes.CDLL(str(library))
        self.device = device
        self.handle = None
        self._declare_abi()
        self.handle = self.lib.tangram_vram_create_ex(
            1, pool_bytes, 0.0, 0.0, 0, 20.0, 0.0, 1, 4, 0, b"vmm"
        )
        if not self.handle or self.lib.tangram_vram_last_error(self.handle):
            error = self.last_error()
            if error:
                self.close()
                raise RuntimeError(error)
        rc = self.lib.tangram_vram_register_model(
            self.handle, 0, os.fsencode(model_path), 1.0, -1
        )
        if rc != 0:
            raise RuntimeError(self.last_error())

    def _declare_abi(self):
        lib = self.lib
        lib.tangram_vram_create_ex.restype = ctypes.c_void_p
        lib.tangram_vram_create_ex.argtypes = [
            ctypes.c_int, ctypes.c_uint64, ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
        ]
        lib.tangram_vram_destroy.argtypes = [ctypes.c_void_p]
        lib.tangram_vram_last_error.argtypes = [ctypes.c_void_p]
        lib.tangram_vram_last_error.restype = ctypes.c_char_p
        lib.tangram_vram_register_model.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_double,
            ctypes.c_int,
        ]
        lib.tangram_vram_load_model.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(_LoadResult),
        ]
        lib.tangram_vram_tensor_count.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ]
        lib.tangram_vram_tensor_count.restype = ctypes.c_int
        lib.tangram_vram_get_tensor.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(_TensorBinding),
        ]

    def last_error(self):
        if not self.handle:
            return ""
        value = self.lib.tangram_vram_last_error(self.handle)
        return value.decode() if value else ""

    def load(self):
        result = _LoadResult()
        if self.lib.tangram_vram_load_model(
            self.handle, 0, self.device, ctypes.byref(result)
        ) != 0:
            raise RuntimeError(self.last_error())
        return result

    def bindings(self):
        count = self.lib.tangram_vram_tensor_count(
            self.handle, 0, self.device
        )
        if count < 0:
            raise RuntimeError(self.last_error())
        result = {}
        for index in range(count):
            name = ctypes.create_string_buffer(4096)
            binding = _TensorBinding()
            rc = self.lib.tangram_vram_get_tensor(
                self.handle, 0, self.device, index, name, len(name),
                ctypes.byref(binding),
            )
            if rc != 0:
                raise RuntimeError(self.last_error())
            result[name.value.decode()] = binding
        return result

    def close(self):
        if self.handle:
            self.lib.tangram_vram_destroy(self.handle)
            self.handle = None


def _dtype_from_config(config, override):
    if override:
        return getattr(torch, override)
    dtype = getattr(config, "torch_dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else torch.float16


def load_hf_exclusive(args):
    dtype = _dtype_from_config(AutoConfig.from_pretrained(
        args.hf_model, trust_remote_code=args.trust_remote_code
    ), args.dtype)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(f"cuda:{args.device}")
    torch.cuda.synchronize(args.device)
    return model.eval(), None, (time.perf_counter() - start) * 1000.0, {}


def load_vmm_reuse(args):
    try:
        from sllm_store._C import restore_tensors
    except ImportError as exc:
        raise RuntimeError(
            "sllm_store._C is required; build/install Tangram/sllm_store first"
        ) from exc

    session = VmmSession(
        args.vmm_library, args.store_model,
        int(args.pool_gib * 1024 ** 3), args.device,
    )
    load_result = session.load()
    bindings = session.bindings()
    with open(Path(args.store_model) / "tensor_meta_index.json") as stream:
        tensor_meta = json.load(stream)
    missing = sorted(set(tensor_meta) - set(bindings))
    if missing:
        session.close()
        raise RuntimeError(f"VMM bindings miss {len(missing)} tensors: {missing[:5]}")

    names = list(tensor_meta)
    meta = {name: tuple(tensor_meta[name]) for name in names}
    bases = [bindings[name].group_base for name in names]
    offsets = [bindings[name].offset for name in names]
    state_dict = restore_tensors(meta, bases, offsets)

    config = AutoConfig.from_pretrained(
        args.store_model, trust_remote_code=args.trust_remote_code
    )
    config.torch_dtype = _dtype_from_config(config, args.dtype)
    start = time.perf_counter()
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=args.trust_remote_code
        )
    with torch.no_grad():
        for name, value in state_dict.items():
            set_module_tensor_to_device(model, name, value.device, value=value)
        for name, value in list(model.named_buffers()):
            if value.device.type != "cuda":
                set_module_tensor_to_device(
                    model, name, f"cuda:{args.device}", value=value
                )
    model.tie_weights()
    model.eval()
    torch.cuda.synchronize(args.device)
    bind_ms = (time.perf_counter() - start) * 1000.0
    load_stats = {
        "cached_bytes": load_result.cached_bytes,
        "to_load_bytes": load_result.to_load_bytes,
        "vmm_load_ms": load_result.wall_load_ms,
        "bind_ms": bind_ms,
    }
    return model, session, load_result.wall_load_ms + bind_ms, load_stats


def benchmark(model, tokenizer, args):
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {key: value.to(f"cuda:{args.device}") for key, value in encoded.items()}

    @torch.inference_mode()
    def operation():
        if args.mode == "forward":
            return model(**encoded, use_cache=False).logits
        return model.generate(
            **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
            use_cache=True, pad_token_id=tokenizer.eos_token_id,
        )

    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize(args.device)
    samples = []
    output = None
    for _ in range(args.iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = operation()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return {
        "latency_ms_mean": statistics.mean(samples),
        "latency_ms_median": statistics.median(samples),
        "latency_ms_min": min(samples),
        "latency_ms_samples": samples,
        "output_checksum": float(output.float().sum().item()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hf_exclusive", "vmm_reuse"), required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--store-model")
    parser.add_argument("--vmm-library", default="build/libtangram_vram_backend.so")
    parser.add_argument("--pool-gib", type=float, default=24.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--mode", choices=("forward", "generate"), default="forward")
    parser.add_argument("--prompt", default="Explain virtual memory in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.backend == "vmm_reuse" and not args.store_model:
        parser.error("--store-model is required for vmm_reuse")
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model, trust_remote_code=args.trust_remote_code
    )
    if args.backend == "hf_exclusive":
        model, session, load_ms, load_stats = load_hf_exclusive(args)
    else:
        model, session, load_ms, load_stats = load_vmm_reuse(args)
    try:
        result = benchmark(model, tokenizer, args)
        result.update({
            "backend": args.backend,
            "mode": args.mode,
            "load_ms": load_ms,
            "model": args.hf_model,
            **load_stats,
        })
        print("TANGRAM_BENCH_RESULT=" + json.dumps(result, sort_keys=True))
    finally:
        torch.cuda.synchronize(args.device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if session:
            session.close()


if __name__ == "__main__":
    main()
