#!/usr/bin/env python3
"""Fast VMM KV ABI + segmented write/decode kernel smoke test."""

import argparse
import ctypes
import json

import torch
from vllm import _custom_ops as ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library",
        default=("/mnt/n0/Tangram/Tangram/tools/mock_allocation/build/"
                 "libtangram_vram_backend.so"),
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    lib = ctypes.CDLL(args.library)
    lib.tangram_vram_create_ex.restype = ctypes.c_void_p
    lib.tangram_vram_create_ex.argtypes = [
        ctypes.c_int, ctypes.c_uint64, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
    ]
    lib.tangram_vram_destroy.argtypes = [ctypes.c_void_p]
    lib.tangram_vram_last_error.argtypes = [ctypes.c_void_p]
    lib.tangram_vram_last_error.restype = ctypes.c_char_p
    lib.tangram_vram_kv_base.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.tangram_vram_kv_base.restype = ctypes.c_uint64
    lib.tangram_vram_kv_allocate.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.tangram_vram_kv_release.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
    ]

    torch.cuda.set_device(args.device)
    handle = lib.tangram_vram_create_ex(
        1, 8 * 1024**2, 0.0, 0.0, 0, 20.0, 0.0, 1, 4, 0,
        b"vmm",
    )
    if not handle:
        raise RuntimeError("Cannot create VMM smoke pool")

    logical_ids = (ctypes.c_uint64 * 1)(0)
    offsets = (ctypes.c_uint64 * 1)()
    block_tokens = 32
    num_layers = 1
    num_kv_heads = 1
    head_size = 128
    dtype_bytes = 2
    physical_block_bytes = (
        2 * num_layers * block_tokens * num_kv_heads * head_size *
        dtype_bytes
    )
    try:
        base = int(lib.tangram_vram_kv_base(handle, args.device))
        status = lib.tangram_vram_kv_allocate(
            handle, -1, args.device, physical_block_bytes,
            logical_ids, 1, offsets,
        )
        if status != 0:
            error = lib.tangram_vram_last_error(handle)
            raise RuntimeError(error.decode() if error else "KV allocation failed")

        torch.manual_seed(20260721)
        key = torch.randn(
            1, num_kv_heads, head_size, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        query = torch.randn_like(key)
        write_table = torch.tensor(
            [offsets[0]], device="cuda", dtype=torch.int64)
        slot_mapping = torch.tensor([0], device="cuda", dtype=torch.int64)
        ops.reshape_and_cache_segment(
            key, value, base, write_table, 0, slot_mapping, block_tokens,
            "auto", 1.0)

        output = torch.empty_like(query)
        decode_table = write_table.view(1, 1)
        seq_lens = torch.tensor([1], device="cuda", dtype=torch.int32)
        ops.segmented_attention_v1(
            output, query, base, num_kv_heads, head_size**-0.5,
            decode_table, 0, seq_lens, block_tokens, 1, None, "auto", 1.0)
        torch.cuda.synchronize()
        max_error = float((output - value).abs().max().item())
        result = {
            "base": base,
            "offset": int(offsets[0]),
            "physical_block_bytes": physical_block_bytes,
            "max_error": max_error,
            "passed": max_error <= 1e-3,
        }
        print("TANGRAM_ODKV_SMOKE=" + json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise RuntimeError(f"Segmented KV mismatch: {max_error}")
    finally:
        torch.cuda.synchronize()
        lib.tangram_vram_kv_release(
            handle, args.device, logical_ids, 1)
        lib.tangram_vram_destroy(handle)


if __name__ == "__main__":
    main()
