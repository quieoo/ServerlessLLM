#!/usr/bin/env python3
"""Replay model switches in one VMM pool and compare target inference phases."""

import argparse
import gc
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def load_models(config_path):
    with open(config_path) as stream:
        config = json.load(stream)
    models = {}
    for item in config["model_lists"]:
        rank_path = str(Path(item["path"]).resolve())
        models[int(item["id"])] = {
            "rank_path": rank_path,
            "model_path": str(Path(rank_path).parent),
        }
    return models


def load_trace(trace_path, requests):
    result = []
    with open(trace_path) as stream:
        for line in stream:
            fields = line.split()
            if len(fields) >= 2:
                result.append(int(fields[1]))
            if len(result) == requests:
                break
    if len(result) != requests:
        raise ValueError(
            f"Trace has only {len(result)} requests, expected {requests}"
        )
    return result


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def make_random_prompt(characters, seed):
    """Make a reproducible, approximately-sized natural-language request."""
    if characters <= 0:
        return None
    words = (
        "memory model request scheduler tensor layer attention context cache "
        "system service virtual physical address mapping performance workload "
        "inference analysis experiment result latency throughput compute page "
        "allocation model parameter execution benchmark sequence observation "
        "application response design implementation measurement evaluation"
    ).split()
    rng = random.Random(seed)
    prefix = (
        "Analyze the following synthetic system notes and explain the main "
        "performance tradeoffs in a concise response. Notes: "
    )
    parts = [prefix]
    length = len(prefix)
    while length < characters:
        word = rng.choice(words)
        parts.append(word)
        parts.append(" ")
        length += len(word) + 1
    return "".join(parts)[:characters]


def measure_target(args, pool, target, phase, backend, model_path=None):
    if backend == "vmm_reuse":
        load_result = pool.load_model(target["rank_path"])
        layout = pool.model_layout(target["rank_path"])
        vmm_load = {
            "cached_bytes": load_result.cached_bytes,
            "to_load_bytes": load_result.to_load_bytes,
            "wall_load_ms": load_result.wall_load_ms,
            "full_model_hit": bool(load_result.full_model_hit),
        }
    else:
        layout = None
        vmm_load = None
    start = time.perf_counter()
    llm = LLM(
        model=model_path or target["model_path"],
        load_format=("safetensors" if backend == "exclusive"
                     else "serverless_llm"),
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        num_gpu_blocks_override=args.num_gpu_blocks,
    )
    engine_init_ms = (time.perf_counter() - start) * 1000.0
    sampling = SamplingParams(
        temperature=0, max_tokens=args.output_tokens, ignore_eos=True
    )
    try:
        prompt_tokens = len(llm.get_tokenizer().encode(args.prompt))
    except Exception:  # Metrics remain useful if an older vLLM lacks this API.
        prompt_tokens = None
    for _ in range(args.warmup):
        llm.generate([args.prompt], sampling, use_tqdm=False)

    ttft_ms = []
    scheduled_prefill_ms = []
    decode_total_ms = []
    decode_per_token_ms = []
    e2e_ms = []
    outputs = []
    for _ in range(args.measurements):
        result = llm.generate([args.prompt], sampling, use_tqdm=False)[0]
        metrics = result.metrics
        ttft_ms.append(
            (metrics.first_token_time - metrics.arrival_time) * 1000.0
        )
        scheduled_start = (metrics.first_scheduled_time
                           if metrics.first_scheduled_time is not None
                           else metrics.arrival_time)
        scheduled_prefill_ms.append(
            (metrics.first_token_time - scheduled_start) * 1000.0
        )
        decode_ms = (
            metrics.finished_time - metrics.first_token_time
        ) * 1000.0
        output_count = len(result.outputs[0].token_ids)
        decode_tokens = max(1, output_count - 1)
        decode_total_ms.append(decode_ms)
        decode_per_token_ms.append(decode_ms / decode_tokens)
        e2e_ms.append(
            (metrics.finished_time - metrics.arrival_time) * 1000.0
        )
        outputs.append(result.outputs[0].text)

    record = {
        "phase": phase,
        "backend": backend,
        "layout": layout,
        "vmm_load": vmm_load,
        "engine_init_ms": engine_init_ms,
        "prompt_characters": len(args.prompt),
        "prompt_tokens": prompt_tokens,
        "ttft_ms": summarize(ttft_ms),
        "scheduled_prefill_ms": summarize(scheduled_prefill_ms),
        "decode_total_ms": summarize(decode_total_ms),
        "decode_per_token_ms": summarize(decode_per_token_ms),
        "e2e_ms": summarize(e2e_ms),
        "output": outputs[-1],
    }
    print("TANGRAM_SWITCH_PHASE=" + json.dumps(record, sort_keys=True))
    torch.cuda.synchronize()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/servegen_8_models.json")
    parser.add_argument(
        "--trace", default="evaluation/traces/servegen_tangram.trace")
    parser.add_argument("--switch-requests", type=int, default=1000)
    parser.add_argument("--target-model-id", type=int, default=0)
    parser.add_argument("--pool-gib", type=float, default=40.0)
    parser.add_argument(
        "--vmm-page-size-mib", type=int, default=0,
        help=("VMM physical allocation page size in MiB. Zero uses the CUDA "
              "device native minimum granularity."),
    )
    parser.add_argument(
        "--merge-tensor-groups", nargs="?", type=int, const=40, default=None,
        metavar="TARGET_COUNT",
        help=("Merge adjacent stored tensor groups toward TARGET_COUNT before "
              "rounding each merged group to VMM pages. With no value, use "
              "40. By default groups are not merged."),
    )
    parser.add_argument(
        "-tensor-only", "--tensor-only", action="store_true",
        help=("Split stored tensor groups into tensor-sized file/copy units. "
              "Units are packed into the model's stable VA arena and may "
              "share a physical VMM page."),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measurements", type=int, default=10)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default=("Virtual memory maps virtual addresses to physical memory "
                 "pages. Explain why this abstraction is useful."),
    )
    parser.add_argument(
        "--random-prompt-chars", type=int, default=0,
        help=("Replace --prompt with a deterministic pseudo-random natural "
              "language string of approximately this many characters."),
    )
    parser.add_argument("--random-prompt-seed", type=int, default=20260721)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--num-gpu-blocks", type=int, default=64)
    parser.add_argument(
        "--kv-backend", choices=("default", "vmm"), default="default",
        help=("Use ordinary vLLM KV tensors or lazily map segmented KV "
              "blocks from the same in-process VMM physical-page pool."),
    )
    parser.add_argument(
        "--exclusive-model", default="/mnt/n0/models/qwen2-14",
        help=("Native Hugging Face model used by --exclusive-run. It must "
              "contain the safetensors shards for --target-model-id."),
    )
    parser.add_argument(
        "--exclusive-run", action="store_true",
        help=("Directly load --exclusive-model safetensors into ordinary CUDA "
              "allocations and measure it; no VMM pool or reuse."),
    )
    parser.add_argument("--output", default="vllm_vmm_switch_result.json")
    args = parser.parse_args()
    if args.vmm_page_size_mib < 0:
        parser.error("--vmm-page-size-mib must be zero or positive")
    if (args.vmm_page_size_mib != 0 and
            args.vmm_page_size_mib & (args.vmm_page_size_mib - 1)):
        parser.error("--vmm-page-size-mib must be zero or a power of two")
    if (args.merge_tensor_groups is not None and
            args.merge_tensor_groups <= 0):
        parser.error("--merge-tensor-groups TARGET_COUNT must be positive")
    if args.tensor_only and args.merge_tensor_groups is not None:
        parser.error(
            "-tensor-only/--tensor-only cannot be combined with "
            "--merge-tensor-groups")
    if args.tensor_only and args.exclusive_run:
        parser.error(
            "-tensor-only/--tensor-only only applies to the VMM backend")

    random_prompt = make_random_prompt(args.random_prompt_chars,
                                       args.random_prompt_seed)
    if random_prompt is not None:
        args.prompt = random_prompt

    torch.cuda.set_device(args.device)
    models = load_models(args.config)
    if args.target_model_id not in models:
        raise ValueError(f"Unknown target model id {args.target_model_id}")

    target = models[args.target_model_id]
    if args.exclusive_run:
        if args.kv_backend != "default":
            raise ValueError("--exclusive-run only supports --kv-backend default")
        native_model = str(Path(args.exclusive_model).resolve())
        if not list(Path(native_model).glob("*.safetensors")):
            raise ValueError(
                f"--exclusive-model has no safetensors shards: {native_model}")
        os.environ["TANGRAM_VMM_IN_PROCESS"] = "0"
        exclusive = measure_target(args, None, target, "exclusive", "exclusive",
                                   model_path=native_model)
        result = {
            "backend": "exclusive",
            "target_model_id": args.target_model_id,
            "target_model": native_model,
            "vmm_equivalent_model": target["model_path"],
            "warmup": args.warmup,
            "measurements": args.measurements,
            "output_tokens": args.output_tokens,
            "num_gpu_blocks": args.num_gpu_blocks,
            "kv_backend": "default",
            "prompt": args.prompt,
            "random_prompt_characters": args.random_prompt_chars,
            "random_prompt_seed": args.random_prompt_seed,
            "exclusive": exclusive,
        }
        with open(args.output, "w") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print("TANGRAM_EXCLUSIVE_RESULT=" + json.dumps(result, sort_keys=True))
        return

    trace = load_trace(args.trace, args.switch_requests)

    os.environ["TANGRAM_KV_BACKEND"] = args.kv_backend
    if args.kv_backend == "vmm":
        os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

    from vllm.model_executor.model_loader.tangram_vmm import (
        TangramVmmPool,
        set_shared_vmm_pool,
    )
    pool = TangramVmmPool(
        args.pool_gib, args.device, page_size_mib=args.vmm_page_size_mib)
    print(
        f"VMM_POLICY={pool.vmm_policy} "
        f"device={args.device} pool_gib={args.pool_gib} "
        f"page_size_mib={args.vmm_page_size_mib}",
        flush=True,
    )
    set_shared_vmm_pool(pool)
    required_model_ids = sorted(set(trace) | {args.target_model_id})
    for model_id in required_model_ids:
        model = models[model_id]
        start = time.perf_counter()
        pool.register_model(
            model["rank_path"], model_id,
            merge_tensor_groups=args.merge_tensor_groups,
            tensor_only=args.tensor_only)
        print(
            f"REGISTER model_id={model_id} path={model['rank_path']} "
            f"tensor_only={args.tensor_only} "
            f"seconds={time.perf_counter() - start:.3f}", flush=True
        )

    initial = measure_target(args, pool, target, "initial", "vmm_reuse")

    transitions = 0
    previous = None
    replay_start = time.perf_counter()
    for index, model_id in enumerate(trace, 1):
        pool.load_model(models[model_id]["rank_path"])
        if previous is not None and previous != model_id:
            transitions += 1
        previous = model_id
        if index % 25 == 0 or index == args.switch_requests:
            print(
                f"REPLAY requests={index}/{args.switch_requests} "
                f"transitions={transitions}", flush=True
            )
    replay_seconds = time.perf_counter() - replay_start
    after = measure_target(args, pool, target, "after_switches", "vmm_reuse")

    result = {
        "backend": "vmm_reuse",
        "vmm_policy": pool.vmm_policy,
        "target_model_id": args.target_model_id,
        "target_model": target["model_path"],
        "switch_requests": args.switch_requests,
        "actual_transitions": transitions,
        "replay_seconds": replay_seconds,
        "pool_gib": args.pool_gib,
        "vmm_page_size_mib": args.vmm_page_size_mib,
        "merge_tensor_groups_target_count": args.merge_tensor_groups,
        "tensor_only": args.tensor_only,
        "warmup": args.warmup,
        "measurements": args.measurements,
        "output_tokens": args.output_tokens,
        "num_gpu_blocks": args.num_gpu_blocks,
        "kv_backend": args.kv_backend,
        "prompt": args.prompt,
        "random_prompt_characters": args.random_prompt_chars,
        "random_prompt_seed": args.random_prompt_seed,
        "initial": initial,
        "after_switches": after,
    }
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print("TANGRAM_SWITCH_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
