#!/usr/bin/env python3
"""Trace-driven multi-model VMM weight reuse with vLLM segmented ODKV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Dict, List


# CUDA visibility must be fixed before importing layerpipe_bench because that
# module imports torch at module scope. Changing CUDA_VISIBLE_DEVICES after a
# torch import can leave PyTorch's deferred CUDA initialization with a stale
# device count.
def _bootstrap_cuda_visibility(argv: List[str]) -> str:
    physical_device = "0"
    for index, value in enumerate(argv):
        if value == "--device" and index + 1 < len(argv):
            physical_device = argv[index + 1]
            break
        if value.startswith("--device="):
            physical_device = value.split("=", 1)[1]
            break
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to logical GPU 0.
    os.environ["USE_GPU"] = "0"
    return physical_device


_PHYSICAL_CUDA_DEVICE = _bootstrap_cuda_visibility(sys.argv[1:])


REPO_ROOT = Path(__file__).resolve().parents[2]
ELASTIC_KV = str(REPO_ROOT / "ElasticKV")
if ELASTIC_KV not in sys.path:
    sys.path.insert(0, ELASTIC_KV)
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from layerpipe_bench import (  # noqa: E402
    RequestResult,
    TraceRequest,
    apply_request_input_limits,
    log_event,
    load_model_input_limits,
    parse_trace,
    pop_same_model,
    summarize,
)


def load_models(config_path: Path,
                overrides: List[str]) -> Dict[int, dict]:
    with config_path.open() as stream:
        config = json.load(stream)
    models = {}
    for item in config["model_lists"]:
        model_id = int(item["id"])
        raw_rank = item.get("packed_rank_path", item["path"])
        rank_path = Path(raw_rank).resolve()
        if rank_path.name != "rank_0":
            rank_path = rank_path / "rank_0"
        if not list(rank_path.glob("tensor.data_*")):
            raise ValueError(
                f"Model {model_id} has no packed tensor.data_* files: "
                f"{rank_path}"
            )
        model_config = {}
        model_config_path = rank_path.parent / "config.json"
        if model_config_path.exists():
            with model_config_path.open() as stream:
                model_config = json.load(stream)
        tokenizer_mode = item.get("tokenizer_mode")
        if tokenizer_mode is None:
            # This converted LLaVA checkpoint carries an older tokenizer.json
            # that tokenizers 0.19 cannot deserialize, while tokenizer.model
            # remains valid. Limit the compatibility fallback to LLaVA.
            tokenizer_mode = (
                "slow" if model_config.get("model_type") == "llava"
                else "auto"
            )
        vision_options = {}
        if model_config.get("model_type") == "llava":
            vision_config = model_config.get("vision_config", {})
            image_size = int(vision_config.get("image_size", 336))
            patch_size = int(vision_config.get("patch_size", 14))
            feature_size = (image_size // patch_size) ** 2
            if model_config.get(
                    "vision_feature_select_strategy", "default") == "full":
                feature_size += 1
            vision_options = {
                "image_input_type": "pixel_values",
                "image_token_id": int(
                    model_config.get("image_token_index", 32000)
                ),
                "image_input_shape": f"1,3,{image_size},{image_size}",
                "image_feature_size": feature_size,
                # Trace replay supplies token IDs rather than raw images.
                # Keep the vision tower (and thus checkpoint key layout), but
                # avoid constructing an unused HF image processor.
                "disable_image_processor": True,
            }
        models[model_id] = {
            "rank_path": str(rank_path),
            "model_path": str(rank_path.parent),
            "tokenizer_mode": tokenizer_mode,
            "vision_options": vision_options,
        }
    for value in overrides:
        model_id_text, path_text = value.split("=", 1)
        rank_path = Path(path_text).resolve()
        if rank_path.name != "rank_0":
            rank_path = rank_path / "rank_0"
        models[int(model_id_text)] = {
            "rank_path": str(rank_path),
            "model_path": str(rank_path.parent),
            "tokenizer_mode": "auto",
            "vision_options": {},
        }
    return models


def prepare_requests(args, model_limits) -> List[TraceRequest]:
    requests = parse_trace(args.trace, args.max_requests)
    for request in requests:
        request.trace_input_tokens = request.input_tokens
        request.trace_output_tokens = request.output_tokens
        if args.output_tokens_override > 0:
            request.output_tokens = args.output_tokens_override
    apply_request_input_limits(
        requests, model_limits, args.max_model_len,
        args.truncate_input_to_model_limit,
    )
    return requests


def make_prompt_tokens(request: TraceRequest, vocab_size: int,
                       seed: int) -> List[int]:
    rng = random.Random(seed + request.request_id)
    upper = max(4, vocab_size)
    return [rng.randrange(3, upper) for _ in range(request.input_tokens)]


def get_worker(llm):
    executor = llm.llm_engine.model_executor
    worker = getattr(executor, "driver_worker", None)
    if worker is None:
        raise RuntimeError(
            "VMM ODKV trace benchmark requires the single-GPU GPUExecutor"
        )
    return worker


def get_layerweave_controller(worker):
    controller = getattr(worker.model_runner.model,
                         "_tangram_layerweave", None)
    if controller is None:
        raise RuntimeError(
            "vLLM model has no Tangram LayerWeave controller")
    return controller


def compact_layerweave_metrics(metrics):
    """Keep per-layer records in result JSON, not in the line-oriented log."""
    if metrics is None:
        return None
    stages = metrics.get("stages", [])
    return {
        "layer_path": metrics["layer_path"],
        "prefetch_enabled": metrics["prefetch_enabled"],
        "page_count": metrics["page_count"],
        "required_pages": metrics["required_pages"],
        "stage_count": len(stages),
        "mapped_pages": sum(item["mapped_pages"] for item in stages),
        "cached_pages": sum(item["cached_pages"] for item in stages),
        "to_load_bytes": metrics["to_load_bytes"],
        "cached_bytes": metrics["cached_bytes"],
        "h2d_ms": metrics["h2d_ms"],
    }


def log_cuda_memory(point, model_id):
    debug_model = os.environ.get("LAYERWEAVE_MEMORY_DEBUG_MODEL_ID", "")
    if debug_model == "" or int(debug_model) != model_id:
        return
    import torch
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    log_event(
        "CUDA_MEMORY",
        point=point,
        model_id=model_id,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        torch_allocated_bytes=torch.cuda.memory_allocated(),
        torch_reserved_bytes=torch.cuda.memory_reserved(),
    )


def initialize_engines(args, requests, models):
    # vLLM's single-GPU worker addresses the selected CUDA_VISIBLE_DEVICES
    # entry as local rank 0. Keep the VMM backend on that same logical device.
    os.environ["TANGRAM_VMM_IN_PROCESS"] = "1"
    os.environ["TANGRAM_KV_BACKEND"] = "vmm"
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    os.environ["TANGRAM_VMM_POOL_GIB"] = str(args.vmm_pool_gib)
    os.environ["TANGRAM_WEIGHT_LOAD_MODE"] = args.load_mode

    import torch
    from vllm import LLM
    from vllm.model_executor.model_loader.tangram_vmm import (
        TangramVmmPool,
        set_shared_vmm_pool,
    )

    logical_device = 0
    torch.cuda.set_device(logical_device)
    pool = TangramVmmPool(
        args.vmm_pool_gib,
        logical_device,
        page_size_mib=args.vmm_page_size_mib,
    )
    set_shared_vmm_pool(pool)
    required = sorted({item.model_id for item in requests})
    for model_id in required:
        pool.register_model(models[model_id]["rank_path"], model_id)

    print(
        f"VMM_POLICY={pool.vmm_policy} device={args.device} "
        f"pool_gib={args.vmm_pool_gib} "
        f"page_size_mib={args.vmm_page_size_mib} "
        f"weight_load_mode={args.load_mode} kv_backend=odkv",
        f"layerweave_prefetch={os.environ.get('LAYERWEAVE_PREFETCH', '1')}",
        flush=True,
    )

    engines = {}
    startup = {}
    engine_limits = {}
    for model_id in required:
        started = time.perf_counter()
        model_requests = [
            item for item in requests if item.model_id == model_id
        ]
        max_num_seqs = min(
            len(model_requests),
            (args.max_batch_size if args.max_batch_size > 0
             else len(model_requests)),
        )
        engine_max_model_len = max(
            item.input_tokens + item.output_tokens
            for item in model_requests
        )
        largest_inputs = sorted(
            (item.input_tokens for item in model_requests), reverse=True
        )[:max_num_seqs]
        max_output_tokens = max(
            item.output_tokens for item in model_requests
        )
        logical_token_capacity = (
            sum(largest_inputs) + max_output_tokens * max_num_seqs
        )
        options = {
            "model": models[model_id]["model_path"],
            "load_format": "serverless_llm",
            "dtype": args.dtype,
            "enforce_eager": True,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "swap_space": 0,
            "trust_remote_code": args.trust_remote_code,
            "max_num_seqs": max_num_seqs,
            "tokenizer_mode": models[model_id]["tokenizer_mode"],
            "max_model_len": engine_max_model_len,
            # These are scheduler-visible logical IDs, not eagerly reserved
            # physical KV pages. Physical pages are allocated lazily from VMM.
            "num_gpu_blocks_override": math.ceil(
                logical_token_capacity / 16
            ),
        }
        options.update(models[model_id]["vision_options"])
        engines[model_id] = LLM(**options)
        engine_limits[model_id] = {
            "max_model_len": engine_max_model_len,
            "max_num_seqs": max_num_seqs,
            "logical_token_capacity": logical_token_capacity,
            "num_gpu_blocks": options["num_gpu_blocks_override"],
        }
        startup[model_id] = (time.perf_counter() - started) * 1000.0
        log_event(
            "ENGINE_READY",
            gpu=args.device,
            load_mode=args.load_mode,
            kv_backend="odkv",
            model_id=model_id,
            model_path=models[model_id]["model_path"],
            tokenizer_mode=models[model_id]["tokenizer_mode"],
            vision_input_type=models[model_id]["vision_options"].get(
                "image_input_type", "none"
            ),
            **engine_limits[model_id],
            startup_ms=startup[model_id],
        )
    return pool, engines, startup, engine_limits


def run(args) -> dict:
    import torch
    from vllm import SamplingParams

    model_limits = load_model_input_limits(args.config)
    requests = prepare_requests(args, model_limits)
    models = load_models(args.config, args.model_path)
    missing = sorted({item.model_id for item in requests} - set(models))
    if missing:
        raise ValueError(f"Trace references missing model ids: {missing}")
    pool, engines, startup, engine_limits = initialize_engines(
        args, requests, models
    )

    pending: Deque[TraceRequest] = deque()
    next_index = 0
    results = []
    batches = []
    replay_start = time.perf_counter()
    active_model_id = None
    batch_id = 0

    while next_index < len(requests) or pending:
        now_s = time.perf_counter() - replay_start
        while (next_index < len(requests) and
               requests[next_index].arrival_s * args.trace_time_scale
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

        head_model_id = pending[0].model_id
        batch = pop_same_model(
            pending,
            args.max_batch_size,
            model_limits.get(head_model_id, {}).get(
                "max_batch_input_tokens", 0
            ),
        )
        model_id = batch[0].model_id
        dispatch_s = time.perf_counter() - replay_start
        dispatch_wall = time.time()
        switched = active_model_id != model_id
        load_result = None
        allocator_trim_ms = 0.0
        load_started = time.perf_counter()
        if switched:
            log_event(
                "MODEL_SWITCH",
                elapsed_s=dispatch_s,
                gpu=args.device,
                load_mode=args.load_mode,
                kv_backend="odkv",
                from_model_id=active_model_id,
                to_model_id=model_id,
                to_model_path=models[model_id]["model_path"],
                waiting_requests=len(pending) + len(batch),
            )
            if args.load_mode == "vmm":
                load_result = pool.load_model(
                    models[model_id]["rank_path"])
            else:
                trim_started = time.perf_counter()
                torch.cuda.empty_cache()
                allocator_trim_ms = (
                    time.perf_counter() - trim_started
                ) * 1000.0
                log_event(
                    "ALLOCATOR_TRIM",
                    gpu=args.device,
                    load_mode=args.load_mode,
                    model_id=model_id,
                    trim_ms=allocator_trim_ms,
                )
            active_model_id = model_id
        weight_load_ms = (time.perf_counter() - load_started) * 1000.0

        llm = engines[model_id]
        worker = get_worker(llm)
        layerweave = None
        if args.load_mode == "layerweave":
            layerweave = get_layerweave_controller(worker)
            layerweave.reset_metrics()
        vocab_size = int(worker.model_config.get_vocab_size())
        prompts = [
            make_prompt_tokens(item, vocab_size, args.seed) for item in batch
        ]
        sampling = [
            SamplingParams(
                temperature=0,
                max_tokens=item.output_tokens,
                ignore_eos=True,
            )
            for item in batch
        ]
        odkv_before = worker.odkv_stats()
        log_cuda_memory("before_generate", model_id)
        log_event(
            "DISPATCH",
            elapsed_s=dispatch_s,
            gpu=args.device,
            load_mode=args.load_mode,
            kv_backend="odkv",
            batch_id=batch_id,
            model_id=model_id,
            concurrent_requests=len(batch),
            waiting_requests_after_dispatch=len(pending),
            request_ids=[item.request_id for item in batch],
            input_lengths=[item.input_tokens for item in batch],
            output_lengths=[item.output_tokens for item in batch],
            max_batch_input_tokens=model_limits.get(
                model_id, {}
            ).get("max_batch_input_tokens", 0),
        )
        outputs = llm.generate(
            prompt_token_ids=prompts,
            sampling_params=sampling,
            use_tqdm=False,
        )
        layerweave_metrics = (
            layerweave.metrics() if layerweave is not None else None
        )
        if layerweave_metrics is not None:
            weight_load_ms = layerweave_metrics["h2d_ms"]
        odkv_after = worker.odkv_stats()
        output_by_index = list(outputs)
        batch_results = []
        prefill_values = []
        for request, output in zip(batch, output_by_index):
            metrics = output.metrics
            first_token_s = (
                dispatch_s + metrics.first_token_time - dispatch_wall
            )
            finish_s = dispatch_s + metrics.finished_time - dispatch_wall
            scheduled = (
                metrics.first_scheduled_time
                if metrics.first_scheduled_time is not None
                else metrics.arrival_time
            )
            prefill_values.append(
                (metrics.first_token_time - scheduled) * 1000.0
            )
            batch_results.append(RequestResult(
                request_id=request.request_id,
                model_id=model_id,
                arrival_s=request.arrival_s,
                dispatch_s=dispatch_s,
                first_token_s=first_token_s,
                finish_s=finish_s,
                input_tokens=request.input_tokens,
                output_tokens=request.output_tokens,
                batch_id=batch_id,
                batch_size=len(batch),
                trace_input_tokens=request.trace_input_tokens,
                trace_output_tokens=request.trace_output_tokens,
            ))
        odkv_delta = {
            key: odkv_after.get(key, 0) - odkv_before.get(key, 0)
            for key in (
                "allocate_calls", "release_calls", "allocated_blocks",
                "released_blocks", "allocate_ms", "release_ms",
            )
        }
        batch_metric = {
            "batch_id": batch_id,
            "model_id": model_id,
            "batch_size": len(batch),
            "input_tokens": sum(item.input_tokens for item in batch),
            "output_tokens": sum(item.output_tokens for item in batch),
            "cold_model_load": switched,
            "weight_load_ms": weight_load_ms,
            "allocator_trim_ms": allocator_trim_ms,
            "prefill_ms": max(prefill_values, default=0.0),
            "odkv": odkv_delta,
            "vmm_load": (
                {
                    "cached_bytes": int(load_result.cached_bytes),
                    "to_load_bytes": int(load_result.to_load_bytes),
                    "total_model_bytes": int(load_result.total_model_bytes),
                    "wall_load_ms": float(load_result.wall_load_ms),
                    "full_model_hit": bool(load_result.full_model_hit),
                } if load_result is not None else None
            ),
            "layerweave": layerweave_metrics,
        }
        log_event(
            "BATCH_COMPLETE",
            elapsed_s=time.perf_counter() - replay_start,
            gpu=args.device,
            load_mode=args.load_mode,
            kv_backend="odkv",
            batch_id=batch_id,
            model_id=model_id,
            concurrent_requests=len(batch),
            weight_load_ms=weight_load_ms,
            allocator_trim_ms=allocator_trim_ms,
            prefill_ms=batch_metric["prefill_ms"],
            odkv=odkv_delta,
            layerweave=compact_layerweave_metrics(layerweave_metrics),
            ttft_ms=[item.ttft_ms for item in batch_results],
            service_ttft_ms=[
                item.service_ttft_ms for item in batch_results
            ],
        )
        results.extend(batch_results)
        batches.append(batch_metric)
        batch_id += 1

    records = [
        item.record() for item in sorted(
            results, key=lambda item: item.request_id
        )
    ]
    summary = {
        "requests": len(records),
        "batches": len(batches),
        "ttft_ms": summarize([item["ttft_ms"] for item in records]),
        "service_ttft_ms": summarize(
            [item["service_ttft_ms"] for item in records]
        ),
        "queue_ms": summarize([item["queue_ms"] for item in records]),
        "e2e_ms": summarize([item["e2e_ms"] for item in records]),
        "weight_load_ms": summarize(
            [item["weight_load_ms"] for item in batches]
        ),
        "prefill_ms": summarize([item["prefill_ms"] for item in batches]),
    }
    final_odkv = {
        str(model_id): get_worker(engine).odkv_stats()
        for model_id, engine in engines.items()
    }
    return {
        "backend": f"{args.load_mode}_odkv",
        "load_mode": args.load_mode,
        "execution_engine": "vllm",
        "kv_backend": "segmented_odkv",
        "trace": str(args.trace.resolve()),
        "config": str(args.config.resolve()),
        "trace_time_scale": args.trace_time_scale,
        "output_tokens_override": args.output_tokens_override,
        "max_model_len": args.max_model_len,
        "truncate_input_to_model_limit":
            args.truncate_input_to_model_limit,
        "engine_startup_ms": startup,
        "engine_limits": engine_limits,
        "summary": summary,
        "requests": records,
        "batch_metrics": batches,
        "odkv": final_odkv,
        "vmm": {
            "policy": pool.vmm_policy,
            "pool_gib": args.vmm_pool_gib,
            "page_size_mib": args.vmm_page_size_mib,
            "source_format": "vllm_packed_rank",
            "layerweave_prefetch": os.environ.get(
                "LAYERWEAVE_PREFETCH", "1"),
        },
    }


def write_outputs(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(result, stream, indent=2)
    csv_path = output.with_suffix(".requests.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=result["requests"][0].keys()
        )
        writer.writeheader()
        writer.writerows(result["requests"])
    print("LAYERPIPE_SUMMARY=" + json.dumps(
        result["summary"], sort_keys=True
    ))
    print(f"LAYERPIPE_RESULT={output}")
    print(f"LAYERPIPE_REQUESTS={csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-path", action="append", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load-mode", choices=("vmm", "layerweave"),
                        default="vmm")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--output-tokens-override", type=int, default=0)
    parser.add_argument("--trace-time-scale", type=float, default=1.0)
    parser.add_argument("--vmm-pool-gib", type=float, default=40.0)
    parser.add_argument("--vmm-page-size-mib", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"),
                        default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--truncate-input-to-model-limit",
                        action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_batch_size < 0:
        parser.error("--max-batch-size must be non-negative")
    if args.max_model_len < 0:
        parser.error("--max-model-len must be non-negative")
    result = run(args)
    write_outputs(result, args.output)


if __name__ == "__main__":
    main()
