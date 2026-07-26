#!/usr/bin/env python3
"""Measure LayerPipe residual loading stall with matched cold/warm pairs."""

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from layerpipe_bench import (
    CpuModelCache,
    LayerPipeline,
    apply_request_input_limits,
    execute_batch,
    load_model_input_limits,
    load_model_paths,
    parse_trace,
    scale_request_inputs,
    summarize,
)


def prepare_requests(args):
    requests = parse_trace(args.trace, args.max_requests)
    for request in requests:
        request.trace_input_tokens = request.input_tokens
        request.trace_output_tokens = request.output_tokens
        request.output_tokens = 1
    scale_request_inputs(requests, args.input_scale)
    limits = load_model_input_limits(args.config)
    apply_request_input_limits(
        requests,
        limits,
        args.max_model_len,
        args.truncate_input_to_model_limit,
    )
    if args.max_pairs_per_model > 0:
        counts = Counter()
        selected = []
        for request in requests:
            if counts[request.model_id] >= args.max_pairs_per_model:
                continue
            selected.append(request)
            counts[request.model_id] += 1
        requests = selected
    return requests


def execute_once(model, loader, request, device, seed, replay_start,
                 batch_id):
    dispatch_s = time.perf_counter() - replay_start
    results, metrics = execute_batch(
        model,
        loader,
        [request],
        device,
        seed,
        dispatch_s,
        replay_start,
        batch_id,
    )
    return results[0].service_ttft_ms, metrics


def run(args):
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    requests = prepare_requests(args)
    paths = load_model_paths(args.config, args.model_path)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    cache = CpuModelCache(paths, dtype, args.trust_remote_code)
    if args.preload_cpu_models:
        cache.preload(item.model_id for item in requests)

    warmed_models = set()
    pairs = []
    replay_start = time.perf_counter()
    batch_id = 0
    active_loader = None
    try:
        for pair_id, request in enumerate(requests):
            model = cache.get(request.model_id)
            weights = cache.get_weight_store(request.model_id)

            if request.model_id not in warmed_models:
                warmup_loader = LayerPipeline(model, device, weights)
                execute_once(
                    model, warmup_loader, request, device, args.seed,
                    replay_start, batch_id,
                )
                batch_id += 1
                execute_once(
                    model, warmup_loader, request, device, args.seed,
                    replay_start, batch_id,
                )
                batch_id += 1
                warmup_loader.close()
                warmed_models.add(request.model_id)

            # Restore-to-CPU and allocator cleanup happen before cold timing.
            # The resulting cold service time therefore excludes cleanup of
            # the preceding pair.
            active_loader = LayerPipeline(model, device, weights)
            cold_ms, cold_metrics = execute_once(
                model, active_loader, request, device, args.seed,
                replay_start, batch_id,
            )
            batch_id += 1
            loading = active_loader.loading_metrics()

            # The same model, request, generated tokens, and loader are reused
            # immediately. All weights are resident, so this is the matched
            # compute-only service baseline for the cold execution above.
            # Drop cached activation allocations first so the warm baseline
            # does not get an allocator-cache advantage over the cold run.
            torch.cuda.empty_cache()
            warm_ms, warm_metrics = execute_once(
                model, active_loader, request, device, args.seed,
                replay_start, batch_id,
            )
            batch_id += 1
            signed_delta_ms = cold_ms - warm_ms
            clear_ms = active_loader.close()
            active_loader = None

            pair = {
                "pair_id": pair_id,
                "source_request_id": request.request_id,
                "model_id": request.model_id,
                "trace_input_tokens": request.trace_input_tokens,
                "input_tokens": request.input_tokens,
                "input_scale": args.input_scale,
                "cold_service_ttft_ms": cold_ms,
                "warm_service_ttft_ms": warm_ms,
                "signed_cold_minus_warm_ms": signed_delta_ms,
                "residual_loading_stall_ms": max(0.0, signed_delta_ms),
                "cold_synchronous_load_ms":
                    cold_metrics["synchronous_load_ms"],
                "cold_prefill_ms": cold_metrics["prefill_ms"],
                "warm_prefill_ms": warm_metrics["prefill_ms"],
                "sum_h2d_ms": loading["sum_h2d_ms"],
                "post_pair_clear_ms": clear_ms,
            }
            pairs.append(pair)
            print(
                "PAIRED_RESIDUAL="
                + json.dumps(pair, sort_keys=True),
                flush=True,
            )
    finally:
        if active_loader is not None:
            active_loader.close()

    residuals = [item["residual_loading_stall_ms"] for item in pairs]
    signed = [item["signed_cold_minus_warm_ms"] for item in pairs]
    return {
        "backend": "layerpipe_paired_residual",
        "metric_definition": (
            "max(0, cold_service_ttft_ms - warm_service_ttft_ms); "
            "same model/input/tokens; cleanup excluded; first pair per model "
            "preceded by an unrecorded cold/warm warm-up"
        ),
        "config": str(args.config.resolve()),
        "trace": str(args.trace.resolve()),
        "input_scale": args.input_scale,
        "max_requests": args.max_requests,
        "max_pairs_per_model": args.max_pairs_per_model,
        "summary": {
            "pairs": len(pairs),
            "models": len({item["model_id"] for item in pairs}),
            "residual_loading_stall_ms": summarize(residuals),
            "signed_cold_minus_warm_ms": summarize(signed),
            "negative_delta_pairs": sum(value < 0 for value in signed),
            "mean_cold_service_ttft_ms": statistics.mean(
                item["cold_service_ttft_ms"] for item in pairs
            ),
            "mean_warm_service_ttft_ms": statistics.mean(
                item["warm_service_ttft_ms"] for item in pairs
            ),
        },
        "pairs": pairs,
    }


def write_outputs(result, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    csv_path = output.with_suffix(".pairs.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=result["pairs"][0].keys()
        )
        writer.writeheader()
        writer.writerows(result["pairs"])
    print("PAIRED_RESIDUAL_SUMMARY=" + json.dumps(
        result["summary"], sort_keys=True
    ))
    print(f"PAIRED_RESIDUAL_RESULT={output}")
    print(f"PAIRED_RESIDUAL_PAIRS={csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-path", action="append", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--max-pairs-per-model", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument(
        "--truncate-input-to-model-limit", action="store_true"
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--preload-cpu-models", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be non-negative")
    if args.max_pairs_per_model < 0:
        parser.error("--max-pairs-per-model must be non-negative")
    if args.max_model_len < 0:
        parser.error("--max-model-len must be non-negative")
    if not math.isfinite(args.input_scale) or args.input_scale <= 0:
        parser.error("--input-scale must be a finite positive number")
    result = run(args)
    if not result["pairs"]:
        parser.error("the selected trace contains no request pairs")
    write_outputs(result, args.output)


if __name__ == "__main__":
    main()
