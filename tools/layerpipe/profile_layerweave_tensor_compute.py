#!/usr/bin/env python3
"""Sequential GPU microprofile for tensor-shaped linear operators.

This profiles isolated FP16 GEMMs using safetensors weight shapes.  Repeated
shapes are measured once and shared.  It is a useful tensor-granular proxy,
not a replacement for CUDA-event instrumentation in the real vLLM forward.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as functional


def parse_ints(text):
    return [int(value) for value in text.split(",") if value]


def measure_linear(shape, tokens, repeats, device):
    out_features, in_features = shape
    weight = torch.empty(
        (out_features, in_features), dtype=torch.float16, device=device)
    torch.nn.init.normal_(weight, mean=0.0, std=0.01)
    samples = {}
    for token_count in tokens:
        source = torch.empty(
            (token_count, in_features), dtype=torch.float16, device=device)
        torch.nn.init.normal_(source, mean=0.0, std=0.01)
        for _ in range(2):
            functional.linear(source, weight)
        torch.cuda.synchronize(device)
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = functional.linear(source, weight)
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
            del output
        samples[str(token_count)] = {
            "median_ms": statistics.median(values),
            "samples_ms": values,
        }
        del source
    del weight
    torch.cuda.empty_cache()
    return samples


def linear_fit(samples):
    points = sorted(
        (int(tokens), float(item["median_ms"]))
        for tokens, item in samples.items()
    )
    if len(points) == 1:
        return [0.0, points[0][1] / max(1, points[0][0])]
    count = len(points)
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)
    sum_xx = sum(point[0] ** 2 for point in points)
    sum_xy = sum(point[0] * point[1] for point in points)
    denominator = count * sum_xx - sum_x * sum_x
    slope = (
        (count * sum_xy - sum_x * sum_y) / denominator
        if denominator else 0.0
    )
    intercept = (sum_y - slope * sum_x) / count
    return [max(0.0, intercept), max(0.0, slope)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", default="")
    parser.add_argument("--tokens", default="32,128")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-unique-shapes", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    selected = set(parse_ints(args.models)) if args.models else None
    tokens = parse_ints(args.tokens)
    layout = json.loads(args.tensor_layout.read_text())
    shape_profiles = {}
    tensor_profiles = {}
    started = time.perf_counter()
    for model_key, model in layout["models"].items():
        model_id = int(model_key)
        if selected is not None and model_id not in selected:
            continue
        tensor_profiles[model_key] = {}
        for tensor in model["tensors"]:
            shape = tensor.get("shape") or []
            if len(shape) != 2:
                continue
            shape_key = f"{int(shape[0])}x{int(shape[1])}"
            if shape_key not in shape_profiles:
                if (
                    args.max_unique_shapes > 0
                    and len(shape_profiles) >= args.max_unique_shapes
                ):
                    continue
                print(f"profiling shape={shape_key}", flush=True)
                samples = measure_linear(
                    (int(shape[0]), int(shape[1])),
                    tokens, args.repeats, device)
                shape_profiles[shape_key] = {
                    "shape": [int(shape[0]), int(shape[1])],
                    "samples": samples,
                    "coefficients": linear_fit(samples),
                }
            tensor_profiles[model_key][tensor["name"]] = {
                "shape_key": shape_key,
                "coefficients":
                    shape_profiles[shape_key]["coefficients"],
            }
    document = {
        "format": "layerweave-tensor-compute-profile-v1",
        "source": "isolated-fp16-linear-gpu-microbenchmark",
        "caveat":
            "not full vLLM fused-operator or attention/runtime profiling",
        "gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(device),
        "tokens": tokens,
        "repeats": args.repeats,
        "elapsed_seconds": time.perf_counter() - started,
        "shape_profiles": shape_profiles,
        "models": tensor_profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "gpu": document["gpu_name"],
        "unique_shapes": len(shape_profiles),
        "profiled_tensors": sum(
            len(items) for items in tensor_profiles.values()),
        "elapsed_seconds": document["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
