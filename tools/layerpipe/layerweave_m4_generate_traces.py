#!/usr/bin/env python3
"""Generate deterministic multi-model M4 calibration/evaluation traces."""

import argparse
import json
from pathlib import Path


def distinct_lengths(cap, fractions):
    values = []
    for fraction in fractions:
        value = max(3, min(cap, round(cap * fraction)))
        while value in values and value < cap:
            value += 1
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"token cap {cap} is too small for distinct points")
    return values


def write_trace(
    path, models, batch_size, fractions, repeats, input_scale
):
    lines = []
    for model in models:
        model_id = int(model["id"])
        input_limit = int(model.get("l40_safe_max_input_length", 0))
        batch_limit = int(model.get(
            "l40_safe_max_batch_input_tokens", 0))
        caps = [3072]
        if input_limit > 0:
            caps.append(input_limit)
        if batch_limit > 0:
            caps.append(batch_limit // batch_size)
        # The benchmark applies INPUT_SCALE after reading the trace. Generate
        # unscaled lengths so the measured lengths still cover the configured
        # safe-input fractions instead of all clipping at the model limit.
        cap = int(min(caps) // input_scale)
        if cap < 8:
            raise ValueError(
                f"model {model_id} has token cap {cap} at batch "
                f"size {batch_size}")
        lengths = distinct_lengths(cap, fractions)
        for _ in range(repeats):
            for tokens in lengths:
                for _request in range(batch_size):
                    lines.append(
                        f"0.000000 {model_id} {tokens} 1\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    return len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes", default="1,2,4,8",
        help="Comma-separated batch sizes")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--input-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.input_scale <= 0:
        raise ValueError("--input-scale must be positive")

    document = json.loads(args.config.read_text())
    models = document["model_lists"]
    batch_sizes = [
        int(value) for value in args.batch_sizes.split(",") if value
    ]
    scale_tag = (
        "" if args.input_scale == 1.0
        else f"_scale{args.input_scale:g}".replace(".", "p")
    )
    for batch_size in batch_sizes:
        for split, fractions in (
            ("calibration", (0.08, 0.34, 0.68)),
            ("evaluation", (0.16, 0.50, 0.90)),
        ):
            path = (
                args.output_dir
                / f"layerweave_m4_full_{split}_b{batch_size}"
                f"{scale_tag}.trace"
            )
            requests = write_trace(
                path, models, batch_size, fractions, args.repeats,
                args.input_scale)
            print(
                f"M4_TRACE={path} split={split} batch_size={batch_size} "
                f"input_scale={args.input_scale:g} requests={requests}")


if __name__ == "__main__":
    main()
