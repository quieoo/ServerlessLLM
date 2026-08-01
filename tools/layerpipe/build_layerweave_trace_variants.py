#!/usr/bin/env python3
"""Build deterministic model-sequence variants with matched token shapes.

The source trace supplies each model's empirical (input, output) token pairs.
Variants change only the model sequence: every selected model consumes the next
shape from that model's cyclic, seed-shuffled source pool.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def read_shapes(path):
    shapes = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            shapes[int(fields[1])].append(
                (int(fields[2]), int(fields[3])))
    if not shapes:
        raise ValueError(f"No requests in {path}")
    return dict(shapes)


def balanced_iid(model_ids, count, rng):
    sequence = []
    while len(sequence) < count:
        choices = [
            model_id for model_id in model_ids
            if not sequence or model_id != sequence[-1]
        ]
        sequence.append(rng.choice(choices))
    return sequence


def round_robin(model_ids, count, rng):
    sequence = []
    while len(sequence) < count:
        block = list(model_ids)
        rng.shuffle(block)
        if sequence and block[0] == sequence[-1]:
            block[0], block[1] = block[1], block[0]
        sequence.extend(block)
    return sequence[:count]


def phase_shift(model_ids, count, rng):
    phase_hotsets = (
        (0, 1, 4),
        (2, 3, 5),
        (0, 6, 7),
        (1, 4, 5),
        (2, 6, 7),
    )
    phase_length = max(1, count // len(phase_hotsets))
    sequence = []
    for position in range(count):
        phase = min(
            len(phase_hotsets) - 1, position // phase_length)
        hot = phase_hotsets[phase]
        cold = tuple(model_id for model_id in model_ids if model_id not in hot)
        population = hot if rng.random() < 0.70 else cold
        choices = [
            model_id for model_id in population
            if not sequence or model_id != sequence[-1]
        ]
        if not choices:
            choices = [
                model_id for model_id in model_ids
                if model_id != sequence[-1]
            ]
        sequence.append(rng.choice(choices))
    return sequence


def working_set_shift(model_ids, count, rng):
    """Alternate overlapping four-model working sets every 100 requests."""
    working_sets = (
        (0, 1, 4, 5),
        (2, 3, 4, 5),
        (0, 2, 6, 7),
        (1, 3, 6, 7),
    )
    sequence = []
    for position in range(count):
        active = working_sets[(position // 100) % len(working_sets)]
        choices = [
            model_id for model_id in active
            if not sequence or model_id != sequence[-1]
        ]
        sequence.append(rng.choice(choices))
    return sequence


def write_variant(path, sequence, shapes, seed):
    rng = random.Random(seed)
    shuffled = {
        model_id: rng.sample(values, len(values))
        for model_id, values in shapes.items()
    }
    positions = defaultdict(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for index, model_id in enumerate(sequence):
            values = shuffled[model_id]
            input_tokens, output_tokens = values[
                positions[model_id] % len(values)]
            positions[model_id] += 1
            stream.write(
                f"{index / 100.0:.6f} {model_id} "
                f"{input_tokens} {output_tokens}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path("evaluation/traces/servegen_tangram.trace"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("evaluation/traces/layerweave_variants"))
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    shapes = read_shapes(args.source)
    model_ids = sorted(shapes)
    rng = random.Random(args.seed)
    variants = {
        "balanced_iid": balanced_iid(
            model_ids, args.requests, random.Random(rng.randrange(2**32))),
        "round_robin": round_robin(
            model_ids, args.requests, random.Random(rng.randrange(2**32))),
        "phase_shift": phase_shift(
            model_ids, args.requests, random.Random(rng.randrange(2**32))),
        "working_set_shift": working_set_shift(
            model_ids, args.requests, random.Random(rng.randrange(2**32))),
    }
    for offset, (name, sequence) in enumerate(variants.items()):
        path = args.output_dir / f"{name}.trace"
        write_variant(path, sequence, shapes, args.seed + offset)
        print(f"{name}: requests={len(sequence)} output={path}")


if __name__ == "__main__":
    main()
