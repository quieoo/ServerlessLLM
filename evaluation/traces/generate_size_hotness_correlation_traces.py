#!/usr/bin/env python3
"""Generate matched traces by permuting popularity ranks onto model sizes."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from generate_target_trace import (
    largest_remainder_counts,
    read_shapes,
    read_source_model_sequence,
    sequence_metrics,
    token_multiset_hash,
)


LEVELS = (
    ("neg1", -1.0),
    ("neg0p5", -0.5),
    ("zero", 0.0),
    ("pos0p5", 0.5),
    ("pos1", 1.0),
)


def correlation(xs, ys):
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = (
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)) ** 0.5
    return numerator / denominator


def choose_mappings(model_sizes):
    size_order = sorted(model_sizes, key=lambda model: (model_sizes[model], model))
    size_rank = {model: rank for rank, model in enumerate(size_order)}
    permutations = list(itertools.permutations(size_order))

    def rank_correlation(mapping):
        return correlation(
            [size_rank[model] for model in mapping],
            [len(mapping) - 1 - rank for rank in range(len(mapping))])

    result = {}
    for name, target in LEVELS:
        mapping = min(
            permutations,
            key=lambda item: (abs(rank_correlation(item) - target), item))
        result[name] = {
            "target": target,
            "rank_to_model": mapping,
            "spearman_size_hotness": rank_correlation(mapping),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-layout", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--source-requests", type=int, default=1000)
    parser.add_argument("--fixed-input-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238")
    parser.add_argument("--lookahead-k", type=int, default=64)
    args = parser.parse_args()

    layout = json.loads(args.tensor_layout.read_text())
    model_sizes = {
        int(model): item["logical_bytes"]
        for model, item in layout["models"].items()
    }
    model_ids = sorted(model_sizes)
    source_sequence = read_source_model_sequence(
        args.source, args.source_requests)
    source_counts = Counter(source_sequence)
    rank_models = sorted(
        model_ids, key=lambda model: (-source_counts[model], model))
    counts_by_model = largest_remainder_counts(
        source_counts, model_ids, args.requests)
    rank_counts = [counts_by_model[model] for model in rank_models]
    shapes = read_shapes(args.source)
    mappings = choose_mappings(model_sizes)
    seeds = [int(value) for value in args.seeds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        sequence_rng = random.Random(seed)
        rank_sequence = [
            rank for rank, count in enumerate(rank_counts)
            for _ in range(count)
        ]
        sequence_rng.shuffle(rank_sequence)
        token_rng = random.Random(seed ^ 0x5A17C0DE)
        token_pools = {
            model: token_rng.sample(values, len(values))
            for model, values in shapes.items()
        }
        for level, mapping_data in mappings.items():
            mapping = mapping_data["rank_to_model"]
            model_sequence = [mapping[rank] for rank in rank_sequence]
            positions = Counter()
            rows = []
            rows_by_model = defaultdict(list)
            for index, model in enumerate(model_sequence):
                source_input, source_output = token_pools[model][
                    positions[model] % len(token_pools[model])]
                positions[model] += 1
                input_tokens = (
                    source_input if args.fixed_input_tokens is None
                    else args.fixed_input_tokens)
                output_tokens = (
                    source_output if args.output_tokens is None
                    else args.output_tokens)
                rows.append(
                    f"{index / 100.0:.6f} {model} "
                    f"{input_tokens} {output_tokens}\n")
                rows_by_model[model].append((input_tokens, output_tokens))
            output = args.output_dir / f"{level}-seed{seed}.trace"
            output.write_text("".join(rows))
            counts = Counter(model_sequence)
            shares = [counts[model] / args.requests for model in model_ids]
            size_values = [model_sizes[model] for model in model_ids]
            metadata = {
                "format": "size-hotness-correlation-v1",
                "trace": str(output),
                "source": str(args.source),
                "seed": seed,
                "requests": args.requests,
                "level": level,
                "target_spearman_size_hotness": mapping_data["target"],
                "spearman_size_hotness":
                    mapping_data["spearman_size_hotness"],
                "pearson_bytes_share": correlation(size_values, shares),
                "rank_to_model": list(mapping),
                "rank_counts": rank_counts,
                "counts": dict(sorted(counts.items())),
                "model_shares": {
                    model: counts[model] / args.requests for model in model_ids
                },
                "model_sizes_bytes": model_sizes,
                "metrics": sequence_metrics(
                    model_sequence, args.lookahead_k),
                "lookahead_k": args.lookahead_k,
                "fixed_input_tokens": args.fixed_input_tokens,
                "output_tokens": args.output_tokens,
                "token_multiset_sha256": token_multiset_hash(rows_by_model),
            }
            output.with_suffix(".trace.json").write_text(
                json.dumps(metadata, indent=2) + "\n")
            print(
                f"{output} rho={mapping_data['spearman_size_hotness']:+.2f} "
                f"pearson={metadata['pearson_bytes_share']:+.3f}")


if __name__ == "__main__":
    main()
