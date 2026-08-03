#!/usr/bin/env python3
"""Generate long-context traces and synthetic model-replica catalogs."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path


TEMPLATES = (0, 1, 2, 4)
INPUT_LEVELS = (128, 512, 2048, 8192, 16384)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def balanced_sequence(model_ids, requests, seed):
    """Balanced, seed-varying blocks without changing model popularity."""
    rng = random.Random(seed)
    sequence = []
    while len(sequence) < requests:
        block = list(model_ids)
        rng.shuffle(block)
        if sequence and block[0] == sequence[-1] and len(block) > 1:
            block[0], block[1] = block[1], block[0]
        sequence.extend(block)
    return sequence[:requests]


def write_trace(path, model_ids, requests, input_tokens, seed, metadata):
    sequence = balanced_sequence(model_ids, requests, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for index, model_id in enumerate(sequence):
            stream.write(
                f"{index * 0.01:.6f} {model_id} {input_tokens} 1\n")
    counts = Counter(sequence)
    write_json(path.with_suffix(path.suffix + ".json"), {
        "format": "sgrp-revised-balanced-v1",
        "trace": str(path),
        "seed": seed,
        "requests": requests,
        "input_tokens": input_tokens,
        "model_ids": list(model_ids),
        "counts": {str(key): counts[key] for key in model_ids},
        **metadata,
    })


def expanded_catalog(config, table, copies):
    by_id = {int(item["id"]): item for item in config["model_lists"]}
    expanded_config = copy.deepcopy(config)
    expanded_config.pop("model_mapping_order", None)
    expanded_config["model_lists"] = []
    expanded_models = {}
    replica_map = {}
    next_id = 0
    for replica in range(copies):
        for template_id in TEMPLATES:
            logical_id = next_id
            next_id += 1
            item = copy.deepcopy(by_id[template_id])
            item["id"] = logical_id
            item["synthetic_replica"] = replica
            item["template_model_id"] = template_id
            expanded_config["model_lists"].append(item)
            profile = copy.deepcopy(table["models"][str(template_id)])
            profile["model_id"] = logical_id
            profile["synthetic_replica"] = replica
            profile["template_model_id"] = template_id
            expanded_models[str(logical_id)] = profile
            replica_map[str(logical_id)] = {
                "template_model_id": template_id,
                "replica": replica,
            }
    expanded_config["synthetic_model_replicas"] = replica_map
    expanded_table = copy.deepcopy(table)
    expanded_table["models"] = expanded_models
    expanded_table["metadata"] = copy.deepcopy(table["metadata"])
    expanded_table["metadata"]["synthetic_model_replicas"] = replica_map
    expanded_table["metadata"]["replica_semantics"] = (
        "Independent logical weights and cache identities sharing only the "
        "template model size and offline execution profile")
    return expanded_config, expanded_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stall-table", type=Path, required=True)
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238")
    parser.add_argument("--input-requests", type=int, default=1000)
    parser.add_argument("--routing-requests-per-gpu", type=int, default=1000)
    parser.add_argument("--pressure-requests-per-model", type=int, default=125)
    parser.add_argument(
        "--pressure-requests", type=int,
        help="Fix the total requests at every pressure level; overrides "
             "--pressure-requests-per-model")
    parser.add_argument(
        "--pressure-copies",
        help="Comma-separated pressure catalog copies; default: 1,2,4,...")
    parser.add_argument("--max-copies", type=int, default=12)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    config = json.loads(args.config.read_text())
    table = json.loads(args.stall_table.read_text())

    for copies in range(1, args.max_copies + 1):
        expanded_config, expanded_table = expanded_catalog(
            config, table, copies)
        catalog = args.output_dir / "catalogs" / f"copies{copies}"
        write_json(catalog / "config.json", expanded_config)
        write_json(catalog / "stall-table.json", expanded_table)

    for seed in seeds:
        for tokens in INPUT_LEVELS:
            write_trace(
                args.output_dir / "input" / f"i{tokens}-seed{seed}.trace",
                range(8), args.input_requests, tokens, seed,
                {"dimension": "input", "effective_input_tokens": tokens})
        for gpus in range(1, 5):
            ids = range(4 * gpus)
            write_trace(
                args.output_dir / "routing" / f"g{gpus}-seed{seed}.trace",
                ids, args.routing_requests_per_gpu * gpus, 256, seed,
                {"dimension": "routing", "gpus": gpus,
                 "catalog_copies": gpus})
        pressure_copies = (
            tuple(int(value) for value in args.pressure_copies.split(","))
            if args.pressure_copies else
            (1, *range(2, args.max_copies + 1, 2)))
        for copies in pressure_copies:
            ids = range(4 * copies)
            requests = (args.pressure_requests if args.pressure_requests
                        is not None else
                        args.pressure_requests_per_model * len(ids))
            write_trace(
                args.output_dir / "pressure"
                / f"c{copies}-seed{seed}.trace",
                ids, requests, 256, seed,
                {"dimension": "pressure", "gpus": 4,
                 "catalog_copies": copies})


if __name__ == "__main__":
    main()
