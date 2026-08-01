#!/usr/bin/env python3
"""Create config variants that remap trace model ids to physical models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mapping", action="append", required=True,
        help="NAME=comma-separated physical model ids in trace-id order")
    args = parser.parse_args()

    base = json.loads(args.base_config.read_text())
    model_ids = sorted(int(item["id"]) for item in base["model_lists"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for specification in args.mapping:
        name, raw_order = specification.split("=", 1)
        order = [int(value) for value in raw_order.split(",")]
        if sorted(order) != model_ids:
            raise ValueError(
                f"{name}: mapping must be a permutation of {model_ids}")
        document = dict(base)
        document["model_mapping_order"] = order
        document["model_mapping_semantics"] = (
            "array index is trace model_id; value is physical/profile "
            "model_id")
        output = args.output_dir / f"{args.base_config.stem}-{name}.json"
        output.write_text(json.dumps(document, indent=2) + "\n")
        print(output)


if __name__ == "__main__":
    main()
