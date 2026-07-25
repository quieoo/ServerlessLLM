#!/usr/bin/env python3
"""Convert every packed model in a ServeGen config and write an HF config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="2GiB")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dry-run-first", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    output_config = args.output_config.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open() as stream:
        config = json.load(stream)
    models = config.get("model_lists")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{config_path} has no model_lists")

    converter = Path(__file__).with_name(
        "convert_rank0_to_safetensors.py"
    )
    converted = []
    for item in models:
        model_id = int(item["id"])
        rank_path = Path(item["path"]).resolve()
        destination = output_root / f"model_{model_id}"
        summary_path = destination / "conversion_summary.json"
        if summary_path.is_file():
            print(
                f"SKIP model_id={model_id}: completed at {destination}",
                flush=True,
            )
        else:
            common = [
                sys.executable,
                str(converter),
                str(rank_path),
            ]
            if args.trust_remote_code:
                common.append("--trust-remote-code")
            if args.dry_run_first:
                print(f"DRY_RUN model_id={model_id}", flush=True)
                subprocess.run(common + ["--dry-run"], check=True)
            print(
                f"CONVERT model_id={model_id} -> {destination}", flush=True
            )
            subprocess.run(
                common + [
                    "--output", str(destination),
                    "--max-shard-size", args.max_shard_size,
                ],
                check=True,
            )
        updated = dict(item)
        updated["packed_rank_path"] = str(rank_path)
        updated["path"] = str(destination)
        updated["hf_path"] = str(destination)
        converted.append(updated)

    new_config = dict(config)
    new_config["model_lists"] = converted
    temporary = output_config.with_suffix(output_config.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(new_config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, output_config)
    print(f"SERVEGEN_HF_CONFIG={output_config}", flush=True)


if __name__ == "__main__":
    main()
