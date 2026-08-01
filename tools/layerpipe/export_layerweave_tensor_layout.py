#!/usr/bin/env python3
"""Export tensor-level layouts from Hugging Face safetensors checkpoints.

The compact offset is the deterministic concatenation order used by this
exporter.  It is suitable for the CPU simulator.  A GPU/runtime export should
replace it when exact stable-VA offsets are required.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)transformer\.blocks\.(\d+)(?:\.|$)"),
)
_OPERATOR_ORDER = (
    "input_layernorm",
    "attention_norm",
    "qkv_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "query_key_value",
    "o_proj",
    "out_proj",
    "dense",
    "post_attention_layernorm",
    "ffn_norm",
    "gate_up_proj",
    "gate_proj",
    "up_proj",
    "w1",
    "w3",
    "down_proj",
    "w2",
)


def _layer_id(name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def _operator_rank(name: str) -> int:
    for rank, token in enumerate(_OPERATOR_ORDER):
        if token in name:
            return rank
    return len(_OPERATOR_ORDER)


def _stage_sort_key(item: dict) -> tuple:
    name = item["name"]
    layer = item["layer_id"]
    if layer is None:
        if any(token in name for token in ("embed", "wte", "tok_embeddings")):
            return (-1, 0, name)
        return (10**9, _operator_rank(name), name)
    return (layer, _operator_rank(name), name)


def _shards(model_path: Path) -> list[Path]:
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return [model_path / name for name in sorted(set(weight_map.values()))]
    return sorted(model_path.glob("*.safetensors"))


def read_tensors(model_path: Path) -> list[dict]:
    tensors = []
    for shard in _shards(model_path):
        with shard.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            begin, end = metadata["data_offsets"]
            tensors.append({
                "name": name,
                "logical_bytes": int(end) - int(begin),
                "shape": [int(value) for value in metadata["shape"]],
                "dtype": metadata["dtype"],
                "layer_id": _layer_id(name),
                "source_shard": shard.name,
            })
    if not tensors:
        raise ValueError(f"No safetensors found under {model_path}")
    tensors.sort(key=_stage_sort_key)
    compact_offset = 0
    for tensor_id, tensor in enumerate(tensors):
        tensor["tensor_id"] = tensor_id
        tensor["stage_id"] = tensor_id
        tensor["compact_offset"] = compact_offset
        compact_offset += tensor["logical_bytes"]
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    models = {}
    for item in config["model_lists"]:
        model_id = int(item["id"])
        model_path = Path(item.get("hf_path") or item["path"])
        tensors = read_tensors(model_path)
        models[str(model_id)] = {
            "model_id": model_id,
            "model_path": str(model_path),
            "logical_bytes": sum(t["logical_bytes"] for t in tensors),
            "tensor_count": len(tensors),
            "layout_source": "safetensors-header-derived",
            "compact_offset_semantics":
                "exporter-order proxy; not runtime stable-VA verified",
            "tensors": tensors,
        }
    document = {
        "format": "layerweave-tensor-layout-v1",
        "config": str(args.config.resolve()),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "models": len(models),
        "tensors": sum(m["tensor_count"] for m in models.values()),
        "logical_bytes": sum(m["logical_bytes"] for m in models.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
