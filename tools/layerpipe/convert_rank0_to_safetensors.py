#!/usr/bin/env python3
"""Convert Tangram rank_0/tensor.data_* into HF safetensors shards.

The source format stores raw tensor storages in numbered partitions and keeps
global offsets in tensor_group_index.txt.  Some vLLM models use packed QKV and
gate/up tensors; this converter consults the Hugging Face model created from
config.json on the meta device and splits packed tensors only when the target
architecture requires separate weights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForVision2Seq)


DTYPES = {
    "torch.bool": torch.bool,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


@dataclass(frozen=True)
class SourceTensor:
    name: str
    global_offset: int
    storage_bytes: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class OutputTensor:
    name: str
    source_name: str
    split_start: Optional[int]
    split_length: Optional[int]
    shape: Tuple[int, ...]
    dtype: torch.dtype

    @property
    def size_bytes(self) -> int:
        return (
            int(torch.tensor(self.shape).prod().item())
            * torch.empty((), dtype=self.dtype).element_size()
        )


def parse_size(value: str) -> int:
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*",
        value,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Invalid size {value!r}")
    number = float(match.group(1))
    suffix = (match.group(2) or "b").lower()
    factors = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
        "tb": 1000**4,
        "tib": 1024**4,
    }
    return int(number * factors[suffix])


def parse_group_index(path: Path) -> Dict[str, Tuple[int, int]]:
    offsets: Dict[str, Tuple[int, int]] = {}
    group_offset: Optional[int] = None
    current_name: Optional[str] = None
    current_offset: Optional[int] = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Group Offset:"):
            group_offset = int(line.split(":", 1)[1])
        elif line.startswith("Tensor Name:"):
            current_name = line.split(":", 1)[1].strip()
            current_offset = None
        elif current_name is not None and line.startswith("Offset:"):
            current_offset = int(line.split(":", 1)[1])
        elif current_name is not None and line.startswith("Size:"):
            if group_offset is None or current_offset is None:
                raise ValueError(
                    f"Malformed tensor entry for {current_name} in {path}"
                )
            if current_name in offsets:
                raise ValueError(f"Duplicate tensor {current_name} in {path}")
            offsets[current_name] = (
                group_offset + current_offset,
                int(line.split(":", 1)[1]),
            )
            current_name = None
            current_offset = None
    if not offsets:
        raise ValueError(f"No tensor entries found in {path}")
    return offsets


def load_source_tensors(rank_path: Path) -> List[SourceTensor]:
    with (rank_path / "tensor_meta_index.json").open() as stream:
        metadata = json.load(stream)
    offsets = parse_group_index(rank_path / "tensor_group_index.txt")
    missing = sorted(set(metadata) - set(offsets))
    extra = sorted(set(offsets) - set(metadata))
    if missing or extra:
        raise ValueError(
            f"Index mismatch: missing offsets={missing[:8]}, "
            f"missing metadata={extra[:8]}"
        )
    result = []
    for name, (shape, stride, dtype_text) in metadata.items():
        if dtype_text not in DTYPES:
            raise ValueError(f"Unsupported dtype {dtype_text} for {name}")
        global_offset, storage_bytes = offsets[name]
        result.append(
            SourceTensor(
                name=name,
                global_offset=global_offset,
                storage_bytes=storage_bytes,
                shape=tuple(shape),
                stride=tuple(stride),
                dtype=DTYPES[dtype_text],
            )
        )
    return result


class PartitionReader:
    def __init__(self, paths: Sequence[Path]):
        if not paths:
            raise ValueError("No tensor.data_* partitions found")
        self.paths = list(paths)
        self.sizes = [path.stat().st_size for path in self.paths]
        self.starts = []
        total = 0
        for size in self.sizes:
            self.starts.append(total)
            total += size
        self.total_bytes = total
        self.files = [path.open("rb", buffering=0) for path in self.paths]

    def read(self, offset: int, size: int) -> bytearray:
        if offset < 0 or size < 0 or offset + size > self.total_bytes:
            raise ValueError(
                f"Read [{offset}, {offset + size}) exceeds packed size "
                f"{self.total_bytes}"
            )
        output = bytearray(size)
        view = memoryview(output)
        written = 0
        position = offset
        for file, start, partition_size in zip(
                self.files, self.starts, self.sizes):
            if position >= start + partition_size:
                continue
            if position < start:
                continue
            local_offset = position - start
            count = min(size - written, partition_size - local_offset)
            file.seek(local_offset)
            chunk = file.read(count)
            if len(chunk) != count:
                raise IOError(
                    f"Short read from {file.name}: expected {count}, "
                    f"received {len(chunk)}"
                )
            view[written:written + count] = chunk
            written += count
            position += count
            if written == size:
                return output
        raise IOError(f"Could only read {written}/{size} bytes at {offset}")

    def close(self) -> None:
        for file in self.files:
            file.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


def target_spec(model_path: Path,
                trust_remote_code: bool) -> Tuple[Dict[str, Tuple[
                    Tuple[int, ...], torch.dtype]], object]:
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    with init_empty_weights():
        try:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=trust_remote_code
            )
        except ValueError as causal_error:
            try:
                model = AutoModelForVision2Seq.from_config(
                    config, trust_remote_code=trust_remote_code
                )
            except ValueError:
                raise causal_error
    spec = {
        name: (tuple(tensor.shape), tensor.dtype)
        for name, tensor in model.state_dict().items()
    }
    del model
    return spec, config


def direct_output(source: SourceTensor, name: str,
                  target: Dict[str, Tuple[Tuple[int, ...], torch.dtype]]
                  ) -> OutputTensor:
    shape, dtype = target[name]
    if shape != source.shape:
        raise ValueError(
            f"Shape mismatch for {source.name} -> {name}: "
            f"source={source.shape}, target={shape}"
        )
    return OutputTensor(
        name=name,
        source_name=source.name,
        split_start=None,
        split_length=None,
        shape=shape,
        dtype=source.dtype,
    )


def split_outputs(
    source: SourceTensor,
    target_names: Sequence[str],
    target: Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
) -> List[OutputTensor]:
    if any(name not in target for name in target_names):
        return []
    shapes = [target[name][0] for name in target_names]
    dtypes = [source.dtype for _name in target_names]
    if any(len(shape) != len(source.shape) for shape in shapes):
        raise ValueError(f"Rank mismatch while splitting {source.name}")
    if any(shape[1:] != source.shape[1:] for shape in shapes):
        raise ValueError(f"Trailing shape mismatch while splitting {source.name}")
    lengths = [shape[0] for shape in shapes]
    if sum(lengths) != source.shape[0]:
        raise ValueError(
            f"Split sizes {lengths} do not cover {source.name} "
            f"shape {source.shape}"
        )
    outputs = []
    start = 0
    for name, shape, dtype, length in zip(
            target_names, shapes, dtypes, lengths):
        outputs.append(
            OutputTensor(
                name=name,
                source_name=source.name,
                split_start=start,
                split_length=length,
                shape=shape,
                dtype=dtype,
            )
        )
        start += length
    return outputs


def plan_outputs(
    sources: Sequence[SourceTensor],
    target: Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
    tie_word_embeddings: bool,
) -> Tuple[List[OutputTensor], List[str]]:
    outputs: List[OutputTensor] = []
    generated = set()
    for source in sources:
        candidate_names = [source.name]
        if source.name == "lm_head.weight":
            candidate_names.append("language_model.lm_head.weight")
        if source.name.startswith("language_model."):
            suffix = source.name[len("language_model."):]
            if suffix == "lm_head.weight":
                candidate_names.append("language_model.lm_head.weight")
            else:
                candidate_names.append(f"language_model.model.{suffix}")
        direct_name = next(
            (name for name in candidate_names if name in target), None
        )
        packed_name = candidate_names[-1]
        if direct_name is not None:
            items = [direct_output(source, direct_name, target)]
        elif source.name == "lm_head_weight":
            aliases = (
                ["model.embed_tokens.weight"]
                if tie_word_embeddings
                else ["lm_head.weight"]
            )
            items = [
                direct_output(source, name, target)
                for name in aliases if name in target
            ]
        elif ".qkv_proj." in source.name:
            items = split_outputs(
                source,
                [
                    packed_name.replace(".qkv_proj.", ".q_proj."),
                    packed_name.replace(".qkv_proj.", ".k_proj."),
                    packed_name.replace(".qkv_proj.", ".v_proj."),
                ],
                target,
            )
        elif ".gate_up_proj." in source.name:
            items = split_outputs(
                source,
                [
                    packed_name.replace(".gate_up_proj.", ".gate_proj."),
                    packed_name.replace(".gate_up_proj.", ".up_proj."),
                ],
                target,
            )
        else:
            items = []
        if not items:
            raise ValueError(
                f"Cannot map packed tensor {source.name} with shape "
                f"{source.shape} to the Hugging Face architecture"
            )
        for item in items:
            if item.name in generated:
                raise ValueError(f"Duplicate generated key {item.name}")
            generated.add(item.name)
            outputs.append(item)

    required = set(target)
    # Hugging Face ties these after loading; only one physical copy is needed.
    if tie_word_embeddings:
        required.discard("lm_head.weight")
    missing = sorted(required - generated)
    return outputs, missing


def tensor_from_storage(raw: bytearray, source: SourceTensor) -> torch.Tensor:
    element_size = torch.empty((), dtype=source.dtype).element_size()
    if source.storage_bytes % element_size:
        raise ValueError(
            f"{source.name} storage size {source.storage_bytes} is not "
            f"aligned to dtype size {element_size}"
        )
    flat = torch.frombuffer(
        raw, dtype=source.dtype, count=source.storage_bytes // element_size
    )
    required_elements = 1
    if source.shape:
        required_elements = 1 + sum(
            (dimension - 1) * stride
            for dimension, stride in zip(source.shape, source.stride)
            if dimension
        )
    if required_elements > flat.numel():
        raise ValueError(
            f"{source.name} metadata needs {required_elements} elements, "
            f"storage only has {flat.numel()}"
        )
    return torch.as_strided(
        flat, size=source.shape, stride=source.stride
    )


def copy_assets(model_path: Path, output_path: Path) -> None:
    excluded_names = {
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    for source in model_path.iterdir():
        if source.name in excluded_names or source.name == "rank_0":
            continue
        if source.suffix in {".safetensors", ".bin"}:
            continue
        target = output_path / source.name
        if source.is_file():
            shutil.copy2(source, target)
        elif source.is_dir():
            shutil.copytree(source, target)


def convert(args) -> dict:
    rank_path = args.rank_path.resolve()
    model_path = (
        args.model_path.resolve()
        if args.model_path is not None else rank_path.parent
    )
    output_path = args.output.resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output_path}. "
            "Use a new directory to avoid overwriting a checkpoint."
        )
    output_path.mkdir(parents=True, exist_ok=True)

    sources = load_source_tensors(rank_path)
    target, config = target_spec(model_path, args.trust_remote_code)
    outputs, missing = plan_outputs(
        sources, target, bool(getattr(config, "tie_word_embeddings", False))
    )
    if missing:
        raise ValueError(
            f"Packed checkpoint does not cover {len(missing)} required HF "
            f"keys: {missing[:16]}"
        )

    partitions = sorted(
        rank_path.glob("tensor.data_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    source_by_name = {item.name: item for item in sources}
    output_by_source: Dict[str, List[OutputTensor]] = {}
    for item in outputs:
        output_by_source.setdefault(item.source_name, []).append(item)

    max_shard_bytes = parse_size(args.max_shard_size)
    shard_paths: List[Path] = []
    weight_map = {}
    shard: Dict[str, torch.Tensor] = {}
    shard_bytes = 0
    converted_bytes = 0

    def flush_shard() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        path = output_path / f"model-{len(shard_paths) + 1:05d}.safetensors"
        save_file(shard, str(path), metadata={"format": "pt"})
        shard_paths.append(path)
        for name in shard:
            weight_map[name] = path.name
        shard = {}
        shard_bytes = 0

    with PartitionReader(partitions) as reader:
        for source in sources:
            raw = reader.read(source.global_offset, source.storage_bytes)
            tensor = tensor_from_storage(raw, source)
            for planned in output_by_source[source.name]:
                if planned.split_start is None:
                    value = tensor
                else:
                    value = tensor.narrow(
                        0, planned.split_start, planned.split_length
                    )
                value = value.contiguous()
                if shard and shard_bytes + planned.size_bytes > max_shard_bytes:
                    flush_shard()
                shard[planned.name] = value
                shard_bytes += planned.size_bytes
                converted_bytes += planned.size_bytes
            # Values stored in the active shard keep raw alive as necessary.
            if args.verbose:
                names = [item.name for item in output_by_source[source.name]]
                print(f"{source.name} -> {', '.join(names)}", flush=True)
        flush_shard()

    shard_count = len(shard_paths)
    final_weight_map = {}
    for index, old_path in enumerate(shard_paths, 1):
        new_name = (
            f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        )
        new_path = old_path.with_name(new_name)
        os.rename(old_path, new_path)
        for name, shard_name in weight_map.items():
            if shard_name == old_path.name:
                final_weight_map[name] = new_name

    index = {
        "metadata": {"total_size": converted_bytes},
        "weight_map": final_weight_map,
    }
    with (output_path / "model.safetensors.index.json").open("w") as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
    copy_assets(model_path, output_path)
    summary = {
        "source": str(rank_path),
        "model_config": str(model_path),
        "output": str(output_path),
        "source_tensors": len(sources),
        "hf_tensors": len(outputs),
        "shards": shard_count,
        "total_size": converted_bytes,
    }
    with (output_path / "conversion_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    return summary


def dry_run(args) -> dict:
    rank_path = args.rank_path.resolve()
    model_path = (
        args.model_path.resolve()
        if args.model_path is not None else rank_path.parent
    )
    sources = load_source_tensors(rank_path)
    target, config = target_spec(model_path, args.trust_remote_code)
    outputs, missing = plan_outputs(
        sources, target, bool(getattr(config, "tie_word_embeddings", False))
    )
    if missing:
        raise ValueError(
            f"Packed checkpoint does not cover {len(missing)} required HF "
            f"keys: {missing[:16]}"
        )
    packed_size = sum(
        path.stat().st_size for path in rank_path.glob("tensor.data_*")
    )
    return {
        "source": str(rank_path),
        "model_config": str(model_path),
        "source_tensors": len(sources),
        "hf_tensors": len(outputs),
        "packed_size": packed_size,
        "converted_size": sum(item.size_bytes for item in outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rank_path", type=Path,
        help="Directory containing tensor.data_*, tensor_group_index.txt, "
             "and tensor_meta_index.json.",
    )
    parser.add_argument(
        "--model-path", type=Path,
        help="Directory containing config.json/tokenizer files. Defaults to "
             "the parent of rank_path.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-shard-size", default="2GiB")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        summary = dry_run(args)
    else:
        if args.output is None:
            parser.error("--output is required unless --dry-run is used")
        summary = convert(args)
    print("RANK0_CONVERSION=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
