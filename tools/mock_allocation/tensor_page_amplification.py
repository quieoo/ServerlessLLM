#!/usr/bin/env python3
"""Estimate fixed-page fragmentation for checkpoint groups or tensors."""

import argparse
import json
import math
import re
from pathlib import Path


DTYPE_BYTES = {
    "torch.bool": 1,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e4m3fnuz": 1,
    "torch.float8_e5m2": 1,
    "torch.float8_e5m2fnuz": 1,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int16": 2,
    "torch.uint16": 2,
    "torch.float32": 4,
    "torch.int32": 4,
    "torch.uint32": 4,
    "torch.complex32": 4,
    "torch.float64": 8,
    "torch.int64": 8,
    "torch.uint64": 8,
    "torch.complex64": 8,
    "torch.complex128": 16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate fixed-page space usage from checkpoint tensor groups. "
            "Use -tensor-only to split groups and page-align every tensor "
            "independently."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Model configuration containing a model_lists array.",
    )
    parser.add_argument(
        "--page-sizes-mib",
        type=int,
        nargs="+",
        required=True,
        help="One or more positive fixed page sizes in MiB.",
    )
    parser.add_argument(
        "--json-output",
        help="Optionally write the complete machine-readable result as JSON.",
    )
    parser.add_argument(
        "-tensor-only",
        "--tensor-only",
        action="store_true",
        help=(
            "Use each tensor as an independent allocation unit. Every tensor "
            "is rounded to whole pages, so different tensors never share a "
            "page."
        ),
    )
    parser.add_argument(
        "--stable-model-layout",
        action="store_true",
        help=(
            "Match the current VMM backend: pack units at 256-byte alignment "
            "in one stable per-model VA arena and round only the complete "
            "arena to physical pages."
        ),
    )
    parser.add_argument(
        "--merge-tensor-groups",
        type=int,
        nargs="?",
        const=40,
        metavar="TARGET_COUNT",
        help=(
            "Read tensor_group_index.txt and reproduce MergeTGsRatio. With no "
            "value, use the Allocateion default target of 40 groups; an "
            "explicit positive target count may also be supplied."
        ),
    )
    args = parser.parse_args()
    if any(size <= 0 for size in args.page_sizes_mib):
        parser.error("--page-sizes-mib values must be positive")
    if args.merge_tensor_groups is not None and args.merge_tensor_groups <= 0:
        parser.error("--merge-tensor-groups TARGET_COUNT must be positive")
    if args.tensor_only and args.merge_tensor_groups is not None:
        parser.error(
            "-tensor-only/--tensor-only cannot be combined with "
            "--merge-tensor-groups")
    args.page_sizes_mib = list(dict.fromkeys(args.page_sizes_mib))
    return args


def resolve_model_path(raw_path, config_path):
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def tensor_bytes(meta_path):
    with meta_path.open() as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(f"empty or invalid tensor metadata: {meta_path}")

    sizes = []
    for name, entry in metadata.items():
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(f"invalid metadata for tensor {name!r}: {entry!r}")
        shape, _stride, dtype = entry
        if dtype not in DTYPE_BYTES:
            raise ValueError(
                f"unsupported dtype {dtype!r} for tensor {name!r} in "
                f"{meta_path}"
            )
        if not isinstance(shape, (list, tuple)):
            raise ValueError(f"invalid shape for tensor {name!r}: {shape!r}")
        elements = math.prod(int(dimension) for dimension in shape)
        sizes.append(elements * DTYPE_BYTES[dtype])
    return sizes


def tensor_group_bytes(index_path):
    sizes = []
    pattern = re.compile(r"^Group Size:\s*(\d+)\s*$")
    with index_path.open() as stream:
        for line in stream:
            match = pattern.match(line)
            if match:
                sizes.append(int(match.group(1)))
    if not sizes:
        raise ValueError(f"no tensor groups found in {index_path}")
    return sizes


def merge_tensor_groups_ratio(sizes, model_size, target_count):
    """Reproduce RegisteredModel::MergeTGsRatio/MergeTGs size behavior."""
    minimum_size = model_size // target_count
    merged_sizes = list(sizes)
    while True:
        merged = False
        index = 0
        while index < len(merged_sizes):
            if merged_sizes[index] >= minimum_size:
                index += 1
                continue
            merge_end = index + 1
            if merge_end >= len(merged_sizes):
                break
            total_size = merged_sizes[index]
            while merge_end < len(merged_sizes) and total_size < minimum_size:
                total_size += merged_sizes[merge_end]
                merge_end += 1
            merged_sizes[index:merge_end] = [total_size]
            merged = True
        if not merged:
            break
    if len(merged_sizes) > 1 and merged_sizes[-1] < minimum_size:
        merged_sizes[-2] += merged_sizes[-1]
        merged_sizes.pop()
    return merged_sizes


def gib(value):
    return value / 1024**3


def analyze_model(model, config_path, page_sizes_mib, merge_target,
                  tensor_only, stable_model_layout):
    model_path = resolve_model_path(model["path"], config_path)
    meta_path = model_path / "tensor_meta_index.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"tensor metadata not found: {meta_path}")

    tensors = tensor_bytes(meta_path)
    logical_bytes = sum(tensors)
    group_path = model_path / "tensor_group_index.txt"
    if not group_path.is_file():
        raise FileNotFoundError(f"tensor group index not found: {group_path}")
    groups = tensor_group_bytes(group_path)
    original_group_count = len(groups)
    merged_group_count = None
    allocation_units = tensors if tensor_only else groups
    if merge_target is not None:
        allocation_units = merge_tensor_groups_ratio(
            groups, logical_bytes, merge_target)
        merged_group_count = len(allocation_units)
    pages = {}
    for page_mib in page_sizes_mib:
        page_bytes = page_mib * 1024**2
        if stable_model_layout:
            packed_bytes = 0
            for size in allocation_units:
                packed_bytes = (packed_bytes + 255) & ~255
                packed_bytes += size
            page_count = (packed_bytes + page_bytes - 1) // page_bytes
        else:
            page_count = sum((size + page_bytes - 1) // page_bytes
                             for size in allocation_units)
        mapped_bytes = page_count * page_bytes
        pages[str(page_mib)] = {
            "page_count": page_count,
            "mapped_bytes": mapped_bytes,
            "wasted_bytes": mapped_bytes - logical_bytes,
            "amplification": mapped_bytes / logical_bytes,
            "overhead_percent": (
                (mapped_bytes - logical_bytes) / logical_bytes * 100.0
            ),
        }
    return {
        "id": int(model["id"]),
        "path": str(model_path),
        "tensor_count": len(tensors),
        "original_tensor_group_count": original_group_count,
        "merged_tensor_group_count": merged_group_count,
        "allocation_unit_count": len(allocation_units),
        "logical_bytes": logical_bytes,
        "pages": pages,
    }


def print_results(models, page_sizes_mib, merge_target, tensor_only,
                  stable_model_layout):
    if stable_model_layout:
        print(
            "Semantics: allocation units are packed at 256-byte alignment "
            "inside one stable model VA arena; only the arena tail is rounded "
            "to a physical page."
        )
    elif tensor_only:
        print(
            "Semantics: every tensor owns whole pages exclusively; "
            "different tensors never share a page."
        )
    elif merge_target is None:
        print(
            "Semantics: checkpoint tensor groups are allocation units; "
            "tensors in the same group may share pages."
        )
    else:
        print(
            "Semantics: tensor_group_index.txt groups are merged with "
            f"MergeTGsRatio(target_count={merge_target}) for group-count "
            "allocation; page usage is rounded per merged group."
        )
    for page_mib in page_sizes_mib:
        key = str(page_mib)
        print(f"\nPage size: {page_mib} MiB")
        # print(
        #     f"{'ID':>3}  {'Tensors':>7}  {'Logical GiB':>11}  "
        #     f"{'Mapped GiB':>10}  {'Waste GiB':>9}  "
        #     f"{'Amplification':>13}  Model"
        # )
        print(
            f"{'ID':>3} "
            f"{'Amplification':>13}  Model"
        )
        total_logical = 0
        total_mapped = 0
        for model in models:
            result = model["pages"][key]
            total_logical += model["logical_bytes"]
            total_mapped += result["mapped_bytes"]
            print(
                f"{model['id']:>3} "
                # f"{gib(model['logical_bytes']):>11.3f}  "
                # f"{gib(result['mapped_bytes']):>10.3f}  "
                # f"{gib(result['wasted_bytes']):>9.3f}  "
                f"{result['amplification']:>12.4f}  "
                f"{model['path']}"
            )
        amplification = total_mapped / total_logical
        # print(
        #     f"{'ALL':>3}  {sum(m['tensor_count'] for m in models):>7}  "
        #     f"{gib(total_logical):>11.3f}  {gib(total_mapped):>10.3f}  "
        #     f"{gib(total_mapped - total_logical):>9.3f}  "
        #     f"{amplification:>12.4f}  all models"
        # )
        print(
            f"{'ALL':>3} "
            f"{amplification:>12.4f}  all models"
        )

def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open() as stream:
        config = json.load(stream)
    model_list = config.get("model_lists")
    if not isinstance(model_list, list) or not model_list:
        raise ValueError(f"model_lists is empty or missing in {config_path}")

    models = [
        analyze_model(
            model, config_path, args.page_sizes_mib,
            args.merge_tensor_groups, args.tensor_only,
            args.stable_model_layout)
        for model in model_list
    ]
    print_results(
        models, args.page_sizes_mib, args.merge_tensor_groups,
        args.tensor_only, args.stable_model_layout)

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output = {
            "config": str(config_path),
            "semantics": (
                "stable_model_va_arena"
                if args.stable_model_layout
                else ("round_each_tensor_exclusively"
                      if args.tensor_only
                      else ("round_each_merged_tensor_group"
                            if args.merge_tensor_groups is not None
                            else "round_each_checkpoint_tensor_group"))
            ),
            "tensor_only": args.tensor_only,
            "stable_model_layout": args.stable_model_layout,
            "merge_tensor_groups_target_count": args.merge_tensor_groups,
            "page_sizes_mib": args.page_sizes_mib,
            "models": models,
        }
        with output_path.open("w") as stream:
            json.dump(output, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"\nJSON written to {output_path}")


if __name__ == "__main__":
    main()
