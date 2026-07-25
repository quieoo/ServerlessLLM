#!/usr/bin/env python3
"""Generate per-model safe input budgets for the L40 LayerPipe config."""

import argparse
import glob
import json
import math
import os
from pathlib import Path


def text_config(config: dict) -> dict:
    value = dict(config.get("text_config") or config)
    if config.get("model_type") == "llava":
        # The converted LLaVA config omits the standard Vicuna-7B dimensions.
        value.setdefault("num_hidden_layers", 32)
        value.setdefault("hidden_size", 4096)
        value.setdefault("num_attention_heads", 32)
        value.setdefault("num_key_value_heads", 32)
    return value


def dimensions(config: dict) -> tuple:
    value = text_config(config)
    layers = value.get("num_hidden_layers", value.get("n_layer"))
    hidden = value.get("hidden_size", value.get("n_embd"))
    intermediate = value.get(
        "intermediate_size", value.get("n_inner", 4 * hidden if hidden else 0)
    )
    heads = value.get("num_attention_heads", value.get("n_head"))
    kv_heads = value.get("num_key_value_heads", heads)
    head_dim = value.get("head_dim")
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads
    context = value.get(
        "max_position_embeddings", value.get("n_positions", 0)
    )
    if not all((
        layers, hidden, intermediate, heads, kv_heads, head_dim, context
    )):
        raise ValueError(f"Incomplete model dimensions: {value}")
    return layers, hidden, intermediate, kv_heads, head_dim, context


def packed_size(rank_path: Path) -> int:
    files = glob.glob(str(rank_path / "tensor.data_*"))
    if not files:
        raise ValueError(f"No packed tensors under {rank_path}")
    return sum(os.path.getsize(path) for path in files)


def generate(config_path: Path, pool_gib: float, reserve_gib: float,
             block_size: int, layout_factor: float, gpu_memory_gib: float,
             runtime_reserve_gib: float,
             prefill_activation_safety_factor: float) -> dict:
    with config_path.open() as stream:
        config = json.load(stream)
    pool_bytes = int(pool_gib * 1024**3)
    reserve_bytes = int(reserve_gib * 1024**3)
    gpu_memory_bytes = int(gpu_memory_gib * 1024**3)
    runtime_reserve_bytes = int(runtime_reserve_gib * 1024**3)
    if min(
        pool_gib, reserve_gib, gpu_memory_gib, runtime_reserve_gib,
        prefill_activation_safety_factor,
    ) < 0:
        raise ValueError("memory budgets and safety factor must be non-negative")
    if prefill_activation_safety_factor == 0:
        raise ValueError("prefill activation safety factor must be positive")
    if reserve_bytes > pool_bytes:
        raise ValueError("reserve_gib cannot exceed pool_gib")
    if pool_bytes > gpu_memory_bytes:
        raise ValueError(
            f"pool_gib={pool_gib} exceeds gpu_memory_gib={gpu_memory_gib}"
        )
    for item in config["model_lists"]:
        rank_path = Path(item["packed_rank_path"]).resolve()
        if rank_path.name != "rank_0":
            rank_path /= "rank_0"
        model_path = rank_path.parent
        with (model_path / "config.json").open() as stream:
            model_config = json.load(stream)
        (layers, hidden, intermediate, kv_heads, head_dim,
         context) = dimensions(model_config)
        weight_bytes = packed_size(rank_path)
        # K and V, FP16, all layers. layout_factor matches the current
        # segmented ODKV provider's observed physical block accounting.
        kv_bytes_per_token = math.ceil(
            2 * layers * kv_heads * head_dim * 2 * layout_factor
        )
        kv_block_bytes = kv_bytes_per_token * block_size
        kv_available = max(0, pool_bytes - reserve_bytes - weight_bytes)
        safe_blocks = kv_available // kv_block_bytes
        kv_token_budget = int(safe_blocks * block_size)

        # During a SwiGLU/GeGLU MLP, gate_up (2 * intermediate), the
        # activated output (intermediate), and hidden/residual-sized tensors
        # can be live at the same time. This memory is allocated by CUDA,
        # outside the ODKV page accounting. The safety factor covers attention
        # workspaces, allocator fragmentation, and implementation differences.
        activation_bytes_per_token = math.ceil(
            (3 * intermediate + 4 * hidden) * 2
            * prefill_activation_safety_factor
        )
        # reserve_gib is deliberately left uncommitted inside the VMM pool, so
        # it also remains physically available to runtime CUDA allocations.
        runtime_available = max(
            0,
            gpu_memory_bytes - pool_bytes + reserve_bytes
            - runtime_reserve_bytes,
        )
        activation_token_budget = (
            runtime_available // activation_bytes_per_token
        )
        batch_budget = int(min(kv_token_budget, activation_token_budget))
        input_limit = min(max(1, context - 1), max(1, batch_budget))
        item["l40_safe_max_input_length"] = input_limit
        item["l40_safe_max_batch_input_tokens"] = batch_budget
        item["l40_memory_budget"] = {
            "gpu_memory_gib": gpu_memory_gib,
            "pool_gib": pool_gib,
            "reserve_gib": reserve_gib,
            "runtime_reserve_gib": runtime_reserve_gib,
            "packed_weight_bytes": weight_bytes,
            "odkv_bytes_per_token": kv_bytes_per_token,
            "odkv_block_size_tokens": block_size,
            "kv_token_budget": kv_token_budget,
            "prefill_activation_bytes_per_token":
                activation_bytes_per_token,
            "prefill_activation_safety_factor":
                prefill_activation_safety_factor,
            "prefill_activation_token_budget":
                activation_token_budget,
            "model_context_tokens": context,
        }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pool-gib", type=float, default=41.5)
    parser.add_argument("--reserve-gib", type=float, default=0.5)
    parser.add_argument(
        "--gpu-memory-gib", type=float, default=44.98,
        help="Usable physical GPU memory; this L40 reports 46068 MiB",
    )
    parser.add_argument(
        "--runtime-reserve-gib", type=float, default=1.0,
        help="Fixed CUDA/vLLM headroom excluded from Prefill activations",
    )
    parser.add_argument(
        "--prefill-activation-safety-factor", type=float, default=2.0,
        help="Multiplier for MLP live tensors and CUDA workspace/fragmentation",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--odkv-layout-factor", type=float, default=2.0)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        help="Write to a new config path instead of replacing --config",
    )
    args = parser.parse_args()
    result = generate(
        args.config, args.pool_gib, args.reserve_gib,
        args.block_size, args.odkv_layout_factor, args.gpu_memory_gib,
        args.runtime_reserve_gib,
        args.prefill_activation_safety_factor,
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.write or args.output:
        output_path = args.output or args.config
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
        print(f"updated {output_path}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
