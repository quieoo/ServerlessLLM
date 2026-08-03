"""Trace, configuration, and legacy compute-profile I/O helpers."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Request:
    request_id: int
    model_id: int
    input_tokens: int
    output_tokens: int
    trace_output_tokens: int = 1


class ComputeProfile:
    """Minimal profile container consumed by compute_durations()."""

    def __init__(self):
        self.models = {}


def load_profile(path: Path) -> ComputeProfile:
    document = json.loads(path.read_text())
    profile_set = ComputeProfile()
    for model_id, profile in document["profiles"].items():
        profile = copy.deepcopy(profile)
        profile["compute"] = {
            int(layer): value
            for layer, value in profile["compute"].items()
        }
        profile_set.models[int(model_id)] = profile
    return profile_set


def load_model_limits(
        path: Path, input_limit_policy: str = "safe") -> dict[int, int]:
    document = json.loads(path.read_text())
    limits = {}
    for item in document["model_lists"]:
        model_id = int(item["id"])
        context = int(item["l40_memory_budget"]["model_context_tokens"])
        if input_limit_policy == "safe":
            limit = int(item.get("l40_safe_max_input_length") or context)
        elif input_limit_policy == "context":
            limit = context
        elif input_limit_policy == "none":
            # load_trace still enforces simulated pool/KV feasibility.
            limit = 2**31 - 1
        else:
            raise ValueError(
                f"unknown input limit policy: {input_limit_policy}")
        limits[model_id] = limit
    return limits


def load_trace(
    path: Path,
    layouts,
    max_requests: int,
    input_scale: float,
    output_tokens: int,
    pool_pages: int,
    kv_block_tokens: int,
    trace_mode: str = "all",
) -> list[Request]:
    """Load and normalize the four-column ServeGen trace format."""
    requests = []
    source_requests = 0
    previous_model_id = None
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            model_id = int(fields[1])
            source_requests += 1
            keep = not (
                trace_mode == "model-switches"
                and previous_model_id == model_id
            )
            previous_model_id = model_id
            if not keep:
                if max_requests > 0 and source_requests >= max_requests:
                    break
                continue
            scaled = max(
                1, int(math.floor(int(fields[2]) * input_scale + 0.5)))
            trace_output_tokens = int(fields[3])
            scaled = min(scaled, layouts[model_id].max_input_tokens)
            scaled = min(
                scaled,
                max(
                    1,
                    (pool_pages - layouts[model_id].page_count)
                    * kv_block_tokens - max(1, output_tokens),
                ),
            )
            requests.append(Request(
                request_id=len(requests),
                model_id=model_id,
                input_tokens=scaled,
                output_tokens=(
                    output_tokens
                    if output_tokens > 0 else trace_output_tokens),
                trace_output_tokens=trace_output_tokens,
            ))
            if max_requests > 0 and source_requests >= max_requests:
                break
    return requests
