"""Compute-model features used by the Tensor-level simulator."""

from __future__ import annotations

import math


def batch_features(batch):
    """Return the scaled prefill features used by layer regressions."""
    batch_size = max(1, int(batch.get("batch_size", 1)))
    total_tokens = max(1, int(batch["input_tokens"]))
    max_tokens = max(
        1,
        int(batch.get(
            "max_input_tokens", math.ceil(total_tokens / batch_size))),
    )
    sum_squared = float(batch.get(
        "sum_input_tokens_squared",
        batch_size * max_tokens * max_tokens,
    ))
    return [
        1.0,
        float(batch_size - 1),
        total_tokens / 1000.0,
        max_tokens / 1000.0,
        sum_squared / 1_000_000.0,
    ]
