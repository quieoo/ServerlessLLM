"""Dependency-free summary statistics for simulator output."""

from __future__ import annotations

import math
import statistics


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0.0
    position = fraction * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return (
        values[lower] * (upper - position)
        + values[upper] * (position - lower)
    )


def summary(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0.0),
    }
