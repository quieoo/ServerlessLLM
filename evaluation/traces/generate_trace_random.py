#!/usr/bin/env python3
"""Generate a fully random multi-model trace.

This script keeps the same command-line inputs and output columns as
generate_trace_tangram.py, but it does not depend on ServeGen. It only reads the
model IDs from the config file, then generates a random request stream.

Output format:
    timestamp model_id input_tokens output_tokens
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[2]

DEFAULT_CONFIG = WORKSPACE_ROOT / "configs" / "servegen_8_models.json"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "evaluation" / "traces" / "servegen_random.trace"
DEFAULT_DATA_DIR = THIS_FILE.parents[1] / "data"
DEFAULT_RPS = 100.0


@dataclass(frozen=True)
class ModelConfig:
    model_id: int
    pattern: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a random experimental trace with columns: "
            "timestamp model_id input_tokens output_tokens."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Accepted for CLI compatibility with generate_trace_tangram.py; unused.",
    )
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument(
        "--start-time",
        type=int,
        default=0,
        help="Accepted for CLI compatibility; generated timestamps remain relative to 0.",
    )
    parser.add_argument(
        "--target-rps",
        type=float,
        default=None,
        help=(
            "Aggregate average requests per second. "
            f"Defaults to {DEFAULT_RPS:g} because this standalone random generator "
            "does not read ServeGen rates."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skew-alpha",
        type=float,
        default=0.0,
        help="Accepted for CLI compatibility with generate_trace_tangram.py; unused.",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Write a header line before the trace rows.",
    )
    parser.add_argument(
        "--delimiter",
        default=" ",
        help="Column delimiter. Defaults to one space.",
    )
    return parser.parse_args()


def load_model_configs(config_path: Path) -> List[ModelConfig]:
    with config_path.open() as f:
        raw = json.load(f)

    models = raw.get("model_lists")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{config_path} must contain a non-empty 'model_lists' array")

    result: List[ModelConfig] = []
    for item in models:
        if "id" not in item:
            raise ValueError(f"Model entry is missing 'id': {item}")
        result.append(
            ModelConfig(
                model_id=int(item["id"]),
                pattern=str(item.get("pattern", "unknown")),
            )
        )
    return result


def sample_request_count(duration: int, target_rps: float, rng: random.Random) -> int:
    """Sample a Poisson request count using Knuth for small means and Gaussian for large ones."""
    expected = duration * target_rps
    if expected <= 0:
        return 0
    if expected < 1000:
        threshold = pow(2.718281828459045, -expected)
        product = 1.0
        count = 0
        while product > threshold:
            count += 1
            product *= rng.random()
        return count - 1

    return max(0, round(rng.gauss(expected, expected**0.5)))


def sample_input_tokens(rng: random.Random) -> int:
    """Draw a broad positive token length without any external dataset."""
    value = int(rng.lognormvariate(7.3, 1.0))
    return max(1, min(value, 131072))


def sample_output_tokens(rng: random.Random) -> int:
    """Draw a broad positive generation length without any external dataset."""
    value = int(rng.lognormvariate(5.8, 0.9))
    return max(1, min(value, 32768))


def generate_requests(
    model_ids: Sequence[int],
    duration: int,
    target_rps: float,
    seed: int,
) -> List[Tuple[float, int, int, int]]:
    if not model_ids:
        raise ValueError("No model IDs found in config")

    rng = random.Random(seed)
    request_count = sample_request_count(duration, target_rps, rng)
    rows: List[Tuple[float, int, int, int]] = []

    for _ in range(request_count):
        rows.append(
            (
                rng.random() * duration,
                rng.choice(model_ids),
                sample_input_tokens(rng),
                sample_output_tokens(rng),
            )
        )

    rows.sort(key=lambda row: (row[0], row[1]))
    return rows


def write_trace(
    rows: Sequence[Tuple[float, int, int, int]],
    output_path: Path,
    delimiter: str,
    header: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        if header:
            f.write(delimiter.join(["timestamp", "model_id", "input_tokens", "output_tokens"]) + "\n")
        for timestamp, model_id, input_len, output_len in rows:
            f.write(
                delimiter.join(
                    [
                        f"{timestamp:.6f}",
                        str(model_id),
                        str(input_len),
                        str(output_len),
                    ]
                )
                + "\n"
            )


def print_summary(
    rows: Sequence[Tuple[float, int, int, int]],
    models: Sequence[ModelConfig],
    duration: int,
    target_rps: float,
    output_path: Path,
) -> None:
    actual_rps = len(rows) / duration if duration > 0 else 0.0
    model_ids = sorted(model.model_id for model in models)
    counts = {model_id: 0 for model_id in model_ids}
    for _, model_id, _, _ in rows:
        counts[model_id] = counts.get(model_id, 0) + 1

    print(f"Wrote {len(rows)} requests to {output_path}")
    print(f"Target average RPS: {target_rps:.4f}")
    print(f"Actual average RPS: {actual_rps:.4f}")
    print("Random model assignment:")
    print(f"  model_ids={model_ids}")
    print(f"  per-request probability={1.0 / len(model_ids):.6f}")
    print("Actual per-model request counts:")
    for model_id in model_ids:
        share = counts[model_id] / len(rows) if rows else 0.0
        print(f"  model_id={model_id} count={counts[model_id]} share={share:.4f}")


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    target_rps: Optional[float] = args.target_rps
    if target_rps is None:
        target_rps = DEFAULT_RPS
    if target_rps <= 0:
        raise ValueError("--target-rps must be positive")

    models = load_model_configs(args.config)
    model_ids = sorted(model.model_id for model in models)
    rows = generate_requests(model_ids, args.duration, target_rps, args.seed)
    write_trace(rows, args.output, args.delimiter, args.header)
    print_summary(rows, models, args.duration, target_rps, args.output)


if __name__ == "__main__":
    main()
