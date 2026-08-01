#!/usr/bin/env python3
"""Generate deterministic traces with controlled model locality and tokens.

This generator is intentionally independent of ServeGen.  It can reuse each
model's empirical token-shape pool while controlling the model-ID sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def parse_ids(raw):
    return [int(value) for value in raw.replace(",", " ").split()]


def parse_weights(raw, model_ids):
    if raw is None:
        return {model_id: 1.0 for model_id in model_ids}
    result = {model_id: 0.0 for model_id in model_ids}
    for item in raw.replace(",", " ").split():
        model_id, weight = item.split(":", 1)
        result[int(model_id)] = float(weight)
    if any(value < 0 for value in result.values()) or sum(result.values()) <= 0:
        raise ValueError("model weights must be nonnegative with positive sum")
    return result


def read_shapes(path):
    shapes = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            shapes[int(fields[1])].append(
                (int(fields[2]), int(fields[3])))
    if not shapes:
        raise ValueError(f"no requests in {path}")
    return dict(shapes)


def read_source_model_sequence(path, source_requests):
    sequence = []
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            sequence.append(int(fields[1]))
            if source_requests > 0 and len(sequence) >= source_requests:
                break
    if not sequence:
        raise ValueError(f"no requests in {path}")
    return sequence


def load_model_ids(path):
    document = json.loads(path.read_text())
    return sorted(int(item["id"]) for item in document["model_lists"])


def weighted_choice_no_repeat(rng, model_ids, weights, previous):
    choices = [
        model_id for model_id in model_ids
        if model_id != previous and weights.get(model_id, 0.0) > 0
    ]
    if not choices:
        raise ValueError("weights cannot avoid a consecutive model repeat")
    return rng.choices(
        choices, weights=[weights[model_id] for model_id in choices], k=1)[0]


def period_sequence(
        count, period, target_ids, background_ids, background_weights, rng):
    if period < len(target_ids):
        raise ValueError("target period must be >= number of target models")
    target_positions = {
        round(index * period / len(target_ids)): model_id
        for index, model_id in enumerate(target_ids)
    }
    sequence = []
    for index in range(count):
        within = index % period
        model_id = target_positions.get(within)
        if model_id is None:
            model_id = weighted_choice_no_repeat(
                rng, background_ids, background_weights,
                sequence[-1] if sequence else None)
        if sequence and model_id == sequence[-1]:
            model_id = weighted_choice_no_repeat(
                rng, background_ids, background_weights, sequence[-1])
        sequence.append(model_id)
    return sequence


def cyclic_sequence(count, model_ids, rng):
    sequence = []
    while len(sequence) < count:
        block = list(model_ids)
        rng.shuffle(block)
        if sequence and block[0] == sequence[-1]:
            swap = next(
                index for index, value in enumerate(block)
                if value != sequence[-1])
            block[0], block[swap] = block[swap], block[0]
        sequence.extend(block)
    return sequence[:count]


def iid_sequence(count, model_ids, weights, rng):
    sequence = []
    while len(sequence) < count:
        sequence.append(weighted_choice_no_repeat(
            rng, model_ids, weights, sequence[-1] if sequence else None))
    return sequence


def largest_remainder_counts(source_counts, model_ids, count):
    total = sum(source_counts[model_id] for model_id in model_ids)
    if total <= 0:
        raise ValueError("source model counts must be positive")
    exact = {
        model_id: source_counts[model_id] * count / total
        for model_id in model_ids
    }
    result = {model_id: int(exact[model_id]) for model_id in model_ids}
    remaining = count - sum(result.values())
    order = sorted(
        model_ids,
        key=lambda model_id: (
            exact[model_id] - result[model_id], -model_id),
        reverse=True)
    for model_id in order[:remaining]:
        result[model_id] += 1
    return result


def model_switch_sequence(sequence):
    result = []
    for model_id in sequence:
        if not result or result[-1] != model_id:
            result.append(model_id)
    return result


def balanced_count_sequence(counts, rng):
    remaining = Counter(counts)
    sequence = []
    while sum(remaining.values()):
        candidates = [
            model_id for model_id, value in remaining.items()
            if value > 0 and (not sequence or model_id != sequence[-1])
        ]
        if not candidates:
            raise ValueError(
                "model counts cannot be arranged without consecutive repeats")
        maximum = max(remaining[model_id] for model_id in candidates)
        maximum_models = [
            model_id for model_id in candidates
            if remaining[model_id] == maximum
        ]
        model_id = rng.choice(maximum_models)
        sequence.append(model_id)
        remaining[model_id] -= 1
    return sequence


def max_distance_tail(prefix, counts, rng):
    """Complete a fixed prefix by choosing the least recently used model."""
    remaining = Counter(counts)
    for model_id in prefix:
        remaining[model_id] -= 1
    sequence = list(prefix)
    last_seen = {
        model_id: index for index, model_id in enumerate(sequence)
    }
    while sum(remaining.values()):
        candidates = [
            model_id for model_id, value in remaining.items()
            if value > 0 and (not sequence or model_id != sequence[-1])
        ]
        if not candidates:
            raise ValueError(
                "model counts cannot be arranged without consecutive repeats")
        oldest = min(last_seen.get(model_id, -1) for model_id in candidates)
        candidates = [
            model_id for model_id in candidates
            if last_seen.get(model_id, -1) == oldest
        ]
        maximum = max(remaining[model_id] for model_id in candidates)
        candidates = [
            model_id for model_id in candidates
            if remaining[model_id] == maximum
        ]
        model_id = rng.choice(candidates)
        sequence.append(model_id)
        remaining[model_id] -= 1
        last_seen[model_id] = len(sequence) - 1
    return sequence


def min_distance_tail(prefix, counts, rng):
    """Complete a fixed prefix in alternating two-model locality chunks."""
    remaining = Counter(counts)
    for model_id in prefix:
        remaining[model_id] -= 1
    sequence = list(prefix)
    while sum(remaining.values()):
        available = [
            model_id for model_id, value in remaining.items() if value > 0
        ]
        if len(available) == 1:
            model_id = available[0]
            if sequence and sequence[-1] == model_id:
                raise ValueError(
                    "low-distance tail cannot avoid a consecutive repeat")
            sequence.append(model_id)
            remaining[model_id] -= 1
            continue
        maximum = max(remaining[model_id] for model_id in available)
        first_choices = [
            model_id for model_id in available
            if remaining[model_id] == maximum
            and (not sequence or model_id != sequence[-1])
        ]
        if not first_choices:
            first_choices = [
                model_id for model_id in available
                if not sequence or model_id != sequence[-1]
            ]
        first = rng.choice(first_choices)
        second_choices = [
            model_id for model_id in available if model_id != first
        ]
        second_maximum = max(
            remaining[model_id] for model_id in second_choices)
        second = rng.choice([
            model_id for model_id in second_choices
            if remaining[model_id] == second_maximum
        ])
        pair_count = min(remaining[first], remaining[second])
        for _ in range(pair_count):
            for model_id in (first, second):
                if sequence and sequence[-1] == model_id:
                    first, second = second, first
                    model_id = first
                sequence.append(model_id)
                remaining[model_id] -= 1
    return sequence


def sequence_metrics(sequence, lookahead_k=32):
    last_index = {}
    last_seen = {}
    distinct = []
    request_gaps = []
    by_model_distinct = defaultdict(list)
    by_model_gaps = defaultdict(list)
    for index, model_id in enumerate(sequence):
        previous_index = last_index.get(model_id)
        previous_seen = last_seen.get(model_id, -1)
        if previous_index is not None:
            gap = index - previous_index
            distance = sum(
                int(other != model_id and seen > previous_seen)
                for other, seen in last_seen.items())
            distinct.append(distance)
            request_gaps.append(gap)
            by_model_distinct[model_id].append(distance)
            by_model_gaps[model_id].append(gap)
        last_index[model_id] = index
        last_seen[model_id] = index
    counts = Counter(sequence)
    transitions = Counter(zip(sequence, sequence[1:]))
    transition_total = max(1, len(sequence) - 1)
    transition_entropy = -sum(
        (value / transition_total) * math.log2(value / transition_total)
        for value in transitions.values())
    return {
        "requests": len(sequence),
        "counts": dict(sorted(counts.items())),
        "model_shares": {
            model_id: counts[model_id] / len(sequence)
            for model_id in sorted(counts)
        },
        "model_switch_rate": (
            sum(a != b for a, b in zip(sequence, sequence[1:]))
            / transition_total),
        "distinct_mean": statistics.fmean(distinct) if distinct else 0.0,
        "distinct_p50": percentile(distinct, .50),
        "distinct_p90": percentile(distinct, .90),
        "distinct_p95": percentile(distinct, .95),
        "distinct_p99": percentile(distinct, .99),
        "distinct_histogram": dict(sorted(Counter(distinct).items())),
        "request_gap_mean":
            statistics.fmean(request_gaps) if request_gaps else 0.0,
        "request_gap_p50": percentile(request_gaps, .50),
        "request_gap_p90": percentile(request_gaps, .90),
        "request_gap_p95": percentile(request_gaps, .95),
        "request_gap_p99": percentile(request_gaps, .99),
        "request_gap_within_k_fraction": (
            sum(gap <= lookahead_k for gap in request_gaps)
            / len(request_gaps) if request_gaps else 0.0),
        "by_model_distinct_mean": {
            model_id: statistics.fmean(values)
            for model_id, values in sorted(by_model_distinct.items())
        },
        "by_model_request_gap_mean": {
            model_id: statistics.fmean(values)
            for model_id, values in sorted(by_model_gaps.items())
        },
        "transition_entropy_bits": transition_entropy,
    }


def valid_swap(sequence, first, second, frozen_prefix):
    if first < frozen_prefix or second < frozen_prefix or first == second:
        return False
    if sequence[first] == sequence[second]:
        return False
    changed = {first, second}
    values = {
        first: sequence[second],
        second: sequence[first],
    }
    for index in changed:
        value = values[index]
        for neighbor in (index - 1, index + 1):
            if not 0 <= neighbor < len(sequence):
                continue
            neighbor_value = (
                values[neighbor] if neighbor in changed
                else sequence[neighbor])
            if value == neighbor_value:
                return False
    return True


def controlled_loss(
        metrics, target, reference, objective, gap_constraint):
    constraint_penalty = 0.0
    if gap_constraint == "reference":
        within_k_delta = abs(
            metrics["request_gap_within_k_fraction"]
            - reference["request_gap_within_k_fraction"])
        gap_p50_delta = abs(
            metrics["request_gap_p50"] - reference["request_gap_p50"])
        gap_p90_delta = abs(
            metrics["request_gap_p90"] - reference["request_gap_p90"])
        constraint_penalty = (
            12.0 * max(0.0, within_k_delta - 0.02)
            + 0.08 * max(0.0, gap_p50_delta - 1)
            + 0.04 * max(0.0, gap_p90_delta - 1))
    if objective == "min":
        primary = metrics["distinct_mean"]
    elif objective == "max":
        primary = -metrics["distinct_mean"]
    else:
        primary = abs(metrics["distinct_mean"] - target)
    return primary + constraint_penalty


def controlled_distinct_sequence(
        counts, rng, frozen_prefix, iterations, objective, target,
        lookahead_k, gap_constraint, initialization):
    balanced = balanced_count_sequence(counts, rng)
    if initialization == "max-distance":
        sequence = max_distance_tail(
            balanced[:frozen_prefix], counts, rng)
    elif initialization == "min-distance":
        sequence = min_distance_tail(
            balanced[:frozen_prefix], counts, rng)
    else:
        sequence = balanced
    reference = sequence_metrics(sequence, lookahead_k)
    current = list(sequence)
    current_metrics = reference
    current_loss = controlled_loss(
        current_metrics, target, reference, objective, gap_constraint)
    best = list(current)
    best_metrics = current_metrics
    best_loss = current_loss
    for iteration in range(iterations):
        first = rng.randrange(frozen_prefix, len(current))
        second = rng.randrange(frozen_prefix, len(current))
        if not valid_swap(current, first, second, frozen_prefix):
            continue
        current[first], current[second] = current[second], current[first]
        candidate_metrics = sequence_metrics(current, lookahead_k)
        candidate_loss = controlled_loss(
            candidate_metrics, target, reference, objective, gap_constraint)
        progress = iteration / max(1, iterations - 1)
        temperature = 0.10 * (1.0 - progress) + 0.001
        accept = (
            candidate_loss <= current_loss
            or rng.random()
            < math.exp((current_loss - candidate_loss) / temperature))
        if accept:
            current_metrics = candidate_metrics
            current_loss = candidate_loss
            if candidate_loss < best_loss:
                best = list(current)
                best_metrics = candidate_metrics
                best_loss = candidate_loss
        else:
            current[first], current[second] = current[second], current[first]
    return best, reference, best_metrics


def reuse_distances(sequence):
    last = {}
    result = defaultdict(list)
    for index, model_id in enumerate(sequence):
        if model_id in last:
            result[model_id].append(index - last[model_id])
        last[model_id] = index
    return result


def percentile(values, fraction):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def token_multiset_hash(rows_by_model):
    result = {}
    for model_id, rows in sorted(rows_by_model.items()):
        payload = "\n".join(
            f"{input_tokens},{output_tokens}"
            for input_tokens, output_tokens in sorted(rows))
        result[model_id] = hashlib.sha256(payload.encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument(
        "--sequence",
        choices=("period", "cyclic", "iid", "controlled-distinct"),
        default="period")
    parser.add_argument("--target-models", default="4,5,7")
    parser.add_argument("--background-models", default="0,1,2,3")
    parser.add_argument("--target-period", type=int, default=24)
    parser.add_argument(
        "--model-weights",
        help="Comma-separated MODEL:WEIGHT values for iid/background draws.")
    parser.add_argument("--token-scale", type=float, default=1.0)
    parser.add_argument("--fixed-input-tokens", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument(
        "--source-requests", type=int, default=1000,
        help=(
            "Source rows used to derive model-switch hotness for "
            "controlled-distinct."))
    parser.add_argument(
        "--controlled-count-policy",
        choices=("source-switch", "balanced"),
        default="source-switch",
        help=(
            "Derive controlled-distinct model counts from source model "
            "switches, or distribute requests equally across models."))
    parser.add_argument(
        "--common-prefix-requests", type=int, default=32,
        help="Frozen common prefix length for controlled-distinct.")
    parser.add_argument(
        "--controlled-objective",
        choices=("min", "max", "target"), default="target")
    parser.add_argument("--target-distinct-mean", type=float)
    parser.add_argument("--search-iterations", type=int, default=20000)
    parser.add_argument("--lookahead-k", type=int, default=32)
    parser.add_argument(
        "--controlled-gap-constraint",
        choices=("reference", "none"), default="reference",
        help=(
            "Keep request-gap quantiles and lookahead coverage near the "
            "initial sequence, or optimize distinct distance without a "
            "request-gap penalty."))
    parser.add_argument(
        "--controlled-initialization",
        choices=("balanced", "min-distance", "max-distance"),
        default="balanced",
        help=(
            "Initialize with the balanced randomized sequence, or preserve "
            "its frozen prefix and complete the tail by least-recent use."))
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.requests <= 0 or args.token_scale <= 0:
        raise ValueError("requests and token scale must be positive")

    rng = random.Random(args.seed)
    model_ids = load_model_ids(args.config)
    shapes = read_shapes(args.source)
    missing = sorted(set(model_ids) - set(shapes))
    if missing:
        raise ValueError(f"source trace is missing model IDs {missing}")
    weights = parse_weights(args.model_weights, model_ids)
    controlled_reference = None
    if args.sequence == "period":
        target_ids = parse_ids(args.target_models)
        background_ids = parse_ids(args.background_models)
        sequence = period_sequence(
            args.requests, args.target_period, target_ids, background_ids,
            weights, rng)
    elif args.sequence == "cyclic":
        sequence = cyclic_sequence(args.requests, model_ids, rng)
    elif args.sequence == "iid":
        sequence = iid_sequence(args.requests, model_ids, weights, rng)
    else:
        if (
            args.controlled_objective == "target"
            and args.target_distinct_mean is None
        ):
            raise ValueError(
                "--target-distinct-mean is required for target objective")
        if args.controlled_count_policy == "balanced":
            base, remainder = divmod(args.requests, len(model_ids))
            counts = {
                model_id: base + int(index < remainder)
                for index, model_id in enumerate(model_ids)
            }
        else:
            source_sequence = read_source_model_sequence(
                args.source, args.source_requests)
            switch_counts = Counter(model_switch_sequence(source_sequence))
            counts = largest_remainder_counts(
                switch_counts, model_ids, args.requests)
        sequence, controlled_reference, _ = controlled_distinct_sequence(
            counts, rng, args.common_prefix_requests,
            args.search_iterations, args.controlled_objective,
            args.target_distinct_mean, args.lookahead_k,
            args.controlled_gap_constraint, args.controlled_initialization)

    token_rng = random.Random(args.seed ^ 0x5A17C0DE)
    pools = {
        model_id: token_rng.sample(values, len(values))
        for model_id, values in shapes.items()
    }
    positions = Counter()
    rows = []
    rows_by_model = defaultdict(list)
    for index, model_id in enumerate(sequence):
        source_input, source_output = pools[model_id][
            positions[model_id] % len(pools[model_id])]
        positions[model_id] += 1
        input_tokens = (
            args.fixed_input_tokens
            if args.fixed_input_tokens is not None
            else max(1, round(source_input * args.token_scale))
        )
        if args.max_input_tokens is not None:
            input_tokens = min(input_tokens, args.max_input_tokens)
        output_tokens = (
            source_output if args.output_tokens is None
            else args.output_tokens)
        rows_by_model[model_id].append((input_tokens, output_tokens))
        rows.append(
            f"{index / 100.0:.6f} {model_id} "
            f"{input_tokens} {output_tokens}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(rows))

    distances = reuse_distances(sequence)
    counts = Counter(sequence)
    metrics = sequence_metrics(sequence, args.lookahead_k)
    metadata = {
        "format": "controlled-distinct-reuse-v1",
        "trace": str(args.output),
        "source": str(args.source),
        "source_requests": args.source_requests,
        "controlled_count_policy": args.controlled_count_policy,
        "sequence": args.sequence,
        "seed": args.seed,
        "requests": len(sequence),
        "common_prefix_requests": (
            args.common_prefix_requests
            if args.sequence == "controlled-distinct" else 0),
        "controlled_objective": (
            args.controlled_objective
            if args.sequence == "controlled-distinct" else None),
        "target_distinct_mean": args.target_distinct_mean,
        "lookahead_k": args.lookahead_k,
        "controlled_gap_constraint": (
            args.controlled_gap_constraint
            if args.sequence == "controlled-distinct" else None),
        "controlled_initialization": (
            args.controlled_initialization
            if args.sequence == "controlled-distinct" else None),
        "metrics": metrics,
        "reference_metrics": controlled_reference,
        "token_multiset_sha256": token_multiset_hash(rows_by_model),
    }
    metadata_path = (
        args.metadata_output
        if args.metadata_output is not None
        else args.output.with_suffix(args.output.suffix + ".json"))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"output={args.output} requests={len(sequence)} "
        f"sequence={args.sequence} token_scale={args.token_scale:g} "
        f"distinct_mean={metrics['distinct_mean']:.6f} "
        f"distinct_p90={metrics['distinct_p90']} "
        f"gap_within_k={metrics['request_gap_within_k_fraction']:.6f} "
        f"metadata={metadata_path}")
    for model_id in sorted(counts):
        values = distances[model_id]
        print(
            f"model={model_id} requests={counts[model_id]} "
            f"reuse_p50={percentile(values, .50)} "
            f"reuse_p90={percentile(values, .90)}")


if __name__ == "__main__":
    main()
