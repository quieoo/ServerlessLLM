#!/usr/bin/env python3
"""Profile-driven cache simulator for Minimal versus pipeline-aware eviction.

The simulator deliberately serializes request execution.  It models only the
weight-cache/Prefill pipeline mechanism: each request chooses between idle GPU
caches, reserves request KV pages from the same VMM pool, loads missing weight
pages, executes the M4 per-layer compute/H2D critical-path model, and leaves the
full active model resident.

Minimal reproduces the current stable-page value plus page-LRU tie break.
Pipeline-aware eviction values a page by the reduction in predicted GPU
critical path when that page is retained for a representative request shape.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from layerweave_estimator import PipelineEstimator, batch_features


@dataclass(frozen=True)
class Request:
    request_id: int
    model_id: int
    input_tokens: int
    output_tokens: int
    trace_output_tokens: int = 1


@dataclass
class ModelLayout:
    model_id: int
    stage_pages: list[int]
    page_stage: list[int]
    page_count: int
    max_input_tokens: int


@dataclass
class CacheState:
    resident: dict[int, set[int]]
    last_access: dict[tuple[int, int], int] = field(default_factory=dict)
    model_accesses: dict[int, int] = field(default_factory=dict)
    protected: dict[int, set[int]] = field(default_factory=dict)
    demand_scores: dict[int, float] = field(default_factory=dict)
    clock: int = 0


def _percentile(values, fraction):
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


def _summary(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values, default=0.0),
    }


def load_profile(path: Path) -> PipelineEstimator:
    document = json.loads(path.read_text())
    estimator = PipelineEstimator()
    estimator.models = {}
    for model_id, profile in document["profiles"].items():
        profile = copy.deepcopy(profile)
        profile["compute"] = {
            int(layer): value
            for layer, value in profile["compute"].items()
        }
        estimator.models[int(model_id)] = profile
    return estimator


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
            # load_trace still applies the simulated pool/KV feasibility cap.
            limit = 2**31 - 1
        else:
            raise ValueError(
                f"unknown input limit policy: {input_limit_policy}")
        limits[model_id] = limit
    return limits


def load_layouts(
    calibration_paths: list[Path],
    model_limits: dict[int, int],
) -> dict[int, ModelLayout]:
    best = {}
    for path in calibration_paths:
        document = json.loads(path.read_text())
        for batch in document.get("batch_metrics", []):
            metrics = batch.get("layerweave") or {}
            stages = metrics.get("stages") or []
            if not stages:
                continue
            mapped = sum(int(stage["mapped_pages"]) for stage in stages)
            model_id = int(batch["model_id"])
            if mapped <= best.get(model_id, (-1, None))[0]:
                continue
            best[model_id] = (mapped, metrics)
    layouts = {}
    for model_id, (mapped, metrics) in best.items():
        page_count = int(metrics["page_count"])
        if mapped != page_count:
            raise ValueError(
                f"Model {model_id} has no fully cold layout sample: "
                f"mapped={mapped}, pages={page_count}")
        stage_pages = [
            int(stage["mapped_pages"]) for stage in metrics["stages"]
        ]
        page_stage = [
            stage
            for stage, count in enumerate(stage_pages)
            for _ in range(count)
        ]
        if len(page_stage) != page_count:
            raise ValueError(
                f"Model {model_id} cold stage pages do not sum to page_count")
        layouts[model_id] = ModelLayout(
            model_id=model_id,
            stage_pages=stage_pages,
            page_stage=page_stage,
            page_count=page_count,
            max_input_tokens=model_limits[model_id],
        )
    missing = sorted(set(model_limits) - set(layouts))
    if missing:
        raise ValueError(f"Calibration layouts missing models: {missing}")
    return layouts


def load_trace(
    path: Path,
    layouts: dict[int, ModelLayout],
    max_requests: int,
    input_scale: float,
    output_tokens: int,
    pool_pages: int,
    kv_block_tokens: int,
    trace_mode: str = "all",
) -> list[Request]:
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
                    (
                        pool_pages - layouts[model_id].page_count
                    ) * kv_block_tokens - max(1, output_tokens),
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


def make_batch(
    request: Request,
    layout: ModelLayout,
    resident_pages: set[int],
) -> dict:
    missing_by_stage = [0] * len(layout.stage_pages)
    for page, stage in enumerate(layout.page_stage):
        if page not in resident_pages:
            missing_by_stage[stage] += 1
    stages = []
    for stage, missing in enumerate(missing_by_stage):
        stages.append({
            "stage": "initial" if stage == 0 else stage,
            "mapped_pages": missing,
            "to_load_bytes": missing * 64 * 1024 * 1024,
        })
    return {
        "model_id": request.model_id,
        "batch_size": 1,
        "input_tokens": request.input_tokens,
        "max_input_tokens": request.input_tokens,
        "sum_input_tokens_squared": request.input_tokens ** 2,
        "_prefetch": True,
        "_phase": "steady",
        "layerweave": {
            "page_count": layout.page_count,
            "stages": stages,
            "computes": [
                {"layer": layer}
                for layer in range(len(layout.stage_pages))
            ],
        },
    }


def predict(
    estimator: PipelineEstimator,
    request: Request,
    layout: ModelLayout,
    resident_pages: set[int],
) -> dict:
    batch = make_batch(request, layout, resident_pages)
    profile = estimator.models[request.model_id]
    gpu = estimator._simulate_gpu(batch, profile)
    hot = estimator._simulate_gpu(
        make_batch(request, layout, set(range(layout.page_count))),
        profile,
    )
    missing = layout.page_count - len(resident_pages)
    missing_by_stage = [
        int(stage["mapped_pages"])
        for stage in batch["layerweave"]["stages"]
    ]
    h2d_ms = (
        missing * 64 * 1024 * 1024 / profile["bytes_per_ms"]
        + sum(
            profile.get("fixed_stage_ms", 0.0)
            for stage in batch["layerweave"]["stages"]
            if stage["mapped_pages"]
        )
    )
    return {
        "critical_path_ms": gpu["predicted_gpu_critical_path_ms"],
        "hot_compute_ms": hot["predicted_gpu_critical_path_ms"],
        "exposed_load_ms": max(
            0.0,
            gpu["predicted_gpu_critical_path_ms"]
            - hot["predicted_gpu_critical_path_ms"],
        ),
        "h2d_ms": h2d_ms,
        "missing_pages": missing,
        "missing_pages_by_stage": missing_by_stage,
    }


def pipeline_timeline(
    estimator,
    request,
    layout,
    missing_pages_by_stage,
):
    resident = set()
    first_page = 0
    for count, missing in zip(layout.stage_pages, missing_pages_by_stage):
        resident.update(range(
            first_page,
            first_page + count - int(missing),
        ))
        first_page += count
    batch = make_batch(request, layout, resident)
    profile = estimator.models[request.model_id]
    features = batch_features(batch)
    stages = batch["layerweave"]["stages"]
    stage_by_layer = {
        int(stage["stage"]): stage
        for stage in stages if isinstance(stage["stage"], int)
    }
    initial = next(stage for stage in stages if stage["stage"] == "initial")

    def load_ms(stage):
        size = int(stage["to_load_bytes"])
        if size == 0:
            return profile["fixed_stage_ms"]
        return profile["fixed_stage_ms"] + size / profile["bytes_per_ms"]

    def compute_ms(layer):
        item = profile["compute"].get(layer, {})
        coefficients = item.get("coefficients")
        if coefficients is not None:
            return max(0.0, sum(
                coefficient * feature
                for coefficient, feature in zip(coefficients, features)
            ))
        total_tokens = max(1, request.input_tokens)
        return max(
            0.0,
            item.get("intercept_ms", 0.0)
            + item.get("ms_per_token", 0.0) * total_tokens
            + item.get("ms_per_token_squared", 0.0)
            * total_tokens * total_tokens,
        )

    initial_end = load_ms(initial)
    loads = [{
        "stage": 0,
        "missing_pages": int(initial["mapped_pages"]),
        "start_ms": 0.0,
        "end_ms": initial_end,
        "duration_ms": initial_end,
    }]
    computes = []
    compute_start = initial_end
    compute_end = initial_end
    copy_end = initial_end
    total_ready_stall = initial_end
    layer_count = len(batch["layerweave"]["computes"])
    for layer in range(layer_count):
        ready_stall = 0.0
        if layer > 0:
            stage = stage_by_layer[layer]
            copy_start = max(copy_end, previous_compute_start)
            copy_end = copy_start + load_ms(stage)
            ready_stall = max(0.0, copy_end - compute_end)
            total_ready_stall += ready_stall
            compute_start = max(compute_end, copy_end)
            loads.append({
                "stage": layer,
                "missing_pages": int(stage["mapped_pages"]),
                "start_ms": copy_start,
                "end_ms": copy_end,
                "duration_ms": copy_end - copy_start,
            })
        previous_compute_start = compute_start
        duration = compute_ms(layer)
        compute_end = compute_start + duration
        computes.append({
            "layer": layer,
            "start_ms": compute_start,
            "end_ms": compute_end,
            "duration_ms": duration,
            "ready_stall_before_ms": ready_stall,
        })
    return {
        "predicted_total_ms": compute_end,
        "initial_ready_ms": initial_end,
        "total_ready_stall_ms": total_ready_stall,
        "compute_sum_ms": sum(item["duration_ms"] for item in computes),
        "last_load_end_ms": loads[-1]["end_ms"],
        "tail_compute_after_last_load_ms": max(
            0.0, compute_end - loads[-1]["end_ms"]),
        "loads": loads,
        "computes": computes,
    }


def representative_requests(requests):
    by_model = defaultdict(list)
    for request in requests:
        by_model[request.model_id].append(request.input_tokens)
    return {
        model_id: Request(
            request_id=-1,
            model_id=model_id,
            input_tokens=max(1, round(statistics.median(tokens))),
            output_tokens=1,
        )
        for model_id, tokens in by_model.items()
    }


def pipeline_page_values(
    estimator,
    layouts,
    representatives,
) -> dict[tuple[int, int], float]:
    """Build a conditional greedy residency ladder.

    A one-page leave-one-out score from the fully cold state is insufficient:
    any saved copy shifts the serialized copy stream and appears equally
    valuable. Instead, start cold and repeatedly retain the stage page that
    gives the largest *conditional* critical-path reduction. This exposes the
    interaction between an immediately runnable prefix and overlapable suffix.
    """
    values = {}
    for model_id, layout in layouts.items():
        request = representatives[model_id]
        pages_by_stage = {
            stage: [
                page for page, page_stage in enumerate(layout.page_stage)
                if page_stage == stage
            ]
            for stage in range(len(layout.stage_pages))
        }
        retained = set()
        current = predict(
            estimator, request, layout, retained)["critical_path_ms"]
        for _ in range(layout.page_count):
            candidates = []
            for stage, pages in pages_by_stage.items():
                remaining = [page for page in pages if page not in retained]
                if not remaining:
                    continue
                page = remaining[0]
                candidate = predict(
                    estimator, request, layout,
                    retained | {page})["critical_path_ms"]
                candidates.append((
                    current - candidate,
                    -stage,
                    page,
                    candidate,
                ))
            gain, _, page, candidate = max(candidates)
            # Store the marginal value of retaining this page at the point it
            # enters the greedy ladder. A tiny rank term makes later, equally
            # valuable pages evict before earlier ladder entries.
            values[(model_id, page)] = (
                max(0.0, gain)
                + (layout.page_count - len(retained)) * 1e-9
            )
            retained.add(page)
            current = candidate
    return values


CONFIGURATION_LAYERS = (0, 2, 4, 8, 16, 32)


def configuration_layers(layout, step=0):
    if step <= 0:
        return CONFIGURATION_LAYERS
    return tuple(range(0, len(layout.stage_pages), step))


def prefix_pages(layout, layers):
    if layers == "full" or int(layers) >= len(layout.stage_pages):
        return frozenset(range(layout.page_count))
    page_count = sum(layout.stage_pages[:int(layers)])
    return frozenset(range(page_count))


def configuration_curves(estimator, layouts, requests, configuration_step=0):
    """Profile each model's protected-prefix configurations over its shapes."""
    by_model = defaultdict(list)
    for request in requests:
        by_model[request.model_id].append(request)
    total_requests = max(1, len(requests))
    curves = {}
    for model_id, layout in layouts.items():
        model_requests = by_model[model_id]
        probability = len(model_requests) / total_requests
        configurations = [
            *configuration_layers(layout, configuration_step),
            "full",
        ]
        candidates = []
        seen = set()
        for configuration in configurations:
            pages = prefix_pages(layout, configuration)
            if pages in seen:
                continue
            seen.add(pages)
            gains = []
            shape_gains = {}
            for request in model_requests:
                cold = predict(estimator, request, layout, set())
                configured = predict(
                    estimator, request, layout, set(pages))
                gain = (
                    cold["exposed_load_ms"]
                    - configured["exposed_load_ms"])
                gains.append(gain)
                shape_gains[request.input_tokens] = gain
            expected_gain = statistics.mean(gains) if gains else 0.0
            candidates.append({
                "configuration": (
                    "full"
                    if len(pages) == layout.page_count
                    else str(configuration)
                ),
                "pages": pages,
                "page_count": len(pages),
                "expected_gain_ms": max(0.0, expected_gain),
                "model_probability": probability,
                "shape_gains_ms": shape_gains,
                "expected_value_ms": (
                    probability * max(0.0, expected_gain)),
            })
        curves[model_id] = candidates
    return curves


def pareto_candidates(candidates):
    frontier = []
    best_value = -1.0
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["page_count"], -item["expected_value_ms"]),
    ):
        if candidate["expected_value_ms"] <= best_value:
            continue
        frontier.append(candidate)
        best_value = candidate["expected_value_ms"]
    return frontier


def solve_mckp(model_candidates, capacity_pages):
    states = {0: (0.0, {})}
    for model_id in sorted(model_candidates):
        next_states = {}
        for used, (value, choices) in states.items():
            for candidate in model_candidates[model_id]:
                new_used = used + candidate["page_count"]
                if new_used > capacity_pages:
                    continue
                new_value = value + candidate["expected_value_ms"]
                previous = next_states.get(new_used)
                if previous is None or new_value > previous[0]:
                    next_states[new_used] = (
                        new_value,
                        {**choices, model_id: candidate},
                    )
        if not next_states:
            raise RuntimeError(
                "No feasible MCKP state for "
                f"model={model_id}, capacity={capacity_pages}")
        pruned = {}
        best_value = -1.0
        for used in sorted(next_states):
            value, choices = next_states[used]
            if value > best_value:
                pruned[used] = (value, choices)
                best_value = value
        states = pruned
    used, (value, choices) = max(
        states.items(), key=lambda item: (item[1][0], -item[0]))
    return {
        "used_pages": used,
        "expected_value_ms": value,
        "choices": choices,
    }


def solve_mckp_topk(model_candidates, capacity_pages, topk):
    """Return heuristic Top-B configuration plans for exact re-ranking."""
    states = {0: [(0.0, {})]}
    for model_id in sorted(model_candidates):
        next_states = defaultdict(list)
        for used, entries in states.items():
            for value, choices in entries:
                for candidate in model_candidates[model_id]:
                    new_used = used + candidate["page_count"]
                    if new_used > capacity_pages:
                        continue
                    next_states[new_used].append((
                        value + candidate["expected_value_ms"],
                        {**choices, model_id: candidate},
                    ))
        if not next_states:
            raise RuntimeError(
                "No feasible Top-B MCKP state for "
                f"model={model_id}, capacity={capacity_pages}")
        states = {
            used: sorted(
                entries, key=lambda item: item[0], reverse=True)[:topk]
            for used, entries in next_states.items()
        }
    all_plans = [
        {
            "used_pages": used,
            "expected_value_ms": value,
            "choices": choices,
        }
        for used, entries in states.items()
        for value, choices in entries
    ]
    return sorted(
        all_plans,
        key=lambda item: (
            -item["expected_value_ms"], item["used_pages"]),
    )[:topk]


def clone_cache(cache):
    return CacheState(
        resident={
            model_id: set(pages)
            for model_id, pages in cache.resident.items()
        },
        last_access=dict(cache.last_access),
        model_accesses=dict(cache.model_accesses),
        protected={
            model_id: set(pages)
            for model_id, pages in cache.protected.items()
        },
        demand_scores=dict(cache.demand_scores),
        clock=cache.clock,
    )


def evict_for_request(
    policy,
    cache,
    request,
    layouts,
    pool_pages,
    kv_pages,
    pipeline_values,
):
    layout = layouts[request.model_id]
    active = cache.resident[request.model_id]
    missing = layout.page_count - len(active)
    resident_count = sum(len(pages) for pages in cache.resident.values())
    shortage = max(
        0,
        resident_count + missing + kv_pages - pool_pages,
    )
    victims = []
    total_accesses = max(1, sum(cache.model_accesses.values()))
    for model_id, pages in cache.resident.items():
        if model_id == request.model_id:
            continue
        probability = cache.model_accesses.get(model_id, 0) / total_accesses
        for page in pages:
            if policy == "minimal":
                value = probability
            else:
                value = probability * pipeline_values[(model_id, page)]
            victims.append((
                value,
                cache.last_access.get((model_id, page), 0),
                model_id,
                page,
            ))
    victims.sort()
    if len(victims) < shortage:
        raise RuntimeError(
            f"Cannot expose {shortage} pages for request "
            f"{request.request_id} model {request.model_id}")
    evicted_by_stage = defaultdict(int)
    for _, _, model_id, page in victims[:shortage]:
        cache.resident[model_id].remove(page)
        evicted_by_stage[layouts[model_id].page_stage[page]] += 1
    return shortage, dict(evicted_by_stage)


def configuration_rank(configuration):
    if configuration == "full":
        return math.inf
    return int(configuration)


def evict_for_request_mckp(
    cache,
    request,
    layouts,
    pool_pages,
    kv_pages,
    curves,
    future_requests=None,
    lookahead_discount=1.0,
    soft_victim_policy="configuration-tier",
    pipeline_values=None,
):
    active_layout = layouts[request.model_id]
    scored_curves = {}
    for model_id, candidates in curves.items():
        scored = []
        for candidate in candidates:
            if future_requests is None:
                value = (
                    cache.demand_scores.get(model_id, 0.0)
                    * candidate["expected_gain_ms"]
                )
            else:
                value = sum(
                    (lookahead_discount ** position)
                    * candidate["shape_gains_ms"][future.input_tokens]
                    for position, future in enumerate(future_requests)
                    if future.model_id == model_id
                )
            scored.append({
                **candidate,
                "expected_value_ms": value,
            })
        scored_curves[model_id] = scored
    protected_capacity = max(
        0, pool_pages - kv_pages - active_layout.page_count)
    inactive_candidates = {}
    for model_id, candidates in scored_curves.items():
        if model_id == request.model_id:
            continue
        current_pages = cache.protected[model_id]
        current_rank = max(
            (
                configuration_rank(candidate["configuration"])
                for candidate in candidates
                if candidate["pages"] == frozenset(current_pages)
            ),
            default=0,
        )
        if future_requests is None:
            allowed = [
                candidate for candidate in pareto_candidates(candidates)
                if (
                    configuration_rank(candidate["configuration"])
                    <= current_rank
                )
            ]
        else:
            allowed = [
                candidate for candidate in pareto_candidates(candidates)
                if candidate["pages"].issubset(
                    cache.resident[model_id])
            ]
            if not allowed:
                allowed = [candidates[0]]
        inactive_candidates[model_id] = allowed
    solution = solve_mckp(inactive_candidates, protected_capacity)
    protected = {
        model_id: set(solution["choices"][model_id]["pages"])
        for model_id in solution["choices"]
    }
    protected[request.model_id] = set(range(active_layout.page_count))

    active = cache.resident[request.model_id]
    missing = active_layout.page_count - len(active)
    resident_count = sum(len(pages) for pages in cache.resident.values())
    shortage = max(
        0, resident_count + missing + kv_pages - pool_pages)
    victims = []
    for model_id, pages in cache.resident.items():
        if model_id == request.model_id:
            continue
        soft = pages - protected[model_id]
        if soft_victim_policy == "page-greedy":
            if pipeline_values is None:
                raise ValueError(
                    "page-greedy soft victims require pipeline values")
            if future_requests is None:
                page_demand = cache.demand_scores.get(model_id, 0.0)
            else:
                page_demand = sum(
                    lookahead_discount ** position
                    for position, future in enumerate(future_requests)
                    if future.model_id == model_id
                )
            for page in soft:
                victims.append((
                    page_demand * pipeline_values[(model_id, page)],
                    cache.last_access.get((model_id, page), 0),
                    model_id,
                    -page,
                    page,
                ))
            continue
        page_metadata = {}
        model_curve = scored_curves[model_id]
        for tier_index, (lower, upper) in enumerate(
            zip(model_curve, model_curve[1:]), start=1
        ):
            added = upper["pages"] - lower["pages"]
            marginal_value = max(
                0.0,
                upper["expected_value_ms"]
                - lower["expected_value_ms"],
            )
            value_per_page = (
                marginal_value / len(added) if added else 0.0)
            for page in added:
                page_metadata[page] = (tier_index, value_per_page)
        for page in soft:
            tier_index, value_per_page = page_metadata.get(
                page, (len(model_curve), 0.0))
            victims.append((
                -tier_index,
                value_per_page,
                cache.last_access.get((model_id, page), 0),
                model_id,
                -page,
                page,
            ))
    victims.sort()
    if len(victims) < shortage:
        raise RuntimeError(
            "Configuration MCKP cannot expose enough soft pages: "
            f"request={request.request_id}, need={shortage}, "
            f"soft={len(victims)}")
    evicted_by_stage = defaultdict(int)
    for victim in victims[:shortage]:
        model_id, page = victim[-3], victim[-1]
        cache.resident[model_id].remove(page)
        evicted_by_stage[layouts[model_id].page_stage[page]] += 1
    cache.protected = protected
    return shortage, dict(evicted_by_stage), solution


def evict_for_request_transition_aware(
    cache,
    request,
    layouts,
    estimator,
    pool_pages,
    kv_pages,
    curves,
    future_requests,
    lookahead_discount,
    beam_width,
):
    """Re-rank configuration plans by their exact post-eviction residency."""
    active_layout = layouts[request.model_id]
    scored_curves = {}
    for model_id, candidates in curves.items():
        scored_curves[model_id] = [
            {
                **candidate,
                "expected_value_ms": sum(
                    (lookahead_discount ** position)
                    * candidate["shape_gains_ms"][future.input_tokens]
                    for position, future in enumerate(future_requests)
                    if future.model_id == model_id
                ),
            }
            for candidate in candidates
        ]
    protected_capacity = max(
        0, pool_pages - kv_pages - active_layout.page_count)
    inactive_candidates = {}
    for model_id, candidates in scored_curves.items():
        if model_id == request.model_id:
            continue
        allowed = [
            candidate for candidate in pareto_candidates(candidates)
            if candidate["pages"].issubset(cache.resident[model_id])
        ]
        inactive_candidates[model_id] = (
            allowed if allowed else [candidates[0]])
    plans = solve_mckp_topk(
        inactive_candidates, protected_capacity, beam_width)

    active = cache.resident[request.model_id]
    missing = active_layout.page_count - len(active)
    resident_count = sum(len(pages) for pages in cache.resident.values())
    shortage = max(
        0, resident_count + missing + kv_pages - pool_pages)

    evaluated = []
    seen_victims = set()
    for plan in plans:
        protected = {
            model_id: set(plan["choices"][model_id]["pages"])
            for model_id in plan["choices"]
        }
        protected[request.model_id] = set(
            range(active_layout.page_count))
        victims = []
        for model_id, pages in cache.resident.items():
            if model_id == request.model_id:
                continue
            model_curve = scored_curves[model_id]
            page_metadata = {}
            for tier_index, (lower, upper) in enumerate(
                zip(model_curve, model_curve[1:]), start=1
            ):
                added = upper["pages"] - lower["pages"]
                marginal_value = max(
                    0.0,
                    upper["expected_value_ms"]
                    - lower["expected_value_ms"],
                )
                value_per_page = (
                    marginal_value / len(added) if added else 0.0)
                for page in added:
                    page_metadata[page] = (
                        tier_index, value_per_page)
            for page in pages - protected[model_id]:
                tier_index, value_per_page = page_metadata.get(
                    page, (len(model_curve), 0.0))
                victims.append((
                    -tier_index,
                    value_per_page,
                    cache.last_access.get((model_id, page), 0),
                    model_id,
                    -page,
                    page,
                ))
        victims.sort()
        if len(victims) < shortage:
            continue
        selected = victims[:shortage]
        victim_key = tuple(
            sorted((item[3], item[5]) for item in selected))
        if victim_key in seen_victims:
            continue
        seen_victims.add(victim_key)
        retained = {
            model_id: set(pages)
            for model_id, pages in cache.resident.items()
        }
        for _, _, _, model_id, _, page in selected:
            retained[model_id].discard(page)
        retained[request.model_id] = set(
            range(active_layout.page_count))
        exact_future_ms = sum(
            (lookahead_discount ** position)
            * predict(
                estimator,
                future,
                layouts[future.model_id],
                retained[future.model_id],
            )["critical_path_ms"]
            for position, future in enumerate(future_requests)
        )
        evaluated.append((
            exact_future_ms,
            -plan["expected_value_ms"],
            plan,
            protected,
            selected,
        ))
    if not evaluated:
        raise RuntimeError(
            "Transition-aware planner found no feasible physical transition")
    (
        exact_future_ms, _, solution, protected, selected
    ) = min(evaluated, key=lambda item: (item[0], item[1]))
    evicted_by_stage = defaultdict(int)
    for _, _, _, model_id, _, page in selected:
        cache.resident[model_id].remove(page)
        evicted_by_stage[layouts[model_id].page_stage[page]] += 1
    cache.protected = protected
    solution = {
        **solution,
        "exact_future_critical_path_ms": exact_future_ms,
        "evaluated_transitions": len(evaluated),
    }
    return shortage, dict(evicted_by_stage), solution


def evict_for_request_residency_transition(
    cache,
    request,
    layouts,
    estimator,
    pool_pages,
    kv_pages,
    future_requests,
    lookahead_discount,
):
    """Choose stage-page victims by exact marginal damage from current R."""
    active_layout = layouts[request.model_id]
    active = cache.resident[request.model_id]
    missing = active_layout.page_count - len(active)
    resident_count = sum(len(pages) for pages in cache.resident.values())
    shortage = max(
        0, resident_count + missing + kv_pages - pool_pages)
    projected = {
        model_id: set(pages)
        for model_id, pages in cache.resident.items()
    }
    projected[request.model_id] = set(
        range(active_layout.page_count))
    future_by_model = defaultdict(list)
    for position, future in enumerate(future_requests):
        future_by_model[future.model_id].append((position, future))

    candidates = []
    for model_id, pages in cache.resident.items():
        if model_id == request.model_id or not pages:
            continue
        model_future = future_by_model[model_id]
        base = sum(
            (lookahead_discount ** position)
            * predict(
                estimator, future, layouts[model_id],
                projected[model_id])["critical_path_ms"]
            for position, future in model_future
        )
        pages_by_stage = defaultdict(list)
        for page in pages:
            pages_by_stage[
                layouts[model_id].page_stage[page]].append(page)
        for stage, stage_pages in pages_by_stage.items():
            without = projected[model_id] - set(stage_pages)
            damaged = sum(
                (lookahead_discount ** position)
                * predict(
                    estimator, future, layouts[model_id],
                    without)["critical_path_ms"]
                for position, future in model_future
            )
            damage = max(0.0, damaged - base)
            candidates.append((
                damage / len(stage_pages),
                -stage,
                min(
                    cache.last_access.get((model_id, page), 0)
                    for page in stage_pages
                ),
                model_id,
                sorted(stage_pages, reverse=True),
                damage,
            ))
    candidates.sort(key=lambda item: item[:4])
    selected = []
    remaining = shortage
    predicted_damage = 0.0
    for _, _, _, model_id, stage_pages, damage in candidates:
        if remaining <= 0:
            break
        take = min(remaining, len(stage_pages))
        chosen = stage_pages[:take]
        selected.extend((model_id, page) for page in chosen)
        predicted_damage += damage * take / len(stage_pages)
        remaining -= take
    if remaining:
        raise RuntimeError(
            "Residency-transition planner cannot expose enough pages: "
            f"request={request.request_id}, remaining={remaining}")
    evicted_by_stage = defaultdict(int)
    for model_id, page in selected:
        cache.resident[model_id].remove(page)
        evicted_by_stage[layouts[model_id].page_stage[page]] += 1
    cache.protected = {
        model_id: (
            set(range(active_layout.page_count))
            if model_id == request.model_id
            else set(cache.resident[model_id])
        )
        for model_id in layouts
    }
    solution = {
        "used_pages": sum(
            len(pages) for model_id, pages in cache.protected.items()
            if model_id != request.model_id),
        "expected_value_ms": -predicted_damage,
        "exact_future_critical_path_ms": None,
        "evaluated_transitions": len(candidates),
        "choices": {
            model_id: {
                "configuration": "exact-residency",
            }
            for model_id in layouts if model_id != request.model_id
        },
    }
    return shortage, dict(evicted_by_stage), solution


def execute(
    policy,
    cache,
    request,
    layouts,
    estimator,
    pool_pages,
    kv_block_tokens,
    pipeline_values,
    configuration_curves_by_model=None,
    demand_decay=0.9,
    future_requests=None,
    lookahead_discount=1.0,
    transition_beam_width=8,
):
    planner_started = time.perf_counter()
    kv_pages = math.ceil(
        (request.input_tokens + request.output_tokens) / kv_block_tokens)
    mckp = None
    if policy == "residency-transition":
        evicted, evicted_by_stage, mckp = (
            evict_for_request_residency_transition(
                cache, request, layouts, estimator, pool_pages, kv_pages,
                future_requests or [], lookahead_discount))
    elif policy == "configuration-transition":
        evicted, evicted_by_stage, mckp = (
            evict_for_request_transition_aware(
                cache, request, layouts, estimator, pool_pages, kv_pages,
                configuration_curves_by_model, future_requests or [],
                lookahead_discount, transition_beam_width))
    elif policy in {
        "configuration-mckp",
        "configuration-mckp-page-greedy",
    }:
        for model_id in cache.demand_scores:
            cache.demand_scores[model_id] *= demand_decay
        cache.demand_scores[request.model_id] += 1.0
        evicted, evicted_by_stage, mckp = evict_for_request_mckp(
            cache, request, layouts, pool_pages, kv_pages,
            configuration_curves_by_model, future_requests,
            lookahead_discount,
            soft_victim_policy=(
                "page-greedy"
                if policy == "configuration-mckp-page-greedy"
                else "configuration-tier"
            ),
            pipeline_values=pipeline_values)
    else:
        evicted, evicted_by_stage = evict_for_request(
            policy, cache, request, layouts, pool_pages, kv_pages,
            pipeline_values)
    planner_time_ms = (
        time.perf_counter() - planner_started) * 1000.0
    layout = layouts[request.model_id]
    before = set(cache.resident[request.model_id])
    prediction = predict(estimator, request, layout, before)
    cache.model_accesses[request.model_id] = (
        cache.model_accesses.get(request.model_id, 0) + 1)
    for page in range(layout.page_count):
        cache.clock += 1
        cache.last_access[(request.model_id, page)] = cache.clock
    cache.resident[request.model_id] = set(range(layout.page_count))
    return {
        **prediction,
        "evicted_pages": evicted,
        "evicted_by_stage": evicted_by_stage,
        "kv_pages": kv_pages,
        "planner_time_ms": planner_time_ms,
        "mckp": (
            {
                "used_inactive_protected_pages": mckp["used_pages"],
                "expected_value_ms": mckp["expected_value_ms"],
                "exact_future_critical_path_ms":
                    mckp.get("exact_future_critical_path_ms"),
                "evaluated_transitions":
                    mckp.get("evaluated_transitions", 0),
                "configurations": {
                    str(model_id):
                        choice["configuration"]
                    for model_id, choice in mckp["choices"].items()
                },
            }
            if mckp is not None else None
        ),
    }


def simulate_policy(
    policy,
    requests,
    layouts,
    estimator,
    pool_pages,
    gpu_count,
    kv_block_tokens,
    pipeline_values,
    seed,
    routing_replay=None,
    configuration_curves_by_model=None,
    demand_decay=0.9,
    lookahead_k=0,
    lookahead_discount=1.0,
    transition_beam_width=8,
):
    caches = [
        CacheState(
            resident={model_id: set() for model_id in layouts},
            model_accesses={model_id: 0 for model_id in layouts},
            protected={model_id: set() for model_id in layouts},
            demand_scores={model_id: 0.0 for model_id in layouts},
        )
        for _ in range(gpu_count)
    ]
    rng = random.Random(seed)
    rows = []
    routing = []
    future_positions = None
    if routing_replay is not None and lookahead_k > 0:
        requests_by_gpu = defaultdict(list)
        for replay_request in requests:
            requests_by_gpu[
                routing_replay[replay_request.request_id]].append(
                    replay_request)
        future_positions = {
            replay_request.request_id: position
            for gpu_requests in requests_by_gpu.values()
            for position, replay_request in enumerate(gpu_requests)
        }
    for request in requests:
        candidate_predictions = [
            predict(
                estimator,
                request,
                layouts[request.model_id],
                cache.resident[request.model_id],
            )
            for cache in caches
        ]
        if routing_replay is not None:
            chosen = routing_replay[request.request_id]
        else:
            best = min(
                prediction["critical_path_ms"]
                for prediction in candidate_predictions)
            candidates = [
                gpu for gpu, prediction in enumerate(candidate_predictions)
                if math.isclose(
                    prediction["critical_path_ms"], best,
                    rel_tol=0.0, abs_tol=1e-9)
            ]
            chosen = rng.choice(candidates)
        future_requests = None
        queue_read_time_ms = 0.0
        if lookahead_k > 0:
            if routing_replay is None:
                raise ValueError(
                    "Lookahead simulation requires fixed routing replay")
            queue_read_started = time.perf_counter()
            gpu_requests = requests_by_gpu[chosen]
            position = future_positions[request.request_id]
            future_requests = gpu_requests[
                position + 1:position + 1 + lookahead_k]
            queue_read_time_ms = (
                time.perf_counter() - queue_read_started) * 1000.0
        result = execute(
            policy, caches[chosen], request, layouts, estimator,
            pool_pages, kv_block_tokens, pipeline_values,
            configuration_curves_by_model, demand_decay,
            future_requests, lookahead_discount,
            transition_beam_width)
        rows.append({
            "request_id": request.request_id,
            "model_id": request.model_id,
            "gpu": chosen,
            "input_tokens": request.input_tokens,
            **result,
            "queue_read_time_ms": queue_read_time_ms,
            "controller_time_ms":
                queue_read_time_ms + result["planner_time_ms"],
        })
        routing.append(chosen)
    return {
        "policy": policy,
        "requests": rows,
        "routing": routing,
        "summary": {
            "requests": len(rows),
            "critical_path_ms": _summary(
                row["critical_path_ms"] for row in rows),
            "exposed_load_ms": _summary(
                row["exposed_load_ms"] for row in rows),
            "h2d_ms": _summary(row["h2d_ms"] for row in rows),
            "missing_pages": sum(row["missing_pages"] for row in rows),
            "evicted_pages": sum(row["evicted_pages"] for row in rows),
            "planner_time_ms": _summary(
                row["planner_time_ms"] for row in rows),
            "queue_read_time_ms": _summary(
                row["queue_read_time_ms"] for row in rows),
            "controller_time_ms": _summary(
                row["controller_time_ms"] for row in rows),
            "requests_by_gpu": {
                str(gpu): sum(row["gpu"] == gpu for row in rows)
                for gpu in range(gpu_count)
            },
            "evicted_pages_by_stage": {
                str(stage): sum(
                    row["evicted_by_stage"].get(stage, 0)
                    for row in rows
                )
                for stage in sorted({
                    stage
                    for row in rows
                    for stage in row["evicted_by_stage"]
                })
            },
            "mckp_decisions": sum(
                row["mckp"] is not None for row in rows),
            "mean_inactive_protected_pages": (
                statistics.mean(
                    row["mckp"]["used_inactive_protected_pages"]
                    for row in rows if row["mckp"] is not None
                )
                if any(row["mckp"] is not None for row in rows)
                else 0.0
            ),
        },
    }


def simulate_joint_policy(
    requests,
    layouts,
    estimator,
    pool_pages,
    gpu_count,
    kv_block_tokens,
    pipeline_values,
    seed,
    configuration_curves_by_model,
    demand_decay=0.9,
    lookahead_k=8,
    lookahead_discount=1.0,
    transition_weight=0.1,
    transition_credit_cap_ms=100.0,
):
    """Serial-choice joint placement plus MCKP/PageGreedy cache planning.

    Each candidate GPU is planned from its actual cache snapshot. Placement
    minimizes current critical path plus a bounded estimate of the change in
    future best-GPU critical path caused by committing that candidate state.
    """
    caches = [
        CacheState(
            resident={model_id: set() for model_id in layouts},
            model_accesses={model_id: 0 for model_id in layouts},
            protected={model_id: set() for model_id in layouts},
            demand_scores={model_id: 0.0 for model_id in layouts},
        )
        for _ in range(gpu_count)
    ]
    rng = random.Random(seed)
    rows = []
    routing = []

    def future_best_cost(future_requests, projected_caches):
        return sum(
            (lookahead_discount ** position) * min(
                predict(
                    estimator,
                    future,
                    layouts[future.model_id],
                    cache.resident[future.model_id],
                )["critical_path_ms"]
                for cache in projected_caches
            )
            for position, future in enumerate(future_requests)
        )

    for position, request in enumerate(requests):
        routing_started = time.perf_counter()
        future_requests = requests[
            position + 1:position + 1 + lookahead_k]
        before_future_cost = future_best_cost(future_requests, caches)
        candidates = []
        for gpu in range(gpu_count):
            candidate_cache = clone_cache(caches[gpu])
            result = execute(
                "configuration-mckp-page-greedy",
                candidate_cache,
                request,
                layouts,
                estimator,
                pool_pages,
                kv_block_tokens,
                pipeline_values,
                configuration_curves_by_model,
                demand_decay,
                future_requests,
                lookahead_discount,
            )
            projected_caches = list(caches)
            projected_caches[gpu] = candidate_cache
            after_future_cost = future_best_cost(
                future_requests, projected_caches)
            transition_cost = after_future_cost - before_future_cost
            bounded_transition = max(
                -transition_credit_cap_ms,
                min(transition_credit_cap_ms, transition_cost),
            )
            score = (
                result["critical_path_ms"]
                + transition_weight * bounded_transition
            )
            candidates.append({
                "gpu": gpu,
                "cache": candidate_cache,
                "result": result,
                "score": score,
                "transition_cost_ms": transition_cost,
                "bounded_transition_cost_ms": bounded_transition,
            })
        best_score = min(candidate["score"] for candidate in candidates)
        best = [
            candidate for candidate in candidates
            if math.isclose(
                candidate["score"], best_score,
                rel_tol=0.0, abs_tol=1e-9)
        ]
        chosen = rng.choice(best)
        caches[chosen["gpu"]] = chosen["cache"]
        routing_time_ms = (
            time.perf_counter() - routing_started) * 1000.0
        rows.append({
            "request_id": request.request_id,
            "model_id": request.model_id,
            "gpu": chosen["gpu"],
            "input_tokens": request.input_tokens,
            **chosen["result"],
            "placement_score_ms": chosen["score"],
            "transition_cost_ms": chosen["transition_cost_ms"],
            "bounded_transition_cost_ms":
                chosen["bounded_transition_cost_ms"],
            "queue_read_time_ms": 0.0,
            "routing_planner_time_ms": routing_time_ms,
            "controller_time_ms": routing_time_ms,
        })
        routing.append(chosen["gpu"])
    result = {
        "policy": "joint-configuration-mckp-page-greedy",
        "requests": rows,
        "routing": routing,
        "summary": {
            "requests": len(rows),
            "critical_path_ms": _summary(
                row["critical_path_ms"] for row in rows),
            "exposed_load_ms": _summary(
                row["exposed_load_ms"] for row in rows),
            "h2d_ms": _summary(row["h2d_ms"] for row in rows),
            "missing_pages": sum(row["missing_pages"] for row in rows),
            "evicted_pages": sum(row["evicted_pages"] for row in rows),
            "planner_time_ms": _summary(
                row["planner_time_ms"] for row in rows),
            "routing_planner_time_ms": _summary(
                row["routing_planner_time_ms"] for row in rows),
            "controller_time_ms": _summary(
                row["controller_time_ms"] for row in rows),
            "transition_cost_ms": _summary(
                row["transition_cost_ms"] for row in rows),
            "requests_by_gpu": {
                str(gpu): sum(row["gpu"] == gpu for row in rows)
                for gpu in range(gpu_count)
            },
            "mckp_decisions": len(rows),
            "mean_inactive_protected_pages": statistics.mean(
                row["mckp"]["used_inactive_protected_pages"]
                for row in rows
            ) if rows else 0.0,
        },
    }
    return result


def whole_model_h2d_ms(estimator, layout):
    """Aegaeon uses one contiguous whole-model CPU-to-GPU transfer."""
    profile = estimator.models[layout.model_id]
    return (
        layout.page_count * 64 * 1024 * 1024 / profile["bytes_per_ms"]
        + profile.get("fixed_stage_ms", 0.0)
    )


def simulate_aegaeon(
    requests,
    layouts,
    estimator,
    pool_pages,
    gpu_count,
    kv_block_tokens,
    routing,
    decode_ms_per_token=40.0,
    gpu_copy_gbps=864.0,
):
    """Replay scheduled whole-model double-buffer prefetch.

    On each GPU, request i's real trace decode length is the overlap window for
    prefetching request i+1. Prefetch is allowed only when the current model,
    its request KV, and the complete next model fit together. A prefetched
    model pays the remaining H2D time plus a contiguous GPU-buffer relocation
    at switch time. This deliberately does not grant Aegaeon LayerWeave's
    persistent multi-model page cache or per-layer execution pipeline.
    """
    if len(routing) != len(requests):
        raise ValueError("Aegaeon routing length must match requests")
    requests_by_gpu = defaultdict(list)
    for request, gpu in zip(requests, routing):
        requests_by_gpu[gpu].append(request)

    rows_by_request = {}
    bytes_per_gpu_copy_ms = gpu_copy_gbps * 1_000_000.0
    for gpu in range(gpu_count):
        gpu_requests = requests_by_gpu[gpu]
        for position, request in enumerate(gpu_requests):
            layout = layouts[request.model_id]
            h2d_ms = whole_model_h2d_ms(estimator, layout)
            hot_compute_ms = predict(
                estimator, request, layout,
                set(range(layout.page_count)))["hot_compute_ms"]
            predecessor = (
                gpu_requests[position - 1] if position > 0 else None)
            same_model_hit = (
                predecessor is not None
                and predecessor.model_id == request.model_id
            )
            decode_window_ms = 0.0
            capacity_required_pages = layout.page_count
            capacity_feasible = False
            prefetched_h2d_ms = 0.0
            gpu_relocation_ms = 0.0
            if predecessor is not None and not same_model_hit:
                previous_layout = layouts[predecessor.model_id]
                previous_kv_pages = math.ceil(
                    (
                        predecessor.input_tokens
                        + predecessor.output_tokens
                    ) / kv_block_tokens
                )
                capacity_required_pages = (
                    previous_layout.page_count
                    + previous_kv_pages
                    + layout.page_count
                )
                capacity_feasible = capacity_required_pages <= pool_pages
                if capacity_feasible:
                    decode_window_ms = (
                        predecessor.trace_output_tokens
                        * decode_ms_per_token
                    )
                    prefetched_h2d_ms = min(h2d_ms, decode_window_ms)
                    gpu_relocation_ms = (
                        layout.page_count * 64 * 1024 * 1024
                        / bytes_per_gpu_copy_ms
                    )
            if same_model_hit:
                loading_stall_ms = 0.0
                transfer_h2d_ms = 0.0
                capacity_feasible = True
            elif capacity_feasible:
                loading_stall_ms = (
                    h2d_ms - prefetched_h2d_ms + gpu_relocation_ms)
                transfer_h2d_ms = h2d_ms
            else:
                loading_stall_ms = h2d_ms
                transfer_h2d_ms = h2d_ms
            rows_by_request[request.request_id] = {
                "request_id": request.request_id,
                "model_id": request.model_id,
                "gpu": gpu,
                "input_tokens": request.input_tokens,
                "output_tokens": request.output_tokens,
                "trace_output_tokens": request.trace_output_tokens,
                "hot_compute_ms": hot_compute_ms,
                "critical_path_ms": hot_compute_ms + loading_stall_ms,
                "loading_stall_ms": loading_stall_ms,
                "h2d_ms": transfer_h2d_ms,
                "decode_overlap_window_ms": decode_window_ms,
                "prefetched_h2d_ms": prefetched_h2d_ms,
                "gpu_relocation_ms": gpu_relocation_ms,
                "capacity_required_pages": capacity_required_pages,
                "capacity_feasible": capacity_feasible,
                "same_model_hit": same_model_hit,
                "has_predecessor": predecessor is not None,
            }
    rows = [rows_by_request[request.request_id] for request in requests]
    return {
        "policy": "aegaeon-whole-model-double-buffer-prefetch",
        "routing": list(routing),
        "requests": rows,
        "summary": {
            "requests": len(rows),
            "critical_path_ms": _summary(
                row["critical_path_ms"] for row in rows),
            "loading_stall_ms": _summary(
                row["loading_stall_ms"] for row in rows),
            "h2d_ms": _summary(row["h2d_ms"] for row in rows),
            "decode_overlap_window_ms": _summary(
                row["decode_overlap_window_ms"] for row in rows),
            "prefetched_h2d_ms": _summary(
                row["prefetched_h2d_ms"] for row in rows),
            "gpu_relocation_ms": _summary(
                row["gpu_relocation_ms"] for row in rows),
            "capacity_feasible_prefetches": sum(
                row["capacity_feasible"] and not row["same_model_hit"]
                and row["has_predecessor"]
                for row in rows
            ),
            "capacity_blocked_prefetches": sum(
                not row["capacity_feasible"]
                and row["has_predecessor"]
                for row in rows
            ),
            "same_model_hits": sum(
                row["same_model_hit"] for row in rows),
            "fully_hidden_prefetches": sum(
                row["capacity_feasible"]
                and not row["same_model_hit"]
                and row["has_predecessor"]
                and row["prefetched_h2d_ms"] >= row["h2d_ms"]
                and row["h2d_ms"] > 0.0
                for row in rows
            ),
            "requests_by_gpu": {
                str(gpu): sum(row["gpu"] == gpu for row in rows)
                for gpu in range(gpu_count)
            },
        },
    }


def summarize_ablation_rows(rows):
    return {
        "requests": len(rows),
        "predicted_total_ms": _summary(
            row["predicted_total_ms"] for row in rows),
        "hot_compute_ms": _summary(
            row["hot_compute_ms"] for row in rows),
        "h2d_ms": _summary(row["h2d_ms"] for row in rows),
        "exposed_load_ms": _summary(
            row["exposed_load_ms"] for row in rows),
        "missing_pages": sum(row["missing_pages"] for row in rows),
    }


def build_five_way_ablation(
    requests,
    layouts,
    estimator,
    minimal,
    layerweave_fixed,
    inspect_request_id=None,
):
    """Construct an additive five-way cache/pipeline decomposition.

    Baseline and pipe-only are fully cold on every request. Reuse-only and
    Minimal use the exact same Minimal cache states and routing. LayerWeave
    replays Minimal routing but uses its pipeline-aware cache policy.
    """
    cold_by_request = {}
    for request in requests:
        cold_by_request[request.request_id] = predict(
            estimator, request, layouts[request.model_id], set())

    baseline_rows = []
    pipe_only_rows = []
    reuse_only_rows = []
    minimal_rows = []
    layerweave_rows = []
    for request, minimal_row, layerweave_row in zip(
        requests,
        minimal["requests"],
        layerweave_fixed["requests"],
    ):
        cold = cold_by_request[request.request_id]
        cold_serial = cold["hot_compute_ms"] + cold["h2d_ms"]
        baseline_rows.append({
            "predicted_total_ms": cold_serial,
            "hot_compute_ms": cold["hot_compute_ms"],
            "h2d_ms": cold["h2d_ms"],
            "exposed_load_ms": cold["h2d_ms"],
            "missing_pages": cold["missing_pages"],
        })
        pipe_only_rows.append({
            "predicted_total_ms": cold["critical_path_ms"],
            "hot_compute_ms": cold["hot_compute_ms"],
            "h2d_ms": cold["h2d_ms"],
            "exposed_load_ms": cold["exposed_load_ms"],
            "missing_pages": cold["missing_pages"],
        })
        reuse_serial = (
            minimal_row["hot_compute_ms"] + minimal_row["h2d_ms"])
        reuse_only_rows.append({
            "predicted_total_ms": reuse_serial,
            "hot_compute_ms": minimal_row["hot_compute_ms"],
            "h2d_ms": minimal_row["h2d_ms"],
            "exposed_load_ms": minimal_row["h2d_ms"],
            "missing_pages": minimal_row["missing_pages"],
        })
        minimal_rows.append({
            "predicted_total_ms": minimal_row["critical_path_ms"],
            "hot_compute_ms": minimal_row["hot_compute_ms"],
            "h2d_ms": minimal_row["h2d_ms"],
            "exposed_load_ms": minimal_row["exposed_load_ms"],
            "missing_pages": minimal_row["missing_pages"],
        })
        layerweave_rows.append({
            "predicted_total_ms": layerweave_row["critical_path_ms"],
            "hot_compute_ms": layerweave_row["hot_compute_ms"],
            "h2d_ms": layerweave_row["h2d_ms"],
            "exposed_load_ms": layerweave_row["exposed_load_ms"],
            "missing_pages": layerweave_row["missing_pages"],
        })

    stages = {
        "baseline_load_plus_compute": summarize_ablation_rows(baseline_rows),
        "pipe_only": summarize_ablation_rows(pipe_only_rows),
        "reuse_only": summarize_ablation_rows(reuse_only_rows),
        "minimal": summarize_ablation_rows(minimal_rows),
        "layerweave": summarize_ablation_rows(layerweave_rows),
    }
    baseline_mean = stages[
        "baseline_load_plus_compute"]["predicted_total_ms"]["mean"]
    for stage in stages.values():
        mean = stage["predicted_total_ms"]["mean"]
        stage["gain_vs_baseline_ms"] = baseline_mean - mean
        stage["gain_vs_baseline_fraction"] = (
            (baseline_mean - mean) / baseline_mean
            if baseline_mean else 0.0
        )
    minimal_mean = stages["minimal"]["predicted_total_ms"]["mean"]
    layerweave_mean = stages["layerweave"]["predicted_total_ms"]["mean"]
    if inspect_request_id is None:
        positive_gains = [
            (
                minimal_rows[index]["predicted_total_ms"]
                - layerweave_rows[index]["predicted_total_ms"],
                index,
            )
            for index in range(len(requests))
            if (
                minimal_rows[index]["predicted_total_ms"]
                - layerweave_rows[index]["predicted_total_ms"]
            ) > 1e-9
        ]
        if positive_gains:
            median_gain = statistics.median(
                gain for gain, _ in positive_gains)
            _, inspect_index = min(
                positive_gains,
                key=lambda item: abs(item[0] - median_gain),
            )
            selection = (
                "closest to median positive LayerWeave-vs-Minimal gain")
        else:
            median_pipe = statistics.median(
                row["predicted_total_ms"] for row in pipe_only_rows)
            inspect_index = min(
                range(len(requests)),
                key=lambda index: abs(
                    pipe_only_rows[index]["predicted_total_ms"] - median_pipe),
            )
            selection = (
                "closest to pipe-only median; no positive LayerWeave gain")
    else:
        matches = [
            index for index, request in enumerate(requests)
            if request.request_id == inspect_request_id
        ]
        if not matches:
            raise ValueError(
                f"--inspect-request-id {inspect_request_id} is outside "
                "the simulated request range")
        inspect_index = matches[0]
        selection = "explicit --inspect-request-id"
    inspected = requests[inspect_index]
    inspected_layout = layouts[inspected.model_id]
    cold_missing = inspected_layout.stage_pages
    minimal_missing = minimal["requests"][inspect_index][
        "missing_pages_by_stage"]
    layerweave_missing = layerweave_fixed["requests"][inspect_index][
        "missing_pages_by_stage"]
    pipe_timeline = pipeline_timeline(
        estimator, inspected, inspected_layout, cold_missing)
    minimal_timeline = pipeline_timeline(
        estimator, inspected, inspected_layout, minimal_missing)
    layerweave_timeline = pipeline_timeline(
        estimator, inspected, inspected_layout, layerweave_missing)

    def serial_timeline(row, missing_by_stage):
        load_end = row["h2d_ms"]
        return {
            "predicted_total_ms": row["predicted_total_ms"],
            "load": {
                "start_ms": 0.0,
                "end_ms": load_end,
                "duration_ms": load_end,
                "missing_pages_by_stage": list(missing_by_stage),
            },
            "compute": {
                "start_ms": load_end,
                "end_ms": row["predicted_total_ms"],
                "duration_ms": row["hot_compute_ms"],
            },
        }

    inspected_request = {
        "selection": selection,
        "request_id": inspected.request_id,
        "model_id": inspected.model_id,
        "input_tokens": inspected.input_tokens,
        "output_tokens": inspected.output_tokens,
        "baseline_load_plus_compute": serial_timeline(
            baseline_rows[inspect_index], cold_missing),
        "pipe_only": pipe_timeline,
        "reuse_only": serial_timeline(
            reuse_only_rows[inspect_index], minimal_missing),
        "minimal": minimal_timeline,
        "layerweave": layerweave_timeline,
    }
    return {
        "comparison_control":
            "same requests; Minimal routing replayed for LayerWeave",
        "definitions": {
            "baseline_load_plus_compute":
                "fully cold H2D plus hot compute, serialized",
            "pipe_only":
                "fully cold H2D and per-layer compute, pipelined",
            "reuse_only":
                "Minimal cache reuse, but missing-page H2D plus hot compute "
                "serialized",
            "minimal":
                "Minimal cache reuse with per-layer H2D/compute pipeline",
            "layerweave":
                "configuration-MCKP protected prefixes with deferred soft "
                "reclamation, per-layer H2D/compute pipeline, and Minimal "
                "routing replay",
        },
        "stages": stages,
        "inspected_request": inspected_request,
        "layerweave_gain_vs_minimal_ms": minimal_mean - layerweave_mean,
        "layerweave_gain_vs_minimal_fraction": (
            (minimal_mean - layerweave_mean) / minimal_mean
            if minimal_mean else 0.0
        ),
    }


def run(args):
    estimator = load_profile(args.profile)
    limits = load_model_limits(args.config)
    layouts = load_layouts(args.calibration, limits)
    requests = load_trace(
        args.trace, layouts, args.max_requests,
        args.input_scale, args.output_tokens_override,
        (
            args.request_cap_pool_pages
            if args.request_cap_pool_pages > 0
            else args.pool_pages
        ),
        args.kv_block_tokens,
        args.trace_mode)
    representatives = representative_requests(requests)
    pipeline_values = pipeline_page_values(
        estimator, layouts, representatives)
    configuration_candidates = configuration_curves(
        estimator, layouts, requests, args.configuration_step)
    minimal = simulate_policy(
        "minimal", requests, layouts, estimator, args.pool_pages,
        args.gpus, args.kv_block_tokens, pipeline_values, args.seed)
    run_all = args.policy_suite == "all"
    page_greedy_independent = None
    page_greedy_fixed = None
    layerweave_independent = None
    layerweave_fixed = None
    if run_all:
        page_greedy_independent = simulate_policy(
            "pipeline-aware", requests, layouts, estimator, args.pool_pages,
            args.gpus, args.kv_block_tokens, pipeline_values, args.seed)
        page_greedy_fixed = simulate_policy(
            "pipeline-aware", requests, layouts, estimator, args.pool_pages,
            args.gpus, args.kv_block_tokens, pipeline_values, args.seed,
            routing_replay=minimal["routing"])
        layerweave_independent = simulate_policy(
            "configuration-mckp", requests, layouts, estimator,
            args.pool_pages, args.gpus, args.kv_block_tokens,
            pipeline_values, args.seed,
            configuration_curves_by_model=configuration_candidates,
            demand_decay=args.demand_decay)
        layerweave_fixed = simulate_policy(
            "configuration-mckp", requests, layouts, estimator,
            args.pool_pages, args.gpus, args.kv_block_tokens,
            pipeline_values, args.seed,
            routing_replay=minimal["routing"],
            configuration_curves_by_model=configuration_candidates,
            demand_decay=args.demand_decay)
    hybrid_fixed = simulate_policy(
        "configuration-mckp-page-greedy", requests, layouts, estimator,
        args.pool_pages, args.gpus, args.kv_block_tokens, pipeline_values,
        args.seed, routing_replay=minimal["routing"],
        configuration_curves_by_model=configuration_candidates,
        demand_decay=args.demand_decay)
    lookahead_results = {}
    hybrid_lookahead_results = {}
    transition_results = {}
    residency_transition_results = {}
    for lookahead_k in args.lookahead_k:
        hybrid_lookahead_results[str(lookahead_k)] = simulate_policy(
            "configuration-mckp-page-greedy", requests, layouts, estimator,
            args.pool_pages, args.gpus, args.kv_block_tokens,
            pipeline_values, args.seed,
            routing_replay=minimal["routing"],
            configuration_curves_by_model=configuration_candidates,
            demand_decay=args.demand_decay,
            lookahead_k=lookahead_k,
            lookahead_discount=args.lookahead_discount,
        )
        if run_all:
            lookahead_results[str(lookahead_k)] = simulate_policy(
                "configuration-mckp", requests, layouts, estimator,
                args.pool_pages, args.gpus, args.kv_block_tokens,
                pipeline_values, args.seed,
                routing_replay=minimal["routing"],
                configuration_curves_by_model=configuration_candidates,
                demand_decay=args.demand_decay,
                lookahead_k=lookahead_k,
                lookahead_discount=args.lookahead_discount,
            )
            transition_results[str(lookahead_k)] = simulate_policy(
                "configuration-transition", requests, layouts, estimator,
                args.pool_pages, args.gpus, args.kv_block_tokens,
                pipeline_values, args.seed,
                routing_replay=minimal["routing"],
                configuration_curves_by_model=configuration_candidates,
                demand_decay=args.demand_decay,
                lookahead_k=lookahead_k,
                lookahead_discount=args.lookahead_discount,
                transition_beam_width=args.transition_beam_width,
            )
            residency_transition_results[str(lookahead_k)] = simulate_policy(
                "residency-transition", requests, layouts, estimator,
                args.pool_pages, args.gpus, args.kv_block_tokens,
                pipeline_values, args.seed,
                routing_replay=minimal["routing"],
                configuration_curves_by_model=configuration_candidates,
                demand_decay=args.demand_decay,
                lookahead_k=lookahead_k,
                lookahead_discount=args.lookahead_discount,
            )
    joint_results = {}
    if args.joint_routing:
        joint_lookahead = (
            args.lookahead_k[-1] if args.lookahead_k else 8)
        joint_results[str(args.configuration_step)] = simulate_joint_policy(
            requests, layouts, estimator, args.pool_pages, args.gpus,
            args.kv_block_tokens, pipeline_values, args.seed,
            configuration_candidates, args.demand_decay,
            joint_lookahead, args.lookahead_discount,
            args.transition_weight, args.transition_credit_cap_ms)
    aegaeon_routing = minimal["routing"]
    aegaeon_routing_source = "minimal"
    layerweave_comparison = (
        hybrid_lookahead_results[str(args.lookahead_k[-1])]
        if args.lookahead_k else hybrid_fixed
    )
    if joint_results:
        joint_key = str(args.configuration_step)
        aegaeon_routing = joint_results[joint_key]["routing"]
        aegaeon_routing_source = "layerweave_joint"
        layerweave_comparison = joint_results[joint_key]
    aegaeon = simulate_aegaeon(
        requests, layouts, estimator, args.pool_pages, args.gpus,
        args.kv_block_tokens, aegaeon_routing,
        args.decode_ms_per_token, args.aegaeon_gpu_copy_gbps)
    layerweave_loading = (
        layerweave_comparison["summary"]["exposed_load_ms"])
    aegaeon_loading = aegaeon["summary"]["loading_stall_ms"]
    cold_profiles = {}
    for model_id, layout in layouts.items():
        cold = predict(
            estimator, representatives[model_id], layout, set())
        cold_profiles[str(model_id)] = {
            **cold,
            "hidden_load_ms": max(
                0.0, cold["h2d_ms"] - cold["exposed_load_ms"]),
            "hidden_fraction": (
                max(0.0, cold["h2d_ms"] - cold["exposed_load_ms"])
                / cold["h2d_ms"]
                if cold["h2d_ms"] else 0.0
            ),
        }
    minimal_critical = minimal["summary"]["critical_path_ms"]["mean"]
    comparison_policy = (
        layerweave_fixed if layerweave_fixed is not None else hybrid_fixed)
    fixed_critical = (
        comparison_policy["summary"]["critical_path_ms"]["mean"])
    minimal_exposed = minimal["summary"]["exposed_load_ms"]["mean"]
    fixed_exposed = (
        comparison_policy["summary"]["exposed_load_ms"]["mean"])
    five_way_ablation = (
        build_five_way_ablation(
            requests, layouts, estimator, minimal, layerweave_fixed,
            args.inspect_request_id)
        if run_all else None
    )
    return {
        "model": {
            "profile": str(args.profile.resolve()),
            "calibration": [
                str(path.resolve()) for path in args.calibration
            ],
            "trace": str(args.trace.resolve()),
            "trace_mode": args.trace_mode,
            "source_trace_requests": args.max_requests,
            "simulated_requests": len(requests),
            "config": str(args.config.resolve()),
            "pool_pages_per_gpu": args.pool_pages,
            "request_cap_pool_pages": (
                args.request_cap_pool_pages
                if args.request_cap_pool_pages > 0
                else args.pool_pages
            ),
            "page_size_mib": 64,
            "kv_block_tokens": args.kv_block_tokens,
            "gpus": args.gpus,
            "demand_decay": args.demand_decay,
            "lookahead_k": args.lookahead_k,
            "lookahead_discount": args.lookahead_discount,
            "transition_beam_width": args.transition_beam_width,
            "policy_suite": args.policy_suite,
            "configuration_step": args.configuration_step,
            "joint_routing": args.joint_routing,
            "transition_weight": args.transition_weight,
            "transition_credit_cap_ms": args.transition_credit_cap_ms,
            "decode_ms_per_token": args.decode_ms_per_token,
            "aegaeon_gpu_copy_gbps": args.aegaeon_gpu_copy_gbps,
            "aegaeon_routing_source": aegaeon_routing_source,
            "execution": "system-wide serial request order",
            "minimal_eviction":
                "model-frequency value, page-LRU tie break",
            "pipeline_eviction":
                "model-frequency times marginal critical-path value",
            "configuration_planning":
                "multi-shape expected prefix gain times decay-0.9 online "
                "demand, per-GPU MCKP, protected floors, and deferred "
                "soft-suffix reclamation",
            "hybrid_planning":
                "MCKP protected floors plus demand-weighted pipeline "
                "page-greedy reclamation over all inactive soft pages",
        },
        "configuration_curves": {
            str(model_id): [
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"pages", "shape_gains_ms"}
                }
                for candidate in candidates
            ]
            for model_id, candidates in configuration_candidates.items()
        },
        "layouts": {
            str(model_id): {
                "page_count": layout.page_count,
                "unique_pages_by_stage": layout.stage_pages,
                "representative_input_tokens":
                    representatives[model_id].input_tokens,
                "cold_profile": cold_profiles[str(model_id)],
            }
            for model_id, layout in layouts.items()
        },
        "minimal": minimal,
        "layerweave_independent_routing": layerweave_independent,
        "layerweave_minimal_routing_replay": layerweave_fixed,
        "layerweave_mckp_page_greedy_minimal_routing_replay":
            hybrid_fixed,
        "layerweave_page_greedy_independent_routing":
            page_greedy_independent,
        "layerweave_page_greedy_minimal_routing_replay":
            page_greedy_fixed,
        "layerweave_lookahead_minimal_routing_replay":
            lookahead_results,
        "layerweave_mckp_page_greedy_lookahead_minimal_routing_replay":
            hybrid_lookahead_results,
        "layerweave_transition_lookahead_minimal_routing_replay":
            transition_results,
        "layerweave_residency_transition_minimal_routing_replay":
            residency_transition_results,
        "layerweave_joint_routing_mckp_page_greedy": joint_results,
        "aegaeon": aegaeon,
        "aegaeon_vs_layerweave": {
            "comparison_control": (
                "same requests and routing; Aegaeon uses trace decode length "
                "only for its prefetch window"
            ),
            "layerweave_policy": layerweave_comparison["policy"],
            "routing_source": aegaeon_routing_source,
            "layerweave_loading_stall_ms": layerweave_loading,
            "aegaeon_loading_stall_ms": aegaeon_loading,
            "layerweave_loading_stall_gain_ms": {
                key: aegaeon_loading[key] - layerweave_loading[key]
                for key in ("mean", "p50", "p90", "p95", "p99", "max")
            },
            "layerweave_loading_stall_gain_fraction": {
                key: (
                    (aegaeon_loading[key] - layerweave_loading[key])
                    / aegaeon_loading[key]
                    if aegaeon_loading[key] else 0.0
                )
                for key in ("mean", "p50", "p90", "p95", "p99", "max")
            },
        },
        "five_way_ablation": five_way_ablation,
        "fixed_routing_comparison": {
            "critical_path_gain_ms":
                minimal_critical - fixed_critical,
            "critical_path_gain_fraction": (
                (minimal_critical - fixed_critical) / minimal_critical
                if minimal_critical else 0.0
            ),
            "exposed_load_gain_ms":
                minimal_exposed - fixed_exposed,
            "exposed_load_gain_fraction": (
                (minimal_exposed - fixed_exposed) / minimal_exposed
                if minimal_exposed else 0.0
            ),
            "additional_h2d_ms": (
                comparison_policy["summary"]["h2d_ms"]["mean"]
                - minimal["summary"]["h2d_ms"]["mean"]
            ),
            "additional_missing_pages": (
                comparison_policy["summary"]["missing_pages"]
                - minimal["summary"]["missing_pages"]
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", type=Path,
        default=Path(
            "docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json"))
    parser.add_argument(
        "--calibration", type=Path, nargs="+",
        default=[Path(
            "docs/m4-m5.5-b1-scale4/"
            "m4-calibration-b1-p16-f0.json")])
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "configs/servegen_8_models_layerpipe_l40_pool42.json"))
    parser.add_argument(
        "--trace", type=Path,
        default=Path("evaluation/traces/servegen_tangram.trace"))
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument(
        "--trace-mode",
        choices=("all", "model-switches"),
        default="model-switches",
        help=(
            "Replay every source request, or keep only the first request "
            "of each consecutive same-model run. --max-requests always "
            "limits source rows before filtering. Defaults to "
            "model-switches."))
    parser.add_argument("--input-scale", type=float, default=4.0)
    parser.add_argument("--output-tokens-override", type=int, default=1)
    parser.add_argument(
        "--decode-ms-per-token", type=float, default=40.0,
        help=(
            "Aegaeon overlap-window estimate per real trace decode token; "
            "does not change request KV sizing."))
    parser.add_argument(
        "--aegaeon-gpu-copy-gbps", type=float, default=864.0,
        help=(
            "Effective bandwidth for Aegaeon's prefetched-model relocation "
            "inside the GPU."))
    parser.add_argument("--pool-pages", type=int, default=672)
    parser.add_argument(
        "--request-cap-pool-pages", type=int, default=0,
        help=(
            "Pool size used only to cap trace input lengths; 0 uses "
            "--pool-pages. Fix this across capacity sweeps for matched "
            "requests."))
    parser.add_argument("--kv-block-tokens", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--demand-decay", type=float, default=0.9)
    parser.add_argument(
        "--policy-suite",
        choices=("all", "mckp-page-greedy"),
        default="all",
        help=(
            "Run every simulator policy, or isolate Minimal plus "
            "MCKP+PageGreedy without transition/Exact-residency planners"))
    parser.add_argument(
        "--lookahead-k", type=int, nargs="*", default=[],
        help=(
            "Fixed-routing per-GPU future queue lengths to sweep for "
            "configuration-MCKP oracle lookahead"))
    parser.add_argument(
        "--lookahead-discount", type=float, default=1.0)
    parser.add_argument(
        "--configuration-step", type=int, default=0,
        help=(
            "Protected-prefix layer step; 0 keeps the coarse "
            "0/2/4/8/16/32/full configuration set"))
    parser.add_argument(
        "--joint-routing", action="store_true",
        help=(
            "Jointly choose the GPU and MCKP+PageGreedy cache placement "
            "using bounded future cache-transition cost"))
    parser.add_argument("--transition-weight", type=float, default=0.1)
    parser.add_argument(
        "--transition-credit-cap-ms", type=float, default=100.0)
    parser.add_argument(
        "--transition-beam-width", type=int, default=8)
    parser.add_argument(
        "--inspect-request-id", type=int,
        help=(
            "Request ID for the detailed five-way timeline; defaults to the "
            "median positive LayerWeave-vs-Minimal gain request"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "docs/layerweave-pipeline-cache-sim-100.json"))
    args = parser.parse_args()
    if args.gpus <= 0 or args.pool_pages <= 0:
        parser.error("--gpus and --pool-pages must be positive")
    if args.request_cap_pool_pages < 0:
        parser.error("--request-cap-pool-pages must be non-negative")
    if not 0.0 <= args.demand_decay <= 1.0:
        parser.error("--demand-decay must be between 0 and 1")
    if any(value <= 0 for value in args.lookahead_k):
        parser.error("--lookahead-k values must be positive")
    if not 0.0 < args.lookahead_discount <= 1.0:
        parser.error("--lookahead-discount must be in (0, 1]")
    if args.transition_beam_width <= 0:
        parser.error("--transition-beam-width must be positive")
    if args.configuration_step < 0:
        parser.error("--configuration-step must be non-negative")
    if args.transition_weight < 0.0:
        parser.error("--transition-weight must be non-negative")
    if args.transition_credit_cap_ms < 0.0:
        parser.error("--transition-credit-cap-ms must be non-negative")
    if args.decode_ms_per_token < 0.0:
        parser.error("--decode-ms-per-token must be non-negative")
    if args.aegaeon_gpu_copy_gbps <= 0.0:
        parser.error("--aegaeon-gpu-copy-gbps must be positive")
    run_started = time.perf_counter()
    result = run(args)
    result["model"]["simulator_wall_time_ms"] = (
        time.perf_counter() - run_started) * 1000.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    output_summary = {
        "policy_suite": args.policy_suite,
        "simulator_wall_time_ms":
            result["model"]["simulator_wall_time_ms"],
        "minimal": result["minimal"]["summary"],
        "layerweave_mckp_page_greedy_minimal_routing_replay":
            result[
                "layerweave_mckp_page_greedy_minimal_routing_replay"
            ]["summary"],
        "layerweave_mckp_page_greedy_lookahead_minimal_routing_replay": {
            key: value["summary"]
            for key, value in result[
                "layerweave_mckp_page_greedy_lookahead_minimal_routing_replay"
            ].items()
        },
        "layerweave_joint_routing_mckp_page_greedy": {
            key: value["summary"]
            for key, value in result[
                "layerweave_joint_routing_mckp_page_greedy"].items()
        },
        "aegaeon": result["aegaeon"]["summary"],
        "aegaeon_vs_layerweave": result["aegaeon_vs_layerweave"],
    }
    if args.policy_suite == "all":
        ablation = result["five_way_ablation"]
        output_summary["five_way_ablation"] = {
            "stages": ablation["stages"],
            "inspected_request": {
                key: ablation["inspected_request"][key]
                for key in (
                    "selection", "request_id", "model_id",
                    "input_tokens", "output_tokens")
            },
            "layerweave_gain_vs_minimal_ms":
                ablation["layerweave_gain_vs_minimal_ms"],
            "layerweave_gain_vs_minimal_fraction":
                ablation["layerweave_gain_vs_minimal_fraction"],
        }
        output_summary["layerweave_independent_routing"] = (
            result["layerweave_independent_routing"]["summary"])
        output_summary["layerweave_minimal_routing_replay"] = (
            result["layerweave_minimal_routing_replay"]["summary"])
        output_summary[
            "layerweave_page_greedy_minimal_routing_replay"] = (
                result[
                    "layerweave_page_greedy_minimal_routing_replay"
                ]["summary"])
        output_summary[
            "layerweave_lookahead_minimal_routing_replay"] = {
                key: value["summary"]
                for key, value in result[
                    "layerweave_lookahead_minimal_routing_replay"].items()
            }
        output_summary[
            "layerweave_transition_lookahead_minimal_routing_replay"] = {
            key: value["summary"]
            for key, value in result[
                "layerweave_transition_lookahead_minimal_routing_replay"
            ].items()
            }
        output_summary[
            "layerweave_residency_transition_minimal_routing_replay"] = {
            key: value["summary"]
            for key, value in result[
                "layerweave_residency_transition_minimal_routing_replay"
            ].items()
            }
    print("LAYERWEAVE_PIPELINE_CACHE_SIM=" + json.dumps(
        output_summary, sort_keys=True))
    print(f"LAYERWEAVE_PIPELINE_CACHE_SIM_RESULT={args.output}")


if __name__ == "__main__":
    main()
