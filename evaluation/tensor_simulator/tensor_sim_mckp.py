"""Prefix-stall table access and future MCKP planner primitives."""

from __future__ import annotations

import bisect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path


class PrefixStallTable:
    """Validated piecewise-linear lookup over measured input lengths."""

    def __init__(self, document, source=None, input_extrapolation="clamp"):
        if document.get("format") != "layerweave-prefix-stall-v1":
            raise ValueError("unsupported prefix stall table format")
        self.document = document
        self.source = None if source is None else str(source)
        if input_extrapolation not in ("clamp", "linear"):
            raise ValueError("input extrapolation must be clamp or linear")
        self.input_extrapolation = input_extrapolation
        self.metadata = document["metadata"]
        self.models = {}
        self._lookup_cache = {}
        self._component_cache = {}
        for key, item in document["models"].items():
            model_id = int(key)
            group_count = int(item["group_count"])
            prefix_bytes = [int(value) for value in item["prefix_bytes"]]
            if len(prefix_bytes) != group_count + 1:
                raise ValueError(
                    f"model {model_id}: prefix byte count mismatch")
            if any(a > b for a, b in zip(prefix_bytes, prefix_bytes[1:])):
                raise ValueError(
                    f"model {model_id}: prefix bytes are not monotonic")
            rows = sorted(item["rows"], key=lambda row: row["input_tokens"])
            tokens = [int(row["input_tokens"]) for row in rows]
            if not rows or len(tokens) != len(set(tokens)):
                raise ValueError(
                    f"model {model_id}: empty or duplicate input rows")
            for row in rows:
                stalls = [float(value) for value in row["prefix_stall_ms"]]
                if len(stalls) != group_count + 1:
                    raise ValueError(
                        f"model {model_id}: prefix stall count mismatch")
                if stalls[-1] != 0.0:
                    raise ValueError(
                        f"model {model_id}: full-resident stall is not zero")
                if any(
                    left < right
                    for left, right in zip(stalls, stalls[1:])
                ):
                    raise ValueError(
                        f"model {model_id}: stall is not prefix-monotonic")
                suffix = row.get("suffix_stall_ms")
                if suffix is not None:
                    suffix = [float(value) for value in suffix]
                    if len(suffix) != group_count + 1:
                        raise ValueError(
                            f"model {model_id}: suffix stall count mismatch")
                    if suffix[0] != 0.0:
                        raise ValueError(
                            f"model {model_id}: full suffix stall is not zero")
                    if any(
                        left > right
                        for left, right in zip(suffix, suffix[1:])
                    ):
                        raise ValueError(
                            f"model {model_id}: stall is not "
                            "suffix-monotonic")
            self.models[model_id] = {
                "group_count": group_count,
                "prefix_bytes": prefix_bytes,
                "rows": rows,
                "tokens": tokens,
                "groups": item.get("groups"),
            }

    @classmethod
    def load(cls, path):
        path = Path(path)
        return cls(json.loads(path.read_text()), path)

    def group_count(self, model_id):
        return self.models[int(model_id)]["group_count"]

    def prefix_bytes(self, model_id, prefix_count):
        model = self.models[int(model_id)]
        if not 0 <= prefix_count <= model["group_count"]:
            raise IndexError("prefix count is outside the model")
        return model["prefix_bytes"][prefix_count]

    def _lookup_field(
            self, model_id, input_tokens, state_count, field):
        cache_key = (
            int(model_id), int(input_tokens), int(state_count), field,
            self.input_extrapolation,
        )
        cached = self._lookup_cache.get(cache_key)
        if cached is not None:
            return cached
        model = self.models[int(model_id)]
        if not 0 <= state_count <= model["group_count"]:
            raise IndexError("residency state is outside the model")
        tokens = model["tokens"]
        rows = model["rows"]
        if field not in rows[0]:
            raise ValueError(f"stall table does not contain {field}")
        if input_tokens <= tokens[0]:
            result = float(rows[0][field][state_count])
            self._lookup_cache[cache_key] = result
            return result
        if input_tokens >= tokens[-1] and (
                self.input_extrapolation == "clamp" or len(tokens) == 1):
            result = float(rows[-1][field][state_count])
            self._lookup_cache[cache_key] = result
            return result
        if input_tokens >= tokens[-1]:
            lower, upper = len(tokens) - 2, len(tokens) - 1
        else:
            upper = bisect.bisect_right(tokens, input_tokens)
            lower = upper - 1
        left_tokens = tokens[lower]
        right_tokens = tokens[upper]
        weight = (
            (float(input_tokens) - left_tokens)
            / (right_tokens - left_tokens)
        )
        left = float(rows[lower][field][state_count])
        right = float(rows[upper][field][state_count])
        result = max(0.0, left + weight * (right - left))
        self._lookup_cache[cache_key] = result
        return result

    def lookup(self, model_id, input_tokens, prefix_count):
        return self._lookup_field(
            model_id, input_tokens, prefix_count, "prefix_stall_ms")

    def lookup_suffix(self, model_id, input_tokens, suffix_start):
        return self._lookup_field(
            model_id, input_tokens, suffix_start, "suffix_stall_ms")

    @staticmethod
    def _simulate_component_stall(
            boundaries, groups, resident, h2d_bytes_per_ms, allocation_ms):
        """Replay a profile row with selected loading-cost components."""
        if not boundaries:
            return 0.0
        actual_previous = None
        baseline_previous = None
        copy_end = 0.0
        for index, (baseline, group) in enumerate(zip(boundaries, groups)):
            baseline = float(baseline)
            compute_ready = (
                baseline if actual_previous is None else
                actual_previous + max(0.0, baseline - baseline_previous)
            )
            if bool(group.get("observed", True)) and not resident(index):
                load_ms = (
                    int(group["logical_bytes"]) / h2d_bytes_per_ms
                    if h2d_bytes_per_ms is not None else 0.0
                ) + allocation_ms
                load_start = (
                    copy_end if actual_previous is None else
                    max(copy_end, actual_previous)
                )
                copy_end = load_start + load_ms
                actual = max(compute_ready, copy_end)
            else:
                actual = compute_ready
            actual_previous = actual
            baseline_previous = baseline
        return max(0.0, actual_previous - float(boundaries[-1]))

    def _interpolate_scalar(self, model, input_tokens, value):
        tokens = model["tokens"]
        rows = model["rows"]
        if input_tokens <= tokens[0]:
            return value(rows[0])
        if input_tokens >= tokens[-1] and (
                self.input_extrapolation == "clamp" or len(tokens) == 1):
            return value(rows[-1])
        if input_tokens >= tokens[-1]:
            lower, upper = len(tokens) - 2, len(tokens) - 1
        else:
            upper = bisect.bisect_right(tokens, input_tokens)
            lower = upper - 1
        weight = ((float(input_tokens) - tokens[lower]) /
                  (tokens[upper] - tokens[lower]))
        return max(0.0, value(rows[lower]) + weight * (
            value(rows[upper]) - value(rows[lower])))

    def exposed_loading_components(
            self, model_id, input_tokens, state_count, residency_kind):
        """Shapley attribution of profiled stall to H2D and allocation.

        The stored profile combines these costs before load/compute overlap.
        Replaying its measured group boundaries with each cost enabled alone
        preserves that overlap while producing an additive decomposition.
        """
        cache_key = (
            int(model_id), int(input_tokens), int(state_count),
            residency_kind, self.input_extrapolation,
        )
        cached = self._component_cache.get(cache_key)
        if cached is not None:
            return cached
        model = self.models[int(model_id)]
        groups = model.get("groups")
        h2d_gbps = self.metadata.get("h2d_gbps")
        allocation_ms = self.metadata.get("segment_allocation_ms")
        if not groups or h2d_gbps is None or allocation_ms is None or any(
                "group_boundary_ms" not in row for row in model["rows"]):
            exposed = (
                self.lookup(model_id, input_tokens, state_count)
                if residency_kind == "prefix" else
                self.lookup_suffix(model_id, input_tokens, state_count)
            )
            result = (exposed, 0.0)
            self._component_cache[cache_key] = result
            return result
        if residency_kind == "prefix":
            resident = lambda index: index < state_count
            target = self.lookup(model_id, input_tokens, state_count)
        elif residency_kind == "suffix":
            resident = lambda index: index >= state_count
            target = self.lookup_suffix(model_id, input_tokens, state_count)
        else:
            raise ValueError(f"unsupported residency kind: {residency_kind}")
        bandwidth = float(h2d_gbps) * 1_000_000.0

        def replay(row, use_h2d, use_allocation):
            return self._simulate_component_stall(
                row["group_boundary_ms"], groups, resident,
                bandwidth if use_h2d else None,
                float(allocation_ms) if use_allocation else 0.0)

        h2d_only = self._interpolate_scalar(
            model, input_tokens, lambda row: replay(row, True, False))
        allocation_only = self._interpolate_scalar(
            model, input_tokens, lambda row: replay(row, False, True))
        both = self._interpolate_scalar(
            model, input_tokens, lambda row: replay(row, True, True))
        h2d = 0.5 * (h2d_only + both - allocation_only)
        allocation = 0.5 * (allocation_only + both - h2d_only)
        attributed = h2d + allocation
        if target <= 0.0 or attributed <= 0.0:
            result = (max(0.0, target), 0.0)
            self._component_cache[cache_key] = result
            return result
        scale = target / attributed
        result = (
            max(0.0, h2d * scale), max(0.0, allocation * scale))
        self._component_cache[cache_key] = result
        return result

    def curve(self, model_id, input_tokens):
        count = self.group_count(model_id)
        values = [
            self.lookup(model_id, input_tokens, prefix)
            for prefix in range(count + 1)
        ]
        # Interpolation of two monotonic curves should remain monotonic, but
        # retain a defensive envelope for serialized/rounded tables.
        values[-1] = 0.0
        for index in range(len(values) - 2, -1, -1):
            values[index] = max(values[index], values[index + 1])
        return values

    def input_median(self, model_id):
        tokens = self.models[int(model_id)]["tokens"]
        return tokens[len(tokens) // 2]

    def baseline_ms(self, model_id, input_tokens):
        """Interpolate the uninstrumented hot-prefill wall time."""
        model = self.models[int(model_id)]
        tokens = model["tokens"]
        rows = model["rows"]

        def value(index):
            result = rows[index].get("median_baseline_wall_ms")
            return 0.0 if result is None else float(result)

        if input_tokens <= tokens[0]:
            return value(0)
        if input_tokens >= tokens[-1] and (
                self.input_extrapolation == "clamp" or len(tokens) == 1):
            return value(-1)
        if input_tokens >= tokens[-1]:
            lower, upper = len(tokens) - 2, len(tokens) - 1
        else:
            upper = bisect.bisect_right(tokens, input_tokens)
            lower = upper - 1
        weight = (
            (float(input_tokens) - tokens[lower])
            / (tokens[upper] - tokens[lower])
        )
        return value(lower) + weight * (value(upper) - value(lower))


@dataclass(frozen=True)
class PrefixOption:
    model_id: int
    prefix_count: int
    freed_bytes: int
    damage_ms: float
    evicted_groups: int


@dataclass
class MckpEvictionPlan:
    feasible: bool
    required_free_bytes: int
    planned_free_bytes: int
    predicted_future_damage_ms: float
    choices: dict[int, int]
    victim_keys: list[tuple[int, int]]
    solver_time_ms: float
    frontier_peak_states: int
    options_considered: int

    @property
    def overrelease_bytes(self):
        return max(0, self.planned_free_bytes - self.required_free_bytes)


def build_model_options(
        table, model_id, resident_prefix, prediction_samples,
        prefix_bytes=None):
    """Enumerate all suffix-release choices for one model.

    prediction_samples contains ``(input_tokens, weight)`` pairs.  Weights may
    be decayed historical probabilities or discounted lookahead occurrences.
    """
    if prefix_bytes is None:
        prefix_bytes = table.prefix_bytes
    current_bytes = prefix_bytes(model_id, resident_prefix)
    baselines = [
        (tokens, float(weight),
         table.lookup(model_id, tokens, resident_prefix))
        for tokens, weight in prediction_samples
        if weight > 0
    ]
    options = []
    for prefix in range(resident_prefix, -1, -1):
        damage = sum(
            weight * max(
                0.0, table.lookup(model_id, tokens, prefix) - baseline)
            for tokens, weight, baseline in baselines
        )
        options.append(PrefixOption(
            model_id=model_id,
            prefix_count=prefix,
            freed_bytes=(
                current_bytes - prefix_bytes(model_id, prefix)),
            damage_ms=damage,
            evicted_groups=resident_prefix - prefix,
        ))
    return options


def _state_rank(state, required_free_bytes):
    freed, cost, evicted, choices = state
    return (
        cost,
        max(0, freed - required_free_bytes),
        evicted,
        choices,
    )


def _prune_frontier(states, required_free_bytes):
    """Retain the exact cost/freed Pareto frontier plus best feasible state."""
    best_by_freed = {}
    for state in states:
        freed = state[0]
        old = best_by_freed.get(freed)
        if old is None or _state_rank(
                state, required_free_bytes) < _state_rank(
                    old, required_free_bytes):
            best_by_freed[freed] = state
    feasible = [
        state for state in best_by_freed.values()
        if state[0] >= required_free_bytes
    ]
    best_feasible = (
        min(feasible, key=lambda state: _state_rank(
            state, required_free_bytes))
        if feasible else None
    )
    result = []
    best_cost = math.inf
    # For increasing freed bytes, a state survives iff no state with at least
    # as many freed bytes has lower/equal cost.
    for state in sorted(
            (item for item in best_by_freed.values()
             if item[0] < required_free_bytes),
            key=lambda item: item[0], reverse=True):
        if state[1] < best_cost:
            result.append(state)
            best_cost = state[1]
    result.reverse()
    if best_feasible is not None:
        result.append(best_feasible)
    return result


def solve_covering_mckp(
        table, resident_prefixes, active_model, required_free_bytes,
        prediction_samples, prefix_bytes=None):
    """Solve model-choice minimum-cost covering reclamation exactly."""
    started = time.perf_counter()
    required = max(0, int(required_free_bytes))
    choices = {
        int(model_id): int(prefix)
        for model_id, prefix in resident_prefixes.items()
    }
    if required == 0:
        return MckpEvictionPlan(
            True, 0, 0, 0.0, choices, [],
            (time.perf_counter() - started) * 1000.0, 1, 0)
    # Models are processed in sorted order, so this append-only tuple is
    # already the deterministic choice rank. Avoid a dict allocation and
    # sorted(items) call for every expanded frontier state.
    frontier = [(0, 0.0, 0, ())]
    peak = 1
    considered = 0
    for model_id in sorted(resident_prefixes):
        if model_id == active_model:
            continue
        options = build_model_options(
            table, model_id, resident_prefixes[model_id],
            prediction_samples.get(model_id, ()), prefix_bytes)
        considered += len(options)
        expanded = []
        for freed, cost, evicted, selected in frontier:
            for option in options:
                expanded.append((
                    freed + option.freed_bytes,
                    cost + option.damage_ms,
                    evicted + option.evicted_groups,
                    selected + ((model_id, option.prefix_count),),
                ))
        frontier = _prune_frontier(expanded, required)
        peak = max(peak, len(frontier))
    feasible = [state for state in frontier if state[0] >= required]
    elapsed = (time.perf_counter() - started) * 1000.0
    if not feasible:
        return MckpEvictionPlan(
            False, required, 0, math.inf, choices, [], elapsed, peak,
            considered)
    freed, cost, _, selected = min(
        feasible, key=lambda state: _state_rank(state, required))
    choices.update(dict(selected))
    victims = [
        (model_id, group_id)
        for model_id, old_prefix in sorted(resident_prefixes.items())
        for group_id in range(choices[model_id], old_prefix)
        if model_id != active_model
    ]
    return MckpEvictionPlan(
        True, required, freed, cost, choices, victims, elapsed, peak,
        considered)
