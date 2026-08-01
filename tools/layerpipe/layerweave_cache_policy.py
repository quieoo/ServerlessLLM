#!/usr/bin/env python3
"""Configuration planning plus deferred LayerWeave page reclamation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from layerweave_estimator import PipelineEstimator


PREFIX_CANDIDATES = ("0", "2", "4", "8", "16", "32", "full")


@dataclass
class ModelDemand:
    score: float = 0.0
    batch_size: int = 1
    input_tokens: int = 1
    max_input_tokens: int = 1
    sum_input_tokens_squared: int = 1


@dataclass
class ModelCacheState:
    """Logical protection state; physical residency remains VMM-owned."""

    protected_configuration: str = "0"


class LayerWeaveSingleGpuCachePolicy:
    """Plan protected configurations and reclaim only unprotected soft pages.

    Launch configurations are logical protected-residency floors. Physical
    pages above those floors remain resident as soft cache until the next
    dispatch genuinely needs capacity. The MCKP changes only protection
    metadata; cuMemUnmap is restricted to inactive soft pages.
    """

    def __init__(
        self,
        controllers: Dict[int, object],
        profile_path: Path,
        pool_pages: int,
        expected_input_scale: float,
        decay: float = 0.9,
        uncertainty_ms: float = 100.0,
        policy_mode: str = "demand",
    ):
        if not 0.0 <= decay <= 1.0:
            raise ValueError("M5.5 demand decay must be between 0 and 1")
        if pool_pages <= 0:
            raise ValueError("M5.5 requires a positive shared-pool size")
        document = json.loads(Path(profile_path).read_text())
        scope = document.get("m4_scope", {})
        profile_scale = float(scope.get("input_scale", 1.0))
        if not math.isclose(
                profile_scale, expected_input_scale,
                rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "M5.5 runtime INPUT_SCALE does not match M4 profile: "
                f"runtime={expected_input_scale:g}, "
                f"profile={profile_scale:g}")
        if scope.get("batch_sizes") != [1]:
            raise ValueError(
                "M5.5 v3 requires a batch-size-1 M4 profile")
        if int(scope.get("output_tokens_override", 1)) != 1:
            raise ValueError(
                "M5.5 v3 requires an output_tokens_override=1 profile")
        profiles = document.get("profiles")
        if not profiles:
            raise ValueError(
                f"M4 report has no estimator profiles: {profile_path}")
        self.estimator = PipelineEstimator()
        self.estimator.models = {}
        for model_id, profile in profiles.items():
            profile = dict(profile)
            profile["compute"] = {
                int(layer): value
                for layer, value in profile.get("compute", {}).items()
            }
            self.estimator.models[int(model_id)] = profile
        self.controllers = controllers
        missing = sorted(set(controllers) - set(self.estimator.models))
        if missing:
            raise ValueError(f"M4 report is missing models: {missing}")
        page_sizes = {item.page_size for item in controllers.values()}
        if len(page_sizes) != 1:
            raise ValueError("M5.5 requires one common VMM page size")
        self.page_size = next(iter(page_sizes))
        self.pool_pages = int(pool_pages)
        self.decay = decay
        self.uncertainty_ms = max(0.0, uncertainty_ms)
        if policy_mode not in {"demand", "next-use"}:
            raise ValueError(
                "LayerWeave cache policy mode must be demand or next-use")
        self.policy_mode = policy_mode
        self.demands = {
            model_id: ModelDemand() for model_id in controllers
        }
        self.pending_counts = {
            model_id: 0 for model_id in controllers
        }
        self.next_use_positions = {
            model_id: math.inf for model_id in controllers
        }
        self.cache_states = {
            model_id: ModelCacheState() for model_id in controllers
        }
        # Estimator predictions for a configuration depend on the model and
        # request shape, but not on demand history. Cache that expensive base
        # curve; apply effective demand to a cheap copy on every decision.
        self._candidate_curve_cache = {}

    @staticmethod
    def _effective_configuration(controller, configuration):
        if configuration == "full":
            return "full"
        if int(configuration) >= len(controller.layer_pages):
            return "full"
        return configuration

    def observe(
        self,
        model_id: int,
        batch_size: int,
        input_tokens: int,
        max_input_tokens: int,
        sum_input_tokens_squared: int,
    ):
        for demand in self.demands.values():
            demand.score *= self.decay
        demand = self.demands[model_id]
        demand.score += 1.0
        demand.batch_size = max(1, int(batch_size))
        demand.input_tokens = max(1, int(input_tokens))
        demand.max_input_tokens = max(1, int(max_input_tokens))
        demand.sum_input_tokens_squared = max(
            1, int(sum_input_tokens_squared))

    def set_pending(self, requests: Iterable[object]):
        requests = list(requests)
        self.pending_counts = {
            model_id: 0 for model_id in self.controllers
        }
        self.next_use_positions = {
            model_id: math.inf for model_id in self.controllers
        }
        first = {}
        for position, request in enumerate(requests):
            model_id = int(request.model_id)
            if model_id not in self.pending_counts:
                continue
            self.pending_counts[model_id] += 1
            first.setdefault(model_id, request)
            self.next_use_positions[model_id] = min(
                self.next_use_positions[model_id], position)
        for model_id, request in first.items():
            demand = self.demands[model_id]
            demand.batch_size = 1
            demand.input_tokens = max(1, int(request.input_tokens))
            demand.max_input_tokens = demand.input_tokens
            demand.sum_input_tokens_squared = demand.input_tokens ** 2

    def _hypothetical_batch(
        self, model_id, controller, retained_pages
    ):
        demand = self.demands[model_id]
        resident = set(retained_pages)
        stages = []

        def add_stage(stage, required):
            missing = set(required) - resident
            stages.append({
                "stage": stage,
                "to_load_bytes": len(missing) * controller.page_size,
                "mapped_pages": len(missing),
            })
            resident.update(required)

        add_stage(
            "initial",
            controller.initial_pages | controller.layer_pages[0],
        )
        for layer in range(1, len(controller.layer_pages)):
            add_stage(layer, controller.layer_pages[layer])
        return {
            "model_id": model_id,
            "batch_size": demand.batch_size,
            "input_tokens": demand.input_tokens,
            "max_input_tokens": demand.max_input_tokens,
            "sum_input_tokens_squared":
                demand.sum_input_tokens_squared,
            "_prefetch": True,
            "_phase": "steady",
            "layerweave": {
                "page_count": len(controller.required_pages),
                "stages": stages,
                "computes": [
                    {"layer": layer}
                    for layer in range(len(controller.layer_pages))
                ],
            },
        }

    def _score_candidates(self, model_id, controller):
        demand = self.demands[model_id]
        if not hasattr(self, "_candidate_curve_cache"):
            self._candidate_curve_cache = {}
        shape_key = (
            int(model_id),
            int(demand.batch_size),
            int(demand.input_tokens),
            int(demand.max_input_tokens),
            int(demand.sum_input_tokens_squared),
        )
        base_candidates = self._candidate_curve_cache.get(shape_key)
        if base_candidates is None:
            base_candidates = []
            seen = set()
            best_ttft = float("inf")
            cold_ttft = None
            for configuration in PREFIX_CANDIDATES:
                pages = frozenset(
                    controller.prefix_cache_pages(configuration))
                if pages in seen:
                    continue
                seen.add(pages)
                raw_prediction = self.estimator.predict(
                    self._hypothetical_batch(
                        model_id, controller, pages)
                )["predicted_service_ttft_ms"]
                prediction = min(best_ttft, raw_prediction)
                best_ttft = prediction
                if cold_ttft is None:
                    cold_ttft = prediction
                gain = max(0.0, cold_ttft - prediction)
                base_candidates.append({
                    "configuration": self._effective_configuration(
                        controller, configuration),
                    "pages": pages,
                    "page_count": len(pages),
                    "predicted_service_ttft_ms": prediction,
                    "predicted_gain_ms": gain,
                })
            self._candidate_curve_cache[shape_key] = base_candidates
        effective_demand = (
            demand.score + float(self.pending_counts[model_id]))
        return [
            {
                **candidate,
                "effective_demand": effective_demand,
                "expected_value_ms": (
                    effective_demand
                    * float(candidate["predicted_gain_ms"])
                ),
            }
            for candidate in base_candidates
        ]

    @staticmethod
    def _configuration_rank(configuration):
        return (
            len(PREFIX_CANDIDATES)
            if configuration == "full"
            else PREFIX_CANDIDATES.index(str(configuration))
        )

    @staticmethod
    def _pareto_candidates(candidates):
        """Drop configurations dominated in both pages and expected value."""
        frontier = []
        best_value = -1.0
        for candidate in sorted(
                candidates,
                key=lambda item: (
                    item["page_count"], -item["expected_value_ms"])):
            if candidate["expected_value_ms"] <= best_value:
                continue
            frontier.append(candidate)
            best_value = candidate["expected_value_ms"]
        return frontier

    def _allowed_candidates(
        self, model_id, candidates, active_model_id
    ):
        candidates = self._pareto_candidates(candidates)
        if model_id == active_model_id:
            return candidates
        current_rank = self._configuration_rank(
            self.cache_states[model_id].protected_configuration)
        allowed = [
            candidate for candidate in candidates
            if self._configuration_rank(candidate["configuration"])
            <= current_rank
        ]
        if not allowed:
            raise RuntimeError(
                f"Model {model_id} has no non-upgrading cache configuration")
        return allowed

    @staticmethod
    def _solve_mckp(
        model_candidates: Dict[int, List[dict]],
        capacity_pages: int,
    ):
        """Exact page-granular MCKP for the small per-GPU model set."""
        states = {0: (0.0, {})}
        for model_id in sorted(model_candidates):
            next_states = {}
            for used, (value, choices) in states.items():
                for candidate in model_candidates[model_id]:
                    new_used = used + int(candidate["page_count"])
                    if new_used > capacity_pages:
                        continue
                    new_value = value + float(
                        candidate["expected_value_ms"])
                    previous = next_states.get(new_used)
                    if previous is None or new_value > previous[0]:
                        next_states[new_used] = (
                            new_value,
                            {**choices, model_id: candidate},
                        )
            if not next_states:
                raise RuntimeError(
                    "No feasible protected-cache configuration: "
                    f"model={model_id}, capacity_pages={capacity_pages}")
            # A state is useless when a no-larger state has at least its value.
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

    def _marginal_tiers(self, model_id, controller):
        candidates = self._score_candidates(model_id, controller)
        tier_count = max(1, len(candidates) - 1)
        tiers = []
        page_metadata = {}
        for index, (lower, upper) in enumerate(
                zip(candidates, candidates[1:]), start=1):
            added = set(upper["pages"]) - set(lower["pages"])
            marginal_gain = max(
                0.0,
                lower["predicted_service_ttft_ms"] -
                upper["predicted_service_ttft_ms"],
            )
            conservative_gain = max(
                0.0,
                marginal_gain - self.uncertainty_ms / tier_count,
            )
            value_per_page = (
                upper["effective_demand"] * conservative_gain / len(added)
                if added else 0.0
            )
            label = (
                f"{lower['configuration']}->{upper['configuration']}")
            tier = {
                "index": index,
                "label": label,
                "added_pages": len(added),
                "marginal_gain_ms": marginal_gain,
                "conservative_gain_ms": conservative_gain,
                "value_per_page": value_per_page,
            }
            tiers.append(tier)
            for page in added:
                page_metadata[page] = tier
        return candidates, tiers, page_metadata

    def estimate_dispatch(
        self,
        active_model_id: int,
        batch: Iterable[object],
        pending: Iterable[object],
        request_kv_pages: int,
        current_kv_pages: int = 0,
        residencies: Dict[int, Iterable[int]] | None = None,
    ) -> dict:
        """Read-only route estimate for a candidate GPU.

        This mirrors the value calculation used by ``plan_dispatch`` without
        changing demand history, protected floors, residency, or physical
        mappings.  A multi-GPU coordinator can therefore compare workers
        before committing a request to exactly one of them.
        """
        batch = list(batch)
        pending = list(pending)
        if not batch:
            raise ValueError("estimate_dispatch requires a non-empty batch")
        controller = self.controllers[active_model_id]
        if residencies is None:
            residencies = {
                model_id: {
                    page for page, resident in enumerate(
                        item.pool.layerweave_residency(item.model_path))
                    if resident
                }
                for model_id, item in self.controllers.items()
            }
        else:
            residencies = {
                int(model_id): set(pages)
                for model_id, pages in residencies.items()
            }
        saved_demands = {
            model_id: (
                demand.score,
                demand.batch_size,
                demand.input_tokens,
                demand.max_input_tokens,
                demand.sum_input_tokens_squared,
            )
            for model_id, demand in self.demands.items()
        }
        saved_pending = dict(self.pending_counts)
        saved_next_use = dict(self.next_use_positions)
        try:
            self.observe(
                model_id=active_model_id,
                batch_size=len(batch),
                input_tokens=sum(
                    int(item.input_tokens) for item in batch),
                max_input_tokens=max(
                    int(item.input_tokens) for item in batch),
                sum_input_tokens_squared=sum(
                    int(item.input_tokens) ** 2 for item in batch),
            )
            self.set_pending(pending)
            prediction = self.estimator.predict(
                self._hypothetical_batch(
                    active_model_id,
                    controller,
                    residencies[active_model_id],
                )
            )
            marginal = {
                model_id: self._marginal_tiers(model_id, item)
                for model_id, item in self.controllers.items()
            }
            protected_capacity = max(
                0,
                self.pool_pages - int(current_kv_pages)
                - int(request_kv_pages)
                - len(controller.required_pages),
            )
            inactive_candidates = {
                model_id: self._allowed_candidates(
                    model_id, marginal[model_id][0], active_model_id)
                for model_id in self.controllers
                if model_id != active_model_id
            }
            solution = self._solve_mckp(
                inactive_candidates, protected_capacity)
            active_candidates = self._allowed_candidates(
                active_model_id,
                marginal[active_model_id][0],
                active_model_id,
            )
            active_choice = max(
                active_candidates,
                key=lambda item: (
                    item["expected_value_ms"], item["page_count"]))
            value_before = 0.0
            for model_id, candidates in marginal.items():
                current = self.cache_states[
                    model_id].protected_configuration
                value_before += next(
                    (
                        candidate["expected_value_ms"]
                        for candidate in candidates[0]
                        if candidate["configuration"] == current
                    ),
                    0.0,
                )
            value_after = (
                float(solution["expected_value_ms"])
                + float(active_choice["expected_value_ms"])
            )
            return {
                "predicted_service_ttft_ms": float(
                    prediction["predicted_service_ttft_ms"]),
                # This is the PSE prediction that is directly comparable to
                # TangramLayerWeaveController.metrics()["exposed_load_ms"].
                # Keep it separate from service TTFT so compute, host gaps,
                # Decode, and queueing cannot contaminate the accuracy study.
                "predicted_exposed_load_ms": float(
                    prediction.get(
                        "predicted_gpu_ready_stall_ms",
                        prediction["predicted_service_ttft_ms"],
                    )),
                "transition_cost_ms": float(value_before - value_after),
                "protected_capacity_pages": protected_capacity,
                "resident_pages": len(residencies[active_model_id]),
                "required_pages": len(controller.required_pages),
                "missing_pages": len(
                    set(controller.required_pages)
                    - residencies[active_model_id]),
            }
        finally:
            for model_id, values in saved_demands.items():
                demand = self.demands[model_id]
                (
                    demand.score,
                    demand.batch_size,
                    demand.input_tokens,
                    demand.max_input_tokens,
                    demand.sum_input_tokens_squared,
                ) = values
            self.pending_counts = saved_pending
            self.next_use_positions = saved_next_use

    def plan_dispatch(
        self,
        active_model_id: int,
        batch: Iterable[object],
        pending: Iterable[object],
        kv_block_size_tokens: int,
        kv_block_size_bytes: int,
        current_kv_pages: int,
        residencies: Dict[int, Iterable[int]] | None = None,
    ):
        plan_started = time.perf_counter()
        batch = list(batch)
        self.observe(
            model_id=active_model_id,
            batch_size=len(batch),
            input_tokens=sum(item.input_tokens for item in batch),
            max_input_tokens=max(
                (item.input_tokens for item in batch), default=1),
            sum_input_tokens_squared=sum(
                item.input_tokens * item.input_tokens for item in batch),
        )
        self.set_pending(pending)
        residency_started = time.perf_counter()
        if residencies is None:
            residencies = {
                model_id: {
                    page for page, resident in enumerate(
                        controller.pool.layerweave_residency(
                            controller.model_path))
                    if resident
                }
                for model_id, controller in self.controllers.items()
            }
        else:
            residencies = {
                int(model_id): set(pages)
                for model_id, pages in residencies.items()
            }
        residency_query_ms = (
            time.perf_counter() - residency_started) * 1000.0
        marginal_started = time.perf_counter()
        marginal = (
            {}
            if self.policy_mode == "next-use"
            else {
                model_id: self._marginal_tiers(model_id, controller)
                for model_id, controller in self.controllers.items()
            }
        )
        marginal_ms = (time.perf_counter() - marginal_started) * 1000.0
        controller = self.controllers[active_model_id]
        missing_weight_pages = len(
            set(controller.required_pages) - residencies[active_model_id])
        kv_blocks = sum(
            math.ceil(
                (int(item.input_tokens) + int(item.output_tokens)) /
                kv_block_size_tokens
            )
            for item in batch
        )
        kv_pages_per_block = math.ceil(
            kv_block_size_bytes / self.page_size)
        request_kv_pages = kv_blocks * kv_pages_per_block
        resident_weight_pages = sum(map(len, residencies.values()))
        current_kv_pages = int(current_kv_pages)
        free_pages_before = max(
            0, self.pool_pages -
            resident_weight_pages - current_kv_pages)
        reservation_pages = missing_weight_pages + request_kv_pages
        eviction_required = max(
            0, reservation_pages - free_pages_before)

        # During the request the full active model is non-reclaimable. Its
        # final protection choice therefore consumes no additional dispatch
        # capacity; all inactive protected floors must fit beside that active
        # working set and the request's KV reservation.
        protected_capacity = max(
            0,
            self.pool_pages - current_kv_pages - request_kv_pages
            - len(controller.required_pages),
        )
        solve_started = time.perf_counter()
        if self.policy_mode == "next-use":
            choices = {
                model_id: {
                    "configuration": (
                        "full" if model_id == active_model_id else "0"),
                    "pages": (
                        frozenset(item.required_pages)
                        if model_id == active_model_id else frozenset()),
                    "page_count": (
                        len(item.required_pages)
                        if model_id == active_model_id else 0),
                    "expected_value_ms": 0.0,
                }
                for model_id, item in self.controllers.items()
            }
            solution = {
                "choices": {
                    model_id: choice
                    for model_id, choice in choices.items()
                    if model_id != active_model_id
                },
                "used_pages": 0,
                "expected_value_ms": 0.0,
            }
        else:
            inactive_candidates = {
                model_id: self._allowed_candidates(
                    model_id, marginal[model_id][0], active_model_id)
                for model_id in self.controllers
                if model_id != active_model_id
            }
            solution = self._solve_mckp(
                inactive_candidates, protected_capacity)
            active_candidates = self._allowed_candidates(
                active_model_id,
                marginal[active_model_id][0],
                active_model_id,
            )
            # The full active working set is resident during execution, so
            # retain its highest-value configuration as the protection floor.
            active_choice = max(
                active_candidates,
                key=lambda item: (
                    item["expected_value_ms"], item["page_count"]))
            choices = dict(solution["choices"])
            choices[active_model_id] = active_choice
        mckp_solve_ms = (time.perf_counter() - solve_started) * 1000.0

        value_before = 0.0
        if self.policy_mode != "next-use":
            for model_id, candidates in marginal.items():
                current = self.cache_states[
                    model_id].protected_configuration
                value_before += next(
                    (
                        candidate["expected_value_ms"]
                        for candidate in candidates[0]
                        if candidate["configuration"] == current
                    ),
                    0.0,
                )
        value_after = sum(
            float(choice["expected_value_ms"])
            for choice in choices.values())
        transition_cost = value_before - value_after
        for model_id, choice in choices.items():
            self.cache_states[
                model_id].protected_configuration = choice["configuration"]

        protected = {
            model_id: set(choice["pages"])
            for model_id, choice in choices.items()
        }
        soft = {
            model_id: (
                set(residencies[model_id]) - protected[model_id]
                if model_id != active_model_id else set()
            )
            for model_id in self.controllers
        }

        # Reclaim only inactive soft pages. Higher suffix tiers are more
        # overlapable; within a tier prefer lower PSE value density.
        victim_started = time.perf_counter()
        victims = []
        for model_id, soft_pages in soft.items():
            if model_id == active_model_id:
                continue
            if self.policy_mode == "next-use":
                next_use = self.next_use_positions[model_id]
                for page in soft_pages:
                    victims.append((
                        -next_use, 0.0, model_id, page, "next-use",
                    ))
                continue
            _, tiers, page_metadata = marginal[model_id]
            for page in soft_pages:
                tier = page_metadata.get(page)
                if tier is None:
                    victims.append((
                        -len(tiers) - 1, 0.0, model_id, page,
                        "fragment",
                    ))
                    continue
                victims.append((
                    -tier["index"], tier["value_per_page"],
                    model_id, page, tier["label"],
                ))
        victims.sort(
            key=lambda item: (item[0], item[1], item[2], -item[3]))
        selected = victims[:eviction_required]
        victim_select_ms = (
            time.perf_counter() - victim_started) * 1000.0
        if len(selected) < eviction_required:
            raise RuntimeError(
                "Protected-cache MCKP cannot expose enough soft pages for "
                f"dispatch: need={eviction_required}, soft={len(selected)}")

        retained = {
            model_id: set(pages)
            for model_id, pages in residencies.items()
        }
        for _, _, model_id, page, _ in selected:
            retained[model_id].discard(page)
        apply_started = time.perf_counter()
        applied = {}
        for model_id, item in self.controllers.items():
            if model_id == active_model_id:
                continue
            before = len(residencies[model_id])
            if len(retained[model_id]) == before:
                continue
            applied[str(model_id)] = item.apply_retained_pages(
                retained[model_id], synchronize=False)
        eviction_ms = (time.perf_counter() - apply_started) * 1000.0
        plan_total_ms = (time.perf_counter() - plan_started) * 1000.0

        return {
            "mode": (
                "next_use_deferred_reclamation"
                if self.policy_mode == "next-use"
                else "protected_mckp_deferred_reclamation"
            ),
            "cache_policy_mode": self.policy_mode,
            "active_model_id": active_model_id,
            "pool_pages": self.pool_pages,
            "resident_weight_pages_before": resident_weight_pages,
            "current_kv_pages": current_kv_pages,
            "free_pages_before": free_pages_before,
            "missing_active_weight_pages": missing_weight_pages,
            "request_kv_blocks": kv_blocks,
            "kv_pages_per_block": kv_pages_per_block,
            "request_kv_pages": request_kv_pages,
            "reservation_pages": reservation_pages,
            "eviction_required_pages": eviction_required,
            "evicted_pages": len(selected),
            "evicted_soft_pages": len(selected),
            "evicted_protected_pages": 0,
            "evicted_pages_by_tier": {
                label: sum(
                    item_label == label
                    for _, _, _, _, item_label in selected)
                for label in sorted({
                    item_label
                    for _, _, _, _, item_label in selected
                })
            },
            "eviction_ms": eviction_ms,
            "plan_total_ms": plan_total_ms,
            "residency_query_ms": residency_query_ms,
            "marginal_ms": marginal_ms,
            "victim_select_ms": victim_select_ms,
            "candidate_curve_cache_entries": len(
                getattr(self, "_candidate_curve_cache", {})),
            "runtime_eviction_expected": False,
            "mckp": {
                "capacity_pages": protected_capacity,
                "used_inactive_protected_pages": solution["used_pages"],
                "solve_ms": mckp_solve_ms,
                "value_before_ms": value_before,
                "value_after_ms": value_after,
                "transition_cost_ms": transition_cost,
            },
            "targets": {
                str(model_id): {
                    "configuration": choices[model_id]["configuration"],
                    "protected_pages": len(protected[model_id]),
                    "soft_pages_before": len(soft[model_id]),
                    "active_pages": (
                        len(controller.required_pages)
                        if model_id == active_model_id else 0),
                    "resident_pages_before": len(residencies[model_id]),
                    "resident_pages_after": len(retained[model_id]),
                    "effective_demand": (
                        0.0
                        if self.policy_mode == "next-use"
                        else marginal[model_id][0][0][
                            "effective_demand"]
                    ),
                    "next_use_position": (
                        self.next_use_positions[model_id]
                        if math.isfinite(
                            self.next_use_positions[model_id])
                        else None
                    ),
                    "marginal_tiers": (
                        []
                        if self.policy_mode == "next-use"
                        else marginal[model_id][1]
                    ),
                }
                for model_id in self.controllers
            },
            "applied": applied,
        }
