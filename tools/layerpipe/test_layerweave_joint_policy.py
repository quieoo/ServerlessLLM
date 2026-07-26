#!/usr/bin/env python3
"""CPU-only invariants for protected planning and deferred reclamation."""

import unittest
from dataclasses import dataclass

from layerweave_cache_policy import (
    LayerWeaveSingleGpuCachePolicy,
    ModelCacheState,
    ModelDemand,
)


class FakePool:
    def __init__(self, residencies):
        self.residencies = residencies

    def layerweave_residency(self, model_path):
        return list(self.residencies[model_path])


class FakeController:
    def __init__(self, pool, model_path):
        self.pool = pool
        self.model_path = model_path
        self.page_size = 1
        self.required_pages = set(range(6))
        self.initial_pages = set()
        self.layer_pages = [{page} for page in range(6)]
        self.applied = None

    def apply_retained_pages(self, pages, synchronize=True):
        self.applied = set(pages)
        self.pool.residencies[self.model_path] = [
            page in self.applied for page in range(6)
        ]
        return {"resident_pages": len(self.applied)}


@dataclass
class FakeRequest:
    model_id: int
    input_tokens: int = 1
    output_tokens: int = 0


class FakeEstimator:
    def predict(self, batch):
        missing = sum(
            int(stage["mapped_pages"])
            for stage in batch["layerweave"]["stages"])
        return {"predicted_service_ttft_ms": float(missing * 10)}


def candidate(configuration, pages, gain, demand=1.0):
    return {
        "configuration": configuration,
        "pages": frozenset(pages),
        "page_count": len(pages),
        "predicted_service_ttft_ms": 100.0 - gain,
        "predicted_gain_ms": gain,
        "effective_demand": demand,
        "expected_value_ms": demand * gain,
    }


class JointPolicyTest(unittest.TestCase):
    def make_policy(self):
        pool = FakePool({
            "m0": [True, True, False, False, False, False],
            "m1": [True, True, True, True, True, True],
        })
        controllers = {
            0: FakeController(pool, "m0"),
            1: FakeController(pool, "m1"),
        }
        policy = LayerWeaveSingleGpuCachePolicy.__new__(
            LayerWeaveSingleGpuCachePolicy)
        policy.controllers = controllers
        policy.page_size = 1
        policy.pool_pages = 10
        policy.decay = 0.9
        policy.uncertainty_ms = 0.0
        policy.demands = {0: ModelDemand(), 1: ModelDemand()}
        policy.pending_counts = {0: 0, 1: 0}
        policy.cache_states = {
            0: ModelCacheState("0"),
            1: ModelCacheState("full"),
        }
        policy.estimator = FakeEstimator()
        ladders = {
            0: [
                candidate("0", set(), 0),
                candidate("2", {0, 1}, 10),
                candidate("full", set(range(6)), 20),
            ],
            1: [
                candidate("0", set(), 0),
                candidate("2", {0, 1}, 30),
                candidate("full", set(range(6)), 40),
            ],
        }

        def marginal(model_id, _controller):
            candidates = ladders[model_id]
            tiers = []
            metadata = {}
            for index, (lower, upper) in enumerate(
                    zip(candidates, candidates[1:]), 1):
                added = set(upper["pages"]) - set(lower["pages"])
                tier = {
                    "index": index,
                    "label": (
                        f"{lower['configuration']}->"
                        f"{upper['configuration']}"),
                    "added_pages": len(added),
                    "marginal_gain_ms": (
                        upper["predicted_gain_ms"]
                        - lower["predicted_gain_ms"]),
                    "conservative_gain_ms": (
                        upper["predicted_gain_ms"]
                        - lower["predicted_gain_ms"]),
                    "value_per_page": 1.0,
                }
                tiers.append(tier)
                for page in added:
                    metadata[page] = tier
            return candidates, tiers, metadata

        policy._marginal_tiers = marginal
        return policy, controllers

    def test_mckp_keeps_best_feasible_combination(self):
        result = LayerWeaveSingleGpuCachePolicy._solve_mckp({
            0: [
                candidate("0", set(), 0),
                candidate("2", {0, 1}, 8),
            ],
            1: [
                candidate("0", set(), 0),
                candidate("2", {0, 1}, 10),
            ],
        }, 2)
        self.assertEqual(result["choices"][1]["configuration"], "2")
        self.assertEqual(result["choices"][0]["configuration"], "0")

    def test_dispatch_reclaims_only_soft_pages(self):
        policy, controllers = self.make_policy()
        decision = policy.plan_dispatch(
            active_model_id=0,
            batch=[FakeRequest(0)],
            pending=[],
            kv_block_size_tokens=1,
            kv_block_size_bytes=1,
            current_kv_pages=0,
        )
        self.assertEqual(
            decision["mode"],
            "protected_mckp_deferred_reclamation")
        self.assertEqual(decision["eviction_required_pages"], 3)
        self.assertEqual(decision["evicted_soft_pages"], 3)
        self.assertEqual(decision["evicted_protected_pages"], 0)
        self.assertEqual(
            decision["targets"]["1"]["configuration"], "2")
        self.assertTrue({0, 1}.issubset(controllers[1].applied))
        self.assertEqual(len(controllers[1].applied), 3)
        self.assertEqual(
            policy.cache_states[0].protected_configuration, "full")

    def test_route_estimate_is_read_only(self):
        policy, controllers = self.make_policy()
        before_residency = {
            path: list(value)
            for path, value in controllers[0].pool.residencies.items()
        }
        before_floors = {
            model_id: state.protected_configuration
            for model_id, state in policy.cache_states.items()
        }
        estimate = policy.estimate_dispatch(
            active_model_id=0,
            batch=[FakeRequest(0, input_tokens=4)],
            pending=[FakeRequest(1)],
            request_kv_pages=1,
        )
        self.assertEqual(estimate["missing_pages"], 4)
        self.assertEqual(estimate["predicted_service_ttft_ms"], 40.0)
        self.assertEqual(
            before_residency, controllers[0].pool.residencies)
        self.assertEqual(
            before_floors,
            {
                model_id: state.protected_configuration
                for model_id, state in policy.cache_states.items()
            },
        )


if __name__ == "__main__":
    unittest.main()
