#!/usr/bin/env python3
"""CPU-only tests for the profile-driven cache simulator."""

import json
import tempfile
import unittest
from pathlib import Path

from layerweave_pipeline_cache_sim import (
    CacheState,
    ModelLayout,
    Request,
    build_five_way_ablation,
    configuration_layers,
    evict_for_request_mckp,
    load_profile,
    load_trace,
    prefix_pages,
    simulate_aegaeon,
    solve_mckp,
    solve_mckp_topk,
)


class PipelineCacheSimulatorTest(unittest.TestCase):
    def test_json_compute_layer_keys_are_normalized_to_int(self):
        document = {
            "profiles": {
                "3": {
                    "bytes_per_ms": 1.0,
                    "fixed_stage_ms": 0.0,
                    "compute": {
                        "0": {
                            "coefficients": [7.0, 0.0, 0.0, 0.0, 0.0]
                        }
                    },
                    "host_gap_ms": {},
                    "service_residual_ms": {},
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(document))
            estimator = load_profile(path)
        self.assertIn(0, estimator.models[3]["compute"])
        self.assertNotIn("0", estimator.models[3]["compute"])

    def test_five_way_ablation_has_expected_decomposition(self):
        request = Request(0, 0, 8, 1)
        minimal_row = {
            "critical_path_ms": 13.0,
            "hot_compute_ms": 10.0,
            "h2d_ms": 5.0,
            "exposed_load_ms": 3.0,
            "missing_pages": 2,
            "missing_pages_by_stage": [2],
        }
        layerweave_row = {
            "critical_path_ms": 12.0,
            "hot_compute_ms": 10.0,
            "h2d_ms": 4.0,
            "exposed_load_ms": 2.0,
            "missing_pages": 1,
            "missing_pages_by_stage": [1],
        }
        cold = {
            "critical_path_ms": 17.0,
            "hot_compute_ms": 10.0,
            "h2d_ms": 9.0,
            "exposed_load_ms": 7.0,
            "missing_pages": 4,
            "missing_pages_by_stage": [4],
        }
        import layerweave_pipeline_cache_sim as simulator
        original_predict = simulator.predict
        original_timeline = simulator.pipeline_timeline
        simulator.predict = lambda *args, **kwargs: cold
        simulator.pipeline_timeline = lambda *args, **kwargs: {
            "predicted_total_ms": 0.0,
            "loads": [],
            "computes": [],
        }
        try:
            class Layout:
                stage_pages = [4]

            result = build_five_way_ablation(
                [request],
                {0: Layout()},
                object(),
                {"requests": [minimal_row]},
                {"requests": [layerweave_row]},
                inspect_request_id=0,
            )
        finally:
            simulator.predict = original_predict
            simulator.pipeline_timeline = original_timeline

        stages = result["stages"]
        self.assertEqual(
            19.0,
            stages["baseline_load_plus_compute"]
            ["predicted_total_ms"]["mean"],
        )
        self.assertEqual(
            17.0, stages["pipe_only"]["predicted_total_ms"]["mean"])
        self.assertEqual(
            15.0, stages["reuse_only"]["predicted_total_ms"]["mean"])
        self.assertEqual(
            13.0, stages["minimal"]["predicted_total_ms"]["mean"])
        self.assertEqual(
            12.0, stages["layerweave"]["predicted_total_ms"]["mean"])
        self.assertEqual(1.0, result["layerweave_gain_vs_minimal_ms"])

    def test_prefix_configuration_uses_leading_stages(self):
        layout = ModelLayout(
            model_id=0,
            stage_pages=[3, 2, 4],
            page_stage=[0, 0, 0, 1, 1, 2, 2, 2, 2],
            page_count=9,
            max_input_tokens=32,
        )
        self.assertEqual(frozenset(), prefix_pages(layout, 0))
        self.assertEqual(
            frozenset(range(5)), prefix_pages(layout, 2))
        self.assertEqual(
            frozenset(range(9)), prefix_pages(layout, "full"))
        self.assertEqual((0, 1, 2), configuration_layers(layout, 1))
        self.assertEqual((0, 2), configuration_layers(layout, 2))
        self.assertEqual(
            (0, 2, 4, 8, 16, 32), configuration_layers(layout, 0))

    def test_switch_only_trace_keeps_first_request_of_each_model_run(self):
        layouts = {
            model_id: ModelLayout(
                model_id=model_id,
                stage_pages=[1],
                page_stage=[0],
                page_count=1,
                max_input_tokens=128,
            )
            for model_id in (0, 1)
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            path.write_text(
                "0 0 8 1\n"
                "1 0 16 1\n"
                "2 1 24 1\n"
                "3 1 32 1\n"
                "4 0 40 1\n"
            )
            requests = load_trace(
                path, layouts, max_requests=4, input_scale=1.0,
                output_tokens=1, pool_pages=8, kv_block_tokens=32,
                trace_mode="model-switches")
        self.assertEqual([0, 1], [request.model_id for request in requests])
        self.assertEqual([8, 24], [
            request.input_tokens for request in requests])
        self.assertEqual([1, 1], [
            request.trace_output_tokens for request in requests])

    def test_aegaeon_uses_real_decode_window_and_capacity_gate(self):
        layouts = {
            model_id: ModelLayout(
                model_id=model_id,
                stage_pages=[pages],
                page_stage=[0] * pages,
                page_count=pages,
                max_input_tokens=128,
            )
            for model_id, pages in ((0, 2), (1, 3), (2, 7))
        }
        requests = [
            Request(0, 0, 8, 1, trace_output_tokens=3),
            Request(1, 1, 8, 1, trace_output_tokens=1),
            Request(2, 2, 8, 1, trace_output_tokens=1),
        ]
        import layerweave_pipeline_cache_sim as simulator
        original_predict = simulator.predict
        original_h2d = simulator.whole_model_h2d_ms
        simulator.predict = lambda *args, **kwargs: {
            "hot_compute_ms": 10.0,
        }
        simulator.whole_model_h2d_ms = (
            lambda estimator, layout: layout.page_count * 10.0)
        try:
            result = simulate_aegaeon(
                requests, layouts, object(), pool_pages=10, gpu_count=1,
                kv_block_tokens=32, routing=[0, 0, 0],
                decode_ms_per_token=10.0, gpu_copy_gbps=1_000_000.0)
        finally:
            simulator.predict = original_predict
            simulator.whole_model_h2d_ms = original_h2d
        rows = result["requests"]
        self.assertEqual(20.0, rows[0]["loading_stall_ms"])
        self.assertAlmostEqual(
            rows[1]["gpu_relocation_ms"], rows[1]["loading_stall_ms"])
        self.assertFalse(rows[2]["capacity_feasible"])
        self.assertEqual(70.0, rows[2]["loading_stall_ms"])

    def test_mckp_allocates_cross_model_prefix_budget(self):
        result = solve_mckp({
            0: [
                {
                    "configuration": "0", "page_count": 0,
                    "expected_value_ms": 0.0,
                },
                {
                    "configuration": "2", "page_count": 3,
                    "expected_value_ms": 5.0,
                },
            ],
            1: [
                {
                    "configuration": "0", "page_count": 0,
                    "expected_value_ms": 0.0,
                },
                {
                    "configuration": "2", "page_count": 2,
                    "expected_value_ms": 7.0,
                },
            ],
        }, capacity_pages=3)
        self.assertEqual("0", result["choices"][0]["configuration"])
        self.assertEqual("2", result["choices"][1]["configuration"])
        self.assertEqual(2, result["used_pages"])

    def test_topk_mckp_returns_distinct_candidate_plans(self):
        candidates = {
            0: [
                {
                    "configuration": "0", "page_count": 0,
                    "expected_value_ms": 0.0,
                },
                {
                    "configuration": "2", "page_count": 2,
                    "expected_value_ms": 5.0,
                },
            ],
            1: [
                {
                    "configuration": "0", "page_count": 0,
                    "expected_value_ms": 0.0,
                },
                {
                    "configuration": "2", "page_count": 2,
                    "expected_value_ms": 4.0,
                },
            ],
        }
        plans = solve_mckp_topk(candidates, 2, topk=3)
        choices = {
            tuple(
                plan["choices"][model]["configuration"]
                for model in sorted(plan["choices"])
            )
            for plan in plans
        }
        self.assertGreaterEqual(len(choices), 2)

    def test_mckp_page_greedy_selects_soft_page_by_pipeline_value(self):
        layouts = {
            model_id: ModelLayout(
                model_id=model_id,
                stage_pages=[1, 1],
                page_stage=[0, 1],
                page_count=2,
                max_input_tokens=32,
            )
            for model_id in range(3)
        }
        cache = CacheState(
            resident={0: set(), 1: {0, 1}, 2: {0, 1}},
            protected={0: set(), 1: set(), 2: set()},
            demand_scores={0: 0.0, 1: 1.0, 2: 1.0},
        )
        curves = {
            model_id: [{
                "configuration": "0",
                "pages": frozenset(),
                "page_count": 0,
                "expected_gain_ms": 0.0,
                "expected_value_ms": 0.0,
                "shape_gains_ms": {},
            }]
            for model_id in range(3)
        }
        values = {
            (1, 0): 10.0,
            (1, 1): 1.0,
            (2, 0): 8.0,
            (2, 1): 4.0,
        }
        evicted, _, _ = evict_for_request_mckp(
            cache,
            Request(0, 0, 8, 1),
            layouts,
            pool_pages=5,
            kv_pages=0,
            curves=curves,
            soft_victim_policy="page-greedy",
            pipeline_values=values,
        )
        self.assertEqual(1, evicted)
        self.assertEqual({0}, cache.resident[1])
        self.assertEqual({0, 1}, cache.resident[2])


if __name__ == "__main__":
    unittest.main()
