#!/usr/bin/env python3
"""CPU-only checks for global-pending multi-GPU placement."""

import random
import unittest

from layerweave_multi_gpu_trace_bench import (
    _choose_immediate_pair,
    _cold_tie_choice,
    _route_scores,
)


class MultiGpuSchedulerTest(unittest.TestCase):
    def test_minimal_can_queue_on_busy_cache_hit(self):
        scores, _ = _route_scores(
            "minimal",
            [0, 1],
            {
                0: {"estimated_load_ms": 0.0},
                1: {"estimated_load_ms": 700.0},
            },
            {0: 80.0, 1: 0.0},
            2,
            {0: {}, 1: {}},
            750.0,
            1.0,
        )
        self.assertLess(scores[0], scores[1])

    def test_minimal_uses_idle_gpu_when_busy_wait_is_too_long(self):
        scores, _ = _route_scores(
            "minimal",
            [0, 1],
            {
                0: {"estimated_load_ms": 0.0},
                1: {"estimated_load_ms": 700.0},
            },
            {0: 900.0, 1: 0.0},
            2,
            {0: {}, 1: {}},
            750.0,
            1.0,
        )
        self.assertLess(scores[1], scores[0])

    def test_joint_includes_transition_cost(self):
        scores, predicted = _route_scores(
            "joint",
            [0, 1],
            {
                0: {
                    "predicted_service_ttft_ms": 200.0,
                    "transition_cost_ms": 50.0,
                },
                1: {
                    "predicted_service_ttft_ms": 300.0,
                    "transition_cost_ms": 0.0,
                },
            },
            {0: 0.0, 1: 0.0},
            2,
            {0: {}, 1: {}},
            750.0,
            1.0,
        )
        self.assertEqual(predicted[0], 200.0)
        self.assertEqual(scores[0], 250.0)
        self.assertLess(scores[0], scores[1])

    def test_negative_transition_credit_is_bounded(self):
        scores, _ = _route_scores(
            "joint",
            [0],
            {
                0: {
                    "predicted_service_ttft_ms": 650.0,
                    "transition_cost_ms": -10000.0,
                },
            },
            {0: 0.0},
            2,
            {0: {}},
            750.0,
            1.0,
            100.0,
        )
        self.assertEqual(scores[0], 550.0)

    def test_positive_transition_penalty_is_bounded(self):
        scores, _ = _route_scores(
            "joint",
            [0],
            {
                0: {
                    "predicted_service_ttft_ms": 650.0,
                    "transition_cost_ms": 10000.0,
                },
            },
            {0: 0.0},
            2,
            {0: {}},
            750.0,
            1.0,
            100.0,
            100.0,
        )
        self.assertEqual(scores[0], 750.0)

    def test_waits_briefly_for_better_busy_gpu(self):
        selection, deferred = _choose_immediate_pair(
            [{"request_id": 7, "arrival_s": 0.0}],
            [0, 1],
            [1],
            {7: {0: 300.0, 1: 700.0}},
            {0: 80.0, 1: 0.0},
            1.0,
            150.0,
            25.0,
            {},
        )
        self.assertIsNone(selection)
        self.assertEqual(deferred[0]["preferred_device"], 0)

    def test_dispatches_after_placement_wait_cap(self):
        selection, deferred = _choose_immediate_pair(
            [{"request_id": 7, "arrival_s": 0.0}],
            [0, 1],
            [1],
            {7: {0: 300.0, 1: 700.0}},
            {0: 80.0, 1: 0.0},
            1.2,
            150.0,
            25.0,
            {7: 1.0},
        )
        self.assertEqual(selection["chosen_device"], 1)
        self.assertFalse(deferred)

    def test_lookahead_keeps_free_gpu_work_conserving(self):
        selection, deferred = _choose_immediate_pair(
            [
                {"request_id": 7, "arrival_s": 0.0},
                {"request_id": 8, "arrival_s": 0.1},
            ],
            [0, 1],
            [1],
            {
                7: {0: 300.0, 1: 700.0},
                8: {0: 800.0, 1: 200.0},
            },
            {0: 80.0, 1: 0.0},
            1.0,
            150.0,
            25.0,
            {},
        )
        self.assertEqual(selection["request"]["request_id"], 8)
        self.assertEqual(selection["chosen_device"], 1)
        self.assertEqual(deferred[0]["request_id"], 7)

    def test_serial_choice_selects_between_two_idle_gpus(self):
        selection, deferred = _choose_immediate_pair(
            [{"request_id": 9, "arrival_s": 0.0}],
            [0, 1],
            [0, 1],
            {9: {0: 420.0, 1: 310.0}},
            {0: 0.0, 1: 0.0},
            1.0,
            150.0,
            25.0,
            {},
        )
        self.assertEqual(selection["chosen_device"], 1)
        self.assertFalse(deferred)

    def test_minimal_cold_tie_random_is_seeded(self):
        estimates = {
            0: {"cached_pages": 0},
            1: {"cached_pages": 0},
        }
        first = _cold_tie_choice(
            "minimal", "serial-choice", True,
            False,
            estimates, [0, 1], random.Random(1234))
        second = _cold_tie_choice(
            "minimal", "serial-choice", True,
            False,
            estimates, [0, 1], random.Random(1234))
        self.assertEqual(first, second)

    def test_minimal_cold_tie_preserves_partial_cache_affinity(self):
        choice = _cold_tie_choice(
            "minimal", "serial-choice", True,
            False,
            {
                0: {"cached_pages": 1},
                1: {"cached_pages": 0},
            },
            [0, 1],
            random.Random(1234),
        )
        self.assertIsNone(choice)

    def test_joint_cold_tie_can_be_enabled_independently(self):
        choice = _cold_tie_choice(
            "joint", "serial-choice", False, True,
            {
                0: {"cached_pages": 0},
                1: {"cached_pages": 0},
            },
            [0, 1],
            random.Random(1234),
        )
        self.assertIn(choice, (0, 1))


if __name__ == "__main__":
    unittest.main()
