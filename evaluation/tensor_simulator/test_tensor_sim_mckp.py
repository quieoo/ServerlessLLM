#!/usr/bin/env python3
"""CPU invariants for prefix-stall table lookup."""

import unittest

from tensor_sim_mckp import (
    PrefixStallTable, build_model_options, solve_covering_mckp,
)


def table():
    return PrefixStallTable({
        "format": "layerweave-prefix-stall-v1",
        "metadata": {},
        "models": {
            "0": {
                "group_count": 2,
                "prefix_bytes": [0, 10, 30],
                "rows": [
                    {
                        "input_tokens": 10,
                        "prefix_stall_ms": [8, 3, 0],
                        "suffix_stall_ms": [0, 6, 8],
                    },
                    {
                        "input_tokens": 20,
                        "prefix_stall_ms": [12, 5, 0],
                        "suffix_stall_ms": [0, 9, 12],
                    },
                ],
            },
        },
    })


class PrefixStallTableTest(unittest.TestCase):
    def test_exact_and_interpolated_lookup(self):
        profile = table()
        self.assertEqual(8, profile.lookup(0, 10, 0))
        self.assertEqual(10, profile.lookup(0, 15, 0))
        self.assertEqual(4, profile.lookup(0, 15, 1))

    def test_lookup_clamps_outside_measured_range(self):
        profile = table()
        self.assertEqual(8, profile.lookup(0, 1, 0))
        self.assertEqual(12, profile.lookup(0, 100, 0))

    def test_lookup_can_linearly_extrapolate_above_measured_range(self):
        profile = table()
        profile.input_extrapolation = "linear"
        self.assertEqual(16, profile.lookup(0, 30, 0))
        self.assertEqual(12, profile.lookup_suffix(0, 30, 1))

    def test_suffix_lookup_exact_interpolated_and_clamped(self):
        profile = table()
        self.assertEqual(0, profile.lookup_suffix(0, 10, 0))
        self.assertEqual(7.5, profile.lookup_suffix(0, 15, 1))
        self.assertEqual(12, profile.lookup_suffix(0, 100, 2))

    def test_prefix_bytes_and_curve(self):
        profile = table()
        self.assertEqual(30, profile.prefix_bytes(0, 2))
        self.assertEqual([10, 4, 0], profile.curve(0, 15))

    def test_rejects_nonmonotonic_stall(self):
        document = table().document
        document["models"]["0"]["rows"][0]["prefix_stall_ms"] = [8, 9, 0]
        with self.assertRaisesRegex(ValueError, "not prefix-monotonic"):
            PrefixStallTable(document)

    def test_model_options_use_delta_from_current_prefix(self):
        profile = table()
        options = build_model_options(
            profile, 0, 2, [(10, 0.5), (20, 0.5)])
        by_prefix = {option.prefix_count: option for option in options}
        self.assertEqual(0, by_prefix[2].damage_ms)
        self.assertEqual(4, by_prefix[1].damage_ms)
        self.assertEqual(30, by_prefix[0].freed_bytes)

    def test_covering_mckp_selects_minimum_damage_combination(self):
        document = table().document
        document["models"]["1"] = {
            "group_count": 2,
            "prefix_bytes": [0, 10, 30],
            "rows": [{
                "input_tokens": 10,
                "prefix_stall_ms": [20, 10, 0],
            }],
        }
        profile = PrefixStallTable(document)
        plan = solve_covering_mckp(
            profile, {0: 2, 1: 2}, active_model=99,
            required_free_bytes=20,
            prediction_samples={0: [(10, 1)], 1: [(10, 1)]})
        self.assertTrue(plan.feasible)
        self.assertEqual(20, plan.planned_free_bytes)
        self.assertEqual({0: 1, 1: 2}, plan.choices)
        self.assertEqual(3, plan.predicted_future_damage_ms)

    def test_covering_mckp_never_evicts_active_model(self):
        profile = table()
        plan = solve_covering_mckp(
            profile, {0: 2}, active_model=0, required_free_bytes=1,
            prediction_samples={0: [(10, 1)]})
        self.assertFalse(plan.feasible)


if __name__ == "__main__":
    unittest.main()
