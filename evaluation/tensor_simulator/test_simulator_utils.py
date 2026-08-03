"""Tests for helpers extracted from the legacy page-level simulator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from compute_features import batch_features
from simulator_io import load_model_limits, load_profile, load_trace
from statistics_utils import summary


class SimulatorUtilsTest(unittest.TestCase):
    def test_batch_features(self):
        self.assertEqual(
            batch_features({
                "batch_size": 2,
                "input_tokens": 3000,
                "max_input_tokens": 2000,
                "sum_input_tokens_squared": 5_000_000,
            }),
            [1.0, 1.0, 3.0, 2.0, 5.0],
        )

    def test_summary_interpolates_percentiles(self):
        result = summary([0, 10])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["mean"], 5.0)
        self.assertEqual(result["p90"], 9.0)

    def test_config_profile_and_trace_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"model_lists": [{
                "id": 0,
                "l40_safe_max_input_length": 12,
                "l40_memory_budget": {"model_context_tokens": 16},
            }]}))
            self.assertEqual(load_model_limits(config, "safe"), {0: 12})
            self.assertEqual(load_model_limits(config, "context"), {0: 16})

            profile = root / "profile.json"
            profile.write_text(json.dumps({"profiles": {"0": {
                "compute": {"2": {"coefficients": [1.0]}},
            }}}))
            loaded = load_profile(profile)
            self.assertIn(2, loaded.models[0]["compute"])

            trace = root / "trace.txt"
            trace.write_text("0 0 10 7\n1 0 20 9\n")
            layout = type("Layout", (), {
                "page_count": 1,
                "max_input_tokens": 12,
            })()
            requests = load_trace(
                trace, {0: layout}, max_requests=2, input_scale=1.0,
                output_tokens=1, pool_pages=10, kv_block_tokens=32,
                trace_mode="model-switches")
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].input_tokens, 10)
            self.assertEqual(requests[0].output_tokens, 1)
            self.assertEqual(requests[0].trace_output_tokens, 7)


if __name__ == "__main__":
    unittest.main()
