#!/usr/bin/env python3
"""CPU-only invariants for the tensor-level simulator."""

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from layerweave_tensor_pipeline_sim import (
    CompactPageMemory,
    Extent,
    LoadCost,
    MockAllocationSegmentMemory,
    SegmentMemory,
    TensorModel,
    TensorPageMemory,
    TensorSpec,
    _aegaeon_profile_row,
    build_pipeline_benefits,
    build_gpu_availability,
    choose_online_cache_bytes_candidate,
    choose_routing_candidate,
    execute_profile_cold_request,
    execute_profile_request,
    execute_request,
    apply_online_timing,
    load_trace_arrivals,
    load_model_mapping,
    remap_requests,
    set_tensor_groups,
    write_per_model_csv,
)
from tensor_sim_mckp import PrefixStallTable


def args():
    return argparse.Namespace(
        h2d_bytes_per_ms=1000.0,
        gpu_copy_bytes_per_ms=1000.0,
        compaction_fixed_ms=0.0,
        compaction_policy="on-allocation-failure",
        segment_allocation_ms=0.0,
        map_fixed_ms=0.0,
        map_per_page_ms=0.0,
        unmap_fixed_ms=0.0,
        unmap_per_page_ms=0.0,
        replacement_policy="frequency-lru",
        pipeline_protected_threshold_ms=0.1,
        pipeline_robust_stat="p90",
        shadow_max_churn_ratio=1.0,
        memory_layout="segment",
        page_size_bytes=8,
        kv_block_bytes=4,
    )


def model(sizes):
    offset = 0
    tensors = []
    for index, size in enumerate(sizes):
        tensors.append(TensorSpec(index, f"t{index}", 0, size, offset))
        offset += size
    result = TensorModel(0, tensors, offset, 128)
    set_tensor_groups(result, 0)
    return {0: result}


def prefix_table(models):
    entries = {}
    for model_id, item in models.items():
        prefix_bytes = [0]
        for group in item.groups:
            prefix_bytes.append(prefix_bytes[-1] + group.logical_bytes)
        entries[str(model_id)] = {
            "group_count": len(item.groups),
            "prefix_bytes": prefix_bytes,
            "rows": [{
                "input_tokens": 8,
                "prefix_stall_ms": list(
                    reversed(range(len(item.groups) + 1))),
            }],
        }
    return PrefixStallTable({
        "format": "layerweave-prefix-stall-v1",
        "metadata": {},
        "models": entries,
    })


class TensorMemoryTest(unittest.TestCase):
    def test_load_cost_adds_pbp_planner_time(self):
        total = LoadCost(
            pbp_planner_time_ms=0.25, model_reload_events=1,
            naive_relocation_bytes=10)
        total.add(LoadCost(
            pbp_planner_time_ms=0.75, model_reload_events=1,
            naive_relocation_bytes=20))
        self.assertEqual(1.0, total.pbp_planner_time_ms)
        self.assertEqual(2, total.model_reload_events)
        self.assertEqual(30, total.naive_relocation_bytes)

    def test_aegaeon_decode_window_uses_trace_output_tokens(self):
        models = model([10])
        current_model = models[0]
        models[1] = TensorModel(
            1, current_model.tensors, current_model.logical_bytes,
            current_model.max_input_tokens)
        table = prefix_table(models)
        options = args()
        options.decode_ms_per_token = 2.0
        options.pool_bytes = 1000
        options.kv_block_tokens = 32
        previous = type("Request", (), {
            "request_id": 0,
            "model_id": 0,
            "input_tokens": 8,
            "output_tokens": 1,
            "trace_output_tokens": 7,
        })()
        request = type("Request", (), {
            "request_id": 1,
            "model_id": 1,
            "input_tokens": 8,
            "output_tokens": 1,
            "trace_output_tokens": 3,
        })()

        row = _aegaeon_profile_row(
            request, 0, previous, models[1], models, table, options)

        self.assertEqual(14.0, row["decode_overlap_window_ms"])
        self.assertEqual(0.01, row["prefetched_h2d_ms"])

    def test_online_cache_bytes_uses_only_same_dispatch_time_gpus(self):
        candidates = [
            (0, {
                "routing_cached_model_bytes": 100,
                "critical_path_ms": 1.0,
            }),
            (1, {
                "routing_cached_model_bytes": 10,
                "critical_path_ms": 100.0,
            }),
            (2, {
                "routing_cached_model_bytes": 50,
                "critical_path_ms": 1.0,
            }),
        ]
        chosen, _, _ = choose_online_cache_bytes_candidate(
            candidates, [20.0, 10.0, 10.0], 0.0, 0, 3)
        self.assertEqual(2, chosen)
        chosen, _, _ = choose_online_cache_bytes_candidate(
            candidates, [20.0, 10.0, 0.0], 5.0, 0, 3)
        self.assertEqual(2, chosen)

    def test_trace_scaled_arrivals_match_target_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            path.write_text(
                "10.0 0 1 1\n"
                "11.0 1 1 1\n"
                "13.0 0 1 1\n")
            arrivals = load_trace_arrivals(
                path, 3, "all", 3, "trace-scaled", 2.0, 1234)
        self.assertEqual(0.0, arrivals[0])
        self.assertAlmostEqual(1 / 3, arrivals[1])
        self.assertEqual(1.0, arrivals[2])
        self.assertAlmostEqual(2.0, 2 / (arrivals[-1] - arrivals[0]))

    def test_online_timing_includes_queue_and_decode_occupancy(self):
        rows = [
            {
                "critical_path_ms": 10.0,
                "hot_compute_ms": 5.0,
                "exposed_load_ms": 5.0,
                "h2d_ms": 5.0,
                "allocation_ms": 0.0,
                "map_ms": 0.0,
                "unmap_ms": 0.0,
                "compaction_ms": 0.0,
                "h2d_bytes": 10,
                "compaction_moved_bytes": 0,
                "evicted_objects": 0,
                "evicted_bytes": 0,
                "map_calls": 0,
                "unmap_calls": 0,
                "mapped_pages": 0,
                "unmapped_pages": 0,
                "memory": {},
            }
            for _ in range(2)
        ]
        requests = [
            type("Request", (), {
                "output_tokens": 2,
                "trace_output_tokens": 99,
            })()
            for _ in rows
        ]
        options = argparse.Namespace(
            gpus=1, decode_ms_per_token=10.0,
            slo_scale=5.0, request_rate_rps=100.0,
            arrival_mode="poisson", warmup_requests=0)
        routing, summary = apply_online_timing(
            rows, [], requests, [0.0, 0.005], options, "test")
        self.assertEqual([0, 0], routing)
        self.assertEqual(25.0, rows[1]["queue_ms"])
        self.assertEqual(35.0, rows[1]["ttft_ms"])
        self.assertEqual(60.0, rows[1]["finish_ms"])
        self.assertEqual(0.5, summary["slo_attainment"])

    def test_online_warmup_executes_but_is_excluded_from_summary(self):
        rows = [
            {
                "critical_path_ms": 10.0,
                "hot_compute_ms": 5.0,
                "exposed_load_ms": 5.0,
                "h2d_ms": 5.0,
                "allocation_ms": 0.0,
                "map_ms": 0.0,
                "unmap_ms": 0.0,
                "compaction_ms": 0.0,
                "h2d_bytes": 10,
                "compaction_moved_bytes": 0,
                "evicted_objects": 0,
                "evicted_bytes": 0,
                "map_calls": 0,
                "unmap_calls": 0,
                "mapped_pages": 0,
                "unmapped_pages": 0,
                "memory": {},
            }
            for _ in range(2)
        ]
        requests = [
            type("Request", (), {"output_tokens": 0})()
            for _ in rows
        ]
        options = argparse.Namespace(
            gpus=1, decode_ms_per_token=0.0,
            slo_scale=2.0, request_rate_rps=100.0,
            arrival_mode="poisson", warmup_requests=1)
        _, summary = apply_online_timing(
            rows, [], requests, [0.0, 0.005], options, "test")
        self.assertFalse(rows[0]["measured"])
        self.assertTrue(rows[1]["measured"])
        self.assertEqual(1, summary["requests"])
        self.assertEqual(1, summary["warmup_requests"])
        self.assertEqual(1, summary["measured_requests"])
        self.assertEqual(0.0, summary["slo_attainment"])

    def test_config_model_mapping_is_a_permutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "model_mapping_order": [2, 0, 1],
            }))
            self.assertEqual(
                {0: 2, 1: 0, 2: 1},
                load_model_mapping(path, {0, 1, 2}))
            path.write_text(json.dumps({
                "model_mapping_order": [0, 0, 2],
            }))
            with self.assertRaises(ValueError):
                load_model_mapping(path, {0, 1, 2})

    def test_remap_requests_preserves_shape_and_maps_model(self):
        request = type("Request", (), {
            "request_id": 7,
            "model_id": 0,
            "input_tokens": 11,
            "output_tokens": 2,
            "trace_output_tokens": 3,
        })()
        mapped = remap_requests([request], {0: 2})[0]
        self.assertEqual(
            (7, 2, 11, 2, 3),
            (
                mapped.request_id, mapped.model_id, mapped.input_tokens,
                mapped.output_tokens, mapped.trace_output_tokens,
            ))

    def test_gpu_availability_is_deterministic_and_nonempty(self):
        requests = [
            type("Request", (), {"request_id": index})()
            for index in range(20)
        ]
        first = build_gpu_availability(requests, 4, 0.5, 1234)
        second = build_gpu_availability(requests, 4, 0.5, 1234)
        self.assertEqual(first, second)
        for item in first:
            self.assertGreaterEqual(len(item["available_gpus"]), 1)
            self.assertFalse(
                set(item["available_gpus"]) & set(item["busy_gpus"]))
            self.assertEqual(
                set(range(4)),
                set(item["available_gpus"]) | set(item["busy_gpus"]))

    def test_gpu_busy_probability_endpoints(self):
        requests = [
            type("Request", (), {"request_id": index})()
            for index in range(20)
        ]
        idle = build_gpu_availability(requests, 4, 0.0, 1234)
        busy = build_gpu_availability(requests, 4, 1.0, 1234)
        self.assertTrue(all(
            len(item["available_gpus"]) == 4
            and item["forced_available_gpu"] is None
            for item in idle))
        self.assertTrue(all(
            len(item["available_gpus"]) == 1
            and len(item["busy_gpus"]) == 3
            and item["forced_available_gpu"] in item["available_gpus"]
            for item in busy))

    def test_gpu_availability_can_sample_exact_count(self):
        requests = [
            type("Request", (), {"request_id": index})()
            for index in range(20)
        ]
        first = build_gpu_availability(
            requests, 6, 0.5, 1234, exact_available=3)
        second = build_gpu_availability(
            requests, 6, 0.5, 1234, exact_available=3)
        self.assertEqual(first, second)
        self.assertTrue(all(
            len(item["available_gpus"]) == 3
            and len(item["busy_gpus"]) == 3
            and item["forced_available_gpu"] is None
            for item in first))

    def test_prefix_lru_kv_reclaim_preserves_prefix(self):
        models = {}
        for model_id in (0, 1):
            item = TensorModel(
                model_id,
                [
                    TensorSpec(0, "t0", 0, 6, 0),
                    TensorSpec(1, "t1", 1, 6, 6),
                ],
                12,
                128,
            )
            set_tensor_groups(item, 0)
            models[model_id] = item
        options = args()
        options.replacement_policy = "lru-prefix"
        memory = MockAllocationSegmentMemory(models, 32, 8, options)
        memory.extents = [
            Extent(0, 6, (1, 0)),
            Extent(6, 6, (1, 1)),
            Extent(12, 20),
        ]
        memory.allocated_extents = {
            extent.key: extent for extent in memory.extents
            if extent.key is not None
        }
        memory.last_access[(1, 0)] = 0
        memory.last_access[(1, 1)] = 1
        memory.pending_kv_bytes = 24

        cost = LoadCost()
        memory._allocate_kv_blocks(0, cost)

        self.assertEqual([0], memory.resident_group_ids(1))
        self.assertEqual(6, cost.evicted_bytes)

    def test_profile_cold_baseline_and_pipe_share_prefill(self):
        models = model([10])
        options = args()
        options.kv_block_tokens = 32
        options.kv_block_bytes = 4
        options.pool_bytes = 64
        table = PrefixStallTable({
            "format": "layerweave-prefix-stall-v1",
            "metadata": {},
            "models": {
                "0": {
                    "group_count": 1,
                    "prefix_bytes": [0, 10],
                    "rows": [{
                        "input_tokens": 8,
                        "median_baseline_wall_ms": 3.0,
                        "prefix_stall_ms": [4.0, 0.0],
                    }],
                },
            },
        })
        request = type("Request", (), {
            "request_id": 0,
            "model_id": 0,
            "input_tokens": 8,
            "output_tokens": 1,
        })()
        baseline = execute_profile_cold_request(
            request, models[0], table, options, False)
        pipe = execute_profile_cold_request(
            request, models[0], table, options, True)
        self.assertEqual(3.0, baseline["hot_compute_ms"])
        self.assertEqual(3.0, pipe["hot_compute_ms"])
        self.assertEqual(0.01, baseline["exposed_load_ms"])
        self.assertEqual(4.0, pipe["exposed_load_ms"])
        self.assertEqual(4.0, pipe["exposed_h2d_ms"])
        self.assertEqual(0.0, pipe["exposed_allocation_ms"])

    def test_profile_exposed_components_are_additive(self):
        models = model([10, 20])
        options = args()
        options.replacement_policy = "mckp-prefix"
        options.segment_allocation_ms = 0.5
        options.kv_block_tokens = 32
        options.kv_block_bytes = 4
        options.pool_bytes = 64
        table = PrefixStallTable({
            "format": "layerweave-prefix-stall-v1",
            "metadata": {"h2d_gbps": 0.000001,
                         "segment_allocation_ms": 0.5},
            "models": {"0": {
                "group_count": 2,
                "prefix_bytes": [0, 10, 30],
                "groups": [
                    {"logical_bytes": 10, "observed": True},
                    {"logical_bytes": 20, "observed": True},
                ],
                "rows": [{
                    "input_tokens": 8,
                    "group_boundary_ms": [1.0, 2.0],
                    "prefix_stall_ms": [29.5, 19.5, 0.0],
                }],
            }},
        })
        request = type("Request", (), {
            "request_id": 0, "model_id": 0,
            "input_tokens": 8, "output_tokens": 1,
        })()
        result = execute_profile_cold_request(
            request, models[0], table, options, True)
        components = sum(result[field] for field in (
            "exposed_h2d_ms", "exposed_allocation_ms",
            "exposed_compaction_ms", "exposed_page_map_ms",
            "exposed_page_unmap_ms"))
        self.assertAlmostEqual(result["exposed_load_ms"], components)
        self.assertGreater(result["exposed_h2d_ms"], 0.0)
        self.assertGreater(result["exposed_allocation_ms"], 0.0)

    def test_profile_reuse_only_charges_serialized_missing_transfer(self):
        models = model([10])
        options = args()
        options.replacement_policy = "lru"
        options.kv_block_tokens = 32
        options.kv_block_bytes = 4
        options.pool_bytes = 64
        backend = MockAllocationSegmentMemory(models, 64, 8, options)
        table = PrefixStallTable({
            "format": "layerweave-prefix-stall-v1",
            "metadata": {},
            "models": {
                "0": {
                    "group_count": 1,
                    "prefix_bytes": [0, 10],
                    "rows": [{
                        "input_tokens": 8,
                        "median_baseline_wall_ms": 3.0,
                        "prefix_stall_ms": [4.0, 0.0],
                        "suffix_stall_ms": [0.0, 4.0],
                    }],
                },
            },
        })
        request = type("Request", (), {
            "request_id": 0,
            "model_id": 0,
            "input_tokens": 8,
            "output_tokens": 1,
        })()
        result = execute_profile_request(
            backend, request, models[0], table, options, pipelined=False)
        self.assertEqual(3.0, result["hot_compute_ms"])
        self.assertEqual(0.01, result["h2d_ms"])
        self.assertEqual(0.01, result["exposed_load_ms"])
        self.assertNotIn("profile_exposed_load_ms", result)

    def test_per_model_csv_uses_hot_compute_as_prefill(self):
        rows = [
            {"model_id": 1, "exposed_load_ms": 2.0, "hot_compute_ms": 4.0},
            {"model_id": 1, "exposed_load_ms": 6.0, "hot_compute_ms": 8.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            write_per_model_csv(path, "test", rows)
            with path.open(newline="") as stream:
                result = list(csv.DictReader(stream))
        self.assertEqual("4.0", result[0]["exposed_load_ms_mean"])
        self.assertEqual("6.0", result[0]["prefill_ms_mean"])

    def test_profile_ttft_charges_vmm_cost_per_physical_page(self):
        models = model([10])
        options = args()
        options.replacement_policy = "mckp-prefix"
        options.kv_block_tokens = 32
        options.kv_block_bytes = 0
        options.map_fixed_ms = 0.001
        options.map_per_page_ms = 0.001
        backend = TensorPageMemory(models, 64, 8, options)
        table = prefix_table(models)
        backend.set_prefix_stall_table(table)
        request = type("Request", (), {
            "request_id": 0,
            "model_id": 0,
            "input_tokens": 8,
            "output_tokens": 1,
        })()
        result = execute_profile_request(
            backend, request, models[0], table, options)
        self.assertEqual(2, result["map_calls"])
        self.assertEqual(result["mapped_pages"], result["map_calls"])
        self.assertEqual(0.004, result["vmm_ms"])
        self.assertEqual(1.004, result["critical_path_ms"])

    def test_page_prefix_mckp_uses_page_aligned_group_bytes(self):
        models = model([10, 10])
        options = args()
        options.replacement_policy = "mckp-prefix"
        backend = TensorPageMemory(models, 64, 8, options)
        backend.set_prefix_stall_table(prefix_table(models))
        backend.set_mckp_prediction_samples({0: [(8, 1.0)]})
        for tensor in models[0].tensors:
            backend.prepare_tensor(0, tensor)
        cost = backend._mckp_reclaim(active_model=1, required_free_bytes=8)
        self.assertEqual(16, cost.evicted_bytes)
        self.assertEqual(1, backend.resident_prefix_count(0))

    def test_compact_prefix_mckp_keeps_shared_boundary_page(self):
        models = model([10, 10])
        options = args()
        options.replacement_policy = "mckp-prefix"
        backend = CompactPageMemory(models, 64, 8, options)
        backend.set_prefix_stall_table(prefix_table(models))
        backend.set_mckp_prediction_samples({0: [(8, 1.0)]})
        for tensor in models[0].tensors:
            backend.prepare_tensor(0, tensor)
        cost = backend._mckp_reclaim(active_model=1, required_free_bytes=8)
        self.assertEqual(8, cost.evicted_bytes)
        self.assertEqual({0, 1}, backend.resident_pages[0])
        self.assertEqual(1, backend.resident_prefix_count(0))

    def test_tensor_page_rounding_is_per_tensor(self):
        models = model([6, 6])
        backend = TensorPageMemory(models, 32, 8, args())
        backend.begin_request(0, 0)
        for tensor in models[0].tensors:
            backend.prepare_tensor(0, tensor)
        self.assertEqual(16, backend.used_bytes())
        self.assertEqual(4, backend.metrics()["internal_fragmentation_bytes"])

    def test_page_prefix_mckp_batches_request_reclaim(self):
        for backend_type in (TensorPageMemory, CompactPageMemory):
            with self.subTest(backend=backend_type.__name__):
                first = model([6, 6])[0]
                second = model([6, 6])[0]
                second.model_id = 1
                models = {0: first, 1: second}
                options = args()
                options.replacement_policy = "mckp-prefix"
                backend = backend_type(models, 24, 8, options)
                backend.set_prefix_stall_table(prefix_table(models))
                backend.set_mckp_prediction_samples({
                    0: [(8, 1.0)], 1: [(8, 1.0)]})

                backend.begin_request(0, 0)
                backend.plan_request(0, first.tensors)
                for tensor in first.tensors:
                    backend.prepare_tensor(0, tensor)
                backend.finish_request(0)

                calls = 0
                reclaim = backend._prefix_reclaim

                def counted_reclaim(active_model, required_free_bytes):
                    nonlocal calls
                    if required_free_bytes > 0:
                        calls += 1
                    return reclaim(active_model, required_free_bytes)

                backend._prefix_reclaim = counted_reclaim
                backend.begin_request(0, 1)
                result = backend.plan_request(1, second.tensors)
                self.assertEqual(1, calls)
                self.assertGreater(result.mckp_solver_time_ms, 0)
                for tensor in second.tensors:
                    backend.prepare_tensor(1, tensor)
                self.assertEqual(1, calls)
                self.assertLessEqual(backend.used_bytes(), 24)

    def test_tensor_page_prefix_physical_bytes_are_precomputed(self):
        models = model([6, 9, 3])
        backend = TensorPageMemory(models, 64, 8, args())
        self.assertEqual(0, backend._prefix_physical_bytes(0, 0))
        self.assertEqual(8, backend._prefix_physical_bytes(0, 1))
        self.assertEqual(24, backend._prefix_physical_bytes(0, 2))
        self.assertEqual(32, backend._prefix_physical_bytes(0, 3))
        clone = backend.clone()
        self.assertIs(
            backend.prefix_physical_bytes, clone.prefix_physical_bytes)

    def test_compact_pages_can_cross_tensor_boundary(self):
        models = model([6, 6])
        backend = CompactPageMemory(models, 32, 8, args())
        self.assertEqual(
            frozenset({0}),
            backend.tensor_pages[(0, 0)])
        self.assertEqual(
            frozenset({0, 1}),
            backend.tensor_pages[(0, 1)])
        first, second = models[0].tensors
        backend.prepare_tensor(0, first)
        self.assertFalse(backend.tensor_resident(0, second))
        result = backend.prepare_tensor(0, second)
        self.assertEqual(1, result.mapped_pages)
        self.assertTrue(backend.tensor_resident(0, second))

    def test_segment_compaction_moves_only_shifted_tensors(self):
        models = model([4, 4, 4, 6])
        backend = SegmentMemory(models, 14, 8, args())
        backend.begin_request(0, 0)
        for tensor in models[0].tensors[:3]:
            backend.prepare_tensor(0, tensor)
        backend._free((0, 1))
        # Free extents are [4,4] and [12,2], so the new 6-byte tensor requires
        # compaction despite enough total free space.
        result = backend.prepare_tensor(0, models[0].tensors[3])
        self.assertEqual(4, result.compaction_moved_bytes)

    def test_segment_allocated_index_tracks_free_and_compaction(self):
        models = model([4, 4, 4])
        backend = SegmentMemory(models, 12, 8, args())
        for tensor in models[0].tensors:
            backend.prepare_tensor(0, tensor)
        self.assertEqual(
            backend.allocated_extents[(0, 1)],
            backend._allocated_extent((0, 1)))
        backend._free((0, 1))
        self.assertNotIn((0, 1), backend.allocated_extents)
        backend._compact()
        self.assertEqual(
            {(0, 0), (0, 2)}, set(backend.allocated_extents))
        self.assertTrue(all(
            extent in backend.extents
            for extent in backend.allocated_extents.values()))

    def test_mock_segment_plans_all_missing_groups_at_request_level(self):
        models = model([4, 4, 4])
        backend = MockAllocationSegmentMemory(models, 12, 8, args())
        plan = backend.plan_request(0, models[0].tensors)
        self.assertEqual(0, plan.h2d_bytes)
        self.assertEqual(3, len(backend.planned_loads))
        loads = [
            backend.prepare_tensor(0, tensor)
            for tensor in models[0].tensors
        ]
        self.assertEqual(12, sum(load.h2d_bytes for load in loads))
        self.assertFalse(backend.planned_loads)

    def test_mock_segment_uses_explicit_kv_extents_without_weight_tail(self):
        models = model([4, 4])
        backend = MockAllocationSegmentMemory(models, 12, 8, args())
        backend.begin_request(4, 0)
        backend.plan_request(0, models[0].tensors)
        self.assertEqual(12, backend.used_bytes())
        self.assertEqual(4, backend.metrics()["kv_resident_bytes"])
        self.assertTrue(any(
            extent.key and extent.key[0] == "kv"
            for extent in backend.extents))
        backend.begin_request(0, 0)
        self.assertEqual(8, backend.used_bytes())
        self.assertEqual(0, backend.metrics()["kv_resident_bytes"])

    def test_mock_segment_kv_evicts_other_model_not_active_weights(self):
        first = model([4])[0]
        second = model([4])[0]
        second.model_id = 1
        models = {0: first, 1: second}
        backend = MockAllocationSegmentMemory(models, 12, 8, args())
        backend.begin_request(0, 1)
        backend.plan_request(1, second.tensors)
        backend.finish_request(1)
        backend.begin_request(8, 0)
        cost = backend.plan_request(0, first.tensors)
        self.assertEqual(1, cost.evicted_objects)
        self.assertNotIn((1, 0), backend.allocated_extents)
        self.assertIn((0, 0), backend.allocated_extents)
        self.assertEqual(8, backend.metrics()["kv_resident_bytes"])

    def test_compact_model_uses_whole_model_rounding(self):
        models = model([6, 6])
        compact = CompactPageMemory(models, 32, 8, args())
        tensor_page = TensorPageMemory(models, 32, 8, args())
        self.assertEqual(16, compact.physical_model_bytes(0))
        self.assertEqual(16, tensor_page.physical_model_bytes(0))

    def test_second_tensor_pipeline_activation_is_full_hit(self):
        models = model([4, 4])
        backend = SegmentMemory(models, 16, 8, args())

        class Estimator:
            models = {
                0: {
                    "compute": {
                        0: {"coefficients": [2.0, 0, 0, 0, 0]},
                    },
                },
            }

        request = type("Request", (), {
            "request_id": 0,
            "model_id": 0,
            "input_tokens": 8,
            "output_tokens": 1,
        })()
        options = args()
        options.kv_block_tokens = 32
        options.kv_block_bytes = 0
        first = execute_request(
            backend, request, models[0], Estimator(), options, True)
        second = execute_request(
            backend, request, models[0], Estimator(), options, True)
        self.assertGreater(first["h2d_bytes"], 0)
        self.assertEqual(0, second["h2d_bytes"])
        self.assertEqual(
            second["hot_compute_ms"], second["critical_path_ms"])

    def test_tensor_group_merges_compute_adjacent_tensors(self):
        models = model([3, 3, 7, 2])
        set_tensor_groups(models[0], 6)
        self.assertEqual(
            [(0, 1), (2, 3)],
            [group.tensor_ids for group in models[0].groups])
        self.assertEqual([6, 9], [
            group.logical_bytes for group in models[0].groups])

    def test_tensor_group_reduces_tensor_page_rounding(self):
        models = model([6, 6])
        set_tensor_groups(models[0], 8)
        backend = TensorPageMemory(models, 32, 8, args())
        backend.begin_request(0, 0)
        backend.prepare_tensor(0, models[0].tensors[0])
        self.assertEqual(16, backend.used_bytes())
        self.assertTrue(backend.tensor_resident(0, models[0].tensors[1]))
        self.assertEqual(
            12, backend.metrics()["logical_resident_bytes"])

    def test_pipeline_value_prefers_larger_stall_reduction(self):
        models = model([4, 12])
        options = args()
        options.replacement_policy = "pipeline-value"
        backend = SegmentMemory(models, 16, 8, options)
        backend.set_pipeline_benefits({(0, 0): 1.0, (0, 1): 4.0})
        backend.begin_request(0, 0)
        for tensor in models[0].tensors:
            backend.prepare_tensor(0, tensor)
        self.assertLess(
            backend._value((0, 0)),
            backend._value((0, 1)))

    def test_lru_uses_storage_unit_recency_only(self):
        models = model([4, 4])
        options = args()
        options.replacement_policy = "lru"
        backend = SegmentMemory(models, 8, 8, options)
        backend.model_accesses = {0: 100}
        backend.last_access = {(0, 0): 1, (0, 1): 2}
        self.assertEqual((1,), backend._value((0, 0)))
        self.assertLess(backend._value((0, 0)), backend._value((0, 1)))

    def test_suffix_first_lru_evicts_later_group_first(self):
        models = model([4, 4, 4])
        options = args()
        options.replacement_policy = "frequency-lru-suffix-first"
        backend = SegmentMemory(models, 12, 8, options)
        backend.last_access = {(0, 0): 1, (0, 1): 2, (0, 2): 3}
        self.assertEqual(
            (0, 2), backend._select_victim({(0, 0), (0, 1), (0, 2)}))

    def test_suffix_first_lru_keeps_model_frequency_priority(self):
        first = model([4, 4])[0]
        second = model([4])[0]
        second.model_id = 1
        models = {0: first, 1: second}
        options = args()
        options.replacement_policy = "frequency-lru-suffix-first"
        backend = SegmentMemory(models, 12, 8, options)
        backend.model_accesses = {0: 10, 1: 0}
        backend.last_access = {(0, 1): 1, (1, 0): 2}
        self.assertEqual(
            (1, 0), backend._select_victim({(0, 1), (1, 0)}))

    def test_compact_page_suffix_first_projects_compute_position(self):
        models = model([8, 8, 8])
        options = args()
        options.replacement_policy = "frequency-lru-suffix-first"
        backend = CompactPageMemory(models, 24, 8, options)
        for tensor in models[0].tensors:
            backend.prepare_tensor(0, tensor)
        self.assertLess(
            backend._page_value(0, 2),
            backend._page_value(0, 0))

    def test_pipeline_benefit_uses_local_ready_stall(self):
        models = model([4, 12])

        class Estimator:
            models = {
                0: {
                    "compute": {
                        0: {"coefficients": [2.0, 0, 0, 0, 0]},
                    },
                },
            }

        request = type("Request", (), {
            "request_id": 0,
            "model_id": 0,
            "input_tokens": 8,
            "output_tokens": 1,
        })()
        options = args()
        values = build_pipeline_benefits(
            models, Estimator(), [request], options)
        self.assertGreater(values[(0, 0)], 0)
        self.assertEqual(0.0, values[(0, 1)])

    def test_exposed_stall_routing_uses_round_robin_for_ties(self):
        candidates = [
            (0, {"exposed_load_ms": 3.0, "critical_path_ms": 8.0}),
            (1, {"exposed_load_ms": 3.0, "critical_path_ms": 8.0}),
        ]
        first, _, next_gpu = choose_routing_candidate(
            candidates, "exposed-stall", 0, 2)
        second, _, _ = choose_routing_candidate(
            candidates, "exposed-stall", next_gpu, 2)
        self.assertEqual((0, 1), (first, second))

    def test_cache_bytes_routing_prefers_more_current_model_residency(self):
        candidates = [
            (0, {"routing_cached_model_bytes": 4}),
            (1, {"routing_cached_model_bytes": 8}),
        ]
        chosen, _, _ = choose_routing_candidate(
            candidates, "cache-bytes", 0, 2)
        self.assertEqual(1, chosen)

    def test_mckp_transition_routes_on_stall_plus_weighted_damage(self):
        candidates = [
            (0, {
                "exposed_load_ms": 2.0,
                "mckp_predicted_damage_ms": 10.0,
                "mckp_transition_weight": 0.5,
            }),
            (1, {
                "exposed_load_ms": 4.0,
                "mckp_predicted_damage_ms": 1.0,
                "mckp_transition_weight": 0.5,
            }),
        ]
        chosen, _, _ = choose_routing_candidate(
            candidates, "mckp-transition", 0, 2)
        self.assertEqual(1, chosen)

    def test_mock_segment_mckp_releases_only_model_suffix(self):
        first = model([4, 4])[0]
        second = model([4, 4])[0]
        second.model_id = 1
        models = {0: first, 1: second}
        options = args()
        options.replacement_policy = "mckp-prefix"
        backend = MockAllocationSegmentMemory(models, 12, 8, options)
        profile = PrefixStallTable({
            "format": "layerweave-prefix-stall-v1",
            "metadata": {},
            "models": {
                str(model_id): {
                    "group_count": 2,
                    "prefix_bytes": [0, 4, 8],
                    "rows": [{
                        "input_tokens": 8,
                        "prefix_stall_ms": [8, 2, 0],
                    }],
                }
                for model_id in models
            },
        })
        backend.set_prefix_stall_table(profile)
        backend.set_mckp_prediction_samples({
            0: [(8, 1.0)], 1: [(8, 1.0)]})
        backend.begin_request(0, 0)
        backend.plan_request(0, first.tensors)
        for tensor in first.tensors:
            backend.prepare_tensor(0, tensor)
        backend.finish_request(0)
        backend.begin_request(0, 1)
        result = backend.plan_request(1, second.tensors)
        self.assertEqual(1, result.mckp_evicted_groups)
        self.assertIn((0, 0), backend.allocated_extents)
        self.assertNotIn((0, 1), backend.allocated_extents)
        self.assertEqual(1, backend.resident_prefix_count(0))

    def test_protected_pipeline_evicts_soft_group_first(self):
        models = model([4, 4])
        options = args()
        options.replacement_policy = "protected-pipeline"
        backend = SegmentMemory(models, 8, 8, options)
        backend.extents = [
            Extent(0, 4, (0, 0)),
            Extent(4, 4, (0, 1)),
        ]
        backend.set_pipeline_benefits({(0, 0): 0.0, (0, 1): 2.0})
        backend.set_future_model_probabilities({0: 1.0})
        self.assertEqual(
            (0, 0), backend._select_victim({(0, 0), (0, 1)}))

    def test_shadow_falls_back_when_soft_victim_has_more_churn(self):
        first = model([12])[0]
        second = model([4])[0]
        second.model_id = 1
        models = {0: first, 1: second}
        options = args()
        options.replacement_policy = "protected-pipeline-shadow"
        backend = SegmentMemory(models, 16, 8, options)
        backend.extents = [
            Extent(0, 12, (0, 0)),
            Extent(12, 4, (1, 0)),
        ]
        backend.model_accesses = {0: 10, 1: 0}
        backend.set_pipeline_benefits({(0, 0): 0.0, (1, 0): 2.0})
        backend.set_future_model_probabilities({0: 0.8, 1: 0.2})
        self.assertEqual(
            (1, 0), backend._select_victim({(0, 0), (1, 0)}))


if __name__ == "__main__":
    unittest.main()
