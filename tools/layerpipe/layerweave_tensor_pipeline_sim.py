#!/usr/bin/env python3
"""Tensor-level loading/cache simulator with three memory organizations.

Execution is tensor-granular.  Physical residency is selected by:
  segment      - one variable-sized contiguous cudaMalloc-pool segment/tensor
  tensor-page  - independently rounded VMM pages per tensor
  compact-page - pages over one compact stable-VA model arena

The initial implementation derives tensor compute windows by distributing the
measured M4 layer compute profile in proportion to tensor bytes.  This is an
explicit proxy until tensor/operator CUDA-event profiles are collected.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

from layerweave_estimator import batch_features
from layerweave_pipeline_cache_sim import (
    Request,
    _summary,
    load_model_limits,
    load_profile,
    load_trace as load_page_trace,
)


from tensor_sim_memory_common import (
    MIB, Extent, LoadCost, MemoryBackend, TensorGroup, TensorModel, TensorSpec,
    ceil_div, contiguous_runs,
)
from tensor_sim_segment_backend import (
    MockAllocationSegmentMemory, SegmentMemory,
)
from tensor_sim_page_backends import CompactPageMemory, TensorPageMemory
from tensor_sim_mckp import PrefixStallTable


BACKENDS = {
    "segment": MockAllocationSegmentMemory,
    "legacy-segment": SegmentMemory,
    "tensor-page": TensorPageMemory,
    "compact-page": CompactPageMemory,
}


def load_tensor_models(layout_path, limits):
    document = json.loads(layout_path.read_text())
    models = {}
    for key, item in document["models"].items():
        model_id = int(key)
        tensors = [
            TensorSpec(
                tensor_id=int(t["tensor_id"]),
                name=t["name"],
                layer_id=(
                    None if t.get("layer_id") is None else int(t["layer_id"])
                ),
                logical_bytes=int(t["logical_bytes"]),
                compact_offset=int(t["compact_offset"]),
            )
            for t in item["tensors"]
        ]
        model = TensorModel(
            model_id=model_id,
            tensors=tensors,
            logical_bytes=sum(t.logical_bytes for t in tensors),
            max_input_tokens=limits[model_id],
        )
        set_tensor_groups(model, 0)
        models[model_id] = model
    return models


def set_tensor_groups(model, minimum_bytes):
    """Greedily merge compute-adjacent tensors to a minimum group size."""
    groups = []
    current = []
    current_bytes = 0
    current_offset = 0
    for tensor in model.tensors:
        if not current:
            current_offset = tensor.compact_offset
        current.append(tensor.tensor_id)
        current_bytes += tensor.logical_bytes
        if minimum_bytes <= 0 or current_bytes >= minimum_bytes:
            groups.append(TensorGroup(
                len(groups), tuple(current), current_bytes, current_offset))
            current = []
            current_bytes = 0
    if current:
        if groups and minimum_bytes > 0:
            previous = groups.pop()
            groups.append(TensorGroup(
                previous.group_id,
                previous.tensor_ids + tuple(current),
                previous.logical_bytes + current_bytes,
                previous.compact_offset,
            ))
        else:
            groups.append(TensorGroup(
                len(groups), tuple(current), current_bytes, current_offset))
    model.groups = groups
    model.tensor_to_group = {
        tensor_id: group.group_id
        for group in groups for tensor_id in group.tensor_ids
    }


def models_with_group_size(models, minimum_bytes):
    grouped = copy.deepcopy(models)
    for model in grouped.values():
        set_tensor_groups(model, minimum_bytes)
    return grouped


def models_from_prefix_stall_table(table, limits):
    """Build synthetic execution units matching runtime-profile groups."""
    models = {}
    for model_id, item in table.models.items():
        prefix_bytes = item["prefix_bytes"]
        tensors = []
        groups = []
        for group_id, (left, right) in enumerate(zip(
                prefix_bytes, prefix_bytes[1:])):
            size = right - left
            tensors.append(TensorSpec(
                group_id, f"runtime_group_{group_id}", group_id, size, left))
            groups.append(TensorGroup(
                group_id, (group_id,), size, left))
        models[model_id] = TensorModel(
            model_id=model_id,
            tensors=tensors,
            logical_bytes=prefix_bytes[-1],
            max_input_tokens=limits[model_id],
            groups=groups,
            tensor_to_group={
                group.group_id: group.group_id for group in groups
            },
        )
    return models


def load_model_mapping(config_path, model_ids):
    """Return the trace-model-id -> physical-model-id mapping."""
    document = json.loads(config_path.read_text())
    physical_ids = set(int(model_id) for model_id in model_ids)
    order = document.get("model_mapping_order")
    if order is None:
        return {model_id: model_id for model_id in physical_ids}
    if not isinstance(order, list):
        raise ValueError("model_mapping_order must be a list")
    mapping = {
        trace_model_id: int(physical_model_id)
        for trace_model_id, physical_model_id in enumerate(order)
    }
    if set(mapping) != physical_ids or set(mapping.values()) != physical_ids:
        raise ValueError(
            "model_mapping_order must be a permutation of all model ids; "
            f"expected {sorted(physical_ids)}, got {order}")
    return mapping


def remap_requests(requests, model_mapping):
    """Map trace-facing model ids to physical profile/table model ids."""
    return [
        Request(
            request.request_id,
            model_mapping[request.model_id],
            request.input_tokens,
            request.output_tokens,
            request.trace_output_tokens,
        )
        for request in requests
    ]


def compute_durations(estimator, request, model):
    tensor_compute = getattr(estimator, "tensor_compute", {})
    tensor_model = tensor_compute.get(str(request.model_id), {})
    profile = estimator.models[request.model_id]
    features = batch_features({
        "batch_size": 1,
        "input_tokens": request.input_tokens,
        "max_input_tokens": request.input_tokens,
        "sum_input_tokens_squared": request.input_tokens ** 2,
    })
    by_layer = {}
    for tensor in model.tensors:
        by_layer.setdefault(tensor.layer_id, []).append(tensor)
    durations = {}
    for layer_id, tensors in by_layer.items():
        if layer_id is None:
            for tensor in tensors:
                durations[tensor.tensor_id] = 0.0
            continue
        item = profile["compute"].get(layer_id, {})
        coefficients = item.get("coefficients")
        layer_ms = (
            max(0.0, sum(c * f for c, f in zip(coefficients, features)))
            if coefficients is not None else 0.0
        )
        total_bytes = sum(t.logical_bytes for t in tensors)
        for tensor in tensors:
            durations[tensor.tensor_id] = (
                layer_ms * tensor.logical_bytes / total_bytes
                if total_bytes else 0.0
            )
    for tensor in model.tensors:
        item = tensor_model.get(tensor.name)
        if item is None:
            continue
        intercept, per_token = item["coefficients"]
        durations[tensor.tensor_id] = max(
            0.0, float(intercept)
            + float(per_token) * request.input_tokens)
    return durations


def kv_bytes(request, args):
    blocks = ceil_div(
        request.input_tokens + request.output_tokens, args.kv_block_tokens)
    return blocks * args.kv_block_bytes


def _pipeline_group_stalls(model, durations, group_load_ms):
    copy_end = 0.0
    compute_end = 0.0
    previous_compute_start = 0.0
    loaded = set()
    stalls = {group.group_id: 0.0 for group in model.groups}
    for index, tensor in enumerate(model.tensors):
        group_id = model.tensor_to_group[tensor.tensor_id]
        load_ms = 0.0
        if group_id not in loaded:
            loaded.add(group_id)
            load_ms = group_load_ms[group_id]
        load_start = (
            copy_end if index == 0
            else max(copy_end, previous_compute_start)
        )
        load_end = load_start + load_ms
        if load_ms:
            stalls[group_id] = max(0.0, load_end - compute_end)
        compute_start = max(compute_end, load_end)
        previous_compute_start = compute_start
        compute_end = compute_start + durations[tensor.tensor_id]
        copy_end = load_end
    return stalls


def build_pipeline_benefits(
        models, estimator, requests, args, representative_requests=None):
    """Estimate each storage unit's marginal exposed-stall reduction."""
    by_model = {}
    for request in requests:
        by_model.setdefault(request.model_id, []).append(request)
    benefits = {}
    for model_id, model in models.items():
        candidates = by_model.get(model_id, [])
        ordered = sorted(candidates, key=lambda item: item.input_tokens)
        representative = (
            None if representative_requests is None
            else representative_requests.get(model_id)
        )
        if representative is not None:
            profile_requests = [representative]
        elif args.replacement_policy.startswith("protected-pipeline") and ordered:
            indexes = sorted(set(
                round(fraction * (len(ordered) - 1))
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
            ))
            profile_requests = [ordered[index] for index in indexes]
        elif ordered:
            profile_requests = [ordered[len(ordered) // 2]]
        else:
            profile_requests = [Request(0, model_id, 1, 1, 1)]
        loads = {}
        for group in model.groups:
            load = group.logical_bytes / args.h2d_bytes_per_ms
            if args.memory_layout in ("segment", "legacy-segment"):
                load += args.segment_allocation_ms
            elif args.memory_layout == "tensor-page":
                pages = ceil_div(group.logical_bytes, args.page_size_bytes)
                load += pages * (
                    args.map_fixed_ms + args.map_per_page_ms)
            else:
                first = group.compact_offset // args.page_size_bytes
                last = (
                    group.compact_offset + group.logical_bytes - 1
                ) // args.page_size_bytes
                pages = last - first + 1
                load += pages * (
                    args.map_fixed_ms + args.map_per_page_ms)
            loads[group.group_id] = load
        samples = {group.group_id: [] for group in model.groups}
        for request in profile_requests:
            durations = compute_durations(estimator, request, model)
            stalls = _pipeline_group_stalls(model, durations, loads)
            for group_id, stall in stalls.items():
                samples[group_id].append(stall)
        for group in model.groups:
            values = sorted(samples[group.group_id])
            if args.replacement_policy.startswith("protected-pipeline"):
                if args.pipeline_robust_stat == "max":
                    marginal = values[-1]
                elif args.pipeline_robust_stat == "mean":
                    marginal = statistics.mean(values)
                else:
                    marginal = values[math.ceil(0.9 * len(values)) - 1]
            else:
                marginal = values[0]
            if args.memory_layout != "compact-page":
                benefits[(model_id, group.group_id)] = marginal
                continue
            first = group.compact_offset // args.page_size_bytes
            last = (
                group.compact_offset + group.logical_bytes - 1
            ) // args.page_size_bytes
            pages = max(1, last - first + 1)
            contribution = marginal / pages
            for page in range(first, last + 1):
                benefits[(model_id, page)] = (
                    benefits.get((model_id, page), 0.0) + contribution
                )
    return benefits


def execute_request(
        backend, request, model, estimator, args, pipelined,
        collect_timeline=False):
    total = backend.begin_request(kv_bytes(request, args), request.model_id)
    total.add(backend.plan_request(request.model_id, model.tensors))
    durations = compute_durations(estimator, request, model)
    copy_end = total.ready_ms
    compute_end = total.ready_ms if pipelined else 0.0
    previous_compute_start = 0.0
    timeline = []
    for index, tensor in enumerate(model.tensors):
        load = backend.prepare_tensor(request.model_id, tensor)
        total.add(load)
        if pipelined:
            load_start = (
                copy_end if index == 0
                else max(copy_end, previous_compute_start)
            )
            load_end = load_start + load.ready_ms
            compute_start = max(compute_end, load_end)
            ready_stall = max(0.0, load_end - compute_end)
            previous_compute_start = compute_start
            compute_end = compute_start + durations[tensor.tensor_id]
            copy_end = load_end
        else:
            load_start = 0.0
            load_end = 0.0
            compute_start = 0.0
            ready_stall = 0.0
        if collect_timeline:
            timeline.append({
                "stage_id": tensor.tensor_id,
                "tensor": tensor.name,
                "logical_bytes": tensor.logical_bytes,
                "load_start_ms": load_start,
                "load_end_ms": load_end,
                "load_ms": load.ready_ms,
                "compute_start_ms": compute_start,
                "compute_ms": durations[tensor.tensor_id],
                "ready_stall_ms": ready_stall,
            })
    hot_compute = sum(durations.values())
    if not pipelined:
        compute_end = total.ready_ms + hot_compute
    backend.finish_request(request.model_id)
    return {
        "request_id": request.request_id,
        "model_id": request.model_id,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "critical_path_ms": compute_end,
        "hot_compute_ms": hot_compute,
        "exposed_load_ms": max(0.0, compute_end - hot_compute),
        "h2d_ms": total.h2d_ms,
        "h2d_bytes": total.h2d_bytes,
        "allocation_ms": total.allocation_ms,
        "map_ms": total.map_ms,
        "unmap_ms": total.unmap_ms,
        "compaction_ms": total.compaction_ms,
        "compaction_moved_bytes": total.compaction_moved_bytes,
        "evicted_objects": total.evicted_objects,
        "evicted_bytes": total.evicted_bytes,
        "map_calls": total.map_calls,
        "unmap_calls": total.unmap_calls,
        "mapped_pages": total.mapped_pages,
        "unmapped_pages": total.unmapped_pages,
        "mckp_required_free_bytes": total.mckp_required_free_bytes,
        "mckp_planned_free_bytes": total.mckp_planned_free_bytes,
        "mckp_overrelease_bytes": max(
            0, total.mckp_planned_free_bytes
            - total.mckp_required_free_bytes),
        "mckp_predicted_damage_ms": total.mckp_predicted_damage_ms,
        "mckp_solver_time_ms": total.mckp_solver_time_ms,
        "mckp_frontier_peak_states": total.mckp_frontier_peak_states,
        "mckp_evicted_groups": total.mckp_evicted_groups,
        "mckp_evicted_models": total.mckp_evicted_models,
        "pbp_planner_time_ms": total.pbp_planner_time_ms,
        "model_reload_events": total.model_reload_events,
        "naive_relocation_bytes": total.naive_relocation_bytes,
        "memory": backend.metrics(),
        "timeline": timeline,
    }


def execute_profile_request(
        backend, request, model, table, args, collect_timeline=False,
        pipelined=True):
    """Execute memory actions and settle latency from the shared profile."""
    if args.replacement_policy == "lru":
        residency_kind = "suffix"
        residency_state = backend.resident_suffix_start(request.model_id)
    else:
        residency_kind = "prefix"
        residency_state = backend.resident_prefix_count(request.model_id)
    total = backend.begin_request(kv_bytes(request, args), request.model_id)
    total.add(backend.plan_request(request.model_id, model.tensors))
    for tensor in model.tensors:
        total.add(backend.prepare_tensor(request.model_id, tensor))
    backend.finish_request(request.model_id)
    hot_compute = table.baseline_ms(
        request.model_id, request.input_tokens)
    profile_exposed = None
    if pipelined:
        profile_exposed = (
            table.lookup_suffix(
                request.model_id, request.input_tokens, residency_state)
            if residency_kind == "suffix" else
            table.lookup(
                request.model_id, request.input_tokens, residency_state)
        )
        exposed_h2d_ms, exposed_allocation_ms = (
            table.exposed_loading_components(
                request.model_id, request.input_tokens, residency_state,
                residency_kind)
        )
    else:
        exposed_h2d_ms = total.h2d_ms
        exposed_allocation_ms = total.allocation_ms
    # Request-level PBP relocation and CUDA VMM mapping operations happen
    # outside the offline prefix-loading profile. They are fully charged to
    # request TTFT; page backends count one API operation per physical page.
    vmm_ms = total.map_ms + total.unmap_ms
    exposed = (
        total.compaction_ms + vmm_ms + profile_exposed
        if pipelined else
        (
            total.h2d_ms + total.allocation_ms
            + total.compaction_ms + vmm_ms
        )
    )
    exposed_compaction_ms = total.compaction_ms
    exposed_page_map_ms = total.map_ms
    exposed_page_unmap_ms = total.unmap_ms
    row = {
        "request_id": request.request_id,
        "model_id": request.model_id,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "critical_path_ms": hot_compute + exposed,
        "hot_compute_ms": hot_compute,
        "exposed_load_ms": exposed,
        "exposed_h2d_ms": exposed_h2d_ms,
        "exposed_allocation_ms": exposed_allocation_ms,
        "exposed_compaction_ms": exposed_compaction_ms,
        "exposed_page_map_ms": exposed_page_map_ms,
        "exposed_page_unmap_ms": exposed_page_unmap_ms,
        "exposed_page_map_unmap_ms": (
            exposed_page_map_ms + exposed_page_unmap_ms),
        "pbp_relocation_ms": total.compaction_ms,
        "vmm_ms": vmm_ms,
        "h2d_ms": total.h2d_ms,
        "h2d_bytes": total.h2d_bytes,
        "allocation_ms": total.allocation_ms,
        "map_ms": total.map_ms,
        "unmap_ms": total.unmap_ms,
        "compaction_ms": total.compaction_ms,
        "compaction_moved_bytes": total.compaction_moved_bytes,
        "evicted_objects": total.evicted_objects,
        "evicted_bytes": total.evicted_bytes,
        "map_calls": total.map_calls,
        "unmap_calls": total.unmap_calls,
        "mapped_pages": total.mapped_pages,
        "unmapped_pages": total.unmapped_pages,
        "mckp_required_free_bytes": total.mckp_required_free_bytes,
        "mckp_planned_free_bytes": total.mckp_planned_free_bytes,
        "mckp_overrelease_bytes": max(
            0, total.mckp_planned_free_bytes
            - total.mckp_required_free_bytes),
        "mckp_predicted_damage_ms": total.mckp_predicted_damage_ms,
        "mckp_solver_time_ms": total.mckp_solver_time_ms,
        "mckp_frontier_peak_states": total.mckp_frontier_peak_states,
        "mckp_evicted_groups": total.mckp_evicted_groups,
        "mckp_evicted_models": total.mckp_evicted_models,
        "pbp_planner_time_ms": total.pbp_planner_time_ms,
        "model_reload_events": total.model_reload_events,
        "naive_relocation_bytes": total.naive_relocation_bytes,
        "residency_kind": residency_kind,
        "residency_state_before": residency_state,
        "resident_prefix_before": (
            residency_state if residency_kind == "prefix" else None),
        "resident_suffix_start_before": (
            residency_state if residency_kind == "suffix" else None),
        "resident_prefix_after": (
            backend.resident_prefix_count(request.model_id)
            if residency_kind == "prefix" else None),
        "resident_suffix_start_after": (
            backend.resident_suffix_start(request.model_id)
            if residency_kind == "suffix" else None),
        "memory": backend.metrics(),
        "timeline": [] if not collect_timeline else [{
            "profile_source": table.source if pipelined else None,
            "loading_overlap": "pipelined" if pipelined else "serialized",
            "residency_kind": residency_kind,
            "residency_state_before": residency_state,
            "profile_exposed_load_ms": profile_exposed,
            "h2d_ms": total.h2d_ms,
            "allocation_ms": total.allocation_ms,
            "pbp_relocation_ms": total.compaction_ms,
            "vmm_ms": vmm_ms,
            "exposed_load_ms": exposed,
        }],
    }
    if profile_exposed is not None:
        row["profile_exposed_load_ms"] = profile_exposed
    return row


def execute_profile_cold_request(
        request, model, table, args, pipelined):
    """Execute a stateless cold request with a shared profiled prefill."""
    backend = make_backend({request.model_id: model}, args)
    total = backend.begin_request(kv_bytes(request, args), request.model_id)
    total.add(backend.plan_request(request.model_id, model.tensors))
    for tensor in model.tensors:
        total.add(backend.prepare_tensor(request.model_id, tensor))
    backend.finish_request(request.model_id)
    hot_compute = table.baseline_ms(request.model_id, request.input_tokens)
    vmm_ms = total.map_ms + total.unmap_ms
    if pipelined:
        profile_exposed = table.lookup(
            request.model_id, request.input_tokens, 0)
        exposed_h2d_ms, exposed_allocation_ms = (
            table.exposed_loading_components(
                request.model_id, request.input_tokens, 0, "prefix")
        )
        exposed = profile_exposed + total.compaction_ms + vmm_ms
    else:
        profile_exposed = None
        exposed_h2d_ms = total.h2d_ms
        exposed_allocation_ms = total.allocation_ms
        exposed = (
            total.h2d_ms + total.allocation_ms
            + total.compaction_ms + vmm_ms
        )
    row = {
        "request_id": request.request_id,
        "model_id": request.model_id,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "critical_path_ms": hot_compute + exposed,
        "hot_compute_ms": hot_compute,
        "exposed_load_ms": exposed,
        "exposed_h2d_ms": exposed_h2d_ms,
        "exposed_allocation_ms": exposed_allocation_ms,
        "exposed_compaction_ms": total.compaction_ms,
        "exposed_page_map_ms": total.map_ms,
        "exposed_page_unmap_ms": total.unmap_ms,
        "exposed_page_map_unmap_ms": vmm_ms,
        "h2d_ms": total.h2d_ms,
        "h2d_bytes": total.h2d_bytes,
        "allocation_ms": total.allocation_ms,
        "map_ms": total.map_ms,
        "unmap_ms": total.unmap_ms,
        "compaction_ms": total.compaction_ms,
        "compaction_moved_bytes": total.compaction_moved_bytes,
        "evicted_objects": total.evicted_objects,
        "evicted_bytes": total.evicted_bytes,
        "map_calls": total.map_calls,
        "unmap_calls": total.unmap_calls,
        "mapped_pages": total.mapped_pages,
        "unmapped_pages": total.unmapped_pages,
        "pbp_planner_time_ms": total.pbp_planner_time_ms,
        "model_reload_events": total.model_reload_events,
        "naive_relocation_bytes": total.naive_relocation_bytes,
        "memory": backend.metrics(),
    }
    if profile_exposed is not None:
        row["profile_exposed_load_ms"] = profile_exposed
        row["pbp_relocation_ms"] = total.compaction_ms
        row["vmm_ms"] = vmm_ms
    return row


def summarize(rows):
    fields = (
        "critical_path_ms", "hot_compute_ms", "exposed_load_ms", "h2d_ms",
        "allocation_ms", "map_ms", "unmap_ms", "compaction_ms",
    )
    if rows and "exposed_h2d_ms" in rows[0]:
        fields += (
            "exposed_h2d_ms", "exposed_allocation_ms",
            "exposed_compaction_ms", "exposed_page_map_ms",
            "exposed_page_unmap_ms", "exposed_page_map_unmap_ms",
        )
    if rows and "profile_exposed_load_ms" in rows[0]:
        fields += (
            "profile_exposed_load_ms", "pbp_relocation_ms", "vmm_ms")
    result = {"requests": len(rows)}
    for field in fields:
        result[field] = _summary(row[field] for row in rows)
    for field in (
        "h2d_bytes", "compaction_moved_bytes", "evicted_objects",
        "evicted_bytes", "map_calls", "unmap_calls", "mapped_pages",
        "unmapped_pages", "mckp_required_free_bytes",
        "mckp_planned_free_bytes", "mckp_overrelease_bytes",
        "mckp_frontier_peak_states", "mckp_evicted_groups",
        "mckp_evicted_models",
        "model_reload_events", "naive_relocation_bytes",
    ):
        result[field] = sum(row.get(field, 0) for row in rows)
    for field in ("mckp_predicted_damage_ms", "mckp_solver_time_ms"):
        result[field] = _summary(row.get(field, 0.0) for row in rows)
    result["pbp_planner_time_ms"] = _summary(
        row.get("pbp_planner_time_ms", 0.0) for row in rows)
    for field in (
        "logical_resident_bytes", "physical_resident_bytes",
        "internal_fragmentation_bytes", "external_fragmentation_ratio",
        "largest_free_extent_bytes", "stranded_resident_bytes",
        "kv_resident_bytes",
    ):
        result[f"memory_{field}"] = _summary(
            row["memory"].get(field, 0) for row in rows)
    return result


def write_per_model_csv(path, policy, rows):
    """Write the requested compact policy-by-model mean breakdown."""
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "policy", "model_id", "request_count",
            "exposed_load_ms_mean", "prefill_ms_mean",
        ))
        writer.writeheader()
        for model_id in sorted(by_model):
            model_rows = by_model[model_id]
            writer.writerow({
                "policy": policy,
                "model_id": model_id,
                "request_count": len(model_rows),
                "exposed_load_ms_mean": (
                    sum(row["exposed_load_ms"] for row in model_rows)
                    / len(model_rows)
                ),
                "prefill_ms_mean": (
                    sum(row["hot_compute_ms"] for row in model_rows)
                    / len(model_rows)
                ),
            })


def load_trace_arrivals(
        path, max_requests, trace_mode, request_count, arrival_mode,
        request_rate_rps, seed):
    """Return normalized request arrivals matching load_page_trace filtering."""
    source_requests = 0
    previous_model_id = None
    raw_arrivals = []
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
            if keep:
                raw_arrivals.append(float(fields[0]))
            if max_requests > 0 and source_requests >= max_requests:
                break
    if len(raw_arrivals) != request_count:
        raise ValueError(
            "arrival/request filtering mismatch: "
            f"arrivals={len(raw_arrivals)} requests={request_count}")
    if not raw_arrivals:
        return []
    if arrival_mode == "poisson":
        rng = random.Random(seed)
        arrivals = [0.0]
        for _ in range(1, request_count):
            arrivals.append(
                arrivals[-1] + rng.expovariate(request_rate_rps))
        return arrivals
    normalized = [value - raw_arrivals[0] for value in raw_arrivals]
    span = normalized[-1]
    if request_count <= 1:
        return [0.0]
    if span <= 0:
        raise ValueError(
            "trace-scaled arrivals require a positive source trace span")
    original_rate = (request_count - 1) / span
    scale = original_rate / request_rate_rps
    return [value * scale for value in normalized]


def apply_online_timing(
        rows, routing, requests, arrivals_s, args, system):
    """Add arrival-to-first-token timing and Decode GPU occupancy."""
    if len(rows) != len(requests) or len(rows) != len(arrivals_s):
        raise ValueError("online timing inputs have different lengths")
    available_at_ms = [0.0] * args.gpus
    online_routing = list(routing)
    next_gpu = 0
    for index, (row, request, arrival_s) in enumerate(
            zip(rows, requests, arrivals_s)):
        arrival_ms = arrival_s * 1000.0
        if index < len(online_routing):
            gpu = online_routing[index]
        else:
            earliest = min(available_at_ms)
            tied = {
                gpu for gpu, value in enumerate(available_at_ms)
                if math.isclose(value, earliest, abs_tol=1e-9)
            }
            gpu = next(
                candidate for offset in range(args.gpus)
                if (
                    candidate := (next_gpu + offset) % args.gpus
                ) in tied
            )
            next_gpu = (gpu + 1) % args.gpus
            online_routing.append(gpu)
            row["gpu"] = gpu
        dispatch_ms = max(arrival_ms, available_at_ms[gpu])
        queue_ms = dispatch_ms - arrival_ms
        service_ttft_ms = row["critical_path_ms"]
        first_token_ms = dispatch_ms + service_ttft_ms
        decode_ms = (
            request.output_tokens * args.decode_ms_per_token)
        finish_ms = first_token_ms + decode_ms
        available_at_ms[gpu] = finish_ms
        ttft_ms = first_token_ms - arrival_ms
        slo_ms = args.slo_scale * row["hot_compute_ms"]
        row.update({
            "arrival_ms": arrival_ms,
            "dispatch_ms": dispatch_ms,
            "first_token_ms": first_token_ms,
            "finish_ms": finish_ms,
            "queue_ms": queue_ms,
            "service_ttft_ms": service_ttft_ms,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "slo_ms": slo_ms,
            "slo_attained": ttft_ms <= slo_ms,
            "measured": index >= args.warmup_requests,
        })
    measured_rows = rows[args.warmup_requests:]
    summary = summarize(measured_rows)
    for field in (
            "queue_ms", "service_ttft_ms", "ttft_ms", "decode_ms",
            "slo_ms"):
        summary[field] = _summary(row[field] for row in measured_rows)
    summary["slo_attainment"] = (
        sum(row["slo_attained"] for row in measured_rows)
        / len(measured_rows)
        if measured_rows else 0.0)
    measurement_start_ms = (
        measured_rows[0]["arrival_ms"] if measured_rows else 0.0)
    makespan_ms = max(
        (row["finish_ms"] for row in measured_rows),
        default=measurement_start_ms)
    measurement_span_ms = max(0.0, makespan_ms - measurement_start_ms)
    summary["makespan_ms"] = makespan_ms
    summary["throughput_requests_s"] = (
        len(measured_rows) * 1000.0 / measurement_span_ms
        if measurement_span_ms > 0 else 0.0)
    summary["total_requests"] = len(rows)
    summary["warmup_requests"] = args.warmup_requests
    summary["measured_requests"] = len(measured_rows)
    summary["measurement_start_ms"] = measurement_start_ms
    summary["measurement_span_ms"] = measurement_span_ms
    summary["offered_request_rate_rps"] = args.request_rate_rps
    summary["arrival_mode"] = args.arrival_mode
    summary["execution_mode"] = "online"
    summary["slo_semantics"] = (
        f"TTFT <= {args.slo_scale:g} * fully-resident Prefill")
    return online_routing, summary


def write_request_csv(path, policy, rows):
    """Write one row per request, including online TTFT when available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "policy", "request_id", "model_id", "gpu", "input_tokens",
        "output_tokens", "arrival_ms", "dispatch_ms", "first_token_ms",
        "finish_ms", "queue_ms", "service_ttft_ms", "ttft_ms",
        "decode_ms", "critical_path_ms", "exposed_load_ms",
        "hot_compute_ms", "slo_ms", "slo_attained", "measured",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: policy if field == "policy" else row.get(field)
                for field in fields
            })


def make_backend(models, args, benefits=None):
    backend = BACKENDS[args.memory_layout](
        models, args.pool_bytes, args.page_size_bytes, args)
    if benefits is not None:
        backend.set_pipeline_benefits(benefits)
    if getattr(args, "prefix_stall_table", None) is not None:
        backend.set_prefix_stall_table(args.prefix_stall_table)
    return backend


def validate_request_feasibility(models, requests, args):
    probe = make_backend(models, args)
    failures = []
    for request in requests:
        footprint = probe.physical_model_bytes(request.model_id)
        request_kv_bytes = kv_bytes(request, args)
        required = footprint + request_kv_bytes
        if required > args.pool_bytes:
            failures.append({
                "request_id": request.request_id,
                "model_id": request.model_id,
                "weight_footprint_bytes": footprint,
                "kv_bytes": request_kv_bytes,
                "required_bytes": required,
            })
    if failures:
        raise ValueError(
            "Memory layout is infeasible before simulation: "
            f"layout={args.memory_layout}, page_size={args.page_size_bytes}, "
            f"pool={args.pool_bytes}, first_failure={failures[0]}, "
            f"failure_count={len(failures)}")


def choose_routing_candidate(candidates, routing_policy, next_gpu, gpus):
    if routing_policy == "cache-bytes":
        scored = [
            (-result["routing_cached_model_bytes"], gpu, result)
            for gpu, result in candidates
        ]
    elif routing_policy == "mckp-transition":
        scored = [
            (
                result["exposed_load_ms"]
                + result.get("mckp_predicted_damage_ms", 0.0)
                * result.get("mckp_transition_weight", 1.0),
                gpu,
                result,
            )
            for gpu, result in candidates
        ]
    else:
        score_field = (
            "exposed_load_ms"
            if routing_policy == "exposed-stall" else "critical_path_ms"
        )
        scored = [
            (result[score_field], gpu, result)
            for gpu, result in candidates
        ]
    best = min(item[0] for item in scored)
    tied = {
        gpu: result for score, gpu, result in scored
        if math.isclose(score, best, abs_tol=1e-9)
    }
    chosen = next(
        gpu for offset in range(gpus)
        if (gpu := (next_gpu + offset) % gpus) in tied
    )
    return chosen, tied[chosen], (chosen + 1) % gpus


def choose_online_routing_candidate(
        candidates, available_at_ms, arrival_ms, next_gpu, gpus,
        transition_weight=0.0):
    """Choose the lowest arrival-to-first-token candidate."""
    scored = []
    for gpu, result in candidates:
        queue_ms = max(0.0, available_at_ms[gpu] - arrival_ms)
        score = (
            queue_ms + result["critical_path_ms"]
            + transition_weight
            * result.get("mckp_predicted_damage_ms", 0.0)
        )
        result["predicted_queue_ms"] = queue_ms
        result["online_routing_score_ms"] = score
        scored.append((score, gpu, result))
    best = min(score for score, _, _ in scored)
    tied = {
        gpu: result for score, gpu, result in scored
        if math.isclose(score, best, abs_tol=1e-9)
    }
    chosen = next(
        gpu for offset in range(gpus)
        if (gpu := (next_gpu + offset) % gpus) in tied
    )
    return chosen, tied[chosen], (chosen + 1) % gpus


def choose_online_cache_bytes_candidate(
        candidates, available_at_ms, arrival_ms, next_gpu, gpus):
    """Apply cache-bytes only among GPUs dispatchable at the same time."""
    idle = [
        (gpu, result) for gpu, result in candidates
        if available_at_ms[gpu] <= arrival_ms
    ]
    if idle:
        eligible = idle
    else:
        earliest = min(available_at_ms[gpu] for gpu, _ in candidates)
        eligible = [
            (gpu, result) for gpu, result in candidates
            if math.isclose(
                available_at_ms[gpu], earliest, abs_tol=1e-9)
        ]
    return choose_routing_candidate(
        eligible, "cache-bytes", next_gpu, gpus)


def update_online_gpu_availability(args, request, gpu, critical_path_ms):
    if args.execution_mode != "online":
        return
    arrival_ms = args.online_arrivals_ms[request.request_id]
    dispatch_ms = max(arrival_ms, args.online_available_at_ms[gpu])
    args.online_available_at_ms[gpu] = (
        dispatch_ms + critical_path_ms
        + request.output_tokens * args.decode_ms_per_token
    )


def build_gpu_availability(
        requests, gpus, busy_probability, seed, exact_available=None):
    """Build a policy-independent Bernoulli GPU availability mask."""
    rng = random.Random(seed)
    all_gpus = tuple(range(gpus))
    result = []
    for request in requests:
        if exact_available is not None:
            available_set = set(rng.sample(all_gpus, exact_available))
            busy_set = set(all_gpus) - available_set
            forced_available_gpu = None
        else:
            busy_set = {
                gpu for gpu in all_gpus if rng.random() < busy_probability
            }
            forced_available_gpu = None
            if len(busy_set) == gpus:
                forced_available_gpu = rng.choice(all_gpus)
                busy_set.remove(forced_available_gpu)
        busy = tuple(sorted(busy_set))
        available = tuple(gpu for gpu in all_gpus if gpu not in busy_set)
        result.append({
            "request_id": request.request_id,
            "available_gpus": available,
            "busy_gpus": busy,
            "forced_available_gpu": forced_available_gpu,
        })
    return result


def _aegaeon_profile_row(
        request, gpu, previous, model, models, table, args):
    hot = table.baseline_ms(request.model_id, request.input_tokens)
    footprint = model.logical_bytes
    h2d = footprint / args.h2d_bytes_per_ms
    same = previous is not None and previous.model_id == request.model_id
    window = (
        previous.trace_output_tokens * args.decode_ms_per_token
        if previous is not None and not same else 0.0
    )
    previous_footprint = (
        models[previous.model_id].logical_bytes
        if previous is not None else 0
    )
    previous_kv = kv_bytes(previous, args) if previous is not None else 0
    feasible = (
        previous is not None
        and previous_footprint + previous_kv + footprint <= args.pool_bytes
    )
    hidden = min(h2d, window) if feasible and not same else 0.0
    relocation = (
        footprint / args.gpu_copy_bytes_per_ms
        if feasible and not same else 0.0
    )
    allocation_ms = args.segment_allocation_ms if not same else 0.0
    exposed = (
        0.0 if same
        else h2d - hidden + relocation + allocation_ms
    )
    return {
        "request_id": request.request_id,
        "model_id": request.model_id,
        "gpu": gpu,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "critical_path_ms": hot + exposed,
        "hot_compute_ms": hot,
        "exposed_load_ms": exposed,
        "h2d_ms": 0.0 if same else h2d,
        "h2d_bytes": 0 if same else footprint,
        "allocation_ms": allocation_ms,
        "map_ms": 0.0,
        "unmap_ms": 0.0,
        "compaction_ms": 0.0,
        "compaction_moved_bytes": 0,
        "evicted_objects": 0,
        "evicted_bytes": 0,
        "map_calls": 0,
        "unmap_calls": 0,
        "mapped_pages": 0,
        "unmapped_pages": 0,
        "memory": {
            "logical_resident_bytes": footprint,
            "physical_resident_bytes": footprint,
            "internal_fragmentation_bytes": 0,
            "external_fragmentation_ratio": 0.0,
            "largest_free_extent_bytes":
                max(0, args.pool_bytes - footprint),
            "stranded_resident_bytes": 0,
            "kv_resident_bytes": kv_bytes(request, args),
        },
        "decode_overlap_window_ms": window,
        "prefetched_h2d_ms": hidden,
        "gpu_relocation_ms": relocation,
        "capacity_feasible": feasible,
        "same_model_hit": same,
    }


def simulate_single_system(args, models, table, requests):
    """Run one independently routed system using the shared Offline table."""
    system = args.system
    rows = []
    routing = []
    routing_candidates = {}
    if system in ("baseline", "pipe-only"):
        pipelined = system == "pipe-only"
        rows = [
            execute_profile_cold_request(
                request, models[request.model_id], table, args, pipelined)
            for request in requests
        ]
    elif system == "reuse-only":
        caches = [make_backend(models, args) for _ in range(args.gpus)]
        next_gpu = 0
        for request_index, request in enumerate(requests):
            available = args.gpu_availability[request_index][
                "available_gpus"]
            candidates = []
            for gpu in available:
                cache = caches[gpu]
                probe = cache.clone()
                result = execute_profile_request(
                    probe, request, models[request.model_id], table, args,
                    pipelined=False)
                result["routing_cached_model_bytes"] = (
                    cache.resident_model_bytes(request.model_id))
                candidates.append((gpu, result))
            if args.execution_mode == "online":
                chosen, _, next_gpu = choose_online_cache_bytes_candidate(
                    candidates, args.online_available_at_ms,
                    args.online_arrivals_ms[request_index],
                    next_gpu, args.gpus)
            else:
                chosen, _, next_gpu = choose_routing_candidate(
                    candidates, "cache-bytes", next_gpu, args.gpus)
            row = execute_profile_request(
                caches[chosen], request, models[request.model_id], table, args,
                pipelined=False)
            row["gpu"] = chosen
            update_online_gpu_availability(
                args, request, chosen, row["critical_path_ms"])
            rows.append(row)
            routing.append(chosen)
            if request_index == args.inspect_request_id:
                routing_candidates[str(request_index)] = [
                    {"gpu": gpu, **result} for gpu, result in candidates
                ]
    elif system == "aegaeon":
        previous_by_gpu = {}
        next_gpu = 0
        for request_index, request in enumerate(requests):
            available = args.gpu_availability[request_index][
                "available_gpus"]
            candidates = [
                (
                    gpu,
                    _aegaeon_profile_row(
                        request, gpu, previous_by_gpu.get(gpu),
                        models[request.model_id], models, table, args),
                )
                for gpu in available
            ]
            if args.execution_mode == "online":
                chosen, row, next_gpu = choose_online_routing_candidate(
                    candidates, args.online_available_at_ms,
                    args.online_arrivals_ms[request_index],
                    next_gpu, args.gpus)
            else:
                best = min(
                    result["critical_path_ms"] for _, result in candidates)
                tied = {
                    gpu: result for gpu, result in candidates
                    if math.isclose(
                        result["critical_path_ms"], best, abs_tol=1e-9)
                }
                chosen = next(
                    gpu for offset in range(args.gpus)
                    if (gpu := (next_gpu + offset) % args.gpus) in tied
                )
                next_gpu = (chosen + 1) % args.gpus
                row = tied[chosen]
            rows.append(row)
            routing.append(chosen)
            previous_by_gpu[chosen] = request
            update_online_gpu_availability(
                args, request, chosen, row["critical_path_ms"])
            if request_index == args.inspect_request_id:
                routing_candidates[str(request_index)] = [
                    result for _, result in candidates
                ]
    elif system == "tangram":
        result = simulate(args, models, None, requests)
        result["system"] = system
        result["policies"][system] = result["policies"].pop("minimal")
        result["model"]["gpu_busy_probability"] = (
            args.gpu_busy_probability)
        result["model"]["availability_seed"] = args.availability_seed
        return result, result["policies"][system]["requests"]
    else:
        raise ValueError(f"unsupported system: {system}")
    result = {
        "format": "layerweave-tensor-simulator-v1",
        "system": system,
        "model": {
            "memory_layout": args.memory_layout,
            "pool_bytes": args.pool_bytes,
            "gpus": args.gpus,
            "gpu_busy_probability": args.gpu_busy_probability,
            "availability_seed": args.availability_seed,
            "model_mapping_order": args.model_mapping_order,
            "execution": "system-wide serial request order",
            "prefix_stall_table": table.source,
            "latency_profile_source": (
                "offline prefix-stall table"
                if system == "pipe-only"
                else (
                    "serialized missing-weight transfer/accounting"
                    if system in ("baseline", "reuse-only")
                    else "whole-model transfer/decode-overlap model"
                )
            ),
            "loading_overlap": (
                "pipelined" if system == "pipe-only"
                else (
                    "decode-prefetch" if system == "aegaeon"
                    else "serialized"
                )
            ),
            "replacement_policy": args.replacement_policy,
            "routing_policy": (
                "none" if system in ("baseline", "pipe-only")
                else (
                    "aegaeon-greedy-critical-path"
                    if system == "aegaeon" else args.routing_policy
                )
            ),
        },
        "routing": routing,
        "gpu_availability": args.gpu_availability,
        "routing_candidates": routing_candidates,
        "policies": {
            system: {
                "summary": summarize(rows),
                "requests": rows,
            },
        },
    }
    return result, rows


def simulate(args, models, estimator, requests):
    enabled_policies = (
        ("minimal",)
        if args.policy_suite == "minimal-only"
        else ("baseline", "pipe_only", "reuse_only", "minimal", "aegaeon")
    )
    profile_driven = args.prefix_stall_table is not None
    policy_models = {
        policy: (
            copy.deepcopy(models)
            if profile_driven else models_with_group_size(
                models, args.tensor_group_bytes_by_policy[policy])
        )
        for policy in enabled_policies
    }
    policy_benefits = (
        {policy: {} for policy in policy_models}
        if profile_driven else {
            policy: build_pipeline_benefits(
                policy_models[policy], estimator, requests, args)
            for policy in policy_models
        }
    )
    execute = (
        (lambda backend, request, model, _estimator, _args, _pipelined,
                collect_timeline=False: execute_profile_request(
                    backend, request, model, args.prefix_stall_table, args,
                    collect_timeline))
        if profile_driven else execute_request
    )
    baseline_rows = []
    pipe_rows = []
    if args.policy_suite == "all":
        for request_index, request in enumerate(requests):
            collect_timeline = request_index == args.inspect_request_id
            baseline_model = policy_models["baseline"][request.model_id]
            pipe_model = policy_models["pipe_only"][request.model_id]
            baseline_rows.append(execute(
                make_backend(
                    policy_models["baseline"], args,
                    policy_benefits["baseline"]),
                request, baseline_model, estimator, args, False,
                collect_timeline))
            pipe_rows.append(execute(
                make_backend(
                    policy_models["pipe_only"], args,
                    policy_benefits["pipe_only"]),
                request, pipe_model, estimator, args, True,
                collect_timeline))

    caches = [
        make_backend(
            policy_models["minimal"], args, policy_benefits["minimal"])
        for _ in range(args.gpus)
    ]
    reuse_caches = (
        [
            make_backend(
                policy_models["reuse_only"], args,
                policy_benefits["reuse_only"])
            for _ in range(args.gpus)
        ]
        if args.policy_suite == "all" else []
    )
    minimal_rows = []
    reuse_rows = []
    routing = []
    round_robin_next = 0
    demand = {model_id: 1.0 for model_id in models}
    input_history = {model_id: [] for model_id in models}
    routing_candidates = {}
    next_use_requests = [None] * len(requests)
    next_by_model = {}
    for request_index in range(len(requests) - 1, -1, -1):
        next_use_requests[request_index] = dict(next_by_model)
        request = requests[request_index]
        next_by_model[request.model_id] = request
    for request_index, request in enumerate(requests):
        if (
            args.future_probability_policy == "oracle-next"
            and request_index + 1 < len(requests)
        ):
            next_model = requests[request_index + 1].model_id
            probabilities = {
                model_id: float(model_id == next_model)
                for model_id in models
            }
        else:
            demand_total = sum(demand.values())
            probabilities = {
                model_id: value / demand_total
                for model_id, value in demand.items()
            }
        if args.replacement_policy == "mckp-prefix":
            if args.mckp_prediction_mode == "lookahead":
                prediction_samples = {model_id: [] for model_id in models}
                for offset, future in enumerate(
                        requests[
                            request_index + 1:
                            request_index + 1 + args.mckp_lookahead_k]):
                    prediction_samples[future.model_id].append((
                        future.input_tokens,
                        args.mckp_lookahead_discount ** offset,
                    ))
            else:
                prediction_samples = {}
                for model_id in models:
                    values = input_history[model_id]
                    tokens = (
                        int(statistics.median(values))
                        if values else
                        args.prefix_stall_table.input_median(model_id)
                    )
                    prediction_samples[model_id] = [
                        (tokens, probabilities[model_id])]
            for cache in caches:
                cache.set_mckp_prediction_samples(prediction_samples)
        for cache in caches:
            cache.set_future_model_probabilities(probabilities)
            if args.pipeline_stall_profile == "oracle-next-use":
                cache.set_pipeline_benefits(build_pipeline_benefits(
                    policy_models["minimal"], estimator, requests, args,
                    next_use_requests[request_index]))
            elif args.pipeline_stall_profile == "rolling-median":
                cache.set_pipeline_benefits(build_pipeline_benefits(
                    policy_models["minimal"], estimator, requests, args,
                    args.rolling_stall_representatives[request_index]))
        for cache in reuse_caches:
            cache.set_future_model_probabilities(probabilities)
            if args.pipeline_stall_profile == "oracle-next-use":
                cache.set_pipeline_benefits(build_pipeline_benefits(
                    policy_models["reuse_only"], estimator, requests, args,
                    next_use_requests[request_index]))
            elif args.pipeline_stall_profile == "rolling-median":
                cache.set_pipeline_benefits(build_pipeline_benefits(
                    policy_models["reuse_only"], estimator, requests, args,
                    args.rolling_stall_representatives[request_index]))
        if args.routing_replay_values is not None:
            chosen = args.routing_replay_values[request_index]
        elif args.gpus == 1:
            chosen = 0
        else:
            candidates = []
            available = args.gpu_availability[request_index][
                "available_gpus"]
            for gpu in available:
                cache = caches[gpu]
                probe = cache.clone()
                result = execute(
                    probe, request,
                    policy_models["minimal"][request.model_id],
                    estimator, args, True)
                if profile_driven:
                    result["routing_cached_model_bytes"] = (
                        cache.resident_model_bytes(request.model_id))
                    result["mckp_transition_weight"] = (
                        args.mckp_transition_weight)
                    result["routing_score_ms"] = (
                        result["exposed_load_ms"]
                        + args.mckp_transition_weight
                        * result.get("mckp_predicted_damage_ms", 0.0))
                candidates.append((gpu, result))
            if args.execution_mode == "online":
                chosen, _, round_robin_next = (
                    choose_online_routing_candidate(
                        candidates, args.online_available_at_ms,
                        args.online_arrivals_ms[request_index],
                        round_robin_next, args.gpus,
                        args.mckp_transition_weight)
                )
            else:
                chosen, _, round_robin_next = choose_routing_candidate(
                    candidates, args.routing_policy, round_robin_next,
                    args.gpus)
            if request_index == args.inspect_request_id:
                routing_candidates[str(request_index)] = [
                    {"gpu": gpu, **result} for gpu, result in candidates
                ]
        minimal = execute(
            caches[chosen], request,
            policy_models["minimal"][request.model_id],
            estimator, args, True,
            request_index == args.inspect_request_id)
        minimal["gpu"] = chosen
        minimal_rows.append(minimal)
        update_online_gpu_availability(
            args, request, chosen, minimal["critical_path_ms"])
        if args.policy_suite == "all":
            reuse = execute_request(
                reuse_caches[chosen], request,
                policy_models["reuse_only"][request.model_id],
                estimator, args, False,
                request_index == args.inspect_request_id)
            reuse["gpu"] = chosen
            reuse_rows.append(reuse)
        routing.append(chosen)
        for model_id in demand:
            demand[model_id] *= args.demand_decay
        demand[request.model_id] += 1.0
        history = input_history[request.model_id]
        history.append(request.input_tokens)
        if len(history) > args.mckp_history_window:
            del history[0]

    aegaeon_rows = []
    per_gpu_previous = {}
    for request, gpu in (
        zip(requests, routing) if args.policy_suite == "all" else ()
    ):
        model = policy_models["aegaeon"][request.model_id]
        hot = sum(compute_durations(estimator, request, model).values())
        footprint = make_backend(
            policy_models["aegaeon"], args).physical_model_bytes(
            request.model_id)
        h2d = model.logical_bytes / args.h2d_bytes_per_ms
        previous = per_gpu_previous.get(gpu)
        same = previous is not None and previous.model_id == request.model_id
        window = (
            previous.output_tokens * args.decode_ms_per_token
            if previous is not None and not same else 0.0
        )
        previous_footprint = (
            make_backend(
                policy_models["aegaeon"], args
            ).physical_model_bytes(previous.model_id)
            if previous is not None else 0
        )
        previous_kv = kv_bytes(previous, args) if previous is not None else 0
        feasible = (
            previous is not None and
            previous_footprint + previous_kv + footprint <= args.pool_bytes
        )
        hidden = min(h2d, window) if feasible and not same else 0.0
        relocation = (
            footprint / args.gpu_copy_bytes_per_ms
            if feasible and not same else 0.0
        )
        pages = ceil_div(footprint, args.page_size_bytes)
        if args.memory_layout in ("segment", "legacy-segment"):
            allocation_ms = (
                args.segment_allocation_ms
                if not same else 0.0
            )
            mapping_ms = 0.0
            map_calls = 0
            mapped_pages = 0
        elif args.memory_layout == "tensor-page":
            allocation_ms = 0.0
            map_calls = len(model.groups) if not same else 0
            mapped_pages = pages if not same else 0
            mapping_ms = (
                map_calls * args.map_fixed_ms
                + mapped_pages * args.map_per_page_ms
            )
        else:
            allocation_ms = 0.0
            map_calls = 1 if not same else 0
            mapped_pages = pages if not same else 0
            mapping_ms = (
                map_calls * args.map_fixed_ms
                + mapped_pages * args.map_per_page_ms
            )
        loading = (
            0.0 if same
            else h2d - hidden + relocation + allocation_ms + mapping_ms
        )
        aegaeon_rows.append({
            "request_id": request.request_id,
            "model_id": request.model_id,
            "gpu": gpu,
            "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens,
            "critical_path_ms": hot + loading,
            "hot_compute_ms": hot,
            "exposed_load_ms": loading,
            "h2d_ms": 0.0 if same else h2d,
            "h2d_bytes": 0 if same else model.logical_bytes,
            "allocation_ms": allocation_ms,
            "map_ms": mapping_ms,
            "unmap_ms": 0.0,
            "compaction_ms": 0.0,
            "compaction_moved_bytes": 0,
            "evicted_objects": 0,
            "evicted_bytes": 0,
            "map_calls": map_calls,
            "unmap_calls": 0,
            "mapped_pages": mapped_pages,
            "unmapped_pages": 0,
            "memory": {
                "logical_resident_bytes": model.logical_bytes,
                "physical_resident_bytes": footprint,
                "internal_fragmentation_bytes":
                    max(0, footprint - model.logical_bytes),
                "external_fragmentation_ratio": 0.0,
                "largest_free_extent_bytes":
                    max(0, args.pool_bytes - footprint),
                "stranded_resident_bytes": 0,
            },
            "decode_overlap_window_ms": window,
            "prefetched_h2d_ms": hidden,
            "gpu_relocation_ms": relocation,
            "capacity_feasible": feasible,
            "same_model_hit": same,
        })
        per_gpu_previous[gpu] = request

    policies = (
        {
            "baseline": baseline_rows,
            "pipe_only": pipe_rows,
            "reuse_only": reuse_rows,
            "minimal": minimal_rows,
            "aegaeon": aegaeon_rows,
        }
        if args.policy_suite == "all"
        else {"minimal": minimal_rows}
    )
    inspect = args.inspect_request_id
    return {
        "format": "layerweave-tensor-simulator-v1",
        "model": {
            "memory_layout": args.memory_layout,
            "pool_bytes": args.pool_bytes,
            "page_size_bytes": args.page_size_bytes,
            "gpus": args.gpus,
            "gpu_busy_probability": args.gpu_busy_probability,
            "availability_seed": args.availability_seed,
            "model_mapping_order": args.model_mapping_order,
            "tensor_compute_profile_source":
                (
                    "offline prefix-stall table median_baseline_wall_ms"
                    if profile_driven else
                    getattr(
                        estimator, "tensor_compute_source",
                        "M4 layer profile distributed by tensor logical bytes")
                ),
            "latency_profile_source": (
                "offline prefix-stall table"
                if profile_driven else "layer compute estimator"),
            "loading_overlap": (
                "pipelined"
                if profile_driven or args.policy_suite == "minimal-only"
                else "policy-dependent"),
            "execution": "system-wide serial request order",
            "policy_suite": args.policy_suite,
            "single_gpu_routing_probe":
                "skipped" if args.gpus == 1 else "not-applicable",
            "segment_compaction_policy": args.compaction_policy,
            "segment_allocator": (
                "mock-allocation request-level partitioned bin packing"
                if args.memory_layout == "segment"
                else (
                    "legacy per-group first-fit/global compaction"
                    if args.memory_layout == "legacy-segment"
                    else "not-applicable"
                )
            ),
            "kv_memory_model": (
                "explicit blocks in unified Segment extent pool"
                if args.memory_layout == "segment"
                else "capacity reservation"
            ),
            "segment_retention_policy":
                (
                    "prefix MCKP minimum predicted future stall damage"
                    if args.replacement_policy == "mckp-prefix"
                    else (
                    "model LRU with suffix release preserving prefix state"
                    if args.replacement_policy == "lru-prefix"
                    else
                    "storage-unit least recently used"
                    if args.replacement_policy == "lru"
                    else
                    "future-model probability times marginal eliminated "
                    "exposed loading stall, then storage-unit LRU"
                    if args.replacement_policy == "pipeline-value"
                    else (
                        "robust pipeline protected floor, soft cache, "
                        "frequency and LRU ordering"
                        if args.replacement_policy.startswith(
                            "protected-pipeline")
                        else (
                            "model-frequency value, suffix-first compute "
                            "position, then storage-unit LRU"
                            if args.replacement_policy
                            == "frequency-lru-suffix-first"
                            else (
                                "model-frequency value then storage-unit LRU"
                            )
                        )
                    )
                    )
                ),
            "replacement_policy": args.replacement_policy,
            "routing_policy": args.routing_policy,
            "routing_source": (
                str(args.routing_replay)
                if args.routing_replay_values is not None
                else "online-policy"
            ),
            "routing_tie_break": "round-robin",
            "future_model_probability":
                (
                    "oracle next request model"
                    if args.future_probability_policy == "oracle-next"
                    else (
                        "global online exponentially decayed demand, "
                        f"decay={args.demand_decay}"
                    )
                ),
            "pipeline_stall_profile": args.pipeline_stall_profile,
            "pipeline_robust_stat": args.pipeline_robust_stat,
            "pipeline_protected_threshold_ms":
                args.pipeline_protected_threshold_ms,
            "shadow_max_churn_ratio": args.shadow_max_churn_ratio,
            "tensor_group_min_bytes_by_policy":
                args.tensor_group_bytes_by_policy,
            "tensor_group_count_by_policy": {
                policy: sum(
                    len(model.groups) for model in grouped.values())
                for policy, grouped in policy_models.items()
            },
            "prefix_stall_table": (
                args.prefix_stall_table.source
                if args.prefix_stall_table is not None else None),
            "prefix_stall_fingerprint": (
                args.prefix_stall_table.metadata.get("fingerprint")
                if args.prefix_stall_table is not None else None),
            "mckp_prediction_mode": args.mckp_prediction_mode,
            "mckp_lookahead_k": args.mckp_lookahead_k,
            "mckp_lookahead_discount": args.mckp_lookahead_discount,
            "mckp_transition_weight": args.mckp_transition_weight,
            "mckp_history_window": args.mckp_history_window,
            "input_limit_policy": args.input_limit_policy,
            "stall_table_input_policy": args.stall_table_input_policy,
        },
        "routing": routing,
        "gpu_availability": args.gpu_availability,
        "routing_candidates": routing_candidates,
        "policies": {
            name: {
                "summary": summarize(rows),
                "requests": [
                    {
                        key: value for key, value in row.items()
                        if key != "timeline"
                    }
                    for row in rows
                ],
            }
            for name, rows in policies.items()
        },
        "inspected_request": (
            {
                name: rows[inspect]
                for name, rows in policies.items()
                if inspect < len(rows)
            }
            if inspect is not None else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-layout", type=Path, required=True)
    parser.add_argument(
        "--profile", type=Path,
        help=(
            "Layer-compute estimator for runs without an offline prefix-stall "
            "table. Profile-driven End2End runs use the offline table's "
            "median_baseline_wall_ms and do not require this option."))
    parser.add_argument("--tensor-compute-profile", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--memory-layout", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--pool-gib", type=float, default=42.0)
    parser.add_argument("--page-size-mib", type=float, default=64.0)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument(
        "--trace-mode", choices=("all", "model-switches"),
        default="model-switches")
    parser.add_argument("--input-scale", type=float, default=4.0)
    parser.add_argument(
        "--input-limit-policy",
        choices=("safe", "context", "none"), default="safe",
        help=(
            "Use real-GPU safe caps, model context caps, or only simulated "
            "pool/KV feasibility caps for effective input tokens."))
    parser.add_argument(
        "--stall-table-input-policy",
        choices=("clamp", "linear"), default="clamp",
        help=(
            "Clamp above the largest Offline stall row or linearly "
            "extrapolate from its last two input rows."))
    parser.add_argument("--output-tokens-override", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument(
        "--gpu-busy-probability", type=float, default=0.0,
        help=(
            "Independent probability in [0, 1] that each GPU is busy for "
            "each request. If all GPUs are sampled busy, one is randomly "
            "made available. The mask is policy-independent and controlled "
            "by --availability-seed."))
    parser.add_argument("--availability-seed", type=int, default=1234)
    parser.add_argument(
        "--available-gpus-per-request", type=int,
        help=(
            "Sample exactly this many policy-independent available GPUs "
            "per request. This overrides Bernoulli --gpu-busy-probability "
            "and is useful for matched GPU-count sensitivity sweeps."))
    parser.add_argument(
        "--execution-mode", choices=("serial", "online"),
        default="serial")
    parser.add_argument(
        "--arrival-mode", choices=("trace-scaled", "poisson"),
        default="trace-scaled")
    parser.add_argument("--request-rate-rps", type=float)
    parser.add_argument("--arrival-seed", type=int, default=1234)
    parser.add_argument(
        "--slo-scale", type=float, default=2.0,
        help="Per-request SLO is this multiple of fully-resident Prefill.")
    parser.add_argument(
        "--warmup-requests", type=int, default=0,
        help=(
            "Execute these initial online requests normally but exclude "
            "them from latency and SLO summaries."))
    parser.add_argument(
        "--system",
        choices=(
            "suite", "baseline", "pipe-only", "reuse-only",
            "aegaeon", "tangram"),
        default="suite",
        help=(
            "Run the legacy suite or one independently defined End2End "
            "system. Single-system runs require the Offline prefix table."))
    parser.add_argument(
        "--policy-suite", choices=("all", "minimal-only"), default="all",
        help=(
            "Run all five top-level policies, or only minimal for cache-policy "
            "sweeps."))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--tensor-group-min-mib", type=float, default=0.0,
        help=(
            "Default minimum TensorGroup size. Zero keeps one allocation "
            "unit per tensor."))
    parser.add_argument(
        "--policy-tensor-group-mib", action="append", default=[],
        metavar="POLICY=MiB",
        help=(
            "Override TensorGroup minimum size for baseline, pipe_only, "
            "reuse_only, minimal, or aegaeon. Repeat as needed."))
    parser.add_argument(
        "--replacement-policy",
        choices=(
            "lru", "lru-prefix", "mckp-prefix",
            "frequency-lru", "frequency-lru-suffix-first",
            "pipeline-value", "protected-pipeline",
            "protected-pipeline-shadow"),
        default="frequency-lru")
    parser.add_argument(
        "--pipeline-robust-stat",
        choices=("mean", "p90", "max"), default="mean")
    parser.add_argument(
        "--pipeline-protected-threshold-ms", type=float, default=5.0)
    parser.add_argument(
        "--shadow-max-churn-ratio", type=float, default=1.25)
    parser.add_argument(
        "--routing-policy",
        choices=(
            "critical-path", "exposed-stall", "cache-bytes",
            "mckp-transition"),
        default="critical-path")
    parser.add_argument("--mckp-stall-table-input", type=Path)
    parser.add_argument(
        "--mckp-prediction-mode",
        choices=("history", "lookahead"), default="history")
    parser.add_argument("--mckp-lookahead-k", type=int, default=8)
    parser.add_argument(
        "--mckp-lookahead-discount", type=float, default=1.0)
    parser.add_argument(
        "--mckp-transition-weight", type=float, default=1.0)
    parser.add_argument("--mckp-history-window", type=int, default=32)
    parser.add_argument(
        "--routing-replay", type=Path,
        help=(
            "Replay the top-level routing array from a previous simulator "
            "JSON, bypassing online GPU selection."))
    parser.add_argument("--demand-decay", type=float, default=0.9)
    parser.add_argument(
        "--future-probability-policy",
        choices=("decayed-demand", "oracle-next"),
        default="decayed-demand",
        help="Diagnostic oracle predicts the model of request i+1.")
    parser.add_argument(
        "--pipeline-stall-profile",
        choices=("static-median", "oracle-next-use", "rolling-median"),
        default="static-median",
        help=(
            "Diagnostic oracle recomputes each model's stall values from its "
            "next actual input shape after the current request; rolling mode "
            "uses the median of each model's previous K source requests."))
    parser.add_argument("--pipeline-stall-window", type=int, default=5)
    parser.add_argument("--kv-block-tokens", type=int, default=32)
    parser.add_argument("--kv-block-bytes", type=int, default=2 * MIB)
    parser.add_argument(
        "--compaction-policy",
        choices=("never", "on-allocation-failure", "cost-aware"),
        default="on-allocation-failure")
    parser.add_argument("--h2d-gbps", type=float, default=20.0)
    parser.add_argument(
        "--gpu-copy-gbps", type=float, default=432.0,
        help=("Effective GPU-local relocation payload bandwidth. The L40 "
              "has 864 GB/s peak memory bandwidth, while a copy consumes "
              "one read and one write, giving a 432 GB/s payload ceiling."))
    parser.add_argument("--segment-allocation-ms", type=float, default=0.005)
    parser.add_argument("--map-fixed-ms", type=float, default=0.010)
    parser.add_argument("--map-per-page-ms", type=float, default=0.002)
    parser.add_argument("--unmap-fixed-ms", type=float, default=0.010)
    parser.add_argument("--unmap-per-page-ms", type=float, default=0.002)
    parser.add_argument("--compaction-fixed-ms", type=float, default=0.020)
    parser.add_argument("--decode-ms-per-token", type=float, default=40.0)
    parser.add_argument("--inspect-request-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--requests-csv-output", type=Path)
    args = parser.parse_args()
    if (
        args.pool_gib <= 0 or args.page_size_mib <= 0 or args.gpus <= 0
        or not 0.0 <= args.gpu_busy_probability <= 1.0
        or args.tensor_group_min_mib < 0
        or not 0.0 <= args.demand_decay <= 1.0
        or args.pipeline_protected_threshold_ms < 0
        or args.shadow_max_churn_ratio <= 0
        or args.pipeline_stall_window <= 0
        or args.mckp_lookahead_k < 0
        or not 0 < args.mckp_lookahead_discount <= 1
        or args.mckp_transition_weight < 0
        or args.mckp_history_window <= 0
        or args.slo_scale <= 0
        or args.warmup_requests < 0
        or (
            args.available_gpus_per_request is not None
            and not 1 <= args.available_gpus_per_request <= args.gpus)
    ):
        parser.error(
            "pool, page size, and GPUs must be positive; GPU busy "
            "probability must be in [0, 1]; sizes and demand decay must "
            "be valid")
    if args.execution_mode == "online":
        if args.request_rate_rps is None or args.request_rate_rps <= 0:
            parser.error(
                "online execution requires a positive --request-rate-rps")
        if args.gpu_busy_probability != 0:
            parser.error(
                "online execution derives GPU occupancy from requests; "
                "set --gpu-busy-probability 0")
        if (
            args.max_requests > 0
            and args.warmup_requests >= args.max_requests
        ):
            parser.error(
                "--warmup-requests must be smaller than --max-requests")
    args.pool_bytes = int(args.pool_gib * 1024**3)
    args.page_size_bytes = int(args.page_size_mib * MIB)
    args.h2d_bytes_per_ms = args.h2d_gbps * 1_000_000.0
    args.gpu_copy_bytes_per_ms = args.gpu_copy_gbps * 1_000_000.0
    args.prefix_stall_table = (
        PrefixStallTable.load(args.mckp_stall_table_input)
        if args.mckp_stall_table_input is not None else None)
    if args.profile is None and args.prefix_stall_table is None:
        parser.error(
            "--profile is required when --mckp-stall-table-input is absent")
    if args.tensor_compute_profile is not None and args.profile is None:
        parser.error("--tensor-compute-profile requires --profile")
    if args.prefix_stall_table is not None:
        args.prefix_stall_table.input_extrapolation = (
            args.stall_table_input_policy)
    if args.system != "suite" and args.prefix_stall_table is None:
        parser.error(
            "single-system End2End runs require "
            "--mckp-stall-table-input")
    if args.system != "suite" and args.memory_layout != "segment":
        parser.error("single-system End2End runs currently require segment")
    if args.system == "reuse-only" and args.replacement_policy != "lru":
        parser.error("reuse-only requires --replacement-policy lru")
    if (
        args.system == "tangram"
        and (
            args.replacement_policy != "mckp-prefix"
            or args.routing_policy != "mckp-transition"
        )
    ):
        parser.error(
            "tangram requires mckp-prefix replacement and "
            "mckp-transition routing")
    if args.system == "tangram" and args.policy_suite != "minimal-only":
        parser.error("tangram requires --policy-suite minimal-only")
    if args.replacement_policy in ("mckp-prefix", "lru-prefix"):
        if args.prefix_stall_table is None:
            parser.error(
                "prefix replacement requires --mckp-stall-table-input")
        if (
            args.replacement_policy == "mckp-prefix"
            and args.memory_layout not in (
                "segment", "tensor-page", "compact-page")
        ):
            parser.error(
                "prefix replacement requires segment, tensor-page, "
                "or compact-page")
        if (
            args.replacement_policy == "lru-prefix"
            and args.memory_layout != "segment"
        ):
            parser.error("lru-prefix currently requires segment")
        if args.policy_suite != "minimal-only":
            parser.error(
                "profile-driven prefix experiments require "
                "--policy-suite minimal-only")
    if (
        args.routing_policy in ("cache-bytes", "mckp-transition")
        and args.prefix_stall_table is None
    ):
        parser.error(
            f"{args.routing_policy} requires --mckp-stall-table-input")
    if (
        args.routing_policy == "mckp-transition"
        and args.replacement_policy != "mckp-prefix"
    ):
        parser.error("mckp-transition requires mckp-prefix replacement")
    policies = (
        "baseline", "pipe_only", "reuse_only", "minimal", "aegaeon")
    policy_mib = {
        policy: args.tensor_group_min_mib for policy in policies
    }
    for override in args.policy_tensor_group_mib:
        try:
            policy, value = override.split("=", 1)
            size_mib = float(value)
        except ValueError:
            parser.error(
                f"invalid --policy-tensor-group-mib {override!r}; "
                "expected POLICY=MiB")
        if policy not in policy_mib or size_mib < 0:
            parser.error(
                f"invalid --policy-tensor-group-mib {override!r}")
        policy_mib[policy] = size_mib
    args.tensor_group_bytes_by_policy = {
        policy: int(size_mib * MIB)
        for policy, size_mib in policy_mib.items()
    }

    limits = load_model_limits(args.config, args.input_limit_policy)
    models = (
        models_from_prefix_stall_table(args.prefix_stall_table, limits)
        if args.prefix_stall_table is not None
        else load_tensor_models(args.tensor_layout, limits)
    )
    estimator = (
        None if args.profile is None else load_profile(args.profile))
    if args.tensor_compute_profile:
        tensor_compute_document = json.loads(
            args.tensor_compute_profile.read_text())
        estimator.tensor_compute = tensor_compute_document["models"]
        estimator.tensor_compute_source = (
            tensor_compute_document["source"]
            + "; unprofiled tensors fall back to M4 layer-byte split"
        )
    proxy_layouts = {
        model_id: type("Layout", (), {
            "page_count": ceil_div(model.logical_bytes, 64 * MIB),
            "max_input_tokens": model.max_input_tokens,
        })()
        for model_id, model in models.items()
    }
    model_mapping = load_model_mapping(args.config, models)
    args.model_mapping_order = [
        model_mapping[model_id] for model_id in sorted(model_mapping)
    ]
    trace_proxy_layouts = {
        trace_model_id: proxy_layouts[physical_model_id]
        for trace_model_id, physical_model_id in model_mapping.items()
    }
    requests = remap_requests(load_page_trace(
        args.trace, trace_proxy_layouts, args.max_requests, args.input_scale,
        args.output_tokens_override,
        max(1, args.pool_bytes // (64 * MIB)),
        args.kv_block_tokens, args.trace_mode), model_mapping)
    if (
        args.execution_mode == "online"
        and args.warmup_requests >= len(requests)
    ):
        parser.error(
            "--warmup-requests must be smaller than the number of "
            "filtered requests")
    args.gpu_availability = build_gpu_availability(
        requests, args.gpus, args.gpu_busy_probability,
        args.availability_seed, args.available_gpus_per_request)
    arrivals_s = None
    if args.execution_mode == "online":
        arrivals_s = load_trace_arrivals(
            args.trace, args.max_requests, args.trace_mode, len(requests),
            args.arrival_mode, args.request_rate_rps, args.arrival_seed)
        args.online_arrivals_ms = [
            arrival_s * 1000.0 for arrival_s in arrivals_s]
        args.online_available_at_ms = [0.0] * args.gpus
    args.rolling_stall_representatives = None
    if args.pipeline_stall_profile == "rolling-median":
        source_requests = remap_requests(load_page_trace(
            args.trace, trace_proxy_layouts, args.max_requests, args.input_scale,
            args.output_tokens_override,
            max(1, args.pool_bytes // (64 * MIB)),
            args.kv_block_tokens, "all"), model_mapping)
        histories = {model_id: [] for model_id in models}
        representatives = []
        previous_model_id = None
        for source_request in source_requests:
            if source_request.model_id != previous_model_id:
                representatives.append({
                    model_id: Request(
                        0, model_id,
                        int(statistics.median(values)), 1, 1)
                    for model_id, values in histories.items()
                    if len(values) >= args.pipeline_stall_window
                })
            history = histories[source_request.model_id]
            history.append(source_request.input_tokens)
            if len(history) > args.pipeline_stall_window:
                del history[0]
            previous_model_id = source_request.model_id
        if len(representatives) != len(requests):
            parser.error(
                "rolling stall history did not align with simulated requests")
        args.rolling_stall_representatives = representatives
    args.routing_replay_values = None
    if args.routing_replay is not None:
        replay_document = json.loads(args.routing_replay.read_text())
        replay = replay_document.get("routing", replay_document)
        if not isinstance(replay, list) or len(replay) != len(requests):
            parser.error(
                "--routing-replay must contain one GPU id per simulated "
                f"request; expected {len(requests)} entries")
        if any(
            not isinstance(gpu, int) or gpu < 0 or gpu >= args.gpus
            for gpu in replay
        ):
            parser.error("--routing-replay contains an invalid GPU id")
        if any(
            gpu not in args.gpu_availability[index]["available_gpus"]
            for index, gpu in enumerate(replay)
        ):
            parser.error(
                "--routing-replay selects a GPU marked busy by the "
                "availability mask")
        args.routing_replay_values = replay
    checked_group_sizes = set()
    for group_bytes in args.tensor_group_bytes_by_policy.values():
        if group_bytes in checked_group_sizes:
            continue
        checked_group_sizes.add(group_bytes)
        validate_request_feasibility(
            models_with_group_size(models, group_bytes),
            requests, args)
    started = time.perf_counter()
    if args.system == "suite":
        result = simulate(args, models, estimator, requests)
        csv_rows = None
    else:
        result, csv_rows = simulate_single_system(
            args, models, args.prefix_stall_table, requests)
    if args.execution_mode == "online":
        online_routing, online_summary = apply_online_timing(
            csv_rows, result.get("routing", []), requests, arrivals_s,
            args, args.system)
        result["routing"] = online_routing
        result["policies"][args.system]["summary"] = online_summary
        result["model"].update({
            "execution": "online arrival replay with per-GPU FIFO occupancy",
            "arrival_mode": args.arrival_mode,
            "request_rate_rps": args.request_rate_rps,
            "arrival_seed": args.arrival_seed,
            "decode_occupancy_source":
                "output_tokens * decode_ms_per_token",
            "slo_scale": args.slo_scale,
            "warmup_requests": args.warmup_requests,
        })
    result["model"]["simulator_wall_time_ms"] = (
        time.perf_counter() - started) * 1000.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    if args.csv_output is not None:
        if csv_rows is None:
            parser.error("--csv-output requires a single-system run")
        write_per_model_csv(args.csv_output, args.system, csv_rows)
    if args.requests_csv_output is not None:
        write_request_csv(
            args.requests_csv_output, args.system, csv_rows)
    print(json.dumps({
        "output": str(args.output),
        "csv_output": (
            None if args.csv_output is None else str(args.csv_output)),
        "requests_csv_output": (
            None if args.requests_csv_output is None
            else str(args.requests_csv_output)),
        "requests": len(requests),
        "layout": args.memory_layout,
        "system": args.system,
        "wall_time_ms": result["model"]["simulator_wall_time_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
