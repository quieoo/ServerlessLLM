"""Segment backends, including mock_allocation request-level PBP."""

from __future__ import annotations

import time

import copy
import math

from tensor_sim_memory_common import Extent, LoadCost, MemoryBackend
from tensor_sim_mckp import solve_covering_mckp

class SegmentMemory(MemoryBackend):
    def __init__(self, models, capacity_bytes, page_size, args):
        super().__init__(models, capacity_bytes, page_size, args)
        self.extents = [Extent(0, capacity_bytes)]
        # Mirror mock_allocation's allocated_regions fingerprint index:
        # address order remains in extents, while reuse lookup is O(1).
        self.allocated_extents = {}
        self.compaction_count = 0

    def clone(self):
        """Clone mutable allocator state without deep-copying model metadata."""
        result = type(self).__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result.extents = [
            Extent(extent.offset, extent.size, extent.key)
            for extent in self.extents
        ]
        result.allocated_extents = {
            extent.key: extent for extent in result.extents
            if extent.key is not None
        }
        result.model_accesses = dict(self.model_accesses)
        result.last_access = dict(self.last_access)
        placement = getattr(self, "_placement_rank_cache", None)
        result._placement_rank_cache = (
            None if placement is None else dict(placement))
        if hasattr(self, "planned_loads"):
            result.planned_loads = {
                key: copy.copy(value)
                for key, value in self.planned_loads.items()
            }
        if hasattr(self, "kv_keys"):
            result.kv_keys = set(self.kv_keys)
        return result

    def _allocated_extent(self, key):
        extent = self.allocated_extents.get(key)
        if extent is not None and extent.key == key:
            return extent
        # Tests and old snapshots may replace extents directly. Rebuild the
        # missing entry lazily without changing allocator behavior.
        extent = next((item for item in self.extents if item.key == key), None)
        if extent is not None:
            self.allocated_extents[key] = extent
        return extent

    def used_bytes(self):
        return sum(item.size for item in self.extents if item.key is not None)

    def _object_bytes(self, key):
        extent = self._allocated_extent(key)
        if extent is None:
            raise KeyError(key)
        return extent.size

    def _placement_rank(self, key):
        cached = getattr(self, "_placement_rank_cache", None)
        if cached is not None:
            return cached.get(key, 0)
        for index, extent in enumerate(self.extents):
            if extent.key != key:
                continue
            reclaim_span = extent.size
            if index > 0 and self.extents[index - 1].key is None:
                reclaim_span += self.extents[index - 1].size
            if (
                index + 1 < len(self.extents)
                and self.extents[index + 1].key is None
            ):
                reclaim_span += self.extents[index + 1].size
            return -reclaim_span
        return 0

    def _coalesce(self):
        result = []
        for extent in self.extents:
            if result and result[-1].key is None and extent.key is None:
                result[-1].size += extent.size
            else:
                result.append(extent)
        self.extents = result

    def _free(self, key, coalesce=True) -> int:
        extent = self.allocated_extents.pop(key, None)
        if extent is None or extent.key != key:
            extent = next(
                (item for item in self.extents if item.key == key), None)
        if extent is None:
            return 0
        released = extent.size
        extent.key = None
        if coalesce:
            self._coalesce()
        return released

    def _free_bytes(self):
        limit = self.weight_capacity_bytes
        return sum(
            max(0, min(e.offset + e.size, limit) - e.offset)
            for e in self.extents if e.key is None and e.offset < limit
        )

    def _largest_free(self):
        limit = self.weight_capacity_bytes
        return max([
            max(0, min(e.offset + e.size, limit) - e.offset)
            for e in self.extents if e.key is None and e.offset < limit
        ] or [0])

    def _victims(self, active_model):
        keys = [e.key for e in self.extents
                if e.key is not None and e.key[0] != active_model]
        remaining = set(keys)
        self._placement_rank_cache = {}
        for index, extent in enumerate(self.extents):
            if extent.key not in remaining:
                continue
            reclaim_span = extent.size
            if index > 0 and self.extents[index - 1].key is None:
                reclaim_span += self.extents[index - 1].size
            if (
                index + 1 < len(self.extents)
                and self.extents[index + 1].key is None
            ):
                reclaim_span += self.extents[index + 1].size
            self._placement_rank_cache[extent.key] = -reclaim_span
        selected = self._select_victim(remaining)
        self._placement_rank_cache = None
        if selected is None:
            return []
        return [selected]

    def _evict_one(self, active_model, cost):
        victims = self._victims(active_model)
        if not victims:
            return False
        key = victims[0]
        released = self._free(key)
        cost.evicted_objects += 1
        cost.evicted_bytes += released
        return True

    def _compact(self) -> LoadCost:
        cost = LoadCost()
        cursor = 0
        allocated = []
        moved = 0
        for extent in self.extents:
            if extent.key is None:
                continue
            if extent.offset != cursor:
                moved += extent.size
            allocated.append(Extent(cursor, extent.size, extent.key))
            cursor += extent.size
        if cursor < self.capacity_bytes:
            allocated.append(Extent(cursor, self.capacity_bytes - cursor))
        self.extents = allocated
        self.allocated_extents = {
            extent.key: extent
            for extent in allocated if extent.key is not None
        }
        self.compaction_count += 1
        cost.compaction_moved_bytes = moved
        cost.compaction_ms = (
            moved / self.args.gpu_copy_bytes_per_ms
            + self.args.compaction_fixed_ms
        )
        return cost

    def _eviction_alternative_ms(self, size, active_model):
        """Estimate reload damage of evicting until a contiguous fit exists."""
        probe = self.clone()
        cost = LoadCost()
        while probe._largest_free() < size:
            victims = probe._victims(active_model)
            if not victims:
                return math.inf
            key = victims[0]
            released = probe._free(key)
            cost.evicted_bytes += released
        return cost.evicted_bytes / self.args.h2d_bytes_per_ms

    def _trim_to_capacity(self, active_model):
        cost = LoadCost()
        while self.used_bytes() > self.weight_capacity_bytes:
            if not self._evict_one(active_model, cost):
                raise RuntimeError("No evictable segment for KV capacity")
        if any(
            e.key is not None and e.offset + e.size > self.weight_capacity_bytes
            for e in self.extents
        ):
            cost.add(self._compact())
        return cost

    def tensor_resident(self, model_id, tensor):
        group = self._group(model_id, tensor)
        return self._allocated_extent(
            (model_id, group.group_id)) is not None

    def prepare_tensor(self, model_id, tensor):
        group = self._group(model_id, tensor)
        key = (model_id, group.group_id)
        if self.tensor_resident(model_id, tensor):
            self._touch(key)
            return LoadCost()
        cost = LoadCost()
        size = group.logical_bytes
        while self._free_bytes() < size:
            if not self._evict_one(model_id, cost):
                raise RuntimeError(f"Segment pool cannot fit tensor {key}")
        if self._largest_free() < size:
            if self.args.compaction_policy == "never":
                while self._largest_free() < size:
                    if not self._evict_one(model_id, cost):
                        raise RuntimeError("Fragmented segment allocation")
            elif self.args.compaction_policy == "cost-aware":
                allocated = [
                    extent for extent in self.extents
                    if extent.key is not None
                ]
                cursor = 0
                moved = 0
                for extent in allocated:
                    if extent.offset != cursor:
                        moved += extent.size
                    cursor += extent.size
                compact_ms = (
                    moved / self.args.gpu_copy_bytes_per_ms
                    + self.args.compaction_fixed_ms
                )
                eviction_ms = self._eviction_alternative_ms(
                    size, model_id)
                if compact_ms <= eviction_ms:
                    cost.add(self._compact())
                else:
                    while self._largest_free() < size:
                        if not self._evict_one(model_id, cost):
                            raise RuntimeError(
                                "Cost-aware segment allocation failed")
            else:
                cost.add(self._compact())
        for index, extent in enumerate(self.extents):
            available = (
                extent.key is None and extent.offset < self.weight_capacity_bytes
                and min(extent.size,
                        self.weight_capacity_bytes - extent.offset) >= size
            )
            if not available:
                continue
            replacement = [Extent(extent.offset, size, key)]
            if extent.size > size:
                replacement.append(
                    Extent(extent.offset + size, extent.size - size))
            self.extents[index:index + 1] = replacement
            self.allocated_extents[key] = replacement[0]
            break
        else:
            raise RuntimeError("No contiguous segment after reclamation")
        self._touch(key)
        cost.h2d_bytes = size
        cost.h2d_ms = size / self.args.h2d_bytes_per_ms
        cost.allocation_ms = self.args.segment_allocation_ms
        return cost

    def physical_model_bytes(self, model_id):
        return self.models[model_id].logical_bytes

    def metrics(self):
        logical = self.used_bytes()
        free = self._free_bytes()
        largest = self._largest_free()
        return {
            "logical_resident_bytes": logical,
            "physical_resident_bytes": logical,
            "internal_fragmentation_bytes": 0,
            "external_fragmentation_ratio":
                1.0 - largest / free if free else 0.0,
            "largest_free_extent_bytes": largest,
            "stranded_resident_bytes": 0,
            "compaction_count": self.compaction_count,
        }


class MockAllocationSegmentMemory(SegmentMemory):
    """Request-level Segment planner following mock_allocation's PBP shape."""

    def __init__(self, models, capacity_bytes, page_size, args):
        super().__init__(models, capacity_bytes, page_size, args)
        self.planned_loads = {}
        self.kv_keys = set()
        self.next_kv_extent_id = 0
        self.pending_kv_bytes = 0

    def begin_request(self, kv_bytes, active_model):
        # mock_allocation uses one unified region pool. Old request KV blocks
        # are released in-place; weights retain the full address range.
        for key in tuple(self.kv_keys):
            self._free(key)
        self.kv_keys.clear()
        self.pending_kv_bytes = kv_bytes
        self.weight_capacity_bytes = self.capacity_bytes
        self.last_mckp_plan = None
        return LoadCost()

    def resident_prefix_count(self, model_id):
        resident = self.resident_group_ids(model_id)
        prefix = len(resident)
        if resident != list(range(prefix)):
            raise RuntimeError(
                f"MCKP prefix invariant failed for model {model_id}: "
                f"{resident}")
        return prefix

    def resident_group_ids(self, model_id):
        return sorted(
            key[1] for key in self.allocated_extents
            if self._is_weight_key(key) and key[0] == model_id)

    def resident_suffix_start(self, model_id):
        resident = self.resident_group_ids(model_id)
        group_count = len(self.models[model_id].groups)
        start = group_count - len(resident)
        if resident != list(range(start, group_count)):
            raise RuntimeError(
                f"LRU suffix invariant failed for model {model_id}: "
                f"{resident}")
        return start

    def resident_model_bytes(self, model_id):
        return sum(
            self._object_bytes((model_id, group_id))
            for group_id in self.resident_group_ids(model_id))

    def resident_prefixes(self):
        return {
            model_id: self.resident_prefix_count(model_id)
            for model_id in self.models
        }

    def _mckp_reclaim(self, active_model, required_free_bytes):
        if required_free_bytes <= 0:
            return LoadCost()
        if self.prefix_stall_table is None:
            raise RuntimeError("mckp-prefix requires a prefix stall table")
        before = self.resident_prefixes()
        plan = solve_covering_mckp(
            self.prefix_stall_table, before, active_model,
            required_free_bytes, self.mckp_prediction_samples)
        if not plan.feasible:
            raise RuntimeError(
                "Prefix-MCKP cannot reclaim enough space: "
                f"required={required_free_bytes}")
        released = 0
        evicted_models = set()
        for key in plan.victim_keys:
            amount = self._free(key, coalesce=False)
            if amount:
                released += amount
                evicted_models.add(key[0])
        if released:
            self._coalesce()
        if released != plan.planned_free_bytes:
            raise RuntimeError(
                "Prefix-MCKP preview/commit byte mismatch: "
                f"planned={plan.planned_free_bytes} actual={released}")
        after = self.resident_prefixes()
        if after != plan.choices:
            raise RuntimeError(
                "Prefix-MCKP preview/commit prefix mismatch: "
                f"planned={plan.choices} actual={after}")
        self.last_mckp_plan = plan
        return LoadCost(
            evicted_objects=len(plan.victim_keys),
            evicted_bytes=released,
            mckp_required_free_bytes=plan.required_free_bytes,
            mckp_planned_free_bytes=released,
            mckp_predicted_damage_ms=plan.predicted_future_damage_ms,
            mckp_solver_time_ms=plan.solver_time_ms,
            mckp_frontier_peak_states=plan.frontier_peak_states,
            mckp_evicted_groups=len(plan.victim_keys),
            mckp_evicted_models=len(evicted_models),
        )

    def _lru_prefix_reclaim(self, active_model, required_free_bytes):
        """Pure model-LRU baseline that preserves the prefix ABI."""
        required = max(0, int(required_free_bytes))
        released = 0
        victims = []
        prefixes = self.resident_prefixes()
        while released < required:
            candidates = [
                model_id for model_id, prefix in prefixes.items()
                if model_id != active_model and prefix > 0
            ]
            if not candidates:
                raise RuntimeError(
                    "Prefix-LRU cannot reclaim enough space: "
                    f"required={required} released={released}")
            victim_model = min(candidates, key=lambda model_id: (
                max(
                    (self.last_access.get((model_id, group_id), -1)
                     for group_id in range(prefixes[model_id])),
                    default=-1),
                model_id,
            ))
            group_id = prefixes[victim_model] - 1
            key = (victim_model, group_id)
            amount = self._free(key, coalesce=False)
            if not amount:
                raise RuntimeError(f"Prefix-LRU failed to free {key}")
            prefixes[victim_model] -= 1
            released += amount
            victims.append(key)
        if released:
            self._coalesce()
        return LoadCost(
            evicted_objects=len(victims),
            evicted_bytes=released,
        )

    def _prefix_reclaim(self, active_model, required_free_bytes):
        if self.args.replacement_policy == "mckp-prefix":
            return self._mckp_reclaim(active_model, required_free_bytes)
        if self.args.replacement_policy == "lru-prefix":
            return self._lru_prefix_reclaim(
                active_model, required_free_bytes)
        raise RuntimeError("not a prefix replacement policy")

    @staticmethod
    def _is_weight_key(key):
        return key is not None and isinstance(key[0], int)

    def _victims(self, active_model):
        keys = [
            extent.key for extent in self.extents
            if self._is_weight_key(extent.key)
            and extent.key[0] != active_model
        ]
        remaining = set(keys)
        self._placement_rank_cache = {}
        for index, extent in enumerate(self.extents):
            if extent.key not in remaining:
                continue
            reclaim_span = extent.size
            if index > 0 and self.extents[index - 1].key is None:
                reclaim_span += self.extents[index - 1].size
            if (
                index + 1 < len(self.extents)
                and self.extents[index + 1].key is None
            ):
                reclaim_span += self.extents[index + 1].size
            self._placement_rank_cache[extent.key] = -reclaim_span
        selected = self._select_victim(remaining)
        self._placement_rank_cache = None
        return [] if selected is None else [selected]

    def _allocate_kv_blocks(self, active_model, cost):
        required = self.pending_kv_bytes
        self.pending_kv_bytes = 0
        block_size = self.args.kv_block_bytes
        def available_block_bytes():
            return sum(
                (extent.size // block_size) * block_size
                for extent in self.extents if extent.key is None
            )
        while available_block_bytes() < required:
            if self.args.replacement_policy in ("mckp-prefix", "lru-prefix"):
                before = available_block_bytes()
                cost.add(self._prefix_reclaim(
                    active_model, required - before))
                if available_block_bytes() <= before:
                    raise RuntimeError(
                        "Prefix policy cannot free KV-aligned Segment bytes")
            elif not self._evict_one(active_model, cost):
                raise RuntimeError("Unified Segment pool cannot fit KV blocks")
        remaining = required
        index = 0
        while remaining > 0 and index < len(self.extents):
            extent = self.extents[index]
            if extent.key is not None:
                index += 1
                continue
            allocatable = min(
                remaining, (extent.size // block_size) * block_size)
            if allocatable == 0:
                index += 1
                continue
            key = ("kv", self.next_kv_extent_id)
            self.next_kv_extent_id += 1
            allocated = Extent(extent.offset, allocatable, key)
            replacement = [allocated]
            if extent.size > allocatable:
                replacement.append(Extent(
                    extent.offset + allocatable,
                    extent.size - allocatable))
            self.extents[index:index + 1] = replacement
            self.allocated_extents[key] = allocated
            self.kv_keys.add(key)
            remaining -= allocatable
            index += len(replacement)
        if remaining:
            raise RuntimeError(
                "Unified Segment pool has bytes but not enough KV blocks")

    @staticmethod
    def _split_requests(sizes, left_capacity, right_capacity):
        left, right = [], []
        left_free, right_free = left_capacity, right_capacity
        for key, size in sizes:
            if size <= left_free and (
                    left_free >= right_free or size > right_free):
                left.append((key, size))
                left_free -= size
            elif size <= right_free:
                right.append((key, size))
                right_free -= size
            else:
                return None
        return left, right

    def _partition_requests(self, requests):
        free_extents = [extent for extent in self.extents
                        if extent.key is None
                        and extent.offset < self.weight_capacity_bytes]
        free_capacity = {
            id(extent): max(
                0,
                min(
                    extent.offset + extent.size,
                    self.weight_capacity_bytes,
                ) - extent.offset,
            )
            for extent in free_extents
        }
        if not requests:
            return []
        groups = [(free_extents, requests)]
        changed = True
        while changed:
            changed = False
            result = []
            for frees, pending in groups:
                if len(frees) < 2 or not pending:
                    result.append((frees, pending))
                    continue
                best = None
                for split in range(len(frees) - 1):
                    left_capacity = sum(
                        free_capacity[id(item)]
                        for item in frees[:split + 1])
                    right_capacity = sum(
                        free_capacity[id(item)]
                        for item in frees[split + 1:])
                    assignment = self._split_requests(
                        pending, left_capacity, right_capacity)
                    if assignment is None:
                        continue
                    begin = frees[split].offset + frees[split].size
                    end = frees[split + 1].offset
                    profit = sum(
                        extent.size for extent in self.extents
                        if extent.key is not None
                        and extent.offset >= begin
                        and extent.offset + extent.size <= end)
                    candidate = (profit, split, assignment)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
                if best is None:
                    result.append((frees, pending))
                    continue
                _, split, (left, right) = best
                result.extend([
                    (frees[:split + 1], left),
                    (frees[split + 1:], right),
                ])
                changed = True
            groups = result
        return groups

    def _merge_partition(self, first_free, last_free):
        start = next(
            index for index, extent in enumerate(self.extents)
            if extent is first_free)
        end = next(
            index for index, extent in enumerate(self.extents)
            if extent is last_free)
        region = self.extents[start:end + 1]
        allocated = [extent for extent in region if extent.key is not None]
        partition_end = min(
            last_free.offset + last_free.size,
            self.weight_capacity_bytes,
        )
        free_bytes = sum(
            max(
                0,
                min(extent.offset + extent.size, partition_end)
                - extent.offset,
            )
            for extent in region if extent.key is None
        )
        cursor = region[0].offset
        moved = 0
        replacement = []
        for extent in allocated:
            if extent.offset != cursor:
                moved += extent.size
            replacement.append(Extent(cursor, extent.size, extent.key))
            cursor += extent.size
        replacement.append(Extent(cursor, free_bytes))
        original_end = region[-1].offset + region[-1].size
        if partition_end < original_end:
            replacement.append(Extent(
                partition_end, original_end - partition_end))
        self.extents[start:end + 1] = replacement
        self.allocated_extents = {
            extent.key: extent for extent in self.extents
            if extent.key is not None
        }
        return replacement[-2] if partition_end < original_end else replacement[-1], moved

    def _reserve(self, free_extent, key, size):
        index = next(
            index for index, extent in enumerate(self.extents)
            if extent is free_extent)
        allocated = Extent(free_extent.offset, size, key)
        replacement = [allocated]
        if free_extent.size > size:
            replacement.append(Extent(
                free_extent.offset + size, free_extent.size - size))
        self.extents[index:index + 1] = replacement
        self.allocated_extents[key] = allocated
        return replacement[1] if len(replacement) == 2 else None

    def plan_request(self, model_id, tensors):
        ordered = []
        seen = set()
        for tensor in tensors:
            group = self._group(model_id, tensor)
            key = (model_id, group.group_id)
            if key in seen or self._allocated_extent(key) is not None:
                continue
            seen.add(key)
            ordered.append((key, group.logical_bytes))
        cost = LoadCost()
        if not ordered:
            if self.args.replacement_policy in ("mckp-prefix", "lru-prefix"):
                block_size = self.args.kv_block_bytes
                available = sum(
                    (extent.size // block_size) * block_size
                    for extent in self.extents if extent.key is None)
                cost.add(self._prefix_reclaim(
                    model_id, max(0, self.pending_kv_bytes - available)))
            self._allocate_kv_blocks(model_id, cost)
            return cost
        # Naive global relocation repacks every currently cached weight from
        # every other model whenever any part of the active model is reloaded.
        # Snapshot before MCKP reclamation so the baseline sees the same
        # request-start cache state as PBP.
        cost.model_reload_events = 1
        cost.naive_relocation_bytes = sum(
            extent.size for extent in self.extents
            if self._is_weight_key(extent.key) and extent.key[0] != model_id)
        required = sum(size for _, size in ordered)
        if self.args.replacement_policy in ("mckp-prefix", "lru-prefix"):
            shortage = max(
                0, required + self.pending_kv_bytes - self._free_bytes())
            cost.add(self._prefix_reclaim(model_id, shortage))
        else:
            while self._free_bytes() < required:
                if not self._evict_one(model_id, cost):
                    raise RuntimeError(
                        "Segment PBP cannot reclaim enough space")
        pbp_started = time.perf_counter()
        requests = sorted(ordered, key=lambda item: item[1], reverse=True)
        partitions = self._partition_requests(requests)
        moved = 0
        for frees, pending in partitions:
            if not pending:
                continue
            free_extent, local_moved = self._merge_partition(
                frees[0], frees[-1])
            moved += local_moved
            for key, size in pending:
                free_extent = self._reserve(free_extent, key, size)
        cost.pbp_planner_time_ms += (
            time.perf_counter() - pbp_started) * 1000.0
        if moved:
            self.compaction_count += 1
            first_key = ordered[0][0]
            self.planned_loads[first_key] = LoadCost(
                compaction_moved_bytes=moved,
                compaction_ms=(
                    moved / self.args.gpu_copy_bytes_per_ms
                    + self.args.compaction_fixed_ms))
        for key, size in ordered:
            load = self.planned_loads.setdefault(key, LoadCost())
            load.h2d_bytes = size
            load.h2d_ms = size / self.args.h2d_bytes_per_ms
            load.allocation_ms = self.args.segment_allocation_ms
        if self.args.replacement_policy in ("mckp-prefix", "lru-prefix"):
            block_size = self.args.kv_block_bytes
            available = sum(
                (extent.size // block_size) * block_size
                for extent in self.extents if extent.key is None)
            cost.add(self._prefix_reclaim(
                model_id, max(0, self.pending_kv_bytes - available)))
        self._allocate_kv_blocks(model_id, cost)
        return cost

    def metrics(self):
        result = super().metrics()
        kv_bytes = sum(
            extent.size for extent in self.extents
            if extent.key in self.kv_keys)
        result["kv_resident_bytes"] = kv_bytes
        result["logical_resident_bytes"] -= kv_bytes
        result["physical_resident_bytes"] -= kv_bytes
        return result

    def prepare_tensor(self, model_id, tensor):
        group = self._group(model_id, tensor)
        key = (model_id, group.group_id)
        planned = self.planned_loads.pop(key, None)
        if planned is not None:
            self._touch(key)
            return planned
        return super().prepare_tensor(model_id, tensor)
