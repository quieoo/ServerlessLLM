"""VMM Tensor-page and Compact-page memory backends."""

from __future__ import annotations

from tensor_sim_memory_common import (
    LoadCost, MemoryBackend, ceil_div, contiguous_runs,
)
from tensor_sim_mckp import solve_covering_mckp


class PagePrefixMckpMixin:
    """Strict group-prefix replacement shared by both page layouts."""

    def resident_prefix_count(self, model_id):
        resident = self.resident_group_ids(model_id)
        prefix = len(resident)
        if resident != list(range(prefix)):
            raise RuntimeError(
                f"MCKP prefix invariant failed for model {model_id}: "
                f"{resident}")
        return prefix

    def resident_prefixes(self):
        return {
            model_id: self.resident_prefix_count(model_id)
            for model_id in self.models
        }

    def resident_model_bytes(self, model_id):
        return self._prefix_physical_bytes(
            model_id, self.resident_prefix_count(model_id))

    def _mckp_reclaim(self, active_model, required_free_bytes):
        required = max(0, int(required_free_bytes))
        if required == 0:
            return LoadCost()
        if self.prefix_stall_table is None:
            raise RuntimeError("mckp-prefix requires a prefix stall table")
        before = self.resident_prefixes()
        plan = solve_covering_mckp(
            self.prefix_stall_table, before, active_model, required,
            self.mckp_prediction_samples,
            prefix_bytes=self._prefix_physical_bytes)
        if not plan.feasible:
            raise RuntimeError(
                "Prefix-MCKP cannot reclaim enough page space: "
                f"required={required}")
        cost = LoadCost()
        evicted_models = set()
        for model_id, old_prefix in before.items():
            new_prefix = plan.choices[model_id]
            if new_prefix < old_prefix:
                self._commit_prefix(
                    model_id, old_prefix, new_prefix, cost)
                evicted_models.add(model_id)
        if cost.evicted_bytes != plan.planned_free_bytes:
            raise RuntimeError(
                "Prefix-MCKP page preview/commit byte mismatch: "
                f"planned={plan.planned_free_bytes} "
                f"actual={cost.evicted_bytes}")
        after = self.resident_prefixes()
        if after != plan.choices:
            raise RuntimeError(
                "Prefix-MCKP page preview/commit prefix mismatch: "
                f"planned={plan.choices} actual={after}")
        cost.mckp_required_free_bytes = plan.required_free_bytes
        cost.mckp_planned_free_bytes = cost.evicted_bytes
        cost.mckp_predicted_damage_ms = plan.predicted_future_damage_ms
        cost.mckp_solver_time_ms = plan.solver_time_ms
        cost.mckp_frontier_peak_states = plan.frontier_peak_states
        cost.mckp_evicted_groups = sum(
            before[model_id] - prefix
            for model_id, prefix in plan.choices.items())
        cost.mckp_evicted_models = len(evicted_models)
        self.last_mckp_plan = plan
        return cost

    def _prefix_reclaim(self, active_model, required_free_bytes):
        if self.args.replacement_policy != "mckp-prefix":
            raise RuntimeError("not a page prefix replacement policy")
        return self._mckp_reclaim(active_model, required_free_bytes)

    def plan_request(self, model_id, tensors):
        """Reclaim once for all pages missing from this request.

        Page backends used to invoke Prefix-MCKP independently from every
        ``prepare_tensor`` call.  Request execution is sequential and no
        other allocation can intervene, so one batch shortage is sufficient.
        Keep the per-tensor reclaim as a defensive fallback.
        """
        if self.args.replacement_policy != "mckp-prefix":
            return LoadCost()
        needed = self._missing_request_physical_bytes(model_id, tensors)
        shortage = max(
            0, self.used_bytes() + needed - self.weight_capacity_bytes)
        return self._prefix_reclaim(model_id, shortage)


class TensorPageMemory(PagePrefixMckpMixin, MemoryBackend):
    def __init__(self, models, capacity_bytes, page_size, args):
        super().__init__(models, capacity_bytes, page_size, args)
        self.resident = set()
        self.prefix_physical_bytes = {}
        for model_id, model in models.items():
            total = 0
            self.prefix_physical_bytes[(model_id, 0)] = 0
            for group in model.groups:
                total += ceil_div(group.logical_bytes, page_size) * page_size
                self.prefix_physical_bytes[
                    (model_id, group.group_id + 1)] = total

    def _clone_shared_objects(self):
        return (self.prefix_physical_bytes,)

    def _pages(self, model_id, group):
        return ceil_div(group.logical_bytes, self.page_size)

    def used_bytes(self):
        return sum(
            self._pages(model_id, self.models[model_id].groups[group_id])
            * self.page_size
            for model_id, group_id in self.resident
        )

    def _object_bytes(self, key):
        group = self.models[key[0]].groups[key[1]]
        return self._pages(key[0], group) * self.page_size

    def _evict_one(self, active_model, cost):
        key = self._select_victim(
            key for key in self.resident if key[0] != active_model)
        if key is None:
            return False
        group = self.models[key[0]].groups[key[1]]
        pages = self._pages(key[0], group)
        self.resident.remove(key)
        cost.evicted_objects += 1
        cost.evicted_bytes += pages * self.page_size
        cost.unmapped_pages += pages
        cost.unmap_calls += pages
        cost.unmap_ms += pages * (
            self.args.unmap_fixed_ms + self.args.unmap_per_page_ms)
        return True

    def _trim_to_capacity(self, active_model):
        if self.args.replacement_policy == "mckp-prefix":
            shortage = max(
                0, self.used_bytes() - self.weight_capacity_bytes)
            return self._prefix_reclaim(active_model, shortage)
        cost = LoadCost()
        while self.used_bytes() > self.weight_capacity_bytes:
            if not self._evict_one(active_model, cost):
                raise RuntimeError("No evictable tensor-page allocation")
        return cost

    def tensor_resident(self, model_id, tensor):
        group = self._group(model_id, tensor)
        return (model_id, group.group_id) in self.resident

    def resident_group_ids(self, model_id):
        return sorted(
            group_id for resident_model, group_id in self.resident
            if resident_model == model_id)

    def _prefix_physical_bytes(self, model_id, prefix):
        return self.prefix_physical_bytes[(model_id, prefix)]

    def _missing_request_physical_bytes(self, model_id, tensors):
        missing = set()
        for tensor in tensors:
            group = self._group(model_id, tensor)
            key = (model_id, group.group_id)
            if key not in self.resident:
                missing.add(group.group_id)
        return sum(
            self._pages(model_id, self.models[model_id].groups[group_id])
            * self.page_size
            for group_id in missing)

    def _commit_prefix(self, model_id, old_prefix, new_prefix, cost):
        victims = list(range(new_prefix, old_prefix))
        pages = 0
        for group_id in victims:
            key = (model_id, group_id)
            if key not in self.resident:
                raise RuntimeError(f"missing resident tensor-page {key}")
            self.resident.remove(key)
            pages += self._pages(
                model_id, self.models[model_id].groups[group_id])
        cost.evicted_objects += len(victims)
        cost.evicted_bytes += pages * self.page_size
        cost.unmapped_pages += pages
        cost.unmap_calls += pages
        cost.unmap_ms += pages * (
            self.args.unmap_fixed_ms + self.args.unmap_per_page_ms)

    def prepare_tensor(self, model_id, tensor):
        group = self._group(model_id, tensor)
        key = (model_id, group.group_id)
        if key in self.resident:
            self._touch(key)
            return LoadCost()
        pages = self._pages(model_id, group)
        needed = pages * self.page_size
        cost = LoadCost()
        if self.args.replacement_policy == "mckp-prefix":
            shortage = max(
                0,
                self.used_bytes() + needed - self.weight_capacity_bytes)
            cost.add(self._prefix_reclaim(model_id, shortage))
        while self.used_bytes() + needed > self.weight_capacity_bytes:
            if not self._evict_one(model_id, cost):
                raise RuntimeError(f"Tensor-page pool cannot fit {key}")
        self.resident.add(key)
        self._touch(key)
        cost.h2d_bytes = group.logical_bytes
        cost.h2d_ms = group.logical_bytes / self.args.h2d_bytes_per_ms
        cost.map_calls = pages
        cost.mapped_pages = pages
        cost.map_ms = pages * (
            self.args.map_fixed_ms + self.args.map_per_page_ms)
        return cost

    def physical_model_bytes(self, model_id):
        return sum(
            self._pages(model_id, group) * self.page_size
            for group in self.models[model_id].groups
        )

    def metrics(self):
        logical = sum(
            self.models[m].groups[g].logical_bytes for m, g in self.resident)
        physical = self.used_bytes()
        return {
            "logical_resident_bytes": logical,
            "physical_resident_bytes": physical,
            "internal_fragmentation_bytes": physical - logical,
            "external_fragmentation_ratio": 0.0,
            "largest_free_extent_bytes":
                max(0, self.weight_capacity_bytes - physical),
            "stranded_resident_bytes": 0,
        }


class CompactPageMemory(PagePrefixMckpMixin, MemoryBackend):
    def __init__(self, models, capacity_bytes, page_size, args):
        super().__init__(models, capacity_bytes, page_size, args)
        self.resident_pages = {model_id: set() for model_id in models}
        self.tensor_pages = {}
        self.group_pages = {}
        self.prefix_page_counts = {}
        self.page_tensors = {}
        for model_id, model in models.items():
            reverse = {}
            for tensor in model.tensors:
                begin = tensor.compact_offset // page_size
                end = (
                    tensor.compact_offset + tensor.logical_bytes - 1
                ) // page_size
                pages = frozenset(range(begin, end + 1))
                self.tensor_pages[(model_id, tensor.tensor_id)] = pages
                for page in pages:
                    reverse.setdefault(page, set()).add(tensor.tensor_id)
            for group in model.groups:
                begin = group.compact_offset // page_size
                end = (
                    group.compact_offset + group.logical_bytes - 1
                ) // page_size
                self.group_pages[(model_id, group.group_id)] = frozenset(
                    range(begin, end + 1))
            page_count = 0
            self.prefix_page_counts[(model_id, 0)] = 0
            for group in model.groups:
                group_pages = self.group_pages[
                    (model_id, group.group_id)]
                if group_pages:
                    page_count = max(page_count, max(group_pages) + 1)
                self.prefix_page_counts[
                    (model_id, group.group_id + 1)] = page_count
            self.page_tensors[model_id] = reverse

    def _clone_shared_objects(self):
        return (
            self.tensor_pages,
            self.group_pages,
            self.prefix_page_counts,
            self.page_tensors,
        )

    def used_bytes(self):
        return sum(len(pages) for pages in self.resident_pages.values()) * (
            self.page_size)

    def _object_bytes(self, key):
        return self.page_size

    def tensor_resident(self, model_id, tensor):
        pages = self.tensor_pages[(model_id, tensor.tensor_id)]
        return pages <= self.resident_pages[model_id]

    def resident_group_ids(self, model_id):
        prefix = self.resident_prefix_count(model_id)
        return list(range(prefix))

    def resident_prefix_count(self, model_id):
        resident = self.resident_pages[model_id]
        canonical = (
            not resident
            or (
                min(resident) == 0
                and max(resident) == len(resident) - 1
            )
        )
        matches = [
            prefix
            for prefix in range(len(self.models[model_id].groups) + 1)
            if (
                canonical
                and
                len(resident)
                == self.prefix_page_counts[(model_id, prefix)]
            )
        ]
        if not matches:
            raise RuntimeError(
                f"MCKP compact-page prefix invariant failed for model "
                f"{model_id}: {len(resident)} resident pages")
        return max(matches)

    def _prefix_physical_bytes(self, model_id, prefix):
        return (
            self.prefix_page_counts[(model_id, prefix)] * self.page_size)

    def _missing_request_physical_bytes(self, model_id, tensors):
        required = set()
        for tensor in tensors:
            group = self._group(model_id, tensor)
            required.update(
                self.group_pages[(model_id, group.group_id)])
        return len(required - self.resident_pages[model_id]) * self.page_size

    def _commit_prefix(self, model_id, old_prefix, new_prefix, cost):
        keep_count = self.prefix_page_counts[(model_id, new_prefix)]
        old_count = self.prefix_page_counts[(model_id, old_prefix)]
        keep = set(range(keep_count))
        victims = self.resident_pages[model_id] - keep
        expected = set(range(keep_count, old_count))
        if victims != expected:
            raise RuntimeError(
                f"non-canonical compact prefix for model {model_id}: "
                f"extra={sorted(victims - expected)}")
        self.resident_pages[model_id].difference_update(victims)
        count = len(victims)
        cost.evicted_objects += count
        cost.evicted_bytes += count * self.page_size
        cost.unmapped_pages += count
        cost.unmap_calls += count
        cost.unmap_ms += count * (
            self.args.unmap_fixed_ms + self.args.unmap_per_page_ms)

    def _page_value(self, model_id, page):
        if (
            self.args.replacement_policy == "pipeline-value"
            or self.args.replacement_policy.startswith("protected-pipeline")
        ):
            return self._value((model_id, page))
        broken = []
        resident = self.resident_pages[model_id]
        for tensor_id in self.page_tensors[model_id].get(page, ()):
            tensor = self.models[model_id].tensors[tensor_id]
            pages = self.tensor_pages[(model_id, tensor_id)]
            # `page` is a required member of `pages`; if the tensor is
            # currently complete, removing this page necessarily breaks it.
            # Avoid materializing `resident - {page}` for every candidate.
            if pages <= resident:
                broken.append((model_id, tensor_id))
        if not broken:
            return (-1, self.last_access.get((model_id, page), -1))
        return min(self._value(key) for key in broken)

    def _evict_pages(self, count, active_model, cost):
        if count <= 0:
            return True
        candidates = [
            (model_id, page)
            for model_id, pages in self.resident_pages.items()
            if model_id != active_model
            for page in pages
        ]
        if len(candidates) < count:
            return False
        if self.args.replacement_policy == "protected-pipeline-shadow":
            remaining = {(model_id, page) for model_id, page in candidates}
            selected_keys = []
            for _ in range(count):
                victim = self._select_victim(remaining)
                selected_keys.append(victim)
                remaining.remove(victim)
            selected = [
                (self._value(key), key[0], key[1])
                for key in selected_keys
            ]
        else:
            selected = sorted(
                (self._page_value(model_id, page), model_id, page)
                for model_id, page in candidates
            )[:count]
        by_model = {}
        for _, model_id, page in selected:
            self.resident_pages[model_id].remove(page)
            by_model.setdefault(model_id, set()).add(page)
        cost.evicted_objects += count
        cost.evicted_bytes += count * self.page_size
        cost.unmapped_pages += count
        cost.unmap_calls += count
        cost.unmap_ms += count * (
            self.args.unmap_fixed_ms + self.args.unmap_per_page_ms)
        return True

    def _evict_one(self, active_model, cost):
        return self._evict_pages(1, active_model, cost)

    def _trim_to_capacity(self, active_model):
        if self.args.replacement_policy == "mckp-prefix":
            shortage = max(
                0, self.used_bytes() - self.weight_capacity_bytes)
            return self._prefix_reclaim(active_model, shortage)
        cost = LoadCost()
        shortage = ceil_div(
            max(0, self.used_bytes() - self.weight_capacity_bytes),
            self.page_size,
        )
        if not self._evict_pages(shortage, active_model, cost):
            raise RuntimeError("No evictable compact page")
        return cost

    def prepare_tensor(self, model_id, tensor):
        group = self._group(model_id, tensor)
        required = self.group_pages[(model_id, group.group_id)]
        missing = set(required - self.resident_pages[model_id])
        if not missing:
            for page in required:
                self._touch((model_id, page))
            return LoadCost()
        cost = LoadCost()
        needed = len(missing) * self.page_size
        shortage = ceil_div(
            max(
                0,
                self.used_bytes() + needed - self.weight_capacity_bytes,
            ),
            self.page_size,
        )
        if self.args.replacement_policy == "mckp-prefix":
            cost.add(self._prefix_reclaim(
                model_id, shortage * self.page_size))
        elif not self._evict_pages(shortage, model_id, cost):
            raise RuntimeError("Compact-page pool cannot fit tensor")
        self.resident_pages[model_id].update(missing)
        for page in required:
            self._touch((model_id, page))
        logical_h2d = 0
        for page in missing:
            begin = page * self.page_size
            end = min(
                begin + self.page_size, self.models[model_id].logical_bytes)
            logical_h2d += max(0, end - begin)
        cost.h2d_bytes = logical_h2d
        cost.h2d_ms = logical_h2d / self.args.h2d_bytes_per_ms
        cost.map_calls = len(missing)
        cost.mapped_pages = len(missing)
        cost.map_ms = len(missing) * (
            self.args.map_fixed_ms + self.args.map_per_page_ms)
        return cost

    def physical_model_bytes(self, model_id):
        return ceil_div(
            self.models[model_id].logical_bytes, self.page_size
        ) * self.page_size

    def metrics(self):
        physical = self.used_bytes()
        logical_resident = 0
        stranded = 0
        internal = 0
        for model_id, model in self.models.items():
            resident = self.resident_pages[model_id]
            complete = set()
            for tensor in model.tensors:
                pages = self.tensor_pages[(model_id, tensor.tensor_id)]
                if pages <= resident:
                    complete.add(tensor.tensor_id)
                    logical_resident += tensor.logical_bytes
            for page in resident:
                page_begin = page * self.page_size
                valid_bytes = max(
                    0, min(self.page_size, model.logical_bytes - page_begin))
                internal += self.page_size - valid_bytes
                if not (
                    self.page_tensors[model_id].get(page, set()) & complete
                ):
                    stranded += valid_bytes
        return {
            "logical_resident_bytes": logical_resident,
            "physical_resident_bytes": physical,
            "internal_fragmentation_bytes": internal,
            "external_fragmentation_ratio": 0.0,
            "largest_free_extent_bytes":
                max(0, self.weight_capacity_bytes - physical),
            "stranded_resident_bytes": stranded,
        }
