"""Shared memory-backend types for the tensor-level simulator."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

MIB = 1024 * 1024


@dataclass(frozen=True)
class TensorSpec:
    tensor_id: int
    name: str
    layer_id: int | None
    logical_bytes: int
    compact_offset: int


@dataclass(frozen=True)
class TensorGroup:
    group_id: int
    tensor_ids: tuple[int, ...]
    logical_bytes: int
    compact_offset: int


@dataclass
class TensorModel:
    model_id: int
    tensors: list[TensorSpec]
    logical_bytes: int
    max_input_tokens: int
    groups: list[TensorGroup] = field(default_factory=list)
    tensor_to_group: dict[int, int] = field(default_factory=dict)


@dataclass
class LoadCost:
    h2d_bytes: int = 0
    h2d_ms: float = 0.0
    allocation_ms: float = 0.0
    map_ms: float = 0.0
    unmap_ms: float = 0.0
    compaction_ms: float = 0.0
    evicted_objects: int = 0
    evicted_bytes: int = 0
    map_calls: int = 0
    unmap_calls: int = 0
    mapped_pages: int = 0
    unmapped_pages: int = 0
    compaction_moved_bytes: int = 0
    mckp_required_free_bytes: int = 0
    mckp_planned_free_bytes: int = 0
    mckp_predicted_damage_ms: float = 0.0
    mckp_solver_time_ms: float = 0.0
    mckp_frontier_peak_states: int = 0
    mckp_evicted_groups: int = 0
    mckp_evicted_models: int = 0
    pbp_planner_time_ms: float = 0.0
    model_reload_events: int = 0
    naive_relocation_bytes: int = 0

    @property
    def ready_ms(self) -> float:
        return (
            self.h2d_ms + self.allocation_ms + self.map_ms
            + self.unmap_ms + self.compaction_ms
        )

    def add(self, other: "LoadCost") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class Extent:
    offset: int
    size: int
    key: tuple[int, int] | None = None


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def contiguous_runs(pages: set[int]) -> int:
    if not pages:
        return 0
    ordered = sorted(pages)
    return 1 + sum(
        current != previous + 1
        for previous, current in zip(ordered, ordered[1:])
    )


class MemoryBackend:
    def __init__(self, models, capacity_bytes, page_size, args):
        self.models = models
        self.capacity_bytes = capacity_bytes
        self.weight_capacity_bytes = capacity_bytes
        self.page_size = page_size
        self.args = args
        self.clock = 0
        self.model_accesses = {model_id: 0 for model_id in models}
        self.last_access = {}
        self.pipeline_benefits = {}
        self.future_model_probabilities = {}
        self.prefix_stall_table = None
        self.mckp_prediction_samples = {}
        self.last_mckp_plan = None

    def clone(self):
        shared = [self.models, self.args]
        if self.prefix_stall_table is not None:
            shared.append(self.prefix_stall_table)
        shared.extend(self._clone_shared_objects())
        memo = {id(item): item for item in shared}
        return copy.deepcopy(self, memo)

    def _clone_shared_objects(self):
        """Read-only backend indexes that routing previews may safely share."""
        return ()

    def begin_request(self, kv_bytes: int, active_model: int) -> LoadCost:
        self.weight_capacity_bytes = max(0, self.capacity_bytes - kv_bytes)
        return self._trim_to_capacity(active_model)

    def plan_request(
            self, model_id: int, tensors: list[TensorSpec]) -> LoadCost:
        return LoadCost()

    def _trim_to_capacity(self, active_model: int) -> LoadCost:
        raise NotImplementedError

    def prepare_tensor(self, model_id: int, tensor: TensorSpec) -> LoadCost:
        raise NotImplementedError

    def tensor_resident(self, model_id: int, tensor: TensorSpec) -> bool:
        raise NotImplementedError

    def physical_model_bytes(self, model_id: int) -> int:
        raise NotImplementedError

    def used_bytes(self) -> int:
        raise NotImplementedError

    def finish_request(self, model_id: int) -> None:
        self.model_accesses[model_id] += 1

    def set_pipeline_benefits(self, benefits) -> None:
        self.pipeline_benefits = benefits

    def set_future_model_probabilities(self, probabilities) -> None:
        self.future_model_probabilities = probabilities

    def set_prefix_stall_table(self, table) -> None:
        self.prefix_stall_table = table

    def set_mckp_prediction_samples(self, samples) -> None:
        self.mckp_prediction_samples = samples

    def _touch(self, key) -> None:
        self.clock += 1
        self.last_access[key] = self.clock

    def _frequency_value(self, key) -> tuple:
        if self.args.replacement_policy == "frequency-lru-suffix-first":
            return (
                self.model_accesses.get(key[0], 0),
                -key[1],
                self.last_access.get(key, -1),
            )
        return (
            self.model_accesses.get(key[0], 0),
            self.last_access.get(key, -1),
        )

    def _pipeline_loss(self, key) -> float:
        probability = self.future_model_probabilities.get(
            key[0], 1.0 / len(self.models))
        return probability * self.pipeline_benefits.get(key, 0.0)

    def _placement_rank(self, key):
        return 0

    def _object_bytes(self, key):
        raise NotImplementedError

    def _value(self, key) -> tuple:
        if self.args.replacement_policy == "lru":
            return (self.last_access.get(key, -1),)
        if self.args.replacement_policy == "pipeline-value":
            return (
                self._pipeline_loss(key),
                self.last_access.get(key, -1),
            )
        if self.args.replacement_policy.startswith("protected-pipeline"):
            benefit = self.pipeline_benefits.get(key, 0.0)
            protected = benefit > self.args.pipeline_protected_threshold_ms
            probability = self.future_model_probabilities.get(
                key[0], 1.0 / len(self.models))
            return (
                int(protected),
                probability,
                benefit,
                self._placement_rank(key),
                self.last_access.get(key, -1),
            )
        return self._frequency_value(key)

    def _select_victim(self, keys):
        keys = list(keys)
        if not keys:
            return None
        pipeline = min(keys, key=self._value)
        if self.args.replacement_policy != "protected-pipeline-shadow":
            return pipeline
        frequency = min(keys, key=self._frequency_value)
        if pipeline == frequency:
            return pipeline
        pipeline_soft = (
            self.pipeline_benefits.get(pipeline, 0.0)
            <= self.args.pipeline_protected_threshold_ms
        )
        frequency_protected = (
            self.pipeline_benefits.get(frequency, 0.0)
            > self.args.pipeline_protected_threshold_ms
        )
        probabilities = self.future_model_probabilities
        default_probability = 1.0 / len(self.models)
        pipeline_churn = (
            probabilities.get(pipeline[0], default_probability)
            * self._object_bytes(pipeline)
        )
        frequency_churn = (
            probabilities.get(frequency[0], default_probability)
            * self._object_bytes(frequency)
        )
        churn_safe = (
            pipeline_churn
            <= frequency_churn * self.args.shadow_max_churn_ratio
        )
        return (
            pipeline
            if pipeline_soft and frequency_protected and churn_safe
            else frequency
        )

    def _group(self, model_id, tensor):
        model = self.models[model_id]
        return model.groups[model.tensor_to_group[tensor.tensor_id]]

    def metrics(self) -> dict:
        return {
            "logical_resident_bytes": 0,
            "physical_resident_bytes": self.used_bytes(),
            "internal_fragmentation_bytes": 0,
            "external_fragmentation_ratio": 0.0,
            "largest_free_extent_bytes":
                max(0, self.weight_capacity_bytes - self.used_bytes()),
            "stranded_resident_bytes": 0,
        }
