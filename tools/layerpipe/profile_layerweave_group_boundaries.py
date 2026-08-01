#!/usr/bin/env python3
"""Profile real vLLM weight-use boundaries and build runtime TensorGroups.

One vLLM prefill records every parameter-owning module's first invocation.
Consecutive complete runtime stages are merged until the requested minimum
group size is reached.  The result contains measured group-ready boundaries;
it deliberately does not distribute layer time by Tensor bytes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import types
from collections import defaultdict
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

MIB = 1024 * 1024


def parse_ints(text: str) -> list[int]:
    return sorted(set(int(value) for value in text.split(",") if value))


def direct_parameters(module):
    return list(module.named_parameters(recurse=False))


class BoundaryRecorder:
    def __init__(self, model):
        self.model = model
        self.enabled = False
        self.start = None
        self.events = {}
        self.pending_events = {}
        self.order = []
        self.hooks = []
        self.stage_hooks = {}
        self.active_stage_names = None
        self.original_compute_logits = model.compute_logits
        self.parameter_owner = {}
        self.stage_parameters = defaultdict(list)
        self._index_parameters()
        self._install()

    def _index_parameters(self):
        seen = set()
        for module_name, module in self.model.named_modules():
            if module is self.model:
                continue
            for parameter_name, parameter in direct_parameters(module):
                identity = id(parameter)
                if identity in seen:
                    continue
                seen.add(identity)
                full_name = (
                    f"{module_name}.{parameter_name}"
                    if module_name else parameter_name
                )
                self.parameter_owner[identity] = module_name
                self.stage_parameters[module_name].append({
                    "name": full_name,
                    "shape": list(parameter.shape),
                    "bytes": parameter.numel() * parameter.element_size(),
                })
        for parameter_name, parameter in direct_parameters(self.model):
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            self.parameter_owner[identity] = "__compute_logits__"
            self.stage_parameters["__compute_logits__"].append({
                "name": parameter_name,
                "shape": list(parameter.shape),
                "bytes": parameter.numel() * parameter.element_size(),
            })

    def _record(self, name):
        if (
            not self.enabled or name in self.events
            or (
                self.active_stage_names is not None
                and name not in self.active_stage_names
            )
        ):
            return
        event = self.pending_events.pop(
            name, None) or torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream())
        self.events[name] = event
        self.order.append(name)

    def _install(self):
        def model_pre(_module, _args):
            if not self.enabled:
                return
            self.start.record(torch.cuda.current_stream())

        self.hooks.append(self.model.register_forward_pre_hook(model_pre))
        for module_name, module in self.model.named_modules():
            if module is self.model or not self.stage_parameters[module_name]:
                continue

            def hook(_module, _args, name=module_name):
                self._record(name)

            handle = module.register_forward_pre_hook(hook)
            self.hooks.append(handle)
            self.stage_hooks[module_name] = handle

        original = self.original_compute_logits

        def compute_logits(instance, *args, **kwargs):
            self._record("__compute_logits__")
            return original(*args, **kwargs)

        self.model.compute_logits = types.MethodType(
            compute_logits, self.model)

    def begin(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.events = {}
        self.pending_events = (
            {
                name: torch.cuda.Event(enable_timing=True)
                for name in self.active_stage_names
            }
            if self.active_stage_names is not None else {}
        )
        self.order = []
        self.enabled = True

    def restrict_to(self, stage_names):
        self.active_stage_names = set(stage_names)
        for name, handle in tuple(self.stage_hooks.items()):
            if name in self.active_stage_names:
                continue
            handle.remove()
            self.stage_hooks.pop(name)
            if handle in self.hooks:
                self.hooks.remove(handle)

    def finish(self):
        self.enabled = False
        torch.cuda.synchronize()
        if self.start is None:
            raise RuntimeError("vLLM model forward hook did not run")
        return {
            name: self.start.elapsed_time(event)
            for name, event in self.events.items()
        }, list(self.order)

    def close(self):
        self.enabled = False
        self.model.compute_logits = self.original_compute_logits
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.stage_hooks.clear()


def runtime_stages(recorder, observed_order):
    observed = []
    seen_parameters = set()
    for stage_name in observed_order:
        parameters = []
        for item in recorder.stage_parameters.get(stage_name, []):
            if item["name"] in seen_parameters:
                continue
            seen_parameters.add(item["name"])
            parameters.append(item)
        if parameters:
            observed.append({
                "stage_name": stage_name,
                "parameters": parameters,
                "bytes": sum(item["bytes"] for item in parameters),
                "observed": True,
            })
    unobserved = []
    for stage_name, parameters in recorder.stage_parameters.items():
        for item in parameters:
            if item["name"] not in seen_parameters:
                unobserved.append(item)
    if unobserved:
        observed.append({
            "stage_name": "__unobserved__",
            "parameters": unobserved,
            "bytes": sum(item["bytes"] for item in unobserved),
            "observed": False,
        })
    return observed


def merge_stages(stages, minimum_bytes):
    groups = []
    pending = []
    pending_bytes = 0
    for stage in stages:
        if pending and pending[-1]["observed"] != stage["observed"]:
            groups.append({
                "group_id": len(groups),
                "stages": [item["stage_name"] for item in pending],
                "parameters": [
                    parameter
                    for item in pending for parameter in item["parameters"]
                ],
                "logical_bytes": pending_bytes,
                "observed": all(item["observed"] for item in pending),
            })
            pending = []
            pending_bytes = 0
        pending.append(stage)
        pending_bytes += stage["bytes"]
        if pending_bytes >= minimum_bytes:
            groups.append({
                "group_id": len(groups),
                "stages": [item["stage_name"] for item in pending],
                "parameters": [
                    parameter
                    for item in pending for parameter in item["parameters"]
                ],
                "logical_bytes": pending_bytes,
                "observed": all(item["observed"] for item in pending),
            })
            pending = []
            pending_bytes = 0
    if pending:
        if groups and groups[-1]["observed"] == all(
                item["observed"] for item in pending):
            group = groups[-1]
            group["stages"].extend(
                item["stage_name"] for item in pending)
            group["parameters"].extend(
                parameter for item in pending
                for parameter in item["parameters"])
            group["logical_bytes"] += pending_bytes
            group["observed"] = (
                group["observed"]
                and all(item["observed"] for item in pending)
            )
        else:
            groups.append({
                "group_id": 0,
                "stages": [item["stage_name"] for item in pending],
                "parameters": [
                    parameter
                    for item in pending for parameter in item["parameters"]
                ],
                "logical_bytes": pending_bytes,
                "observed": all(item["observed"] for item in pending),
            })
    return groups


def group_boundaries(groups, stage_times, forward_end_ms):
    result = []
    previous = 0.0
    for group in groups:
        observed = [
            stage_times[name] for name in group["stages"]
            if name in stage_times
        ]
        boundary = min(observed) if observed else forward_end_ms
        boundary = max(previous, boundary)
        result.append(boundary)
        previous = boundary
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-group-min-mib", type=float, default=64.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--kv-block-tokens", type=int, default=16)
    args = parser.parse_args()
    tokens = parse_ints(args.tokens)
    if not tokens or min(tokens) <= 0:
        parser.error("--tokens must contain positive integers")
    max_tokens = max(tokens)
    blocks = (max_tokens + args.kv_block_tokens - 1) // args.kv_block_tokens
    started = time.perf_counter()
    model_config = json.loads((args.model / "config.json").read_text())
    extra_options = {}
    if model_config.get("model_type") == "llava":
        vision = model_config["vision_config"]
        image_size = int(vision["image_size"])
        patch_size = int(vision["patch_size"])
        image_feature_size = (image_size // patch_size) ** 2
        extra_options = {
            "image_input_type": "pixel_values",
            "image_token_id": int(model_config["image_token_index"]),
            "image_input_shape": f"1,3,{image_size},{image_size}",
            "image_feature_size": image_feature_size,
        }
    llm = LLM(
        model=str(args.model),
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_tokens + 1,
        num_gpu_blocks_override=blocks + 8,
        skip_tokenizer_init=True,
        **extra_options,
    )
    runtime_model = (
        llm.llm_engine.model_executor.driver_worker.model_runner.model)
    recorder = BoundaryRecorder(runtime_model)
    sampling = SamplingParams(
        temperature=0, max_tokens=1, ignore_eos=True)

    def run(token_count, measured):
        recorder.begin()
        envelope_start = torch.cuda.Event(enable_timing=True)
        envelope_end = torch.cuda.Event(enable_timing=True)
        envelope_start.record(torch.cuda.current_stream())
        wall_start = time.perf_counter()
        llm.generate(
            prompt_token_ids=[[1] * token_count],
            sampling_params=sampling,
            use_tqdm=False,
        )
        envelope_end.record(torch.cuda.current_stream())
        envelope_end.synchronize()
        stage_times, order = recorder.finish()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        end_ms = max(stage_times.values(), default=0.0)
        if measured:
            return (
                stage_times, order, wall_ms, end_ms,
                envelope_start.elapsed_time(envelope_end),
            )
        return None

    discovery_token = min(tokens)
    for _ in range(args.warmup):
        run(discovery_token, False)
    stage_times, observed_order, _, _, _ = run(discovery_token, True)
    stages = runtime_stages(recorder, observed_order)
    groups = merge_stages(
        stages, int(args.tensor_group_min_mib * MIB))
    boundary_stages = {
        next(name for name in group["stages"] if name in stage_times)
        for group in groups if group["observed"]
    }
    recorder.restrict_to(boundary_stages | {"__compute_logits__"})
    profiles = {}
    order_mismatches = []
    for token_count in tokens:
        for _ in range(args.warmup):
            run(token_count, False)
        baseline_wall_ms = []
        baseline_gpu_ms = []
        samples = []
        for _ in range(args.repeats):
            recorder.enabled = False
            envelope_start = torch.cuda.Event(enable_timing=True)
            envelope_end = torch.cuda.Event(enable_timing=True)
            envelope_start.record(torch.cuda.current_stream())
            wall_start = time.perf_counter()
            llm.generate(
                prompt_token_ids=[[1] * token_count],
                sampling_params=sampling,
                use_tqdm=False,
            )
            envelope_end.record(torch.cuda.current_stream())
            envelope_end.synchronize()
            baseline_wall_ms.append(
                (time.perf_counter() - wall_start) * 1000.0)
            baseline_gpu_ms.append(
                envelope_start.elapsed_time(envelope_end))
            times, order, wall_ms, forward_end_ms, gpu_ms = run(
                token_count, True)
            expected_boundary_order = [
                name for name in observed_order
                if name in boundary_stages or name == "__compute_logits__"
            ]
            if order != expected_boundary_order:
                order_mismatches.append({
                    "tokens": token_count,
                    "expected": expected_boundary_order,
                    "actual": order,
                })
            samples.append({
                "stage_ms": times,
                "group_boundary_ms": group_boundaries(
                    groups, times, forward_end_ms),
                "wall_ms": wall_ms,
                "gpu_envelope_ms": gpu_ms,
                "last_boundary_ms": forward_end_ms,
            })
        profiles[str(token_count)] = {
            "samples": samples,
            "median_group_boundary_ms": [
                statistics.median(
                    sample["group_boundary_ms"][index]
                    for sample in samples)
                for index in range(len(groups))
            ],
            "median_wall_ms":
                statistics.median(sample["wall_ms"] for sample in samples),
            "baseline_wall_ms": baseline_wall_ms,
            "baseline_gpu_ms": baseline_gpu_ms,
            "median_baseline_wall_ms":
                statistics.median(baseline_wall_ms),
            "median_baseline_gpu_ms":
                statistics.median(baseline_gpu_ms),
            "event_wall_overhead_ms": (
                statistics.median(sample["wall_ms"] for sample in samples)
                - statistics.median(baseline_wall_ms)
            ),
            "event_gpu_overhead_ms": (
                statistics.median(
                    sample["gpu_envelope_ms"] for sample in samples)
                - statistics.median(baseline_gpu_ms)
            ),
        }
    recorder.close()
    document = {
        "format": "layerweave-runtime-group-boundary-profile-v1",
        "source": "real-vllm-prefill-cuda-event-weight-use-boundaries",
        "model_id": args.model_id,
        "model_path": str(args.model),
        "gpu": torch.cuda.current_device(),
        "gpu_name": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "workload_modality": (
            "text-only prefill; vision weights are unobserved"
            if model_config.get("model_type") == "llava"
            else "text prefill"
        ),
        "tokens": tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "tensor_group_min_bytes": int(args.tensor_group_min_mib * MIB),
        "runtime_parameter_bytes":
            sum(stage["bytes"] for stage in stages),
        "observed_parameter_bytes":
            sum(stage["bytes"] for stage in stages if stage["observed"]),
        "unobserved_parameter_bytes":
            sum(stage["bytes"] for stage in stages if not stage["observed"]),
        "stages": stages,
        "groups": groups,
        "profiles": profiles,
        "observed_stage_order": observed_order,
        "order_mismatches": order_mismatches,
        "elapsed_seconds": time.perf_counter() - started,
        "caveats": [
            "One-token output records the prefill path; no decode iteration.",
            "Unobserved parameters are appended as a tail storage group.",
            "CUDA event insertion overhead must be measured separately.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "model_id": args.model_id,
        "tokens": tokens,
        "runtime_parameter_gib":
            document["runtime_parameter_bytes"] / 1024**3,
        "observed_parameter_gib":
            document["observed_parameter_bytes"] / 1024**3,
        "stages": len(stages),
        "groups": len(groups),
        "order_mismatches": len(order_mismatches),
        "elapsed_seconds": document["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
