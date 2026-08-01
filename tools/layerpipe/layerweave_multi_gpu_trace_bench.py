#!/usr/bin/env python3
"""Two-GPU LayerWeave trace replay with one global arrival queue.

Each GPU lives in a persistent spawned process because CUDA visibility and the
Tangram VMM pool are process-local.  The coordinator reads the trace once and
dispatches every request exactly once. Requests remain in a global pending
queue until a GPU actually starts them. ``minimal`` scores queue plus
missing-weight load, while ``joint`` scores queue plus PSE service and cache
transition cost. A bounded placement wait may preserve cache affinity without
pre-binding a FIFO request behind a busy GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace


def _request_dict(item) -> dict:
    return {
        "request_id": int(item.request_id),
        "arrival_s": float(item.arrival_s),
        "model_id": int(item.model_id),
        "input_tokens": int(item.input_tokens),
        "output_tokens": int(item.output_tokens),
        "trace_input_tokens": int(item.trace_input_tokens),
        "trace_output_tokens": int(item.trace_output_tokens),
    }


def _worker_main(device: int, options: dict, commands, events) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        os.environ["USE_GPU"] = "0"
        os.environ["LAYERWEAVE_PREFETCH"] = "1"
        os.environ["LAYERWEAVE_PREFIX_LAYERS"] = (
            "full" if options["system_policy"] == "minimal" else "full")
        os.environ["LAYERWEAVE_CACHE_POLICY"] = (
            "fixed" if options["system_policy"] == "minimal" else "joint")
        os.environ["LAYERWEAVE_M4_PROFILE"] = options["profile"]
        os.environ["LAYERWEAVE_DEMAND_DECAY"] = str(options["demand_decay"])
        os.environ["LAYERWEAVE_UNCERTAINTY_MS"] = str(
            options["uncertainty_ms"])
        # vllm_odkv_trace_bench fixes CUDA visibility while importing torch.
        sys.argv = [sys.argv[0], "--device", str(device)]
        import vllm_odkv_trace_bench as single
        import torch
        from vllm import SamplingParams
        from layerpipe_bench import RequestResult, TraceRequest

        args = SimpleNamespace(**options["single_args"])
        args.device = device
        requests = [
            TraceRequest(
                request_id=item["request_id"],
                arrival_s=item["arrival_s"],
                model_id=item["model_id"],
                input_tokens=item["input_tokens"],
                output_tokens=item["output_tokens"],
            )
            for item in options["requests"]
        ]
        for request, source in zip(requests, options["requests"]):
            request.trace_input_tokens = source["trace_input_tokens"]
            request.trace_output_tokens = source["trace_output_tokens"]
        model_limits = single.load_model_input_limits(args.config)
        single.apply_request_input_limits(
            requests,
            model_limits,
            args.max_model_len,
            args.truncate_input_to_model_limit,
        )
        models = single.load_models(args.config, args.model_path)
        pool, engines, startup, engine_limits = single.initialize_engines(
            args, requests, models)
        joint_pool_input_limits = single.apply_joint_pool_input_limits(
            args, requests, engines)
        controllers = {
            model_id: single.get_layerweave_controller(
                single.get_worker(engine))
            for model_id, engine in engines.items()
        }
        route_kv = {
            model_id: {
                "block_size_tokens": int(
                    single.get_worker(engine).cache_config.block_size),
                "block_size_bytes": int(
                    single.get_worker(engine).odkv_stats()[
                        "block_size_bytes"]),
            }
            for model_id, engine in engines.items()
        }
        cache_policy = None
        if options["system_policy"] == "joint":
            from layerweave_cache_policy import LayerWeaveSingleGpuCachePolicy
            cache_policy = LayerWeaveSingleGpuCachePolicy(
                controllers=controllers,
                profile_path=Path(options["profile"]),
                pool_pages=int(
                    args.vmm_pool_gib * (1024 ** 3)
                    // next(iter(controllers.values())).page_size),
                expected_input_scale=args.input_scale,
                decay=options["demand_decay"],
                uncertainty_ms=options["uncertainty_ms"],
                policy_mode=options["cache_policy_mode"],
            )

        by_id = {item.request_id: item for item in requests}
        active_model_id = None
        batch_id = 0
        state_lock = threading.Lock()
        projected_residencies = {
            model_id: {
                page for page, resident in enumerate(
                    controller.pool.layerweave_residency(
                        controller.model_path))
                if resident
            }
            for model_id, controller in controllers.items()
        }
        route_residency_snapshots = {}
        execution_results = queue.Queue()

        def execute(command, projection_ready=None):
            nonlocal active_model_id, batch_id, projected_residencies
            try:
                request = by_id[int(command["request_id"])]
                dispatch_s = time.perf_counter() - command["replay_start"]
                dispatch_wall = time.time()
                switched = active_model_id != request.model_id
                allocator_trim_ms = 0.0
                if switched or options["allocator_trim_every_request"]:
                    trim_started = time.perf_counter()
                    torch.cuda.empty_cache()
                    allocator_trim_ms = (
                        time.perf_counter() - trim_started) * 1000.0
                    active_model_id = request.model_id

                llm = engines[request.model_id]
                worker = single.get_worker(llm)
                controller = controllers[request.model_id]
                controller.reset_metrics()
                odkv_before = worker.odkv_stats()
                policy_decision = None
                with state_lock:
                    if cache_policy is not None:
                        route_residencies = (
                            route_residency_snapshots.pop(
                                request.request_id, None)
                            if options["scheduler_mode"]
                            == "serial-choice" else None
                        )
                        pending = [
                            by_id[int(request_id)]
                            for request_id in command.get("pending_ids", [])
                        ]
                        policy_decision = cache_policy.plan_dispatch(
                            active_model_id=request.model_id,
                            batch=[request],
                            pending=pending,
                            kv_block_size_tokens=int(
                                worker.cache_config.block_size),
                            kv_block_size_bytes=int(
                                odkv_before["block_size_bytes"]),
                            current_kv_pages=0,
                            residencies=route_residencies,
                        )
                    # Freeze a CPU-only projection for estimates received while
                    # this forward is in flight.  Reclamation has already run;
                    # the active model will be fully resident by forward end.
                    projected_residencies = {
                        model_id: {
                            page for page, resident in enumerate(
                                item.pool.layerweave_residency(
                                    item.model_path))
                            if resident
                        }
                        for model_id, item in controllers.items()
                    }
                    projected_residencies[request.model_id] = set(
                        controller.required_pages)
                if projection_ready is not None:
                    projection_ready.set()

                prompts = [single.make_prompt_tokens(
                    request,
                    int(worker.model_config.get_vocab_size()),
                    args.seed,
                )]
                sampling = [SamplingParams(
                    temperature=0,
                    max_tokens=request.output_tokens,
                    ignore_eos=True,
                    detokenize=False,
                )]
                outputs = llm.generate(
                    prompt_token_ids=prompts,
                    sampling_params=sampling,
                    use_tqdm=False,
                )
                layerweave = controller.metrics()
                prefix_cache = (
                    policy_decision["targets"][str(request.model_id)]
                    if policy_decision is not None
                    else controller.apply_prefix_cache()
                )
                with state_lock:
                    projected_residencies = {
                        model_id: {
                            page for page, resident in enumerate(
                                item.pool.layerweave_residency(
                                    item.model_path))
                            if resident
                        }
                        for model_id, item in controllers.items()
                    }
                odkv_after = worker.odkv_stats()
                output = outputs[0]
                metrics = output.metrics
                scheduled = (
                    metrics.first_scheduled_time
                    if metrics.first_scheduled_time is not None
                    else metrics.arrival_time
                )
                first_token_s = (
                    dispatch_s + metrics.first_token_time - dispatch_wall)
                finish_s = (
                    dispatch_s + metrics.finished_time - dispatch_wall)
                result = RequestResult(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    arrival_s=request.arrival_s,
                    dispatch_s=dispatch_s,
                    first_token_s=first_token_s,
                    finish_s=finish_s,
                    input_tokens=request.input_tokens,
                    output_tokens=request.output_tokens,
                    batch_id=batch_id,
                    batch_size=1,
                    trace_input_tokens=request.trace_input_tokens,
                    trace_output_tokens=request.trace_output_tokens,
                ).record()
                odkv_delta = {
                    key: odkv_after.get(key, 0) - odkv_before.get(key, 0)
                    for key in (
                        "allocate_calls", "release_calls",
                        "allocated_blocks", "released_blocks",
                        "allocate_ms", "release_ms",
                    )
                }
                route = command.get("routing") or {}
                predicted_exposed_load_ms = route.get(
                    "chosen_predicted_exposed_load_ms")
                measured_exposed_load_ms = float(
                    layerweave["exposed_load_ms"])
                batch_metric = {
                    "batch_id": batch_id,
                    "request_id": request.request_id,
                    "device": device,
                    "model_id": request.model_id,
                    "batch_size": 1,
                    "input_tokens": request.input_tokens,
                    "max_input_tokens": request.input_tokens,
                    "sum_input_tokens_squared": request.input_tokens ** 2,
                    "output_tokens": request.output_tokens,
                    "cold_model_load": switched,
                    "weight_load_ms": layerweave["h2d_ms"],
                    "allocator_trim_ms": allocator_trim_ms,
                    "prefill_ms": (
                        metrics.first_token_time - scheduled) * 1000.0,
                    "service_ttft_ms": result["service_ttft_ms"],
                    "ttft_ms": result["ttft_ms"],
                    "odkv": odkv_delta,
                    "layerweave": layerweave,
                    "prefix_cache": prefix_cache,
                    "cache_policy": policy_decision,
                    "routing": route,
                    "predicted_exposed_load_ms": (
                        None if predicted_exposed_load_ms is None else
                        float(predicted_exposed_load_ms)
                    ),
                    "measured_exposed_load_ms": measured_exposed_load_ms,
                    "pse_exposed_error_ms": (
                        None if predicted_exposed_load_ms is None else
                        float(predicted_exposed_load_ms)
                        - measured_exposed_load_ms
                    ),
                    "assigned_wait_ms": max(
                        0.0,
                        (
                            dispatch_s
                            - float(route.get("assignment_s", dispatch_s))
                        ) * 1000.0,
                    ),
                    "route_prediction_error_ms": (
                        result["service_ttft_ms"]
                        - float(route.get(
                            "chosen_predicted_service_ms",
                            result["service_ttft_ms"]))
                    ),
                }
                execution_results.put({
                    "type": "complete",
                    "device": device,
                    "request": result,
                    "batch_metric": batch_metric,
                })
                batch_id += 1
            except BaseException as exc:
                if projection_ready is not None:
                    projection_ready.set()
                execution_results.put({
                    "type": "error",
                    "device": device,
                    "request_id": command.get("request_id"),
                    "model_id": (
                        request.model_id
                        if "request" in locals() else None),
                    "input_tokens": (
                        request.input_tokens
                        if "request" in locals() else None),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })

        running = None
        assigned = deque()
        events.put({
            "type": "ready",
            "device": device,
            "startup_ms": startup,
            "engine_limits": engine_limits,
            "joint_pool_input_limits": joint_pool_input_limits,
            "vmm_policy": pool.vmm_policy,
        })

        while True:
            if running is not None and not running.is_alive():
                running.join()
                running = None
                event = execution_results.get()
                if event["type"] == "error":
                    events.put(event)
                    return
                if assigned:
                    next_command = assigned.popleft()
                    projection_ready = threading.Event()
                    running = threading.Thread(
                        target=execute,
                        args=(next_command, projection_ready),
                        daemon=True,
                    )
                    running.start()
                    projection_ready.wait()
                events.put(event)
            try:
                command = commands.get(timeout=0.01)
            except queue.Empty:
                continue
            kind = command["type"]
            if kind == "stop":
                if running is not None or assigned:
                    raise RuntimeError(
                        "Coordinator stopped a worker with queued work")
                final_odkv = {
                    str(model_id): single.get_worker(engine).odkv_stats()
                    for model_id, engine in engines.items()
                }
                events.put({
                    "type": "stopped",
                    "device": device,
                    "odkv": final_odkv,
                })
                return

            request = by_id[int(command["request_id"])]
            controller = controllers[request.model_id]
            if kind == "discard_estimate":
                route_residency_snapshots.pop(
                    request.request_id, None)
                continue
            if kind == "estimate":
                with state_lock:
                    snapshot = {
                        model_id: set(pages)
                        for model_id, pages
                        in projected_residencies.items()
                    }
                    missing_pages = len(
                        set(controller.required_pages)
                        - snapshot[request.model_id])
                    missing_bytes = missing_pages * controller.page_size
                    estimate = {
                        "required_pages": len(controller.required_pages),
                        "missing_pages": missing_pages,
                        "cached_pages": (
                            len(controller.required_pages) - missing_pages),
                        "missing_bytes": missing_bytes,
                        "estimated_load_ms": (
                            missing_bytes
                            / options["pcie_bandwidth_gbps"]
                            / 1e9 * 1000.0
                        ),
                    }
                    if cache_policy is not None:
                        estimate_started = time.perf_counter()
                        pages_per_block = math.ceil(
                            route_kv[request.model_id][
                                "block_size_bytes"]
                            / controller.page_size)
                        kv_blocks = math.ceil(
                            (request.input_tokens + request.output_tokens)
                            / route_kv[request.model_id][
                                "block_size_tokens"])
                        pending = [
                            by_id[int(request_id)]
                            for request_id
                            in command.get("pending_ids", [])
                        ]
                        estimate.update(cache_policy.estimate_dispatch(
                            active_model_id=request.model_id,
                            batch=[request],
                            pending=pending,
                            request_kv_pages=(
                                kv_blocks * pages_per_block),
                            current_kv_pages=0,
                            residencies=snapshot,
                        ))
                        if options["scheduler_mode"] == "serial-choice":
                            route_residency_snapshots[
                                request.request_id] = {
                                    model_id: set(pages)
                                    for model_id, pages in snapshot.items()
                                }
                        estimate["estimate_cpu_ms"] = (
                            time.perf_counter() - estimate_started
                        ) * 1000.0
                events.put({
                    "type": "estimate",
                    "device": device,
                    "request_id": request.request_id,
                    "estimate": estimate,
                })
                continue
            if kind != "execute":
                raise RuntimeError(f"Unknown worker command: {kind}")
            if running is None:
                running = threading.Thread(
                    target=execute, args=(command,), daemon=True)
                running.start()
            else:
                if len(assigned) >= options["max_assigned_queue_per_gpu"]:
                    raise RuntimeError(
                        f"GPU {device} assigned queue overflow")
                assigned.append(command)
    except BaseException as exc:
        events.put({
            "type": "error",
            "device": device,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })


def _summarize(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0}

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return values[lower]
        return (
            values[lower] * (upper - index)
            + values[upper] * (index - lower)
        )

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": values[-1],
    }


def _route_scores(
    system_policy,
    candidates,
    estimates,
    queue_estimates,
    model_id,
    service_ewma,
    default_service_ms,
    transition_weight,
    transition_credit_cap_ms=100.0,
    transition_penalty_cap_ms=100.0,
):
    predicted_services = {}
    scores = {}
    for device in candidates:
        estimate = estimates[device]
        if system_policy == "minimal":
            compute_estimate = service_ewma[device].get(
                model_id, default_service_ms)
            predicted_services[device] = max(
                estimate["estimated_load_ms"], compute_estimate)
            route_cost = estimate["estimated_load_ms"]
        else:
            predicted_services[device] = estimate[
                "predicted_service_ttft_ms"]
            weighted_transition = (
                transition_weight * estimate["transition_cost_ms"])
            # A configuration change may have negative transition cost when it
            # improves future cache value. Bound that credit so it cannot make
            # a current request's latency score arbitrarily negative.
            weighted_transition = max(
                -transition_credit_cap_ms, weighted_transition)
            weighted_transition = min(
                transition_penalty_cap_ms, weighted_transition)
            route_cost = (
                predicted_services[device]
                + weighted_transition
            )
        scores[device] = queue_estimates[device] + route_cost
    return scores, predicted_services


def _choose_immediate_pair(
    pending_requests,
    devices,
    free_devices,
    scores_by_request,
    queue_estimates,
    now_s,
    max_placement_wait_ms,
    placement_hysteresis_ms,
    first_deferred_s,
):
    """Choose a request/free-GPU pair or defer for a better busy placement."""
    immediate = []
    deferred = []
    for request in pending_requests:
        request_id = int(request["request_id"])
        scores = scores_by_request[request_id]
        preferred = min(devices, key=lambda device: (
            scores[device], device))
        best_free = min(free_devices, key=lambda device: (
            scores[device], device))
        busy_advantage_ms = scores[best_free] - scores[preferred]
        deferred_for_ms = max(
            0.0,
            (now_s - first_deferred_s.get(request_id, now_s)) * 1000.0,
        )
        should_defer = (
            preferred not in free_devices
            and queue_estimates[preferred] <= max_placement_wait_ms
            and busy_advantage_ms >= placement_hysteresis_ms
            and deferred_for_ms < max_placement_wait_ms
        )
        if should_defer:
            deferred.append({
                "request_id": request_id,
                "preferred_device": preferred,
                "best_free_device": best_free,
                "busy_advantage_ms": busy_advantage_ms,
            })
            continue
        immediate.append((
            scores[best_free],
            float(request["arrival_s"]),
            request_id,
            best_free,
            request,
            preferred,
        ))
    if not immediate:
        return None, deferred
    _, _, _, chosen, request, preferred = min(immediate)
    return {
        "request": request,
        "chosen_device": chosen,
        "preferred_device": preferred,
    }, deferred


def _cold_tie_choice(
    system_policy,
    scheduler_mode,
    minimal_enabled,
    joint_enabled,
    estimates,
    devices,
    rng,
):
    enabled = (
        minimal_enabled if system_policy == "minimal"
        else joint_enabled if system_policy == "joint"
        else False
    )
    if not (scheduler_mode == "serial-choice" and enabled):
        return None
    if not all(
        int(estimates[device]["cached_pages"]) == 0
        for device in devices
    ):
        return None
    return rng.choice(sorted(devices))


def _parse_requests(args):
    # Coordinator intentionally avoids importing torch-bearing benchmark code.
    requests = []
    with args.trace.open() as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            requests.append({
                "request_id": len(requests),
                "arrival_s": float(fields[0]),
                "model_id": int(fields[1]),
                "input_tokens": max(
                    1, int(math.floor(
                        int(fields[2]) * args.input_scale + 0.5))),
                "output_tokens": (
                    args.output_tokens_override
                    if args.output_tokens_override > 0 else int(fields[3])),
                "trace_input_tokens": int(fields[2]),
                "trace_output_tokens": int(fields[3]),
            })
            if args.max_requests > 0 and len(requests) >= args.max_requests:
                break
    if not requests:
        raise ValueError(f"No requests found in {args.trace}")
    base = requests[0]["arrival_s"]
    for request in requests:
        request["arrival_s"] = (
            request["arrival_s"] - base) * args.trace_time_scale
    return requests


def run(args) -> dict:
    requests = _parse_requests(args)
    routing_replay = {}
    if args.routing_replay is not None:
        replay_document = json.loads(args.routing_replay.read_text())
        routing_replay = {
            int(item["request_id"]): int(item["chosen_device"])
            for item in replay_document["routing"]
        }
        request_ids = {int(item["request_id"]) for item in requests}
        if set(routing_replay) != request_ids:
            raise ValueError(
                "Routing replay request ids do not match this trace slice")
        invalid_devices = (
            set(routing_replay.values()) - set(args.devices))
        if invalid_devices:
            raise ValueError(
                f"Routing replay names unavailable GPUs: "
                f"{sorted(invalid_devices)}")
    original_trace_span_s = (
        requests[-1]["arrival_s"] - requests[0]["arrival_s"])
    if args.scheduler_mode == "serial-choice":
        # Arrival time is intentionally outside this idealized placement
        # experiment. All requests are available at replay start, while the
        # coordinator permits only one system-wide execution at a time.
        for request in requests:
            request["arrival_s"] = 0.0
    options = {
        "system_policy": args.system_policy,
        "profile": str(args.profile.resolve()),
        "demand_decay": args.demand_decay,
        "uncertainty_ms": args.uncertainty_ms,
        "cache_policy_mode": args.cache_policy_mode,
        "pcie_bandwidth_gbps": args.pcie_bandwidth_gbps,
        "max_assigned_queue_per_gpu": args.max_assigned_queue_per_gpu,
        "scheduler_mode": args.scheduler_mode,
        "allocator_trim_every_request":
            args.allocator_trim_every_request,
        "requests": requests,
        "single_args": {
            "config": args.config.resolve(),
            "model_path": [],
            "load_mode": "layerweave",
            "max_batch_size": 1,
            "input_scale": args.input_scale,
            "output_tokens_override": args.output_tokens_override,
            "vmm_pool_gib": args.vmm_pool_gib,
            "vmm_page_size_mib": args.vmm_page_size_mib,
            "dtype": "float16",
            "gpu_memory_utilization": 0.05,
            "max_model_len": 0,
            "truncate_input_to_model_limit": True,
            "seed": args.seed,
            "trust_remote_code": False,
        },
    }
    context = mp.get_context("spawn")
    events = context.Queue()
    command_queues = {}
    processes = {}
    for index, device in enumerate(args.devices):
        command_queues[device] = context.Queue()
        process = context.Process(
            target=_worker_main,
            args=(device, options, command_queues[device], events),
        )
        process.start()
        processes[device] = process
        if index + 1 < len(args.devices):
            time.sleep(args.startup_stagger_seconds)

    def abort_workers():
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        for process in processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)

    def raise_worker_error(event):
        context = (
            f"GPU {event.get('device')} request "
            f"{event.get('request_id')} model {event.get('model_id')} "
            f"input_tokens={event.get('input_tokens')} failed")
        abort_workers()
        raise RuntimeError(f"{context}:\n{event['traceback']}")

    ready = {}
    while len(ready) < len(args.devices):
        event = events.get()
        if event["type"] == "error":
            raise_worker_error(event)
        if event["type"] == "ready":
            ready[event["device"]] = event

    replay_start = time.perf_counter()
    pending = (
        deque(requests)
        if args.scheduler_mode == "serial-choice" else deque()
    )
    next_index = (
        len(requests) if args.scheduler_mode == "serial-choice" else 0
    )
    assigned_count = {device: 0 for device in args.devices}
    predicted_available_at = {device: 0.0 for device in args.devices}
    service_ewma_ms = {
        device: {} for device in args.devices
    }
    records = []
    batches = []
    routing_records = []
    scheduler_rng = random.Random(args.seed)
    first_deferred_s = {}
    deferred_request_ids = set()

    def handle_complete(event):
        device = event["device"]
        assigned_count[device] -= 1
        if assigned_count[device] < 0:
            raise RuntimeError(f"GPU {device} completion underflow")
        records.append(event["request"])
        batches.append(event["batch_metric"])
        metric = event["batch_metric"]
        model_id = int(metric["model_id"])
        actual = float(metric["service_ttft_ms"])
        previous = service_ewma_ms[device].get(model_id)
        service_ewma_ms[device][model_id] = (
            actual if previous is None else 0.8 * previous + 0.2 * actual)
        error_ms = float(metric["route_prediction_error_ms"])
        predicted_available_at[device] += error_ms / 1000.0
        if assigned_count[device] == 0:
            predicted_available_at[device] = (
                time.perf_counter() - replay_start)

    def dispatch_one():
        nonlocal pending
        if (
            args.scheduler_mode == "serial-choice"
            and any(assigned_count.values())
        ):
            return False
        free_devices = sorted(
            device for device in args.devices
            if assigned_count[device] == 0
        )
        if not free_devices:
            return False
        if args.scheduler_mode == "serial-choice":
            if len(free_devices) != len(args.devices):
                raise RuntimeError(
                    "serial-choice placement requires both GPUs idle")
            considered = [pending[0]]
        else:
            considered = list(pending)[:args.routing_lookahead]
        pending_ids = [item["request_id"] for item in pending]
        selection = None
        selected_estimates = None
        selected_scores = None
        selected_predicted = None
        selected_now = None
        for request in considered:
            request_id = int(request["request_id"])
            estimates = {}
            estimate_pending_ids = [
                item for item in pending_ids if int(item) != request_id
            ]
            for device in args.devices:
                device_pending_ids = (
                    [
                        pending_id
                        for pending_id in estimate_pending_ids
                        if routing_replay[int(pending_id)] == device
                    ]
                    if routing_replay else estimate_pending_ids
                )
                command_queues[device].put({
                    "type": "estimate",
                    "request_id": request["request_id"],
                    "pending_ids": device_pending_ids,
                })
            while len(estimates) < len(args.devices):
                event = events.get()
                if event["type"] == "error":
                    raise_worker_error(event)
                if event["type"] == "complete":
                    handle_complete(event)
                    continue
                if event["type"] != "estimate":
                    raise RuntimeError(
                        f"Unexpected event during route estimate: {event}")
                if int(event["request_id"]) != request_id:
                    raise RuntimeError(
                        f"Estimate request mismatch: {event}")
                estimates[event["device"]] = event["estimate"]
            now = time.perf_counter() - replay_start
            queue_estimates = {
                device: max(
                    0.0, predicted_available_at[device] - now) * 1000.0
                for device in args.devices
            }
            # A completion may arrive while estimates are collected.
            free_devices = sorted(
                device for device in args.devices
                if assigned_count[device] == 0
            )
            scores, predicted = _route_scores(
                args.system_policy,
                args.devices,
                estimates,
                queue_estimates,
                request["model_id"],
                service_ewma_ms,
                args.default_service_ms,
                args.transition_weight,
                args.transition_credit_cap_ms,
                args.transition_penalty_cap_ms,
            )
            cold_tie_chosen = _cold_tie_choice(
                args.system_policy,
                args.scheduler_mode,
                args.minimal_cold_tie_random,
                args.joint_cold_tie_random,
                estimates,
                args.devices,
                scheduler_rng,
            )
            if cold_tie_chosen is not None:
                # Preserve normal selection and routing records while making
                # the seeded random cold-start decision the unique minimum.
                scores = dict(scores)
                scores[cold_tie_chosen] = min(scores.values()) - 1e-9
            if routing_replay:
                replay_device = routing_replay[request_id]
                scores = dict(scores)
                scores[replay_device] = min(scores.values()) - 1e-6
            selection, deferred = _choose_immediate_pair(
                [request],
                args.devices,
                free_devices,
                {request_id: scores},
                queue_estimates,
                now,
                args.max_placement_wait_ms,
                args.placement_hysteresis_ms,
                first_deferred_s,
            )
            if deferred:
                first_deferred_s.setdefault(request_id, now)
                deferred_request_ids.add(request_id)
                continue
            selected_estimates = estimates
            selected_scores = scores
            selected_predicted = predicted
            selected_now = now
            break
        if selection is None:
            return False
        request = selection["request"]
        chosen = selection["chosen_device"]
        request_id = int(request["request_id"])
        scores = selected_scores
        predicted_services = selected_predicted
        estimates = selected_estimates
        now = selected_now
        pending.remove(request)
        affinity_wait_ms = max(
            0.0,
            (now - first_deferred_s.pop(request_id, now)) * 1000.0,
        )
        routing = {
            "request_id": request_id,
            "chosen_device": chosen,
            "preferred_device": selection["preferred_device"],
            "candidate_scores": scores,
            "candidate_queue_estimate_ms": queue_estimates,
            "candidate_predicted_service_ms": predicted_services,
            "candidate_estimates": estimates,
            "policy": args.system_policy,
            "chosen_queue_estimate_ms": queue_estimates[chosen],
            "chosen_predicted_service_ms": predicted_services[chosen],
            "candidate_predicted_exposed_load_ms": {
                device: estimate.get("predicted_exposed_load_ms")
                for device, estimate in estimates.items()
            },
            "chosen_predicted_exposed_load_ms": estimates[chosen].get(
                "predicted_exposed_load_ms"),
            "queue_depth_at_assignment": assigned_count[chosen],
            "both_gpus_idle_at_assignment": (
                len(free_devices) == len(args.devices)),
            "assignment_s": now,
            "affinity_placement_wait_ms": affinity_wait_ms,
            "was_deferred_for_affinity": request_id in deferred_request_ids,
            "minimal_cold_tie_randomized": (
                cold_tie_chosen is not None
                and args.system_policy == "minimal"
            ),
            "cold_tie_randomized": cold_tie_chosen is not None,
            "routing_replayed": bool(routing_replay),
        }
        routing_records.append(routing)
        predicted_available_at[chosen] = (
            max(now, predicted_available_at[chosen])
            + predicted_services[chosen] / 1000.0
        )
        assigned_count[chosen] += 1
        if args.scheduler_mode == "serial-choice":
            for device in args.devices:
                if device != chosen:
                    command_queues[device].put({
                        "type": "discard_estimate",
                        "request_id": request["request_id"],
                    })
        command_queues[chosen].put({
            "type": "execute",
            "request_id": request["request_id"],
            "pending_ids": [
                item["request_id"]
                for item in pending
                if (
                    not routing_replay
                    or routing_replay[int(item["request_id"])] == chosen
                )
            ],
            "replay_start": replay_start,
            "routing": routing,
        })
        return True

    while (
        next_index < len(requests)
        or pending
        or any(assigned_count.values())
    ):
        now = time.perf_counter() - replay_start
        if args.scheduler_mode != "serial-choice":
            while (
                next_index < len(requests)
                and requests[next_index]["arrival_s"] <= now
            ):
                pending.append(requests[next_index])
                next_index += 1
        while pending and dispatch_one():
            pass
        if (
            args.scheduler_mode != "serial-choice"
            and
            not any(assigned_count.values())
            and not pending
            and next_index < len(requests)
        ):
            delay = requests[next_index]["arrival_s"] - (
                time.perf_counter() - replay_start)
            if delay > 0:
                time.sleep(delay)
            continue
        if any(assigned_count.values()):
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event["type"] == "error":
                raise_worker_error(event)
            if event["type"] != "complete":
                raise RuntimeError(f"Unexpected worker event: {event}")
            handle_complete(event)

    for device in args.devices:
        command_queues[device].put({"type": "stop"})
    stopped = {}
    while len(stopped) < len(args.devices):
        event = events.get()
        if event["type"] == "error":
            raise_worker_error(event)
        if event["type"] == "stopped":
            stopped[event["device"]] = event
    for process in processes.values():
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"LayerWeave worker exited with code {process.exitcode}")

    records.sort(key=lambda item: item["request_id"])
    batches.sort(key=lambda item: item["request_id"])
    makespan_s = max(
        (item["finish_s"] for item in records), default=0.0)
    return {
        "backend": "layerweave_odkv_multi_gpu",
        "system_policy": args.system_policy,
        "scheduler_mode": args.scheduler_mode,
        "arrival_time_ignored": args.scheduler_mode == "serial-choice",
        "original_scaled_trace_span_s": original_trace_span_s,
        "throughput_semantics": (
            "system-wide serialized diagnostic throughput"
            if args.scheduler_mode == "serial-choice"
            else "online concurrent trace throughput"
        ),
        "devices": args.devices,
        "trace": str(args.trace.resolve()),
        "config": str(args.config.resolve()),
        "trace_time_scale": args.trace_time_scale,
        "input_scale": args.input_scale,
        "output_tokens_override": args.output_tokens_override,
        "vmm_pool_gib_per_gpu": args.vmm_pool_gib,
        "vmm_page_size_mib": args.vmm_page_size_mib,
        "max_assigned_queue_per_gpu": args.max_assigned_queue_per_gpu,
        "routing_lookahead": args.routing_lookahead,
        "max_placement_wait_ms": args.max_placement_wait_ms,
        "placement_hysteresis_ms": args.placement_hysteresis_ms,
        "transition_weight": args.transition_weight,
        "transition_credit_cap_ms": args.transition_credit_cap_ms,
        "transition_penalty_cap_ms": args.transition_penalty_cap_ms,
        "minimal_cold_tie_random": args.minimal_cold_tie_random,
        "joint_cold_tie_random": args.joint_cold_tie_random,
        "allocator_trim_every_request":
            args.allocator_trim_every_request,
        "cache_policy_mode": args.cache_policy_mode,
        "batch_unmap": os.environ.get(
            "LAYERWEAVE_BATCH_UNMAP", "0") != "0",
        "routing_replay": (
            str(args.routing_replay.resolve())
            if args.routing_replay is not None else None
        ),
        "summary": {
            "requests": len(records),
            "batches": len(batches),
            "makespan_s": makespan_s,
            "throughput_requests_s": (
                len(records) / makespan_s if makespan_s > 0 else 0.0),
            "ttft_ms": _summarize(
                item["ttft_ms"] for item in records),
            "service_ttft_ms": _summarize(
                item["service_ttft_ms"] for item in records),
            "queue_ms": _summarize(
                item["queue_ms"] for item in records),
            "e2e_ms": _summarize(
                item["e2e_ms"] for item in records),
            "weight_load_ms": _summarize(
                item["weight_load_ms"] for item in batches),
            "prefill_ms": _summarize(
                item["prefill_ms"] for item in batches),
            "requests_by_device": {
                str(device): sum(
                    item["device"] == device for item in batches)
                for device in args.devices
            },
            "routed_to_busy_gpu": sum(
                int(item["queue_depth_at_assignment"] > 0)
                for item in routing_records
            ),
            "requests_deferred_for_affinity": sum(
                int(item["was_deferred_for_affinity"])
                for item in routing_records
            ),
            "affinity_placement_wait_ms": _summarize(
                item["affinity_placement_wait_ms"]
                for item in routing_records
            ),
            "two_gpu_route_decisions": sum(
                int(len(item["candidate_estimates"]) == len(args.devices))
                for item in routing_records
            ),
            "both_gpus_idle_route_decisions": sum(
                int(item["both_gpus_idle_at_assignment"])
                for item in routing_records
            ),
            "minimal_cold_tie_random_decisions": sum(
                int(item["minimal_cold_tie_randomized"])
                for item in routing_records
            ),
            "cold_tie_random_decisions": sum(
                int(item["cold_tie_randomized"])
                for item in routing_records
            ),
            "route_prediction_error_ms": _summarize(
                item["route_prediction_error_ms"] for item in batches),
            "pse_exposed_error_ms": _summarize(
                item["pse_exposed_error_ms"]
                for item in batches
                if item.get("pse_exposed_error_ms") is not None
            ),
        },
        "worker_startup": ready,
        "requests": records,
        "batch_metrics": batches,
        "routing": routing_records,
        "worker_odkv": stopped,
    }


def write_outputs(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(result, stream, indent=2)
    csv_path = output.with_suffix(".requests.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=result["requests"][0].keys())
        writer.writeheader()
        writer.writerows(result["requests"])
    print("LAYERWEAVE_MULTI_GPU_SUMMARY=" + json.dumps(
        result["summary"], sort_keys=True))
    print(f"LAYERWEAVE_MULTI_GPU_RESULT={output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument(
        "--system-policy", choices=("minimal", "joint"), required=True)
    parser.add_argument(
        "--scheduler-mode",
        choices=("online", "serial-choice"),
        default="online",
        help=(
            "online replays arrivals concurrently; serial-choice ignores "
            "arrivals and completes one request before choosing between two "
            "idle GPUs for the next request."
        ),
    )
    parser.add_argument(
        "--minimal-cold-tie-random",
        action="store_true",
        help=(
            "In serial-choice minimal mode only, use the seeded RNG when "
            "neither idle GPU caches any page of the requested model."
        ),
    )
    parser.add_argument(
        "--joint-cold-tie-random",
        action="store_true",
        help=(
            "In serial-choice joint mode only, use the seeded RNG when "
            "neither idle GPU caches any page of the requested model."
        ),
    )
    parser.add_argument(
        "--allocator-trim-every-request",
        action="store_true",
        help=(
            "Call torch.cuda.empty_cache before every request on both "
            "policies; useful for serial diagnostics with long same-model "
            "request sequences."
        ),
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--trace-time-scale", type=float, default=4.0)
    parser.add_argument("--input-scale", type=float, default=4.0)
    parser.add_argument("--output-tokens-override", type=int, default=1)
    parser.add_argument("--vmm-pool-gib", type=float, default=42.0)
    parser.add_argument("--vmm-page-size-mib", type=int, default=64)
    parser.add_argument("--pcie-bandwidth-gbps", type=float, default=25.0)
    parser.add_argument("--transition-weight", type=float, default=0.1)
    parser.add_argument(
        "--transition-credit-cap-ms", type=float, default=100.0,
        help="Maximum latency credit from a negative transition cost.")
    parser.add_argument(
        "--transition-penalty-cap-ms", type=float, default=100.0,
        help="Maximum latency penalty from a positive transition cost.")
    parser.add_argument(
        "--max-assigned-queue-per-gpu", type=int, default=0,
        help="Worker safety limit; global-pending routing does not pre-assign.")
    parser.add_argument(
        "--routing-lookahead", type=int, default=8,
        help="Oldest global-pending requests considered at each placement.")
    parser.add_argument(
        "--max-placement-wait-ms", type=float, default=150.0,
        help="Maximum affinity wait before dispatching to a free GPU.")
    parser.add_argument(
        "--placement-hysteresis-ms", type=float, default=25.0,
        help="Required predicted benefit before waiting for a busy GPU.")
    parser.add_argument(
        "--default-service-ms", type=float, default=750.0,
        help="Initial minimal-policy busy-until estimate before EWMA exists.")
    parser.add_argument("--demand-decay", type=float, default=0.9)
    parser.add_argument("--uncertainty-ms", type=float, default=100.0)
    parser.add_argument(
        "--cache-policy-mode",
        choices=("demand", "next-use"),
        default="demand",
        help=(
            "demand is the deployable online policy; next-use is a "
            "serial-choice-only clairvoyant cache upper-bound experiment."
        ),
    )
    parser.add_argument(
        "--routing-replay",
        type=Path,
        help=(
            "Replay request-to-GPU choices from a prior result JSON; useful "
            "for a cache-only serial-choice ablation."
        ),
    )
    parser.add_argument("--startup-stagger-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.devices = [int(value) for value in args.devices.split(",")]
    if len(args.devices) != 2 or len(set(args.devices)) != 2:
        parser.error("--devices must name two distinct physical GPUs")
    if args.max_requests < 0:
        parser.error("--max-requests must be non-negative")
    if args.max_assigned_queue_per_gpu < 0:
        parser.error("--max-assigned-queue-per-gpu must be non-negative")
    if args.routing_lookahead <= 0:
        parser.error("--routing-lookahead must be positive")
    if args.max_placement_wait_ms < 0:
        parser.error("--max-placement-wait-ms must be non-negative")
    if args.placement_hysteresis_ms < 0:
        parser.error("--placement-hysteresis-ms must be non-negative")
    if args.transition_credit_cap_ms < 0:
        parser.error("--transition-credit-cap-ms must be non-negative")
    if args.transition_penalty_cap_ms < 0:
        parser.error("--transition-penalty-cap-ms must be non-negative")
    if args.default_service_ms <= 0:
        parser.error("--default-service-ms must be positive")
    if (
        args.cache_policy_mode == "next-use"
        and (
            args.system_policy != "joint"
            or args.scheduler_mode != "serial-choice"
        )
    ):
        parser.error(
            "--cache-policy-mode next-use requires "
            "--system-policy joint --scheduler-mode serial-choice")
    if (
        args.routing_replay is not None
        and args.scheduler_mode != "serial-choice"
    ):
        parser.error("--routing-replay requires serial-choice mode")
    if args.trace_time_scale <= 0 or args.input_scale <= 0:
        parser.error("trace/input scale must be positive")
    result = run(args)
    write_outputs(result, args.output)


if __name__ == "__main__":
    main()
