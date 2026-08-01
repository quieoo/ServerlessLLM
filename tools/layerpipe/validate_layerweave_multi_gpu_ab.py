#!/usr/bin/env python3
"""Validate fairness and safety of the two-GPU LayerWeave A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _workload(document):
    return [
        (
            item["request_id"],
            item["model_id"],
            item["trace_input_tokens"],
            item["input_tokens"],
            item["trace_output_tokens"],
            item["output_tokens"],
        )
        for item in document["requests"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    args = parser.parse_args()
    minimal = json.loads(args.minimal.read_text())
    joint = json.loads(args.joint.read_text())
    failures = []

    common_fields = (
        "devices",
        "scheduler_mode",
        "arrival_time_ignored",
        "trace",
        "config",
        "trace_time_scale",
        "input_scale",
        "output_tokens_override",
        "vmm_pool_gib_per_gpu",
        "vmm_page_size_mib",
        "max_assigned_queue_per_gpu",
        "routing_lookahead",
        "max_placement_wait_ms",
        "placement_hysteresis_ms",
        "transition_weight",
        "transition_credit_cap_ms",
        "transition_penalty_cap_ms",
        "minimal_cold_tie_random",
        "joint_cold_tie_random",
        "allocator_trim_every_request",
    )
    for field in common_fields:
        if minimal.get(field) != joint.get(field):
            failures.append(
                f"{field} differs: minimal={minimal.get(field)!r}, "
                f"joint={joint.get(field)!r}")
    if minimal.get("system_policy") != "minimal":
        failures.append("minimal result did not enable minimal policy")
    if joint.get("system_policy") != "joint":
        failures.append("joint result did not enable joint policy")
    if _workload(minimal) != _workload(joint):
        failures.append("processed request workload differs")
    for label, document in (("minimal", minimal), ("joint", joint)):
        request_ids = [item["request_id"] for item in document["requests"]]
        if len(request_ids) != len(set(request_ids)):
            failures.append(f"{label} executed duplicate requests")
        if set(request_ids) != {
                item["request_id"] for item in document["routing"]}:
            failures.append(f"{label} routing/request IDs do not match")
        assigned = sum(
            document["summary"]["requests_by_device"].values())
        if assigned != document["summary"]["requests"]:
            failures.append(f"{label} device assignment count is incomplete")
        if document.get("scheduler_mode") == "serial-choice":
            both_idle = document["summary"].get(
                "both_gpus_idle_route_decisions", 0)
            if both_idle != document["summary"]["requests"]:
                failures.append(
                    f"{label} serial-choice did not route every request "
                    f"with both GPUs idle: {both_idle}/"
                    f"{document['summary']['requests']}")
    evicted_protected = sum(
        int((item.get("cache_policy") or {}).get(
            "evicted_protected_pages", 0))
        for item in joint["batch_metrics"]
    )
    if evicted_protected:
        failures.append(
            f"joint evicted {evicted_protected} protected pages")

    minimal_mean = minimal["summary"]["service_ttft_ms"]["mean"]
    joint_mean = joint["summary"]["service_ttft_ms"]["mean"]
    report = {
        "validation": "PASS" if not failures else "FAIL",
        "failures": failures,
        "requests": minimal["summary"]["requests"],
        "minimal_service_ttft_mean_ms": minimal_mean,
        "joint_service_ttft_mean_ms": joint_mean,
        "service_ttft_gain_ms": minimal_mean - joint_mean,
        "minimal_throughput_requests_s":
            minimal["summary"]["throughput_requests_s"],
        "joint_throughput_requests_s":
            joint["summary"]["throughput_requests_s"],
        "evicted_protected_pages": evicted_protected,
        "minimal_routed_to_busy_gpu":
            minimal["summary"].get("routed_to_busy_gpu", 0),
        "joint_routed_to_busy_gpu":
            joint["summary"].get("routed_to_busy_gpu", 0),
        "minimal_two_gpu_route_decisions":
            minimal["summary"].get("two_gpu_route_decisions", 0),
        "joint_two_gpu_route_decisions":
            joint["summary"].get("two_gpu_route_decisions", 0),
        "minimal_both_gpus_idle_route_decisions":
            minimal["summary"].get("both_gpus_idle_route_decisions", 0),
        "joint_both_gpus_idle_route_decisions":
            joint["summary"].get("both_gpus_idle_route_decisions", 0),
        "minimal_requests_deferred_for_affinity":
            minimal["summary"].get("requests_deferred_for_affinity", 0),
        "joint_requests_deferred_for_affinity":
            joint["summary"].get("requests_deferred_for_affinity", 0),
        "minimal_cold_tie_random_decisions":
            minimal["summary"].get(
                "minimal_cold_tie_random_decisions", 0),
        "joint_cold_tie_random_decisions":
            joint["summary"].get("cold_tie_random_decisions", 0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("LAYERWEAVE_MULTI_GPU_VALIDATION=" + report["validation"])
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
