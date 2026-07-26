#!/usr/bin/env python3
"""Validate an M5.5 fixed-cache versus dynamic-cache A/B experiment."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def mean(document, metric):
    return float(document["summary"][metric]["mean"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--dynamic", required=True)
    parser.add_argument("--require-speedup", action="store_true")
    args = parser.parse_args()

    fixed = load(args.fixed)
    dynamic = load(args.dynamic)
    failures = []
    if fixed["trace"] != dynamic["trace"]:
        failures.append("trace_mismatch")
    if len(fixed["requests"]) != len(dynamic["requests"]):
        failures.append("request_count_mismatch")
    fixed_ids = [
        (item["request_id"], item["model_id"], item["input_tokens"])
        for item in fixed["requests"]
    ]
    dynamic_ids = [
        (item["request_id"], item["model_id"], item["input_tokens"])
        for item in dynamic["requests"]
    ]
    if fixed_ids != dynamic_ids:
        failures.append("workload_mismatch")
    if dynamic["vmm"].get(
            "layerweave_cache_policy") not in {"m4", "joint"}:
        failures.append("dynamic_policy_not_enabled")
    for key in (
        "input_scale", "output_tokens_override", "max_model_len",
        "trace_time_scale",
    ):
        if fixed.get(key) != dynamic.get(key):
            failures.append(f"fairness_mismatch_{key}")
    if float(dynamic.get("input_scale", 0)) != 4.0:
        failures.append("m5_5_input_scale_not_4")
    if int(dynamic.get("output_tokens_override", 0)) != 1:
        failures.append("m5_5_output_tokens_not_1")
    if any(
        int(batch.get("batch_size", 0)) != 1
        for batch in dynamic.get("batch_metrics", [])
    ):
        failures.append("m5_5_batch_size_not_1")
    for key in ("pool_gib", "page_size_mib", "layerweave_prefetch"):
        if fixed["vmm"].get(key) != dynamic["vmm"].get(key):
            failures.append(f"fairness_mismatch_{key}")

    decisions = [
        batch.get("cache_policy")
        for batch in dynamic.get("batch_metrics", [])
    ]
    if not decisions or any(item is None for item in decisions):
        failures.append("missing_policy_decisions")
    reservation_ok = bool(decisions) and all(
        int(item["evicted_pages"]) ==
        int(item["eviction_required_pages"])
        and int(item["reservation_pages"]) <= int(item["pool_pages"])
        for item in decisions if item is not None
    )
    if not reservation_ok:
        failures.append("invalid_joint_reservation")
    protected_reclamation_ok = bool(decisions) and all(
        item.get("mode") == "protected_mckp_deferred_reclamation"
        and int(item.get("evicted_protected_pages", -1)) == 0
        and int(item.get("evicted_soft_pages", -1))
        == int(item.get("evicted_pages", -2))
        and item.get("mckp") is not None
        for item in decisions if item is not None
    )
    if not protected_reclamation_ok:
        failures.append("invalid_protected_soft_reclamation")
    configurations = sorted({
        model["configuration"]
        for item in decisions if item is not None
        for model in item["targets"].values()
    })
    if not configurations or configurations == ["0"]:
        failures.append("no_prefix_selected")

    mapped = [
        sum(
            int(stage["mapped_pages"])
            for stage in batch["layerweave"]["stages"]
        )
        for batch in dynamic.get("batch_metrics", [])
        if batch.get("layerweave")
    ]
    page_counts = [
        int(batch["layerweave"]["page_count"])
        for batch in dynamic.get("batch_metrics", [])
        if batch.get("layerweave")
    ]
    reuse_observed = any(
        value < total for value, total in zip(mapped[1:], page_counts[1:])
    )
    if not reuse_observed:
        failures.append("no_weight_reuse_observed")
    kv_estimate_ok = all(
        int(batch.get("odkv", {}).get("allocated_blocks", 0))
        <= int(batch["cache_policy"]["request_kv_blocks"])
        for batch in dynamic.get("batch_metrics", [])
        if batch.get("cache_policy") is not None
    )
    if not kv_estimate_ok:
        failures.append("kv_reservation_underestimated")

    fixed_ttft = mean(fixed, "service_ttft_ms")
    dynamic_ttft = mean(dynamic, "service_ttft_ms")
    fixed_h2d = mean(fixed, "weight_load_ms")
    dynamic_h2d = mean(dynamic, "weight_load_ms")
    ttft_gain = fixed_ttft - dynamic_ttft
    h2d_gain = fixed_h2d - dynamic_h2d
    if args.require_speedup and ttft_gain <= 0:
        failures.append("service_ttft_not_improved")

    result = {
        "fixed_service_ttft_mean_ms": fixed_ttft,
        "dynamic_service_ttft_mean_ms": dynamic_ttft,
        "service_ttft_gain_ms": ttft_gain,
        "fixed_weight_h2d_mean_ms": fixed_h2d,
        "dynamic_weight_h2d_mean_ms": dynamic_h2d,
        "weight_h2d_gain_ms": h2d_gain,
        "reservation_ok": reservation_ok,
        "protected_reclamation_ok": protected_reclamation_ok,
        "kv_estimate_ok": kv_estimate_ok,
        "reuse_observed": reuse_observed,
        "selected_configurations": configurations,
        "failures": failures,
        "validation": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"M5_5_VALIDATION={result['validation']}")
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
