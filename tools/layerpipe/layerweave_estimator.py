#!/usr/bin/env python3
"""Calibrate and evaluate the LayerWeave service-TTFT estimator."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def median(values, default=0.0):
    return statistics.median(values) if values else default


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (ordered[lower] * (upper - position)
            + ordered[upper] * (position - lower))


def linear_fit(samples):
    """Return intercept and slope for y = intercept + slope*x."""
    if not samples:
        return 0.0, 0.0
    if len(samples) == 1:
        x, y = samples[0]
        return 0.0, y / max(1.0, x)
    mean_x = sum(x for x, _ in samples) / len(samples)
    mean_y = sum(y for _, y in samples) / len(samples)
    denominator = sum((x - mean_x) ** 2 for x, _ in samples)
    if denominator <= 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y)
                for x, y in samples) / denominator
    intercept = mean_y - slope * mean_x
    return max(0.0, intercept), max(0.0, slope)


def quadratic_fit(samples):
    """Fit y = a + b*x + c*x^2; fall back to linear if singular."""
    distinct = {x for x, _ in samples}
    if len(distinct) < 3:
        intercept, slope = linear_fit(samples)
        return intercept, slope, 0.0
    sums = [sum(x ** power for x, _ in samples) for power in range(5)]
    rhs = [sum((x ** power) * y for x, y in samples)
           for power in range(3)]
    matrix = [
        [sums[0], sums[1], sums[2], rhs[0]],
        [sums[1], sums[2], sums[3], rhs[1]],
        [sums[2], sums[3], sums[4], rhs[2]],
    ]
    for column in range(3):
        pivot = max(range(column, 3),
                    key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) < 1e-12:
            intercept, slope = linear_fit(samples)
            return intercept, slope, 0.0
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(3):
            if row == column:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                matrix[row][index] - factor * matrix[column][index]
                for index in range(4)
            ]
    return tuple(matrix[row][3] for row in range(3))


def batch_features(batch):
    """Scaled prefill features used by each layer compute regression."""
    batch_size = max(1, int(batch.get("batch_size", 1)))
    total_tokens = max(1, int(batch["input_tokens"]))
    max_tokens = max(
        1, int(batch.get("max_input_tokens",
                         math.ceil(total_tokens / batch_size))))
    sum_squared = float(batch.get(
        "sum_input_tokens_squared",
        batch_size * max_tokens * max_tokens,
    ))
    return [
        1.0,
        float(batch_size - 1),
        total_tokens / 1000.0,
        max_tokens / 1000.0,
        sum_squared / 1_000_000.0,
    ]


def ridge_fit(samples, regularization=1e-6):
    """Fit y = features*coefficients with a small diagonal ridge."""
    if not samples:
        return [0.0] * 5
    width = len(samples[0][0])
    matrix = [[0.0] * (width + 1) for _ in range(width)]
    for features, value in samples:
        for row in range(width):
            matrix[row][-1] += features[row] * value
            for column in range(width):
                matrix[row][column] += (
                    features[row] * features[column])
    for index in range(width):
        matrix[index][index] += regularization
    for column in range(width):
        pivot = max(
            range(column, width),
            key=lambda row: abs(matrix[row][column]),
        )
        if abs(matrix[pivot][column]) < 1e-12:
            continue
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(width):
            if row == column:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                matrix[row][index] - factor * matrix[column][index]
                for index in range(width + 1)
            ]
    return [matrix[row][-1] for row in range(width)]


def load_batches(paths):
    batches = []
    for path in paths:
        document = json.loads(Path(path).read_text())
        request_by_batch = defaultdict(list)
        for request in document.get("requests", []):
            request_by_batch[int(request["batch_id"])].append(request)
        seen_models = set()
        for source_batch in document.get("batch_metrics", []):
            metrics = source_batch.get("layerweave")
            if not metrics or not metrics.get("stages") or not metrics.get(
                    "computes"):
                continue
            # vLLM may split a large logical batch into multiple Prefill
            # scheduler waves. M4 models one LayerWeave pipeline activation;
            # M6 composes multiple predicted activations when scheduling a
            # multi-wave batch.
            forward_ids = {
                int(item.get("forward_id", 0))
                for item in metrics["computes"]
            }
            if len(forward_ids) != 1:
                continue
            batch = dict(source_batch)
            model_id = int(batch["model_id"])
            requests = request_by_batch[int(batch["batch_id"])]
            if "service_ttft_ms" not in batch:
                batch["service_ttft_ms"] = max(
                    (float(item["service_ttft_ms"]) for item in requests),
                    default=0.0,
                )
            batch["_source"] = str(path)
            batch["_prefetch"] = bool(metrics["prefetch_enabled"])
            batch["_prefix"] = (
                batch.get("prefix_cache") or {}
            ).get("configuration", "full")
            batch["_phase"] = (
                "first-activation"
                if model_id not in seen_models else "steady"
            )
            seen_models.add(model_id)
            batches.append(batch)
    return batches


def residency_class(batch):
    metrics = batch["layerweave"]
    mapped = sum(int(stage["mapped_pages"]) for stage in metrics["stages"])
    page_count = max(1, int(metrics["page_count"]))
    if mapped == 0:
        return "full-hit"
    if mapped >= page_count:
        return "cold"
    return "partial-hit"


class PipelineEstimator:
    """GPU pipeline model plus measured host/runtime residual profiles."""

    def __init__(self):
        self.models = {}

    @staticmethod
    def _mode_key(prefetch):
        return "prefetch" if prefetch else "no-prefetch"

    @staticmethod
    def _phase_key(batch):
        return batch.get("_phase", "steady")

    def fit(self, batches):
        grouped = defaultdict(list)
        for batch in batches:
            grouped[int(batch["model_id"])].append(batch)
        for model_id, records in grouped.items():
            bandwidth_samples = []
            fixed_samples = []
            compute_samples = defaultdict(list)
            host_gaps = defaultdict(list)
            for batch in records:
                metrics = batch["layerweave"]
                mode = self._mode_key(batch["_prefetch"])
                state = residency_class(batch)
                for stage in metrics["stages"]:
                    duration = float(stage["h2d_ms"])
                    size = int(stage["to_load_bytes"])
                    if size > 0 and duration > 0:
                        bandwidth_samples.append(size / duration)
                    elif duration > 0:
                        fixed_samples.append(duration)
                for compute in metrics["computes"]:
                    # The first request includes lazy kernel/runtime warm-up.
                    # Keep that cost in the service residual instead of
                    # distorting the steady layer compute profile.
                    if self._phase_key(batch) == "steady":
                        compute_samples[int(compute["layer"])].append(
                            (batch_features(batch),
                             float(compute["compute_ms"])))
                phase = self._phase_key(batch)
                host_gaps[(mode, state, phase)].append(float(
                    metrics.get("host_submission_gap_ms", 0.0)))
            compute_profile = {}
            for layer, samples in compute_samples.items():
                coefficients = ridge_fit(samples)
                compute_profile[layer] = {
                    "coefficients": coefficients,
                    "feature_order": [
                        "intercept",
                        "batch_size_minus_one",
                        "total_tokens_per_1000",
                        "max_tokens_per_1000",
                        "sum_tokens_squared_per_1m",
                    ],
                }
            self.models[model_id] = {
                "bytes_per_ms": median(bandwidth_samples, 1.0),
                "fixed_stage_ms": median(fixed_samples, 0.0),
                "compute": compute_profile,
                "host_gap_ms": {
                    f"{mode}/{state}/{phase}": median(values)
                    for (mode, state, phase), values in host_gaps.items()
                },
                "service_residual_ms": {},
            }
            residuals = defaultdict(list)
            for batch in records:
                mode = self._mode_key(batch["_prefetch"])
                state = residency_class(batch)
                phase = self._phase_key(batch)
                gpu = self._simulate_gpu(batch, self.models[model_id])
                residuals[(mode, state, phase)].append(
                    float(batch["service_ttft_ms"])
                    - gpu["predicted_gpu_critical_path_ms"])
            self.models[model_id]["service_residual_ms"] = {
                f"{mode}/{state}/{phase}": median(values)
                for (mode, state, phase), values in residuals.items()
            }
        return self

    @staticmethod
    def _profile_value(profile, name, mode, state, phase):
        values = profile[name]
        key = f"{mode}/{state}/{phase}"
        if key in values:
            return values[key]
        same_mode = [
            value for item, value in values.items()
            if item.startswith(mode + "/")
        ]
        return median(same_mode, median(list(values.values()), 0.0))

    def _simulate_gpu(self, batch, profile):
        metrics = batch["layerweave"]
        features = batch_features(batch)
        prefetch = batch["_prefetch"]
        stages = metrics["stages"]
        stage_by_layer = {
            int(stage["stage"]): stage
            for stage in stages if isinstance(stage["stage"], int)
        }
        initial = next(stage for stage in stages
                       if stage["stage"] == "initial")

        def load_ms(stage):
            size = int(stage["to_load_bytes"])
            if size == 0:
                return profile["fixed_stage_ms"]
            return (profile["fixed_stage_ms"]
                    + size / profile["bytes_per_ms"])

        def compute_ms(layer):
            item = profile["compute"].get(layer, {})
            coefficients = item.get("coefficients")
            if coefficients is None:
                # Backward-compatible profile reader for older reports.
                total_tokens = max(1, int(batch["input_tokens"]))
                return max(
                    0.0,
                    item.get("intercept_ms", 0.0)
                    + item.get("ms_per_token", 0.0) * total_tokens
                    + item.get("ms_per_token_squared", 0.0)
                    * total_tokens * total_tokens,
                )
            return max(0.0, sum(
                coefficient * feature
                for coefficient, feature in zip(coefficients, features)
            ))

        compute_start = load_ms(initial)
        compute_end = compute_start
        copy_end = compute_start
        gpu_ready_stall = compute_start
        layer_count = len(metrics["computes"])
        for layer in range(layer_count):
            if layer > 0:
                stage = stage_by_layer[layer]
                copy_start = (
                    max(copy_end, previous_compute_start)
                    if prefetch else compute_end
                )
                copy_end = copy_start + load_ms(stage)
                stall = max(0.0, copy_end - compute_end)
                gpu_ready_stall += stall
                compute_start = max(compute_end, copy_end)
            previous_compute_start = compute_start
            compute_end = compute_start + compute_ms(layer)

        return {
            "predicted_gpu_ready_stall_ms": gpu_ready_stall,
            "predicted_gpu_critical_path_ms": compute_end,
        }

    def predict(self, batch):
        model_id = int(batch["model_id"])
        profile = self.models[model_id]
        mode = self._mode_key(batch["_prefetch"])
        state = residency_class(batch)
        phase = self._phase_key(batch)
        gpu = self._simulate_gpu(batch, profile)
        host_gap = self._profile_value(
            profile, "host_gap_ms", mode, state, phase)
        residual_ms = self._profile_value(
            profile, "service_residual_ms", mode, state, phase)
        predicted_forward = (
            gpu["predicted_gpu_critical_path_ms"] + host_gap)
        return {
            **gpu,
            "predicted_forward_ms": predicted_forward,
            "predicted_host_submission_gap_ms": host_gap,
            "predicted_service_residual_ms": residual_ms,
            "predicted_service_external_ms": residual_ms - host_gap,
            "predicted_service_ttft_ms": (
                gpu["predicted_gpu_critical_path_ms"] + residual_ms),
        }


def summarize_errors(rows, actual_key, predicted_key):
    errors = [abs(row[predicted_key] - row[actual_key]) for row in rows]
    actual = [abs(row[actual_key]) for row in rows]
    return {
        "count": len(rows),
        "mae_ms": sum(errors) / len(errors) if errors else 0.0,
        "p95_absolute_error_ms": percentile(errors, 0.95),
        "mape": (
            sum(error / max(value, 1e-3)
                for error, value in zip(errors, actual)) / len(errors)
            if errors else 0.0
        ),
    }


def evaluate(estimator, batches):
    rows = []
    for batch in batches:
        model_id = int(batch["model_id"])
        if model_id not in estimator.models:
            continue
        prediction = estimator.predict(batch)
        actual_service = float(batch["service_ttft_ms"])
        rows.append({
            "source": batch["_source"],
            "model_id": model_id,
            "batch_id": int(batch["batch_id"]),
            "input_tokens": int(batch["input_tokens"]),
            "max_input_tokens": int(batch.get(
                "max_input_tokens", batch["input_tokens"])),
            "batch_size": int(batch.get("batch_size", 1)),
            "prefetch_enabled": batch["_prefetch"],
            "prefix_configuration": batch["_prefix"],
            "residency": residency_class(batch),
            "actual_service_ttft_ms": actual_service,
            **prediction,
            "absolute_service_error_ms": abs(
                prediction["predicted_service_ttft_ms"] - actual_service),
        })
    summary = {
        state: summarize_errors(
            [row for row in rows if row["residency"] == state],
            "actual_service_ttft_ms", "predicted_service_ttft_ms")
        for state in ("cold", "partial-hit", "full-hit")
    }
    by_model = {
        str(model_id): summarize_errors(
            [row for row in rows if row["model_id"] == model_id],
            "actual_service_ttft_ms", "predicted_service_ttft_ms")
        for model_id in sorted({row["model_id"] for row in rows})
    }
    by_batch_size = {
        str(batch_size): summarize_errors(
            [row for row in rows if row["batch_size"] == batch_size],
            "actual_service_ttft_ms", "predicted_service_ttft_ms")
        for batch_size in sorted({row["batch_size"] for row in rows})
    }

    paired = defaultdict(dict)
    for row in rows:
        key = (
            row["model_id"], row["batch_id"], row["input_tokens"],
            row["prefix_configuration"], row["residency"],
        )
        paired[key][row["prefetch_enabled"]] = row
    gains = []
    for key, pair in paired.items():
        if True not in pair or False not in pair:
            continue
        enabled, disabled = pair[True], pair[False]
        actual = (disabled["actual_service_ttft_ms"]
                  - enabled["actual_service_ttft_ms"])
        predicted = (disabled["predicted_service_ttft_ms"]
                     - enabled["predicted_service_ttft_ms"])
        gains.append({
            "model_id": key[0],
            "batch_id": key[1],
            "input_tokens": key[2],
            "prefix_configuration": key[3],
            "residency": key[4],
            "actual_overlap_gain_ms": actual,
            "predicted_overlap_gain_ms": predicted,
            "absolute_error_ms": abs(predicted - actual),
        })
    gain_summary = summarize_errors(
        gains, "actual_overlap_gain_ms", "predicted_overlap_gain_ms")
    return {
        "service_ttft_summary": summary,
        "service_ttft_by_model": by_model,
        "service_ttft_by_batch_size": by_batch_size,
        "overlap_gain_summary": gain_summary,
        "rows": rows,
        "overlap_gain_rows": gains,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", nargs="+", required=True)
    parser.add_argument("--evaluation", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    estimator = PipelineEstimator().fit(load_batches(args.calibration))
    report = evaluate(estimator, load_batches(args.evaluation))
    report["profiles"] = estimator.models
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print("SERVICE_TTFT=" + json.dumps(
        report["service_ttft_summary"], sort_keys=True))
    print("OVERLAP_GAIN=" + json.dumps(
        report["overlap_gain_summary"], sort_keys=True))
    print(f"LAYERWEAVE_ESTIMATOR_RESULT={args.output}")


if __name__ == "__main__":
    main()
