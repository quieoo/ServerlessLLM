#!/usr/bin/env python3
"""Plot Tangram end-to-end comparison figures from summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


SYSTEM_LABELS = {
    "sllm_cm": "SLLM-CM",
    "tangram": "OurSystem",
}

SYSTEM_STYLES = {
    "sllm_cm": {"marker": "s", "linestyle": "-", "color": "#253f4b"},
    "tangram": {"marker": "o", "linestyle": "--", "color": "#e76f51"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slo-scale", type=float, default=6.0)
    parser.add_argument("--rps-for-slo-scale", type=float, default=1.6)
    parser.add_argument("--gpu-for-slo", type=int, default=8)
    parser.add_argument("--rps-list", default="1.6,2.4,3.2,4.0")
    parser.add_argument("--gpu-list", default="1,2,4,8")
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(item) for item in raw.replace(",", " ").split()]


def parse_int_list(raw: str) -> List[int]:
    return [int(item) for item in raw.replace(",", " ").split()]


def read_summary(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def nearly_equal(lhs: float, rhs: float) -> bool:
    return abs(lhs - rhs) < 1e-9


def select_rows(
    rows: Iterable[Dict[str, str]],
    *,
    system: str,
    rps: float | None = None,
    gpu_num: int | None = None,
    slo_scale: float | None = None,
) -> List[Dict[str, str]]:
    selected = []
    for row in rows:
        if row["system"] != system:
            continue
        if rps is not None and not nearly_equal(float(row["rps"]), rps):
            continue
        if gpu_num is not None and int(row["gpu_num"]) != gpu_num:
            continue
        if slo_scale is not None and not nearly_equal(float(row["slo_scale"]), slo_scale):
            continue
        selected.append(row)
    return selected


def plot_series(ax, x_values, y_values, system: str) -> None:
    style = SYSTEM_STYLES.get(system, {})
    ax.plot(
        x_values,
        y_values,
        label=SYSTEM_LABELS.get(system, system),
        markersize=4,
        linewidth=1.2,
        **style,
    )


def values_by_x(rows: List[Dict[str, str]], x_name: str, y_name: str, x_values):
    result = []
    for x_value in x_values:
        matched = None
        for row in rows:
            row_x = float(row[x_name]) if x_name != "gpu_num" else int(row[x_name])
            if row_x == x_value:
                matched = float(row[y_name])
                break
        result.append(matched if matched is not None else float("nan"))
    return result


def main() -> None:
    args = parse_args()
    rows = read_summary(args.summary)
    systems = [system for system in ("sllm_cm", "tangram") if any(row["system"] == system for row in rows)]
    rps_values = parse_float_list(args.rps_list)
    gpu_values = parse_int_list(args.gpu_list)

    fig, axes = plt.subplots(1, 4, figsize=(13.8, 2.5))

    ax = axes[0]
    for system in systems:
        selected = select_rows(
            rows, system=system, gpu_num=args.gpu_for_slo, slo_scale=args.slo_scale
        )
        plot_series(
            ax,
            rps_values,
            values_by_x(selected, "rps", "slo_attainment", rps_values),
            system,
        )
    ax.set_xlabel("RPS")
    ax.set_ylabel("SLO Attainment")
    ax.set_title(f"(a) SLO Scale={args.slo_scale:g}, GPU={args.gpu_for_slo}")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, linestyle=":", linewidth=0.5)

    ax = axes[1]
    slo_values = list(range(1, 10))
    for system in systems:
        selected = select_rows(
            rows, system=system, rps=args.rps_for_slo_scale, gpu_num=args.gpu_for_slo
        )
        plot_series(
            ax,
            slo_values,
            values_by_x(selected, "slo_scale", "slo_attainment", slo_values),
            system,
        )
    ax.set_xlabel("SLO Scale")
    ax.set_ylabel("SLO Attainment")
    ax.set_title(f"(b) RPS={args.rps_for_slo_scale:g}, GPU={args.gpu_for_slo}")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, linestyle=":", linewidth=0.5)

    for axis_idx, rps in enumerate((0.4, 1.6), start=2):
        ax = axes[axis_idx]
        for system in systems:
            selected = select_rows(rows, system=system, rps=rps, slo_scale=args.slo_scale)
            p99_seconds = [
                value / 1000.0
                for value in values_by_x(selected, "gpu_num", "p99_ttft_ms", gpu_values)
            ]
            plot_series(ax, gpu_values, p99_seconds, system)
        ax.set_xlabel("Num of GPUs")
        ax.set_ylabel("P99 TTFT (s)")
        ax.set_title(f"({chr(ord('a') + axis_idx)}) RPS={rps:g}")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
