# System Overhead and PSE Accuracy

## CUDA readiness hook/event overhead

Compare warm, fully resident Prefill for:

1. Native vLLM.
2. VMM resident execution without readiness hooks.
3. LayerWeave resident execution with all readiness hooks/events enabled.

The readiness-hook increment is the matched difference between the latter two
resident paths. Report per-model mean and maximum at input lengths 128, 1K, 4K,
and feasible 16K cases. The existing formal runner and implementation record
are:

```text
evaluation/tensor_simulator/results/implemt/run_system_overhead_formal.py
evaluation/tensor_simulator/Implementation_docs/system-overhead-and-pse.md
```

This runner requires the Tangram vLLM/CUDA environment and available GPUs

## PSE exposed-stall accuracy

Evaluate matched predicted and measured exposed loading stall for:

- cold requests;
- partial hits;
- full hits.

Report MAE and non-zero-sample MAPE. Full-hit samples can have near-zero true
stall, so ordinary MAPE is not meaningful for those samples without an
explicit denominator rule.

## Existing outputs

```text
evaluation/tensor_simulator/results/system-overhead-formal/summary.csv
evaluation/tensor_simulator/results/system-overhead-formal/readiness-overhead.svg
evaluation/tensor_simulator/results/implemt/pse-exposed-smoke-summary.csv
evaluation/tensor_simulator/results/implemt/pse-partial-hit-smoke-summary.csv
```

