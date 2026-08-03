# Cache and Routing Policy Evaluation

## Compared strategies

| ID | Routing | Replacement | Plot label |
|---|---|---|---|
| CB-LRU | Cache bytes | LRU | Bytes-LRU |
| CB-SuffixLRU | Cache bytes | Prefix-preserving LRU | Bytes-SuffixLRU |
| CB-MCKP | Cache bytes | Prefix-MCKP | Bytes-MCKP |
| Joint-MCKP | MCKP transition | Prefix-MCKP | Joint |

Pipe-only is included as a horizontal reference for input-length sensitivity.
The primary metric is mean exposed loading stall. Also report p95, mean H2D
GiB/request, and MCKP solver CPU time/request.

## Paper workload dimensions

The current revised suite contains:

- Input length up to the feasible 16K context cases.
- Routing weak scaling while holding model catalog size per GPU constant.
- Cache-pressure scaling at 4 GPUs by increasing independent model copies.

The representative six-workload suite supplies the broader workload
sensitivity anchor: ServeGen, cyclic, reuse-distance 2 and 7, small-hot, and
large-hot.

## Formal revised sweep

Existing trace inputs are under `evaluation/traces/sgrp-revised-r1000`.

```bash
python evaluation/tensor_simulator/experiments/run_sgrp_revised_eval.py \
  --trace-dir evaluation/traces/sgrp-revised-r1000 \
  --output-dir evaluation/tensor_simulator/results/sgrp-revised-r1000 \
  --max-parallel "$(nproc)"

python evaluation/tensor_simulator/experiments/summarize_sgrp_revised_eval.py \
  --trace-dir evaluation/traces/sgrp-revised-r1000 \
  --result-dir evaluation/tensor_simulator/results/sgrp-revised-r1000 \
  --warmup-requests 32 \
  --csv-output evaluation/tensor_simulator/results/sgrp-revised-r1000/summary.csv \
  --aggregate-csv-output evaluation/tensor_simulator/results/sgrp-revised-r1000/aggregate.csv

python evaluation/tensor_simulator/experiments/plot_sgrp_revised_eval.py \
  --aggregate-csv evaluation/tensor_simulator/results/sgrp-revised-r1000/aggregate.csv \
  --output evaluation/tensor_simulator/results/sgrp-revised-r1000/revised-sensitivity.svg
```

For a smoke test, replace both `sgrp-revised-r1000` paths with
`sgrp-revised-fast`.

## Interpretation

- Input length changes the Prefill window available to hide loading. It is not
  solely a cache-hit experiment.
- Routing weak scaling tests whether transition-aware routing remains useful as
  GPUs and independent model copies grow together.
- Cache pressure increases the independent model working set at fixed GPU
  count; `copy=N` means N copies of the four-model base catalog, not N requests.
- Claims about MCKP should use the aggregate across declared seeds, not the
  best seed.

