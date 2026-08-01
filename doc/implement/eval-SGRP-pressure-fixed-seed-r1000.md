# SGRP cache-pressure sweep: fixed seed and request count

## Method

This companion sweep fixes seed 1234 and total trace length at 1000 requests
for every catalog size. It uses four GPUs, a 42-GiB pool per GPU, two available
routing candidates per request, input length 256, one output token, and a
32-request warmup. Catalog copies range from one through six. Four policies are
run at every point, for 24 simulator runs in total.

Unlike the five-seed pressure experiment, fixing the total request count means
that requests/model decrease as the catalog grows. The approximate values are
250, 125, 83.3, 62.5, 50, and 41.7 requests/model for copies 1--6. Results from
this sweep therefore should not be mixed with the original fixed-125-
requests/model confidence intervals.

## Results

All percentage improvements are paired against the same seed-1234 CB-LRU run.
The final column isolates the incremental routing benefit over CB-MCKP.

| Copies | Pressure | LRU stall | SuffixLRU vs LRU | CB-MCKP vs LRU | Joint vs LRU | Joint vs CB-MCKP |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.33x | 33.10 ms | +16.68% | +39.14% | **+39.27%** | +0.21% |
| 2 | 0.66x | 243.91 ms | +3.93% | +0.78% | **+9.06%** | +8.35% |
| 3 | 0.99x | 369.82 ms | +2.93% | +11.65% | **+12.73%** | +1.23% |
| 4 | 1.32x | 453.42 ms | +0.95% | +15.20% | **+17.10%** | +2.25% |
| 5 | 1.65x | 487.67 ms | +0.74% | +11.93% | **+16.31%** | +4.97% |
| 6 | 1.98x | 502.53 ms | +0.57% | +10.57% | **+11.11%** | +0.60% |

Joint-MCKP beats CB-LRU at all six points, but neither Joint-vs-LRU nor
Joint-vs-CB-MCKP grows monotonically with pressure. The 0.33x result is mostly
an MCKP replacement effect: Joint adds only 0.21% over CB-MCKP. The largest
incremental routing gain in this single-seed sweep is 8.35% at 0.66x, followed
by 4.97% at 1.65x. Because only one seed is present, these differences have no
cross-seed confidence interval and should be treated as seed-specific.

## Reproduction

```bash
python doc/implement/generate_sgrp_revised_inputs.py \
  --output-dir evaluation/traces/sgrp-pressure-c1-c6-r1000-seed1234 \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --stall-table docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json \
  --seeds 1234 --pressure-requests 1000 \
  --pressure-copies 1,2,3,4,5,6 --max-copies 6

python doc/implement/run_sgrp_revised_eval.py \
  --trace-dir evaluation/traces/sgrp-pressure-c1-c6-r1000-seed1234 \
  --output-dir doc/results/sgrp-pressure-c1-c6-r1000-seed1234 \
  --seeds 1234 --dimensions pressure \
  --levels c1,c2,c3,c4,c5,c6 --max-parallel 24

python tools/layerpipe/summarize_sgrp_revised_eval.py \
  --trace-dir evaluation/traces/sgrp-pressure-c1-c6-r1000-seed1234 \
  --result-dir doc/results/sgrp-pressure-c1-c6-r1000-seed1234 \
  --seeds 1234 --warmup-requests 32 --dimensions pressure \
  --levels c1,c2,c3,c4,c5,c6 \
  --csv-output doc/results/sgrp-pressure-c1-c6-r1000-seed1234/summary.csv \
  --aggregate-csv-output \
    doc/results/sgrp-pressure-c1-c6-r1000-seed1234/aggregate.csv
```

