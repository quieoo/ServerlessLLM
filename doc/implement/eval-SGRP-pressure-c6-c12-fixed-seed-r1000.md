# SGRP cache-pressure sweep: copies 6--12

## Method

This sweep extends the fixed-seed, fixed-request-count experiment to catalog
copies 6--12. Seed 1234 and 1000 total requests are fixed at every level. The
other parameters remain four GPUs, 42 GiB/GPU, two routing candidates per
request, input length 256, one output token, and a 32-request warmup. Four
policies at seven pressure levels give 28 simulator runs.

Because total requests are fixed, requests/model fall from approximately 41.7
at six copies to 20.8 at twelve copies. This differs from the original
fixed-125-requests/model experiment.

## Results

| Copies | Pressure | LRU stall | SuffixLRU vs LRU | CB-MCKP vs LRU | Joint vs LRU | Joint vs CB-MCKP |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1.98x | 502.53 ms | +0.57% | +10.57% | **+11.11%** | +0.60% |
| 7 | 2.31x | 514.20 ms | +0.51% | +10.63% | **+12.36%** | +1.93% |
| 8 | 2.63x | 527.92 ms | +0.40% | +10.27% | **+11.08%** | +0.91% |
| 9 | 2.96x | 524.83 ms | +0.55% | +9.53% | **+10.38%** | +0.94% |
| 10 | 3.29x | 538.69 ms | +0.34% | +9.65% | **+11.89%** | +2.48% |
| 11 | 3.62x | 531.53 ms | +0.51% | **+7.73%** | +7.66% | -0.08% |
| 12 | 3.95x | 536.99 ms | +0.48% | +8.46% | **+9.91%** | +1.58% |

The LRU exposed stall is already close to its high-pressure plateau over this
range. Prefix-preserving LRU remains negligible. MCKP retains a roughly
8--11% replacement gain, while the incremental Joint routing gain is small
and non-monotonic: -0.08% to +2.48%. At copy 11, Joint is marginally worse
than CB-MCKP for this seed. Consequently this high-pressure extension does not
show routing benefit increasing with pressure.

Copy 6 was rerun independently. All modeled latency and traffic metrics match
the preceding copies-1--6 run exactly; only host wall-clock solver timing and
artifact path metadata differ.

