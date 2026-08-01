# SGRP revised evaluation: long context and scaling

## 1. Scope

This revision keeps the representative workload suite in
`eval-SGRP-paper-section.md`, moves the complete reuse-distance and
size--hotness sweeps to appendix material, and adds three experiments with
separate research questions:

1. **Input length:** can a longer Prefill window replace pipelining and cache
   planning?
2. **Routing weak scaling:** when the model catalog grows with the GPU count,
   does transition-aware routing add value beyond MCKP replacement?
3. **Cache pressure:** with the GPU count and routing fanout fixed, how does
   replacement value change with model working-set pressure?

The primary metric is mean `exposed_load_ms`. All reported improvements are
normalized within the same trace seed. Five seeds (`1234`--`1238`) and a
32-request warmup are used.

![Revised SGRP sensitivity](../results/sgrp-revised-r1000/revised-sensitivity.svg)

## 2. Common policy definitions

| Label | Replacement | Routing |
|---|---|---|
| CB-LRU | TensorGroup LRU | cache-bytes |
| CB-SuffixLRU | prefix-preserving LRU | cache-bytes |
| CB-MCKP | Prefix-MCKP | cache-bytes |
| Joint-MCKP | Prefix-MCKP | MCKP-transition |

The input-length experiment additionally includes Pipe-only, which pipelines
each cold load but has no cross-request weight cache.

Common simulator parameters remain Segment, 42 GiB/GPU, 64-MiB TensorGroups,
24.56 GB/s H2D, lookahead K=64, discount 0.9, transition weight 0.1, and
one output token for KV accounting.

## 3. Input-length sensitivity

### 3.1 Method and validity boundary

The same eight existing model profiles and balanced request sequence are used
at 128, 512, 2K, 8K, and 16K effective input tokens. There are 1000 requests
per seed, four GPUs, and exactly two available routing candidates per request.

The existing checkpoints do not all have validated 16K L40 runs. To evaluate
the requested long-context range, this simulator experiment uses
`--input-limit-policy none --stall-table-input-policy linear`: it linearly
extrapolates each existing model's last two measured offline prefix-stall
rows. Therefore 8K/16K are **profile-driven projections**, not measurements of
the current eight checkpoints at those lengths. They are appropriate only as
a provisional trend for the planned same-size, longer-context replacements;
the final paper should regenerate the offline table for those checkpoints.

### 3.2 Results

Mean exposed loading stall (ms); parentheses give improvement over matched
CB-LRU:

| Input | Pipe-only | CB-LRU | CB-SuffixLRU | CB-MCKP | Joint-MCKP |
|---:|---:|---:|---:|---:|---:|
| 128 | 706.62 | 428.88 | 417.94 (+2.55%) | 399.48 (+6.79%) | **396.33 (+7.55%)** |
| 512 | 675.92 | 411.27 | 399.53 (+2.86%) | 378.38 (+7.96%) | **371.04 (+9.73%)** |
| 2K | 500.24 | 308.20 | 296.01 (+3.96%) | 280.37 (+8.98%) | **277.93 (+9.81%)** |
| 8K | 238.47 | 159.23 | 145.51 (+8.63%) | 110.25 (+30.74%) | **91.82 (+42.33%)** |
| 16K | 223.11 | 150.35 | 137.83 (+8.31%) | 97.79 (+34.92%) | **81.54 (+45.76%)** |

Joint-MCKP beats LRU in 25/25 paired runs. Longer Prefill reduces Pipe-only
stall from 706.6 to 223.1 ms, but it does not eliminate exposed loading.
Under the extrapolated curves, value-aware prefix selection becomes more
important at 8K/16K: the policies optimize *when* bytes become exposed rather
than simply minimizing bytes. Consistent with this distinction, at 16K Joint
loads 11.58 GiB/request versus LRU's 11.19 GiB/request while reducing exposed
stall by 45.8%. This long-context increase is a projection-sensitive result
and must be revalidated with measured profiles before it is stated as a
hardware conclusion.

## 4. Routing weak scaling

### 4.1 Synthetic catalog

The base catalog uses model templates 0, 1, 2, and 4, whose weights are 5.75,
7.12, 14.96, and 27.51 GiB (55.34 GiB total). At G GPUs, the experiment creates
G independent copies of this four-model catalog:

| GPUs | Logical models | Requests | Model bytes/GPU | Available GPUs/request |
|---:|---:|---:|---:|---:|
| 1 | 4 | 1000 | 55.34 GiB | 1 |
| 2 | 8 | 2000 | 55.34 GiB | 2 |
| 3 | 12 | 3000 | 55.34 GiB | 3 |
| 4 | 16 | 4000 | 55.34 GiB | 4 |

Each logical replica reuses only its template's size and offline execution
curve. It has a distinct model ID, cache keys, residency, history, and MCKP
state, so no weights are shared across replicas. Requests/model and model
bytes/GPU remain constant.

### 4.2 Results

| GPUs | SuffixLRU vs LRU | CB-MCKP vs LRU | Joint vs LRU | Joint vs CB-MCKP |
|---:|---:|---:|---:|---:|
| 1 | +8.37% | +42.85% | +42.85% | 0.00% |
| 2 | +7.37% | **+44.01%** | +42.82% | -2.31% ±7.17 |
| 3 | +6.98% | +44.39% | **+46.29%** | +3.22% ±7.62 |
| 4 | +5.67% | +46.36% | **+46.70%** | +0.67% ±3.81 |

The G=1 equality is the required sanity check: there is no routing choice.
MCKP replacement scales robustly and Joint beats LRU in all 20/20 paired
runs. However, this balanced replicated workload does **not** establish a
stable incremental routing benefit. At G=2--4, every Joint-vs-CB-MCKP CI
crosses zero. Cache-bytes routing already forms an effective model shard when
all GPUs are available. Consequently, the approximately 43--47% Joint-vs-LRU
gain must be attributed primarily to replacement, not to transition routing.

This is a useful negative boundary, but the subplot should not be presented
as evidence that transition-aware routing improves weak scaling. The existing
heterogeneous workload suite remains the evidence for routing value.

## 5. Fixed-cluster cache-pressure scaling

### 5.1 Method

Four GPUs and exactly two available routing candidates are fixed. A one-copy
low-pressure point is followed by the original sweep, which starts from two
copies of the four-model base catalog and adds two copies at each level.
Requests/model are fixed at 125, so total requests grow with the catalog. The
200-request quick scan showed exposed stall saturating by 8--10 copies; 12
copies added no new trend and was excluded before the five-seed formal run.

| Copies | Logical models | Requests | Working set / aggregate cache |
|---:|---:|---:|---:|
| 1 | 4 | 500 | 0.33x |
| 2 | 8 | 1000 | 0.66x |
| 4 | 16 | 2000 | 1.32x |
| 6 | 24 | 3000 | 1.98x |
| 8 | 32 | 4000 | 2.64x |
| 10 | 40 | 5000 | 3.29x |

### 5.2 Results

| Pressure | LRU stall | SuffixLRU | CB-MCKP | Joint-MCKP | Joint solver CPU |
|---:|---:|---:|---:|---:|---:|
| 0.33x | 38.88 ms | +17.34% | +14.99% | **+36.97% ±25.22** | 1.58 ms/req |
| 0.66x | 242.34 ms | +4.90% | +9.92% | **+11.15% ±3.02** | 31.21 ms/req |
| 1.32x | 442.99 ms | +1.26% | +14.51% | **+16.07% ±0.90** | 46.50 ms/req |
| 1.98x | 500.13 ms | +0.69% | +11.98% | **+14.20% ±1.42** | 55.62 ms/req |
| 2.64x | 525.76 ms | +0.49% | +10.57% | **+12.69% ±0.57** | 56.64 ms/req |
| 3.29x | 536.41 ms | +0.40% | +8.71% | **+10.58% ±0.75** | 56.50 ms/req |

Joint-MCKP beats LRU in all 30 paired runs. The added 0.33x point has a high
mean Joint gain but is not a stable effect: the baseline stall is small and
seed-sensitive, its Joint-vs-LRU 95% CI is ±25.22 percentage points, and its
Joint-vs-CB-MCKP gain is +22.93% ±39.13 with only 4/5 paired wins. Among the
original pressure points, the strongest stable gain occurs near 1.32x, where
the working set just exceeds aggregate cache and policy choices have the most
leverage. At extreme pressure, unavoidable loading grows and the gain falls
but remains 10.6%. Prefix-preserving LRU alone becomes nearly irrelevant above
1.3x, whereas value-aware replacement retains an 8.7--12.0% gain.

The Python solver CPU accounting rises from 1.58 ms/request at 0.33x to
approximately 56 ms/request under high pressure. It is
simulation-host wall time and is not added to modeled TTFT, so these results
also expose a controller scalability problem that an optimized implementation
must address.

## 6. Reproduction

```bash
python doc/implement/generate_sgrp_revised_inputs.py \
  --output-dir evaluation/traces/sgrp-revised-r1000 \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --stall-table \
    docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json \
  --seeds 1234,1235,1236,1237,1238 \
  --input-requests 1000 --routing-requests-per-gpu 1000 \
  --pressure-requests-per-model 125 --max-copies 10

python doc/implement/run_sgrp_revised_eval.py \
  --trace-dir evaluation/traces/sgrp-revised-r1000 \
  --output-dir doc/results/sgrp-revised-r1000

python tools/layerpipe/summarize_sgrp_revised_eval.py \
  --trace-dir evaluation/traces/sgrp-revised-r1000 \
  --result-dir doc/results/sgrp-revised-r1000 \
  --warmup-requests 32 \
  --csv-output doc/results/sgrp-revised-r1000/summary.csv \
  --aggregate-csv-output doc/results/sgrp-revised-r1000/aggregate.csv

python tools/layerpipe/plot_sgrp_revised_eval.py \
  --aggregate-csv doc/results/sgrp-revised-r1000/aggregate.csv \
  --output doc/results/sgrp-revised-r1000/revised-sensitivity.svg
```

Source-of-truth outputs:

```text
doc/results/sgrp-revised-r1000/summary.csv
doc/results/sgrp-revised-r1000/aggregate.csv
doc/results/sgrp-revised-r1000/revised-sensitivity.svg
```
