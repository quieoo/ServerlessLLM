# SGRP backend comparison on six representative traces

> **Implementation update:** Page backends now batch all missing groups in a
> request into one Prefix-MCKP reclaim and Tensor-page precomputes physical
> prefix sizes. The six-trace table below predates that change. The current
> implementation has been revalidated on ServeGen; see
> "Optimized ServeGen rerun" below. Do not use the old Page rows as current
> implementation results.

## Method

This experiment fixes Joint-MCKP (`mckp-prefix` replacement and
`mckp-transition` routing) and changes only the tensor-memory backend:

- `segment`
- `tensor-page`
- `compact-page`

It uses seed 1236 only, 1000 requests per trace, and excludes the first 32
requests as warmup. Common settings are four GPUs, exactly two available GPUs
per request, 42 GiB/GPU, 8-MiB pages, 64-MiB TensorGroups, input scale 4,
output-token override 1, H2D 24.56 GB/s, and MCKP lookahead K=64.

The six existing workload-suite traces are ServeGen, balanced cyclic, reuse
D=2, reuse D approximately 7, small models hot, and large models hot. This is
a single-seed matched comparison; it does not report confidence intervals.

![Backend comparison](../results/sgrp-backend-suite-r1000-seed1236/backend-comparison.svg)

## Mean exposed loading stall

Parentheses are the increase relative to Segment on the same trace.

| Trace | Segment | Tensor-page | Compact-page |
|---|---:|---:|---:|
| ServeGen | **71.65** | 87.39 (+21.96%) | 80.80 (+12.77%) |
| Balanced cyclic | **333.45** | 369.52 (+10.81%) | 356.74 (+6.98%) |
| Reuse D=2 | **139.81** | 157.62 (+12.74%) | 150.20 (+7.43%) |
| Reuse D=7 | **353.02** | 376.00 (+6.51%) | 393.32 (+11.41%) |
| Small models hot | **62.99** | 71.39 (+13.34%) | 67.13 (+6.58%) |
| Large models hot | **324.14** | 370.96 (+14.45%) | 357.99 (+10.44%) |
| Mean across traces | **214.18** | 238.81 (+13.30% mean normalized) | 234.36 (+9.27% mean normalized) |

Segment wins all six traces. Compact-page beats Tensor-page on five of six;
the exception is reuse D=7, where Compact-page transfers 11.29 GiB/request
versus Tensor-page's 10.79 GiB/request and its different shared-page residency
decisions outweigh its lower fragmentation.

## Cost decomposition

The following values are arithmetic means of the six per-trace means.

| Backend | H2D GiB/request | VMM/request | Compaction/request | Map calls/request | Unmap calls/request | Solver CPU/request |
|---|---:|---:|---:|---:|---:|---:|
| Segment | 6.716 | 0.00 | 5.10 | 0 | 0 | 12.27 |
| Tensor-page | 6.960 | 21.98 | 0.00 | 915.80 | 915.77 | 151.85 |
| Compact-page | 6.842 | 21.03 | 0.00 | 876.06 | 876.05 | 63.77 |

Every physical Page operation is charged: all requests satisfy
`map_calls == mapped_pages` and `unmap_calls == unmapped_pages`. Under the
current 8-MiB page-cost model, avoiding Segment relocation saves about 5.1 per
request but introduces approximately 21--22 of VMM map/unmap cost. Therefore
the Page backends lose on exposed stall even before considering their higher
Python planning cost.

Tensor-page also averages roughly 1.16 GiB of internal fragmentation in the
six workloads because each TensorGroup is rounded independently. Compact-page
averages only about 5.7 MiB by packing tensors into one compact model arena,
which explains its advantage over Tensor-page in most traces.

`solver CPU/request` is wall time on the simulation host and is not included
in modeled TTFT or exposed stall. Tensor-page's 151.9 per-request solver cost
is an implementation scalability problem, not a simulated GPU latency.

## Interpretation boundary

These results support the following narrow claim:

> With 8-MiB pages and the simulator's current fixed-plus-per-page VMM cost,
> Segment has the lowest exposed loading stall on all six representative
> traces; Compact-page reduces Tensor-page fragmentation but does not recover
> the physical map/unmap overhead.

They do not establish that Segment is universally better. The ranking is
sensitive to page size and map/unmap calibration. Compact-page also projects
TensorGroup prefix value onto shared pages, and its compact offsets remain an
exporter-order proxy rather than verified runtime stable-VA layout.

## Reproduction

```bash
python evaluation/tensor_simulator/experiments/run_sgrp_backend_workload_suite.py \
  --output-dir doc/results/sgrp-backend-suite-r1000-seed1236 \
  --requests 1000 --max-parallel 18

python evaluation/tensor_simulator/experiments/summarize_sgrp_backend_workload_suite.py \
  --result-dir doc/results/sgrp-backend-suite-r1000-seed1236 \
  --warmup-requests 32 \
  --csv-output doc/results/sgrp-backend-suite-r1000-seed1236/summary.csv \
  --svg-output \
    doc/results/sgrp-backend-suite-r1000-seed1236/backend-comparison.svg
```

Source of truth:

```text
doc/results/sgrp-backend-suite-r1000-seed1236/summary.csv
doc/results/sgrp-backend-suite-r1000-seed1236/backend-comparison.svg
```

## Optimized ServeGen rerun

Only ServeGen was rerun after the implementation optimization, using the same
seed 1236, 1000 requests, and 32-request warmup.

| Backend | Mean exposed load | Mean TTFT | Solver CPU/request | VMM/request | H2D GiB/request |
|---|---:|---:|---:|---:|---:|
| Segment | **71.65** | **419.00** | 7.24 | 0.00 | 2.963 |
| Tensor-page | 90.87 (+26.81%) | 438.21 | 8.02 | 10.61 | 3.359 |
| Compact-page | 82.98 (+15.80%) | 430.32 | 9.47 | 9.39 | 3.056 |

Compared with the pre-optimization run, Tensor-page solver CPU falls from
66.65 to 8.02 (-88.0%) and Compact-page falls from 29.11 to 9.47 (-67.5%).
Batch reclaim replaces repeated per-group greedy plans with one request-level
plan, so the residency trajectory and exposed-load result can also change; it
is not a timing-only memoization.

Current ServeGen outputs:

```text
doc/results/sgrp-backend-suite-r1000-seed1236-optimized/servegen-summary.csv
doc/results/sgrp-backend-suite-r1000-seed1236-optimized/servegen-comparison.svg
```

### Exposed loading stall decomposition

The optimized simulator now emits an additive request-level decomposition:

```text
exposed_load_ms
  = exposed_h2d_ms
  + exposed_compaction_ms
  + exposed_page_map_ms
  + exposed_page_unmap_ms
  + exposed_allocation_ms
```

The offline prefix table combines pipelined H2D and Segment allocation. Their
exposed contributions are separated by replaying the measured group-boundary
timeline with H2D/allocation enabled alone and together, followed by a
two-factor Shapley attribution. Compaction and physical-page map/unmap occur
outside that profiled pipeline and are fully exposed. Thus these are exposed
critical-path contributions, not the larger raw loading-work counters.

ServeGen means after the 32-request warmup, in ms/request:

| Backend | Total exposed | H2D | Segment compaction | Page map | Page unmap | Allocation |
|---|---:|---:|---:|---:|---:|---:|
| Segment | **71.654** | 68.868 | 2.755 | 0 | 0 | 0.032 |
| Tensor-page | 90.867 | 80.221 | 0 | 5.305 | 5.304 | 0.036 |
| Compact-page | 82.977 | 73.551 | 0 | 4.697 | 4.697 | 0.033 |

For every measured request and all three backends, the maximum absolute error
between total exposed stall and the sum of the five components is below
`2.3e-13 ms`. The Page columns count every physical page operation; allocation
is nonzero for Page because the shared offline loading profile was calibrated
with the 0.005-ms per-group allocation term. It is attributed consistently to
all backends rather than silently folded into H2D.

### Compact-page 64-MiB sensitivity

One matched ServeGen run changes only Compact-page `--page-size-mib` from 8
to 64. Both runs use 1000 requests and exclude the first 32 as warmup.

| Metric | 8 MiB | 64 MiB | Change |
|---|---:|---:|---:|
| Mean exposed loading stall | 82.977 | **81.492** | -1.79% |
| Mean TTFT | 430.322 | **428.836** | -0.35% |
| Exposed H2D | 73.551 | 80.182 | +9.02% |
| Page map + unmap | 9.393 | **1.274** | -86.43% |
| H2D GiB/request | 3.056 | 3.308 | +8.25% |
| Map calls/request | 391.40 | 53.08 | -86.44% |
| Unmap calls/request | 391.38 | 53.09 | -86.44% |
| Internal fragmentation MiB | 6.40 | 29.77 | +23.37 MiB |
| Selected-candidate MCKP solver CPU/request | 8.50 | **5.55** | -34.70% |

The larger page recovers 8.12 ms/request of VMM cost, but 6.63 ms/request is
lost to additional exposed H2D. The net exposed-stall improvement is therefore
only 1.49 ms/request, and 64-MiB Compact-page remains 13.73% above Segment's
71.654 ms/request. `MCKP solver CPU` is reported separately and is not included
in modeled TTFT; the current row records the selected routing candidate rather
than total CPU work across every evaluated candidate.

Source result:

```text
doc/results/sgrp-backend-suite-r1000-seed1236-optimized/servegen/compact-page-64mib.json
```

## Page-size fast sweep

Following the evaluation protocol, the first pass uses 200 ServeGen requests
with seed 1236 and excludes the first 32 requests. It sweeps
`1/2/4/8/16/32/64/128 MiB`; this is a trend-selection run rather than a formal
1000-request result.

| Backend | Page MiB | Exposed stall | Exposed H2D | Map+unmap | H2D GiB/req | Internal frag MiB | Selected MCKP CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tensor-page | 1 | 195.137 | 95.171 | 99.923 | 4.050 | 183.24 | 10.03 |
| Tensor-page | 2 | 148.495 | 98.870 | 49.581 | 4.002 | 350.18 | 9.73 |
| Tensor-page | 4 | 126.014 | 100.619 | 25.350 | 4.071 | 562.56 | 7.90 |
| Tensor-page | 8 | 115.764 | 102.682 | 13.036 | 4.131 | 1157.85 | 7.47 |
| Tensor-page | **16** | **103.438** | 96.818 | 6.577 | 4.045 | 2548.74 | 8.62 |
| Tensor-page | 32/64/128 | infeasible | - | - | - | - | - |
| Compact-page | 1 | 192.856 | 94.998 | 97.815 | 3.981 | 0.51 | 15.44 |
| Compact-page | 2 | 143.503 | 94.740 | 48.720 | 3.965 | 1.22 | 11.48 |
| Compact-page | 4 | 123.441 | 98.896 | 24.501 | 3.987 | 4.02 | 11.04 |
| Compact-page | 8 | 111.265 | 98.956 | 12.264 | 3.990 | 5.07 | 10.11 |
| Compact-page | 16 | 105.213 | 99.039 | 6.129 | 3.987 | 8.84 | 9.00 |
| Compact-page | **32** | **97.940** | 94.802 | 3.094 | 4.026 | 11.10 | 8.10 |
| Compact-page | 64 | 100.835 | 99.253 | 1.538 | 3.996 | 27.49 | 7.25 |
| Compact-page | 128 | 101.284 | 100.462 | 0.778 | 4.026 | 55.73 | 5.85 |

Tensor-page becomes infeasible at 32 MiB because per-TensorGroup rounding
makes model 5 alone require 45,566,918,656 bytes before KV, exceeding the
42-GiB pool. Compact-page rounds only the complete packed model arena, so all
page sizes remain feasible. Compact-page is better at 1--8 MiB; at 16 MiB,
Tensor-page is 1.69% lower in this short trace, but cannot continue to the
32-MiB Compact-page optimum. The selected-candidate MCKP CPU column is not
included in exposed stall and still excludes rejected routing candidates.

Reproduction and source data:

```text
evaluation/tensor_simulator/experiments/run_sgrp_backend_page_size_sweep.py
evaluation/tensor_simulator/experiments/summarize_sgrp_backend_page_size_sweep.py
doc/results/sgrp-backend-page-size-fast200-seed1236/summary.csv
```

## Best-backend-configuration cache-pressure sweep

This formal run fixes seed 1236 and 1000 requests at every pressure level,
with a 32-request warmup. It runs 21 jobs: copies 1--7 times Segment,
Tensor-page 16 MiB, and Compact-page 32 MiB. All use Joint-MCKP, four GPUs,
two routing candidates/request, 42 GiB/GPU, input length 256, and K=64.

Mean GPU-side exposed loading stall:

| Copies | Pressure | Segment | Tensor-page 16 MiB | Compact-page 32 MiB | Winner |
|---:|---:|---:|---:|---:|---|
| 1 | 0.329x | 22.080 | 33.610 | **20.593** | Compact-page |
| 2 | 0.659x | 213.187 | 223.971 | **200.867** | Compact-page |
| 3 | 0.988x | **318.361** | 351.517 | 334.451 | Segment |
| 4 | 1.317x | 378.457 | 395.031 | **364.939** | Compact-page |
| 5 | 1.647x | 417.324 | 429.585 | **416.262** | Compact-page |
| 6 | 1.976x | 431.093 | 434.343 | **424.576** | Compact-page |
| 7 | 2.306x | **442.174** | 455.138 | 443.253 | Segment |

Compact-page wins five of seven levels; Segment wins c3 and c7, while
Tensor-page wins none. The result is not monotonic because each backend-aware
MCKP-transition run follows its own routing and residency trajectory. At c7,
Segment and Compact-page differ by only 0.24%.

The selected-candidate MCKP solver CPU grows with catalog size and is not
included above. At c7 it is 48.95, 40.85, and 50.57 ms/request for Segment,
Tensor-page, and Compact-page respectively. These values still omit rejected
routing candidates, so they are not total controller CPU cost.

Reproduction and source data:

```text
evaluation/traces/sgrp-backend-pressure-c1-c7-r1000-seed1236/
evaluation/tensor_simulator/experiments/run_sgrp_backend_pressure_sweep.py
evaluation/tensor_simulator/experiments/summarize_sgrp_backend_pressure_sweep.py
doc/results/sgrp-backend-pressure-c1-c7-r1000-seed1236/summary.csv
```

## Segment PBP planner CPU smoke test

The simulator now records `pbp_planner_time_ms` around only Segment's request-
level sorting, partitioning, local extent merge, and reserve operations. It
excludes Prefix-MCKP, modeled GPU relocation/compaction, H2D, and unrelated
Python bookkeeping.

A 200-request ServeGen smoke test uses the same Segment/Joint-MCKP settings as
the backend suite and excludes the first 32 requests:

| Population | Requests | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| All post-warmup requests | 168 | 0.860 | 0 | 3.418 | 4.678 | 5.283 |
| Requests executing PBP | 82 | 1.762 | 1.566 | 4.088 | 4.903 | 5.283 |

For comparison, selected-candidate MCKP solver CPU averages 10.073 ms/request
in the same smoke test. `pbp_planner_time_ms` is still the selected routing
candidate's planner time; it does not sum PBP work performed for rejected GPU
candidates.

Source result:

```text
doc/results/sgrp-segment-pbp-smoke-r200-seed1236.json
```

### L40 relocation-bandwidth default

The tensor simulator's default `--gpu-copy-gbps` is now 432 rather than 864.
L40's 864-GB/s peak device-memory bandwidth counts memory traffic, while a
relocation copies each payload byte with one read and one write; 432 GB/s is
therefore the optimistic payload-throughput ceiling. An explicit command-line
value still overrides the default. Results generated before this change used
864 GB/s unless their command specified otherwise and are not silently
rewritten.

## Segment PBP versus naive internal movement smoke test

This 200-request ServeGen smoke test excludes the first 32 requests and
defines a model reload as a request missing at least one TensorGroup. PBP uses
the actual `compaction_moved_bytes`. The requested naive baseline snapshots,
before MCKP reclaim, the sum of all resident parameter bytes belonging to
other models and moves that entire amount on every reload.

| Model | Type | Reloads | PBP GiB/reload | Naive GiB/reload | Reduction |
|---:|---|---:|---:|---:|---:|
| 0 | language/m-small | 7 | 9.410 | 41.161 | 77.14% |
| 1 | language/m-small | 13 | 3.570 | 36.718 | 90.28% |
| 2 | language/m-mid | 23 | 5.808 | 30.308 | 80.83% |
| 3 | language/m-mid | 16 | 8.089 | 33.500 | 75.85% |
| 4 | language/m-large | 8 | 3.469 | 27.339 | 87.31% |
| 5 | language/m-large | 6 | 4.469 | 27.493 | 83.74% |
| 6 | multimodal/mm-image | 0 | N/A | N/A | N/A |
| 7 | reason/deepseek-r1 | 4 | 6.689 | 33.258 | 79.89% |
| **All reloads** | - | **77** | **5.930** | **32.666** | **81.85%** |

PBP performs zero internal movement on 29 of the 77 reloads because existing
free extents already satisfy placement. Per-model means are conditional on a
reload and should not be interpreted as averages over all requests.

Source result:

```text
doc/results/sgrp-segment-pbp-vs-naive-movement-r200-seed1236.json
```

### 1000-request rerun

The formal rerun keeps the same configuration, uses 1000 requests, and
excludes the first 32 requests. All means remain conditional on a model reload.

| Model | Type | Requests | Reloads | Reload rate | PBP GiB/reload | Naive GiB/reload | Reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | language/m-small | 207 | 29 | 14.01% | 15.529 | 40.853 | 61.99% |
| 1 | language/m-small | 171 | 101 | 59.06% | 2.450 | 36.216 | 93.23% |
| 2 | language/m-mid | 283 | 127 | 44.88% | 5.854 | 29.849 | 80.39% |
| 3 | language/m-mid | 199 | 72 | 36.18% | 6.243 | 33.468 | 81.35% |
| 4 | language/m-large | 51 | 34 | 66.67% | 3.497 | 30.918 | 88.69% |
| 5 | language/m-large | 41 | 32 | 78.05% | 3.606 | 28.320 | 87.27% |
| 6 | multimodal/mm-image | 5 | 4 | 80.00% | 7.652 | 41.862 | 81.72% |
| 7 | reason/deepseek-r1 | 11 | 11 | 100.00% | 4.947 | 38.415 | 87.12% |
| **All** | - | **968** | **410** | **42.36%** | **5.390** | **33.148** | **83.74%** |

PBP requires zero internal movement on 172 of the 410 reloads. The four and
eleven reload samples for models 6 and 7 remain small, so their per-model means
should not be interpreted as stable model-specific estimates.

Source result:

```text
doc/results/sgrp-segment-pbp-vs-naive-movement-r1000-seed1236.json
```
