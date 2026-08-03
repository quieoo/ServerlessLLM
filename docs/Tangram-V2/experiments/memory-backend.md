# Memory Backend Evaluation

## Compared backends

| Backend | Allocation unit | Main extra cost |
|---|---|---|
| Segment | TensorGroup segment | PBP/compaction |
| Tensor-page | Page-aligned tensor allocation | Internal fragmentation and map/unmap |
| Compact-page | Packed tensors sharing pages | Coarser eviction and map/unmap |

The backend comparison fixes the cache/routing policy and workload so that the
result measures memory representation rather than a policy change.

## Six-workload suite

```bash
python evaluation/tensor_simulator/experiments/run_sgrp_backend_workload_suite.py \
  --output-dir evaluation/tensor_simulator/results/sgrp-backend-suite-r1000-seed1236 \
  --requests 1000 \
  --max-parallel 18

python evaluation/tensor_simulator/experiments/summarize_sgrp_backend_workload_suite.py \
  --result-dir evaluation/tensor_simulator/results/sgrp-backend-suite-r1000-seed1236 \
  --csv-output evaluation/tensor_simulator/results/sgrp-backend-suite-r1000-seed1236/summary.csv \
  --svg-output evaluation/tensor_simulator/results/sgrp-backend-suite-r1000-seed1236/backend-comparison.svg
```

Use the optimized ServeGen rerun as the current implementation result:

```text
evaluation/tensor_simulator/results/
  sgrp-backend-suite-r1000-seed1236-optimized/servegen-summary.csv
```

## Page-size sweep

```bash
python evaluation/tensor_simulator/experiments/run_sgrp_backend_page_size_sweep.py \
  --output-dir evaluation/tensor_simulator/results/sgrp-backend-page-size-fast200-seed1236 \
  --requests 200 \
  --page-sizes 1,2,4,8,16,32,64,128

python evaluation/tensor_simulator/experiments/summarize_sgrp_backend_page_size_sweep.py \
  --result-dir evaluation/tensor_simulator/results/sgrp-backend-page-size-fast200-seed1236 \
  --csv-output evaluation/tensor_simulator/results/sgrp-backend-page-size-fast200-seed1236/summary.csv
```

## Cache-pressure sweep with best page sizes

The runner uses Segment, 16 MiB Tensor-page, and 32 MiB Compact-page.

```bash
python evaluation/tensor_simulator/experiments/run_sgrp_backend_pressure_sweep.py \
  --trace-dir evaluation/traces/sgrp-backend-pressure-c1-c7-r1000-seed1236 \
  --output-dir evaluation/tensor_simulator/results/sgrp-backend-pressure-c1-c7-r1000-seed1236 \
  --copies 1,2,3,4,5,6,7

python evaluation/tensor_simulator/experiments/summarize_sgrp_backend_pressure_sweep.py \
  --trace-dir evaluation/traces/sgrp-backend-pressure-c1-c7-r1000-seed1236 \
  --result-dir evaluation/tensor_simulator/results/sgrp-backend-pressure-c1-c7-r1000-seed1236 \
  --csv-output evaluation/tensor_simulator/results/sgrp-backend-pressure-c1-c7-r1000-seed1236/summary.csv
```

## Reported decomposition

Report exposed H2D, Segment compaction, page map/unmap, and MCKP solver CPU
cost separately. The old label "Allocation" referred to solver overhead and
should not be used because it is ambiguous.
