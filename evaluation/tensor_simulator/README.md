# Tangram Tensor-level Simulator

This directory contains the profile-driven CPU simulator and the experiment
drivers used for the Tensor-level evaluation. It does not require CUDA, vLLM,
PyTorch, NumPy, Pandas, or Matplotlib.

## Layout

- `layerweave_tensor_pipeline_sim.py`: simulator entrypoint and multi-GPU
  routing/system replay.
- `tensor_sim_memory_common.py`: shared TensorGroup and memory-cost types.
- `tensor_sim_segment_backend.py`: Segment/PBP/compaction backend.
- `tensor_sim_page_backends.py`: Tensor-page and Compact-page backends.
- `tensor_sim_mckp.py`: prefix-stall lookup and covering-MCKP solver.
- `simulator_io.py`: request type plus trace, config, and legacy compute-profile
  readers.
- `statistics_utils.py`: percentile and summary helpers.
- `compute_features.py`: prefill regression features used by the optional
  legacy compute-profile path.
- `export_layerweave_tensor_layout.py`: build a Tensor layout from
  safetensors headers.
- `profile_layerweave_tensor_compute.py`: optional GPU GEMM proxy profiler;
  it is not required by the profile-driven paper runs.
- `experiments/`: trace generators, parallel runners, summarizers, and
  dependency-free SVG plotting scripts.

## Requirements

The simulator and paper experiment scripts require Python 3.10 or newer and
only the Python standard library. The optional layout exporter reads model
safetensors files directly. The optional Tensor compute profiler additionally
requires the GPU software environment used by Tangram.

The paper runs use repository inputs rather than installed Python packages:

- `evaluation/tensor_simulator/results/tensor-layout.json`
- `evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json`
- `configs/servegen_8_models_layerpipe_l40_pool42.json`
- traces under `evaluation/traces/`

Run commands from the repository root (`Tangram/Tangram`). For exact
reproduction of the existing results, use
`/home/sdu/.conda/envs/sllm-worker/bin/python`; a clean Python 3.10+
interpreter is sufficient for CPU simulation.

## Validation

```bash
cd evaluation/tensor_simulator
python -m unittest \
  test_layerweave_tensor_pipeline_sim.py test_tensor_sim_mckp.py \
  test_simulator_utils.py
```

## Revised paper evaluation

```bash
python evaluation/tensor_simulator/experiments/generate_sgrp_revised_inputs.py \
  --output-dir evaluation/traces/sgrp-revised-r1000 \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --stall-table \
    evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json

python evaluation/tensor_simulator/experiments/run_sgrp_revised_eval.py \
  --trace-dir evaluation/traces/sgrp-revised-r1000 \
  --output-dir evaluation/tensor_simulator/results/sgrp-revised-r1000

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

The CUDA readiness-hook/event, ODKV, and measured PSE experiments are real-GPU
validation and intentionally remain outside this simulator directory.
