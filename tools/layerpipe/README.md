# Real-GPU LayerPipe and LayerWeave

This directory contains the **real GPU implementation** of LayerPipe and
LayerWeave.  These programs execute real Transformer inference through
PyTorch/Transformers or the modified vLLM runtime, issue real CUDA H2D copies,
and measure request latency on GPU hardware.  They are not performance
simulators.

The tensor-level performance simulator is maintained separately under
`evaluation/tensor_simulator/`.

## Main entry points

- `run_layerpipe_real_gpu.sh`: single-GPU LayerPipe, VMM, and LayerWeave trace
  replay entry point.
- `layerpipe_bench.py`: Transformers-based LayerPipe baseline with layer-wise
  CPU-to-GPU streaming.
- `vllm_odkv_trace_bench.py`: vLLM VMM/ODKV/LayerWeave real-GPU trace runner.
- `run_layerweave_joint_ab_2gpu.sh`: paired two-GPU fixed-versus-joint run.
- `run_layerweave_multi_gpu_ab.sh`: multi-GPU routing/cache-policy A/B run.

## Documentation

- `REAL_GPU_LAYERPIPE.md`: quick start and execution semantics.
- `LAYERWEAVE_REAL_GPU.md`: LayerWeave design, implementation, and real-GPU
  experiment history.
- `results/`: generated M4/M5.5 real-GPU profiles and experiment outputs.

## External runtime components

The implementation deliberately keeps the following components in their
original locations:

- `ElasticKV/vllm/`: modified vLLM loader, hooks, and ODKV integration.
- `tools/mock_allocation/`: CUDA VMM backend and PyTorch binding.
- `configs/`: model and hardware-safe-limit configurations.
- `evaluation/traces/`: trace inputs.

Generated JSON, CSV, and log files are experiment artifacts rather than source
dependencies.  Existing commands continue to write them under `docs/` unless
an explicit output path is supplied.
