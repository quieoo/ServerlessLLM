# HF exclusive vs. VMM reuse

This is the minimal two-backend experiment. Run the backends in separate
processes so the VMM physical pool does not consume memory during the native HF
baseline.

## Build

```bash
cd /mnt/n0/Tangram/Tangram
cmake -S tools/mock_allocation -B tools/mock_allocation/build
cmake --build tools/mock_allocation/build --target tangram_vram_backend -j
```

Install/build `sllm_store` in the same Python environment used for the VMM run.
The VMM loader reuses `sllm_store._C.restore_tensors` and does not require a
running SLLM Store server.

## Run

Set the two paths once. `HF_MODEL` is the original Hugging Face checkpoint;
`STORE_MODEL` is the corresponding Tangram checkpoint produced by
`save_tensor_group_dict`, including config/tokenizer files.

```bash
source /mnt/n0/uv_envs/loaderbench/bin/activate

export HF_MODEL=/absolute/path/to/hf-model
export STORE_MODEL=/absolute/path/to/tangram-model
export BENCH=/mnt/n0/Tangram/Tangram/tools/mock_allocation/transformers_vmm_bench.py
export VMM_LIB=/mnt/n0/Tangram/Tangram/tools/mock_allocation/build/libtangram_vram_backend.so

python "$BENCH" \
  --backend hf_exclusive \
  --hf-model "$HF_MODEL" \
  --mode forward \
  --warmup 10 \
  --iterations 50 \
  | tee hf_exclusive.log

python "$BENCH" \
  --backend vmm_reuse \
  --hf-model "$HF_MODEL" \
  --store-model "$STORE_MODEL" \
  --vmm-library "$VMM_LIB" \
  --pool-gib 24 \
  --mode forward \
  --warmup 10 \
  --iterations 50 \
  | tee vmm_reuse.log
```

Choose `--pool-gib` large enough for the complete Tangram checkpoint plus VMM
page rounding. Both commands print one machine-readable
`TANGRAM_BENCH_RESULT=<json>` line. Compare `latency_ms_median` first; use
`latency_ms_samples` for confidence intervals. Keep model, GPU, dtype, prompt,
warmup count, iteration count, clocks, and process startup conditions identical.

For decode performance, repeat both commands with:

```bash
--mode generate --max-new-tokens 32
```

The first experiment only establishes whether VMM-backed parameter placement
changes inference time. CUDA VMM does not reveal GPU physical addresses, so it
does not by itself prove a physical-address-fragmentation mechanism.
