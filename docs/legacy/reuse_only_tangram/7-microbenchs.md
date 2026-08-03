
# UM Prefetch Benchmark

编译：
````bash
cd /mnt/n0/Tangram/Tangram/tools/mock_allocation
cmake --build build --target um_prefetch_bench
````


Tangram:
````bash
# 单GPU
GPU_NUM=1 /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/1-overall.sh

MAX_REQUESTS=1000 GPU_NUM=1 GPU_POOL_SIZE=43 /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/1-overall.sh

MAX_REQUESTS=100 GPU_NUM=1 GPU_POOL_SIZE=43 /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/1-overall.sh
MAX_REQUESTS=100 GPU_NUM=2 GPU_POOL_SIZE=43 /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/1-overall.sh
MAX_REQUESTS=100 GPU_NUM=4 GPU_POOL_SIZE=43 /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/1-overall.sh
````
```
================ Model Load Summary ================
Model                                                                        Count   Avg Latency(ms)
----------------------------------------------------------------------------------------------------
/mnt/n0/models/vllm/qwen2_3b_tmp/rank_0                                        212           142.989
/mnt/n0/models/vllm/phi3mini4k_tmp/rank_0                                      177           192.605
/mnt/n0/models/vllm/llama3_chinese_tmp/rank_0                                  292           106.885
/mnt/n0/models/vllm/yi_9b_tmp/rank_0                                           201           231.300
/mnt/n0/models/vllm/qwen2_14b_tmp/rank_0                                        56           683.610
/mnt/n0/models/vllm/gpt20b_tmp/rank_0                                           43          1232.991
/mnt/n0/models/vllm/llava7b_tmp/rank_0                                           6           289.829
/mnt/n0/models/vllm/qwenr14b_tmp/rank_0                                         13          1156.597
----------------------------------------------------------------------------------------------------
Overall                                                                       1000           250.182
GPU: 0, total_move_data_volume: 2198851940352
Model: /mnt/n0/models/vllm/gpt20b_tmp/rank_0, avg_move_data_volume_per_load: 2.73474e+08, total_move_data_volume: 11759382528, load_count: 43
Model: /mnt/n0/models/vllm/yi_9b_tmp/rank_0, avg_move_data_volume_per_load: 2.89219e+09, total_move_data_volume: 581331007488, load_count: 201
Model: /mnt/n0/models/vllm/phi3mini4k_tmp/rank_0, avg_move_data_volume_per_load: 1.88603e+09, total_move_data_volume: 333827031040, load_count: 177
Model: /mnt/n0/models/vllm/qwen2_3b_tmp/rank_0, avg_move_data_volume_per_load: 2.31259e+09, total_move_data_volume: 490269370368, load_count: 212
Model: /mnt/n0/models/vllm/qwen2_14b_tmp/rank_0, avg_move_data_volume_per_load: 5.27141e+09, total_move_data_volume: 295199148032, load_count: 56
Model: /mnt/n0/models/vllm/qwenr14b_tmp/rank_0, avg_move_data_volume_per_load: 5.77363e+09, total_move_data_volume: 75057215488, load_count: 13
Model: /mnt/n0/models/vllm/llava7b_tmp/rank_0, avg_move_data_volume_per_load: 1.9103e+09, total_move_data_volume: 11461801984, load_count: 6
Model: /mnt/n0/models/vllm/llama3_chinese_tmp/rank_0, avg_move_data_volume_per_load: 1.36968e+09, total_move_data_volume: 399946983424, load_count: 292
====================================================
Model                                                                       Total Time      Merge Time   Allocate Time       Load Time       Drop Time
------------------------------------------------------------------------------------------------------------------------------------------------------
/mnt/n0/models/vllm/llama3_chinese_tmp/rank_0                                  106.589           4.154           0.000         102.226           0.003
/mnt/n0/models/vllm/llava7b_tmp/rank_0                                         289.167           6.167           0.167         282.500           0.000
/mnt/n0/models/vllm/qwenr14b_tmp/rank_0                                       1155.769          17.538           0.000        1137.231           0.000
/mnt/n0/models/vllm/qwen2_14b_tmp/rank_0                                       682.893          16.482           0.000         665.786           0.000
/mnt/n0/models/vllm/qwen2_3b_tmp/rank_0                                        142.387           7.236           0.000         134.533           0.061
/mnt/n0/models/vllm/phi3mini4k_tmp/rank_0                                      192.034           5.757           0.000         185.729           0.000
/mnt/n0/models/vllm/yi_9b_tmp/rank_0                                           230.821           8.861           0.000         221.637           0.000
/mnt/n0/models/vllm/gpt20b_tmp/rank_0                                         1232.093           0.814           0.000        1230.558           0.000
==================================
Reduced IO: 8691492151296 / 14790913032192 = 0.587624
Model: /mnt/n0/models/vllm/gpt20b_tmp/rank_0 Reduced IO: 433386110976 / 1767692820480 = 0.24517
Model: /mnt/n0/models/vllm/yi_9b_tmp/rank_0 Reduced IO: 2428867215360 / 3549421707264 = 0.684299
Model: /mnt/n0/models/vllm/phi3mini4k_tmp/rank_0 Reduced IO: 527849484288 / 1352662161408 = 0.39023
Model: /mnt/n0/models/vllm/qwen2_3b_tmp/rank_0 Reduced IO: 596086181888 / 1308438003712 = 0.455571
Model: /mnt/n0/models/vllm/qwen2_14b_tmp/rank_0 Reduced IO: 714301747200 / 1654243770368 = 0.4318
Model: /mnt/n0/models/vllm/qwenr14b_tmp/rank_0 Reduced IO: 10968563712 / 384020875264 = 0.0285624
Model: /mnt/n0/models/vllm/llava7b_tmp/rank_0 Reduced IO: 42380562432 / 84761124864 = 0.5
Model: /mnt/n0/models/vllm/llama3_chinese_tmp/rank_0 Reduced IO: 3937652285440 / 4689672568832 = 0.839643
[INFO] Clean Registered Model:/mnt/n0/models/vllm/qwenr14b_tmp/rank_0
[INFO] Clean Registered Model:/mnt/n0/models/vllm/llava7b_tmp/rank_0
[INFO] Clean Registered Model:/mnt/n0/models/vllm/gpt20b_tmp/rank_0
[INFO] Clean Registered Model:/mnt/n0/models/vllm/qwen2_14b_tmp/rank_0
====== End ======
All done
```


````bash

MODE_LIST="explicit-copy" SCENARIO_LIST="flush-each-time" GPU_ID=0 GPU_NUM=1 MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh
VERBOSE=1 GPU_ID=0 GPU_NUM=1 MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh

nohup bash -c '
MAX_REQUESTS=100 GPU_NUM=1 GPU_POOL_SIZE=43 /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/1-overall.sh
MAX_REQUESTS=100 GPU_NUM=2 GPU_POOL_SIZE=43 /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/1-overall.sh
MAX_REQUESTS=100 GPU_NUM=3 GPU_POOL_SIZE=43 /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/1-overall.sh
GPU_ID=0 GPU_NUM=1 MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh
GPU_ID=0 GPU_NUM=2 MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh
GPU_ID=0 GPU_NUM=3 MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh
' > /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/microbench_um_prefetch_100.log 2>&1 &

nohup bash -c '
GPU_ID=0 GPU_NUM=1 MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh
GPU_ID=0 GPU_NUM=2 MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh
GPU_ID=0 GPU_NUM=2 GPU_ASSIGNMENT_MODE=round_robin MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh
GPU_ID=0 GPU_NUM=3 GPU_ASSIGNMENT_MODE=round_robin MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh

GPU_ID=0 GPU_NUM=2 GPU_ASSIGNMENT_MODE=sticky MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh
GPU_ID=0 GPU_NUM=3 GPU_ASSIGNMENT_MODE=sticky MAX_REQUESTS=100 PREPARE_THREADS=64 /mnt/n0/Tangram/Tangram/evaluation/microbench/run_um_prefetch.sh


' >> /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/microbench_um_prefetch_100.1.log 2>&1 &
````





# 内存碎片

# Memory Coalescing Benchmark

目标：

- `tensor-level` 作为 baseline，参数张量连续存放。
- `page-level-random` 将同一逻辑 tensor 的 page 映射到随机离散物理页。
- 通过 sweep `page_size` 观察连续性下降对 LLM 典型 kernel 的影响。

编译：
````bash
cd /mnt/n0/Tangram/Tangram/tools/mock_allocation
cmake -S . -B build
cmake --build build --target memory_coalescing_bench
````

一键跑完整 sweep：
````bash
nohup bash -c '
GPU_ID=0 \
KERNEL_LIST="attn_proj_gemm ffn_up_gemm decode_qk rmsnorm" \
PAGE_SIZE_KB_LIST="4 16 64 256 1024 2048" \
/mnt/n0/Tangram/Tangram/evaluation/microbench/memory_coalescing.sh
' > /mnt/n0/Tangram/Tangram/docs/legacy/reuse_only_tangram/microbench_memory_coalescing.log 2>&1 &

GPU_ID=0 \
KERNEL_LIST="attn_proj_gemm ffn_up_gemm decode_qk rmsnorm" \
PAGE_SIZE_KB_LIST="2048" \
/mnt/n0/Tangram/Tangram/evaluation/microbench/memory_coalescing.sh
````

输出：

- 汇总 CSV：`evaluation/log/memory_coalescing/summary.csv`
- 每行对应一个 `(layout, kernel, page_size)` 组合。
- 重点关注 `avg_ms`、`arithmetic_intensity`、`tflops`。
- `attn_proj_gemm`：prefill 阶段的 attention/output projection 类 GEMM。
- `ffn_up_gemm`：prefill 阶段的 FFN up projection GEMM。
- `decode_qk`：decode 阶段单 query 对 KV cache 的 QK 打分。

单次运行示例：
````bash
/mnt/n0/Tangram/Tangram/tools/mock_allocation/build/memory_coalescing_bench \
  --gpu 0 \
  --layout tensor-level \
  --kernel rmsnorm \
  --page_size_kb 64 \
  --batch_size 1 \
  --seq_len 2048 \
  --hidden_size 4096 \
  --intermediate_size 11008 \
  --num_heads 32 \
  --head_dim 128 \
  --warmup_iters 20 \
  --measure_iters 100 \
  --csv /mnt/n0/Tangram/Tangram/evaluation/log/memory_coalescing/summary.csv
````


# Loader Backend

测试不同的 loader backend 对模型加载时间的影响。
- VLLM default loader
- RunAI Streamer
- Fastsafetensors
- Tensorizer
- SLLM loader (ours)

模型路径：
- 原始模型：/mnt/n0/models/qwen2-14
- SLLM格式化后的模型根目录：/mnt/n0/models/vllm/qwen2_14b_tmp

````bash

# 准备环境
mkdir -p /mnt/n0/uv_envs
uv venv /mnt/n0/uv_envs/loaderbench --python 3.12 --seed
source /mnt/n0/uv_envs/loaderbench/bin/activate

UV_TORCH_BACKEND=cu124 uv pip install 'vllm[runai,tensorizer]' --torch-backend=cu124
uv pip install fastsafetensors

uv pip install --force-reinstall 'transformers>=4.52.1,<4.53'
uv pip install --force-reinstall 'numpy<2.3'

python - <<'PY'
import transformers, tokenizers, torch, vllm
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("torch", torch.__version__, torch.version.cuda)
print("vllm", vllm.__version__)
PY
python - <<'PY'
import numpy, numba
print("numpy", numpy.__version__)
print("numba", numba.__version__)
PY
python -c "import tensorizer; print('tensorizer ok')"
python -c "import fastsafetensors; print('fastsafetensors ok')"


# 生成 Tensorizer artifact
python /mnt/n0/Tangram/Tangram/evaluation/microbench/serialize_tensorizer_artifact.py \
  --model /mnt/n0/models/qwen2-14 \
  --dtype float16 \
  --enforce-eager \
  --max-model-len 32 \
  --serialized-directory /mnt/n0/models/tensorizer \
  --suffix qwen2_14b_fp16


# 运行四种 loader backend 对比
python /mnt/n0/Tangram/Tangram/evaluation/microbench/bench_loader_backends.py \
  --backends vllm_default tensorizer runai_streamer \
  --hf-model-path /mnt/n0/models/qwen2-14 \
  --tensorizer-uri /mnt/n0/models/tensorizer/vllm/qwen2-14/qwen2_14b_fp16/model.tensors \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 32 \
  --enforce-eager \
  --repeats 2 \
  --runai-extra-config-json '{"concurrency":64,"memory_limit":53687091200}' \
  --tag qwen2_14b_all_backends

````

## SLLM

````bash
CUDA_VISIBLE_DEVICES=2 /mnt/n0/Tangram/Tangram/tools/mock_allocation/build/sllm_load_path_bench \
  --model-path /mnt/n0/models/vllm/qwen2_14b_tmp/rank_0 \
  --mode sllm \
  --gpu-id 0 \
  --repeats 3 \
  --pipeline-depth 2

CUDA_VISIBLE_DEVICES=2 /mnt/n0/Tangram/Tangram/tools/mock_allocation/build/sllm_load_path_bench \
  --model-path /mnt/n0/models/vllm/qwen2_14b_tmp/rank_0 \
  --mode sllm_cpu_only \
  --gpu-id 0 \
  --repeats 3
````
