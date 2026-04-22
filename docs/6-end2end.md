
## quick start two models
````bash
sllm-cli deploy --config sllm_opt2.7tmp.json 
sllm-cli deploy --config sllm_opt6.7tmp.json
nohup sllm-cli generate opt2.7tmp_input.json >> 2.7.log 2>&1 &
nohup sllm-cli generate opt6.7tmp_input.json >> 2.7.log 2>&1 &

````


## start models with criu backend

changing "server.py" to switch reuse mode: 
````bash
load_strategy=0 (w/o reuse)
or
load_strategy=4 (Reuse)
````

start the controller node worker nodes and Reuse Store backend (hear we run four workers on a single machine, so distinguish them by different ports):

IMPORTANT：Make sure the worker and stores are started in the directory with "models" in it. Otherwise, the model registration will failed to locate the model files.



````bash
# controller
conda activate sllm-0.6
ray start --head --port=6379 --num-cpus=64 --num-gpus=0 --resources='{"control_node": 1}' --block


# worker-0~7
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=0
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1, "store_port":8073}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_1": 1, "store_port":8074}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=2
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_2": 1, "store_port":8075}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=3
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_3": 1, "store_port":8076}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=4
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_4": 1, "store_port":8077}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=5
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_5": 1, "store_port":8078}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=6
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_6": 1, "store_port":8079}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=7
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_7": 1, "store_port":8080}' --block


# store-0~7
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=0
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8073

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8074

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=2
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8075

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=3
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8076

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=4
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8077

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=5
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8078

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=6
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8079

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=7
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8080

# sllm 
conda activate sllm-0.6
sllm-serve start --enable_storage_aware

````


````bash
conda activate sllm-0.6
cd ServerlessLLM/serverless_scripts/

# batch_size=1, request_length=0, output_length=100, qps=0.4, n=30, req_path=4090_small
sllm-cli deploy --config mock_opt1.3.json
sllm-cli deploy --config mock_opt2.7.json
sllm-cli deploy --config mock_qwen3b.json
sllm-cli deploy --config mock_llama3b.json
sllm-cli deploy --config mock_llama8b.json
sllm-cli deploy --config mock_yi9b.json

sllm-cli deploy --config mock_opt13.json
sllm-cli deploy --config mock_qwen14.json

# simple request
curl http://127.0.0.1:8343/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "llama2_3b_tmp",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name and how are you ?"} 
        ]
    }'


# complex request
# fixed rps but varies gpu_num
python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --max_tokens=50 --batch_size=1 --n=100 --qps=0.4
python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --max_tokens=50 --batch_size=1 --n=200 --qps=1.6
python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --max_tokens=50 --batch_size=1 --n=300 --qps=3.2


python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090-cv1-large.txt --max_tokens=50 --batch_size=1 --n=200 --qps=0.8

python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090-cv1-large.txt --max_tokens=50 --batch_size=1 --n=400 --qps=2.4



# 不同workload的影响
python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --max_tokens=50 --batch_size=1 --n=400 --qps=2.4

````
### Run ServeGen Workload

````bash
# 使用新的ServeGen配置 （固定qps）
python benchmark_v2.py \
  --type ServeGen \
  --trace_file_path=/mnt/n0/sslm/ServerlessLLM/evaluation/traces/servegen_tangram.trace \
  --model_config_path=/mnt/n0/sslm/ServerlessLLM/configs/servegen_8_end2end.json \
  --ignore_trace_timestamps \
  --qps=2.4 \
  --batch_size=1 \
  --n=1000

# ServeGen配置 （默认时间戳） 
nohup python benchmark_v2.py \
  --type ServeGen \
  --trace_file_path=../evaluation/traces/end2end/servegen_tangram_end2end_0p4.trace \
  --model_config_path=../configs/servegen_8_end2end.json \
  --batch_size=1 \
  --n=500 > servegen_end2end_n8_0p4.log 2>&1 &


nohup python benchmark_v2.py \
  --type ServeGen \
  --trace_file_path=../evaluation/traces/end2end/servegen_tangram_end2end_1p6.trace \
  --model_config_path=../configs/servegen_8_end2end.json \
  --batch_size=1 \
  --n=500 > servegen_end2end_n8_1.6.log 2>&1 &

nohup python benchmark_v2.py \
  --type ServeGen \
  --trace_file_path=../evaluation/traces/end2end/servegen_tangram_end2end_1p6.trace \
  --model_config_path=../configs/servegen_8_end2end.json \
  --batch_size=1 \
  --ignore_trace_timestamps \
  --qps=1.6 \
  --n=500 > servegen_end2end_n8_1.6.log 2>&1 &

nohup python benchmark_v2.py \
  --type ServeGen \
  --trace_file_path=../evaluation/traces/end2end/servegen_tangram_end2end_0p4.trace \
  --model_config_path=../configs/servegen_8_end2end.json \
  --batch_size=1 \
  --n=500 > servegen_end2end_n8_0p4.log 2>&1 &

````
### Run ServeGen Workload Without Ray Backend

The end-to-end path above requires Ray workers, the ServerlessLLM controller, and
vLLM backend initialization. For Store-only scalability tests, use the standalone
backend benchmark below. It directly calls the same Reuse Store gRPC path used by
the Ray vLLM backend:

- `LoadModelAsync(..., DEVICE_TYPE_CPU, replica_uuid=device_id)` measures model
  loading against the selected Store.
- `ToLoadSize` is used to choose a Store under the `reuse_aware` policy.
- Prefill, engine overhead, and decode time are simulated to produce approximate
  TTFT.

Start only the 8 Reuse Stores first. Each `sllm-store start` is blocking, so
run them in separate terminals or under `nohup`/`tmux`:

````bash
conda activate sllm-worker-0.6

CUDA_VISIBLE_DEVICES=0 sllm-store start --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8073
CUDA_VISIBLE_DEVICES=1 sllm-store start --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8074
CUDA_VISIBLE_DEVICES=2 sllm-store start --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8075
CUDA_VISIBLE_DEVICES=3 sllm-store start --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8076
CUDA_VISIBLE_DEVICES=4 sllm-store start --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8077
CUDA_VISIBLE_DEVICES=5 sllm-store start --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8078
CUDA_VISIBLE_DEVICES=6 sllm-store start --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8079
CUDA_VISIBLE_DEVICES=7 sllm-store start --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8080
````

Then replay the ServeGen trace directly:

````bash
conda activate sllm-worker-0.6
cd /mnt/n0/sslm

python serverless_benchmark_backend.py \
  --trace-file-path ../evaluation/traces/end2end/servegen_tangram_end2end_1p6.trace \
  --model-config-path ../configs/servegen_8_end2end.json \
  --store-addresses 127.0.0.1:8074,127.0.0.1:8075,127.0.0.1:8076,127.0.0.1:8077,127.0.0.1:8078,127.0.0.1:8079,127.0.0.1:8080 \
  --policy reuse_aware \
  --occupy-until load \
  --engine-overhead-ms 20 \
  --prefill-ms-per-token 0.02 \
  --decode-ms-per-token 2.0 \
  --output-csv ../evaluation/log/serverless_backend_1p6.csv \
  --output-json ../evaluation/log/serverless_backend_1p6.json
````

Useful options:

- `--ignore-trace-timestamps --qps 2.4`: ignore trace timestamps and replay at
  fixed QPS.
- `--limit 100`: run a short subset for debugging.
- `--clear-mem`: call `ClearMem` on every Store before replay.
- `--policy round_robin`: skip `ToLoadSize`-based scheduling.
- `--occupy-until load|ttft|finish`: choose whether each Store is considered busy
  only during loading, through first token, or through the simulated full request.
- `--dry-run`: validate parsing and TTFT simulation without contacting Stores.



## Simulator

用法分两档：先跑一个快速 smoke test，确认链路正常；再跑完整实验画图。Simulator 支持两个后端：

- `BACKEND=python`：默认后端，使用 `simulate_end2end.py` 里的轻量级在线缓存模型。
- `BACKEND=cpp`：通过 C++ binding 调用真实 `VRAMManager`，使用真实的 tensor-group 缓存、驱逐、分配和碎片整理逻辑。

**1. 构建 C++ 后端**

如果只用默认 `BACKEND=python`，可以跳过这一步。使用 `BACKEND=cpp` 前需要构建 C++ binding：

```bash
cd /mnt/n0/Tangram/Tangram/tools/mock_allocation
cmake -S . -B build
cmake --build build --target tangram_vram_backend
```

确认 shared library 存在：

```bash
ls /mnt/n0/Tangram/Tangram/tools/mock_allocation/build/libtangram_vram_backend.so
```


**2. 快速测试：Python 后端**

从 Tangram 仓库根目录运行：

```bash
cd /mnt/n0/Tangram/Tangram
MAX_REQUESTS=1000 docs/6-end2end-sim.sh
```

这会做几件事：

```text
1. 读取已有 RPS traces；如果 GENERATE_TRACES=1，则先调用 generate_trace_tangram.py 生成 traces
2. 对每个 RPS/GPU/system 跑 end-to-end simulator
3. 写 per-request CSV 和 TTFT CDF
4. 汇总 summary.csv
5. 画 end2end_comparison.pdf
```

默认会跑：

```text
RPS: 0.4 0.8 1.2 1.6
GPU: 1 2 4 8
System: sllm_cm tangram
Backend: python
```

`MAX_REQUESTS=1000` 只是为了快。确认没问题后再跑完整。

**3. 快速测试：C++ VRAMManager 后端**

先构建 `tangram_vram_backend`，然后运行：

```bash
cd /mnt/n0/Tangram/Tangram
BACKEND=cpp MAX_REQUESTS=1000 docs/6-end2end-sim.sh
```

默认 C++ 后端使用 `mock_copy`，即使用真实 `VRAMManager` 的内存池状态、缓存命中、驱逐和分配策略，但不做真实 CUDA copy。这样适合端到端模拟，速度更快，也不需要真实读取模型参数到 GPU。

如果要测真实 CUDA allocation/copy，并且让 simulator 使用 C++ `LoadModel()` 的实测墙钟时间：

```bash
BACKEND=cpp \
CPP_REAL_COPY=1 \
CPP_LOAD_TIME_SOURCE=wall \
MAX_REQUESTS=1000 \
docs/6-end2end-sim.sh
```

常用 C++ 后端参数：

```bash
BACKEND=cpp \
CPP_FREE_STRATEGY=1 \
CPP_ALLOCATE_STRATEGY=4 \
CPP_LOAD_TIME_SOURCE=estimated \
MAX_REQUESTS=1000 \
docs/6-end2end-sim.sh
```

含义：

- `CPP_FREE_STRATEGY=1`：C++ `GreedyDrop`，按真实 `VRAMManager` 的 cost function 驱逐。
- `CPP_FREE_STRATEGY=0`：随机驱逐。
- `CPP_ALLOCATE_STRATEGY=4`：默认 `PartitionedBinPacking`。
- `CPP_ALLOCATE_STRATEGY=1/2/3`：分别使用 `GlobalMerge`、`GreedyMerge`、`WeightedBipartiteMatch_GreedyMerge`。
- `CPP_LOAD_TIME_SOURCE=estimated`：用 `to_load_bytes / LOAD_BANDWIDTH_GBPS + LOAD_OVERHEAD_MS` 作为调度里的加载时间，便于和 Python 后端对齐。
- `CPP_LOAD_TIME_SOURCE=wall`：用 C++ `LoadModel()` 调用的实测 wall time 作为加载时间。

**4. 跑完整实验**

```bash
cd /mnt/n0/Tangram/Tangram
MAX_REQUESTS=0 docs/6-end2end-sim.sh
```

输出位置：

```text
evaluation/log/end2end/summary.csv
evaluation/log/end2end/end2end_comparison.pdf
evaluation/log/end2end/per_request/
evaluation/log/end2end/ttft_cdf/
```

最终图就是：

```bash
open evaluation/log/end2end/end2end_comparison.pdf
```

或者在服务器上：

```bash
ls -lh evaluation/log/end2end/end2end_comparison.pdf
```

**常用参数**

只跑少量组合：

```bash
RPS_LIST="0.4 1.6" GPU_LIST="1 4 8" MAX_REQUESTS=1000 docs/6-end2end-sim.sh
```

不重新生成 trace，只复用已有 trace：

```bash
GENERATE_TRACES=0 MAX_REQUESTS=1000 docs/6-end2end-sim.sh
```

如果 trace 不存在，先生成 trace：

```bash
GENERATE_TRACES=1 MAX_REQUESTS=1000 docs/6-end2end-sim.sh
```

不画图，只生成数据：

```bash
PLOT=0 MAX_REQUESTS=1000 docs/6-end2end-sim.sh
```

指定 SLLM-CM keep alive：

```bash
SLLM_KEEP_ALIVE_MS=60000 MAX_REQUESTS=1000 docs/6-end2end-sim.sh
```

默认是：

```bash
SLLM_KEEP_ALIVE_MS=0
```

含义是请求完成后立即过期；设为负数表示永不过期。

调整加载带宽：

```bash
LOAD_BANDWIDTH_GBPS=24 MAX_REQUESTS=1000 docs/6-end2end-sim.sh
```

调整 prefill/decode 近似：

```bash
ENGINE_OVERHEAD_MS=20 \
PREFILL_MS_PER_TOKEN=0.02 \
DECODE_MS_PER_TOKEN=2.0 \
MAX_REQUESTS=1000 \
docs/6-end2end-sim.sh
```

**如果只想手动跑单个 case**

例如 `RPS=1.6, GPU=8, Tangram`：

```bash
cd /mnt/n0/Tangram/Tangram

python evaluation/end2end/simulate_end2end.py \
  --trace evaluation/traces/end2end/servegen_tangram_end2end_1p6.trace \
  --config configs/servegen_8_models.json \
  --system tangram \
  --backend cpp \
  --rps 1.6 \
  --num-gpus 8 \
  --gpu-memory-gb 40 \
  --load-bandwidth-gbps 20 \
  --cpp-backend-lib tools/mock_allocation/build/libtangram_vram_backend.so \
  --cpp-free-strategy 1 \
  --cpp-allocate-strategy 4 \
  --output-request-csv evaluation/log/end2end/per_request/tangram_rps1p6_gpu8.csv \
  --output-summary-csv evaluation/log/end2end/summary.csv \
  --append-summary
```

**注意**

脚本默认需要这个配置文件存在：

```text
configs/servegen_8_models.json
```

如果你的配置文件不在这里，可以这样指定：

```bash
CONFIG_PATH=/path/to/your_config.json MAX_REQUESTS=1000 docs/6-end2end-sim.sh
```

如果 trace 已经提前生成，也可以指定：

```bash
TRACE_BASE=/path/to/servegen_tangram_end2end.trace GENERATE_TRACES=0 docs/6-end2end-sim.sh
```

### TTFT-CDF 绘图

````bash
python evaluation/log/end2end/show_cdf.py \
  --gpu-num 4 \
  --rps-list "0.4,0.8,1.2,1.6" \
  --output evaluation/log/end2end/ttft_cdf_gpu4_selected.pdf \
  --slo-value 2500

python evaluation/log/end2end/show_cdf.py \
  --rps 0.4 \
  --gpu-list "1,2,4,8" \
  --output evaluation/log/end2end/ttft_cdf_rps0p4_selected.pdf \
  --slo-value 1500
python evaluation/log/end2end/show_cdf.py \
  --rps 1.6 \
  --gpu-list "1,2,4,8" \
  --output evaluation/log/end2end/ttft_cdf_rps1p6_selected.pdf \
  --slo-value 1500

````