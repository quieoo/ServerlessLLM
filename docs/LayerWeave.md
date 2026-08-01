LayerWeave: Weaving Parameter Reuse into Pipelined Serverless LLM Loading

# Motivation
- 当前的计算-加载流水线不能总是掩盖加载时延
- 显存内存在可重用的参数
- ODKV可以进一步释放显存

# Design
- 基于VMM的参数重用
- 重用-计算-加载流水线
- 调度策略，缓存策略
具体方案[LayerWeave](../tangram_vmm_pipeline_design.md)

# Baselines
    - 纯计算-加载流水线
        模型切换，清理上一个模型，没有任何重用；
        做当前层的Prefill的同时预先加载下一层模型的参数
    - 纯 chunk-level重用
        建立VMM-based显存池，缓存其它模型的参数chunk
        调度时候看当前可用GPU里面缓存参数量最多的
    - 预取（Aegaeon）
        如果当前模型运行时显存还有剩余空间，提前加载调度器队首的下一个模型

# 实现

开发里程碑：
阶段	完成标准
M0 Layout	所有 tensor 都有唯一 VA range；每层 required pages 可验证；共享页处理正确
M1 Partial VMM	可以只加载指定 page range；未请求页面保持 unmapped；已有页不重复 H2D
M2 LayerWeave execution	LOAD_MODE=layerweave 正确完成 Prefill/Decode，输出与 native 一致
M3 Overlap validation	Nsight/事件时间证明 H2D 与 layer compute 真重叠；没有隐藏的全局 synchronize
M4 Estimator	对 cold、partial-hit、full-hit 三种状态预测 exposed latency，误差可量化
M5 Prefix cache	支持 0/2/4/8/16/full 降级，active model 永不误淘汰
M6 Scheduler	多 GPU 使用 predicted TTFT 选择，而不是 cached bytes
M7 Full evaluation	native、layerpipe、vmm-reuse、LayerWeave、prefetch 使用同一 trace/config/pool


# 实验

## 准备模型

模型组合：
| ID | 推荐模型 | 参数规模 | context | 当前代码架构支持 | 说明 |
|---:|---|---:|---:|---|---|
| 0 | 保持 Qwen2-3B | 3B | 32k | 是 | 已覆盖全部 trace |
| 1 | Phi-3-mini-128k | 3.8B | 128k | 是，`Phi3ForCausalLM` | 与当前模型最接近 |
| 2 | 保持 Llama3-Chinese-8B | 8B | 32k | 是 | 只截断约 0.1%；完全覆盖可换 Qwen2.5-7B-128k |
| 3 | 本地 Qwen-7B-long | 7B | 配置为 1M | 是，`Qwen2ForCausalLM` | 本地已有 packed checkpoint，最容易替换 |
| 4 | 保持 Qwen2-14B | 14B | 32k | 是 | 已覆盖全部 trace |
| 5 | InternLM2.5-20B | 20B | 256k | 是，`InternLM2ForCausalLM` | 参数规模最接近 GPT-20B |
| 6 | LLaVA-v1.6-Mistral-7B | 7B | 32k | 是，`LlavaNextForConditionalGeneration` | 保留多模态模型类型 |
| 7 | 保持 Qwen-14B-long | 14B | 128k | 是 | 已覆盖全部 trace |

将当前本地的rank_0/tensor.data_*这种 vLLM 打包格式转换成所需要的Hugging Face safetensors

转换命令：
```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/convert_servegen_config.py \
  --config configs/servegen_8_models.json \
  --output-root /mnt/n0/models/hf/servegen_8_models \
  --output-config configs/servegen_8_models_layerpipe.json \
  --max-shard-size 2GiB
```

准备模型可用显存空间
```bash
python tools/layerpipe/generate_l40_safe_limits.py \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --pool-gib 42.0 \
  --reserve-gib 0.5 \
  --gpu-memory-gib 44.98 \
  --runtime-reserve-gib 1.0 \
  --prefill-activation-safety-factor 2.5 \
  --write
```
如果仍然出现 Prefill OOM，可以：
更保守地将 --runtime-reserve-gib 调到 1.5
或将 --prefill-activation-safety-factor 调到 2.5


## baseline-naive
```bash
cd /mnt/n0/Tangram/Tangram

nohup env \
    CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
    LOAD_MODE=native \
    MAX_REQUESTS=100 \
    MAX_BATCH_SIZE=1 \
    OUTPUT_TOKENS_OVERRIDE=1 \
    TRACE_TIME_SCALE=150 \
    bash docs/1.2-layerpipe.sh \
    > docs/1.2-naive.log 2>&1 &
echo $!
```

## baseline-layerpipe
```bash
nohup env \
    CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
    LOAD_MODE=layerpipe \
    MAX_REQUESTS=100 \
    MAX_BATCH_SIZE=1 \
    OUTPUT_TOKENS_OVERRIDE=1 \
    TRACE_TIME_SCALE=150 \
    bash docs/1.2-layerpipe.sh \
    > docs/1.2-layerpipe.log 2>&1 &
echo $!
```
关键参数：
- MAX_BATCH_SIZE 限制一次从等待队列中抽取多少个“同模型请求”组成 batch。=0表示不限制

### CDF of residual loading stall
```bash
cd /mnt/n0/Tangram/Tangram

mkdir -p docs/paired-residual-stall-sweep

nohup /home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/run_residual_stall_sweep.py \
  --gpu 0 \
  --max-requests 0 \
  --max-pairs-per-model 30 \
  --input-scales 1,2,4,8 \
  --output-dir docs/paired-residual-stall-sweep \
  > docs/paired-residual-stall-sweep/run.log 2>&1 &
echo $!
```



## baseline-vmm-reuse

```bash
cd /mnt/n0/Tangram/Tangram

nohup env \
    CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
    LOAD_MODE=vmm \
    KV_BACKEND=odkv \
    VMM_POOL_GIB=42.0 \
    VMM_PAGE_SIZE_MIB=64 \
    MAX_REQUESTS=100 \
    MAX_BATCH_SIZE=1 \
    OUTPUT_TOKENS_OVERRIDE=1 \
    TRACE_TIME_SCALE=150 \
    bash docs/1.2-layerpipe.sh \
    > docs/1.2-vmm-odkv-reuse.log 2>&1 &

```

## LayerWeave 最小闭环

`layerweave` 现在有两条执行路径：

- `KV_BACKEND=default`：原 HuggingFace 最小闭环，用于独立验证权重
  page pipeline；
- `KV_BACKEND=odkv`：vLLM 执行栈，LayerWeave 权重 pipeline 与
  segmented ODKV 共用一个 VMM physical-page pool。

vLLM + ODKV 路径实现：

- vLLM scheduler/model runner 负责 Prefill、Decode 和 paged attention；
- ODKV block manager 按请求分配、释放 segmented KV pages；
- 模型注册时预留稳定 VMM VA，模型对象立即绑定稳定 tensor 地址；
- 初始化 vLLM engine 时只绑定权重 VA，不整模型预载；
- 根据真实 tensor 地址建立 non-layer 参数和 decoder layer 到 VMM page
  的映射；
- 每次 vLLM forward 先准备 non-layer 参数和 layer 0；
- 计算 layer i 时，在独立 copy stream 上异步映射并加载 layer i+1 的缺失页；
- compute stream 在每层执行前等待 layer-ready event；
- 每个 vLLM step 重新查询 residency，因此 ODKV 或其它模型导致的页替换
  不会沿用失效的 ready 状态；
- 已驻留的权重页不重复 H2D；权重页和 ODKV 页由同一个 VMM backend
  统一进行容量控制。

LayerWeave + ODKV：

```bash
nohup env \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  LOAD_MODE=layerweave \
  KV_BACKEND=odkv \
  VMM_POOL_GIB=42.0 \
  VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=100 \
  MAX_BATCH_SIZE=1 \
  OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=150 \
  bash docs/1.2-layerpipe.sh \
  > docs/1.2-layerweave.log 2>&1 &
echo $!
```

这条路径保持 42 GiB pool，因为权重和 ODKV 都在 pool 内；vLLM
activation 等临时显存仍在 pool 外。`LAYERWEAVE_RUNTIME_RESERVE_GIB`
只用于旧的 `KV_BACKEND=default` HF 路径，不传给 ODKV benchmark。

当前尚未包含 pipeline cost estimator、prefix cache configuration 和
多 GPU LayerWeave 调度；这些属于控制面后续阶段。

## LayerWeave + Cost estimator + Prefix Cache

### M3 performance metrics

主要是打点，breakdown出来reuse+pipeline的性能
```bash
cd /mnt/n0/Tangram/Tangram

for run in 1 2 3; do
  for prefetch in 0 1; do
    env \
      GPU_ID=0 \
      CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
      LOAD_MODE=layerweave \
      KV_BACKEND=odkv \
      VMM_POOL_GIB=42.0 \
      VMM_PAGE_SIZE_MIB=64 \
      MAX_REQUESTS=100 \
      MAX_BATCH_SIZE=1 \
      OUTPUT_TOKENS_OVERRIDE=1 \
      TRACE_TIME_SCALE=150 \
      LAYERWEAVE_PREFIX_LAYERS=full \
      LAYERWEAVE_PREFETCH="$prefetch" \
      OUTPUT="docs/m3-100-prefetch${prefetch}-run${run}.json" \
      bash docs/1.2-layerpipe.sh \
      > "docs/m3-100-prefetch${prefetch}-run${run}.log" 2>&1
  done
done
```
关键结果：
| 指标 | Prefetch=0 | Prefetch=1 | 改善 |
|---|---:|---:|---:|
| Mean | 647.30 ms | 539.30 ms | **108.00 ms** |
| P50 | 431.61 ms | 397.67 ms | **33.95 ms** |
| P90 | 1420.21 ms | 1203.02 ms | **217.18 ms** |
| P95 | 1751.67 ms | 1223.91 ms | **527.76 ms** |
| P99 | 2103.78 ms | 1683.93 ms | **419.85 ms** |

### M4 Pipeline Cost Estimator

收集完整的模型Profile。
```bash
cd /mnt/n0/Tangram/Tangram

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/run_layerweave_m4_full.py \
  --gpu 0 \
  --batch-sizes 1,2,4,8 \
  --output-dir docs/m4-full \
  --resume \
  --compact-batches
```

不同residency下对计算时间的预测准确率：

| Residency | MAE | P95 | MAPE | 解读 |
|---|---:|---:|---:|---|
| cold | 14.05 ms | 52.47 ms | 1.95% | 效果很好 |
| partial-hit | 22.71 ms | 99.66 ms | 4.22% | 总体可靠，但尾部误差最大 |
| full-hit | 15.26 ms | 55.91 ms | 6.10% | 绝对误差小，百分比被短 TTFT 放大 |

MAE 表示平均一次预测偏离实际值多少。

- cold 的 MAPE 只有 1.95%，是最稳定的一类。原因是 cold TTFT 主要由完整 H2D 加载构成，加载字节数和有效带宽比较容易建模。
- partial-hit 是最难预测的一类，P95 已达到 99.66 ms，接近验收上限。因为 partial-hit 同时受以下因素影响：  具体缺失的是哪些 layer/pages；  缺页加载与不同 layer compute 的相对位置；  prefix 长度；  host submission 和 allocator 波动；  当前 batch compute 是否足够隐藏 H2D。
- full-hit MAE 为 15.26 ms，P95 为 55.91 ms，绝对误差没有问题。

Prefetch gain预测：
count = 182
MAE = 23.65 ms
P95 absolute error = 88.27 ms
MAPE = 28.65%
相对也比较准了



### M5 Prefix Cache

每个模型只保留0，2，4，8，16，32个decoder layer的weight pages。
驱逐时综合考虑模型的冷热以及计算掩盖窗口大小。

```bash
# 生成新的Profile
cd /mnt/n0/Tangram/Tangram

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/run_layerweave_m4_full.py \
  --gpu 0 \
  --batch-sizes 1 \
  --input-scale 4 \
  --marginal-prefix-matrix \
  --output-dir docs/m4-m5.5-b1-scale4

# 最小闭环基线
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=100 MAX_BATCH_SIZE=1 \
  INPUT_SCALE=4 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=4 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=fixed \
  LAYERWEAVE_PREFIX_LAYERS=full \
  OUTPUT=docs/m5.5-marginal-minimal-100.json \
  bash docs/1.2-layerpipe.sh \
  > docs/m5.5-marginal-minimal-100.log 2>&1

# 新版本M5
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=100 MAX_BATCH_SIZE=1 \
  INPUT_SCALE=4 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=4 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=m4 \
  LAYERWEAVE_M4_PROFILE=docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  LAYERWEAVE_DEMAND_DECAY=0.9 \
  LAYERWEAVE_UNCERTAINTY_MS=100 \
  OUTPUT=docs/m5.5-marginal-dynamic-100.json \
  bash docs/1.2-layerpipe.sh \
  > docs/m5.5-marginal-dynamic-100.log 2>&1
```

参数作用：
- LAYERWEAVE_PREFETCH=1
启用下一层权重预取，使权重加载与当前层计算重叠。
- LAYERWEAVE_CACHE_POLICY=m4
启用 M4 estimator 指导的 M5.5 v2 动态策略
当前支持：
  fixed：固定 prefix 策略
  m4：调度期联合权重/KV容量规划
m4 路径会在每个 batch 的 generate() 前：
  检查所有模型的权重 residency。
  计算 active model 缺失的权重页。
  根据输入和输出长度计算本批 KV 页需求。
  判断统一 pool 是否有足够空间。
  只淘汰容量缺口对应的最少权重页。
  再进入实际推理。
- LAYERWEAVE_M4_PROFILE=...m4-full-estimator-report.json
  M4 estimator 的模型参数文件，包含 8 个模型的预测 profile。
- LAYERWEAVE_DEMAND_DECAY=0.9
  历史模型访问热度的指数衰减系数。
  含义：
    越接近 1：更看重长期历史，策略变化较慢
    越接近 0：更看重最近请求，策略变化较快
    0.9：兼顾短期热点和历史稳定性
- LAYERWEAVE_UNCERTAINTY_MS=100
  M4 预测收益的保守阈值，单位为毫秒。
  如果某个 prefix 预计只节省 40 ms，则不会被视为强保护对象；预计节省 250 ms，则按约 150 ms 的可信收益参与优先级计算。


## Multi-GPU

real-time调度很容易碰到busy-GPU queue，导致永远是free-first
```bash
cd /mnt/n0/Tangram/Tangram

nohup env \
  GPU_IDS=0,1 \
  MAX_REQUESTS=100 \
  ROUTING_LOOKAHEAD=8 \
  MAX_PLACEMENT_WAIT_MS=150 \
  PLACEMENT_HYSTERESIS_MS=25 \
  TRANSITION_WEIGHT=0.1 \
  TRANSITION_CREDIT_CAP_MS=100 \
  RUN_TAG=globalq1-100 \
  TRACE_TIME_SCALE=4 \
  bash docs/run_layerweave_multi_gpu_ab.sh \
  > docs/layerweave-2gpu-globalq1-100-driver.log 2>&1 &
echo $!
```

串行请求测试：
```bash
cd /mnt/n0/Tangram/Tangram

nohup env \
  GPU_IDS=0,1 \
  SCHEDULER_MODE=serial-choice \
  MAX_REQUESTS=100 \
  TRANSITION_WEIGHT=0.1 \
  TRANSITION_CREDIT_CAP_MS=100 \
  RUN_TAG=serial-choice-100-r1 \
  bash docs/run_layerweave_multi_gpu_ab.sh \
  > docs/layerweave-2gpu-serial-choice-100-r1-driver.log 2>&1 &
echo $!
```

### 模拟器

当前 Joint LayerWeave 配置：
  MCKP + PageGreedy
  Joint routing/cache placement
  configuration-step = 1
  lookahead K = 8
  transition-weight = 0.2
  transition-credit-cap = 100 ms
  2 GPUs
  672 pages/GPU
  input-scale = 4
  model-switches

```bash

cd /mnt/n0/Tangram/Tangram

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_pipeline_cache_sim.py \
  --policy-suite mckp-page-greedy \
  --max-requests 1000 \
  --gpus 2 \
  --pool-pages 672 \
  --lookahead-k 8 \
  --joint-routing \
  --configuration-step 1 \
  --transition-weight 0.2 \
  --transition-credit-cap-ms 100 \
  --output \
    docs/layerweave-joint-cache-sim-1000-step1-lambda02.json
```



## M3-M5 实施记录（2026-07-25）

本轮边界为 M3、M4、M5；没有实现 M6 多 GPU scheduler。实验固定使用
GPU-0（NVIDIA L40）、42 GiB 统一 weight/ODKV VMM pool、64 MiB page，
并保持 `MAX_BATCH_SIZE=1`、`OUTPUT_TOKENS_OVERRIDE=1`。

### M5 Prefix cache

实现修改：

- 新增 `LAYERWEAVE_PREFIX_LAYERS=0|2|4|8|16|full`，默认 `full`，
  因此不改变此前行为；
- prefix 大于 0 时保留 non-layer pages 和前 N 个 decoder layers 的
  page union；`0` 释放全部 inactive weight pages；
- C++ VMM backend 新增按 retained-page set 降级 stable arena 的接口；
- page 降级只允许在 forward 完成后执行，controller 在 active forward
  时调用会直接报错；执行前先同步 GPU；
- LayerWeave 加载和 ODKV KV 分配仍把当前模型 path 传给 backend 的
  protected set，因此 active model weight pages 不会作为容量回收 victim；
- VMM policy handshake 更新为
  `stable_model_va_layerweave_prefix_cache_v2`，避免 Python 静默加载旧
  shared library。

backend 构建：

```bash
cd /mnt/n0/Tangram/Tangram
cmake --build tools/mock_allocation/build -j2
```

完整默认 target 构建时本机缺少 `gtest/gtest.h`，所以
`gpu_tensor_pool_tests` target 失败；所需的
`libtangram_vram_backend.so` target 已成功构建和链接。符号检查：

```bash
nm -D tools/mock_allocation/build/libtangram_vram_backend.so |
  rg 'layerweave_configure_cache|tangram_vram_vmm_policy'
```

配置矩阵命令：

```bash
for prefix in 0 2 4 8 16 full; do
  env GPU_ID=0 \
    CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
    LOAD_MODE=layerweave KV_BACKEND=odkv \
    VMM_POOL_GIB=42.0 VMM_PAGE_SIZE_MIB=64 \
    MAX_REQUESTS=2 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
    TRACE_TIME_SCALE=150 LAYERWEAVE_PREFETCH=1 \
    LAYERWEAVE_PREFIX_LAYERS="$prefix" \
    OUTPUT="docs/m5-prefix${prefix}-smoke.json" \
    bash docs/1.2-layerpipe.sh \
    > "docs/m5-prefix${prefix}-smoke.log" 2>&1
done
```

每个进程的第一个请求为 cold activation；完成后应用指定配置，第二个
请求验证对应 residency：

| Prefix layers | 请求后 retained pages | 第二次 mapped pages | 第二次 exposed stall ms | 第二次 service TTFT ms |
|---|---:|---:|---:|---:|
| 0 | 0/240 | 240 | 617.37 | 648.15 |
| 2 | 45/240 | 195 | 496.58 | 529.92 |
| 4 | 58/240 | 182 | 463.67 | 499.56 |
| 8 | 84/240 | 156 | 397.37 | 435.34 |
| 16 | 136/240 | 104 | 265.44 | 309.50 |
| full | 240/240 | 0 | 0.41 | 34.47 |

retained pages 随 prefix 单调增加，下一次 mapped pages 和 exposed stall
单调下降；六种配置均完成正确推理，没有出现已释放 page 被访问或 active
model 被误淘汰。

### 本轮完成状态与 M6 前置项

- M3：完成事件级 overlap/no-overlap 验证；
- M4：完成可校准 estimator 和 cold/partial/full 误差报告；
- M5：完成六档 prefix cache、默认兼容、安全降级和真实 GPU 矩阵；
- M6：未开始。下一轮应先补齐全部模型、多个输入长度 bucket 的 profile，
  再把 estimator 接入多 GPU predicted-TTFT scheduler。

## M3-M4 service-TTFT 修正（2026-07-26）

### 修正目标

GPU event 区间相交只能描述设备侧 H2D/compute overlap，不能完整解释
prefetch 对 service TTFT 的收益。本轮保留 event 时间线作为底层 profile，
但将 M4 的最终预测和验收目标改为：

```text
MeasuredOverlapGain =
    service_ttft(prefetch=0) - service_ttft(prefetch=1)
```

### M3 时间分解修改

`TangramLayerWeaveController.metrics()` 新增：

- 每层 `host_submission_gap_ms`：
  `compute_start - max(previous_compute_end, layer_ready)`；
- 总 `host_submission_gap_ms`；
- `accounted_forward_ms`：
  `compute_ms + ready_stall_ms + host_submission_gap_ms`。

这三个分量现在可以闭合 CUDA forward。benchmark 的每个
`batch_metrics` 也直接记录标量 `service_ttft_ms` 和 `ttft_ms`，避免 M4
再从其它摘要反推 ground truth。

smoke 命令：

```bash
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/layerweave_m4_evaluation.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42.0 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=2 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=1 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_PREFIX_LAYERS=4 \
  OUTPUT=docs/m34-service-smoke.json \
  bash docs/1.2-layerpipe.sh
```

其中 cold forward：

```text
compute_ms              = 160.164
ready_stall_ms          = 598.162
host_submission_gap_ms =   1.974
accounted_forward_ms    = 760.299
```

三个分量与 CUDA forward duration 闭合；service TTFT 为 `914.562 ms`，
剩余约 `154.263 ms` 是首次执行的 forward 外 runtime/lazy-init 开销。

### M4 estimator 修改

`tools/layerpipe/layerweave_estimator.py` 现在由两层组成：

1. GPU pipeline 子模型：
   - stage 固定开销和有效 H2D bandwidth；
   - steady-state 逐层 compute profile；
   - compute 使用 token 数二次项，以覆盖 attention 随序列长度增长；
   - 分别模拟 prefetch 和严格 no-prefetch 的 copy/compute dependency。
2. service residual：
   - 按 prefetch mode、residency 和
     `first-activation/steady` 分桶；
   - 吸收 lazy kernel、allocator、host submission 和 forward 外开销；
   - 最终输出 `predicted_service_ttft_ms`。

首次请求不再混入 steady layer compute profile；否则 lazy kernel/runtime
warm-up 会同时污染 compute slope 和 service residual。

新增两份可复现 trace：

- `evaluation/traces/layerweave_m4_calibration.trace`：
  `31/512/2048` tokens，各重复两次；
- `evaluation/traces/layerweave_m4_evaluation.trace`：
  calibration 未见过的 `128/1024/3072` tokens，各重复两次。

calibration 使用 prefix `0/8/full`，evaluation 使用 prefix
`4/16/full`，两边均运行 prefetch `0/1`。单次运行模板：

```bash
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/layerweave_m4_calibration.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42.0 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=6 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=1 \
  LAYERWEAVE_PREFETCH=0 \
  LAYERWEAVE_PREFIX_LAYERS=8 \
  OUTPUT=docs/m34-cal-p8-f0.json \
  bash docs/1.2-layerpipe.sh
```

最终 estimator 命令：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_estimator.py \
  --calibration \
    docs/m34-cal-p0-f0.json docs/m34-cal-p8-f0.json \
    docs/m34-cal-pfull-f0.json docs/m34-cal-p0-f1.json \
    docs/m34-cal-p8-f1.json docs/m34-cal-pfull-f1.json \
  --evaluation \
    docs/m34-eval-p4-f0.json docs/m34-eval-p16-f0.json \
    docs/m34-eval-pfull-f0.json docs/m34-eval-p4-f1.json \
    docs/m34-eval-p16-f1.json docs/m34-eval-pfull-f1.json \
  --output docs/m34-service-ttft-estimator-eval.json
```

独立 evaluation 结果：

| Residency | 样本数 | service TTFT MAE | P95 absolute error | MAPE |
|---|---:|---:|---:|---:|
| cold | 6 | 7.67 ms | 13.72 ms | 0.80% |
| partial-hit | 20 | 19.00 ms | 61.53 ms | 4.23% |
| full-hit | 10 | 10.24 ms | 24.68 ms | 12.89% |

prefetch/no-prefetch 成对 service TTFT gain：

```text
count = 18
MAE = 20.43 ms
P95 absolute error = 49.03 ms
```

full-hit 的 service TTFT 和 overlap gain 都很小，因此百分比误差不稳定；
调度验收以绝对误差为主。当前结果只验证 model 2（8B）和 batch size 1，
在进入 M6 前仍需要为其余模型建立相同 profile。

### M4 最终复跑验收（2026-07-26）

使用上述 6 组 calibration、6 组独立 evaluation 和最终验收脚本重新运行，
实际结果为：

| Residency | 样本数 | service TTFT MAE | P95 absolute error | MAPE |
|---|---:|---:|---:|---:|
| cold | 6 | 7.96 ms | 17.38 ms | 0.82% |
| partial-hit | 20 | 24.06 ms | 78.38 ms | 5.18% |
| full-hit | 10 | 13.97 ms | 24.74 ms | 11.79% |

prefetch/no-prefetch 成对 service TTFT gain：

```text
count = 18
MAE = 19.06 ms
P95 absolute error = 46.52 ms
MAPE = 67.35%
```

验收项：

```text
cold_mape_le_10pct       = PASS
partial_mape_le_10pct    = PASS
cold_p95_le_100ms        = PASS
partial_p95_le_100ms     = PASS
gain_p95_le_100ms        = PASS
M4_VALIDATION            = PASS
```

gain MAPE 较高不表示 estimator 失效：full-hit 等样本的实际 overlap gain
接近零，较小的绝对误差也会被百分比放大。因此 M4 调度验收继续以 service
TTFT 和 prefetch gain 的绝对误差为主。该结果确认 model 2、batch size 1
范围内的 M4 estimator 闭环通过。

## M4 完整多模型版本（2026-07-26）

### 范围与实现修改

M4 从原来的 model 2、batch size 1 扩展为：

- 8 个 ServeGen 模型；
- batch size `1/2/4/8`；
- batch 1 完整覆盖 cold、partial-hit、full-hit 和 prefetch `0/1`；
- batch `2/4/8` 使用 full-hit calibration 建立 compute scaling，并使用
  prefix 16、未见 token 点的 partial-hit evaluation 验证 compute/H2D 组合；
- 最终预测目标仍为 `service_ttft_ms`，overlap 收益仍为成对
  `service_ttft(prefetch=0) - service_ttft(prefetch=1)`。

代码修改：

- `tools/layerpipe/vllm_odkv_trace_bench.py`
  - batch metrics 新增 `max_input_tokens` 和
    `sum_input_tokens_squared`；
  - benchmark 不使用生成文本，因此设置
    `SamplingParams(detokenize=False)`，避免转换后 tokenizer 的稀疏 token
    ID 在 batch 模式下触发 `tokens=None` 的无关 detokenizer 异常。
- `tools/layerpipe/layerweave_estimator.py`
  - 首次激活由全局 `batch_id == 0` 修正为每个
    `(source file, model_id)` 的第一次激活；
  - 逐层 compute regression 使用
    `batch_size/total_tokens/max_tokens/sum(tokens^2)`；
  - 输出按 model 和 batch size 的独立误差；
  - 只拟合单个 vLLM Prefill wave。总 batch tokens 超出 vLLM 单次调度预算
    时产生的多个 `forward_id` 留给 M6 组合多个 M4 pipeline prediction。
- 新增 `tools/layerpipe/layerweave_m4_generate_traces.py`，生成 8 模型、
  batch `1/2/4/8` 的独立 calibration/evaluation trace。
- 新增 `tools/layerpipe/run_layerweave_m4_full.py`：
  - 自动生成 trace、运行实验、拟合、打印验收；
  - `--resume` 只跳过包含完整 `batch_metrics` 的 JSON；
  - `--compact-batches` 使用上述精简但覆盖完整语义的矩阵；
  - 最终 report 持久化 profile scope 和验收结果。

### 最终可复现命令

```bash
cd /mnt/n0/Tangram/Tangram

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/run_layerweave_m4_full.py \
  --gpu 0 \
  --batch-sizes 1,2,4,8 \
  --output-dir docs/m4-full \
  --resume \
  --compact-batches
```

该命令固定：

```text
GPU                    = NVIDIA L40 physical GPU 0
VMM pool               = 42 GiB
VMM page               = 64 MiB
VMM policy             = stable_model_va_layerweave_prefix_cache_v2
output tokens          = 1
models                 = 0..7
```

## M5.5 单 GPU 动态 Prefix Cache（2026-07-26）

### 目标与边界

M5.5 在不实现多 GPU scheduler 的前提下，将 M4 service-TTFT estimator
接入 M5 权重 prefix cache。它只使用已经发生的请求和当前已经到达的
pending queue，不读取未来 arrival。原有固定配置仍是默认路径：

```text
LAYERWEAVE_CACHE_POLICY=fixed
```

动态策略通过 `LAYERWEAVE_CACHE_POLICY=m4` 显式启用。

### 实现修改

- prefix 档位扩展为 `0/2/4/8/16/32/full`；不足 32 层的模型自然将
  `32` 去重为 `full`；
- `TangramLayerWeaveController` 新增按参数计算 prefix page set，以及
  运行时应用任意 prefix 配置的接口；
- 新增 `tools/layerpipe/layerweave_cache_policy.py`：
  - 使用历史请求指数衰减热度和当前 pending queue 估计模型需求；
  - 从 M4 report 加载 8 模型 estimator profile；
  - 根据最近 batch size/input tokens 为每个真实可形成的 prefix 预测
    service TTFT；
  - 对 `0/2/4/8/16/32/full` 做 retained-page-set 去重；
  - 只枚举当前真实 resident 的 prefix，因为 cache configure 只能淘汰，
    不能把已经丢失的 inactive pages 重新加载；
  - 使用 multiple-choice knapsack，在 page budget 内最大化
    `demand × (cold TTFT - prefix TTFT - uncertainty)`；
  - M4 residual bucket 可能造成轻微非单调预测，因此对候选 TTFT 应用
    “更多 retained pages 不更慢”的单调包络；
  - 为 pending queue 队首模型预留 full working-set pages，避免当前
    full cache 在下一模型加载时被动退化为随机残页；
- 动态决策、预测 TTFT、需求分数、pending 数、retained pages、
  planned peak pages 和实际 backend 应用结果都写入每个 batch 的
  `cache_policy`；
- 新增 `tools/layerpipe/validate_layerweave_m5_5.py`，检查 A/B workload
  公平性、策略是否真正启用、budget、prefix 选择、真实 weight reuse，
  并可用 `--require-speedup` 将 service TTFT 改善设为验收条件；
- 新增四请求短 trace：
  `evaluation/traces/layerweave_m5_5_smoke.trace`。

> 这一版固定 `LAYERWEAVE_CACHE_BUDGET_GIB` 的实现已被下文
> “M5.5 v2 调度期联合容量规划”取代；旧命令仅保留为历史记录。

### 静态检查命令

```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python -m py_compile \
  ElasticKV/vllm/model_executor/model_loader/tangram_vmm.py \
  tools/layerpipe/layerweave_cache_policy.py \
  tools/layerpipe/vllm_odkv_trace_bench.py \
  tools/layerpipe/layerweave_estimator.py \
  tools/layerpipe/validate_layerweave_m5_5.py
bash -n docs/1.2-layerpipe.sh
git diff --check
```

### GPU-1 短 smoke

fixed baseline：

```bash
env GPU_ID=1 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/layerweave_m5_5_smoke.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=8 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=4 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=1 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=fixed LAYERWEAVE_PREFIX_LAYERS=full \
  OUTPUT=docs/m5.5-smoke-fixed.json \
  bash docs/1.2-layerpipe.sh
```

M4-guided dynamic cache：

```bash
env GPU_ID=1 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/layerweave_m5_5_smoke.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=8 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=4 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=1 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=m4 LAYERWEAVE_CACHE_BUDGET_GIB=7.5 \
  LAYERWEAVE_M4_PROFILE=docs/m4-full/m4-full-estimator-report.json \
  LAYERWEAVE_DEMAND_DECAY=0.9 LAYERWEAVE_UNCERTAINTY_MS=0 \
  OUTPUT=docs/m5.5-smoke-dynamic.json \
  bash docs/1.2-layerpipe.sh
```

功能验收：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/validate_layerweave_m5_5.py \
  --fixed docs/m5.5-smoke-fixed.json \
  --dynamic docs/m5.5-smoke-dynamic.json
```

结果：

```text
budget_ok                  = true
reuse_observed             = true
selected_configurations    = 0,4,full
fixed service TTFT mean    = 331.42 ms
dynamic service TTFT mean  = 349.28 ms
service TTFT gain          = -17.85 ms
M5_5_VALIDATION            = PASS
```

该 smoke 只有四个请求，且 8 GiB tight pool 只能形成很短的 prefix；
它验收机制和安全性，不作为性能结论。正式性能验收必须使用更长的同 trace
A/B，并增加 `--require-speedup`。

### 正式运行与验证命令

两次运行必须使用同一空闲 GPU、trace、pool、page size、batch 和输出长度。
以下以 GPU-0、100 请求为例。

fixed baseline：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=100 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=1 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=fixed LAYERWEAVE_PREFIX_LAYERS=full \
  OUTPUT=docs/m5.5-fixed-100.json \
  bash docs/1.2-layerpipe.sh \
  > docs/m5.5-fixed-100.log 2>&1
```

M5.5：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=100 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=1 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=m4 LAYERWEAVE_CACHE_BUDGET_GIB=36 \
  LAYERWEAVE_M4_PROFILE=docs/m4-full/m4-full-estimator-report.json \
  LAYERWEAVE_DEMAND_DECAY=0.9 LAYERWEAVE_UNCERTAINTY_MS=100 \
  OUTPUT=docs/m5.5-dynamic-100.json \
  bash docs/1.2-layerpipe.sh \
  > docs/m5.5-dynamic-100.log 2>&1
```

最终性能验收：

```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/validate_layerweave_m5_5.py \
  --fixed docs/m5.5-fixed-100.json \
  --dynamic docs/m5.5-dynamic-100.json \
  --require-speedup
```

`M5_5_VALIDATION=PASS` 表示 workload/backend 公平、动态策略生效、没有
超过规划预算、确实发生 prefix reuse，并且平均 service TTFT 优于固定
缓存基线。P95 是否改善仍应从两个 JSON 的
`summary.service_ttft_ms.p95` 单独报告，不强行包含在首次 M5.5 门槛中。

### M5.5 v2：调度期联合容量规划（2026-07-26）

这一版取消 `LAYERWEAVE_CACHE_BUDGET_GIB`，权重和 ODKV 继续使用同一个
弹性 VMM physical-page pool。动态 prefix 不再是立即裁剪的硬上限，而是
发生显存压力时的保护优先级。

新的 batch 执行顺序：

```text
pop batch
  -> 统计所有模型当前 weight residency 和现存 KV pages
  -> 计算 active model 缺失 weight pages
  -> 按每个请求 ceil((input + max_output) / token_block_size)
     计算本 batch KV blocks，并按 VMM page size 向上取整
  -> reservation = missing weight pages + request KV physical pages
  -> 若 free pages 不足，只淘汰 shortage 对应的最少页
  -> llm.generate()
  -> ODKV 正常释放请求 KV；不做执行后全局 rebalance
```

具体修改：

- `tools/layerpipe/vllm_odkv_trace_bench.py`
  - M4 policy 从 `generate()` 后移到 `generate()` 前；
  - 将 pool 总页数、当前所有 ODKV block、active model 缺失权重和当前
    batch 的 KV 需求送入联合规划；
  - 每个 batch 记录 `reservation_pages`、预测 KV blocks、缺失权重页、
    最小淘汰页数、实际淘汰页数和淘汰耗时；
- `tools/layerpipe/layerweave_cache_policy.py`
  - 删除静态 cache budget 和 multiple-choice budget knapsack；
  - M4 选择的 prefix 只作为 protected floor；
  - prefix 外的 resident suffix 在空间足够时继续驻留；
  - 缓存压力下先淘汰 opportunistic suffix，仍不足时再按 M4
    demand-weighted value/page 降级 protected prefix；
  - active model 永远不是淘汰 victim；
- `TangramLayerWeaveController.apply_retained_pages()`
  - 只应用调度器选中的最小 victim set，不再请求结束后对所有模型按
    prefix 做全局裁剪；
- `docs/1.2-layerpipe.sh`
  - 完全删除 `LAYERWEAVE_CACHE_BUDGET_GIB` 的默认值、打印和传递；
- `tools/layerpipe/validate_layerweave_m5_5.py`
  - 验证 predicted KV blocks 不小于实际分配；
  - 验证 `evicted_pages == eviction_required_pages`；
  - 验证 reservation 不超过统一 pool，并继续检查真实权重复用。

本次执行的静态检查：

```bash
cd /mnt/n0/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python -m py_compile \
  Tangram/tools/layerpipe/layerweave_cache_policy.py \
  Tangram/tools/layerpipe/vllm_odkv_trace_bench.py \
  Tangram/tools/layerpipe/validate_layerweave_m5_5.py \
  Tangram/ElasticKV/vllm/model_executor/model_loader/tangram_vmm.py
bash -n Tangram/docs/1.2-layerpipe.sh
git -C Tangram diff --check
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

GPU-0 四请求 smoke 命令：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/layerweave_m5_5_smoke.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=8 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=4 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=1 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=m4 \
  LAYERWEAVE_M4_PROFILE=docs/m4-full/m4-full-estimator-report.json \
  LAYERWEAVE_DEMAND_DECAY=0.9 LAYERWEAVE_UNCERTAINTY_MS=0 \
  OUTPUT=docs/m5.5-v2-smoke-dynamic.json \
  bash docs/1.2-layerpipe.sh

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/validate_layerweave_m5_5.py \
  --fixed docs/m5.5-smoke-fixed.json \
  --dynamic docs/m5.5-v2-smoke-dynamic.json
```

smoke 结果：

```text
batch                     0       1       2       3
predicted KV blocks       5       5       5       5
actual KV blocks          4       4       4       4
eviction required pages   0      83      83      83
actual evicted pages      0      83      83      83
eviction time (ms)      0.001   3.313   3.002   2.931

reservation_ok          = true
kv_estimate_ok           = true
reuse_observed           = true
M5_5_VALIDATION          = PASS
```

该 smoke 的 8 GiB pool 极紧，主要用于验证联合预留和最小淘汰的正确性；
四请求 service TTFT 不作为性能结论。

### M5.5 v2 新的正式运行命令

先跑最小闭环基线。`fixed + prefix=full` 不主动裁剪已驻留权重，仍由原
LayerWeave/ODKV 执行路径在容量不足时自然替换，是同代码路径基线：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=100 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=1 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=fixed LAYERWEAVE_PREFIX_LAYERS=full \
  OUTPUT=docs/m5.5-v2-minimal-100.json \
  bash docs/1.2-layerpipe.sh \
  > docs/m5.5-v2-minimal-100.log 2>&1
```

再跑 M5.5 v2；命令中不再设置任何 cache budget：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=100 MAX_BATCH_SIZE=1 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=1 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=m4 \
  LAYERWEAVE_M4_PROFILE=docs/m4-full/m4-full-estimator-report.json \
  LAYERWEAVE_DEMAND_DECAY=0.9 LAYERWEAVE_UNCERTAINTY_MS=100 \
  OUTPUT=docs/m5.5-v2-dynamic-100.json \
  bash docs/1.2-layerpipe.sh \
  > docs/m5.5-v2-dynamic-100.log 2>&1
```

最终验证：

```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/validate_layerweave_m5_5.py \
  --fixed docs/m5.5-v2-minimal-100.json \
  --dynamic docs/m5.5-v2-dynamic-100.json \
  --require-speedup
```

`M5_5_VALIDATION=PASS` 表示 workload 和 pool 公平、KV 预留未低估、调度期
只淘汰所需的最少页、发生真实权重复用，且平均 service TTFT 优于最小
闭环基线。

### M5.5 v2 100 请求实测结果（2026-07-26）

结果文件：

```text
docs/m5.5-v2-minimal-100.json
docs/m5.5-v2-minimal-100.log
docs/m5.5-v2-dynamic-100.json
docs/m5.5-v2-dynamic-100.log
```

实际 A/B 的 trace、100 个请求、模型顺序、输入长度、42 GiB pool、
64 MiB page、prefetch、output=1 均一致；两次结果中的
`trace_time_scale` 实际均为 `4.0`，因此比较仍公平。两个日志均完整产生
100 个 `BATCH_COMPLETE` 和最终 `LAYERPIPE_RESULT`，没有 Traceback、
CUDA OOM 或 RuntimeError。

执行的验收命令：

```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/validate_layerweave_m5_5.py \
  --fixed docs/m5.5-v2-minimal-100.json \
  --dynamic docs/m5.5-v2-dynamic-100.json \
  --require-speedup
```

输出：

```text
reservation_ok       = true
kv_estimate_ok        = true
reuse_observed        = true
M5_5_VALIDATION       = PASS
```

核心性能：

| 指标 | 最小闭环 | M5.5 v2 | v2 改善 |
|---|---:|---:|---:|
| mean service TTFT | 527.40 ms | 518.46 ms | 8.94 ms / 1.69% |
| P50 service TTFT | 391.62 ms | 343.99 ms | 47.63 ms |
| P95 service TTFT | 1211.87 ms | 1214.15 ms | -2.28 ms |
| P99 service TTFT | 1654.38 ms | 1670.20 ms | -15.82 ms |
| mean weight H2D | 410.15 ms | 403.02 ms | 7.13 ms / 1.74% |
| mean prefill | 523.33 ms | 506.58 ms | 16.76 ms / 3.20% |
| mean queue | 25151.41 ms | 24744.78 ms | 406.63 ms |
| mean end-to-end | 25678.94 ms | 25263.34 ms | 415.60 ms |

联合预留和淘汰统计：

```text
predicted KV blocks       = 4242
actual KV blocks          = 4211
underestimated batches    = 0
max reservation pages     = 656 / 672
eviction batches          = 68 / 100
total evicted pages       = 14417
opportunistic pages       = 4580
protected pages           = 9837
total pre-run eviction    = 453.01 ms
mean pre-run eviction     = 4.53 ms/request
max pre-run eviction      = 20.93 ms
```

ODKV 的 100 次 allocate 总耗时从 `476.17 ms` 降到 `257.92 ms`，说明
提前清出联合 reservation 后，KV allocate 的慢路径显著减少。动态版本
总 mapped weight pages 从 `15290` 降到 `15068`，少加载 222 页；full hit
从 23 次增加到 33 次。

但当前性能结论应限定为“弱正向”：

- 平均 service TTFT 改善 8.94 ms，验收门槛通过；
- 逐请求配对只有 25 个请求变快、75 个变慢，配对差值中位数为
  `-3.10 ms`；
- 配对 mean gain 的近似 95% 区间为 `[-7.09, 24.97] ms`，仍跨过 0；
- P95/P99 分别退化 2.28/15.82 ms；
- 68% batch 需要调度前淘汰，且 68.2% victim 属于 protected pages，
  表明显存压力下 prefix 保护经常被迫降级。

因此这组实验已经证明 M5.5 v2 的容量规划正确、没有 KV underflow，并且
成功把一部分 ODKV eviction 慢路径移到调度边界；它也得到 1.69% 的 mean
service TTFT 优势。但单次 100 请求还不足以证明稳定性能优势，下一步应
优先减少 protected-page churn，并用至少三次同 trace A/B 报告配对置信
区间和 P95/P99。

### M5.5 marginal-overlap 修订（2026-07-26）

本轮把最终实验语义固定为：

```text
MAX_BATCH_SIZE=1
INPUT_SCALE=4
OUTPUT_TOKENS_OVERRIDE=1
```

`LAYERWEAVE_CACHE_POLICY=m4` 若不满足以上条件会直接报错，避免再次用
短输入或 batch-size 矩阵误验收。验证脚本也检查实际所有 batch size 为
1、input scale 为 4、output override 为 1。

M4 trace/profile 路径修改：

- `layerweave_m4_generate_traces.py` 新增 `--input-scale`；
- trace 中写入的是缩放前 token 数，保证 benchmark 应用
  `INPUT_SCALE=4` 后仍覆盖模型安全输入上限的 calibration/evaluation
  fractions，而不是全部被截断到同一个长度；
- scale-4 trace 使用独立的 `_scale4.trace` 文件名，不覆盖旧 M4 trace；
- `run_layerweave_m4_full.py` 新增 `--input-scale`，并把
  `input_scale/max_batch_size_policy/output_tokens_override` 写入 report
  的 `m4_scope`；
- M5.5 加载 Profile 时强制检查 `batch_sizes=[1]`、`input_scale=4`、
  `output_tokens_override=1`，旧 Profile 不再被静默复用。

生成的新 trace：

```text
evaluation/traces/layerweave_m4_full_calibration_b1_scale4.trace
evaluation/traces/layerweave_m4_full_evaluation_b1_scale4.trace
```

生成命令和静态检查：

```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_m4_generate_traces.py \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --output-dir evaluation/traces \
  --batch-sizes 1 \
  --input-scale 4

/home/sdu/.conda/envs/sllm-worker/bin/python -m py_compile \
  tools/layerpipe/layerweave_m4_generate_traces.py \
  tools/layerpipe/run_layerweave_m4_full.py \
  tools/layerpipe/layerweave_cache_policy.py \
  tools/layerpipe/vllm_odkv_trace_bench.py \
  tools/layerpipe/validate_layerweave_m5_5.py
bash -n docs/1.2-layerpipe.sh
git diff --check
```

另外使用旧 report 的 model-0 profile 和 synthetic nested page sets 对
`_marginal_tiers()` 做了无 GPU 单元检查，确认生成顺序恰为
`0->2/2->4/4->8/8->16/16->32/32->full`，所有 required pages 均恰好
归入一个 tier，输出 `M5_5_MARGINAL_UNIT=PASS`。

缓存价值策略不再从 `0/2/4/8/16/32/full` 中选择一个总收益最大的硬
target，而是预测相邻档位的 marginal service-TTFT benefit：

```text
value_per_page(a->b) =
  effective_demand
  * max(0, predicted_TTFT(a) - predicted_TTFT(b) - uncertainty_per_tier)
  / added_pages(a->b)
```

每个 resident page 即使只形成部分 prefix，也会按它所属的 marginal tier
获得价值，不再因为完整 prefix 缺一页就整体退化成零价值 opportunistic
缓存。需要释放 capacity 时仍只淘汰精确 shortage，tier 顺序为：

```text
fragment
32 -> full
16 -> 32
8  -> 16
4  -> 8
2  -> 4
0  -> 2
```

同一 tier 内再按 demand-weighted marginal value/page 选择 victim。每个
batch 的结果新增 `evicted_pages_by_tier` 和每模型 `marginal_tiers`，
便于确认释放的是否真是容易被 overlap 掩盖的后部参数。

新的 batch-1/scale-4 Profile 需要执行一次以下命令。它只运行 batch=1
矩阵，不再运行 batch 2/4/8：

```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/run_layerweave_m4_full.py \
  --gpu 0 \
  --batch-sizes 1 \
  --input-scale 4 \
  --marginal-prefix-matrix \
  --output-dir docs/m4-m5.5-b1-scale4
```

生成的 Profile 为：

```text
docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json
```

Profile 完成后，新的最小闭环命令：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=100 MAX_BATCH_SIZE=1 \
  INPUT_SCALE=4 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=4 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=fixed LAYERWEAVE_PREFIX_LAYERS=full \
  OUTPUT=docs/m5.5-marginal-minimal-100.json \
  bash docs/1.2-layerpipe.sh \
  > docs/m5.5-marginal-minimal-100.log 2>&1
```

新的 M5.5 marginal-overlap 命令：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=100 MAX_BATCH_SIZE=1 \
  INPUT_SCALE=4 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=4 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=m4 \
  LAYERWEAVE_M4_PROFILE=docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  LAYERWEAVE_DEMAND_DECAY=0.9 LAYERWEAVE_UNCERTAINTY_MS=100 \
  OUTPUT=docs/m5.5-marginal-dynamic-100.json \
  bash docs/1.2-layerpipe.sh \
  > docs/m5.5-marginal-dynamic-100.log 2>&1
```

验证命令：

```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/validate_layerweave_m5_5.py \
  --fixed docs/m5.5-marginal-minimal-100.json \
  --dynamic docs/m5.5-marginal-dynamic-100.json \
  --require-speedup
```

#### scale-4 fixed baseline ODKV allocation failure 与修复

首次执行 `m5.5-marginal-minimal-100` 在 batch 6、model 4、输入 8596
tokens 处失败：

```text
RuntimeError: VMM KV allocation failed
```

根因是旧的模型安全输入上限只考虑 vLLM/model limit，没有考虑统一物理
池的 active 权重：

```text
pool pages                 = 672
model 4 required pages     = 441
KV blocks for 8596+1       = ceil(8597/32) = 269
64 MiB pages per KV block  = 1
total required pages       = 441 + 269 = 710 > 672
```

`vllm_odkv_trace_bench.py` 新增 `apply_joint_pool_input_limits()`。所有
LayerWeave fixed/M5.5 运行在 engine 初始化后按模型计算：

```text
max_kv_blocks =
  floor((pool_pages - full_active_weight_pages) / pages_per_kv_block)
max_input_tokens =
  max_kv_blocks * token_block_size - output_tokens
```

这与 M5.5 dispatch reservation 使用同一物理页语义，并在 A/B 两侧同样
应用。model 4 的可行上限因此为 `231*32-1=7391` input tokens。
结果 JSON 新增 `joint_pool_input_limits`，日志新增
`JOINT_POOL_INPUT_LIMITS`。

修复后在 GPU-0 用原 trace 前 7 个请求覆盖同一失败点：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=7 MAX_BATCH_SIZE=1 \
  INPUT_SCALE=4 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=4 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=fixed LAYERWEAVE_PREFIX_LAYERS=full \
  OUTPUT=docs/m5.5-marginal-minimal-fix-smoke7.json \
  bash docs/1.2-layerpipe.sh
```

验证结果：

```text
truncated_requests          = 1
model 4 dispatched input    = 7391
model 4 allocated KV blocks = 231
batches completed           = 7/7
VMM KV allocation failure   = 0
```

因此上面的 100 请求 Profile、fixed baseline 和 M5.5 命令不需要增加新
参数，更新代码后原样重跑即可。

现有 scale-4 Profile 文件已经完整生成，但严格 M4 验收为：

```text
cold MAPE/P95       = 1.74% / 44.67 ms  PASS
partial MAPE/P95    = 4.18% / 80.25 ms  PASS
gain P95 error      = 116.69 ms          FAIL (>100 ms)
M4_FULL_VALIDATION  = FAIL
```

因此它可以用于下一轮 exploratory M5.5 A/B 和 marginal-tier 日志分析，
但在 gain P95 降到 100 ms 以内前，不能标记为最终通过的 M4 Profile。

#### M5.5 marginal-overlap 100 请求结果

两次运行均完整产生 100 个 batch，无 Traceback/OOM/ODKV failure。trace、
处理后的逐请求 model/input/output、42 GiB pool、64 MiB page、
`batch=1/input_scale=4/output=1/trace_time_scale=4` 完全一致。统一池上限
额外截断 8 个请求；两侧应用相同截断。

验收：

```text
reservation_ok       = true
kv_estimate_ok        = true
reuse_observed        = true
M5_5_VALIDATION       = PASS
```

核心结果：

| 指标 | 最小闭环 | M5.5 marginal | 改善 |
|---|---:|---:|---:|
| mean service TTFT | 723.27 ms | 713.17 ms | 10.09 ms / 1.40% |
| P50 service TTFT | 622.85 ms | 572.85 ms | 50.00 ms |
| P90 service TTFT | 1664.92 ms | 1684.36 ms | -19.43 ms |
| P95 service TTFT | 1968.98 ms | 1883.04 ms | 85.93 ms |
| P99 service TTFT | 2013.13 ms | 2028.71 ms | -15.59 ms |
| mean weight H2D | 430.63 ms | 444.29 ms | -13.66 ms |
| mean prefill | 716.55 ms | 693.06 ms | 23.49 ms / 3.28% |
| mean end-to-end | 36432.20 ms | 36179.93 ms | 252.27 ms |

LayerWeave critical-path 分解：

| 指标 | 最小闭环 | M5.5 marginal | 变化 |
|---|---:|---:|---:|
| ready stall | 263.44 ms | 247.03 ms | -16.41 ms |
| exposed load | 263.36 ms | 246.97 ms | -16.39 ms |
| hidden load | 167.28 ms | 197.32 ms | +30.05 ms |
| overlap ratio | 51.57% | 57.06% | +5.49 pp |

动态版本总 mapped weight pages 从 `16025` 增加到 `16603`，多加载 578
页；full hit 基本不变（22→23），partial hit 从 24 增至 36，cold 从
54 降至 41。这不是 weight-byte cache hit 最大化，而是把有限缓存用于
关键 prefix：尽管总 H2D 增加 13.66 ms，其中更多加载被 Prefill compute
掩盖，exposed load 反而减少 16.39 ms。该结果直接验证 marginal-overlap
策略的目标方向。

ODKV：

```text
predicted KV blocks      = 9032
actual KV blocks         = 8890
underestimated batches   = 0
ODKV allocate total      = 908.32 -> 498.85 ms
```

策略开销和 victim：

```text
eviction batches         = 70 / 100
evicted pages            = 15974
eviction total           = 529.39 ms
mean eviction            = 5.29 ms/request
max eviction             = 21.77 ms
max reservation          = 672 / 672 pages

32->full                 = 2805 pages
16->32                   = 2976 pages
16->full                 = 3510 pages
8->16                    = 3012 pages
4->8                     = 1386 pages
2->4                     = 635 pages
0->2                     = 1650 pages
```

后部 tier（`16->32/32->full/16->full`）占 58.2% victim；包含
`8->16` 后占 77.1%，说明 victim 顺序已经明显偏向更可 overlap 的后部
参数。最终有效 prefix 分布也从旧实现的 `0/full` 两极化变成：

```text
0=46, 2=8, 4=5, 8=6, 16=11, 32=1, full=23
```

但统计显著性仍不足：逐请求 service TTFT 只有 26 个变快、74 个变慢，
配对 gain 中位数 `-6.52 ms`，mean gain 的近似 95% 区间
`[-11.30, 31.49] ms`。少数大收益请求拉高均值；P95 改善明显，但
P90/P99 略退化。当前结论应为：marginal placement 机制生效且 mean/P95
正向，但需多次 A/B 才能确认稳定收益。另因 scale-4 Profile 的 gain P95
error 为 116.69 ms，本轮仍属于 exploratory validation。

## M4 完整多模型版本（结果续）

最终 report：

```text
docs/m4-full/m4-full-estimator-report.json
```

### 实验完整性

选定矩阵包含 24 个结果 JSON、1152 个实际 batch：

```text
single Prefill wave samples = 836
multi Prefill wave samples  = 316（M4 不拟合，留给 M6 组合）
single-wave by batch size   = b1:576, b2:108, b4:80, b8:72
max forward closure error   = 0.081 ms
```

所有选定运行均加载 policy v2；最终日志无 `Traceback`、CUDA OOM 或 backend
错误。实验完成后 GPU-0 显存占用恢复到 13 MiB。

### 最终独立 evaluation

service TTFT：

| Residency | 样本数 | MAE | P95 absolute error | MAPE |
|---|---:|---:|---:|---:|
| cold | 92 | 14.05 ms | 52.47 ms | 1.95% |
| partial-hit | 228 | 22.71 ms | 99.66 ms | 4.22% |
| full-hit | 80 | 15.26 ms | 55.91 ms | 6.10% |

按 batch size：

| Batch size | 样本数 | MAE | P95 absolute error | MAPE |
|---:|---:|---:|---:|---:|
| 1 | 288 | 16.64 ms | 61.30 ms | 4.17% |
| 2 | 40 | 31.80 ms | 109.01 ms | 4.76% |
| 4 | 40 | 21.03 ms | 51.00 ms | 3.36% |
| 8 | 32 | 24.61 ms | 72.14 ms | 3.26% |

prefetch/no-prefetch 成对 service TTFT gain：

```text
count = 182
MAE = 23.65 ms
P95 absolute error = 88.27 ms
MAPE = 28.65%
```

最终验收：

```text
cold_mape_le_10pct       = PASS
partial_mape_le_10pct    = PASS
cold_p95_le_100ms        = PASS
partial_p95_le_100ms     = PASS
gain_p95_le_100ms        = PASS
all_8_models_covered     = PASS
batch_1_2_4_8_covered    = PASS
M4_FULL_VALIDATION       = PASS
```

gain MAPE 仍受接近零的 full-hit gain 放大，因此继续以绝对误差验收。局部
切片中 model 5 P95 为 `104.14 ms`、batch 2 P95 为 `109.01 ms`，略高于
100 ms，但其 MAPE 分别为 `3.48%` 和 `4.76%`；总体 residency 和 gain
门槛均通过。M6 接入时应保留按模型/batch 的误差观测，并由 M6 负责把
multi-wave batch 分解为多个 M4 pipeline activation 后组合预测。

## Configuration-level Joint Planning + Deferred Reclamation（2026-07-26）

根据 `docs/joint_schedule.md` 的新语义，launch configuration 不再是
物理驻留的硬上限，而是 protected residency floor。当前真实驻留状态
被显式拆分为：

```text
actual pages
  = active pages
  + protected pages
  + soft pages
```

本轮完成设计中的第 1--4 步：

- `tools/layerpipe/layerweave_cache_policy.py`
  - 新增每模型持久化 `protected_configuration`；
  - actual pages 始终从 VMM residency 查询；
  - protected pages 来自配置 page set；
  - 非活动模型的 `soft = actual - protected`，active model 的全部工作集
    在执行期间不可回收；
  - 新增 configuration expected value、Pareto dominated configuration
    裁剪和 page-granular MCKP DP；
  - 非目标模型只能保持或降低已有 protected floor；
  - 目标模型完成本轮 full working-set 加载，因此可选择任意新 floor；
  - MCKP 只更新 protection metadata，不立即收缩到配置边界；
  - exact-shortage reclaim 的 victim 集合严格限制为 inactive soft
    pages，`evicted_protected_pages` 必须恒为 0；
  - MCKP 输出 capacity、used pages、solve time、before/after value 和
    transition cost。
- `tools/layerpipe/vllm_odkv_trace_bench.py`
  - 新增正式入口 `LAYERWEAVE_CACHE_POLICY=joint`；
  - 保留 `m4` 作为旧命令兼容别名。
- `docs/1.2-layerpipe.sh`
  - 接受并检查 `joint` policy。
- `tools/layerpipe/validate_layerweave_m5_5.py`
  - 新增 protected/soft 回收不变量验收。
- `tools/layerpipe/test_layerweave_joint_policy.py`
  - CPU-only 验证 MCKP 最优组合；
  - 验证容量压力下只回收 soft pages且 protected prefix 保持驻留。

静态与 CPU 单元检查：

```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/test_layerweave_joint_policy.py -v
/home/sdu/.conda/envs/sllm-worker/bin/python -m py_compile \
  tools/layerpipe/layerweave_cache_policy.py \
  tools/layerpipe/vllm_odkv_trace_bench.py \
  tools/layerpipe/validate_layerweave_m5_5.py \
  tools/layerpipe/test_layerweave_joint_policy.py
bash -n docs/1.2-layerpipe.sh
git diff --check
```

结果：两个 CPU 单元测试均通过。

GPU-0 四请求真实接线 smoke：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_ID=0 \
  CONFIG_PATH=configs/servegen_8_models_layerpipe_l40_pool42.json \
  TRACE_PATH=evaluation/traces/servegen_tangram.trace \
  LOAD_MODE=layerweave KV_BACKEND=odkv \
  VMM_POOL_GIB=42 VMM_PAGE_SIZE_MIB=64 \
  MAX_REQUESTS=4 MAX_BATCH_SIZE=1 \
  INPUT_SCALE=4 OUTPUT_TOKENS_OVERRIDE=1 \
  TRACE_TIME_SCALE=4 LAYERWEAVE_PREFETCH=1 \
  LAYERWEAVE_CACHE_POLICY=joint \
  LAYERWEAVE_M4_PROFILE=docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  LAYERWEAVE_DEMAND_DECAY=0.9 LAYERWEAVE_UNCERTAINTY_MS=100 \
  OUTPUT=docs/joint-mckp-smoke4.json \
  bash docs/1.2-layerpipe.sh
```

结果：

```text
requests/batches             = 4/4
service TTFT mean            = 657.08 ms
MCKP solve time              = 0.009--0.026 ms/dispatch
largest exact reclamation    = 320 soft pages
evicted protected pages      = 0
selected inactive floors     = model 2: prefix 8, model 6: prefix 4
LAYERPIPE_RESULT             = docs/joint-mckp-smoke4.json
```

最后一个请求需要加载 441 个 active weight pages 和 100 个 KV pages。
规划器将旧 model 2/6 分别降到 protected prefix 8/4，只回收 320 个 soft
pages；model 2 的 84 个 protected pages全部保留，model 6 在 42 个
protected pages之外还保留了 5 个未被实际需要回收的 soft pages。这验证
了 protected floor 与 opportunity residency 的解耦。

### 单机双卡、最小闭环对比 Joint MCKP

`docs/run_layerweave_joint_ab_2gpu.sh` 在两张独立 GPU 上同时重放完全相同
的 trace：一侧为最小闭环 `fixed/full`，另一侧为 `joint`。默认再交换
GPU 重复一轮，以减少 GPU 个体和 PCIe 拓扑偏差。每个进程仍是单 GPU
LayerWeave；这不是 M6 的跨 GPU 请求路由。

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_A=0 GPU_B=1 MAX_REQUESTS=100 SWAP_GPUS=1 \
  bash docs/run_layerweave_joint_ab_2gpu.sh
```

快速单轮版本：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_A=0 GPU_B=1 MAX_REQUESTS=100 SWAP_GPUS=0 \
  bash docs/run_layerweave_joint_ab_2gpu.sh
```

输出：

```text
round 1:
  docs/joint-ab-r1-minimal-gpu0.json
  docs/joint-ab-r1-joint-gpu1.json
  docs/joint-ab-r1-validation.txt
round 2 (swap):
  docs/joint-ab-r2-minimal-gpu1.json
  docs/joint-ab-r2-joint-gpu0.json
  docs/joint-ab-r2-validation.txt
```

两侧固定相同 config、trace、42 GiB pool、64 MiB page、batch size 1、
`INPUT_SCALE=4`、输出 1 token、trace scale 4 和 prefetch 1。运行期间
会共享主机 CPU、内存带宽和 PCIe，因此正式结论应同时报告两轮 GPU
互换结果，不只使用单轮。

### 单机双卡 A/B 实测结果（2026-07-26）

首次双卡启动在两侧创建 VMM pool 时失败：

```text
RuntimeError: Cannot create Tangram VMM pool:
cuMemCreate(preallocate native page): out of memory
```

检查发现两侧实际都在物理 GPU-0 分配。根因是 `docs/1.2-layerpipe.sh`
虽然打印 `GPU_ID`，构造 benchmark 命令时却硬编码了 `--device 0`。
修复为 `--device "$GPU_ID"`。同时 VMM page pool 初始化根据 CUDA Runtime
当前可见设备 UUID 查找对应 CUDA Driver device ordinal，使
`cuMemCreate/cuMemSetAccess` 和 torch/vLLM 始终指向同一物理 GPU。
重新构建：

```bash
cd /mnt/n0/Tangram/Tangram
cmake --build tools/mock_allocation/build \
  --target tangram_vram_backend -j2
```

GPU-1 四请求 smoke 期间 `nvidia-smi` 确认只有物理 GPU-1 占用约
43.5 GiB，4/4 请求完成。双卡脚本另外将两侧 pool 初始化错开 5 秒，
避免两个 42 GiB `cuMemCreate` burst 同时进入驱动；正式 trace replay
仍然并行。

正式命令：

```bash
cd /mnt/n0/Tangram/Tangram
env GPU_A=0 GPU_B=1 \
  MAX_REQUESTS=100 SWAP_GPUS=1 \
  STARTUP_STAGGER_SECONDS=5 \
  bash docs/run_layerweave_joint_ab_2gpu.sh
```

两轮 validation 均为：

```text
reservation_ok             = true
kv_estimate_ok             = true
reuse_observed             = true
protected_reclamation_ok   = true
evicted_protected_pages    = 0
M5_5_VALIDATION            = PASS
```

Round 1（minimal GPU-0，joint GPU-1）：

| 指标 | 最小闭环 | Joint | 改善 |
|---|---:|---:|---:|
| mean service TTFT | 715.68 ms | 703.65 ms | 12.03 ms / 1.68% |
| P50 service TTFT | 606.18 ms | 569.94 ms | 36.24 ms |
| P95 service TTFT | 1978.87 ms | 1871.92 ms | 106.95 ms |
| mean Prefill | 708.73 ms | 684.78 ms | 23.95 ms |
| mean weight H2D | 430.24 ms | 427.43 ms | 2.81 ms |

Round 2（minimal GPU-1，joint GPU-0）：

| 指标 | 最小闭环 | Joint | 改善 |
|---|---:|---:|---:|
| mean service TTFT | 720.84 ms | 706.05 ms | 14.79 ms / 2.05% |
| P50 service TTFT | 608.75 ms | 570.37 ms | 38.38 ms |
| P95 service TTFT | 2024.80 ms | 1894.61 ms | 130.20 ms |
| mean Prefill | 713.59 ms | 686.45 ms | 27.14 ms |
| mean weight H2D | 429.83 ms | 427.63 ms | 2.20 ms |

两轮 200 个请求合并：

```text
minimal mean service TTFT = 718.26 ms
joint mean service TTFT   = 704.85 ms
mean gain                 = 13.41 ms / 1.87%
paired faster/slower      = 72 / 128
paired median gain        = -3.37 ms
```

Joint 机制数据在两轮高度一致：

| 指标 | Round 1 | Round 2 |
|---|---:|---:|
| exposed load | 240.90 ms | 240.59 ms |
| 相对 minimal exposed 改善 | 22.14 ms | 22.33 ms |
| hidden load | 186.53 ms | 187.04 ms |
| 相对 minimal hidden 增加 | 19.33 ms | 20.13 ms |
| overlap ratio | 58.81% | 58.88% |
| 相对 minimal overlap 增加 | 7.17 pp | 7.26 pp |
| eviction batches/pages | 69 / 15360 | 69 / 15360 |
| eviction 总耗时 | 474.89 ms | 497.85 ms |
| MCKP mean/max | 0.091 / 0.362 ms | 0.095 / 0.565 ms |

因此 GPU 互换后结果可复现：Joint 没有依靠明显减少总 H2D 获益，而是把
更多 H2D 放入计算窗口，使 exposed load 稳定减少约 22 ms，mean
service TTFT 改善约 1.87%，P95 改善约 107--130 ms。与此同时，逐请求
中位 gain 仍为负，说明多数短请求支付了 eviction/规划之外的固定开销，
均值收益仍由长请求的大幅改善驱动；下一步优化目标应是批量 unmap 和
降低 69 个 eviction batch 的约 4.75--4.98 ms/batch 物理回收开销。

## M6 双 GPU 共享 Trace Runner（2026-07-26）

`tools/layerpipe/layerweave_multi_gpu_trace_bench.py` 实现了真正的双 GPU
系统实验，而不是把 minimal 和 Joint 分别放在两张卡上同时运行：

- coordinator 只读取一份 trace，并维护一个全局 arrival/pending queue；
- GPU 0/1 分别运行一个持久 worker，每个 worker 有独立的 42 GiB VMM
  weight/ODKV pool 和真实 residency；
- 每个请求只派发给一个 worker、只执行一次；
- worker 初始化可以错开，但 coordinator 等待两个 `READY` 后才统一开始
  trace replay；
- worker 完成请求后把真实 TTFT、H2D、overlap、cache policy 和 ODKV
  指标返回 coordinator；
- 最终输出一份系统级 JSON，包括 makespan、吞吐、全局延迟和逐 GPU
  分配。

第一版 work-conserving dispatcher 只比较 idle GPU；在 100 请求实验中
只有 `1/100` 请求真正比较了两张卡，其余请求退化为 first-free routing。
busyq1 随后允许：

```text
1 running request + 1 assigned pending request
```

但 100 请求结果中虽然 `98/100` 被派到 busy GPU，只有 `3/100` 次路由
同时拥有两个候选。原因是两张卡的 assigned slot 很快都被占满；之后每次
只有刚完成请求的 GPU 释放一个 slot，因此仍近似 first-free。该版本
Joint mean service TTFT 比 minimal 慢约 60.08 ms，95% bootstrap CI 为
`[-104.70, -18.89] ms`，吞吐下降约 7.65%，所以不能靠增加 per-GPU queue
depth 继续推进。

当前 globalq1 不再提前绑定 pending 请求：

```text
global pending queue
+ per-GPU running request
+ zero assigned pending request
```

每当至少一张 GPU 空闲，coordinator 从队首开始查询两张 GPU；只有请求
因 affinity 暂缓时才继续查看后续请求，最多检查
`ROUTING_LOOKAHEAD=8` 个 pending 请求：

- `minimal` 使用
  `predicted_queue_remaining + missing_weight_bytes / PCIe_bandwidth`，
  不使用 PSE、MCKP 或 future cache value；
- `joint` 使用
  `predicted_queue_remaining + predicted_service_ttft
  + transition_weight * transition_cost`；
- Joint 候选估价是只读操作，不会在目标 GPU 选定前修改 protection
  metadata 或回收页面；
- 若 busy GPU 至少优于 free GPU `PLACEMENT_HYSTERESIS_MS=25` ms，且预计
  等待不超过 `MAX_PLACEMENT_WAIT_MS=150` ms，请求暂留在 global queue；
- lookahead 可绕过这个请求，把更适合 free GPU 的后续请求立即发出；
- 达到 wait cap 后必须使用 free GPU；选定后才执行 protected MCKP 与
  exact-shortage soft-page reclamation。

worker 执行 forward 时不查询或修改 VMM。它在执行前冻结 projected
residency：已完成规划后的 inactive residency，加上即将完整加载的 active
model。独立 CPU route thread 基于该快照响应 busy-GPU estimate。请求真正
开始前仍基于 actual residency 重新运行 `plan_dispatch`。结果记录
`affinity_placement_wait_ms`、两卡预测 queue/service 和 prediction error。
Joint 的 `transition_weight` 默认从 1.0 降为 0.1；负 transition cost
带来的 future-value credit 另设 100 ms 上限，避免出现负总分主导当前请求。

这一定义避免用 round-robin 人为削弱 minimal，同时保持清晰消融：

```text
minimal = byte-locality placement + fixed/full LayerWeave cache
joint   = PSE/transition-aware placement + protected/soft Joint cache
```

### 双 GPU 两请求 smoke

```bash
cd /mnt/n0/Tangram/Tangram

env \
  GPU_IDS=0,1 \
  MAX_REQUESTS=2 \
  ROUTING_LOOKAHEAD=8 \
  MAX_PLACEMENT_WAIT_MS=150 \
  RUN_TAG=globalq1-smoke2 \
  TRACE_TIME_SCALE=4 \
  bash docs/run_layerweave_multi_gpu_ab.sh
```

该脚本顺序运行：

1. minimal 使用 GPU 0+1 共同处理一份 trace；
2. minimal 完全退出并释放两张卡；
3. Joint 使用 GPU 0+1 共同处理同一份 trace；
4. 检查 workload、设备、pool、请求唯一性和 protected-page invariants。

### 100 请求正式检查

```bash
cd /mnt/n0/Tangram/Tangram

nohup env \
  GPU_IDS=0,1 \
  MAX_REQUESTS=100 \
  ROUTING_LOOKAHEAD=8 \
  MAX_PLACEMENT_WAIT_MS=150 \
  PLACEMENT_HYSTERESIS_MS=25 \
  TRANSITION_WEIGHT=0.1 \
  TRANSITION_CREDIT_CAP_MS=100 \
  RUN_TAG=globalq1-100 \
  TRACE_TIME_SCALE=4 \
  bash docs/run_layerweave_multi_gpu_ab.sh \
  > docs/layerweave-2gpu-globalq1-100-driver.log 2>&1 &
echo $!
```

先检查 100 请求结果中的：

```text
two_gpu_route_decisions
requests_deferred_for_affinity
affinity_placement_wait_ms
route_prediction_error_ms
```

只有两卡比较确实覆盖绝大多数请求、预测误差和吞吐没有恶化后，再决定是否
扩大到 1000 请求；当前不建议直接运行 1000。

### Serial-choice 理想放置上界实验

`SCHEDULER_MODE=serial-choice` 是一个刻意极端的诊断模式：

- 忽略 trace arrival time，replay 开始时所有请求均可用；
- 全系统任意时刻只允许一个请求执行；
- 当前请求完成后才放置下一个请求；
- 每次放置时 GPU 0 和 GPU 1 必定都 idle；
- 请求保持原始 trace 顺序，不使用 lookahead、placement wait 或
  hysteresis；
- 两张 GPU 的 VMM pool、cache residency 和历史状态保持独立并持续演化。

因此每个请求都有一次真实的二选一 placement。该模式用于回答：

> 在完全消除到达负载和 busy-GPU 约束后，LayerWeave 的 Joint
> cache/placement 是否存在理想条件下的收益上界？

该模式的 `service_ttft_ms`、H2D、cache hit、eviction 和逐请求 paired gain
可以比较。`throughput_requests_s` 仅表示系统级串行诊断吞吐，不能解释为
双 GPU 并发吞吐。`trace_time_scale` 在该模式中被记录，但不参与执行。

为避免 missing-byte greedy 在第一次冷启动后永久吸附到同一张卡，可选：

```text
MINIMAL_COLD_TIE_RANDOM=1
```

它只在两张 idle GPU 对当前模型都为 `cached_pages=0` 时使用固定 seed RNG
破局；任一卡已有至少一页可复用缓存时，Minimal 仍严格选择 missing bytes
更少者。Joint 可对应启用 `JOINT_COLD_TIE_RANDOM=1`，使跨 GPU 模型播种
不依赖巨额 transition penalty。

长同模型请求序列在 request 48/model 2/input 8967 上稳定触发过 246 MiB
activation allocator OOM。`ALLOCATOR_TRIM_EVERY_REQUEST=1` 对两版都在每个
请求前执行 `torch.cuda.empty_cache()`，不改变 VMM residency，并记录
`allocator_trim_ms`。

两请求 smoke：

```bash
cd /mnt/n0/Tangram/Tangram

env \
  GPU_IDS=0,1 \
  SCHEDULER_MODE=serial-choice \
  MAX_REQUESTS=2 \
  RUN_TAG=serial-choice-smoke2 \
  bash docs/run_layerweave_multi_gpu_ab.sh
```

100 请求正式运行：

```bash
cd /mnt/n0/Tangram/Tangram

nohup env \
  GPU_IDS=0,1 \
  SCHEDULER_MODE=serial-choice \
  MAX_REQUESTS=100 \
  MINIMAL_COLD_TIE_RANDOM=1 \
  JOINT_COLD_TIE_RANDOM=1 \
  ALLOCATOR_TRIM_EVERY_REQUEST=1 \
  TRANSITION_WEIGHT=0.1 \
  TRANSITION_CREDIT_CAP_MS=100 \
  TRANSITION_PENALTY_CAP_MS=100 \
  RUN_TAG=serial-choice-final-100-r1 \
  bash docs/run_layerweave_multi_gpu_ab.sh \
  > docs/layerweave-2gpu-serial-choice-final-100-r1-driver.log 2>&1 &
echo $!
```

验收要求：

```text
arrival_time_ignored = true
both_gpus_idle_route_decisions = requests
two_gpu_route_decisions = requests
routed_to_busy_gpu = 0
requests_deferred_for_affinity = 0
```

#### 增强 Minimal 后的优化结论

原始 Minimal 因冷启动 tie 固定选择 GPU 0，100 个请求全部落到一张卡；
Joint 相对它曾显示约 10.2% mean service TTFT 收益。启用 cold-tie random
后，Minimal 分配稳定为 56/44，mean service TTFT 从约 725.8 ms 降至
约 650 ms，证明原始 10% 主要包含弱 baseline 的单卡吸附。

Joint 随后完成以下修复：

- route estimate 只读模拟正式 `observe()` 的 demand decay/current demand；
- route estimate 的 pending 集合移除当前请求，与正式 plan 一致；
- estimator 的 request-shape/configuration 曲线缓存，避免 route/plan 重算；
- transition contribution 对称限制为 `[-100,+100] ms`；
- Joint 在双零缓存时同样使用固定 seed 冷播种；
- serial-choice 复用 chosen worker 的 idle residency snapshot，正式 plan
  不再重复查询 VMM；online 模式仍强制 actual-state 重查；
- worker 失败时记录 device/request/model/input 并清理另一 worker。

最终两次完整 100-request A/B 的稳定策略结果为：

| 指标 | 增强 Minimal | Joint |
|---|---:|---:|
| GPU 分配 | 56 / 44 | 51 / 49 |
| mean H2D | 约 288.2 ms | 约 254.4 ms |
| mean exposed load | 约 177.6 ms | 约 157.8 ms |
| eviction pages | 0（由 runtime 管理） | 8178 protected-safe pages |
| mean service gain | - | 10.9--17.1 ms（1.7--2.6%） |
| throughput gain | - | 1.0--2.1% |

两轮逐请求 pooled mean gain 为约 14.0 ms，bootstrap 95% CI
`[1.96, 26.83] ms`。物理 H2D/exposed 收益高度复现，但 service gain
仍受少量长请求影响。Joint 剩余 plan 约 6 ms/request，其中约
4.6 ms/request 是物理 eviction apply；进一步明显提升需要 C++ 批量 unmap
或 next-use/Belady 风格的新 cache policy，而不是继续调整 transition 参数。

输出：

```text
docs/layerweave-minimal-2gpu-<tag>.json
docs/layerweave-joint-2gpu-<tag>.json
docs/layerweave-2gpu-<tag>-validation.txt
```

`STARTUP_STAGGER_SECONDS` 默认 5 秒，只错开同一系统内部两个 42 GiB
pool 的 `cuMemCreate`；两个 worker 都 ready 后才开始共享 trace，因此
不会改变请求到达时间。`RUN_TAG` 进入文件名，100/1000 请求不会互相覆盖。

### C++ batch unmap、next-use 上限与 online 复测（2026-07-27）

#### C++ 连续区间 batch unmap

原 `RetainStablePages()` 对每个淘汰页分别调用一次 `cuMemUnmap`。
`LAYERWEAVE_BATCH_UNMAP=1`（launcher 中为 `BATCH_UNMAP=1`）把连续虚拟页
合并成一个区间调用，然后逐页更新 physical extent 元数据；若 CUDA driver
拒绝跨 physical allocation 的区间 unmap，会自动回退到逐页调用。

相同 100-request serial-choice Joint、相同 51/49 路由的开关消融：

| 指标 | batch off | batch on |
|---|---:|---:|
| evicted pages | 8178 | 8178 |
| eviction apply | 5.30 ms/request | 4.71 ms/request |
| plan total | 7.16 ms/request | 6.39 ms/request |
| service TTFT | 630.98 ms | 630.48 ms |

结论：driver 调用合并降低约 0.6--0.8 ms/request 控制面开销，但不是当前
端到端加速的主要来源。

#### next-use/Belady 风格的离线上限

`CACHE_POLICY_MODE=next-use` 只允许
`SCHEDULER_MODE=serial-choice` + Joint。它利用剩余请求顺序，将 inactive
resident pages 按所属模型的下一次使用位置排序，优先淘汰下一次使用最远或
不再使用的页。它是 clairvoyant 上限诊断，不是可部署 online 策略。

缓存策略必须与路由正交比较。直接替换 demand protection 会改变后续
transition state，使路由从 51/49 退化到 56/44，H2D 从约 254 ms 回到
约 288 ms。因此正式 cache-only 消融用 `--routing-replay` 重放 demand
baseline 的全部 request-to-GPU 决策，并只把未来会到达该 GPU 的请求传给
对应 worker。

固定路由 100/100 一致时：

| 指标 | demand cache | next-use cache |
|---|---:|---:|
| mapped pages | 9351 | 9206 |
| evicted pages | 8178 | 8033 |
| mean H2D | 254.45 ms | 249.59 ms |
| mean service TTFT | 630.480 ms | 630.445 ms |

next-use 只再减少 145 mapped pages（1.55%）和 4.85 ms H2D，端到端基本
不变。此前 whole-model DP 的更大差距同时包含未来感知路由/模型分区收益，
不能归因于单卡 eviction policy。

#### 保留 arrival time 的双 GPU online A/B

最终 online 命令：

```bash
cd /mnt/n0/Tangram/Tangram

nohup env \
  GPU_IDS=0,1 \
  SCHEDULER_MODE=online \
  MAX_REQUESTS=100 \
  TRACE_TIME_SCALE=4 \
  ALLOCATOR_TRIM_EVERY_REQUEST=1 \
  BATCH_UNMAP=1 \
  CACHE_POLICY_MODE=demand \
  TRANSITION_WEIGHT=0.1 \
  TRANSITION_CREDIT_CAP_MS=100 \
  TRANSITION_PENALTY_CAP_MS=100 \
  RUN_TAG=online-final-100-r3 \
  bash docs/run_layerweave_multi_gpu_ab.sh \
  > docs/layerweave-2gpu-online-final-100-r3-driver.log 2>&1 &
echo $!
```

两轮独立 A/B 均为 Minimal 独占 GPU 0+1 完整运行后清理，再由 Joint 独占
GPU 0+1 运行；两版不会同时执行。validation 均 PASS，protected eviction
均为 0。

| 指标 | r1 Minimal | r1 Joint | r2 Minimal | r2 Joint |
|---|---:|---:|---:|---:|
| makespan (s) | 37.60 | 36.54 | 37.43 | 36.30 |
| throughput (req/s) | 2.659 | 2.737 | 2.672 | 2.755 |
| mean service TTFT (ms) | 734.16 | 710.58 | 732.37 | 701.89 |
| mean trace TTFT (s) | 18.95 | 18.33 | 18.76 | 18.32 |
| mean H2D (ms) | 434.75 | 408.02 | 432.30 | 384.30 |
| mean exposed load (ms) | 268.98 | 238.80 | 266.65 | 228.24 |
| mapped pages | 16159 | 15226 | 16069 | 14291 |

系统吞吐收益两轮为 2.91% 和 3.10%，makespan 降低 2.83% 和 3.01%。
两轮 pooled mean service gain 为 27.03 ms，paired bootstrap 95% CI
`[5.47, 50.20] ms`。paired median 为约 -1.67 ms，且 Joint 只在
93/200 个逐请求 pair 上更快：收益集中在避免少数昂贵 reload，而非每个
请求普遍变快。当前 workload 下可复现的完整系统结论应表述为约 3% 系统
吞吐收益，同时单独报告稳定的 mapped-page/exposed-load 下降。

### Profile-driven pipeline cache simulator（2026-07-27）

新增 CPU-only 理论模拟器：

```text
tools/layerpipe/layerweave_pipeline_cache_sim.py
```

输入来自 M4 report 的逐层 compute regression/H2D bandwidth、cold
calibration 中每个 stage 实际新增的 unique 64 MiB pages、ServeGen trace、
config input cap，以及与实际 Joint 一致的 672-page pool 和 32-token KV
block reservation。

模拟器采用 system-wide serial request order，隔离 cache/pipeline 机制。
Minimal 复现当前 VMM 的 model-frequency value 和 page-LRU tie break；
顺序 forward 后同模型前缀最老。Pipeline-aware policy 从 cold state 构造
条件 greedy residency ladder。对于当前 resident set \(R\)，逐步加入使
M4 predicted GPU critical path 下降最多的 stage page：

\[
V_{\mathrm{pipe}}(m,p\mid R)
=T_{\mathrm{crit}}(m,R)-T_{\mathrm{crit}}(m,R\cup\{p\}).
\]

这直接表达“释放可重叠后缀、保留不可重叠前缀”的空间–流水交换。

#### Profile loader 修复

模拟器发现 JSON 中 `compute` 的 layer key 是字符串，而
`PipelineEstimator._simulate_gpu()` 用整数 layer 查询。旧 reload 路径因此
把所有逐层 compute 当成 0。修复后 simulator 和正式
`LayerWeaveSingleGpuCachePolicy` 都在加载时将 compute key 规范化为整数。

这意味着旧 GPU A/B 的 estimator 实际接近 demand-aware byte cache，并未
正确使用 M4 compute overlap。修复后的真实 GPU A/B 必须重新运行，旧的约
3% 结果不能直接代表修复后性能。

#### 理论结果

100-request、固定 Minimal request-to-GPU 路由：

| 指标 | Minimal | Pipeline-aware |
|---|---:|---:|
| predicted critical path | 516.38 ms | 505.40 ms |
| exposed load | 187.44 ms | 176.45 ms |
| H2D | 269.77 ms | 282.08 ms |
| missing pages | 9866 | 10305 |

Pipeline-aware 多加载 439 个可重叠后缀页、增加 12.32 ms H2D，但 exposed
load 减少 10.99 ms（5.86%），critical path 改善 2.13%。

1000-request：

| 指标 | Minimal | Pipeline-aware |
|---|---:|---:|
| predicted critical path | 381.24 ms | 370.55 ms |
| exposed load | 94.97 ms | 84.28 ms |
| H2D | 142.90 ms | 150.11 ms |
| missing pages | 52256 | 54728 |
| stage-0 evicted pages | 8380 | 4223 |

Pipeline-aware 用额外 7.22 ms H2D 换取 10.69 ms exposed-load 下降，
critical path 改善 2.80%。各模型代表请求的 cold H2D hidden fraction
约为 8%--87%。

容量扫描（1000 request、固定路由）：

| Pool pages/GPU | critical-path gain | exposed-load gain |
|---:|---:|---:|
| 640 | 3.13% | 11.29% |
| 672 | 2.80% | 11.25% |
| 896 | 2.26% | 16.13% |

绝对 critical-path 收益在容量更紧时更大，但当前 Profile 下理论端到端收益
仍约为 2%--3%，因为被释放的后缀并不是免费加载，只是更容易 overlap。

运行命令：

```bash
cd /mnt/n0/Tangram/Tangram

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_pipeline_cache_sim.py \
  --max-requests 1000 \
  --pool-pages 672 \
  --output docs/layerweave-pipeline-cache-sim-1000.json
```

模型边界：不模拟 arrival queue、并发 CUDA contention、allocator 波动或
service residual；page-to-stage ownership 使用 cold run 的 unique
mapped-page counts，边界共享页归入首次加载它的 stage。它回答的是
pipeline-aware cache ordering 的 Profile 理论收益，不替代真实 GPU A/B。

#### 五项机制分解

模拟器同时输出 `five_way_ablation`。五项使用相同请求；LayerWeave 固定
replay Minimal 的 request-to-GPU 路由，以免 placement 差异污染 cache
对比：

| 模式 | Cache | H2D/compute |
|---|---|---|
| baseline load+compute | 每次全冷 | 串行 |
| pipe-only | 每次全冷 | 逐层流水 |
| reuse-only | Minimal cache state | 串行 |
| Minimal | Minimal cache state | 逐层流水 |
| LayerWeave | pipeline-aware cache state | 逐层流水 |

前 1000 请求、672 pages/GPU 的均值：

| 模式 | predicted total | H2D | exposed load | gain vs baseline |
|---|---:|---:|---:|---:|
| baseline load+compute | 871.02 ms | 584.75 ms | 584.75 ms | 0% |
| pipe-only | 639.52 ms | 584.75 ms | 353.25 ms | 26.58% |
| reuse-only | 429.17 ms | 142.90 ms | 142.90 ms | 50.73% |
| Minimal | 381.24 ms | 142.90 ms | 94.97 ms | 56.23% |
| LayerWeave | 370.55 ms | 150.11 ms | 84.28 ms | 57.46% |

所有模式的 mean hot compute 均为 286.27 ms。当前 LayerWeave 不是提高
总 page hit rate：相同路由下 missing pages 从 52256 增至 54728；它保留
关键路径价值更高的页，用额外 7.22 ms H2D 换取 10.69 ms exposed-load
下降。因此更准确的目标是提高 pipeline-aware effective hit value，而不是
单纯提高 page hit count。

`five_way_ablation.inspected_request` 进一步输出一个具体请求的完整时间线。
默认选择 LayerWeave 正收益请求中的中位案例，避免只展示最大尾部收益；
也可用 `--inspect-request-id N` 指定。每条 pipeline timeline 包含逐 stage
的 `start_ms/end_ms/missing_pages`、逐层 compute timeline、每层前的
`ready_stall_before_ms`，以及 initial ready、总 ready stall 和 tail compute。

默认 1000-request 结果选择 request 61（model 2，input 1684）：

| 模式 | total | 关键构成 |
|---|---:|---|
| baseline | 839.83 ms | cold load 656.10 + hot compute 183.73 |
| pipe-only | 661.76 ms | initial 106.06，total ready stall 478.16 |
| reuse-only | 262.63 ms | 29 missing pages load 78.90 + compute 183.73 |
| Minimal | 262.50 ms | 29 个 missing page 全在 stage 0，initial stall 78.90 |
| LayerWeave | 183.73 ms | 31 个 missing page 分散在 stage 1--31，几乎全部隐藏 |

这个请求具体展示了 LayerWeave 的目标：即使 missing pages 从 29 增至 31，
只要把缺页从阻塞首层的 prefix 移到可与前层计算重叠的 suffix，关键路径
仍可降低 78.77 ms。

#### Configuration-MCKP 模拟（2026-07-27）

模拟器现已把文档中的完整配置级机制设为默认 LayerWeave，并保留旧
`pipeline-aware page greedy` 作为消融：

1. 每个模型建立 `{0,2,4,8,16,32,full}` protected-prefix curve；
2. 配置收益对该模型在 trace 中的多种 input shape 求期望，不再只用 median；
3. 期望收益乘以 `--demand-decay`（默认 0.9）的在线模型 demand；
4. 每次 dispatch 在 active model full working set 和 KV 预算之外，对所有
   inactive 模型求精确 page-granular MCKP；
5. inactive protected configuration 只能维持或降低，模型再次 active 后可
   恢复 full；
6. MCKP 只更新 protected floor，配置外页面保持 soft，容量不足时按
   configuration tier 和 marginal value/page 回收，不驱逐 protected page。

固定 Minimal 路由的结果：

| Trace | Minimal | page-greedy | configuration-MCKP |
|---|---:|---:|---:|
| 1000 request / 2 GPU | 381.24 ms | 370.55 ms (-2.80%) | 380.09 ms (-0.30%) |
| 5000 request / 2 GPU | 381.15 ms | 368.14 ms (-3.41%) | 381.63 ms (+0.13%) |

5000-request MCKP 平均保护 284.8 个 inactive pages/GPU-decision，但 missing
pages 从 Minimal 的 264378 增至 281933，mean H2D 从 144.50 增至
154.20 ms，最终 exposed load 从 96.54 增至 97.02 ms。正收益与负收益
分别累计 38.29 s 和 -40.71 s，近乎完全抵消。

这是一项负结果，但区分了设计表达与收益：configuration-MCKP、protected
floor 和 deferred reclamation 已被模拟；当前离散 prefix configurations
和 demand/value 模型没有提高总体 hit rate，且比逐页 critical-path value
更粗。后续优化应先在模拟器中改进 configuration set/value，而不能把旧
page-greedy 的约 3% 当作完整 MCKP 的结果。

#### Queue lookahead oracle（1000 request）

模拟器新增 `--lookahead-k` 和 `--lookahead-discount`。固定 Minimal 路由后，
每张 GPU 在当前请求规划时查看其未来 K 个真实请求，直接使用真实 model ID
和 input length 计算 configuration value：

\[
v_{m,k}(t)=
\sum_{j=1}^{K}
\mathbf 1[m_j=m]\gamma^{j-1}
b_{m,k}(r_j).
\]

Lookahead 可以把仍 resident 的 soft prefix 重新提升为 protected，但不会
预取缺失页。该实验查看尚未 arrival 的 trace 请求，因此是 cache oracle，
不是可直接部署的 online 策略。

1000-request、2-GPU、672 pages/GPU、固定路由、\(\gamma=1\)：

| Policy | critical path | vs Minimal | exposed load | H2D | missing pages |
|---|---:|---:|---:|---:|---:|
| Minimal | 381.24 ms | -- | 94.97 ms | 142.90 ms | 52256 |
| demand-MCKP | 380.09 ms | +0.30% | 93.82 ms | 151.82 ms | 55499 |
| lookahead K=1 | 383.52 ms | -0.60% | 97.25 ms | 159.62 ms | 58314 |
| lookahead K=4 | 376.69 ms | +1.19% | 90.42 ms | 147.77 ms | 54019 |
| lookahead K=8 | 376.10 ms | +1.35% | 89.83 ms | 145.97 ms | 53360 |
| lookahead K=16 | 377.05 ms | +1.10% | 90.78 ms | 147.23 ms | 53820 |
| lookahead K=32 | 377.60 ms | +0.95% | 91.33 ms | 148.32 ms | 54220 |
| page-greedy | 370.55 ms | +2.80% | 84.28 ms | 150.11 ms | 54728 |

K=8 最优，与该 trace 的 per-GPU model reuse-distance P90 约 8--9 一致。
K=1 只保护最近一次未来访问，导致更远请求 cache damage；K 太大则累计价值
趋近静态频率并过度保护更多 inactive pages。即使使用真实未来 model/shape，
configuration-MCKP 仍低于 page-greedy，说明访问预测不是唯一或主要上限；
离散 prefix configuration 和配置价值对实际 soft residency 的表达仍是主要
问题。

运行命令：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_pipeline_cache_sim.py \
  --max-requests 1000 \
  --gpus 2 \
  --pool-pages 672 \
  --lookahead-k 1 4 8 16 32 \
  --output docs/layerweave-pipeline-cache-sim-1000-lookahead.json
```

#### Exact residency-transition planner

在 configuration-MCKP 之外，模拟器新增真实物理状态转换规划：

1. 当前请求的 active model 在完成后视为 full resident；
2. 根据真实 resident pages、当前请求 missing pages、KV 和 pool 计算 exact
   shortage；
3. 将 inactive resident pages 按 model/stage 分组；
4. 对每组页面直接计算从当前 \(R\) 驱逐后，对未来 K 个真实请求造成的
   marginal critical-path damage；
5. 按 damage/page 选择恰好满足 shortage 的 victim pages；
6. 输出实际 post-reclamation residency，而非用 prefix-only
   configuration 代替。

1000-request 固定 Minimal 路由结果：

| K | critical path | vs Minimal | exposed load | H2D | missing pages |
|---:|---:|---:|---:|---:|---:|
| 1 | 378.94 ms | +0.61% | 92.66 ms | 150.95 ms | 55145 |
| 4 | 371.76 ms | **+2.49%** | 85.49 ms | **140.25 ms** | **51260** |
| 8 | 371.96 ms | +2.44% | 85.69 ms | 139.63 ms | 51031 |
| 16 | 372.62 ms | +2.26% | 86.35 ms | 141.04 ms | 51548 |
| 32 | 372.94 ms | +2.18% | 86.67 ms | 141.27 ms | 51628 |

对照：

```text
Minimal              381.24 ms, H2D 142.90 ms, missing 52256
page-greedy           370.55 ms, H2D 150.11 ms, missing 54728
configuration K=4     376.69 ms, H2D 147.77 ms, missing 54019
exact-residency K=4   371.76 ms, H2D 140.25 ms, missing 51260
```

Exact-residency K=4 虽比 page-greedy 的关键路径高 1.21 ms，但它首次同时降低
critical path、H2D 和 missing pages，符合“牺牲低 damage 后缀、把空间留给
更有价值页面”的原始目标。Top-B configuration exact re-ranking 几乎没有
改善，因为不同 configuration 通常映射到相同 victim set；真正有效的是直接
在当前物理 residency 上评价 stage-group transition damage。

#### MCKP + Page-greedy hybrid

模拟器新增 `configuration-mckp-page-greedy`，将全局容量规划与物理回收
解耦：

1. MCKP 根据 demand 或 lookahead 为每个 inactive model 选择 protected
   prefix floor；
2. protected pages 不允许被驱逐；
3. 发生 exact shortage 时，在所有 inactive soft pages 上按
   `future demand × pipeline marginal page value` 全局排序；
4. 只驱逐恰好满足 shortage 的最低价值 soft pages。

1000-request、固定 Minimal 路由结果：

| Policy | critical path | vs Minimal | exposed load | H2D | missing pages |
|---|---:|---:|---:|---:|---:|
| Minimal | 381.24 ms | — | 94.97 ms | 142.90 ms | 52256 |
| Page-greedy | 370.55 ms | +2.80% | 84.28 ms | 150.11 ms | 54728 |
| Demand-MCKP | 380.09 ms | +0.30% | 93.82 ms | 151.82 ms | 55499 |
| Hybrid demand | 377.71 ms | +0.93% | 91.43 ms | 150.67 ms | 55065 |
| Exact-residency K=4 | 371.76 ms | +2.49% | 85.49 ms | 140.25 ms | 51260 |
| **Hybrid K=8** | **371.25 ms** | **+2.62%** | **84.98 ms** | **139.56 ms** | **50995** |

只看 Minimal 非完全命中的 296 个请求，Hybrid K=8 相对 Minimal 提升
`6.12%`；Page-greedy 为 `6.47%`，Exact-residency K=4 为 `6.73%`。
因此 Hybrid 已获得与 Page-greedy 相近的关键路径性能，同时避免其额外
H2D/cache churn，并且不需要 Exact-residency 对每个 stage 反复运行
critical-path estimator。结果文件：

```text
docs/layerweave-pipeline-cache-sim-1000-mckp-page-greedy.json
```

Hybrid 可以与高开销 oracle 完全隔离：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_pipeline_cache_sim.py \
  --policy-suite mckp-page-greedy \
  --max-requests 1000 \
  --gpus 2 \
  --pool-pages 672 \
  --lookahead-k 1 4 8 16 32 \
  --output \
    docs/layerweave-pipeline-cache-sim-1000-mckp-page-greedy-isolated.json
```

该模式只执行 Minimal、Hybrid-demand 和指定 K 的 Hybrid，不执行普通
Page-greedy、Demand-MCKP、configuration-transition 或 Exact-residency。
Lookahead 使用预建 per-GPU request index，读取队列前 K 项为 \(O(K)\)。

独立运行中 Hybrid K=8 的 controller time 为：

```text
mean 0.367 ms, P90 0.526 ms, P95 0.602 ms,
P99 0.717 ms, max 0.878 ms
```

其中队列读取均值仅 `0.00084 ms`；主要控制成本是 MCKP 和 soft-page 排序。
K=8 的预测 critical path 为 mean `371.25 ms`、P90 `926.23 ms`、P95
`1269.79 ms`、P99 `1675.23 ms`。P95/P99 与 Minimal 相同，说明当前 trace
的尾部由长输入的 hot compute/cold-like request 主导，缓存策略主要改善均值
和中部 miss 请求。整个 Python 模拟器 wall time 为 `41.89 s`，其中还包括
离线 profile/configuration curve 构建和七次完整 trace replay，不能作为在线
controller latency。
