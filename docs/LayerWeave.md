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
