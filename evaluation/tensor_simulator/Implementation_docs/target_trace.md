# Prefix-MCKP 目标 Trace 探索

## 1. 目标与固定对照

本轮不再继续搜索 Segment 策略参数，而是构造不同类型的 trace，得到多个
可复现的性能提升档位。固定比较：

- 基线：普通 TensorGroup LRU + cache-bytes routing；
- 候选：Prefix-MCKP + transition routing。

除非单独说明，公共配置为：

```text
GPU 数量                    = 2
内存后端                    = Segment request-level PBP
runtime TensorGroup 最小值  = 64 MiB
allocator page              = 8 MiB
H2D                         = 24.56 GB/s
input-scale                 = 4
Offline table               = offline-prefix-stall-runtime-group64.json
lookahead discount          = 0.9
transition weight           = 0.1
```

文中的 mean TTFT 提升定义为：

```text
(LRU mean critical_path_ms - MCKP mean critical_path_ms)
--------------------------------------------------------- * 100%
                LRU mean critical_path_ms
```

平均值和尾延迟分别报告。平均值改善不等价于每个尾延迟分位点都改善。

## 2. 现有 Trace 生成路径

本轮阅读了：

- `docs/legacy/reuse_only_tangram/0.1-trace_gen.md`；
- `evaluation/traces/generate_trace_tangram.py`；
- `evaluation/traces/clientpool.py`；
- `evaluation/traces/generate_trace_random.py`；
- `evaluation/traces/show_trace_pattern.py`；
- `tools/layerpipe/build_layerweave_trace_variants.py`。

`generate_trace_tangram.py` 首先按照 ServeGen `pattern` 对模型分组。对于
共享同一 pattern 的模型，`--skew-alpha` 生成 power-law 目标比例，
`split_clients_shape_preserving()` 再把完整 client 映射到各个模型，
同时近似保持 rate、CV 和 token shape。它适合构造比较真实的
client-to-model skew，但不能直接指定精确的 request-index reuse distance。

`show_trace_pattern.py` 把 reuse distance 定义为两次访问之间出现过多少个
不同模型。`mckp_cache.md` 的 paired analysis 使用两次访问之间相隔的请求
数量。因为这里只有 8 个 model ID，distinct-model distance 无法表达
16--31 这样的距离；分析 MCKP 的 K-window 时，本文统一使用
request-index distance。

## 3. 新增目标 Trace 生成器

新增脚本：

```text
evaluation/traces/generate_target_trace.py
```

它不依赖 ServeGen 安装，但可以从 `servegen_tangram.trace` 中复用每个模型
真实的 `(input_tokens, output_tokens)` 样本池。支持：

- `period`：让指定模型按精确 request period 出现，其余位置由加权背景模型填充；
- `cyclic`：每个 block 都随机排列全部 model ID；
- `iid`：按权重随机抽取模型，并避免连续两次访问相同模型；
- 自定义 target/background model 集合；
- 自定义模型权重，用于模拟不同的 client-to-model mapping；
- 使用原始 token shape、缩放 token shape 或固定输入长度。

### 3.1 精确控制大模型 reuse period

```bash
python3 evaluation/traces/generate_target_trace.py \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --source evaluation/traces/servegen_tangram.trace \
  --output evaluation/traces/target/large-period16.trace \
  --requests 500 --sequence period \
  --target-models 4,5,7 --background-models 0,1,2,3 \
  --target-period 16 --seed 1234
```

这个 trace 中大模型 4/5/7 的 request-index reuse distance 精确为 16。

### 3.2 八模型均衡循环

```bash
python3 evaluation/traces/generate_target_trace.py \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --source evaluation/traces/servegen_tangram.trace \
  --output evaluation/traces/target/all8-cyclic.trace \
  --requests 500 --sequence cyclic --seed 1234
```

### 3.3 当前最大提升 Trace

```bash
python3 evaluation/traces/generate_target_trace.py \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --source evaluation/traces/servegen_tangram.trace \
  --output evaluation/traces/target/all8-cyclic-fixed256.trace \
  --requests 500 --sequence cyclic \
  --fixed-input-tokens 256 --output-tokens 1 --seed 1234
```

原始输入为 256，经过模拟器 `input-scale=4` 后目标 effective input 为 1024。

### 3.4 大模型高热度

```bash
python3 evaluation/traces/generate_target_trace.py \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --source evaluation/traces/servegen_tangram.trace \
  --output evaluation/traces/target/large-hot-iid.trace \
  --requests 500 --sequence iid \
  --model-weights 0:1,1:1,2:1,3:1,4:4,5:4,6:0.2,7:4 \
  --seed 1234
```

### 3.5 小/中模型高热度

```bash
python3 evaluation/traces/generate_target_trace.py \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --source evaluation/traces/servegen_tangram.trace \
  --output evaluation/traces/target/small-hot-iid.trace \
  --requests 500 --sequence iid \
  --model-weights 0:5,1:5,2:3,3:3,4:0.3,5:0.3,6:0.1,7:0.3 \
  --seed 1234
```

## 4. 放宽模拟器输入限制

原来的 trace loader 总是应用 `l40_safe_max_input_length`，因为配置文件是
为真实 45-GB L40 推理生成的。本轮新增两个 opt-in 参数，默认行为不变：

```text
--input-limit-policy safe|context|none
--stall-table-input-policy clamp|linear
```

- `safe`：原行为，使用真实 GPU 验证过的安全上限；
- `context`：使用模型架构的 `model_context_tokens`；
- `none`：不使用模型输入上限，只保留模拟显存/KV 可行性上限；
- `clamp`：超过 Offline table 最大输入点后使用最后一行；
- `linear`：使用 Offline table 最后两个输入点做线性外推。

`linear` 是仅用于模拟器的近似扩展。策略估计和 simulator settlement
使用同一个扩展 table，因此 A/B 内部一致，但它不是新的 GPU 测量结果，
必须与 measured-range 结果分开标记。

即使选择 `none`，仍然保留 pool feasibility cap：单个请求必须能够在目标
GPU 上同时容纳模型权重和请求 KV。

## 5. 公共实验命令

```bash
COMMON="\
  --tensor-layout evaluation/tensor_simulator/results/tensor-layout.json \
  --profile tools/layerpipe/results/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --gpus 2 --trace-mode model-switches --input-scale 4 \
  --memory-layout segment --page-size-mib 8 \
  --h2d-gbps 24.56 --tensor-group-min-mib 64 \
  --policy-suite minimal-only \
  --mckp-stall-table-input \
    evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json"
```

普通 LRU 基线：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
  $COMMON --trace TRACE --pool-gib POOL --max-requests REQUESTS \
  --replacement-policy lru --routing-policy cache-bytes \
  --output OUTPUT_LRU
```

Prefix-MCKP：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
  $COMMON --trace TRACE --pool-gib POOL --max-requests REQUESTS \
  --replacement-policy mckp-prefix \
  --routing-policy mckp-transition \
  --mckp-prediction-mode lookahead \
  --mckp-lookahead-k K \
  --mckp-lookahead-discount 0.9 \
  --mckp-transition-weight 0.1 \
  --output OUTPUT_MCKP
```

## 6. 搜索过程

### 6.1 模型序列与 skew 搜索

以下均为 42 GiB/GPU、K=32、200 请求：

| Trace | 主要特征 | Mean TTFT：LRU → MCKP | 提升 |
|---|---|---:|---:|
| all8-cyclic | 八模型均衡，reuse P50 约 7--8 | 736.6 → 623.1 ms | **15.41%** |
| large-period8 | 模型 4/5/7 period=8 | 699.3 → 604.5 ms | 13.56% |
| large-period16 | 模型 4/5/7 period=16 | 523.3 → 475.8 ms | 9.07% |
| large-period24 | 模型 4/5/7 period=24 | 475.2 → 438.6 ms | 7.70% |
| large-hot-iid | 大模型抽取权重为 4x | 678.8 → 637.4 ms | 6.10% |
| large-period32 | 模型 4/5/7 period=32 | 403.0 → 383.3 ms | 4.89% |
| small-hot-iid | 小/中模型占主导 | 303.2 → 290.7 ms | 4.13% |

结论：仅仅让大模型变冷并不能获得最大提升。让全部大模型稳定地进入一个
持续复用的 working set，收益更明显。

### 6.2 模拟显存与输入长度

八模型 balanced cyclic、39 GiB/GPU、K=32、200 请求：

| Trace 原始固定输入 | scale=4 后目标输入 | Mean TTFT 提升 |
|---:|---:|---:|
| 32 | 128 | 19.02% |
| 256 | 1024 | **20.60%** |
| 1024 | 4096 | 16.96% |
| 4096 | safe/pool cap 前为 16384 | 15.35% |
| 原始 empirical token pool | 依模型变化 | 17.28% |

最大百分比出现在中等输入长度。很短的输入虽然 compute dilution 较小，
但也会改变不同 prefix 的 loading-stall 价值；长输入增加 hot compute，
并隐藏更多 H2D，所以 end-to-end TTFT 的相对收益下降。

固定原始输入 256 时：

| Pool | K=8 | K=16 | K=32 | K=64 |
|---:|---:|---:|---:|---:|
| 38.5 GiB | 20.13% | 19.80% | 19.15% | 19.18% |
| 39 GiB | 19.12% | 18.85% | 20.60% | **20.87%** |

更小的 pool 不一定带来更大 gap。空间过紧时，两个策略都会被迫释放高价值
状态，并且 model 5 的可用 KV 也进一步缩小。

### 6.3 K 与目标 reuse period

200 请求结果：

| 大模型目标 period | 较小/匹配 K | 提升 | 更大 K |
|---:|---:|---:|---:|
| 8 | K=8 | **17.11%** | K=32：13.56% |
| 16 | K=16 | **9.24%** | K=32：9.07% |
| 24 | K=24 | **7.71%** | K=32：7.70% |
| 32 | K=32 | 4.89% | K=64：4.89% |

K 需要覆盖有用的 reuse horizon，但并非越大越好。更多未来请求经过 discount
后仍会改变 MCKP damage ranking；reuse period 较短时，接近 period 的 K
优于 K=32。

### 6.4 放宽 context 与增大显存

额外参数：

```text
--input-limit-policy context
--stall-table-input-policy linear
--trace evaluation/traces/target/all8-cyclic-fixed4096.trace
--mckp-lookahead-k 64
```

| Pool | 实际 effective input | Mean TTFT | 提升 | Exposed-load 提升 |
|---:|---|---:|---:|---:|
| 48 GiB | 2048、4096、10463、16384 | 1565.4 → 1469.8 ms | 6.11% | 74.24% |
| 64 GiB | 2048、4096、16384 | 1881.0 → 1833.5 ms | 2.52% | 82.79% |

64 GiB 的 TTFT 更高并不矛盾：更大的 pool feasibility cap 允许更长输入，
hot compute 因此增加。这两组 trace 的 exposed-load 相对降幅很大，但
exposed load 占总 TTFT 的比例较小；更大 pool 也减少了 replacement 压力。
所以放宽输入适合研究 compute/H2D overlap，但不会产生最大的 end-to-end
TTFT gap。

## 7. 最大提升档位

500 请求复验配置：

```text
trace        = all8-cyclic-fixed256.trace
pool         = 39 GiB/GPU
K            = 64
input        = raw 256，input-scale=4
input limit  = safe
table policy = clamp
```

运行命令：

```bash
# LRU
/home/sdu/.conda/envs/sllm-worker/bin/python \
  evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
  $COMMON \
  --trace evaluation/traces/target/all8-cyclic-fixed256.trace \
  --pool-gib 39 --max-requests 500 \
  --replacement-policy lru --routing-policy cache-bytes \
  --output evaluation/tensor_simulator/results/target-trace-max-lru-r500.json

# Prefix-MCKP
/home/sdu/.conda/envs/sllm-worker/bin/python \
  evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
  $COMMON \
  --trace evaluation/traces/target/all8-cyclic-fixed256.trace \
  --pool-gib 39 --max-requests 500 \
  --replacement-policy mckp-prefix \
  --routing-policy mckp-transition \
  --mckp-prediction-mode lookahead \
  --mckp-lookahead-k 64 \
  --mckp-lookahead-discount 0.9 \
  --mckp-transition-weight 0.1 \
  --output evaluation/tensor_simulator/results/target-trace-max-mckp-r500.json
```

结果：

| 指标 | LRU | Prefix-MCKP | 提升 |
|---|---:|---:|---:|
| Critical mean | 617.47 ms | **488.14 ms** | **20.95%** |
| Critical P50 | 549.26 ms | **376.28 ms** | **31.49%** |
| Critical P90 | 1163.04 ms | **1072.96 ms** | 7.74% |
| Critical P95 | 1657.37 ms | **1151.08 ms** | **30.55%** |
| Critical P99 | 1657.37 ms | **1498.07 ms** | 9.61% |
| Exposed-load mean | 471.10 ms | **341.76 ms** | **27.45%** |
| PBP relocation mean | **7.15 ms** | 7.59 ms | +0.44 ms |
| H2D bytes | 7184.19 GB | **5554.03 GB** | -22.69% |
| Evicted bytes | 7100.76 GB | **5470.77 GB** | -22.96% |

CDF：

```text
evaluation/tensor_simulator/results/target-trace-max-ttft-cdf.svg
```

这个最大档位没有使用线性外推，仍属于原 safe-limit/clamped-profile 实验范围。

## 8. 五个可复现的提升档位

下表都是实际 simulator 输出，不是插值出来的目标值：

| 档位 | Trace/config | 请求数 | Mean TTFT 提升 | 尾延迟说明 |
|---|---|---:|---:|---|
| 最大 | cyclic fixed256、pool39、K64 | 500 | **20.95%** | P95 提升 30.55% |
| 高 | all8-cyclic、pool42、K32 | 200 | **15.41%** | P95 提升 22.69% |
| 中 | large-period16、pool42、K16 | 200 | **9.24%** | P95 回退 4.88% |
| 低 | large-hot-iid、pool42、K32 | 200 | **6.10%** | P95 提升 0.12% |
| 小 | small-hot-iid、pool42、K32 | 200 | **4.13%** | P95 提升 4.68% |

保留 period-16 这一档是为了明确展示：mean gap 档位不一定也是 tail
improvement 档位。如果实验要求主要尾延迟也改善，应使用“最大、高、低、
小”四档，而不是“中”档。

作为对照，`mckp_cache.md` 中原 ServeGen first-1000-source-row trace 的
mean TTFT 提升为 3.99%。

## 9. Segment、Page 与 Page-Compact 的实现区别

### 9.1 名称映射

模拟器 CLI 与代码类的对应关系：

| 文中简称 | CLI `--memory-layout` | 实现类 |
|---|---|---|
| Segment | `segment` | `MockAllocationSegmentMemory` |
| Page | `tensor-page` | `TensorPageMemory` |
| Page-Compact | `compact-page` | `CompactPageMemory` |
| 旧 Segment，仅作诊断 | `legacy-segment` | `SegmentMemory` |

注意：本文的 Segment 是 mock_allocation 风格的 request-level PBP，不是
逐 TensorGroup first-fit 的 `legacy-segment`。

### 9.2 核心数据结构与分配粒度

| 维度 | Segment | Page | Page-Compact |
|---|---|---|---|
| 物理表示 | 有地址的连续 `Extent` 列表 | resident TensorGroup 集合 | 每模型 resident compact-page 集合 |
| 分配对象 | 一个连续 TensorGroup segment | TensorGroup 独占若干页 | 模型 compact address space 中的页 |
| 页能否跨 tensor/group 边界 | 不适用；segment 按 group 连续分配 | 不能；每个 group 独立向上取整 | 可以；相邻 tensor/group 可共享边界页 |
| 权重与 KV | 同一个 extent pool，KV 显式 block | KV 只减少 weight capacity | KV 只减少 weight capacity |
| 地址连续要求 | 每个 group 必须找到连续 extent | 不要求物理连续 extent | 不要求物理连续 extent |

### 9.3 加载和复用

**Segment**

- 请求开始时释放上一个请求的 KV block；
- `plan_request()` 一次收集当前模型缺失的所有 TensorGroup；
- 先做 replacement，再用 request-level partitioned bin packing/PBP
  合并空洞、移动仍驻留的 segment，并为全部缺失 group 预留连续空间；
- 加载一个缺失 group 时，H2D 等于该 group 的逻辑字节数；
- 同一 group 已有 extent 时直接命中，不产生 H2D。

**Page**

- 每个 TensorGroup 单独计算
  `ceil(group_bytes / page_size)`；
- 只要 group 的 resident bit 存在，就认为整个 group 命中；
- 缺失时为整个 group map 页并传输 group 的逻辑字节；
- 不同 group 不能共享向上取整产生的最后一页。

**Page-Compact**

- 先把每个模型的所有 tensor 按 `compact_offset` 放入一个连续逻辑地址空间；
- tensor 和 TensorGroup 映射到覆盖它们的 compact page；
- 请求一个 group 时只加载缺失页，已有页可以部分复用；
- H2D 按缺失页中属于模型的有效逻辑字节计算；
- 一个边界页可能同时覆盖相邻 tensor，因此移除一页可能同时破坏多个 tensor。

### 9.4 Replacement 粒度

| 后端 | 普通 LRU victim | 释放量 |
|---|---|---|
| Segment | TensorGroup segment | group 精确逻辑字节 |
| Page | 完整 TensorGroup page allocation | `ceil(group/page) * page_size` |
| Page-Compact | 单个 compact page | 一个 page |

Page-Compact 的 `_page_value()` 会检查删除该页会破坏哪些当前完整 tensor，
并用这些 tensor 的最低 value 作为该页的近似价值；Page 和 Segment 则直接
以 group 为缓存对象。

当前 `mckp-prefix` 已支持三个正式后端。Page 仍以完整 TensorGroup prefix
为状态，但 MCKP 的释放空间按逐 group 页对齐后的物理字节计算；
Page-Compact 将每个 group prefix 映射成 compact address space 中连续的
物理页 prefix，释放后缀时保留仍被前缀使用的共享边界页。因此三个后端
使用相同的 stall curve 和 MCKP covering solver，但使用各自的
`prefix_physical_bytes()` 和 commit 操作。

Page-Compact 当前仍使用 group-prefix Offline stall table。释放空间和
实际 missing-page 加载由模拟器精确结算，但 MCKP damage 没有建模共享
边界页带来的部分后缀复用，因此其预测是近似值。

### 9.5 碎片与额外代价

**Segment**

- group 按逻辑大小精确分配，因此没有 page rounding internal fragmentation；
- 会产生 external fragmentation：总空闲空间足够，但最大连续 extent 不够；
- PBP/compaction 会移动仍驻留字节；
- 移动代价为
  `moved_bytes / gpu_copy_bytes_per_ms + compaction_fixed_ms`；
- profile-driven 模式把这部分作为 `pbp_relocation_ms` 完整加入
  exposed load 和 critical path。

**Page**

- 没有 external fragmentation，空闲页总数足够即可；
- 每个 TensorGroup 独立向上取整，会产生最大的 internal fragmentation；
- 没有 relocation/compaction；
- map/unmap 调用以完整 group allocation 为单位。

**Page-Compact**

- 没有 external fragmentation，也不需要 relocation；
- 只在每个模型 compact address space 最后一页产生传统 rounding 浪费，
  internal fragmentation 通常明显小于 Page；
- 但可能产生 `stranded_resident_bytes`：某些页仍驻留，却不足以组成任何完整
  tensor；
- 连续缺失页合并为一次 map run，连续被淘汰页合并为一次 unmap run。

### 9.6 适用场景

- **Segment**：最接近当前 mock_allocation 真实路径，支持 request-level KV、
  PBP relocation 和 Prefix-MCKP，是本文策略实验的正式后端；代价是必须
  模拟 external fragmentation 与数据移动。
- **Page**：适合作为“TensorGroup 独占 VMM pages”的简单基线，行为清晰，
  支持严格 Prefix-MCKP，但每组单独 rounding 可能明显夸大物理占用。
- **Page-Compact**：适合研究 compact VA/page sharing 和更细粒度的部分复用，
  通常空间效率优于 Page；但 page-level eviction 可能留下 stranded bytes，
  Prefix-MCKP 的 damage 暂时仍是 group-prefix 近似。

因此不能只比较三者的 TTFT 数字而忽略语义：Segment 同时模拟了显式 KV 和
relocation；Page/Page-Compact 只把 KV 作为 capacity reservation，且没有
数据移动代价。若要做正式后端 A/B，应固定 replacement policy、routing、
page size、trace 和 KV 参数，并同时报告：

```text
critical path / exposed load
H2D bytes
physical and logical resident bytes
internal / external fragmentation
stranded resident bytes
map / unmap calls
compaction moved bytes / PBP relocation
```

## 10. Prefix-MCKP 三后端六档对比

### 10.1 匹配条件与页大小

三后端固定相同 trace、请求数、输入缩放、GPU 数、pool、64-MiB
TensorGroup、Offline stall table、K 和 transition routing。页大小保持原
后端配置：

- Segment 保持原始 TensorGroup segment 语义；命令保留原来的 8 MiB，
  但该参数不参与 Segment 权重分配；
- 常规五档使用 Page=8 MiB、Page-Compact=64 MiB；
- 8-MiB Page 在最大档的 39-GiB pool 下不可行；最大模型逐 group rounding
  后的权重 footprint 为 42,580,574,208 bytes，大于 39 GiB；
- 因此只有最大档的 Page 和 Page-Compact 改用 2 MiB，Segment 仍为原始
  8-MiB CLI/TensorGroup 语义。

这里比较的是三种后端上的同一套 Prefix-MCKP，不是各后端的 LRU 对照。

### 10.2 最大档完整命令

```bash
for LAYOUT in segment tensor-page compact-page; do
  PAGE_SIZE_MIB=2
  if [ "${LAYOUT}" = segment ]; then PAGE_SIZE_MIB=8; fi
  /home/sdu/.conda/envs/sllm-worker/bin/python \
    evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
    --tensor-layout evaluation/tensor_simulator/results/tensor-layout.json \
    --profile tools/layerpipe/results/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
    --config configs/servegen_8_models_layerpipe_l40_pool42.json \
    --trace evaluation/traces/target/all8-cyclic-fixed256.trace \
    --pool-gib 39 --max-requests 500 --gpus 2 \
    --trace-mode model-switches --input-scale 4 \
    --memory-layout "${LAYOUT}" --page-size-mib "${PAGE_SIZE_MIB}" \
    --h2d-gbps 24.56 --tensor-group-min-mib 64 \
    --policy-suite minimal-only \
    --mckp-stall-table-input \
      evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json \
    --replacement-policy mckp-prefix \
    --routing-policy mckp-transition \
    --mckp-prediction-mode lookahead \
    --mckp-lookahead-k 64 --mckp-lookahead-discount 0.9 \
    --mckp-transition-weight 0.1 \
    --output \
      "evaluation/tensor_simulator/results/backend-comparison-max-${LAYOUT}.json"
done

python3 tools/layerpipe/plot_prefix_mckp_ttft_cdf.py \
  --series evaluation/tensor_simulator/results/backend-comparison-max-segment.json Segment \
  --series evaluation/tensor_simulator/results/backend-comparison-max-tensor-page.json Page \
  --series evaluation/tensor_simulator/results/backend-comparison-max-compact-page.json \
    Page-Compact \
  --title "Prefix-MCKP backend comparison: max" \
  --output evaluation/tensor_simulator/results/backend-comparison-max-ttft-cdf.svg
```

### 10.3 高档完整命令

```bash
for LAYOUT in segment tensor-page compact-page; do
  case "${LAYOUT}" in
    compact-page) PAGE_SIZE_MIB=64 ;;
    *) PAGE_SIZE_MIB=8 ;;
  esac
  /home/sdu/.conda/envs/sllm-worker/bin/python \
    evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
    --tensor-layout evaluation/tensor_simulator/results/tensor-layout.json \
    --profile tools/layerpipe/results/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
    --config configs/servegen_8_models_layerpipe_l40_pool42.json \
    --trace evaluation/traces/target/all8-cyclic.trace \
    --pool-gib 42 --max-requests 200 --gpus 2 \
    --trace-mode model-switches --input-scale 4 \
    --memory-layout "${LAYOUT}" --page-size-mib "${PAGE_SIZE_MIB}" \
    --h2d-gbps 24.56 --tensor-group-min-mib 64 \
    --policy-suite minimal-only \
    --mckp-stall-table-input \
      evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json \
    --replacement-policy mckp-prefix \
    --routing-policy mckp-transition \
    --mckp-prediction-mode lookahead \
    --mckp-lookahead-k 32 --mckp-lookahead-discount 0.9 \
    --mckp-transition-weight 0.1 \
    --output \
      "evaluation/tensor_simulator/results/backend-comparison-high-${LAYOUT}.json"
done

python3 tools/layerpipe/plot_prefix_mckp_ttft_cdf.py \
  --series evaluation/tensor_simulator/results/backend-comparison-high-segment.json Segment \
  --series evaluation/tensor_simulator/results/backend-comparison-high-tensor-page.json Page \
  --series evaluation/tensor_simulator/results/backend-comparison-high-compact-page.json \
    Page-Compact \
  --title "Prefix-MCKP backend comparison: high" \
  --output evaluation/tensor_simulator/results/backend-comparison-high-ttft-cdf.svg
```

### 10.4 中档完整命令

```bash
for LAYOUT in segment tensor-page compact-page; do
  case "${LAYOUT}" in
    compact-page) PAGE_SIZE_MIB=64 ;;
    *) PAGE_SIZE_MIB=8 ;;
  esac
  /home/sdu/.conda/envs/sllm-worker/bin/python \
    evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
    --tensor-layout evaluation/tensor_simulator/results/tensor-layout.json \
    --profile tools/layerpipe/results/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
    --config configs/servegen_8_models_layerpipe_l40_pool42.json \
    --trace evaluation/traces/target/large-period16.trace \
    --pool-gib 42 --max-requests 200 --gpus 2 \
    --trace-mode model-switches --input-scale 4 \
    --memory-layout "${LAYOUT}" --page-size-mib "${PAGE_SIZE_MIB}" \
    --h2d-gbps 24.56 --tensor-group-min-mib 64 \
    --policy-suite minimal-only \
    --mckp-stall-table-input \
      evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json \
    --replacement-policy mckp-prefix \
    --routing-policy mckp-transition \
    --mckp-prediction-mode lookahead \
    --mckp-lookahead-k 16 --mckp-lookahead-discount 0.9 \
    --mckp-transition-weight 0.1 \
    --output \
      "evaluation/tensor_simulator/results/backend-comparison-medium-${LAYOUT}.json"
done

python3 tools/layerpipe/plot_prefix_mckp_ttft_cdf.py \
  --series evaluation/tensor_simulator/results/backend-comparison-medium-segment.json Segment \
  --series evaluation/tensor_simulator/results/backend-comparison-medium-tensor-page.json Page \
  --series evaluation/tensor_simulator/results/backend-comparison-medium-compact-page.json \
    Page-Compact \
  --title "Prefix-MCKP backend comparison: medium" \
  --output evaluation/tensor_simulator/results/backend-comparison-medium-ttft-cdf.svg
```

### 10.5 低档完整命令

```bash
for LAYOUT in segment tensor-page compact-page; do
  case "${LAYOUT}" in
    compact-page) PAGE_SIZE_MIB=64 ;;
    *) PAGE_SIZE_MIB=8 ;;
  esac
  /home/sdu/.conda/envs/sllm-worker/bin/python \
    evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
    --tensor-layout evaluation/tensor_simulator/results/tensor-layout.json \
    --profile tools/layerpipe/results/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
    --config configs/servegen_8_models_layerpipe_l40_pool42.json \
    --trace evaluation/traces/target/large-hot-iid.trace \
    --pool-gib 42 --max-requests 200 --gpus 2 \
    --trace-mode model-switches --input-scale 4 \
    --memory-layout "${LAYOUT}" --page-size-mib "${PAGE_SIZE_MIB}" \
    --h2d-gbps 24.56 --tensor-group-min-mib 64 \
    --policy-suite minimal-only \
    --mckp-stall-table-input \
      evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json \
    --replacement-policy mckp-prefix \
    --routing-policy mckp-transition \
    --mckp-prediction-mode lookahead \
    --mckp-lookahead-k 32 --mckp-lookahead-discount 0.9 \
    --mckp-transition-weight 0.1 \
    --output \
      "evaluation/tensor_simulator/results/backend-comparison-low-${LAYOUT}.json"
done

python3 tools/layerpipe/plot_prefix_mckp_ttft_cdf.py \
  --series evaluation/tensor_simulator/results/backend-comparison-low-segment.json Segment \
  --series evaluation/tensor_simulator/results/backend-comparison-low-tensor-page.json Page \
  --series evaluation/tensor_simulator/results/backend-comparison-low-compact-page.json \
    Page-Compact \
  --title "Prefix-MCKP backend comparison: low" \
  --output evaluation/tensor_simulator/results/backend-comparison-low-ttft-cdf.svg
```

### 10.6 小档完整命令

```bash
for LAYOUT in segment tensor-page compact-page; do
  case "${LAYOUT}" in
    compact-page) PAGE_SIZE_MIB=64 ;;
    *) PAGE_SIZE_MIB=8 ;;
  esac
  /home/sdu/.conda/envs/sllm-worker/bin/python \
    evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
    --tensor-layout evaluation/tensor_simulator/results/tensor-layout.json \
    --profile tools/layerpipe/results/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
    --config configs/servegen_8_models_layerpipe_l40_pool42.json \
    --trace evaluation/traces/target/small-hot-iid.trace \
    --pool-gib 42 --max-requests 200 --gpus 2 \
    --trace-mode model-switches --input-scale 4 \
    --memory-layout "${LAYOUT}" --page-size-mib "${PAGE_SIZE_MIB}" \
    --h2d-gbps 24.56 --tensor-group-min-mib 64 \
    --policy-suite minimal-only \
    --mckp-stall-table-input \
      evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json \
    --replacement-policy mckp-prefix \
    --routing-policy mckp-transition \
    --mckp-prediction-mode lookahead \
    --mckp-lookahead-k 32 --mckp-lookahead-discount 0.9 \
    --mckp-transition-weight 0.1 \
    --output \
      "evaluation/tensor_simulator/results/backend-comparison-small-${LAYOUT}.json"
done

python3 tools/layerpipe/plot_prefix_mckp_ttft_cdf.py \
  --series evaluation/tensor_simulator/results/backend-comparison-small-segment.json Segment \
  --series evaluation/tensor_simulator/results/backend-comparison-small-tensor-page.json Page \
  --series evaluation/tensor_simulator/results/backend-comparison-small-compact-page.json \
    Page-Compact \
  --title "Prefix-MCKP backend comparison: small" \
  --output evaluation/tensor_simulator/results/backend-comparison-small-ttft-cdf.svg
```

### 10.7 原默认 trace 完整命令

```bash
for LAYOUT in segment tensor-page compact-page; do
  case "${LAYOUT}" in
    compact-page) PAGE_SIZE_MIB=64 ;;
    *) PAGE_SIZE_MIB=8 ;;
  esac
  /home/sdu/.conda/envs/sllm-worker/bin/python \
    evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
    --tensor-layout evaluation/tensor_simulator/results/tensor-layout.json \
    --profile tools/layerpipe/results/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
    --config configs/servegen_8_models_layerpipe_l40_pool42.json \
    --trace evaluation/traces/servegen_tangram.trace \
    --pool-gib 42 --max-requests 1000 --gpus 2 \
    --trace-mode model-switches --input-scale 4 \
    --memory-layout "${LAYOUT}" --page-size-mib "${PAGE_SIZE_MIB}" \
    --h2d-gbps 24.56 --tensor-group-min-mib 64 \
    --policy-suite minimal-only \
    --mckp-stall-table-input \
      evaluation/tensor_simulator/results/offline-prefix-stall-runtime-group64.json \
    --replacement-policy mckp-prefix \
    --routing-policy mckp-transition \
    --mckp-prediction-mode lookahead \
    --mckp-lookahead-k 32 --mckp-lookahead-discount 0.9 \
    --mckp-transition-weight 0.1 \
    --output \
      "evaluation/tensor_simulator/results/backend-comparison-default-${LAYOUT}.json"
done

python3 tools/layerpipe/plot_prefix_mckp_ttft_cdf.py \
  --series evaluation/tensor_simulator/results/backend-comparison-default-segment.json \
    Segment \
  --series evaluation/tensor_simulator/results/backend-comparison-default-tensor-page.json \
    Page \
  --series evaluation/tensor_simulator/results/backend-comparison-default-compact-page.json \
    Page-Compact \
  --title "Prefix-MCKP backend comparison: default ServeGen" \
  --output evaluation/tensor_simulator/results/backend-comparison-default-ttft-cdf.svg
```

`--max-requests 1000` 作用于原始 source rows；经过 model-switch 过滤后，
三个后端都实际执行 704 个请求。

### 10.8 实测结果

Page 和 Page-Compact 的 TTFT 使用逐物理页 VMM 计费：

```text
map_ms   = mapped_pages   * (map_fixed_ms + map_per_page_ms)
unmap_ms = unmapped_pages * (unmap_fixed_ms + unmap_per_page_ms)
TTFT     = hot compute + Offline exposed stall
           + map_ms + unmap_ms + Segment PBP relocation
```

默认每页 map/unmap 均为 `0.010 + 0.002 = 0.012 ms`。物理页池启动时
预分配产生的 `cuMemCreate` 不计入每请求 TTFT。

| 档位 | 后端 | Mean TTFT (ms) | P95 TTFT (ms) | Mean exposed (ms) | Mean VMM (ms) | H2D (GB) |
|---|---|---:|---:|---:|---:|---:|
| 最大 | Segment | **488.14** | **1151.08** | **341.76** | 0.00 | **5554.03** |
| 最大 | Page | 609.28 | 1408.31 | 462.91 | 129.05 | 5630.60 |
| 最大 | Page-Compact | 610.62 | 1405.38 | 464.24 | 128.71 | 5664.64 |
| 高 | Segment | 623.08 | 1285.17 | 256.51 | 0.00 | **2162.88** |
| 高 | Page | 656.52 | 1310.72 | 289.95 | 32.69 | 2265.79 |
| 高 | Page-Compact | **617.72** | **1219.19** | **251.15** | 3.85 | 2190.98 |
| 中 | Segment | **474.94** | **1389.84** | **176.70** | 0.00 | **1302.69** |
| 中 | Page | 497.27 | 1465.73 | 199.03 | 19.27 | 1354.16 |
| 中 | Page-Compact | 484.77 | 1466.40 | 186.53 | 2.33 | 1342.68 |
| 低 | Segment | 637.37 | **1669.95** | 267.73 | 0.00 | 1921.14 |
| 低 | Page | 643.16 | 1774.42 | 273.51 | 26.91 | 1879.05 |
| 低 | Page-Compact | **609.47** | 1682.37 | **239.82** | 3.18 | **1816.15** |
| 小 | Segment | **290.69** | 916.92 | **43.32** | 0.00 | 330.69 |
| 小 | Page | 297.70 | 911.96 | 50.33 | 4.85 | 374.41 |
| 小 | Page-Compact | 291.03 | **902.04** | 43.66 | 0.51 | **327.66** |
| 默认 | Segment | **422.84** | **1384.29** | **67.23** | 0.00 | **2205.84** |
| 默认 | Page | 438.15 | 1564.01 | 82.54 | 9.52 | 2324.19 |
| 默认 | Page-Compact | 432.19 | 1650.55 | 76.57 | 1.16 | 2317.24 |

常规五档中，8-MiB Page 的平均 internal fragmentation 为约
1.02--1.23 GiB，64-MiB Page-Compact 为约 0.02--0.03 GiB。最大档改用
2 MiB 后，两者分别约为 0.315 GiB 和 0.001 GiB。Segment 没有 page
rounding，但最大档平均有 7.59 ms PBP relocation；两个页后端没有
relocation。

不能把 Page-Compact 的低 fragmentation 直接等同于最低 TTFT：不同物理
prefix 大小会改变 MCKP 组合和 transition routing，Page-Compact 又暂时使用
group-prefix damage 近似。加入逐页 VMM TTFT 后，2-MiB 最大档的两个页后端
平均增加约 129 ms，Segment 明显最好。常规页大小下，Page-Compact 仅在高、
低档取得最低 mean；Segment 在最大、中、小、默认档最低。8-MiB Page 在
六档均未取得最低 mean。

六张本地 CDF：

- [最大档 CDF](../evaluation/tensor_simulator/results/backend-comparison-max-ttft-cdf.svg)
- [高档 CDF](../evaluation/tensor_simulator/results/backend-comparison-high-ttft-cdf.svg)
- [中档 CDF](../evaluation/tensor_simulator/results/backend-comparison-medium-ttft-cdf.svg)
- [低档 CDF](../evaluation/tensor_simulator/results/backend-comparison-low-ttft-cdf.svg)
- [小档 CDF](../evaluation/tensor_simulator/results/backend-comparison-small-ttft-cdf.svg)
- [默认 trace CDF](../evaluation/tensor_simulator/results/backend-comparison-default-ttft-cdf.svg)

## 11. 验证命令

```bash
python3 -m py_compile \
  evaluation/traces/generate_target_trace.py \
  evaluation/tensor_simulator/tensor_sim_mckp.py \
  evaluation/tensor_simulator/tensor_sim_memory_common.py \
  evaluation/tensor_simulator/tensor_sim_page_backends.py \
  tools/layerpipe/layerweave_pipeline_cache_sim.py \
  evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
  tools/layerpipe/plot_prefix_mckp_ttft_cdf.py

python3 evaluation/tensor_simulator/test_tensor_sim_mckp.py
python3 evaluation/tensor_simulator/test_layerweave_tensor_pipeline_sim.py

git diff --check -- \
  evaluation/traces/generate_target_trace.py \
  evaluation/tensor_simulator/tensor_sim_mckp.py \
  tools/layerpipe/layerweave_pipeline_cache_sim.py \
  evaluation/tensor_simulator/layerweave_tensor_pipeline_sim.py \
  evaluation/tensor_simulator/test_tensor_sim_mckp.py \
  doc/target_trace.md
```

CDF：

```bash
python3 tools/layerpipe/plot_prefix_mckp_ttft_cdf.py \
  --baseline evaluation/tensor_simulator/results/target-trace-max-lru-r500.json \
  --mckp evaluation/tensor_simulator/results/target-trace-max-mckp-r500.json \
  --output evaluation/tensor_simulator/results/target-trace-max-ttft-cdf.svg \
  --baseline-label "TensorGroup LRU + cache-bytes" \
  --mckp-label "Prefix-MCKP K=64 + transition"
```

当前验证：

```text
Prefix-stall/MCKP tests = 9 passed
Tensor simulator tests  = 25 passed
py_compile              = passed
git diff --check        = passed
```
