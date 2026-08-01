# LayerWeave 模拟器

LayerWeave 模拟器是一个基于真实请求 trace、模型参数布局和 M4
性能 profile 的 CPU 离线模拟器。入口为：

```text
tools/layerpipe/layerweave_pipeline_cache_sim.py
```

它主要回答：

> 给定请求序列、GPU 数量和显存容量，不同路由与参数缓存策略会产生多少
> 关键路径时延、暴露加载停顿和在线控制开销？

## 1. Trace 与系统配置

模拟器可以读取 ServeGen 请求 trace，并支持两种回放模式：

- `all`：模拟 trace 中的所有请求；
- `model-switches`：每段连续相同模型的请求只保留第一个请求。

可配置的系统和 workload 参数包括：

- 最大请求数；
- 输入 token 缩放比例；
- 输出 token 覆盖值；
- GPU 数量；
- 每张 GPU 的物理页池容量；
- KV block 的 token 数；
- 模型访问热度的指数衰减系数；
- 随机种子。

`--request-cap-pool-pages` 可以将“用于裁剪过长请求的容量”与实际模拟的
GPU pool 容量分开。进行显存容量 sweep 时固定该参数，可以确保不同容量
实验使用完全相同的请求集合。

## 2. Profile-driven Pipeline Stall Estimation

模拟器读取 M4 profile，根据以下信息预测一次请求的性能：

- 每层参数页数量及当前 residency；
- 各层缺失页的 H2D 加载时间；
- 每层 Prefill compute 时间；
- 输入 token 数；
- layer loading 与 compute 的 overlap；
- 前序 stall 对后续流水线时间线的影响。

主要预测结果包括：

- hot compute time；
- 总 H2D time；
- 无法被计算隐藏的 exposed loading stall；
- 请求 critical path。

因此，模拟器不是简单地计算“缺失字节数除以带宽”。相同数量的缺失页，
如果位于不同 layer，或者请求具有不同输入长度，可能产生不同的 exposed
stall。

## 3. 支持的缓存与流水线策略

完整的 `--policy-suite all` 支持以下主要策略和对照组。

### 3.1 五级消融

1. `baseline_load_plus_compute`
   - 权重完全 cold；
   - H2D 与 compute 串行执行。
2. `pipe_only`
   - 权重完全 cold；
   - 按 layer 模拟 H2D/compute pipeline。
3. `reuse_only`
   - 使用 Minimal 的缓存复用结果；
   - 缺失页 H2D 与 compute 串行执行。
4. `minimal`
   - 支持参数页复用；
   - 使用逐层 H2D/compute pipeline；
   - 使用模型访问频率衡量缓存价值；
   - 使用 page LRU 处理相同价值的页面。
5. `layerweave`
   - 使用 configuration MCKP 选择 protected prefix；
   - 保留未被实际回收的 soft pages；
   - 采用 deferred PageGreedy reclamation；
   - 使用逐层 H2D/compute pipeline。

### 3.2 其他策略

完整策略集合还支持：

- Pipeline-aware PageGreedy；
- ordinary configuration MCKP；
- MCKP + PageGreedy；
- configuration-transition planning；
- residency-transition planning；
- Exact-residency；
- Aegaeon 风格的下一模型预取 baseline；
- 不同 Lookahead K 的策略 sweep；
- joint routing + cache placement。

其中 Exact-residency 会反复评估未来请求和精确 residency transition，
属于离线 oracle，用于提供参考上界，不是可直接部署的在线控制器。

## 4. Prefix Configuration

默认的 protected-prefix 候选配置为：

```text
0 / 2 / 4 / 8 / 16 / 32 / full
```

也可以通过 `--configuration-step N` 使用更细的配置。例如
`--configuration-step 1` 会逐层生成 protected-prefix 候选。

这里的 configuration 表示模型参数的 **protected residency floor**，
不是物理驻留的硬上限。模拟器区分三类权重页：

- `active pages`：当前正在执行的模型页面，不允许回收；
- `protected pages`：MCKP 选择的最低保护集合，正常情况下不回收；
- `soft pages`：仍然真实驻留、但位于 protected prefix 之外的机会式页面。

因此，实际驻留关系为：

```text
actual residency = protected pages + remaining soft pages
```

只要 soft page 尚未因容量压力被回收，后续请求仍然可以直接复用它。

## 5. MCKP 与延迟 PageGreedy 回收

模拟器可以根据各模型的在线访问热度和不同 prefix 对 pipeline stall 的
预测收益，为每个模型构造多个 configuration 候选，然后在 GPU page
budget 下求解 Multiple-Choice Knapsack Problem（MCKP）。

MCKP 决定：

- 每个模型至少保护多少 prefix；
- 哪些页面从 protected 状态变为 soft 状态；
- 当前配置的总保护价值。

配置变化不会立即强制释放全部 soft pages。只有模型加载或 KV Cache
需求产生真实容量缺口时，PageGreedy 才执行物理回收：

- victim 只来自非活动模型的 soft pages；
- protected pages 和 active pages 不参与回收；
- 只回收满足当前容量缺口所需的页数；
- 根据模型 demand 和页面对 pipeline critical path 的边际价值选择
  低价值 victim。

这实现了配置级全局规划与页面级精确回收的解耦。

## 6. Lookahead

模拟器通过 `--lookahead-k K...` 支持一个或多个 Lookahead K 的 sweep。

完整 suite 中包含 configuration-MCKP oracle lookahead，它可以分析更完整
的未来状态，适合作为离线参考。

隔离的 MCKP+PageGreedy Lookahead 则面向低开销在线实现：

- 每张 GPU 只读取队列中后续 K 个 `(model_id, input_tokens)`；
- 使用这些请求修正模型 demand 和典型输入窗口；
- 队列访问复杂度为 O(K)；
- 不等待未来请求到齐；
- 不为每个候选 stage 重跑完整的未来状态 PSE。

## 7. 联合 GPU 路由与缓存规划

启用 `--joint-routing` 后，模拟器会联合决定：

- 当前请求应该路由到哪张 GPU；
- 目标 GPU 上每个模型应保护多少 prefix；
- 哪些 soft pages 可以在需要时被回收；
- 当前请求的 pipeline stall；
- cache configuration 变化对未来请求造成的 transition cost。

联合目标可以概括为：

```text
当前请求的排队和 pipeline stall
+ transition_weight × bounded cache-transition cost
```

相关参数包括：

- `--transition-weight`；
- `--transition-credit-cap-ms`；
- `--transition-beam-width`。

## 8. 隔离的 MCKP+PageGreedy 模式

使用：

```bash
--policy-suite mckp-page-greedy
```

时，模拟器只运行：

- Minimal；
- MCKP+PageGreedy demand 策略；
- MCKP+PageGreedy 的不同 Lookahead K；
- 可选的 joint routing + MCKP+PageGreedy。

该模式不会构造：

- Exact-residency；
- configuration-transition；
- ordinary MCKP；
- standalone PageGreedy。

因此，该模式适合评估可在线部署的 MCKP+PageGreedy 策略，并避免昂贵的
离线 oracle 影响模拟器 wall time。

## 9. Baseline 与公平对照

模拟器支持 Aegaeon 风格的预取 baseline。为了避免路由差异混淆缓存或
流水线收益，对比结果可以复用相同请求和相同 routing。

五级消融同样使用相同请求，并可让 LayerWeave replay Minimal routing，
从而分别观察以下因素的贡献：

- 计算—加载 pipeline；
- 参数复用；
- Minimal 缓存策略；
- MCKP protected floor；
- deferred PageGreedy reclamation。

## 10. 指标与输出

模拟器将结果写入 JSON，主要包括：

- critical-path latency；
- exposed loading stall；
- hot compute time；
- H2D time；
- missing pages 及逐 stage 缺页分布；
- cache residency、protected pages 和 soft pages；
- reclaimed page 数量；
- 每个请求的 GPU routing；
- 每个模型的 configuration choice；
- cache-transition cost；
- planner time；
- queue-read time；
- controller time；
- 模拟器总 wall time。

关键路径和控制器开销支持以下统计：

```text
mean / P50 / P90 / P95 / P99 / max
```

`--inspect-request-id` 可以指定一个请求，输出其详细五级时间线。未指定时，
模拟器会选择一个具有代表性的正收益请求。

需要注意：

- `simulator_wall_time_ms` 包含 profile 构造和多策略 replay，不能当作在线
  controller latency；
- 在线开销应使用逐请求的 `planner_time_ms`、`queue_read_time_ms` 和
  `controller_time_ms`；
- 平均 critical-path 改善不代表 P95/P99 一定改善，平均值与尾时延需要
  分别报告。

## 11. 当前模拟边界

当前模拟器存在以下边界：

- 它是 CPU 离散事件/策略模拟器，不执行真实 CUDA kernel 或 H2D；
- 时延来自实测 M4 profile，但最终结果仍是预测值；
- 模拟器按 system-wide serial request order 回放，不等价于完整的并发
  vLLM scheduler；
- KV Cache 主要作为容量需求建模，不模拟 paged attention 的全部运行细节；
- Exact-residency 和完整未来状态 lookahead 是离线 oracle；
- 真正的 LayerWeave + VMM + ODKV 执行由 benchmark/runtime 路径完成，
  模拟器主要用于快速比较路由与缓存策略。

## 12. 当前推荐运行方式

当前 Joint LayerWeave 模拟配置为：

- MCKP + PageGreedy；
- joint routing/cache placement；
- configuration step = 1；
- Lookahead K = 8；
- transition weight = 0.2；
- transition credit cap = 100 ms；
- 2 GPUs；
- 672 pages/GPU；
- input scale = 4；
- 只保留 model-switch 请求。

运行命令：

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


# TensorWeave 模拟器
新的Tensor-level模拟器，当前的Layer Weave模拟器默认计算加载流水是以layer为单位的，换成以更细粒度的tensor为单位，初始只需要支持LayerWeave中的：baseline, pipe_only, reuse_only和minimal以及Aegaeon。需要做哪些工作。
补充：还需要支持以下三种不同的参数重用策略
    - Segment-based tensor memory
        通过cudamalloc申请一整块连续的空间；
        每个 tensor 对应一个连续 segment； tensor 与 cache/reuse/compute 语义完全一致； 可以直接保留或淘汰完整 tensor； 但不规则 segment 会产生外部碎片； 需要 PGP 选择保留对象并控制 compaction (参考“tools/mock_allocation/src/main.cpp”以及“vram_manager.h”下的实现)
    - Page-based tensor allocation
        通过VMM allocate/map/unmap 申请页面；
        每个 tensor 分配若干完整页面； tensor 内仍保持独立 VA 区域； 页面较大时产生 tensor 尾部内部碎片； 页面较小时，VMM allocate/map/unmap 调用数量增加； 存在 page size 在空间效率和管理开销之间的权衡。
    - Page-based compact allocation
        通过VMM allocate/map/unmap 申请页面；
        为整个模型预留连续虚拟地址空间； 参数紧凑排列； 物理页面按模型 offset 映射； 基本消除 per-tensor rounding 带来的内部碎片； 映射数量也可以通过更大的连续范围降低。
        但代价是： 物理管理、加载和 eviction 只能自然地以 page 为单位； 一个 page 可能覆盖 tensor 的一部分，或横跨 tensor 边界； tensor-level cache value 很难直接投影到 page； 回收一个 page 可能破坏一个高价值 tensor 的完整可重用性； 保留某个 page 也可能包含 pipeline 无价值的数据； 因此存储粒度和计算/缓存语义不一致。


## Segment


