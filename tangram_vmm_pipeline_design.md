# Tangram-VMM：基于页面重用的计算-加载流水线设计

## 1. 目标

Tangram-VMM 面向 Serverless LLM 模型切换场景，利用 GPU 中已经缓存的模型参数页，减少模型重新加载的数据量，并通过 layer-wise 计算-加载流水线隐藏剩余加载开销。

系统的核心目标不是最大化缓存命中率或缓存字节数，而是最小化请求在目标 GPU 上的 **暴露加载时延（exposed loading latency）**，即无法被 Prefill 计算隐藏的参数加载时间。

整体方案包含三个核心部分：

1. 基于 CUDA VMM 的模型参数页重用；
2. 基于 layer readiness 的重用-计算-加载流水线；
3. 统一使用流水线代价模型指导 GPU 调度和缓存淘汰。

---

## 2. 设计原则

### 2.1 VMM 负责物理内存管理

每个注册模型预留一段固定、连续的 GPU 虚拟地址空间。模型参数按照正常 checkpoint 布局紧密排列，tensor 和 layer 不强制按照 VMM page 对齐。

模型运行时，仅为需要驻留的参数页分配和映射物理显存：

- 已缓存页继续保持映射；
- 缺失页通过 `cuMemMap` 映射后从 CPU 内存加载；
- 被淘汰页通过 `cuMemUnmap` 释放物理显存；
- 模型虚拟地址和 tensor 指针始终稳定；
- 推理 kernel 无需感知 page-level 物理放置。

VMM page 是物理重用和淘汰单元，layer 是计算 readiness 和流水线调度单元。

### 2.2 缓存价值由流水线效果决定

缓存某一段参数的价值，不仅取决于它所属模型的访问热度和大小，还取决于它能否：

- 让前部 layer 更早进入 ready 状态；
- 使 Prefill 提前启动；
- 消除后续计算关键路径上的加载 stall；
- 为后续缺失参数提供更大的计算掩盖窗口。

因此，调度和淘汰均以预测的暴露加载时延为核心指标，而不是简单使用缓存字节数或命中率。

---

## 3. 核心数据结构

### 3.1 ModelLayout

记录模型静态布局信息：

- 模型虚拟地址空间起始地址和大小；
- tensor 在模型地址空间中的 offset、size、dtype 和 shape；
- 每个 VMM page 覆盖的地址范围；
- 每个 layer 依赖的 page 集合；
- page 到 tensor、layer 的反向映射。

### 3.2 ModelProfile

记录离线性能 Profile：

- 不同 GPU 上每个 layer 的 Prefill 计算时间；
- 计算时间关于 batch size、总输入 token、最大序列长度等特征的估算模型；
- 参数页加载带宽和固定控制开销；
- 推荐的 page copy chunk 大小；
- 模型历史输入长度分布。

### 3.3 GPUResidencyState

记录每张 GPU 的运行时状态：

- 当前 active model；
- live KV cache 占用；
- 每个模型已映射的参数页 bitmap；
- 每个 layer 当前已驻留和缺失的 page 数量；
- GPU 队列长度和预计排队时间；
- 当前可用于参数缓存的物理显存容量。

### 3.4 CacheConfiguration

缓存策略以候选配置为决策对象，而不是直接逐 page 全局搜索。

初始版本只支持 prefix 配置，例如：

- 不缓存；
- 缓存前 2 层；
- 缓存前 4 层；
- 缓存前 8 层；
- 缓存前 16 层；
- 缓存完整模型。

每个配置记录：

- 占用 page 数量；
- 对不同输入长度分布的预计暴露加载时延；
- 预计未来缓存收益。

后续可以扩展到非连续 critical-layer 配置。

---

## 4. 请求执行流程

### 4.1 模型注册

1. 为模型预留连续虚拟地址空间；
2. 按 checkpoint 顺序建立 tensor 和 layer 布局；
3. 构建 page-to-layer dependency map；
4. 建立离线 layer compute profile；
5. 初始化模型 residency bitmap，初始状态全部未映射。

### 4.2 请求到达

对于目标模型和当前 batch，调度器在每张候选 GPU 上执行以下估算：

1. 获取该 GPU 上目标模型的当前 page residency；
2. 计算每个 layer 的缺失页集合；
3. 估计当前 batch 在该 GPU 上每个 layer 的 Prefill 计算时间；
4. 生成需要释放的显存容量；
5. 调用本地缓存控制器生成 eviction plan；
6. 估计淘汰后目标模型的暴露加载时延；
7. 估计该 eviction plan 对其它模型未来启动时延造成的损失；
8. 结合排队时间计算候选 GPU 的总分。

### 4.3 GPU 选择

候选 GPU 的高层评分为：

```text
Score = QueueDelay
      + CurrentRequestLatency
      + FutureCacheDamage
```

其中：

- `QueueDelay`：当前 GPU 的预计等待时间；
- `CurrentRequestLatency`：目标模型在淘汰后的 residency 状态下，执行 live loading 的预计启动和 Prefill 时间；
- `FutureCacheDamage`：本次淘汰对其它模型未来请求造成的预计额外暴露加载时延。

选择总分最低的 GPU。

### 4.4 Live Activation

模型在目标 GPU 上激活时：

1. 保留已有的目标模型参数页；
2. 执行 eviction plan，解除其它 inactive 参数页的映射；
3. 为目标模型缺失页准备物理显存；
4. 启动参数加载 stream 和计算 stream；
5. 第一层所有依赖页 ready 后立即开始 Prefill；
6. 计算当前层时并行加载后续层缺失页；
7. 每层计算前等待对应的 layer-ready event；
8. 所有参数加载完成后恢复普通完整模型执行路径；
9. Decode 阶段不再动态修改模型参数映射。

---

## 5. 重用-计算-加载流水线

### 5.1 Layer Readiness

Layer 只有在它依赖的所有参数页都已映射并完成数据加载后才能执行。

每个 layer 维护：

- required pages；
- resident pages；
- missing pages；
- ready event。

当 missing pages 全部完成加载后，在 copy stream 上记录 ready event；compute stream 在执行该 layer 前等待对应 event。

### 5.2 加载顺序

基础版本按 layer 顺序加载缺失页。

增强版本使用 critical-frontier 策略：

- 根据当前计算进度估计每个未来 layer 的计算 deadline；
- 根据剩余缺失字节估计该 layer 的加载完成时间；
- 优先加载 slack 最小、最可能阻塞计算的 layer；
- 实际 I/O 仍以连续 page run 或较大 chunk 传输，避免过细 API 调用。

### 5.3 执行边界

初始实现只在 layer 粒度流水，不拆分 layer 内部算子。

仅第一次 cold activation Prefill 使用该路径。模型参数完整加载后：

- Decode 使用普通推理路径；
- 不再执行模型参数 map/unmap；
- 可以恢复 CUDA Graph 或其它 steady-state 优化。

---

## 6. Pipeline Cost Estimator

对于模型、GPU、batch 和当前 residency 状态，估算器输出：

- 第一层可开始计算的时间；
- 每个 layer 的参数 ready time；
- 每个 layer 的计算开始和结束时间；
- 总 Prefill 时间；
- 暴露加载时延；
- 可被计算隐藏的加载比例。

估算过程同时模拟：

- 已缓存页；
- 缺失页加载顺序；
- H2D 加载带宽；
- layer 计算时间；
- copy 与 compute 的重叠；
- 必要的固定映射和同步开销。

该估算器由调度器、缓存控制器和实验分析共同使用，保证系统内使用统一的性能目标。

---

## 7. 缓存策略

### 7.1 缓存目标

缓存策略最大化集群未来请求可减少的暴露加载时延，而不是最大化缓存命中率。

缓存价值由以下因素共同决定：

- 模型访问热度；
- 模型历史输入长度分布；
- 缓存层的位置；
- 当前模型在其它 GPU 上是否已有副本；
- 缓存配置能否形成可立即执行的前缀；
- 后续层加载是否能够被该前缀的计算隐藏。

### 7.2 模型输入长度差异

不同模型可能具有稳定但不同的输入长度分布：

- 长输入模型每层 Prefill 时间较长，少量 prefix layers 可能已经足以隐藏剩余加载；
- 短输入模型计算窗口较短，可能需要缓存更长 prefix 或完整模型；
- 因此不能为所有模型使用统一的缓存层数或缓存比例。

### 7.3 Eviction Plan

当目标 GPU 空间不足时，本地缓存控制器从 inactive weight cache 中选择需要降级或移除的缓存配置。

优先淘汰：

- 访问频率低的模型；
- 在其它 GPU 上存在等价缓存副本的模型；
- 后部且可被计算完全隐藏的 layers；
- 对未来暴露加载时延贡献较低的配置。

尽量保留：

- 热模型的有效执行前缀；
- 当前集群中唯一的低启动时延副本；
- 对短输入模型关键的前部 layers；
- 补齐后可以立即让某层 ready 的 pages。

初始实现使用 prefix configuration 降级，例如从前 16 层降为前 8 层，再降为前 4 层，而不是逐 page 进行复杂全局优化。

---

## 8. 调度与缓存的联合关系

调度器和缓存控制器共享同一个 Pipeline Cost Estimator。

对于每张候选 GPU，本地缓存控制器返回：

- 需要释放的容量；
- 建议 eviction plan；
- 淘汰后的目标模型暴露加载时延；
- 对其它模型造成的未来缓存损失。

全局调度器根据这些结果选择目标 GPU。

请求执行完成后，缓存控制器根据最新的：

- 模型热度；
- 输入长度分布；
- GPU cache capacity；
- 集群副本分布；
- 未来流水线收益；

决定当前模型和历史模型应保留的 prefix configuration。

---

## 9. 与现有 Tangram 组件的关系

### 9.1 ODKV

ODKV 保留为支撑组件，用于动态释放未使用的 KV 预算，为 inactive parameter cache 提供更多容量。

KV cache 和 active model weights 为不可淘汰内存；inactive model parameter pages 为可抢占缓存。

### 9.2 GPU Affinity Scheduler

原有基于可复用参数字节数的 affinity 评分被替换为 pipeline-aware score。

调度器不再只判断某张 GPU 命中了多少参数，而是直接估计：

- 当前 batch 能否近乎无损地完成模型加载；
- 缺失加载是否能被 Prefill 隐藏；
- 选择该 GPU 会破坏多少未来缓存价值。

### 9.3 原 Tensor Allocator 和 PGP

删除原来的 variable-size tensor allocator、外部碎片管理和 PGP compaction。

VMM page mapping 负责解决物理空间离散和地址连续性问题。

---

## 10. MVP 实现范围

第一版实现限定为：

- decoder-only dense LLM；
- 单 GPU 或固定 tensor parallel degree；
- 只优化模型首次激活时的 Prefill；
- layer-granularity compute-load pipeline；
- prefix-based cache configurations；
- 基于离线 Profile 的计算时间估算；
- 基于历史统计的模型热度和输入长度分布；
- Decode 阶段恢复普通完整模型执行路径。

第一版暂不实现：

- layer 内 operator-level pipeline；
- 跨 GPU cooperative execution；
- 任意非连续 layer cache configuration；
- 复杂的长期 workload prediction；
- 推理过程中动态卸载 active model weights；
- 动态改变 tensor parallel degree。

---



## 11. 最终方案概括

Tangram-VMM 为每个模型维护稳定连续的虚拟地址空间，并在模型实例生命周期之外保留部分参数页。请求到来时，系统根据目标模型在各 GPU 上的实际 page residency，估计 layer-wise 计算-加载流水线的暴露加载时延。

系统选择当前请求启动成本低且对未来缓存破坏较小的 GPU。在目标 GPU 上，只加载缺失参数页，并在前部 layer ready 后立即启动 Prefill，同时继续加载后续层。缓存策略根据模型热度、输入长度分布、层位置和集群副本状态，优先保留能够减少未来流水线 stall 的参数前缀。

整个系统统一优化的目标是：

> 最小化当前和未来请求无法被计算隐藏的模型加载时间。
