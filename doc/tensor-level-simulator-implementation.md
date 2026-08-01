# Tensor-level Simulator Implementation

本文档记录 Tensor-level LayerWeave 模拟器的实现过程、执行命令、验证结果
以及尚未完成的 GPU profile 工作。本文档会随实现推进持续更新。

## 1. 实现目标

第一版支持五种实验策略：

- baseline；
- pipe_only；
- reuse_only；
- minimal；
- Aegaeon。

同时支持三种参数内存组织：

- Segment-based tensor memory；
- Page-based tensor allocation；
- Page-based compact allocation。

## 2. 实施记录

### 2026-07-28：环境与现有实现检查

- 当前 LayerWeave 模拟器按 decoder layer 构造 loading/compute stage；
- safetensors checkpoint 可直接读取 Tensor 名称、大小和 shape；
- legacy `GPUTensorPool` 使用一整块 `cudaMalloc` pool 和不规则连续
  `GPUMemoryRegion`；
- `GlobalDeFrag()` 会把保留 region 向 pool 起点搬移，并按实际移动字节
  记录 compaction 数据量；
- VMM stable-weight backend 为整个模型预留连续 VA，并以物理 page
  map/unmap；
- 当前会话执行 `nvidia-smi` 失败，错误为无法连接 NVIDIA driver，因此
  GPU Tensor compute profile 暂时排在 CPU 实现和验证之后。

### 2026-07-28：新增文件

- `tools/layerpipe/export_layerweave_tensor_layout.py`
  - 从 Hugging Face safetensors header 导出 Tensor 名称、shape、大小、
    layer ID 和 compact offset；
  - 当前 compact offset 是 exporter 确定性排列得到的 CPU 模拟代理值，
    尚未用 runtime stable VA 核验。
- `tools/layerpipe/layerweave_tensor_pipeline_sim.py`
  - Tensor-level loading/compute timeline；
  - Segment、tensor-page、compact-page 三种 backend；
  - baseline、pipe_only、reuse_only、minimal、Aegaeon；
  - H2D、allocation、map/unmap、compaction、碎片和 tail 指标。
- `tools/layerpipe/test_layerweave_tensor_pipeline_sim.py`
  - CPU-only backend 不变量测试。
- `tools/layerpipe/profile_layerweave_tensor_compute.py`
  - 顺序执行 Tensor shape 对应的 FP16 GEMM GPU microprofile；
  - 相同 shape 只测量一次；
  - 结果可通过 `--tensor-compute-profile` 传给模拟器；
  - 该结果是 isolated GEMM proxy，不等于完整 vLLM fused-operator profile。

## 3. 当前计算 Profile 语义

当前实现先读取已有 M4 layer compute profile，再按照同一 layer 内各 Tensor
的 logical bytes 占比拆分 layer compute time：

```text
tensor_compute_ms
= layer_compute_ms * tensor_bytes / layer_tensor_bytes
```

该方法保证一个 layer 内所有 Tensor compute 之和等于原 M4 layer compute，
适合先验证模拟器代码、物理布局和流水线趋势，但不是 GPU 实测的 Tensor
operator profile。输出 JSON 会明确记录：

```text
tensor_compute_profile_source =
M4 layer profile distributed by tensor logical bytes
```

真实 Tensor/fused-operator CUDA event profile 将在 GPU 可用后最后顺序执行。

## 4. 待执行验证

以下验证已经完成：

- Python syntax/compile；
- CPU backend 单元测试；
- 导出 8 模型 Tensor layout；
- 三种 layout 的 smoke；
- 五种策略输出结构检查；
- 21 个 model-switch 请求对比；
- page-size 可行性检查；
- GPU 可用性复查；
- GPU Tensor GEMM microprofile smoke；
- GPU profile 与模拟器集成 smoke。

完整 vLLM fused-operator CUDA event profile 尚未实现，具体边界见本文档末尾。

## 5. 内存 Backend 实现

### 5.1 Segment-based tensor memory

`SegmentMemory` 维护一条按地址排序的连续 extent list：

```text
free / loading / resident tensor segment
```

当前实现：

- 每个 Tensor 对应一个连续、大小等于 logical bytes 的 segment；
- cache、reuse、eviction 和 compute-ready 都以完整 Tensor 为单位；
- 相邻 free extent 自动合并；
- 分配同时检查 total free bytes 和 largest free extent；
- PGP proxy 使用模型访问次数作为第一价值维度，并用 Tensor LRU
  作为 tie-break；
- 当前模型已经加载的 Tensor 不作为 victim；
- 支持 `never`、`on-allocation-failure`、`cost-aware` 三种 compaction
  策略；
- cost-aware 比较 GPU compaction 时间与继续淘汰 Tensor 后的预计重加载
  时间；
- compaction 只统计地址真实发生移动的 Tensor bytes。

需要说明：当前 PGP 是在线 retention/eviction proxy，并未逐行移植 legacy
`PartitionedBinPacking()` 的递归 placement 搜索。extent、free/coalesce 和
global defrag 语义与 legacy `GPUTensorPool` 对齐。

### 5.2 Page-based tensor allocation

`TensorPageMemory` 为每个 Tensor 分配独立页面：

```text
physical_bytes(tensor)
= ceil(tensor_bytes / page_size) * page_size
```

当前实现：

- Tensor 独占自己的 VA/page allocation；
- Tensor 不与其他 Tensor 共享尾页；
- 只允许完整 Tensor resident 或 absent；
- 加载和淘汰均以完整 Tensor 为单位；
- 统计 per-tensor internal fragmentation；
- 统计 map/unmap calls 和 mapped/unmapped pages；
- 一个 Tensor 的连续 pages 视为一次 map/unmap call。

### 5.3 Page-based compact allocation

`CompactPageMemory` 为每个模型构造一个紧凑的逻辑地址空间：

```text
Tensor A | Tensor B | Tensor C | ...
```

然后按照模型 offset 切分物理页。当前实现：

- Tensor 可以跨多个 page；
- page 可以跨 Tensor 边界；
- Tensor 的所有依赖 page 都 resident 时才算 reusable；
- 加载一个 page 可以同时完成多个 Tensor；
- 淘汰一个 page 可以同时破坏多个完整 Tensor；
- page eviction damage 根据当前完整 Tensor 集合投影；
- 没有任何完整 Tensor 可以使用的 resident page 记为 stranded bytes；
- 相邻 missing/victim pages 合并成 mapping run；
- 小 page 场景下按一次 reclamation 的候选快照批量选 victim，避免每淘汰
  一页都全量扫描 resident pages。

最后一点是 CPU 模拟复杂度与逐页精确重新求值之间的折中。每次新的容量
reclamation 会重新计算 page damage，但同一批次内部不会逐页更新全部候选。

## 6. 五种策略

实现支持：

- `baseline`
  - 每个请求完全 cold；
  - 所有参数加载完成后再执行 compute。
- `pipe_only`
  - 每个请求完全 cold；
  - Tensor i 的 compute 与后续 Tensor loading overlap。
- `reuse_only`
  - replay Minimal 完全相同的 GPU routing 和 request-before cache state；
  - 所有 missing 参数加载完成后再 compute。
- `minimal`
  - 模型访问频率 + LRU 的基本重用策略；
  - 使用 Tensor-level loading/compute pipeline；
  - victim object 由具体 memory backend 决定。
- `Aegaeon`
  - replay Minimal routing；
  - 使用前一请求真实 trace decode tokens 形成预取窗口；
  - 只允许 whole-model double-buffer prefetch；
  - capacity gate 使用各 backend 的真实 physical footprint；
  - Tensor-page 的 whole-model map call 数按 Tensor allocation 数统计；
  - Compact-page 的连续模型 arena 计为一个 mapping run。

## 7. 指标

每种策略都输出：

```text
critical_path_ms
hot_compute_ms
exposed_load_ms
h2d_ms / h2d_bytes
allocation_ms
map_ms / unmap_ms
compaction_ms / compaction_moved_bytes
evicted_objects / evicted_bytes
map_calls / unmap_calls
mapped_pages / unmapped_pages
```

内存指标包括：

```text
logical_resident_bytes
physical_resident_bytes
internal_fragmentation_bytes
external_fragmentation_ratio
largest_free_extent_bytes
stranded_resident_bytes
```

所有主要时延同时统计：

```text
mean / P50 / P90 / P95 / P99 / max
```

`--inspect-request-id` 可以保存指定请求的完整 Tensor loading/compute
时间线，避免为所有请求输出庞大的逐 Tensor timeline。

## 8. 执行命令与结果

### 8.1 CPU 单元测试

```bash
cd /mnt/n0/Tangram/Tangram

python -m py_compile \
  tools/layerpipe/export_layerweave_tensor_layout.py \
  tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  tools/layerpipe/profile_layerweave_tensor_compute.py \
  tools/layerpipe/test_layerweave_tensor_pipeline_sim.py

python tools/layerpipe/test_layerweave_tensor_pipeline_sim.py
```

结果：

```text
Ran 5 tests in 0.001s
OK
```

测试覆盖：

- Tensor-page per-tensor rounding；
- Compact page 跨 Tensor 边界；
- Segment external fragmentation 与 compaction；
- whole-model rounding；
- 第二次 Tensor pipeline activation full hit。

### 8.2 导出真实 8 模型 Tensor layout

```bash
python tools/layerpipe/export_layerweave_tensor_layout.py \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --output docs/tensor-level-sim/tensor-layout.json
```

结果：

```text
models        = 8
tensors       = 3635
logical_bytes = 159421916160
```

该命令只读取 safetensors header，不读取权重 payload。

### 8.3 21 请求三布局验证

公共命令参数：

```bash
python tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --profile docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --trace evaluation/traces/servegen_tangram.trace \
  --pool-gib 42 \
  --max-requests 30 \
  --gpus 2 \
  --trace-mode model-switches \
  --input-scale 4 \
  --inspect-request-id 0
```

三种 layout 分别使用：

```text
segment:      --memory-layout segment
tensor-page:  --memory-layout tensor-page  --page-size-mib 8
compact-page: --memory-layout compact-page --page-size-mib 64
```

`max-requests=30` 在 `model-switches` 过滤后得到 21 个模拟请求。

#### Mean critical path

| Policy | Segment | Tensor-page 8 MiB | Compact-page 64 MiB |
|---|---:|---:|---:|
| baseline | 1275.41 ms | 1282.48 ms | 1275.40 ms |
| pipe_only | 1119.10 ms | 1125.05 ms | 1130.87 ms |
| reuse_only | 902.40 ms | 988.70 ms | 851.11 ms |
| minimal | 815.46 ms | 908.17 ms | 784.13 ms |
| Aegaeon | 740.03 ms | 895.05 ms | 857.13 ms |

这些数值使用 layer-byte-split Tensor compute proxy，不是最终 GPU
Tensor-operator 结论。

#### Minimal 的空间和传输

| Metric | Segment | Tensor-page 8 MiB | Compact-page 64 MiB |
|---|---:|---:|---:|
| H2D total | 195.70 GiB | 253.46 GiB | 203.20 GiB |
| Mean physical resident | 38.94 GiB | 39.19 GiB | 39.13 GiB |
| Mean internal fragmentation | 0 | 3.95 GiB | 0.06 GiB |
| Mean stranded bytes | 0 | 0 | 0.122 GiB |

Tensor-page 的 H2D bytes 只统计 logical payload；较高 H2D 来自
per-tensor rounding 降低了可缓存 Tensor 数量，而不是复制 padding bytes。

### 8.4 Page-size 可行性

8 请求短 smoke 中：

```text
Tensor-page 8 MiB  = PASS
Tensor-page 16 MiB = PASS
Tensor-page 32 MiB = FAIL
Tensor-page 64 MiB = FAIL
Compact-page 8/16/32/64 MiB = PASS
```

覆盖 21 个 model-switch 请求、包含 model 5 后：

```text
Tensor-page 8 MiB  = PASS
Tensor-page 16 MiB = FAIL
```

16 MiB 的首个失败为 model 5：

```text
weight footprint = 47,781,511,168 bytes
KV bytes          =     90,177,536 bytes
42 GiB pool       = 45,097,156,608 bytes
```

32 MiB 的短 sweep 首个失败为 model 7：

```text
weight footprint = 45,063,602,176 bytes
KV bytes          =    209,715,200 bytes
required          = 45,273,317,376 bytes
42 GiB pool       = 45,097,156,608 bytes
```

模拟器现在会在正式 replay 前执行 feasibility validation，并给出具体
request/model/weight/KV/pool，而不是运行到一半才返回不透明 OOM。

### 8.5 Compact-page CPU 热路径修正

最初 8/16 MiB compact-page sweep 很慢。原因是 page damage 计算中为每个
candidate page 构造：

```python
resident_pages - {page}
```

这会反复复制整个 resident set。修正为：

```python
if tensor_pages <= resident_pages:
    removing_required_page_breaks_tensor = True
```

并将一次容量 reclamation 改成候选快照上的批量选择。

修正后的 8 请求 wall time：

```text
Compact-page 8 MiB  = 21.63 s
Compact-page 16 MiB =  9.87 s
Compact-page 64 MiB = 13.04 s
```

小 page 仍然产生更多 physical page state，这是预期的模拟复杂度增长。

## 9. GPU Tensor compute microprofile

### 9.1 GPU 状态

默认 sandbox 中 CUDA driver 不可见，但宿主机检查显示：

```text
GPU 0: NVIDIA L40, 13 MiB / 46068 MiB, utilization 0%
GPU 1: NVIDIA L40, 13 MiB / 46068 MiB, utilization 0%
GPU 2: NVIDIA L40, 13 MiB / 46068 MiB, utilization 0%
GPU 3: NVIDIA L40, 13 MiB / 46068 MiB, utilization 0%
```

### 9.2 顺序 GPU smoke

执行：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/profile_layerweave_tensor_compute.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --models 0 \
  --gpu 0 \
  --tokens 32,128 \
  --repeats 3 \
  --max-unique-shapes 8 \
  --output \
    docs/tensor-level-sim/tensor-compute-gpu0-model0-smoke.json
```

结果：

```text
GPU              = NVIDIA L40
unique shapes    = 5
profiled tensors = 253
elapsed          = 0.381 s
```

实际测量的 shape：

```text
151936 x 2048
2048 x 2048
256 x 2048
11008 x 2048
2048 x 11008
```

随后使用：

```text
--tensor-compute-profile \
docs/tensor-level-sim/tensor-compute-gpu0-model0-smoke.json
```

完成了两请求 simulator integration smoke。未被该文件覆盖的模型或 Tensor
会自动回退到 M4 layer-byte-split profile，输出 metadata 会明确说明 mixed
profile source。

## 10. 当前输出文件

```text
docs/tensor-level-sim/tensor-layout.json
docs/tensor-level-sim/tensor-compute-gpu0-model0-smoke.json
docs/tensor-level-sim/gpu-profile-integration-smoke.json
docs/tensor-level-sim/validation30-segment.json
docs/tensor-level-sim/validation30-tensor-page.json
docs/tensor-level-sim/validation30-compact-page.json
docs/tensor-level-sim/sweep-*.json
```

## 11. 当前边界与后续工作

当前已经完成可运行的第一版，但以下边界必须保留：

1. Tensor 顺序来自 safetensors name/layer/operator 的确定性排序；
   compact offset 也是 exporter 顺序，尚未与 runtime stable VA 逐 Tensor
   对齐。
2. 默认 Tensor compute window 来自 M4 layer compute 按 Tensor bytes
   拆分。
3. GPU microprofile 是 isolated FP16 linear，不包含真实 vLLM 的 fused
   QKV/gate-up、attention、activation、norm、host gap 和 kernel launch
   交互。
4. 完整可信的 Tensor compute profile 仍需要在真实 LayerWeave/vLLM
   forward 中按 fused weight-use stage 增加 CUDA event。
5. Segment PGP 是 frequency+LRU retention 与 cost-aware compaction
   proxy，不是 legacy `PartitionedBinPacking()` 的逐行复刻。
6. Compact-page 在同一次批量 reclamation 内使用候选价值快照；下一次
   reclamation 会重新计算完整 Tensor 到 page 的价值投影。
7. VMM map/unmap、segment allocation 和 compaction fixed cost 当前由 CLI
   参数控制，尚未进行完整 CUDA Driver API calibration sweep。
8. 当前模拟器按 system-wide serial request order 回放，不是并发 vLLM
   scheduler。

下一步若需要论文级结果，优先顺序应为：

1. 导出 runtime stable VA 的精确 Tensor offset；
2. 增加 fused weight-use stage CUDA event；
3. 为 8 个模型、多输入 bucket 建立 Tensor profile；
4. 校准不同 page size 的 create/map/unmap 调用成本；
5. 再运行完整 1000-request matched sweep。

## 12. TensorGroup、pipeline-value replacement 与 exposed-stall routing

### 12.1 TensorGroup

新增 `TensorGroup` 存储粒度。Tensor 仍然保持原来的 compute/profile
语义；group 负责 allocation、H2D、resident hit 和 eviction。

构造方法参考 `tools/mock_allocation/src/registered_model.h::MergeTGs()`：

1. 按 simulator 的 Tensor compute 顺序遍历；
2. 连续合并相邻 Tensor，直到 group 达到最小大小；
3. 最后一个不足最小大小的 group 合并进前一个 group；
4. `0 MiB` 表示保持 one-Tensor-per-group。

公共默认值：

```text
--tensor-group-min-mib 64
```

可按 policy 覆盖，参数可以重复：

```text
--policy-tensor-group-mib baseline=0
--policy-tensor-group-mib pipe_only=32
--policy-tensor-group-mib reuse_only=64
--policy-tensor-group-mib minimal=128
--policy-tensor-group-mib aegaeon=64
```

三个 backend 的语义为：

- Segment：一个 group 对应一个连续 segment；
- Tensor-page：一个 group 独占 `ceil(group_bytes/page_size)` 个 page；
- Compact-page：VA 仍按整个模型紧凑排列，group 只合并 prepare/map/load
  范围，物理 residency 和 eviction 仍以 page 为单位。

### 12.2 Pipeline-value replacement

新增：

```text
--replacement-policy frequency-lru
--replacement-policy pipeline-value
```

`pipeline-value` 对存储对象使用：

```text
value(object) =
    predicted_future_model_probability(model)
    * estimated_eliminated_exposed_stall(object)
```

未来模型概率采用 system-wide 在线 exponentially-decayed demand，默认：

```text
--demand-decay 0.9
```

不读取未来 queue，也不使用 oracle future trace。每个模型用其回放请求的
中位 input length 建立代表性 cold pipeline timeline。一个 group 的
`estimated_eliminated_exposed_stall` 是它在该 timeline 中实际造成的局部
ready stall。Tensor-page/Segment 直接将该值赋给 group；Compact-page
把 group value 均摊到它覆盖的 page，同一个 page 的多个贡献相加。

首版曾使用“从全冷流水单独移除一个 group 后关键路径的全局边际”。由于
pipeline stall 非加性，大量 group 会被错误估值为 0；实现已改成局部
ready-stall，并增加相应 CPU invariant test。

### 12.3 Exposed-stall routing

新增：

```text
--routing-policy critical-path
--routing-policy exposed-stall
```

`exposed-stall` 会 clone 每个 GPU 的当前 backend 状态，预测请求在该 GPU
上的 `exposed_load_ms`，选择最小者。多个 GPU 分数相同时使用 round-robin，
并通过 CPU test 验证 `GPU 0 -> GPU 1 -> ...` 的平局轮转。

当前 system-wide serial request 模型下，同一个请求的 hot compute 在各
GPU 相同，因此：

```text
critical_path = hot_compute + exposed_load
```

`critical-path` 和 `exposed-stall` 在数学上通常选择同一 GPU；新策略主要
用于明确路由目标，并为以后加入 GPU queue/service time 后保留独立口径。

### 12.4 验证

静态和 CPU 测试：

```bash
python -m py_compile \
  tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  tools/layerpipe/test_layerweave_tensor_pipeline_sim.py

python tools/layerpipe/test_layerweave_tensor_pipeline_sim.py
```

结果：

```text
Ran 10 tests in 0.002s
OK
```

测试覆盖：

- 原有三种 backend invariant；
- compute-adjacent TensorGroup 合并；
- TensorGroup hit 和 Tensor-page rounding；
- pipeline-value value ordering；
- cold timeline local ready-stall；
- exposed-stall routing 的 round-robin tie break。

21-request 验证公共参数：

```bash
python tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --profile docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --trace evaluation/traces/servegen_tangram.trace \
  --pool-gib 42 \
  --max-requests 30 \
  --gpus 2 \
  --trace-mode model-switches \
  --input-scale 4 \
  --page-size-mib 8 \
  --h2d-gbps 24.56
```

Minimal critical path：

| Layout/Policy | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Segment, group=0, frequency-LRU | 729.08 | 462.49 | 1477.72 | 2063.66 | 2323.97 | 2389.05 |
| Segment, group=64 MiB, frequency-LRU | 694.89 | 499.36 | 1345.42 | 1770.53 | 1960.49 | 2007.98 |
| Segment, group=64 MiB, pipeline-value | 1739.66 | 1446.75 | 3145.77 | 4941.65 | 5448.68 | 5575.44 |
| Tensor-page, group=0, frequency-LRU | 791.48 | 667.37 | 1488.92 | 2127.72 | 2158.28 | 2165.92 |
| Tensor-page, group=64 MiB, frequency-LRU | 652.10 | 376.83 | 1343.30 | 1429.48 | 1647.46 | 1701.95 |
| Tensor-page, group=64 MiB, pipeline-value | 747.03 | 635.74 | 1500.01 | 1694.06 | 1700.41 | 1702.00 |

TensorGroup 的直接结果：

- Segment mean critical path 降低 34.19 ms，P95 降低 293.13 ms；
- Tensor-page mean 降低 139.38 ms，P95 降低 698.24 ms；
- Tensor-page H2D 从 253.46 GiB 降至 212.92 GiB；
- Tensor-page map calls 从 6183 降至 1487；
- simulator wall time 从约 28 秒降至约 6 秒。

Pipeline-value 在该短 trace 上没有优于 frequency-LRU。它倾向把后续请求
路由到已保留高 pipeline-value 对象的 GPU，21 请求后半段出现单 GPU
cache affinity；Segment 因此产生 291.34 GiB H2D 和约
1018.26 ms/request compaction，明显退化。Tensor-page 也增加到
296.78 GiB H2D。该结果说明 value 定义已经落地，但 value projection
和 routing/cache feedback 仍需更长 trace、matched routing replay 和
transition-conditioned probability 验证，当前不能声称该替换策略更优。

新增结果：

```text
docs/tensor-level-sim/group-validation30-segment-group0.json
docs/tensor-level-sim/group-validation30-segment-group64.json
docs/tensor-level-sim/group-validation30-segment-group64-pipeline.json
docs/tensor-level-sim/group-validation30-segment-group64-pipeline-route.json
docs/tensor-level-sim/group-validation30-tensor-page-group0.json
docs/tensor-level-sim/group-validation30-tensor-page-group64.json
docs/tensor-level-sim/group-validation30-tensor-page-group64-pipeline.json
docs/tensor-level-sim/group-validation30-tensor-page-group64-pipeline-route.json
docs/tensor-level-sim/group-smoke-compact-page.json
```

## 13. Segment 64 MiB TensorGroup 的 1000-source-row cache-policy 对比

### 13.1 公平性控制

本轮使用：

```text
memory layout       = segment
TensorGroup minimum = 64 MiB
pool                = 42 GiB/GPU
GPUs                = 2
H2D                 = 24.56 GB/s
trace mode           = model-switches
max source requests  = 1000
simulated requests   = 704
routing score        = exposed loading stall
```

为了把 cache replacement 与 routing feedback 分开，模拟器新增：

```text
--routing-replay PREVIOUS_RESULT.json
```

它读取旧结果顶层的 `routing` 数组，要求长度与本轮模拟请求完全一致，并
验证所有 GPU id 合法。启用后跳过在线 GPU 选择，但仍正常更新各 GPU cache
状态和全局在线 demand。

本轮顺序运行三组：

1. `frequency-lru + adaptive exposed-stall routing`，生成 matched routing；
2. `pipeline-value + replay frequency-lru routing`，比较 replacement 本身；
3. `pipeline-value + adaptive exposed-stall routing`，观察 cache/routing
   feedback。

第一组：

```bash
python tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --profile docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --trace evaluation/traces/servegen_tangram.trace \
  --pool-gib 42 --max-requests 1000 --gpus 2 \
  --trace-mode model-switches --input-scale 4 \
  --memory-layout segment --page-size-mib 8 \
  --h2d-gbps 24.56 --tensor-group-min-mib 64 \
  --replacement-policy frequency-lru \
  --routing-policy exposed-stall \
  --output \
    docs/tensor-level-sim/segment-group64-1000-frequency.json
```

Matched pipeline-value 在相同命令上替换：

```text
--replacement-policy pipeline-value
--routing-replay \
  docs/tensor-level-sim/segment-group64-1000-frequency.json
--output \
  docs/tensor-level-sim/segment-group64-1000-pipeline-matched.json
```

Adaptive pipeline-value 不传 `--routing-replay`，输出：

```text
docs/tensor-level-sim/segment-group64-1000-pipeline-adaptive.json
```

三个 simulator wall time 分别为：

```text
frequency adaptive       134.68 s
pipeline-value matched    33.38 s
pipeline-value adaptive  157.36 s
```

固定 routing 后不需要逐请求 clone 两个 GPU backend 并进行候选预测，因此
matched replay 明显更快。

### 13.2 Minimal critical path

| Cache/routing | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| frequency-LRU, adaptive | 553.26 | 174.14 | 1401.97 | 2050.82 | 3462.42 | 5332.84 |
| pipeline-value, matched | 691.79 | 224.06 | 1884.77 | 2549.56 | 5273.19 | 7414.50 |
| pipeline-value, adaptive | 823.75 | 299.53 | 2141.75 | 3283.12 | 5188.34 | 6208.65 |

在完全相同 routing 下，pipeline-value 相比 frequency-LRU：

```text
mean critical-path delta = +138.52 ms
improved requests        = 74
identical requests       = 406
regressed requests       = 224
```

因此长 trace 仍不支持“当前 pipeline-value 更优”的结论。

### 13.3 Loading、eviction 和 compaction

| Cache/routing | H2D | Evicted | Compaction moved | Compaction requests |
|---|---:|---:|---:|---:|
| frequency-LRU, adaptive | 2450.29 GiB | 2367.09 GiB | 89921.70 GiB | 213 |
| pipeline-value, matched | 2675.33 GiB | 2592.00 GiB | 172412.42 GiB | 286 |
| pipeline-value, adaptive | 3520.64 GiB | 3438.89 GiB | 232958.72 GiB | 299 |

对应 per-request compaction latency：

| Cache/routing | Mean | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| frequency-LRU, adaptive | 158.86 | 562.83 | 864.36 | 2181.87 | 3654.65 |
| pipeline-value, matched | 304.60 | 1080.91 | 1519.13 | 3607.16 | 5736.31 |
| pipeline-value, adaptive | 411.55 | 1264.67 | 1857.41 | 3662.11 | 4580.93 |

Matched 请求的 critical-path delta 与 compaction delta 的 Pearson
correlation 为 `0.986`；与 H2D delta 的 correlation 只有 `0.305`。
主要退化来自 placement/compaction，而不是纯 H2D。

### 13.4 Pipeline value 为什么退化

64 MiB grouping 后共有 1065 个 group。代表性 cold pipeline 估值中：

```text
zero pipeline-value groups = 216 / 1065
model 3 zero-value bytes    = 10.52 GiB
model 4 zero-value bytes    = 15.61 GiB
```

这些 group 的加载在代表性全冷时间线中能被前序 compute 掩盖，所以局部
exposed-stall value 为 0。但是对于 Segment，它们并非没有系统成本：

- 淘汰后仍需重新申请连续 segment；
- 重载虽然可能与 compute overlap，但会改变 extent 排列；
- 大量零价值 group churn 会增加外部碎片；
- 后续大 group 分配触发 compaction，移动大量仍 resident 的参数。

当前 value：

```text
future probability * eliminated exposed loading stall
```

没有包含：

```text
expected compaction damage
expected placement difficulty
reload/eviction churn
group size or value density
```

因此它对流水语义是局部合理的，但对 Segment allocator 并不完整。

### 13.5 Adaptive routing feedback

frequency-LRU routing：

```text
GPU 0 = 283 requests
GPU 1 = 421 requests
```

pipeline-value adaptive routing：

```text
GPU 0 = 459 requests
GPU 1 = 245 requests
```

Adaptive pipeline-value 形成更强的模型/GPU 吸附。例如 model 0、3、4、5
几乎全部进入 GPU 0，而 frequency-LRU 会把不同热门模型分散到两个 GPU。
在当前没有 queue/service-time 的 serial simulator 中，routing 只看到本次
请求 exposed stall，不会惩罚长期 cache concentration，于是 replacement
与 routing 形成正反馈，H2D 和 compaction 进一步升高。

### 13.6 结论与下一步

当前 704-request 结果表明：

1. Segment 64 MiB group 本身有效，但当前 pipeline-value replacement
   明显不如 frequency-LRU；
2. 即使固定完全相同 routing，pipeline-value 仍退化，说明问题首先在
   replacement value，而不只是 routing；
3. adaptive routing 会进一步放大退化；
4. Segment policy 不能只计算 exposed-stall benefit，还必须建模 allocator
   damage。

下一版建议把 Segment value 扩展为：

```text
P(future model)
* eliminated exposed stall
+ reload_churn_weight * expected reload cost
+ compaction_weight * expected placement/compaction damage
```

并至少增加以下控制：

- 为零 exposed-stall group 设置 reload/placement value floor；
- 按 `value / physical_bytes` 与连续 extent 可放置性共同选 victim；
- routing score 加入 cache concentration 或 predicted transition damage；
- 用本轮 frequency routing 做 matched weight sweep，再启用 adaptive
  routing。

## 14. 默认 model-switches 约定与单 GPU cache-policy 对比

### 14.1 后续实验默认 trace 语义

除非实验明确说明使用 `--trace-mode all`，后续 Tensor-level simulator
实验统一默认：

```text
--trace-mode model-switches
```

处理顺序为：

1. `--max-requests N` 先限制源 trace 的前 N 行；
2. 对这些源请求按顺序扫描；
3. 连续相同 `model_id` 的请求视为同一次模型驻留区间；
4. 只保留每个连续区间的第一个请求；
5. 相邻相同模型不发生模型切换，不纳入 simulator latency/cache-policy
   统计。

因此本轮 `--max-requests 1000` 对应 704 个 model-switch 请求。

### 14.2 单 GPU实验设置

为完全移除多 GPU routing 影响，运行：

```text
memory layout       = segment
TensorGroup minimum = 64 MiB
GPUs                = 1
pool                = 42 GiB
trace source rows   = 1000
model-switch rows   = 704
H2D                 = 24.56 GB/s
```

frequency-LRU：

```bash
python tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --profile docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --trace evaluation/traces/servegen_tangram.trace \
  --pool-gib 42 --max-requests 1000 --gpus 1 \
  --trace-mode model-switches --input-scale 4 \
  --memory-layout segment --page-size-mib 8 \
  --h2d-gbps 24.56 --tensor-group-min-mib 64 \
  --replacement-policy frequency-lru \
  --routing-policy exposed-stall \
  --output \
    docs/tensor-level-sim/segment-group64-1000-1gpu-frequency.json
```

pipeline-value 只替换：

```text
--replacement-policy pipeline-value
--output \
  docs/tensor-level-sim/segment-group64-1000-1gpu-pipeline.json
```

只有 GPU 0，因此所有 routing 都是 0，`critical-path` 与 `exposed-stall`
routing 参数不会改变 GPU 选择。

### 14.3 Critical path

| Policy | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| frequency-LRU | 985.01 | 672.07 | 2720.68 | 4247.25 | 5409.53 | 6028.30 |
| pipeline-value | 1232.48 | 838.44 | 2917.17 | 4424.10 | 5562.54 | 6225.75 |

Pipeline-value 相比 frequency-LRU：

```text
mean delta       = +247.47 ms
mean regression  = +25.12%
improved         = 169 requests
identical        = 88 requests
regressed        = 447 requests
```

所以即使完全移除多 GPU 和 routing，当前 pipeline-value 仍明显更差。

### 14.4 Cache 和 allocator 指标

| Policy | H2D | Evicted | Compaction moved | Compaction requests |
|---|---:|---:|---:|---:|
| frequency-LRU | 4679.82 GiB | 4638.37 GiB | 283915.37 GiB | 436 |
| pipeline-value | 5620.91 GiB | 5579.41 GiB | 420549.47 GiB | 576 |

Compaction latency：

| Policy | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| frequency-LRU | 501.50 | 34.70 | 1344.74 | 2915.56 | 3731.33 | 4350.11 |
| pipeline-value | 742.94 | 423.62 | 1861.23 | 2972.98 | 3935.23 | 4598.03 |

逐请求 critical-path delta 与：

```text
compaction delta correlation = 0.974
H2D delta correlation        = 0.693
```

单 GPU下 H2D churn 和 compaction 都变差，但 compaction 仍是最强的性能
相关因素。

### 14.5 结论

多 GPU cache affinity 不是 pipeline-value 退化的必要条件。它在上一轮
会进一步放大退化，但单 GPU下 replacement 自身已经存在问题：

- pipeline benefit 没有覆盖 Segment placement/compaction damage；
- zero/low exposed-stall group 被频繁回收；
- 回收后增加 H2D、外部碎片和后续大 group compaction；
- 单 GPU容量竞争更强，使这个问题比两 GPU matched routing 更明显。

在修正 value 公式以前，Segment 64 MiB 的默认 cache policy 应继续使用
`frequency-lru`。

## 15. 单 GPU Compact-page 对比：隔离 Segment allocator

### 15.1 实验目的与设置

为了判断上一节 pipeline-value 退化是否只来自 Segment allocator，保持
相同单 GPU、相同 704 个 model-switch 请求，替换为：

```text
memory layout       = compact-page
physical page size  = 64 MiB
TensorGroup minimum = 64 MiB
compaction          = none
```

其他参数与第 14 节一致。分别输出：

```text
docs/tensor-level-sim/compact-page64-group64-1000-1gpu-frequency.json
docs/tensor-level-sim/compact-page64-group64-1000-1gpu-pipeline.json
```

两个 simulator wall time 分别为 222.31 秒和 162.63 秒。wall time 是 CPU
模拟开销，不进入预测 latency；frequency page value 需要投影 page removal
对完整 Tensor reuse 的破坏，因此控制器模拟更重。

### 15.2 Critical path

| Policy | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| frequency-LRU | 487.770 | 324.763 | 1255.118 | 1683.505 | 1772.847 | 1772.887 |
| pipeline-value | 487.257 | 305.611 | 1230.062 | 1685.279 | 1773.629 | 1774.317 |

Pipeline-value：

```text
mean delta   = -0.513 ms
P50 delta    = -19.152 ms
P90 delta    = -25.056 ms
P95 delta    = +1.774 ms
P99 delta    = +0.782 ms
Max delta    = +1.430 ms
improved     = 233 requests
identical    = 149 requests
regressed    = 322 requests
```

去掉 Segment compaction 后，pipeline-value 不再出现明显 latency 退化。
Mean 小幅改善约 0.11%，但 P95/P99/Max 略差，整体应视为 latency 基本持平，
而不是显著胜出。

### 15.3 Page cache 和管理开销

| Metric | frequency-LRU | pipeline-value |
|---|---:|---:|
| H2D | 4655.77 GiB | 5000.81 GiB |
| Evicted physical bytes | 4619.94 GiB | 4964.88 GiB |
| Mapped pages | 74583 | 80102 |
| Unmapped pages | 73919 | 79438 |
| Map calls | 31728 | 37455 |
| Unmap calls | 33124 | 60681 |
| Mean stranded resident bytes | 0.089 GiB | 0.949 GiB |
| Max stranded resident bytes | 1.250 GiB | 8.062 GiB |

Pipeline-value 的 H2D 增加 345.04 GiB，约 7.4%；unmap calls 增加约
83.2%。当前 CLI 假设下 map/unmap latency 较小，所以这些额外管理操作没有
明显拉高 critical path。

逐请求 critical-path delta 与：

```text
H2D delta correlation   = 0.828
unmap delta correlation = 0.386
```

### 15.4 为什么更多 H2D 但 latency 基本持平

Pipeline-value 有意保留能够直接消除 exposed stall 的 page，并允许低
exposed-value page 更频繁换入换出。这会产生：

- 更多总 H2D；
- 更多 page eviction/map/unmap；
- 更高 stranded residency；
- 但额外 H2D 多数仍能与 compute overlap。

因此它改善部分模型的 exposed path，同时牺牲不直接暴露在关键路径上的
数据复用。例如 mean critical-path delta：

```text
model 0 = -32.09 ms
model 1 = -22.60 ms
model 2 = +51.50 ms
model 3 = +16.52 ms
model 4 = -69.25 ms
model 5 = -40.66 ms
```

不同模型的正负效果最终接近抵消。

### 15.5 结论

Compact-page 实验把原因拆开后可以得出：

1. Segment 上 `+247.47 ms` 的巨大退化主要来自外部碎片和 compaction；
2. pipeline-value 并非天然产生更差的流水 latency；Compact-page 上 mean
   与 frequency-LRU 基本持平；
3. 当前 pipeline-value 仍不是更好的综合 cache policy，因为它使用更多
   H2D、page operations 和 stranded memory，却没有获得稳定的 tail
   latency 改善；
4. 所以不能把所有问题都归因于 allocator：allocator 是 Segment latency
   退化的主因，但 value projection 本身仍有 cache/space efficiency
   问题。

下一步应分别针对 backend 调整 value：

- Segment：加入 placement/compaction damage；
- Compact-page：加入 page sharing、stranded bytes、H2D churn 和 value
  density；
- 保留 exposed-stall benefit，但不能把它作为唯一价值项。

## 16. Protected Pipeline Floor、Soft Cache 与 Shadow Frequency-LRU

### 16.1 新策略

新增 replacement policy：

```text
--replacement-policy protected-pipeline
--replacement-policy protected-pipeline-shadow
```

每个模型从本轮 model-switch 请求中选取：

```text
min / P25 / P50 / P75 / max input length
```

分别生成 TensorGroup cold pipeline timeline。最终使用：

```text
--pipeline-robust-stat mean
--pipeline-protected-threshold-ms 5
```

一个 group 跨输入 bucket 的 mean local ready stall 大于 5 ms 时标记为
protected，否则作为 soft cache。当前 64 MiB grouping 下：

```text
protected groups = 244 / 1065
protected bytes  = 64.03 GiB
soft bytes       = 84.44 GiB
```

Victim 使用字典序：

```text
soft before protected
lower future-model probability first
lower stall benefit first
larger contiguous reclaim span first
older group LRU first
```

因此 pipeline 分类是最高优先级；同一类别内仍先驱逐冷模型，符合
“pipeline 找 soft 空间、热度策略保持稳定”的设计。

### 16.2 Shadow guard

`protected-pipeline-shadow` 默认以 frequency-LRU victim 为安全候选。只有
同时满足以下条件时才允许改用 pipeline victim：

1. pipeline victim 是 soft；
2. frequency-LRU victim 是 protected；
3. pipeline victim 的
   `future_probability * reload_bytes` 不超过 LRU victim 的 1.25 倍。

配置：

```text
--shadow-max-churn-ratio 1.25
```

这是 per-eviction conservative guard，不维护第二份完整 shadow cache，
因此限制单次偏离，但不提供全轨迹 dominance 保证。

### 16.3 实现验证和控制器热点

CPU tests 增加：

- protected group 不会先于 soft group 被淘汰；
- soft victim 的预计 churn 超过 shadow budget 时回退 frequency-LRU。

结果：

```text
Ran 12 tests
OK
```

首版 Segment placement rank 在每个候选上重复扫描 extent，长回放控制器很
慢。实现已改为一次 victim selection 预计算所有候选的 contiguous reclaim
span，并且 `_victims()` 只返回调用方实际使用的第一个 victim。该优化不
改变 eviction 结果和预测请求 latency。

还验证了一个重要配置陷阱：五个 bucket 使用 P90 时接近 max；配合 0.1 ms
阈值会把 1065/1065 groups 全部标成 protected，完全失去 soft-cache
语义。最终实验使用 mean/5 ms，上述错误配置不纳入最终结果。

### 16.4 单 GPU 704-request 结果

设置继续使用：

```text
Segment
64 MiB TensorGroup
1 GPU
42 GiB pool
1000 source rows
704 model-switch requests
24.56 GB/s H2D
```

输出：

```text
docs/tensor-level-sim/segment-group64-1000-1gpu-frequency.json
docs/tensor-level-sim/segment-group64-1000-1gpu-protected-pipeline.json
docs/tensor-level-sim/segment-group64-1000-1gpu-protected-pipeline-shadow.json
```

Minimal critical path：

| Policy | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| frequency-LRU | 985.01 | 672.07 | 2720.68 | 4247.25 | 5409.53 | 6028.30 |
| protected-pipeline | 1260.72 | 1105.04 | 2318.16 | 3025.80 | 4440.81 | 5313.33 |
| protected-pipeline-shadow | 1265.52 | 901.19 | 2790.41 | 4231.49 | 5817.50 | 6579.75 |

相对 frequency-LRU：

| Policy | Mean delta | Better | Same | Worse |
|---|---:|---:|---:|---:|
| protected-pipeline | +275.70 ms | 166 | 7 | 531 |
| protected-pipeline-shadow | +280.50 ms | 190 | 77 | 437 |

### 16.5 Cache 和 allocator 指标

| Policy | H2D | Evicted | Compaction moved | Compaction requests |
|---|---:|---:|---:|---:|
| frequency-LRU | 4679.82 GiB | 4638.37 GiB | 283915.37 GiB | 436 |
| protected-pipeline | 7286.33 GiB | 7244.80 GiB | 407242.99 GiB | 690 |
| protected-pipeline-shadow | 5881.65 GiB | 5840.14 GiB | 429045.59 GiB | 583 |

Shadow 把 H2D 从 7286.33 GiB 降到 5881.65 GiB，但仍比 LRU 高
25.7%。它没有降低 compaction moved bytes：保守的逐次 churn guard
不等于全局 placement guard，部分保留决策仍会形成不利 extent layout。

### 16.6 结果解释

无 shadow 的 protected policy 没有改善 mean，但明显改善 tail：

```text
P90 reduction = 14.8%
P95 reduction = 28.8%
P99 reduction = 17.9%
Max reduction = 11.9%
```

它把少数极慢请求的关键参数保护下来，代价是：

- full/near-full cache hits 大幅减少；
- 704 请求中只有 15 个请求 H2D 为 0，LRU 为 271 个；
- 更多 soft groups 被持续换入；
- 690 个请求发生 compaction；
- P50 和 mean 明显退化。

所以它是一种强 tail-oriented policy，而不是综合性能更优的 policy。

Shadow 降低了部分 churn，但基本丢失 tail 优势：

- P90 比 LRU 略差；
- P95 基本持平；
- P99/Max 更差；
- Mean 仍退化约 280.5 ms。

当前 per-victim shadow guard 不足以同时保住 LRU hit rate 和 pipeline tail
benefit。

### 16.7 当前结论

本轮实现验证了 protected/soft 思路可以有效保护 tail-critical groups，
但 Segment 上不能把所有 soft groups 都视为廉价空间：

1. “H2D 可以 overlap”不代表“反复 eviction 没有成本”；
2. soft churn 会降低 hit rate并增加外部碎片；
3. per-victim shadow 不能预测长期 extent layout；
4. protected floor 应是受容量预算约束的每模型 prefix/configuration，而
   不是对所有模型同时永久拥有最高优先级。

下一版更合适的方向是：

- 为 protected floor 设置每模型和全局容量预算；
- 用 MCKP 选择哪些模型的 protected configuration 当前值得驻留；
- 未被选中的 protected groups 回到普通 frequency-LRU，而不是永远压过
  所有 soft groups；
- Segment 单独使用 placement-aware shadow state 或直接比较 victim set
  的预计 compaction cost；
- 将 mean/throughput policy 与 tail-SLO policy 分开报告。

## 17. Compact-page 后缀优先 frequency-LRU

### 17.1 实现语义

新增独立替换策略：

```text
--replacement-policy frequency-lru-suffix-first
```

原有 `frequency-lru` 保持不变。新策略的 victim key 为：

```text
(model_access_count, -compute_position, storage_unit_last_access)
```

因此跨模型仍然先淘汰访问次数更少的冷模型；同一模型内优先淘汰计算顺序
更靠后的 TensorGroup/page；计算位置相同时再使用 storage-unit LRU。

Segment 和 Tensor-page 使用 `group_id` 表示计算位置。Compact-page 仍先
判断回收 page 会破坏哪些完整 Tensor 的复用，再把这些 Tensor 的计算位置
投影到 page value；模型连续 VA 中越靠后的 page 因而优先被回收。没有破坏
任何当前完整 Tensor 的 page 仍然优先于其他 page 回收。

新增三项 CPU invariant：

- 同一模型内后缀 group 先于前缀 group；
- 跨模型时 model frequency 仍高于 suffix position；
- Compact-page 能把后缀 Tensor 的计算位置投影到 page victim value。

共 15 个 CPU 测试通过。

### 17.2 配对实验

与第 15 节原始 LRU 完全相同：

```text
memory layout       = compact-page
physical page size  = 64 MiB
TensorGroup minimum = 64 MiB
GPUs                = 1
pool                = 42 GiB
trace source rows   = 1000
trace mode          = model-switches
model-switch rows   = 704
input scale         = 4
H2D                 = 24.56 GB/s
routing             = exposed-stall (单 GPU，routing 均为 GPU 0)
```

原始 LRU 结果：

```text
docs/tensor-level-sim/compact-page64-group64-1000-1gpu-frequency.json
```

后缀优先 LRU 命令：

```bash
python tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --profile docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --trace evaluation/traces/servegen_tangram.trace \
  --pool-gib 42 --max-requests 1000 --gpus 1 \
  --trace-mode model-switches --input-scale 4 \
  --memory-layout compact-page --page-size-mib 64 \
  --h2d-gbps 24.56 --tensor-group-min-mib 64 \
  --replacement-policy frequency-lru-suffix-first \
  --routing-policy exposed-stall \
  --output \
    docs/tensor-level-sim/compact-page64-group64-1000-1gpu-suffix-first-lru.json
```

模拟器 wall time 为 263.00 秒；这是 CPU controller 模拟时间，不进入预测
请求 latency。

### 17.3 Critical path

| Policy | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| 原始 frequency-LRU | 487.770 | 324.763 | 1255.118 | 1683.505 | 1772.847 | 1772.887 |
| 后缀优先 frequency-LRU | 487.781 | 326.201 | 1255.905 | 1684.356 | 1773.138 | 1773.317 |

后缀优先相对原始 LRU 的逐请求配对结果：

```text
mean delta    = +0.012 ms
median delta  =  0.000 ms
P90 delta     = +0.860 ms
P95 delta     = +3.514 ms
P99 delta     = +31.840 ms
best delta    = -116.653 ms
worst delta   = +276.378 ms
improved      = 114 requests
identical     = 250 requests
regressed     = 340 requests
```

均值基本完全持平，但后缀优先没有形成稳定收益：退化请求多于改善请求，
P50/P90/P95/P99/Max 均略差。

### 17.4 Cache 和 VMM 管理开销

| Metric | 原始 LRU | 后缀优先 LRU | Delta |
|---|---:|---:|---:|
| Mean exposed load | 193.880 ms | 193.892 ms | +0.012 ms |
| Mean H2D time | 289.128 ms | 289.644 ms | +0.515 ms |
| H2D | 4655.77 GiB | 4664.07 GiB | +8.30 GiB |
| Evicted physical bytes | 4619.94 GiB | 4630.50 GiB | +10.56 GiB |
| Zero-H2D requests | 265 | 266 | +1 |
| Mapped pages | 74583 | 74752 | +169 |
| Unmapped pages | 73919 | 74088 | +169 |
| Map calls | 31728 | 31914 | +186 |
| Unmap calls | 33124 | 45081 | +11957 |

后缀 page 的加载更容易被前面的计算隐藏，但“无损隐藏”只在未来使用时有
足够前缀计算窗口才成立。固定后缀优先没有考虑：

- 下次请求的输入长度和实际 compute window；
- page 是否跨 Tensor 边界；
- 多个 victim page 能否合并为连续 unmap range；
- 为保留前缀而造成的额外 H2D churn。

本轮最明显的副作用是 unmap calls 增加 36.1%，说明被选中的 page 更离散。
更高 H2D 和 VMM 管理开销抵消了后缀加载更容易隐藏的优势。结论是：在该
单 GPU Compact-page workload 下，单纯把同模型 LRU 顺序改成后缀优先不
优于原始 LRU。下一版若继续沿这个方向，应按“预计 exposed stall +
连续 victim run”联合选页，而不是固定按层位置排序。

## 18. 后续 cache-policy base 与模拟器 CPU profile

### 18.1 后续实验默认 base

从本节开始，缓存策略优化方向的探索实验默认使用：

```text
memory layout       = segment
TensorGroup minimum = 64 MiB
GPUs                = 1
pool                = 42 GiB
trace mode          = model-switches
input scale         = 4
H2D                 = 24.56 GB/s
```

除非实验明确研究 page projection、VMM map/unmap 或空间效率，否则不再把
Compact-page 作为 cache-policy exploration 的默认 base。Segment 的
TensorGroup 同时是 compute、cache、reuse 和 eviction 单位，可以直接验证
计算语义或 exposed-stall value；实验仍需单独报告 Segment fragmentation
和 compaction 代价。

### 18.2 为什么 1000-request 模拟需要数分钟

模拟器是纯 Python 离散事件模拟，当前慢点不在 GPU compute profile，而在
controller state exploration。主循环对每个请求固定执行：

```text
baseline
pipe_only
minimal routing probe
minimal real execution
reuse_only
```

所以单 GPU 时每个请求仍会走五次 request execution。即使 `--gpus 1`，
routing 代码也会：

1. `deepcopy` 整个 cache backend；
2. 在副本上完整执行一次 minimal；
3. 确定唯一的 GPU 0；
4. 在真实 backend 上再完整执行一次 minimal。

这次 routing probe 在单 GPU 下没有决策价值，是确定的重复工作。

此外，Compact-page 每次需要回收页面时会：

1. 枚举所有非 active model 的驻留 page；
2. 对每个 page 扫描它覆盖的 Tensor；
3. 判断删除该 page 会破坏哪些完整 Tensor；
4. 将 Tensor cache value 投影为 page value；
5. 对所有候选 page 排序；
6. shortage 较大时仍反复处理大量候选。

### 18.3 100-source-request CPU profile

使用与长实验相同参数，把 `--max-requests` 改为 100；折叠后得到 83 个
model-switch 请求。`cProfile` 会显著放大 wall time，因此这里只使用累计
时间占比定位热点，不把 profile wall time当作正常运行时间。

| Layout | Profile wall time | Python calls |
|---|---:|---:|
| Segment | 49.50 s | 112.25 M |
| Compact-page | 122.78 s | 283.42 M |

Compact-page 主要热点：

| Hot path | Cumulative time | 占主模拟时间 |
|---|---:|---:|
| `_evict_pages` | 79.26 s | 64.8% |
| `_page_value` | 71.50 s | 58.5% |
| routing `clone/deepcopy` | 32.96 s | 27.0% |
| `sorted` page candidates | 78.25 s | 64.0% |

这些累计路径存在父子包含关系，不能直接相加。`_page_value` 被调用
8.08 M 次，`_value` 被调用 19.53 M 次；这正是 Compact-page 的
Tensor-to-page value projection 成本。

Segment 主要热点：

| Hot path | Cumulative time | 占主模拟时间 |
|---|---:|---:|
| routing `clone/deepcopy` | 18.53 s | 37.6% |
| `prepare_tensor` | 25.25 s | 51.2% |
| `_victims` | 10.95 s | 22.2% |
| resident extent scan | 7.42 s | 15.0% |

Segment 没有逐页 Tensor value projection，但仍存在：

- 单 GPU routing clone；
- 每次 Tensor resident check 线性扫描 extents；
- eviction 时重复构造和扫描 victim candidates；
- 每个 request 都为 timeline 创建逐 Tensor 字典；
- 即使只分析 minimal，仍同时模拟五种顶层 policy。

因此 1000 source rows/704 model switches 达到分钟级是当前 Python
实现和算法复杂度的结果，不是需要或正在执行真实 Tensor GPU compute。

后续若单独优化模拟器运行时间，优先级应为：

1. `gpus == 1` 时跳过 routing clone/probe，直接选择 GPU 0；
2. 增加只运行指定顶层 policy 的选项，cache-policy sweep 可只运行
   `minimal`；
3. 没有 `--inspect-request-id` 时不构造逐 Tensor timeline；
4. Segment 增加 key-to-extent/resident index 和增量 victim heap；
5. Compact-page 缓存 page-to-value 投影，并使用增量候选结构或
   victim-run selection，避免每次全量排序。

### 18.4 已实现的前三项优化

已完成：

1. `gpus == 1` 且没有 routing replay 时直接选择 GPU 0，不再 clone cache
   和执行 routing probe；多 GPU路径保持不变。
2. 新增 `--policy-suite minimal-only`。默认值仍为 `all`，保证旧命令和
   完整五策略输出兼容；cache-policy sweep 应显式使用 `minimal-only`。
3. 只有 `--inspect-request-id` 指定的请求才构造逐 Tensor timeline；普通
   policy request JSON 仍不包含 timeline。

推荐的后续 Segment cache-policy sweep 参数为：

```text
--memory-layout segment
--tensor-group-min-mib 64
--gpus 1
--policy-suite minimal-only
--trace-mode model-switches
```

使用相同 100 source rows/83 model switches 验证：

| Run | Wall time |
|---|---:|
| 优化后完整 `all` suite | 8.69 s |
| 优化后 `minimal-only` | 4.46 s |

`minimal-only` 相对优化后完整 suite 加速 1.95 倍。使用相同 `cProfile`
工具比较修改前后完整 suite：

```text
before = 49.50 s
after  = 20.15 s
speedup = 2.46x
```

这一项同时包含跳过单 GPU clone/probe 和 lazy timeline 的收益。修改前后
83 条 routing、五个 policy 的所有 request metrics 逐字段完全一致；
`minimal-only` 的 minimal summary/request rows 也与完整 suite 完全一致。
15 项 CPU 单元测试、Python compile 和 diff check 均通过。

## 19. 对齐 mock_allocation Segment 管理

### 19.1 reuse lookup：布局结构与驻留索引分离

`tools/mock_allocation/src/vram_manager.h` 的 Segment pool 同时维护：

```text
memory_regions
    按地址顺序连接的 region 布局

allocated_regions[fingerprint] -> region
    TensorGroup 驻留哈希索引
```

`GetTensor(fingerprint)` 直接查询 `allocated_regions`，不会为每次 reuse
检查遍历完整 region 链表。释放时从 map 删除 fingerprint，并只合并
相邻 free regions；compaction/region move 后同步更新 map 指针。

Tensor simulator 的 `SegmentMemory` 现已采用相同分层：

```text
extents
    维护地址顺序、free holes 和 compaction layout

allocated_extents[(model_id, group_id)] -> Extent
    维护 O(1) 平均复杂度的 reuse lookup
```

以下路径同步维护索引：

- 新 TensorGroup 分裂 free extent并分配；
- eviction/free；
- compaction 重建 Extent；
- clone/deepcopy；
- 测试直接替换 extents 时通过 lazy fallback 修复索引。

相同 100 source rows/83 model switches、Segment 64 MiB、
`minimal-only`：

| 实现 | Wall time |
|---|---:|
| 只有前三项 simulator 优化 | 4.459 s |
| 增加 allocated-extents index | 3.061 s |

驻留索引单项加速 1.46 倍，wall time 减少 31.3%。修改前后 routing、
minimal summary 和所有 request rows 逐字段完全一致。当前共有 16 项 CPU
测试通过。

### 19.2 PartitionedBinPacking 实际做了什么

`PartitionedBinPacking` 不是对单个 TensorGroup 调用一次局部 first-fit。
它先取得本次模型访问的完整 `tg_to_load` 集合，然后：

1. 按 TensorGroup size 降序排列；
2. 把所有 free regions 视为初始可用区域组；
3. 考察相邻 free regions 之间的拆分点；
4. 拆分收益定义为不再需要跨越/移动的 allocated bytes；
5. 用双桶 greedy 判断待加载 TensorGroups 能否分别装进左右 free
   capacity；
6. 选择高收益且可行的拆分并递归处理；
7. 最后只在每个 partition 内调用 `MergeRegions`；
8. 将属于该 partition 的多个 TensorGroups 连续放入合并后的 free
   region。

因此它的关键收益是：

```text
global compaction
    把全池 allocated regions 全部向左搬

PartitionedBinPacking
    联合考虑本次全部 missing TensorGroups
    保留有价值的分区边界
    只在必要的局部范围搬移
```

这既降低 moved bytes，也能把多个待加载 TensorGroups 批量放入规划好的
区域，避免每装一个 group 就重新搜索/compact 一次。

### 19.3 为什么不能直接替换当前 `_compact()`

当前 Tensor simulator 在计算循环中逐 Tensor 调用：

```text
prepare_tensor(group)
```

当某个 group 分配失败时，allocator 只知道当前 group 的大小，不知道同一
请求后面还有哪些 missing groups。此时运行所谓 PBP 会退化成单对象局部
compaction，不具备 mock_allocation 算法的联合 bin-packing 语义。

要忠实加入 PBP，需要把 Segment request path拆成：

```text
request planning phase
    枚举本请求全部 missing TensorGroups
    批量计算需要释放的总空间
    cache policy 批量选择 victims
    PartitionedBinPacking 规划 partitions 和目标 Extents
    记录各 partition 的 moved bytes

execution phase
    仍按 Tensor 计算顺序触发 H2D
    在 group 第一次使用时计入其 load/allocation ready time
    compaction cost 按实际发生位置进入 pipeline timeline
```

这样才能同时保持：

- Tensor-level load/compute pipeline；
- cache reuse 语义；
- PBP 的批量 placement；
- 局部 compaction moved bytes；
- 与 mock_allocation 可对照的 fragmentation metrics。

PBP 会改变 placement、compaction 和最终 latency，因此应作为独立可选
Segment allocation policy 与当前 first-fit/global-compaction 做 matched
comparison，而不应混入本节的结果等价加速。

## 20. Segment 与 Page backend 拆分边界

后续实现不再要求 Segment、Tensor-page 和 Compact-page 共用同一种逐
Tensor allocation control flow。目标文件边界为：

```text
tensor_sim_memory_common.py
    TensorSpec / TensorGroup / TensorModel
    LoadCost
    replacement value 与公共 backend protocol

tensor_sim_page_backends.py
    TensorPageMemory
    CompactPageMemory
    VMM page map/unmap、rounding、Tensor-to-page value projection

tensor_sim_segment_backend.py
    mock_allocation 风格的 region layout
    allocated_regions reuse index
    GreedyDrop
    PartitionedBinPacking
    local MergeRegions

layerweave_tensor_pipeline_sim.py
    trace/profile
    request execution
    routing
    summary/JSON/CLI
```

公共 backend protocol 分为：

```text
begin_request(...)
plan_request(model_id, ordered_groups)
prepare_tensor(...)
finish_request(...)
metrics()
```

Page backend 的 `plan_request` 可以为空操作，继续在 `prepare_tensor`
中按需 map/load。Segment backend 的 `plan_request` 则：

1. 收集本请求所有 missing TensorGroups；
2. 按 mock_allocation GreedyDrop 批量释放足够总空间；
3. 调用 PartitionedBinPacking 生成 region partitions；
4. 对每个 partition 执行局部 MergeRegions；
5. 为 missing groups 预留目标 region；
6. `prepare_tensor` 只在计算顺序第一次访问 group 时计入 H2D 和已经规划
   好的 allocation/relocation cost。

Segment 默认语义将以 `tools/mock_allocation/src/vram_manager.h` 为来源：

- fingerprint/TensorGroup reuse；
- 地址有序 regions + `allocated_regions` 哈希索引；
- free 时相邻 region 合并；
- request-level GreedyDrop；
- size-descending TensorGroup packing；
- recursive partition split；
- partition-local relocation，而不是默认 global left compaction。

旧 Python first-fit/global-compaction backend 在迁移期间只作为
`legacy-segment` matched reference 保留，不再作为后续 cache-policy
探索的 Segment base。Page backend 不复用 Segment placement 或
fragmentation policy。

### 20.1 Request-level PBP 接入状态

主执行路径已经调用统一的 `plan_request(model_id, tensors)`：

- Page backend 使用默认空 planning，行为不变；
- `segment` 使用 mock_allocation 风格 request-level planning；
- 原来的逐 Tensor first-fit/global compaction 暴露为
  `legacy-segment`。

当前 `segment` planning 已实现：

1. 去重并收集全部 missing TensorGroups；
2. 一次计算总 required bytes并批量 GreedyDrop；
3. TensorGroups 按 size 降序；
4. 按 free-region capacity 做 recursive two-bin split；
5. 以被分区边界保留下来的 allocated bytes 作为 split profit；
6. 对 terminal partition 执行局部 merge；
7. 在 partition free range 中批量预留 TensorGroup extents；
8. `prepare_tensor` 按原计算顺序消费 planned load cost。

100 source rows/83 model switches 的 PBP smoke 完成：

```text
layout       = segment
suite        = minimal-only
requests     = 83
wall time    = 2.597 s
```

新增 request-level planning invariant，当前共 17 项 CPU tests。

### 20.2 物理文件拆分完成

backend 实现已经从主模拟器中移出：

```text
tools/layerpipe/tensor_sim_memory_common.py
    235 lines
    TensorSpec/TensorGroup/TensorModel
    LoadCost/Extent
    MemoryBackend 与 replacement 公共逻辑

tools/layerpipe/tensor_sim_segment_backend.py
    447 lines
    legacy Segment
    allocated_extents reuse index
    request-level mock_allocation PBP Segment

tools/layerpipe/tensor_sim_page_backends.py
    288 lines
    TensorPageMemory
    CompactPageMemory

tools/layerpipe/layerweave_tensor_pipeline_sim.py
    881 lines
    CLI、trace/profile、execution、routing、summary
    并 re-export backend classes 保持测试/调用兼容
```

拆分后的 matched checks：

- `legacy-segment` 与拆分前83-request routing 和全部 policy request rows
  逐字段一致；
- 修复后的 PBP 相同输入连续运行两次，routing 和全部 policy rows
  逐字段一致；
- KV weight-capacity tail 不再被 PBP 误算为空闲容量；
- 当前共18项 CPU tests，Python compile 与 diff check通过。

修复后的83-request PBP smoke：

```text
mean critical path       = 659.593 ms
P50/P90/P95/P99/Max      =
  460.275 / 1511.272 / 1680.059 / 1795.128 / 1798.122 ms
H2D                      = 860.2 GiB
partition relocation     = 1292.36 GiB
simulator wall time      = 2.598 s
```

该 smoke 用于结构和确定性验证，不是与旧 Segment 的最终性能结论；长
trace matched comparison 仍应单独运行。

## 21. Segment 统一权重/KV显存池

新版 `segment` 不再使用：

```text
weight_capacity = pool_capacity - current_KV
```

也不再把 `[weight_capacity, pool_end)` 当作固定KV尾区。该抽象会在输入
长度增加时制造动态地址边界，并触发 mock_allocation 中不存在的跨边界
global compaction。

当前统一池生命周期与 `tools/mock_allocation` 对齐：

```text
begin_request
    释放上一请求全部显式KV Extents
    相邻free extents原地合并

plan_request
    在完整pool中复用/加载权重
    request-level GreedyDrop + PBP
    从所有离散free extents分配当前请求KV blocks
    block容量不足时淘汰非active模型权重
```

KV Extent key为：

```text
("kv", allocation_id)
```

权重仍为：

```text
(model_id, tensor_group_id)
```

KV分配按 `kv_block_bytes` 对每个free extent取整，可跨多个离散region，
不要求一个覆盖全部KV的连续大空间。当前active模型权重不会为自己的KV
被淘汰；其他模型权重按当前replacement policy回收。

TTFT计费保持 mock benchmark口径：

- KV block bookkeeping/allocation本身当前为0 ms；
- 为KV淘汰权重本身为0 ms，但改变后续cache状态；
- 不再存在KV尾边界global compaction；
- 权重PBP relocation、H2D、Segment allocation和compute仍进入关键路径。

新增输出：

```text
memory.kv_resident_bytes
summary.memory_kv_resident_bytes
model.kv_memory_model
model.segment_allocator
```

100 source rows/83 model-switch验证：

```text
每条请求KV bytes == ceil((input+output)/32) * 2 MiB
weight resident + KV resident <= 42 GiB
mean KV                = 0.177 GiB
max KV                 = 0.549 GiB
wall time              = 2.829 s
```

| Metric | 人工KV尾边界PBP | 统一池PBP |
|---|---:|---:|
| Mean critical path | 659.593 ms | 649.846 ms |
| P50 | 460.275 ms | 445.737 ms |
| P90 | 1511.272 ms | 1506.500 ms |
| P95 | 1680.059 ms | 1680.156 ms |
| P99 | 1795.128 ms | 1757.916 ms |
| Max | 1798.122 ms | 1760.892 ms |
| H2D | 860.20 GiB | 860.51 GiB |
| Relocation moved | 1292.36 GiB | 635.73 GiB |

该短trace显示，删除人工尾边界后relocation moved bytes下降约50.8%；
critical path mean下降约9.75 ms。它说明旧边界模型确实制造了大量额外
搬移，但最终cache-policy结论仍需使用1000-source-row长trace确认。

## 22. 统一池 Segment replacement policy 的 1000-source-row 对比

### 22.1 实验条件

本轮使用当前 `segment` 的统一权重/KV显存池和 request-level PBP。
除 replacement policy 外，其余条件完全一致：

```text
source rows             = 1000
model-switch requests   = 704
GPU                     = 1
pool                    = 42 GiB
TensorGroup minimum     = 64 MiB
trace mode              = model-switches
input scale             = 4
H2D bandwidth           = 24.56 GB/s
policy suite            = minimal-only
compaction policy       = on-allocation-failure
```

基础命令如下，只替换 `POLICY` 和 `OUTPUT`：

```bash
python tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --profile docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --trace evaluation/traces/servegen_tangram.trace \
  --pool-gib 42 --max-requests 1000 --gpus 1 \
  --trace-mode model-switches --input-scale 4 \
  --memory-layout segment --h2d-gbps 24.56 \
  --tensor-group-min-mib 64 \
  --replacement-policy POLICY \
  --policy-suite minimal-only \
  --output OUTPUT
```

测试的五个 policy 为：

```text
frequency-lru
frequency-lru-suffix-first
pipeline-value
protected-pipeline
protected-pipeline-shadow
```

每个输出都通过以下一致性检查：

- 704条请求的 request/model/token 序列完全相同；
- 每条请求的 hot compute 完全相同；
- `weight physical resident + explicit KV resident <= 42 GiB`；
- 没有 routing 差异，因为本轮固定为单GPU。

### 22.2 TTFT结果

单位均为 ms：

| Replacement policy | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| frequency-lru-suffix-first | **491.999** | **333.483** | 1250.535 | 1678.350 | **1751.075** | 1762.879 |
| frequency-lru | 494.653 | 336.094 | 1250.385 | 1678.195 | 1751.577 | **1760.892** |
| pipeline-value | 511.486 | 353.058 | 1221.513 | 1627.720 | 1754.262 | 1764.375 |
| protected-pipeline-shadow | 529.212 | 366.142 | 1224.047 | 1678.195 | 1756.193 | 1765.174 |
| protected-pipeline | 571.631 | 530.882 | **1123.559** | **1299.529** | 1753.880 | 1766.941 |

按 mean TTFT 排序，最优的是 `frequency-lru-suffix-first`。它相对
`frequency-lru`：

```text
mean  -2.653 ms  (-0.54%)
P50   -2.612 ms  (-0.78%)
P90   +0.150 ms
P95   +0.156 ms
P99   -0.502 ms
```

所以 suffix-first 的收益很小，主要出现在分布中部，不能表述成显著的
tail improvement。

如果目标函数只看 P90/P95，则 `protected-pipeline` 反而最好；但是它将
mean 增加 79.631 ms、P50增加197.399 ms。原因是它用大量普通请求上的
cache churn，换取部分请求在 P90/P95 位置上的更低 loading stall。
不同分位数最优项不同，因此“最好”必须绑定目标。本节后续 breakdown
采用 mean/P50 最优的 `frequency-lru-suffix-first`。

### 22.3 Cache traffic 与 allocator work

| Replacement policy | H2D GiB | PBP moved GiB | Evicted GiB | Evicted objects |
|---|---:|---:|---:|---:|
| frequency-lru-suffix-first | **4678.58** | **2925.99** | **4637.12** | 31411 |
| frequency-lru | 4680.30 | 3694.42 | 4638.85 | **31346** |
| pipeline-value | 5621.18 | 5197.84 | 5579.68 | 43067 |
| protected-pipeline-shadow | 5881.86 | 4454.75 | 5840.35 | 46143 |
| protected-pipeline | 7286.97 | 3892.52 | 7245.43 | 60192 |

`frequency-lru-suffix-first` 的H2D只比普通LRU少1.72 GiB，但PBP搬移少
768.43 GiB。由于GPU relocation带宽为864 GB/s，且一部分搬移被流水
隐藏，这个很大的byte差异最后只转化为约2.65 ms的mean TTFT收益。

三种pipeline-aware policy在当前trace上都产生更多H2D和eviction。
`protected-pipeline-shadow` 相比不带shadow的版本确实降低了H2D、
eviction和mean TTFT，但仍未恢复到frequency-LRU水平：

```text
protected-pipeline mean          = 571.631 ms
protected-pipeline-shadow mean   = 529.212 ms
frequency-lru mean               = 494.653 ms
```

### 22.4 最优组 latency breakdown

`frequency-lru-suffix-first` 的直接关键路径分解为：

| TTFT组成 | Mean ms | TTFT占比 |
|---|---:|---:|
| Hot tensor compute | 293.890 | 59.73% |
| Exposed loading | 198.109 | 40.27% |
| Critical path / TTFT | 491.999 | 100.00% |

这里 `exposed loading = critical path - hot compute`，它是已经考虑
load/compute流水重叠后的真实TTFT增量。

模拟器同时记录的原始loading工作量为：

| 原始工作量 | Mean ms/request |
|---|---:|
| H2D | 290.545 |
| PBP relocation/compaction | 5.173 |
| Segment allocation | 0.225 |
| 合计 | 295.943 |
| 被compute流水隐藏 | 97.834 |
| 暴露在关键路径 | 198.109 |

因此原始loading工作量中约33.06%被compute隐藏。原始H2D、relocation
和allocation不能直接与compute相加：三者按tensor计算顺序进入copy
流水，其中一部分与之前tensor的compute重叠。

为了继续把198.109 ms exposed loading归因到三种成本，本轮对同一
policy补跑H2D、relocation和allocation开/关的全部 `2^3` 组合：

```text
bit order = H2D, PBP relocation, Segment allocation
000 001 010 011 100 101 110 111
```

关闭成本的参数为：

```text
H2D off        : --h2d-gbps 1000000000000
relocation off : --gpu-copy-gbps 1000000000000
                 --compaction-fixed-ms 0
allocation off : --segment-allocation-ms 0
```

8组运行的请求、H2D bytes、PBP moved bytes、evicted bytes和evicted
objects均完全一致，说明开关只改变计时，没有改变placement和cache
决策。对每条请求做三因素Shapley分解，再汇总得到：

| Exposed loading归因 | Mean ms | Exposed占比 | TTFT占比 |
|---|---:|---:|---:|
| H2D | 192.962 | 97.40% | 39.22% |
| PBP relocation | 5.055 | 2.55% | 1.03% |
| Segment allocation | 0.093 | 0.05% | 0.02% |
| 合计 | 198.109 | 100.00% | 40.27% |

Shapley分解后的分位数如下。每个分位点是“逐请求归因值”的分位数，
不是TTFT分位数之间相减：

| 归因 | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| H2D | 192.962 | 40.756 | 579.991 | 1144.525 | 1218.705 | 1445.777 |
| PBP relocation | 5.055 | 0.000 | 21.246 | 26.152 | 35.592 | 45.922 |
| Segment allocation | 0.093 | 0.005 | 0.258 | 0.452 | 0.490 | 0.490 |

结论是：当前最优Segment组仍然主要受H2D影响，而不是Python allocator
固定成本或PBP relocation latency。PBP placement仍然重要，因为它会
间接决定未来复用和H2D；这里只是说在已经产生的关键路径时间里，
直接的relocation计时占比较小。

结果文件：

```text
docs/tensor-level-sim/segment-unified-group64-1000-frequency-lru.json
docs/tensor-level-sim/segment-unified-group64-1000-frequency-lru-suffix-first.json
docs/tensor-level-sim/segment-unified-group64-1000-pipeline-value.json
docs/tensor-level-sim/segment-unified-group64-1000-protected-pipeline.json
docs/tensor-level-sim/segment-unified-group64-1000-protected-pipeline-shadow.json
docs/tensor-level-sim/segment-unified-group64-1000-suffix-breakdown-{000..110}.json
```

## 23. 不同 LayerWeave trace 下的 Segment LRU 对比与部分驻留

### 23.1 Trace选择

此前 LayerWeave 为分离模型序列和输入shape影响，使用
`tools/layerpipe/build_layerweave_trace_variants.py` 从原始
ServeGen trace生成了四条matched-token-shape trace：

| Trace | 模型序列特征 |
|---|---|
| `balanced_iid` | 八模型均匀IID，禁止相邻同模型 |
| `round_robin` | 每8请求包含每个模型一次，block内随机排列 |
| `phase_shift` | 5个阶段，每阶段3个模型占70%，热点集合迁移 |
| `working_set_shift` | 每100请求切换一个4模型working set，集合部分重叠 |

每条trace都是1000请求。生成器只改变model sequence；每个模型的
`(input_tokens, output_tokens)` 仍从该模型在原ServeGen trace中的经验
分布seed-shuffle后循环取样。四条trace均禁止相邻同模型，所以
`--trace-mode model-switches` 不会继续折叠，1000 source rows对应1000个
有效请求。

本轮比较：

```text
frequency-lru
frequency-lru-suffix-first
```

其余实验条件与第22节一致：单GPU、42 GiB统一Segment权重/KV池、
64 MiB TensorGroup、input scale 4、24.56 GB/s H2D、minimal-only和
request-level PBP。

### 23.2 部分驻留口径

对每条请求，在该请求开始加载active model时按权重H2D bytes分类：

```text
full resident:
    h2d_bytes == 0

partial resident:
    0 < h2d_bytes < model_total_group_bytes

not resident:
    h2d_bytes == model_total_group_bytes
```

`model_total_group_bytes` 是该模型全部64 MiB TensorGroup的logical bytes
之和。分类只描述active model的权重，不把当前请求KV或PBP relocation
bytes算入模型驻留量。

此外报告：

```text
resident fraction = 1 - h2d_bytes / model_total_group_bytes
```

“Partial resident mean”只在partial请求内求平均；“All-request resident
mean”包含full、partial和not-resident全部请求。

### 23.3 TTFT对比

| Trace | Policy | Mean | P50 | P90 | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| balanced IID | frequency-LRU | 802.411 | 690.576 | 1274.173 | 1669.667 | 1751.075 | 1772.829 |
| balanced IID | suffix-first | **798.612** | **685.864** | 1277.699 | **1449.920** | 1751.075 | 1780.633 |
| round robin | frequency-LRU | **861.490** | **685.896** | 1679.611 | 1680.484 | **1767.961** | **1776.979** |
| round robin | suffix-first | 862.153 | 691.454 | **1678.195** | **1679.631** | 1770.563 | 1781.105 |
| phase shift | frequency-LRU | 729.519 | **668.483** | 1345.366 | 1664.681 | 1752.613 | **1769.238** |
| phase shift | suffix-first | **727.269** | 669.156 | **1333.783** | **1628.956** | **1751.681** | 1774.417 |
| working-set shift | frequency-LRU | 717.830 | 672.875 | 1336.916 | 1563.151 | 1753.317 | **1767.957** |
| working-set shift | suffix-first | **714.452** | **670.881** | **1321.873** | **1476.304** | **1751.075** | 1768.292 |

Mean TTFT上的suffix-first变化：

| Trace | Delta ms | Delta % |
|---|---:|---:|
| balanced IID | -3.799 | -0.47% |
| round robin | +0.663 | +0.08% |
| phase shift | -2.249 | -0.31% |
| working-set shift | -3.378 | -0.47% |

四条trace再次说明suffix-first只是小幅二级优化。除round-robin基本持平
外，其余三条mean改善0.31%--0.47%，与原始ServeGen switches上的
0.54%改善相近。

### 23.4 模型驻留状态比例

下表每行比例之和为100%：

| Trace | Policy | Full resident | Partial resident | Not resident |
|---|---|---:|---:|---:|
| balanced IID | frequency-LRU | 3.3% | 20.8% | 75.9% |
| balanced IID | suffix-first | 1.9% | 24.2% | 73.9% |
| round robin | frequency-LRU | 4.4% | 4.9% | 90.7% |
| round robin | suffix-first | 0.5% | 30.4% | 69.1% |
| phase shift | frequency-LRU | 15.7% | 27.5% | 56.8% |
| phase shift | suffix-first | 15.6% | 29.1% | 55.3% |
| working-set shift | frequency-LRU | 15.5% | 37.0% | 47.5% |
| working-set shift | suffix-first | 14.0% | 38.0% | 48.0% |

部分驻留请求中实际保留的模型比例：

| Trace | Policy | Partial resident mean | Partial resident median | All-request resident mean |
|---|---|---:|---:|---:|
| balanced IID | frequency-LRU | 47.37% | 37.47% | 13.15% |
| balanced IID | suffix-first | 45.29% | 37.55% | 12.86% |
| round robin | frequency-LRU | 46.11% | 48.02% | 6.66% |
| round robin | suffix-first | 18.44% | 11.49% | 6.11% |
| phase shift | frequency-LRU | 47.72% | 50.66% | 28.82% |
| phase shift | suffix-first | 46.27% | 50.74% | 29.07% |
| working-set shift | frequency-LRU | 41.99% | 45.37% | 31.04% |
| working-set shift | suffix-first | 43.40% | 45.40% | 30.49% |

作为对照，原始ServeGen前1000 source rows过滤后的704个model switches：

| Policy | Full resident | Partial resident | Not resident | Partial mean | All-request mean |
|---|---:|---:|---:|---:|---:|
| frequency-LRU | 38.49% | 25.99% | 35.51% | 73.39% | 57.57% |
| suffix-first | 37.78% | 26.70% | 35.51% | 74.83% | 57.77% |

原始trace有更强的模型局部性：即使只统计switch，其full-hit比例和部分
驻留时的resident fraction也显著高于四条压力trace。

### 23.5 分析

部分驻留请求比例不能单独预测suffix-first收益。

`working_set_shift`最有利于模型级缓存：

- 约52%请求至少部分驻留；
- all-request mean resident约31%；
- suffix-first的mean TTFT改善0.47%。

`balanced_iid`和`round_robin`持续在8个模型间切换，远超42 GiB池能稳定
保留的working set：

- balanced IID有约74%--76%完全未驻留；
- round-robin普通LRU有90.7%完全未驻留；
- 两种策略的大部分请求都接近cold model load，前后缀选择空间有限。

round-robin下suffix-first的partial比例从4.9%升到30.4%，但不能解读为
缓存显著改善。其partial请求平均只保留18.44%、中位只保留11.49%的模型
参数；all-request mean resident反而从6.66%下降到6.11%。它只是把大量
“完全未驻留”变成“残留少量TensorGroup”，不足以隐藏后续大部分H2D，
所以mean TTFT还退化0.08%。

因此评估计算语义感知replacement时，至少应联合报告：

1. full/partial/not-resident请求比例；
2. partial请求的resident byte fraction；
3. all-request mean resident fraction；
4. exposed loading和TTFT；
5. H2D与PBP relocation。

结果文件：

```text
docs/tensor-level-sim/segment-unified-group64-balanced_iid-frequency-lru.json
docs/tensor-level-sim/segment-unified-group64-balanced_iid-frequency-lru-suffix-first.json
docs/tensor-level-sim/segment-unified-group64-round_robin-frequency-lru.json
docs/tensor-level-sim/segment-unified-group64-round_robin-frequency-lru-suffix-first.json
docs/tensor-level-sim/segment-unified-group64-phase_shift-frequency-lru.json
docs/tensor-level-sim/segment-unified-group64-phase_shift-frequency-lru-suffix-first.json
docs/tensor-level-sim/segment-unified-group64-working_set_shift-frequency-lru.json
docs/tensor-level-sim/segment-unified-group64-working_set_shift-frequency-lru-suffix-first.json
```

## 24. Segment、2 GPU：pure LRU 与 pipeline-value

### 24.1 策略与公平性

此前文档中的 `frequency-lru` 不是严格 LRU：它先按模型历史访问次数排序，
只在相同频率内按 TensorGroup 最近访问时间打破平局。为直接回答 pure LRU
与 pipeline-value 的差异，模拟器新增：

```text
--replacement-policy lru
```

该策略只使用 storage-unit `last_access`，淘汰最近最少访问的 TensorGroup，
不读取模型频率、未来请求或 pipeline benefit。

实验设置：

```text
memory layout       = segment
Segment allocator   = request-level PBP
TensorGroup minimum = 64 MiB
pool                = 42 GiB/GPU
GPUs                = 2
trace mode           = model-switches
source requests      = 1000
simulated requests   = 704
input scale          = 4
H2D                  = 24.56 GB/s
```

先用 LRU 和 adaptive exposed-stall routing 生成 GPU routing：

```bash
python tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --profile docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --trace evaluation/traces/servegen_tangram.trace \
  --pool-gib 42 --max-requests 1000 --gpus 2 \
  --trace-mode model-switches --input-scale 4 \
  --memory-layout segment --page-size-mib 8 \
  --h2d-gbps 24.56 --tensor-group-min-mib 64 \
  --policy-suite minimal-only \
  --replacement-policy lru \
  --routing-policy exposed-stall \
  --output \
    docs/tensor-level-sim/segment-group64-1000-2gpu-lru.json
```

pipeline-value 使用 `--routing-replay` 重放完全相同的 704 个 GPU 选择：

```bash
python tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --profile docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --trace evaluation/traces/servegen_tangram.trace \
  --pool-gib 42 --max-requests 1000 --gpus 2 \
  --trace-mode model-switches --input-scale 4 \
  --memory-layout segment --page-size-mib 8 \
  --h2d-gbps 24.56 --tensor-group-min-mib 64 \
  --policy-suite minimal-only \
  --replacement-policy pipeline-value \
  --routing-policy exposed-stall \
  --routing-replay \
    docs/tensor-level-sim/segment-group64-1000-2gpu-lru.json \
  --output \
    docs/tensor-level-sim/segment-group64-1000-2gpu-pipeline-matched-lru.json
```

LRU routing 为 GPU 0/1 分别 `322/382` 个请求；JSON 校验确认两轮 routing
数组完全相同。因此以下差异只包含 replacement/cache-state feedback，
不包含 GPU routing 变化。

### 24.2 Critical path 和 exposed loading

| Metric | Policy | Mean | P50 | P90 | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| Critical path | LRU | 405.81 | **216.60** | 924.16 | 1291.73 | 1540.82 | 1757.78 |
| Critical path | pipeline-value | **396.17** | 241.87 | **828.39** | **1269.65** | **1408.86** | **1678.19** |
| Exposed load | LRU | 111.92 | 0 | 359.93 | 588.75 | 1153.51 | 1220.65 |
| Exposed load | pipeline-value | **102.29** | 0 | **331.18** | **500.39** | **1051.25** | **1219.16** |

在相同 routing 下，pipeline-value 相比 pure LRU：

```text
mean critical-path delta = -9.64 ms
mean improvement         = 2.37%
improved requests        = 166
identical requests       = 424
regressed requests       = 114
```

Pipeline-value 的 mean、P90、P95、P99 和 max 都更低，但 P50 高
`25.26 ms`。大量请求完全相同，是因为相同 routing 下两种策略对这些请求
产生相同 hit/miss 和关键路径。

### 24.3 Cache、H2D 和 PBP relocation

| Metric | LRU | pipeline-value | Pipeline-value delta |
|---|---:|---:|---:|
| H2D | 2567.13 GiB | 2849.76 GiB | +11.01% |
| Evicted bytes | 2484.08 GiB | 2766.72 GiB | +11.38% |
| Evicted objects | 17240 | 22674 | +31.52% |
| PBP moved bytes | 1566.35 GiB | 2160.15 GiB | +37.91% |
| Compaction requests | 165 | 249 | +50.91% |
| Mean compaction | 2.77 ms | 3.82 ms | +1.05 ms |

Pipeline-value 用更多 H2D、eviction 和 PBP relocation 换取更低的 exposed
loading。额外加载更常落在可被 Tensor pipeline compute 掩盖的位置，因此
总传输增加并不等价于 critical path 增加。

### 24.4 结论

在当前 `segment + 2 GPU + matched routing` 模拟中，pipeline-value 相比
pure LRU：

1. mean critical path 改善 `9.64 ms / 2.37%`；
2. P90/P95/P99 分别改善 `95.78/22.08/131.95 ms`；
3. P50 退化 `25.26 ms`；
4. H2D 增加 `11.01%`，PBP moved bytes 增加 `37.91%`。

因此 pipeline-value 在本 workload 上改善了平均和多数 tail latency，
但不是无代价的综合胜出：它显著增加数据移动和 allocator churn。该结论
只适用于 CPU/profile-driven、system-wide serial simulator；不代表真实
双 GPU 并发服务吞吐或 queueing latency。

结果文件：

```text
docs/tensor-level-sim/segment-group64-1000-2gpu-lru.json
docs/tensor-level-sim/segment-group64-1000-2gpu-pipeline-matched-lru.json
```
