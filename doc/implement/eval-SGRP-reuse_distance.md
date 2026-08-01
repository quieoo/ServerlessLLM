# SGRP：Distinct-model Reuse Distance 实现与实验

## 1. 目标与最终设置

本实验比较四种 Stall-Guided Reuse Policy（SGRP）组合：

| Label | Replacement | Routing |
|---|---|---|
| CB-LRU | `lru` | `cache-bytes` |
| CB-SuffixLRU | `lru-prefix` | `cache-bytes` |
| CB-MCKP | `mckp-prefix` | `cache-bytes` |
| Joint-MCKP | `mckp-prefix` | `mckp-transition` |

控制变量是 distinct-model reuse distance：

```text
模型 m 的两次相邻访问之间，出现过的不同模型数量
```

最终实验使用：

```text
8 models, each 125 requests
1000 total requests
4 GPUs
42 GiB/GPU
GPU busy probability = 0.5
availability seed = 1234
Segment request-level PBP
64-MiB minimum TensorGroup
K = 32, discount = 0.9, transition weight = 0.1
```

前一版 Preserve-ServeGen-skew、`gpu-busy-probability=0` 的结果仍保留在：

```text
evaluation/traces/sgrp-reuse-distance/
doc/results/sgrp-reuse-distance/
```

但它不是本文件的主结果。该版本前四个热点模型占 86.5%，mean distinct
只能从 2.345 扫到 3.033；同时 4×42 GiB 容量和全 GPU 可用使大部分请求
成为 full hit，mean exposed loading 过小。

## 2. Segment 模拟器执行效率修复

### 2.1 Profile

在修改前，对 300-request、4-GPU、MCKP-transition + MCKP-prefix 路径运行
`cProfile`：

```text
87,850,371 function calls
40.379 s cumulative profiled execution
```

主要热点：

| Hot path | Cumulative time |
|---|---:|
| `solve_covering_mckp()` | 21.452 s |
| generic `copy.deepcopy()` routing clone | 8.328 s |
| `_prune_frontier()` | 12.540 s（包含于 solver） |
| Segment `_reserve()` | 2.656 s |
| Segment `_free()` / `_coalesce()` | 1.937 / 1.904 s |

### 2.2 修改

修改文件：

```text
tools/layerpipe/tensor_sim_segment_backend.py
tools/layerpipe/tensor_sim_mckp.py
```

实现：

1. Segment 专用轻量 `clone()`：
   - 模型布局、args、stall table 和只读预测数据保持共享；
   - 只复制 extents、allocated index、访问计数、KV keys 和 pending loads；
   - 避免 routing preview 递归 deepcopy 全部模型/Tensor metadata。
2. MCKP frontier state：
   - 原实现每个扩展状态构造 `dict`；
   - tie-break 时重复执行 `sorted(choices.items())`；
   - 新实现按已排序 model 顺序维护 append-only tuple，保持相同确定性排序。
3. Prefix/suffix stall lookup cache：
   - key 包含 model、input tokens、state、field 和 extrapolation policy；
   - 避免 MCKP 对相同 profile 点重复插值。
4. Segment 批量 reclaim：
   - MCKP/LRU-prefix 一批 victim 全部释放后只 coalesce 一次；
   - `_reserve()` 和 `_merge_partition()` 用 extent identity 定位，避免
     dataclass equality 的线性比较开销。

### 2.3 等价性与性能

对修改前后同一 300-request 输入进行了逐请求比较：

```text
routing array                         identical
all non-timing request fields         identical
critical path / exposed load / bytes  identical
only mckp_solver_time_ms differs
```

修改后同一 cProfile：

```text
60,965,515 function calls
29.814 s cumulative profiled execution
```

相对修改前：

```text
profiled cumulative time: -26.2%
function calls:           -30.6%
generic deepcopy path:    removed from top profile
```

非 cProfile 的 300-request wall time在进一步 batch-coalesce 前从约 19.1 s
降到 17.4 s。实际完整 sweep 仍由 exact MCKP frontier solver 主导。

## 3. 均衡 controlled-distinct trace

### 3.1 生成器

扩展：

```text
evaluation/traces/generate_target_trace.py
```

新增：

```bash
--sequence controlled-distinct
--controlled-count-policy balanced
--controlled-objective min|max|target
--target-distinct-mean VALUE
--common-prefix-requests 32
--search-iterations N
--lookahead-k 32
```

生成器采用 count-preserving swap search：

1. 每个模型精确出现 125 次；
2. 冻结前 32 个请求，四档具有相同初始 cache history；
3. 禁止相邻同模型；
4. 四档使用完全相同的 per-model token multiset；
5. 优化 mean distinct-model distance；
6. 约束 request-index gap 与 `gap <= K`，避免 Lookahead 可见率成为主要
   混杂因素；
7. 序列搜索 RNG 和 token-shape RNG 分离。

每个 trace 的 `.trace.json` sidecar 保存 count、distance histogram、
request-gap、transition entropy 和 per-model token SHA-256。

### 3.2 四档

| Trace | Mean distinct | P50 | P90 | Gap P50 | Gap P90 | Gap <= K |
|---|---:|---:|---:|---:|---:|---:|
| `min` | 3.541 | 3 | 6 | 6 | 16 | 97.98% |
| `mid-low` | 4.009 | 4 | 7 | 6 | 15 | 98.99% |
| `mid-high` | 4.476 | 5 | 7 | 7 | 15 | 100.00% |
| `max` | 4.944 | 5 | 7 | 7 | 13 | 100.00% |

硬校验：

```text
requests == 1000
each model count == 125
first 32 model IDs matched
no consecutive repeated model
per-model token multiset SHA-256 matched
PASS
```

Trace 位于：

```text
evaluation/traces/sgrp-reuse-distance-balanced/
```

## 4. 实验配置

```text
Requests                  1000
Common prefix             first 32 requests
Reported requests         requests 32--999, n=968
GPUs                      4
GPU busy probability      0.5
Availability seed         1234
Pool                      42 GiB/GPU
Backend                   Segment request-level PBP
TensorGroup minimum       64 MiB
Allocator page            8 MiB
H2D                       24.56 GB/s
Input scale               4
Input limit               safe
Output override           1
Trace mode                all
MCKP prediction           lookahead
K                         32
Discount                  0.9
Transition weight         0.1
Offline table             offline-prefix-stall-runtime-group64.json
```

`gpu-busy-probability=0.5` 的 availability mask 由 request index 和固定 seed
生成，因此同一 trace 内四策略看到完全相同的候选 GPU 集合。

四组合必须通过默认 `system=suite` 和 `--policy-suite minimal-only` 运行。
suite JSON 的逐请求结果位于 `policies.minimal.requests`。

复现：

```bash
cd /mnt/n0/Tangram/Tangram
MAX_PARALLEL=4 bash doc/implement/run_eval_sgrp_reuse_distance.sh
```

汇总：

```bash
python tools/layerpipe/summarize_sgrp_reuse_distance.py \
  --trace-dir evaluation/traces/sgrp-reuse-distance-balanced \
  --result-dir doc/results/sgrp-reuse-distance-balanced-busy0p5 \
  --warmup-requests 32 \
  --csv-output \
    doc/results/sgrp-reuse-distance-balanced-busy0p5/summary.csv \
  --svg-output \
    doc/results/sgrp-reuse-distance-balanced-busy0p5/reuse-distance.svg
```

## 5. 结果

### 5.1 Mean TTFT

单位为 ms；括号内为相对同一 trace CB-LRU 的改善。

| Mean distinct | CB-LRU | CB-SuffixLRU | CB-MCKP | Joint-MCKP |
|---:|---:|---:|---:|---:|
| 3.541 | 618.90 | 610.39 (+1.38%) | 589.27 (+4.79%) | **583.98 (+5.64%)** |
| 4.009 | 632.35 | 621.72 (+1.68%) | 605.90 (+4.18%) | **598.31 (+5.38%)** |
| 4.476 | 645.51 | 633.68 (+1.83%) | 616.86 (+4.44%) | **604.13 (+6.41%)** |
| 4.944 | 672.64 | 662.89 (+1.45%) | 633.49 (+5.82%) | **621.44 (+7.61%)** |

随着 mean distinct distance 增大，四条策略的 mean TTFT 均总体上升。
Joint-MCKP 在四档均为最低。

### 5.2 Mean exposed loading

| Mean distinct | CB-LRU | CB-SuffixLRU | CB-MCKP | Joint-MCKP |
|---:|---:|---:|---:|---:|
| 3.541 | 283.41 | 274.90 | 253.78 | **248.49** |
| 4.009 | 296.86 | 286.23 | 270.41 | **262.81** |
| 4.476 | 310.01 | 298.18 | 281.37 | **268.63** |
| 4.944 | 337.15 | 327.40 | 297.99 | **285.95** |

这版不再出现“all-request mean exposed loading 只有几到几十毫秒”的低压
现象。CB-LRU 从 283.41 ms 单调增加到 337.15 ms；Joint-MCKP 从
248.49 ms 增加到 285.95 ms。

### 5.3 Tail TTFT

| Mean distinct | Strategy | P95 | P99 |
|---:|---|---:|---:|
| 3.541 | CB-LRU | 1661.9 | 1842.3 |
|  | CB-SuffixLRU | 1661.4 | 1842.3 |
|  | CB-MCKP | 1592.6 | 1851.5 |
|  | Joint-MCKP | **1555.0** | 1843.5 |
| 4.009 | CB-LRU | 1661.9 | 1934.4 |
|  | CB-SuffixLRU | 1661.2 | **1860.1** |
|  | CB-MCKP | **1579.9** | 1862.4 |
|  | Joint-MCKP | 1590.8 | 1936.5 |
| 4.476 | CB-LRU | 1662.1 | 1954.1 |
|  | CB-SuffixLRU | 1661.5 | **1880.3** |
|  | CB-MCKP | 1589.4 | 1908.9 |
|  | Joint-MCKP | **1553.5** | 1892.4 |
| 4.944 | CB-LRU | 1662.2 | 2078.6 |
|  | CB-SuffixLRU | 1661.8 | 2078.2 |
|  | CB-MCKP | 1565.6 | 2007.0 |
|  | Joint-MCKP | **1510.1** | **1965.3** |

Joint-MCKP 的 mean 和大多数 P95 最好，但不是每一档 P99 都最好。特别是
mid-low，CB-SuffixLRU 的 P99 优于 Joint-MCKP。论文中应分别陈述 mean、
P95 和 P99，不应把 mean 优势描述成所有尾延迟均改善。

### 5.4 Traffic

单位 GiB，总量对应 968 个 reported requests。

| Mean distinct | Strategy | H2D | Compaction moved | Evicted |
|---:|---|---:|---:|---:|
| 3.541 | CB-LRU | 8996.9 | 4654.6 | 8996.5 |
|  | CB-SuffixLRU | 9005.7 | **3492.6** | 9005.2 |
|  | CB-MCKP | 8692.4 | 5472.7 | 8691.8 |
|  | Joint-MCKP | **8468.9** | 5215.5 | **8468.7** |
| 4.009 | CB-LRU | 9399.7 | 5216.8 | 9400.5 |
|  | CB-SuffixLRU | 9397.5 | **4017.7** | 9398.3 |
|  | CB-MCKP | 9186.7 | 6139.7 | 9186.0 |
|  | Joint-MCKP | **9109.1** | 5607.2 | **9109.4** |
| 4.476 | CB-LRU | 9976.0 | 5066.1 | 9976.8 |
|  | CB-SuffixLRU | 9926.9 | **3818.4** | 9927.8 |
|  | CB-MCKP | 9831.6 | 6072.9 | 9830.7 |
|  | Joint-MCKP | **9447.4** | 5687.9 | **9447.7** |
| 4.944 | CB-LRU | 10883.7 | 5368.6 | 10885.3 |
|  | CB-SuffixLRU | 10880.7 | **4192.8** | 10882.3 |
|  | CB-MCKP | 10433.8 | 5994.7 | 10434.1 |
|  | Joint-MCKP | **10065.7** | 5857.0 | **10067.0** |

Suffix-LRU 的主要收益来自显著减少 Segment compaction，而不是减少 H2D。
MCKP 两组减少 H2D/eviction，但当前 plan 会形成更多 Segment relocation；
Joint-MCKP 通过 routing 进一步减少 H2D。

### 5.5 Hit/stall incidence

| Mean distinct | Strategy | H2D-free | Nonzero exposed | Conditional exposed |
|---:|---|---:|---:|---:|
| 3.541 | CB-LRU | 40.3% | 59.7% | 474.6 ms |
|  | CB-SuffixLRU | 40.1% | 57.0% | 482.1 ms |
|  | CB-MCKP | 25.8% | 69.6% | 364.5 ms |
|  | Joint-MCKP | 27.1% | 68.8% | 361.2 ms |
| 4.009 | CB-LRU | 34.5% | 65.5% | 453.2 ms |
|  | CB-SuffixLRU | 34.6% | 62.8% | 455.7 ms |
|  | CB-MCKP | 21.5% | 75.1% | 360.1 ms |
|  | Joint-MCKP | 21.9% | 73.9% | 355.8 ms |
| 4.476 | CB-LRU | 32.4% | 67.6% | 458.9 ms |
|  | CB-SuffixLRU | 32.7% | 64.9% | 459.6 ms |
|  | CB-MCKP | 19.2% | 77.5% | 363.2 ms |
|  | Joint-MCKP | 20.6% | 76.9% | 349.5 ms |
| 4.944 | CB-LRU | 30.2% | 69.8% | 482.8 ms |
|  | CB-SuffixLRU | 30.4% | 67.7% | 483.8 ms |
|  | CB-MCKP | 16.9% | 81.0% | 367.9 ms |
|  | Joint-MCKP | 18.1% | 78.6% | 363.7 ms |

MCKP 不一定提高“完全无 H2D”的请求比例。它保留有 pipeline stall value
的 prefix，使更多请求发生较小或可重叠的 partial load；因此评价 MCKP
不能只看 full-hit ratio，应同时看 exposed stall 和 H2D traffic。

主图：

![Balanced reuse-distance results](../results/sgrp-reuse-distance-balanced-busy0p5/reuse-distance.svg)

机器可读结果：

```text
doc/results/sgrp-reuse-distance-balanced-busy0p5/summary.csv
```

## 6. 结论

1. 均衡分布将可控 mean distinct distance 从原来的 `2.345--3.033` 扩展到
   `3.541--4.944`。
2. `gpu-busy-probability=0.5` 打破了 4×42 GiB 下接近静态模型分区的低压
   状态，CB-LRU mean exposed loading 达到 `283--337 ms`。
3. distinct distance 增大时，所有策略 mean TTFT 和 mean exposed loading
   总体上升，说明新的 sweep 确实形成了更强 cache competition。
4. CB-SuffixLRU 稳定改善 mean TTFT `1.38%--1.83%`，主要减少 Segment
   compaction。
5. CB-MCKP 改善 `4.18%--5.82%`；Joint-MCKP 改善
   `5.38%--7.61%`，四档 mean 均为最好。
6. Joint-MCKP 的 P99 不在所有档位最好，因此结论必须区分 mean 与 tail。

## 7. 限制

本轮仍是单个 trace-generation seed。availability mask 在策略间 matched，
但不同 distinct 档的 model ID 会落到相同 index mask 的不同位置，因此
trace ordering 与 busy mask 存在交互。

论文最终结果建议：

1. 每个 distinct 目标生成 5 个 trace seed；
2. 每个 seed 使用多个 availability seed，或预生成并报告 mask；
3. 对 seed 做 paired bootstrap；
4. 继续报告 mean/P95/P99、H2D、compaction、hit incidence；
5. 增加 matched routing replay，隔离 replacement-only effect。

本轮应标记为：

```text
controlled balanced simulator pilot, one trace seed,
one matched availability seed
```

## 8. 验证

```text
47 Segment/MCKP unit tests passed
optimized vs pre-optimization routing identical
optimized vs pre-optimization non-timing request fields identical
4/4 balanced trace invariant checks passed
16/16 simulator combinations completed
summary/result validation passed
py_compile passed
bash syntax check passed
git diff --check passed
```

## 9. 200-request 宽距离 × Pool 二维快扫

### 9.1 修改与口径

后续实验采用两阶段流程：

1. 先用 200 requests 快扫参数空间；
2. 只把候选点扩展到 1000 requests、多 trace seed 和完整四策略。

本轮在生成器中增加：

```bash
--controlled-gap-constraint none
--controlled-initialization max-distance
```

第一项取消 request-gap P50/P90 和 `gap <= K` 对 distinct-distance 搜索的
惩罚；第二项保留相同前缀后，以 least-recently-seen 顺序构造最大距离
tail。MCKP Lookahead 从 `K=32` 增加到 `K=64`。

六档 trace 均包含 200 requests、8 个模型各 25 次，前 16 个请求一致，
实际 mean distinct distance 为：

```text
2.010, 3.000, 4.000, 5.000, 5.927, 6.750
```

各档 `request_gap <= 64` 比例为 `98.96%--100%`。request gap 不再作为
生成约束，但仍记录在 sidecar 和 summary 中。

容量不能使用原计划的 `24--36 GiB`：最大模型的权重 footprint 为
`38.286 GiB`，Segment 后端要求单个请求至少可容纳其完整模型和 KV。
因此实际扫描：

```text
Pool/GPU = 39, 40, 42, 46, 50 GiB
```

快扫只运行 CB-LRU 与 Joint-MCKP，共 `6 × 5 × 2 = 60` 个任务。其他设置：

```text
Requests=200, warmup=16, GPUs=4, busy probability=0.5
Segment, group=64 MiB, K=64, availability seed=1234
```

运行脚本根据 CPU affinity 内的 physical core 数和
`MemAvailable / 2 GiB` 取较小值作为默认并发数。本机为 112 logical CPU、
56 physical cores，后续默认最大并发为 56；可用 `MAX_PARALLEL` 覆盖。

```bash
bash doc/implement/run_eval_sgrp_reuse_distance_fast200_2d.sh

python tools/layerpipe/summarize_sgrp_reuse_distance_2d.py \
  --trace-dir evaluation/traces/sgrp-reuse-distance-fast200-wide \
  --result-dir doc/results/sgrp-reuse-distance-fast200-2d-k64 \
  --warmup-requests 16 \
  --csv-output \
    doc/results/sgrp-reuse-distance-fast200-2d-k64/summary.csv
```

### 9.2 快扫结果

表中是 Joint-MCKP 相对同点 CB-LRU 的 mean TTFT 改善：

| Mean distinct | 39 GiB | 40 GiB | 42 GiB | 46 GiB | 50 GiB |
|---:|---:|---:|---:|---:|---:|
| 2.010 | +3.63% | +3.51% | +3.04% | +3.06% | +2.98% |
| 3.000 | **+10.89%** | +8.36% | +6.61% | +5.15% | +8.30% |
| 4.000 | -0.16% | +2.46% | +0.80% | +2.68% | +3.15% |
| 5.000 | +7.42% | **+10.78%** | +8.06% | +7.94% | +7.20% |
| 5.927 | +4.82% | +6.21% | +4.20% | +5.48% | +4.87% |
| 6.750 | -4.40% | +3.65% | +3.60% | +1.95% | +3.89% |

30 个点的平均改善为 `4.67%`。最值得进入确认实验的候选点：

```text
D=3.000, pool=39 GiB: +10.89%
D=5.000, pool=40 GiB: +10.78%
```

还应加入两个边界点：

```text
D=4.000, pool=39 GiB: -0.16%
D=6.750, pool=39 GiB: -4.40%
```

这些结果不随 distance 或 pool 单调变化，说明 200-request 单 seed 中
trace ordering、availability mask 和冷启动仍有明显影响。因此本节只用于
候选点筛选，不能直接作为论文最终结论。下一阶段应对上述候选和边界点运行
1000 requests、全部四策略和多个 trace/availability seeds。

### 9.3 产物与验证

```text
evaluation/traces/sgrp-reuse-distance-fast200-wide/
doc/results/sgrp-reuse-distance-fast200-2d-k64/
doc/results/sgrp-reuse-distance-fast200-2d-k64/summary.csv
```

验证：

```text
60/60 simulator tasks completed
47 Segment/MCKP unit tests passed
generator and summarizer py_compile passed
runner bash syntax passed
```

## 10. 200-request 模型大小—热点相关性快扫

### 10.1 设计

固定 ServeGen 原始热点 share，经 largest-remainder 缩放为 200 requests：

```text
popularity-rank counts = 58, 42, 40, 35, 11, 9, 3, 2
```

每个 seed 只生成一次 popularity-rank sequence，五档通过置换
`popularity rank -> model ID` 改变模型大小与热点的相关性。五个 trace seeds 为
`1234--1238`，每个档位运行 CB-LRU 与 Joint-MCKP，共 50 个任务。

| Level | Spearman(size, hotness) | Pearson(bytes, share) |
|---|---:|---:|
| `neg1` | -1.0 | -0.884 |
| `neg0p5` | -0.5 | -0.510 |
| `zero` | 0.0 | +0.086 |
| `pos0p5` | +0.5 | +0.436 |
| `pos1` | +1.0 | +0.943 |

其余设置为 200 requests、warmup 16、4 GPU、42 GiB/GPU、busy probability
0.5、K=64。运行器识别 112 logical CPU / 56 physical cores，并使用最多
56 个并发进程。

```bash
python evaluation/traces/generate_size_hotness_correlation_traces.py \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --source evaluation/traces/servegen_tangram.trace \
  --output-dir evaluation/traces/sgrp-size-hotness-fast200 \
  --requests 200 --source-requests 1000 \
  --seeds 1234,1235,1236,1237,1238 --lookahead-k 64

bash doc/implement/run_eval_sgrp_size_hotness_fast200.sh
```

### 10.2 五 seed 汇总

改善为每个 seed 中 Joint-MCKP 相对 CB-LRU 的 paired mean TTFT 改善，再对
五个 seed 求均值和标准差。

| Spearman | LRU TTFT | Joint TTFT | 改善 | Exposed LRU/Joint | H2D LRU/Joint |
|---:|---:|---:|---:|---:|---:|
| -1.0 | 367.1 | 355.1 | +3.21% ± 1.85 | 80.1 / 68.2 | 528.7 / 479.0 |
| -0.5 | 386.9 | 370.4 | +4.28% ± 2.26 | 146.4 / 130.0 | 870.1 / 819.2 |
| 0.0 | 544.9 | 517.7 | +4.91% ± 2.81 | 243.1 / 215.9 | 1430.1 / 1348.7 |
| +0.5 | 609.9 | 584.2 | +4.03% ± 5.45 | 288.4 / 262.7 | 1704.3 / 1639.3 |
| +1.0 | 767.4 | 720.4 | **+6.11% ± 1.46** | 379.5 / 332.5 | 2176.1 / 2036.1 |

单位分别为 ms、ms 和 GiB。强正相关的五个 seed 均改善，范围为
`4.25%--8.28%`。`pos0p5` 方差最大，范围 `-2.25%--10.30%`。

快扫结果否定了“大模型冷时差距一定最大”的初始猜测。当前工作负载中，
大模型越热，absolute loading pressure 越高；Joint-MCKP 通过容量价值选择与
transition-aware routing 减少的 exposed loading 和 H2D 也更多。强正相关是
最值得进入 1000-request 完整四策略实验的候选点，零相关可作为中性对照，
强负相关可作为低压力边界。

本实验仍有两个限制：所有 trace seed 共用一个 availability seed；不同模型
本身的 input-token 分布不同，因此置换热点会同步改变 aggregate input mix。
第二点符合“哪个模型变热”的系统语义，但论文中应另做 fixed-input-length
robustness check，避免将输入长度差异误归因为大小—热点相关性。

结果：

```text
doc/results/sgrp-size-hotness-fast200-k64/summary.csv
doc/results/sgrp-size-hotness-fast200-k64/aggregate.csv
```
