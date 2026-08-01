# SGRP 四策略论文 Evaluation

## 1. 研究问题

比较四种逐步增强的策略组合：

| Label | Replacement | Routing | 目的 |
|---|---|---|---|
| CB-LRU | TensorGroup LRU | cache-bytes | 基线 |
| CB-SuffixLRU | prefix-preserving LRU | cache-bytes | 隔离prefix ABI收益 |
| CB-MCKP | Prefix-MCKP | cache-bytes | 隔离value-aware replacement收益 |
| Joint-MCKP | Prefix-MCKP | MCKP-transition | replacement与routing联合策略 |

本节以`exposed_load_ms`作为主指标，回答复杂策略是否在多数条件下减少无法被
Prefill掩盖的loading stall，以及收益如何随reuse distance、input length、
模型大小—热点相关性和GPU数量变化。TTFT保留为端到端辅助指标。

## 2. 实验方法

### 2.1 两阶段流程

先运行200-request快扫，不根据结果删除正式档位；随后将全部19个档位扩展到
1000 requests。每档5个matched trace seeds，每个seed的四策略使用相同trace
和availability mask。

正式矩阵：

```text
19 levels × 5 seeds × 4 strategies = 380 runs
```

另对ServeGen原始trace使用5个availability seeds运行20个锚点任务。

### 2.2 公共配置

```text
Requests                 1000
Warmup                   32 requests
Backend                  Segment request-level PBP
Pool                     42 GiB/GPU
Default GPUs             4
Default busy probability 0.5
Availability seed        1234, matched within each trace seed
Input scale              4
Output-token override    1
TensorGroup minimum      64 MiB
H2D                      24.56 GB/s
MCKP                     Lookahead K=64, discount=0.9
Transition weight        0.1
```

置信区间以每个seed先相对同seed CB-LRU归一化，再使用五seed均值和
`t(4, 0.975)=2.776`计算95% CI。

### 2.3 四个单维sweep

| Dimension | Levels | 固定条件 |
|---|---|---|
| Distinct reuse distance | 2.00, 3.00, 4.00, 5.00, 6.96 | balanced model counts，raw input=256 |
| Effective input tokens | 128, 256, 512, 1024 | balanced cyclic sequence |
| Size-hotness Spearman | -1, -0.5, 0, +0.5, +1 | matched popularity-rank sequence，raw input=256 |
| GPU count | 2, 3, 4, 6, 8 | balanced cyclic，exact available GPUs=1,2,2,3,4 |

GPU sweep使用`--available-gpus-per-request`精确控制每请求可用GPU数量，避免
Bernoulli busy在不同GPU数下产生不同的全忙概率。

## 3. 正式结果

### 3.1 四维Sensitivity

![SGRP exposed-loading sensitivity](../results/sgrp-paper-r1000/sensitivity-exposed-load.svg)

下表为Joint-MCKP相对CB-LRU的paired mean exposed loading stall改善：

| Dimension | Level | Improvement | 95% CI | Winning seeds |
|---|---:|---:|---:|---:|
| Reuse distance | 2.00 | +11.81% | ±3.90 | 5/5 |
| | 3.00 | +11.27% | ±4.68 | 5/5 |
| | 4.00 | +9.58% | ±2.93 | 5/5 |
| | 5.00 | +9.66% | ±3.74 | 5/5 |
| | 6.96 | +7.93% | ±5.36 | 5/5 |
| Effective input | 128 | +7.82% | ±2.32 | 5/5 |
| | 256 | +5.68% | ±2.02 | 5/5 |
| | 512 | +6.57% | ±1.80 | 5/5 |
| | 1024 | +6.08% | ±1.65 | 5/5 |
| Size-hotness rho | -1.0 | +15.06% | ±3.93 | 5/5 |
| | -0.5 | +11.51% | ±3.65 | 5/5 |
| | 0.0 | +9.15% | ±4.56 | 5/5 |
| | +0.5 | +9.35% | ±2.39 | 5/5 |
| | +1.0 | +9.53% | ±3.65 | 5/5 |
| GPU count | 2 | +9.55% | ±0.52 | 5/5 |
| | 3 | **+15.96%** | ±3.81 | 5/5 |
| | 4 | +12.31% | ±3.10 | 5/5 |
| | 6 | +7.87% | ±5.81 | 5/5 |
| | 8 | -1.48% | ±11.44 | 2/5 |

Joint-MCKP在18/19个档位的五seed平均值上优于LRU，在92/95个paired
seed-level条件中获胜；19个档位的中位改善为9.53%。唯一平均回退是8 GPU：
可用GPU增至4张后，routing/caching压力较低，额外transition routing没有稳定
收益。

### 3.2 策略增量贡献

跨19个档位汇总：

| Strategy | Positive level means | Paired wins | Median exposed-stall improvement | Range |
|---|---:|---:|---:|---:|
| CB-SuffixLRU | 19/19 | 92/95 | +2.93% | +1.05% to +3.59% |
| CB-MCKP | 19/19 | 91/95 | +7.02% | +0.48% to +14.33% |
| Joint-MCKP | 18/19 | 92/95 | **+9.53%** | -1.48% to +15.96% |

这给出清晰的逐步结论：prefix-preserving LRU提供稳定但较小的收益；MCKP
replacement贡献主要增益；transition routing在GPU资源受限时进一步提高收益，
但在8 GPU低压力点可能抵消replacement-only的小幅收益。

Joint-MCKP跨档位的中位指标变化：

```text
mean exposed loading    -9.55%
mean TTFT               -6.06%
H2D bytes               -5.76%
P95 TTFT                -20.82%
P99 TTFT                -0.07%
Segment compaction      +19.23%
CPU MCKP accounting      21.38 ms/request
```

P99几乎不变，因此论文结论应强调mean和P95，不应声称稳定改善P99。MCKP减少
H2D和暴露加载，但更细的保留状态会增加Segment relocation/compaction。
`mckp_solver_time_ms`是模拟主机CPU wall time，没有加入模拟TTFT。

### 3.3 代表性Workload Suite

![SGRP representative workload suite](../results/sgrp-paper-r1000/workload-suite-exposed-load.svg)

| Workload | SuffixLRU | CB-MCKP | Joint-MCKP |
|---|---:|---:|---:|
| ServeGen empirical | +4.21% | +11.46% | **+12.06%** |
| Balanced cyclic | +2.79% | **+6.65%** | +6.08% |
| Reuse D=2 | +2.96% | +7.02% | **+11.81%** |
| Reuse D=7 | +3.09% | +6.07% | **+7.93%** |
| Small models hot | +3.30% | +12.97% | **+15.06%** |
| Large models hot | +3.03% | +6.24% | **+9.53%** |

六类代表性workload中三种增强策略相对LRU均为正。Joint-MCKP在5/6类中
最好；balanced cyclic中CB-MCKP略优于Joint，说明复杂routing并非无条件有效。
ServeGen锚点收益较小但在五个availability seeds中均为正。

## 4. 可用于论文的结论

建议正文表述：

> Across 19 sensitivity settings and five matched seeds per setting,
> Joint-MCKP reduces exposed loading stall in 92 of 95 paired runs, with a
> median improvement of 9.5%. Prefix-preserving LRU provides a consistent
> 2.9% median gain, while MCKP replacement raises the median gain to 7.0%.
> Joint transition-aware routing is most beneficial with 2--4 GPUs, but its
> benefit vanishes in the low-pressure 8-GPU setting. These gains primarily
> come from 9.6% lower exposed loading and 5.8% lower H2D traffic.

边界必须同时报告：8 GPU exposed stall平均回退1.48%；Joint routing在
balanced cyclic中略逊于replacement-only；P99没有稳定改善；全部结果来自CPU/profile-driven
Segment模拟器，不是实际GPU端到端测量。

## 5. 复现

```bash
# 200-request trace与快扫
REQUESTS=200 \
OUTPUT_DIR=evaluation/traces/sgrp-paper-r200 \
  bash doc/implement/generate_sgrp_paper_traces.sh

python doc/implement/run_sgrp_paper_eval.py \
  --trace-dir evaluation/traces/sgrp-paper-r200 \
  --output-dir doc/results/sgrp-paper-r200 \
  --requests 200

# 1000-request正式矩阵
REQUESTS=1000 \
OUTPUT_DIR=evaluation/traces/sgrp-paper-r1000 \
  bash doc/implement/generate_sgrp_paper_traces.sh

python doc/implement/run_sgrp_paper_eval.py \
  --trace-dir evaluation/traces/sgrp-paper-r1000 \
  --output-dir doc/results/sgrp-paper-r1000 \
  --requests 1000

# ServeGen现实性锚点
python doc/implement/run_sgrp_paper_eval.py \
  --trace-dir evaluation/traces/sgrp-paper-r1000 \
  --output-dir doc/results/sgrp-paper-r1000 \
  --requests 1000 --dimensions workload

# 汇总与四维图
python tools/layerpipe/summarize_sgrp_paper_eval.py \
  --trace-dir evaluation/traces/sgrp-paper-r1000 \
  --result-dir doc/results/sgrp-paper-r1000 \
  --warmup-requests 32 \
  --csv-output doc/results/sgrp-paper-r1000/summary.csv \
  --aggregate-csv-output doc/results/sgrp-paper-r1000/aggregate.csv

python tools/layerpipe/plot_sgrp_paper_eval.py \
  --aggregate-csv doc/results/sgrp-paper-r1000/aggregate.csv \
  --metric exposed-load \
  --sensitivity-svg \
    doc/results/sgrp-paper-r1000/sensitivity-exposed-load.svg

python tools/layerpipe/summarize_sgrp_workload_suite.py \
  --aggregate-csv doc/results/sgrp-paper-r1000/aggregate.csv \
  --result-dir doc/results/sgrp-paper-r1000 \
  --warmup-requests 32 \
  --csv-output \
    doc/results/sgrp-paper-r1000/workload-suite-exposed-load.csv \
  --svg-output \
    doc/results/sgrp-paper-r1000/workload-suite-exposed-load.svg
```

## 6. Source of truth

```text
doc/results/sgrp-paper-r1000/summary.csv
doc/results/sgrp-paper-r1000/aggregate.csv
doc/results/sgrp-paper-r1000/workload-suite-exposed-load.csv
doc/results/sgrp-paper-r1000/sensitivity-exposed-load.svg
doc/results/sgrp-paper-r1000/sensitivity-ttft.svg
doc/results/sgrp-paper-r1000/workload-suite-exposed-load.svg
```
