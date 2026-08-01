# Prefix-MCKP Cache Replacement 与 2-GPU Routing 实现规划

本文档规划 Tensor-level LayerWeave 模拟器中的新一版缓存替换与路由策略。
第一阶段只修改 CPU/profile-driven simulator，实验范围限定为：

```text
memory layout       = segment
Segment allocator   = request-level PBP
TensorGroup minimum = 64 MiB（允许通过 CLI 修改）
GPUs                = 2
trace mode           = model-switches
```

现有 `lru`、`frequency-lru` 和 `pipeline-value` 保持不变，作为 matched
baseline。新策略使用独立入口，避免改变已有结果的语义。

## 1. 目标

新策略不再给每个 TensorGroup 独立打分并逐个贪心淘汰，而是：

1. 离线构建每个模型的
   `input_length × resident_prefix -> exposed_loading_stall` 模型；
2. 强制每个模型的缓存状态为 TensorGroup 前缀；
3. 对每个模型枚举“从后向前释放到哪个前缀”的候选配置；
4. 根据需要释放的空间求解 multiple-choice minimum-cost covering
   knapsack；
5. 在 2-GPU 路由时联合考虑当前请求 loading stall 和未来 eviction
   damage。

首版策略名建议为：

```text
--replacement-policy mckp-prefix
--routing-policy mckp-transition
```

## 2. 精确语义

### 2.1 TensorGroup 顺序和前缀状态

模型 `m` 的 TensorGroup 按实际 Tensor compute/loading 顺序编号：

```text
TG[0], TG[1], ..., TG[n-1]
```

缓存配置 `c` 表示：

```text
resident = TG[0:c]
absent   = TG[c:n]
```

因此：

- `c=0`：模型权重完全不驻留；
- `c=n`：模型权重完全驻留；
- eviction 只能将 `c` 减小，即从后向前释放；
- loading 当前请求时仍按原 Tensor pipeline 顺序加载所有 missing group；
- 当前请求结束后模型恢复为完整驻留，之后才可能被其他请求缩短前缀。

MCKP 模式必须增加 prefix invariant。对于每个模型，resident group ID
必须严格等于 `[0, c)`。发现中间空洞时直接报错，不静默修复。

### 2.2 Offline stall table

定义：

```text
stall[m][input_bucket][c]
```

它表示模型 `m` 在输入长度落入 `input_bucket`、当前保留前 `c` 个
TensorGroup 时，请求的预测 `exposed_load_ms`。

该表必须通过当前 Tensor pipeline timeline 计算整个请求，而不是把各
TensorGroup 的局部 ready stall 相加。原因是 loading 和 compute overlap
非线性，多个 group 的边际 stall 不可直接相加。

表的基本不变量：

```text
stall[m][l][n] = 0
stall[m][l][c] >= 0
stall[m][l][c+1] <= stall[m][l][c]
```

浮点误差或 profile 噪声可能轻微破坏单调性。构表后应从完整驻留端做
单调包络，保证更多缓存不会得到更高 stall。

### 2.3 Input-length bucket

首版支持两种 bucket 来源：

```text
--mckp-input-buckets trace-quantiles
--mckp-input-buckets explicit
```

建议默认对每个模型分别取历史 trace 的：

```text
min / P25 / P50 / P75 / P90 / max
```

并保存真实 bucket token 值。查询任意输入长度时先使用线性插值；超出表
范围时 clamp 到首尾 bucket，不外推。

为了验证表本身，保留一个直接调用 Tensor timeline 的 reference path。
单元测试和小 trace 必须比较 table lookup 与 reference 结果。

### 2.4 一个释放选项的空间

模型 `m` 当前保留前 `r_m` 个 group。选择新配置 `c <= r_m` 时：

```text
freed_bytes(m, c)
    = prefix_bytes(m, r_m) - prefix_bytes(m, c)
```

候选配置至少包括：

```text
c = r_m        # 不释放
c = r_m - 1
...
c = 0          # 全部释放
```

如果 group 数量导致 planner 开销过高，可以增加显式配置步长，但默认
64 MiB TensorGroup 实验应保留全部候选，先得到无配置采样误差的结果。

### 2.5 一个释放选项的代价

释放代价必须使用相对当前缓存状态增加的 future stall，不能使用释放后
的绝对 stall：

```text
delta_stall(m, l, c | r_m)
    = stall(m, l, c) - stall(m, l, r_m)
```

在线历史估计版本：

```text
cost(m, c)
    = P_history(m)
      * E_history_l[delta_stall(m, l, c | r_m)]
```

这里：

- `P_history(m)` 使用 exponentially-decayed demand；
- 输入长度默认使用该模型历史窗口的中位数；
- 历史不足时回退到该模型离线 trace 中位数；
- 当前请求只更新完 cache/demand 后才进入后续请求的历史，不能泄漏未来。

未来 K 步 Lookahead 版本：

```text
cost_K(m, c)
    = sum(
        discount^j
        * delta_stall(m, input_tokens[j], c | r_m)
        for j in [1, K]
        if future_request[j].model_id == m
      )
```

Lookahead 直接读取排队的 `(model_id, input_tokens)`，不等待未来请求、
不重新运行 future-state simulator。每个候选配置只做 table lookup 和
加法。

### 2.6 需要释放的空间

对候选 GPU `g` 和当前请求 `(m, l)`，先计算：

```text
missing_weight_bytes
    = 当前模型所有 absent TensorGroup 的 logical bytes

required_bytes
    = missing_weight_bytes
      + request_kv_bytes
      + optional_runtime_reserve_bytes

S = max(0, required_bytes - current_free_bytes)
```

当前 Segment backend 在 `begin_request()` 中先释放上一请求 KV，再为本
请求规划权重和 KV。MCKP 的 `current_free_bytes` 必须在旧 KV 已释放、
新 KV 尚未分配的相同状态下读取。

当前请求所属模型：

- 已驻留前缀完全保护，不作为 MCKP victim；
- 它的 missing suffix 计入 `missing_weight_bytes`；
- 如果保护当前前缀后所有其他模型全部释放仍不足，应返回 infeasible；
- whole-model weight + KV 超过 pool 的情况继续由现有 feasibility
  validation 提前拒绝。

### 2.7 MCKP 定义

每个非 active 模型是一类，每类必须选择一个最终前缀配置：

```text
minimize:
    sum_m cost(m, c_m)

subject to:
    sum_m freed_bytes(m, c_m) >= S
    c_m in {0, ..., r_m}
```

这是 minimum-cost covering 形式，而不是“容量不超过上限、价值最大”的
普通背包。

确定性 tie-break 顺序：

1. 最小 future stall cost；
2. 最小 over-release bytes；
3. 最少 evicted TensorGroup 数；
4. 按 model ID 和 prefix length 做稳定字典序。

输出计划：

```text
MckpEvictionPlan:
    required_free_bytes
    planned_free_bytes
    predicted_future_damage_ms
    choices[model_id] = final_prefix_count
    victim_keys = [(model_id, group_id), ...]
    solver_time_ms
    frontier_peak_states
```

## 3. 求解器设计

### 3.1 首版：Pareto frontier DP

仓库只有 8 个模型，没有必要引入外部整数规划依赖。建议新建：

```text
tools/layerpipe/tensor_sim_mckp.py
```

实现流程：

1. 为每个非 active 模型生成配置选项；
2. 逐模型合并当前 frontier 与该模型全部选项；
3. 将 `freed_bytes` clamp 到 `S`，因为超过 `S` 的额外空间不增加可行性；
4. 删除 dominated state；
5. 保存 backpointer，最终恢复每个模型的选择。

Dominance：

```text
state A dominates B if:
    A.freed_bytes >= B.freed_bytes
    A.cost <= B.cost

and at least one strict
```

由于最终还需要最小 over-release，不能只把所有 `freed >= S` 状态无条件
合并成一个。实现上可以：

- 对 `freed < S` 保留 Pareto frontier；
- 对 `freed >= S` 单独维护按完整 tie-break 最优的 feasible state。

### 3.2 Reference solver

测试中实现小规模穷举 reference solver：

```text
models <= 4
options/model <= 5
```

随机生成空间和代价，验证 frontier DP 与穷举在 cost、over-release 和
最终配置上完全一致。

### 3.3 性能保护

首版先记录而不默认近似：

```text
solver_time_ms
frontier_states_before_prune
frontier_states_after_prune
frontier_peak_states
options_considered
```

如果 704/1000-request 实验的 P99 planner time 不可接受，再增加显式参数：

```text
--mckp-capacity-quantum-mib
--mckp-frontier-limit
```

任何量化或 beam pruning 都必须在结果 metadata 中标为 approximate，
并与 exact small-workload 结果报告 objective gap。

## 4. 代码改动规划

### 4.1 `tensor_sim_mckp.py`

新增纯 CPU、无 backend 副作用的类型和函数：

```text
PrefixStallTable
PrefixOption
MckpState
MckpEvictionPlan

build_prefix_stall_table(...)
lookup_stall(...)
build_model_options(...)
solve_covering_mckp(...)
```

该文件不直接访问 Segment extents，便于单测。

### 4.2 `tensor_sim_memory_common.py`

为 backend 增加显式 planner 输入，而不是继续复用对象 `_value()`：

```text
set_prefix_stall_table(table)
set_request_prediction_context(context)
resident_prefix_count(model_id)
preview_reclamation(active_model, required_bytes)
commit_reclamation(plan)
```

保留现有 `_select_victim()` 给 LRU/frequency/pipeline-value 使用。
`mckp-prefix` 不经过逐对象 `_select_victim()`。

### 4.3 `tensor_sim_segment_backend.py`

在 `MockAllocationSegmentMemory.plan_request()` 中：

1. 收集当前请求 missing groups；
2. 计算权重、KV 和 runtime reserve 的总空间；
3. 若空间不足，调用 `preview_reclamation()`；
4. 一次性按 plan 释放每个模型的后缀；
5. 验证实际释放空间不小于计划；
6. 再执行现有 request-level PBP partition/merge/reserve；
7. 分配 KV blocks；
8. 将 MCKP 和 PBP 成本分别写入请求指标。

必须避免以下错误：

- MCKP 已选 victim 后，PBP 又通过 `_evict_one()` 选择另一套 victim；
- routing probe 调用时修改真实 backend；
- 当前模型的 prefix 被当作 victim；
- 释放 group 后没有更新 `allocated_extents`、LRU clock 或 prefix state；
- PBP 因连续 extent 不足需要 relocation，却把 relocation bytes 算成
  eviction damage。

MCKP 负责“释放哪些缓存”；PBP 继续负责“怎样放置 missing groups”。
两者职责不能混合。

### 4.4 `layerweave_tensor_pipeline_sim.py`

新增 CLI：

```text
--replacement-policy mckp-prefix
--routing-policy mckp-transition
--mckp-prediction-mode history
--mckp-prediction-mode lookahead
--mckp-lookahead-k 8
--mckp-lookahead-discount 1.0
--mckp-transition-weight 1.0
--mckp-history-window 32
--mckp-runtime-reserve-mib 0
--mckp-stall-table-input PATH
--mckp-stall-table-output PATH
```

初始化阶段：

1. 加载或构建 prefix stall table；
2. 校验 table 的 layout、group size、H2D/profile fingerprint；
3. 为每个 GPU backend 注入同一个只读 table；
4. 预构建每个 GPU 的 future request index，Lookahead 查询保持 O(K)。

每个请求开始前：

1. 仅用之前请求更新 history demand/input statistics；
2. 构造 history 或 Lookahead prediction context；
3. 将相同 context 注入各 GPU cache；
4. 分别 clone backend 做候选计划；
5. 选择 GPU；
6. 只在选中 GPU 的真实 backend 上重新规划并执行；
7. 请求结束后更新历史统计。

为了检查 preview/commit 一致性，debug 模式可断言选中 GPU 的 probe plan
与真实执行 plan 相同。

## 5. 路由设计

### 5.1 候选 GPU 分数

对 GPU `g`：

```text
current_stall_g
    = stall[current_model][current_input][resident_prefix_g]

eviction_damage_g
    = solve_covering_mckp(g, S_g).predicted_future_damage_ms

routing_score_g
    = current_stall_g
      + transition_weight * eviction_damage_g
```

建议 `--mckp-transition-weight 1.0` 作为首个实验点，但它必须保留为显式
参数，因为：

- `current_stall` 是当前请求立即付出的 latency；
- `eviction_damage` 是未来请求的期望 latency；
- 二者虽然都用 ms 表示，但时间范围不同。

路由 tie-break 沿用 round-robin。

### 5.2 当前请求真实 critical path

`routing_score` 不能替代请求 latency 指标：

```text
critical_path_ms
    = hot_compute_ms + actual_exposed_load_ms
```

MCKP future damage 仅用于决策和单独统计，不能加进当前请求的
`critical_path_ms`。

### 5.3 Preview 输出

每个 GPU 候选至少记录：

```text
gpu
resident_prefix_count
current_stall_ms
required_free_bytes
planned_free_bytes
predicted_eviction_damage_ms
routing_score_ms
solver_time_ms
feasible
```

`--inspect-request-id` 应保存两个 GPU 的完整 candidate plan 和最终选中
计划。

## 6. 在线与 Lookahead 两个版本

### 6.1 History 模式

严格 answer-blind、future-blind：

```text
model probability = decayed historical demand
input estimate     = per-model rolling median
```

初始状态建议使用：

```text
每个模型 demand = 1
input estimate = offline trace median
```

需要明确 history 是按源 trace 请求还是过滤后的 model-switch 请求更新。
本模拟器默认评估 model-switch latency，因此首版应按模拟请求更新，并在
metadata 中写明。

### 6.2 Lookahead 模式

只读取目标 GPU 后续队列中的：

```text
(model_id, input_tokens)
```

当前 simulator 是 system-wide serial order，没有真实 per-GPU queue。
首版应提供两个清晰口径：

```text
global-next-K:
    当前请求之后的全局 K 个请求

routed-next-K:
    matched routing 已知时，该 GPU 后续 K 个请求
```

Adaptive routing 下未来请求尚未分配 GPU，不能声称知道
`routed-next-K`。因此建议实验顺序：

1. 用 history MCKP adaptive routing 生成 routing；
2. 在相同 routing 上做 history vs routed-next-K replacement；
3. global-next-K 仅作为 diagnostic oracle；
4. 最后再评估 adaptive global-next-K routing，并明确其 oracle 性质。

## 7. Stall table 文件格式

建议 JSON：

```json
{
  "format": "layerweave-prefix-stall-v1",
  "metadata": {
    "memory_layout": "segment",
    "tensor_group_min_bytes": 67108864,
    "h2d_gbps": 24.56,
    "profile_source": "...",
    "tensor_layout_source": "...",
    "compute_profile_source": "...",
    "stall_semantics": "whole-request exposed_load_ms"
  },
  "models": {
    "0": {
      "group_count": 123,
      "prefix_bytes": [0, 67108864],
      "input_tokens": [32, 128, 512],
      "stall_ms": [
        [123.0, 120.0],
        [130.0, 126.0],
        [145.0, 138.0]
      ]
    }
  }
}
```

真实数组长度必须满足：

```text
len(prefix_bytes) = group_count + 1
len(stall_ms[row]) = group_count + 1
```

加载时必须验证 fingerprint，防止用 64 MiB group 的表运行 32 MiB group，
或用不同 H2D/profile 的 table。

## 8. 指标

每请求新增：

```text
mckp_required_free_bytes
mckp_planned_free_bytes
mckp_overrelease_bytes
mckp_predicted_damage_ms
mckp_solver_time_ms
mckp_frontier_peak_states
mckp_evicted_groups
mckp_evicted_models
mckp_prefix_before
mckp_prefix_after
routing_score_ms
routing_current_stall_ms
routing_eviction_damage_ms
```

汇总报告 mean/P50/P90/P95/P99/max：

```text
critical_path_ms
exposed_load_ms
mckp_predicted_damage_ms
mckp_solver_time_ms
mckp_overrelease_bytes
```

并继续报告：

```text
H2D bytes
evicted bytes/groups
PBP moved bytes
PBP relocation latency
full/partial/not-resident request ratio
partial resident fraction
per-GPU request/model distribution
```

需要新增 prediction calibration：

```text
predicted future damage
vs
后续 K 步因本次 eviction 实际增加的 stall
```

该指标只用于离线诊断，不能在在线决策中读取未来。

## 9. 测试计划

### 9.1 Stall table

- full-resident prefix stall 为 0；
- retained prefix 增长时 stall 单调不增；
- table bucket 精确点与 reference timeline 相等；
- 插值和 clamp 正确；
- group size/profile fingerprint 不匹配时拒绝加载；
- 整个 prefix 的 stall 不等于简单 group value 求和的构造用例。

### 9.2 Prefix state

- empty、partial、full 三种 prefix 识别；
- suffix release 后仍满足 prefix invariant；
- 中间缺 group 时立即失败；
- active model prefix 永不淘汰；
- plan 释放字节数等于实际释放字节数。

### 9.3 MCKP solver

- `S=0` 时所有模型选择当前 prefix；
- 单模型选择正确释放深度；
- 多模型时选择联合最小代价；
- 允许 over-release，但 tie-break 选择最小 over-release；
- 空间不可满足时返回 infeasible；
- zero-cost option；
- 相同 cost 的确定性 tie-break；
- 随机小问题与穷举 reference 完全一致。

一个必要的反贪心测试：

```text
逐 group 最小 value/byte 的选择不是全局最优，
但模型级 MCKP 能找到正确组合。
```

### 9.4 Segment/PBP 集成

- MCKP victim 集合与 PBP 实际 victim 集合一致；
- PBP 只 relocation，不额外改变保留前缀；
- KV 空间计入 `S`；
- request 完成后 active model 全部 group resident；
- clone preview 不修改原 backend；
- preview plan 与 commit plan 一致；
- LRU/frequency/pipeline-value 旧测试和结果语义不变。

### 9.5 Routing

- 当前 stall 更低且 eviction damage 相同的 GPU 被选择；
- current stall 相同但 eviction damage 更低的 GPU 被选择；
- `transition_weight=0` 退化为只看当前 stall；
- score 相同按 round-robin；
- infeasible GPU 不参与选择；
- 两个 GPU 都 infeasible 时输出清晰错误；
- matched routing replay 数组逐项完全一致。

## 10. 实施阶段

### M0：冻结语义和 baseline

交付：

- 保存当前 2-GPU pure LRU、frequency-LRU、pipeline-value JSON；
- 记录代码版本、trace、profile 和 CLI；
- 确认现有 20 个 CPU tests 全部通过。

完成标准：

- 后续实现不能静默改变现有策略结果；
- 新 MCKP 只通过新 CLI 激活。

### M1：Prefix stall table

交付：

- `tensor_sim_mckp.py` 的 table 类型、构建和查询；
- stall table 导出/加载 CLI；
- 单调性、插值、fingerprint 测试。

完成标准：

- 8 模型、64 MiB group 的 table 可离线生成；
- exact bucket 上与直接 timeline 的误差小于 `1e-9 ms`；
- table 构建不进入 per-request controller time。

### M2：Prefix ABI 与 MCKP solver

交付：

- resident prefix 查询和 invariant；
- 模型配置生成；
- Pareto frontier DP；
- 穷举 reference 测试。

完成标准：

- 随机小问题与穷举 100% 一致；
- solver 返回确定性 plan；
- 尚不接入真实 eviction。

### M3：Segment eviction 集成

交付：

- `mckp-prefix` replacement；
- plan/commit；
- 与 request-level PBP 集成；
- 请求级 MCKP 指标。

完成标准：

- 不发生非后缀 eviction；
- MCKP 计划空间与实际空间一致；
- PBP 不选择计划外 victim；
- 21-request smoke 无 invariant/OOM 错误。

### M4：History-aware 2-GPU routing

交付：

- `mckp-transition` routing；
- decayed demand + rolling median input；
- 两 GPU candidate inspection；
- transition weight。

完成标准：

- probe 无副作用；
- routing score 分解可复算；
- 21-request 和 704-request 均完成；
- 报告 controller P50/P90/P95/P99/max。

### M5：Lookahead K

交付：

- O(K) future request index；
- global-next-K；
- matched routing 下 routed-next-K；
- K/discount sweep。

完成标准：

- queue read 与 MCKP solve time 分开统计；
- 不等待 K 个请求；
- 不对未来每个候选重新运行 Tensor pipeline；
- K=0 或空窗口明确退化到 history/default 行为。

### M6：Matched evaluation

至少运行：

```text
LRU
frequency-LRU
旧 pipeline-value
MCKP history
MCKP Lookahead K=4/8/16/32
```

公平性控制：

- 相同 trace、source-row cap 和 model-switch 过滤；
- 相同 Tensor layout、group size、pool、KV、H2D 和 compute profile；
- replacement 对比先固定 routing；
- adaptive routing 作为单独实验；
- `request_cap_pool` 或等价输入截断控制固定；
- 不混用不同 stall table fingerprint。

必须报告：

```text
critical path: mean/P50/P90/P95/P99/max
exposed load:  mean/P50/P90/P95/P99/max
H2D / eviction / PBP moved bytes
prefix residency
MCKP solver/controller latency
per-GPU routing distribution
```

## 11. 推荐实现顺序

按以下顺序修改，避免同时调试 profile、solver、allocator 和 routing：

```text
离线 stall table
    -> prefix invariant
    -> 纯函数 MCKP solver
    -> 单 GPU fixed-routing eviction
    -> 2 GPU matched-routing replacement
    -> history adaptive routing
    -> Lookahead replacement
    -> Lookahead adaptive routing
```

第一轮只做：

```text
Segment + 64 MiB group + history MCKP + fixed routing
```

验证 replacement 本身后，再启用 adaptive routing。这样出现性能变化时，
可以区分：

- stall model 是否准确；
- MCKP 是否选错配置；
- Segment/PBP 是否产生额外 relocation；
- routing/cache feedback 是否放大热点吸附。

## 12. 当前实现与目标实现的映射

| 当前代码 | 保留或替换 |
|---|---|
| `build_pipeline_benefits()` | 旧策略保留；MCKP 改用 prefix stall table |
| `MemoryBackend._value()` | LRU 等旧策略保留；MCKP 不经过此接口 |
| `MockAllocationSegmentMemory._evict_one()` | 旧策略保留；MCKP 使用批量 plan |
| `MockAllocationSegmentMemory.plan_request()` | 在 PBP 前插入 MCKP plan/commit |
| `execute_request()` | 保留真实 timeline；增加 planner 指标 |
| backend `clone()` routing probe | 保留，但输出 MCKP score breakdown |
| `choose_routing_candidate()` | 新增 `mckp-transition` score |
| decayed demand | 复用为 history probability |
| rolling median | 复用为 history input estimate |
| oracle-next | 不作为最终 Lookahead；改为显式 future K window |

最终边界是：

```text
Prefix stall table 负责预测缓存配置的流水收益；
MCKP 负责决定释放哪些模型后缀；
PBP 负责把当前请求的 missing groups 放进 Segment extents；
Routing 负责比较每个 GPU 的当前 stall 与未来 eviction damage。
```

## 13. Offline stall table 构建时间 smoke

### 13.1 测量方法

2026-07-28 在当前 CPU 环境使用真实输入：

```text
Tensor layout = docs/tensor-level-sim/tensor-layout.json
Models        = 8
Tensors       = 3635
Compute       = M4 layer profile distributed by Tensor bytes
H2D           = 24.56 GB/s
Allocation    = 0.005 ms/group
Trace         = ServeGen first 1000 source rows
Input scale   = 4
```

smoke 对每个 `(model, input bucket, resident prefix)` 从头计算完整 Tensor
loading/compute timeline，得到 whole-request `exposed_load_ms`。这是最直接
的朴素 Python 实现，其复杂度近似：

```text
O(input_buckets * prefix_configurations * tensors)
```

没有使用跨 prefix 的增量计算或并行化，因此可以作为首版实现的保守估计。

### 13.2 Trace-quantile smoke

先从前 1000 个源请求过滤出的 704 个 model-switch 请求中，为每个模型取：

```text
min / P25 / P50 / P75 / P90 / max
```

由于部分模型的输入长度重复，8 个模型最终共 33 个不同 bucket。

| Granularity | Storage units | Table cells | Repeats | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| 64 MiB TensorGroup | 1065 | 4125 | 7 | 0.802 s | 0.793 s | 1.235 s |
| One Tensor/group | 3635 | 14129 | 3 | 3.051 s | 3.022 s | 3.060 s |

layout、profile 和 704-request trace 的读取与解析额外耗时约 `0.048 s`。

### 13.3 固定 32-bucket 压力 smoke

为了避免 trace 中 bucket 较少导致过于乐观，又为每个模型在
`[1, max_input_tokens]` 上构造 32 个输入长度点，共 256 个
model/input 组合。

| Granularity | Storage units | Table cells | Repeats | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| 64 MiB TensorGroup | 1065 | 34336 | 5 | 6.062 s | 5.998 s | 6.584 s |
| One Tensor/group | 3635 | 116576 | 2 | 21.849 s | 21.842 s | 21.855 s |

每个 cell 对应一个完整 resident-prefix 配置，不是单个 TensorGroup 的
局部 value。

### 13.4 完整构表时间估计

在当前朴素单线程实现下，时间随 bucket 数近似线性。根据 32-bucket
压力结果外推：

| Buckets/model | 64 MiB TensorGroup | One Tensor/group |
|---:|---:|---:|
| 6 | 约 1.1–1.3 s | 约 4.1–4.5 s |
| 32 | 实测约 6.1 s | 实测约 21.8 s |
| 64 | 约 12–14 s | 约 44–48 s |
| 128 | 约 24–28 s | 约 87–96 s |
| 256 | 约 49–56 s | 约 175–192 s |

JSON serialization 和写盘预计相对较小；实现后仍应单独计时确认。

因此对于计划中的默认 `64 MiB TensorGroup + 6 或 32 buckets/model`，
一次完整 8 模型 Offline stall table 构建预计是 `1–7 s`，不是小时级。
即使使用 128 buckets/model，保守估计也在半分钟内。

### 13.5 边界

上述结果只测量：

```text
已有 M4/Tensor compute predictor
+ 已知 Tensor bytes/H2D 参数
+ CPU timeline replay
```

它不包含：

- 在 GPU 上逐 TensorGroup、逐 input length 运行真实 vLLM forward；
- 为 8 个模型采集新的 fused-operator CUDA event profile；
- 加载真实模型权重；
- GPU warm-up、同步或多次统计采样。

正确实现应先采集或加载一份可复用的 compute/loading profile，再在 CPU
上展开所有 prefix table cell；不应该为每个 table cell 单独运行 GPU。
如果未来要求真实 GPU profiling，时间主要由“模型 × input bucket ×
warm-up/repeats”的 profile 采集决定，而不是本节测得的 table expansion，
需要单独做 GPU smoke 后才能可靠估计。

还可以进一步优化 table expansion：

- 预计算每个 input bucket 的 Tensor compute durations；
- 将 TensorGroup 的首 Tensor 和 load cost 编译成紧凑数组；
- 用跨 prefix 的增量/反向动态计算复用 timeline 中间量；
- 按 model 或 input bucket 并行。

这些优化不是首版 blocker；当前 64 MiB TensorGroup 的朴素构表时间已经
足够小。

## 14. 可用的 runtime TensorGroup profile 与 Offline stall table

### 14.1 口径修正

第 13 节只是验证 CPU table expansion 开销的旧结构 smoke，其中
“M4 layer compute 按 Tensor bytes 拆分”不作为最终数据来源。本版改为：

```text
真实 vLLM eager forward
-> 按实际 weight-use 顺序识别 fused runtime stage
-> 只在完整 stage 边界合并出至少 64 MiB 的 TensorGroup
-> 在 TensorGroup 边界采集 CUDA event
-> 计算每个 resident-prefix 的 whole-request exposed loading stall
```

不会把同一 layer 的时间再按 Tensor bytes 比例拆到各 Tensor。

### 14.2 input-scale=4 的准确含义

`profile_layerweave_group_boundaries.py` 不提供 `--input-scale`，也不会在
内部缩放输入；`--tokens N` 就会直接构造 N-token prompt。

本次传入的 token 点已经按模拟器相同语义从 trace 生成：

```text
effective_input =
    min(round(raw_trace_input * 4), model.l40_safe_max_input_length)
```

然后再执行 `model-switches` 过滤和按模型选点。因此本版 profile 已经对应
`input-scale=4`。模拟器查询 table 时应传入它已经缩放并截断后的
`request.input_tokens`，不能再次乘 4。

### 14.3 采集与构表命令

单模型采集命令形式如下；`--tokens` 使用对应模型的 effective token 点：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/mnt/n0/Tangram/Tangram/ElasticKV \
VLLM_LOGGING_LEVEL=ERROR \
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/profile_layerweave_group_boundaries.py \
  --model-id 0 \
  --model /mnt/n0/models/hf/servegen_8_models/model_0 \
  --tokens 56,76,84,152,168,184,196,208,228,276,296,316,336,364,404,428,516,588,640,764,848,936,1068,2296,2800,2832,2932,3056,3244,3400,8652 \
  --output docs/tensor-level-sim/group-boundary-model0.json
```

最终表构建命令：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/build_layerweave_offline_stall_table.py \
  --profiles \
    docs/tensor-level-sim/group-boundary-model0.json \
    docs/tensor-level-sim/group-boundary-model1.json \
    docs/tensor-level-sim/group-boundary-model2.json \
    docs/tensor-level-sim/group-boundary-model3.json \
    docs/tensor-level-sim/group-boundary-model4.json \
    docs/tensor-level-sim/group-boundary-model5.json \
    docs/tensor-level-sim/group-boundary-model6.json \
    docs/tensor-level-sim/group-boundary-model7.json \
  --h2d-gbps 24.56 \
  --segment-allocation-ms .005 \
  --output \
    docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json
```

### 14.4 当前结果

8 个模型最终得到 `834` 个 runtime TensorGroup、`150` 个 input rows 和
`12502` 个 prefix table cells。JSON 输出约 `2 MiB`。最终一次构表和
lookup 单元测试输出：

```text
models = 8
groups = 834
rows   = 150
cells  = 12502
median event/wall overhead = 0.4011 ms
lookup tests = 4 passed
```

8 个 profile 文件中记录的 GPU 采集 wall time 合计约 `512.8 s`
（`8.55 min`）。这是可复用的离线成本；CPU 构表本身约 1 秒。

当前查表采用相邻 input-token 点的分段线性插值，并在边界外 clamp。
模拟器与 table 使用同一份 profile，因此首版不再增加按 tensor/operator
类型的拟合。类型化模型可作为未来减少采样点、迁移到新模型或新 GPU 的
增强项，但不是当前模拟实验的 blocker。

## 15. Prefix-MCKP 与 2-GPU Routing 实现及验证

### 15.1 已实现路径

2026-07-28 已在以下文件落地：

```text
tools/layerpipe/tensor_sim_mckp.py
    PrefixOption / MckpEvictionPlan
    model prefix options
    covering-MCKP Pareto frontier solver

tools/layerpipe/tensor_sim_memory_common.py
    shared immutable PrefixStallTable
    per-request MCKP prediction context and metrics

tools/layerpipe/tensor_sim_segment_backend.py
    resident-prefix invariant
    Prefix-MCKP preview/commit
    prefix-LRU baseline
    MCKP -> PBP -> KV allocation

tools/layerpipe/layerweave_tensor_pipeline_sim.py
    runtime groups loaded directly from Offline table
    profile-driven latency settlement
    cache-bytes and mckp-transition routing
    history/global-lookahead prediction modes
```

使用 `--mckp-stall-table-input` 后，模拟器不会用旧
M4-layer/Tensor-byte timeline 结算请求。它直接从同一张 table：

1. 构造 8 个模型的 834 个 runtime TensorGroup；
2. 查询请求开始时 resident prefix 的 exposed loading stall；
3. 为 MCKP 计算释放后缀的 future stall damage；
4. 为每个 GPU 计算 current stall 和 eviction damage。

因此策略估值和模拟器结算共享相同 profile、group/prefix 编号、input
插值和 clamp 语义。

当前正式基线已经改为普通逐 TensorGroup `lru`。它直接使用原有
storage-unit recency `_select_victim()`，不强制后缀释放。由于一次 forward
按 group 0 到 N 顺序 touch，普通 LRU 在请求边界留下 resident suffix。
Offline table 因此扩展为同时保存：

```text
prefix_stall_ms[prefix_count]   # Prefix-MCKP
suffix_stall_ms[suffix_start]   # ordinary TensorGroup LRU
```

两种曲线都由同一组 runtime boundary profile 计算。基线路由
`cache-bytes` 直接统计当前模型实际 resident groups 的 bytes，选择缓存
参数量最多的 GPU，完全相同时按 round-robin。旧 `lru-prefix` 入口仍保留
用于诊断，但不再作为正式基线。

### 15.2 新增 CLI

```text
--replacement-policy lru|lru-prefix|mckp-prefix
--routing-policy cache-bytes|mckp-transition
--mckp-stall-table-input PATH
--mckp-prediction-mode history|lookahead
--mckp-lookahead-k K
--mckp-lookahead-discount D
--mckp-transition-weight W
--mckp-history-window N
```

`mckp-transition` 的候选 GPU 分数为：

```text
profile_lookup(current input, current resident prefix)
+ W * predicted MCKP eviction damage
```

future damage 不计入当前请求 latency，只作为路由决策分数。

### 15.3 测试

关键命令：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/test_tensor_sim_mckp.py

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/test_layerweave_tensor_pipeline_sim.py

/home/sdu/.conda/envs/sllm-worker/bin/python -m py_compile \
  tools/layerpipe/tensor_sim_mckp.py \
  tools/layerpipe/tensor_sim_memory_common.py \
  tools/layerpipe/tensor_sim_segment_backend.py \
  tools/layerpipe/layerweave_tensor_pipeline_sim.py

git diff --check
```

结果：

```text
Prefix table/MCKP tests = 7 passed
Tensor simulator tests  = 23 passed
py_compile              = passed
git diff --check        = passed
```

新增覆盖包括 MCKP delta cost、minimum-cost covering、active-model
保护、suffix-only commit、cache-bytes routing、stall+damage routing 和
preview/commit prefix invariant。原有 LRU/pipeline/Segment/Page tests
继续通过。

### 15.4 704-request 实验命令

公共参数：

```bash
COMMON_ARGS="\
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --profile docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json \
  --config configs/servegen_8_models_layerpipe_l40_pool42.json \
  --trace evaluation/traces/servegen_tangram.trace \
  --pool-gib 42 --max-requests 1000 --gpus 2 \
  --trace-mode model-switches --input-scale 4 \
  --memory-layout segment --page-size-mib 8 \
  --h2d-gbps 24.56 --tensor-group-min-mib 64 \
  --policy-suite minimal-only \
  --mckp-stall-table-input \
    docs/tensor-level-sim/offline-prefix-stall-runtime-group64.json"
```

普通 TensorGroup LRU + cached-parameter routing：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  $COMMON_ARGS \
  --replacement-policy lru \
  --routing-policy cache-bytes \
  --output \
    docs/tensor-level-sim/runtime-group-lru-cache-route-1000-2gpu.json
```

Prefix-MCKP + transition routing：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_tensor_pipeline_sim.py \
  $COMMON_ARGS \
  --replacement-policy mckp-prefix \
  --routing-policy mckp-transition \
  --mckp-prediction-mode lookahead \
  --mckp-lookahead-k 32 \
  --mckp-lookahead-discount 0.9 \
  --mckp-transition-weight 0.1 \
  --output \
    docs/tensor-level-sim/prefix-mckp-lookahead-k32-route-1000-2gpu.json
```

replacement-only matched routing 额外使用：

```text
--routing-replay \
  docs/tensor-level-sim/runtime-group-lru-cache-route-1000-2gpu.json
```

注意：这里的 Lookahead 是读取后续 32 个全局
`(model_id, effective_input_tokens)` 的 diagnostic oracle，符合 CPU
模拟器实验范围，但不是严格 future-blind online controller。History 模式
也已实现，但短 smoke 的效果不如 K=32 Lookahead，未选为最终效果结果。

### 15.5 最终结果

范围与文档开头一致：Segment request-level PBP、42 GiB/GPU、2 GPUs、
64 MiB minimum runtime groups、first 1000 source rows、704
model-switch requests、input-scale=4。

| Metric (ms) | LRU + cache-bytes | Prefix-MCKP + transition | Change |
|---|---:|---:|---:|
| Critical mean | 440.42 | **422.84** | **-3.99%** |
| Critical P50 | **173.04** | 175.42 | +1.38% |
| Critical P90 | 1130.73 | **1052.96** | **-6.88%** |
| Critical P95 | 1662.48 | **1384.29** | **-16.73%** |
| Critical P99 | 2063.80 | **1936.85** | **-6.15%** |
| Critical max | 2090.70 | **2078.59** | -0.58% |
| Exposed-load mean | 84.80 | **67.23** | **-20.72%** |
| Exposed-load P50 | 0 | 0 | 0 |
| Exposed-load P90 | 243.80 | **224.82** | **-7.79%** |
| Exposed-load P95 | 583.03 | **348.23** | **-40.27%** |
| Exposed-load P99 | 1226.39 | **948.82** | **-22.63%** |
| Exposed-load max | 1378.68 | **1225.02** | **-11.15%** |

其他资源指标：

| Metric | LRU + cache-bytes | Prefix-MCKP + transition | Change |
|---|---:|---:|---:|
| H2D | 2128.34 GiB | **2054.35 GiB** | **-3.48%** |
| Evicted bytes | 2045.14 GiB | **1971.07 GiB** | **-3.62%** |
| PBP moved bytes | **1118.65 GiB** | 1499.30 GiB | +34.03% |
| PBP relocation mean | **1.98 ms** | 2.65 ms | +0.67 ms |
| GPU 0 / GPU 1 requests | 483 / 221 | 436 / 268 | 更均衡 |

MCKP planner time 是 CPU simulator controller 开销，不计入 simulated
critical path：

```text
mean / P50 / P90 / P95 / P99 / max
= 5.68 / 0 / 16.06 / 34.34 / 81.82 / 120.82 ms
```

matched LRU routing 下只替换 replacement 后：

```text
mean critical path : 440.42 -> 429.86 ms  (-2.40%)
mean exposed load  : 84.80  -> 74.25 ms   (-12.44%)
P90 exposed load   : 243.80 -> 233.54 ms  (-4.21%)
P99 exposed load   : 1226.39 -> 1089.15 ms (-11.19%)
```

说明 replacement 本身已经降低 profile-defined loading stall；联合路由
进一步把 P95 critical/exposed tail 明显压低。PBP relocation 在
profile-driven simulator 中作为请求前串行延迟完整加入 exposed load 和
critical path；它使 MCKP mean TTFT 增加 `2.65 ms`，比普通 LRU 的
`1.98 ms` 多 `0.67 ms/request`。即便计入该代价，MCKP mean TTFT 仍降低
`17.57 ms`，说明 profile loading/cache 收益约为 `18.24 ms/request`，
其中约 `0.67 ms` 被额外 relocation 抵消。moved bytes 增加约 34%，说明
MCKP 当前只优化 future stall 和释放空间，没有把
Segment placement/relocation damage 放进目标。后续若要优化 controller
overhead 或 relocation，可以加入 capacity quantum/frontier cache 和
placement-aware secondary cost，但不影响本轮“同 profile 下策略是否改善
loading latency”的结论。

## 16. Trace 特征与 Prefix-MCKP 收益

Segment 版本后续固定使用 15.4 的两个配置，不再继续搜索策略参数：

- baseline：普通 TensorGroup LRU + cache-bytes routing；
- candidate：Prefix-MCKP + transition routing，global lookahead K=32，
  discount=0.9，transition weight=0.1。

### 16.1 Paired analysis 工具

`analyze_prefix_mckp_trace_effects.py` 对两个结果中的同一请求作配对。
定义：

```text
saved_ttft_ms = LRU critical_path_ms - MCKP critical_path_ms
```

正值表示 MCKP 更快。重用距离是在已经执行 `model-switches` 过滤后的
704-request 流中，两次相同 model 请求的 request-index 距离。模型大小取
tensor layout 的 `logical_bytes`，输入长度取 input-scale=4 且经过
per-model safe cap 后的 `effective_input_tokens`。

运行命令：

```bash
python3 tools/layerpipe/analyze_prefix_mckp_trace_effects.py \
  --baseline \
    docs/tensor-level-sim/runtime-group-lru-cache-route-1000-2gpu.json \
  --candidate \
    docs/tensor-level-sim/prefix-mckp-lookahead-k32-route-1000-2gpu.json \
  --tensor-layout docs/tensor-level-sim/tensor-layout.json \
  --output-json \
    docs/tensor-level-sim/prefix-mckp-trace-effects.json \
  --output-csv \
    docs/tensor-level-sim/prefix-mckp-trace-effects.csv \
  --output-svg \
    docs/tensor-level-sim/prefix-mckp-trace-effects.svg
```

产物：

- JSON：分桶汇总；
- CSV：704 条请求的 feature 和 paired delta，可继续作回归；
- SVG：按模型、模型大小、重用距离和输入长度画 mean TTFT saving。

### 16.2 当前 ServeGen trace 的主要发现

全局 mean saving 是 `17.57 ms/request`，但 median 为 0，MCKP 只在
`24.0%` 的请求上严格更快。这不是“所有请求小幅变快”，而是少数昂贵
cold/miss 请求的大幅改善。

按模型大小：

| Logical model size | Requests | Mean TTFT saving | MCKP win rate |
|---|---:|---:|---:|
| < 8 GiB | 281 | -2.71 ms | 17.8% |
| 8--20 GiB | 331 | +0.05 ms | 18.4% |
| >= 20 GiB | 92 | **+142.55 ms** | **63.0%** |

按重用距离：

| Reuse distance | Requests | Mean TTFT saving | MCKP win rate |
|---|---:|---:|---:|
| 1--2 | 175 | -3.86 ms | 1.7% |
| 3--7 | 374 | +1.55 ms | 21.1% |
| 8--15 | 94 | +31.76 ms | 45.7% |
| 16--31 | 36 | **+193.28 ms** | **80.6%** |
| >= 32 | 17 | +147.18 ms | 70.6% |

两个维度结合后最明显：`>=20 GiB` 且 reuse distance 16--31 的 29 个
请求平均节省 `269.14 ms`，win rate `93.1%`。这正好落在 K=32
lookahead 能看见、但普通 LRU 容易因中间请求而破坏 residency 的区间。
大模型 4/5/7 分别只占 5.8%/5.4%/1.8% 的请求，却分别平均节省
54.84/237.09/142.82 ms；当前总收益主要来自“冷但会在不久后复用的大模型”。

route 不变的 529 个请求平均节省 `19.40 ms`，route 改变的 175 个请求
平均节省 `12.05 ms`。这个分组不是 replacement/routing 的因果消融，
但说明当前 trace 的收益并不依赖大量改路由；15.5 的 matched-routing
实验仍是 replacement 因果证据。

输入长度目前没有单调结论：

| Effective input tokens | Requests | Mean TTFT saving |
|---|---:|---:|
| <= 512 | 150 | +12.14 ms |
| 513--2048 | 261 | +28.17 ms |
| 2049--4096 | 128 | +13.28 ms |
| >= 4097 | 165 | +9.07 ms |

原因是模型大小、模型 ID、safe cap、重用距离和 input length 强烈耦合。
例如 model 4/5 的 token 值大量被 cap，不能把全局 input bucket 当成输入
长度的因果效应。输入长度应在固定 model sequence 下单独控制。

### 16.3 哪类 trace 更可能得到明显提升

基于当前 paired evidence，最有利的 trace 应同时满足：

1. working set 总权重大于 2-GPU cache capacity，确实需要 replacement；
2. 大模型不是一次性 cold object，而是在约 8--32 个 model-switch 后复用；
3. 大模型访问有可预测的局部性，下一次访问落在 lookahead K 内；
4. 模型间 prefix stall-per-byte 差异大，使 MCKP 有比 byte/LRU age 更好的
   选择空间；
5. 请求仍存在可暴露的 loading stall，而不是被很长 prefill compute
   完全隐藏。

以下 trace 的提升预计有限，甚至可能略差：

- working set 能完整放入显存，两个策略都几乎不 eviction；
- reuse distance 1--2，LRU 已接近正确答案；
- 完全无复用或复用远超 K，oracle window 也没有可用信号；
- 模型大小和 prefix stall-per-byte 都接近均匀，MCKP 无法区分价值；
- 极长输入把 H2D 大量隐藏，此时即便减少加载，TTFT 也不敏感；
- 高频访问只集中在小模型、稀有大模型之后不再出现。

### 16.4 下一轮受控 trace matrix

不能只用当前一条 trace 宣称普适规律。下一轮保持模型和 tensor profile
不变，每个 cell 都运行上述固定 baseline/candidate：

| Axis | Controlled variants | 要回答的问题 |
|---|---|---|
| size × hotness | large-hot / size-neutral / small-hot | 热度分配给大模型是否放大收益 |
| reuse locality | burst、4-model phases、balanced IID、round-robin | 收益峰值位于哪个 reuse-distance 区间 |
| input | per-model P25/P50/P75，或 scale 1/2/4 后 cap | 在相同 model sequence 下，compute overlap 如何改变收益 |
| predictability | 相同直方图但 phase-local 与全局 shuffle | K=32 信号本身贡献多少 |

生成 model-sequence variant 时，每个模型从自己的 source token-shape pool
循环取样，因此只改变模型序列：

```bash
python3 tools/layerpipe/build_layerweave_trace_variants.py \
  --source evaluation/traces/servegen_tangram.trace \
  --output-dir evaluation/traces/layerweave_variants \
  --requests 1000 --seed 1234
```

报告应至少包含 mean/P50/P90/P95/P99 critical path、mean/P95/P99 exposed
load、H2D、PBP relocation，以及 request-level paired saving；主横轴使用
实际测得的 reuse-distance 分布和 `large-model request share`，不要只用
variant 名称。由于当前 candidate 使用 global-next-K oracle，这些结论只
代表该 diagnostic 上限；最终 online history 版本需要重新做同一矩阵。

已完成一个 `round_robin` pilot：每个 8-request block 随机排列全部 8 个
模型，模型频率完全均衡，token shape 仍从各模型自己的 source pool 取样。
它把大模型从原 trace 的低频长尾变成稳定复用对象，并使典型 reuse
distance 落在 K=32 内：

| Metric | LRU + cache-bytes | Prefix-MCKP + transition | Change |
|---|---:|---:|---:|
| Critical mean | 734.62 ms | **599.42 ms** | **-18.40%** |
| Critical P50 | 631.11 ms | **573.89 ms** | -9.07% |
| Critical P90 | 1449.53 ms | **1094.37 ms** | **-24.50%** |
| Critical P95 | 1662.37 ms | **1266.60 ms** | **-23.81%** |
| Critical P99 | 2079.71 ms | **1960.75 ms** | -5.72% |
| Exposed-load mean | 397.50 ms | **262.30 ms** | **-34.01%** |

请求级 mean saving `135.20 ms`，median saving `37.41 ms`，MCKP win
rate `72.1%`。这比原 trace 的 `17.57 ms` mean saving 和 `24.0%`
win rate 明显更强，支持“**大模型进入稳定但非极短的 K-window 复用区间**
会放大 Prefix-MCKP 收益”这一判断。产物为：

```text
docs/tensor-level-sim/trace-variant-round-robin-lru.json
docs/tensor-level-sim/trace-variant-round-robin-mckp.json
docs/tensor-level-sim/trace-variant-round-robin-effects.{json,csv,svg}
```

这仍只控制了 model sequence，没有独立控制 size-hotness 与 input length；
其余 matrix cell 完成前，不把 18.4% 当作一般 trace 的预期收益。

### 15.6 TTFT CDF

无外部绘图库版本：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/plot_prefix_mckp_ttft_cdf.py \
  --baseline \
    docs/tensor-level-sim/runtime-group-lru-cache-route-1000-2gpu.json \
  --mckp \
    docs/tensor-level-sim/prefix-mckp-lookahead-k32-route-1000-2gpu.json \
  --output \
    docs/tensor-level-sim/prefix-mckp-vs-lru-ttft-cdf.svg
```

输出同时包含完整 CDF 和 `CDF=0.85–1.0` tail detail。基线为普通
TensorGroup LRU。
