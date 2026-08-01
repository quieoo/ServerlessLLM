# LayerWeave 第三项技术：配置级联合规划与延迟页面回收

## 1. 设计目标

LayerWeave 的第三项技术联合解决以下两个问题：

1. 当一个新模型请求到达时，应将其调度到哪张 GPU；
2. 为容纳目标模型、流水线运行时空间和动态 KV Cache，目标 GPU 应保留或回收哪些已有模型参数。

传统方法通常分别处理请求路由与缓存驱逐：

- 调度器倾向于选择缓存目标模型参数最多的 GPU；
- 缓存策略依据 LRU、LFU、访问概率或重加载字节数选择驱逐对象。

但在 reuse-aware computation–loading pipeline 中，参数的实际价值取决于其对关键路径加载停顿的影响：

- 某些缺失参数会直接阻塞模型执行；
- 某些后续参数即使缺失，也可以在前序层计算期间完成加载；
- 当前请求路由到某张 GPU 后，可能驱逐其他模型的高价值参数，从而增加未来请求的暴露加载时延；
- KV Cache 的实际需求动态变化，若提前按照保守上界回收参数，会损失本可继续利用的复用机会。

因此，LayerWeave 将该问题统一为：

> 在控制在线决策复杂度的同时，最小化当前请求暴露在关键路径上的加载停顿，以及缓存变化对未来请求造成的预期停顿。

为此，LayerWeave 将**配置级缓存规划**与**页面级物理回收**解耦：

- 配置级优化负责目标 GPU 选择、缓存价值评估与保护边界规划；
- 页面级运行时根据实际模型加载与 KV Cache 增长需求延迟回收参数。

---

## 2. Pipeline Stall Estimator

LayerWeave 使用 **Pipeline Stall Estimator（PSE）** 统一衡量请求特征、参数驻留状态与计算—加载流水线之间的关系。

给定：

- 当前请求或请求批次 \(r\)；
- 目标模型 \(m\)；
- 候选 GPU \(g\)；
- 目标模型在该 GPU 上的逐层、逐页参数驻留状态；
- 参数传输带宽；
- 离线 profile 得到的逐层计算时间；

PSE 预测 reuse-aware pipeline 执行后，仍有多少参数加载时间无法被计算隐藏：

\[
S(r,m,g)
=
\operatorname{ExposedPipelineStall}(r,m,g).
\]

PSE 需要考虑：

- 每层参数大小与驻留状态；
- 缺失参数的传输时间；
- 每层 Prefill 计算时间；
- 输入长度与 batch composition；
- 前序计算为后续参数加载提供的 overlap window；
- 前序 stall 对后续层时间线的累积影响。

PSE 的输出不是简单的“缺失参数量除以带宽”，而是真正暴露在请求关键路径上的加载停顿。

因此，相同的缺失参数量可能具有完全不同的代价：

- 短输入请求计算窗口较小，更容易暴露加载停顿；
- 长输入请求计算窗口较大，更多传输可以被隐藏；
- 前部层参数通常比后部层参数更容易阻塞模型启动；
- 实际仍驻留的机会式参数页也可以减少当前请求的 stall。

---

## 3. 配置级缓存规划

### 3.1 Launch Configuration

直接对所有参数页进行联合组合优化会带来较高在线开销。LayerWeave 为每个模型定义一组有限的 **launch configurations**，例如：

\[
\mathcal K_m
=
\{0,2,4,8,16,32,L_m\},
\]

其中配置 \(k\) 表示将模型 \(m\) 的前 \(k\) 层设为受保护参数，\(L_m\) 表示保护完整模型。

每个配置具有两个基本属性。

#### 内存开销

\[
s_{m,k}
=
\text{配置 }k\text{ 所覆盖参数的显存占用}.
\]

#### 预期缓存价值

对于代表性请求 \(r\)，配置 \(k\) 相对完全不缓存时减少的暴露加载停顿为：

\[
b_{m,k}(r)
=
S_m(0,r)-S_m(k,r).
\]

根据模型请求分布 \(\mathcal D_m\)，可计算平均收益：

\[
\bar b_{m,k}
=
\mathbb E_{r\sim\mathcal D_m}
\left[
S_m(0,r)-S_m(k,r)
\right].
\]

再结合模型未来再次被请求的概率 \(p_m\)，得到配置的预期价值：

\[
v_{m,k}
=
p_m\cdot \bar b_{m,k}.
\]

该价值表示：

> 在未来模型激活中，保护该配置预计能够减少多少暴露在流水线关键路径上的加载时延。

### 3.2 Configuration 的新语义

Launch configuration 不再表示 GPU 上最终必须精确保留的全部参数集合，而表示：

> 系统承诺保护的最低驻留集合，即 protected residency floor。

因此，配置级优化只决定：

- 哪些参数具有较高长期价值，应当受到保护；
- 在预计内存压力下，每个模型至少保留到哪个配置；
- 哪张 GPU 承接当前请求的综合代价最低。

配置之外但仍实际驻留的页面不需要立即驱逐，而是作为机会式缓存继续保留。

---

## 4. 三类页面状态

LayerWeave 将 GPU 上的参数页划分为三类。

### 4.1 Active Pages

当前正在执行的模型参数，以及当前请求使用的 KV Cache、workspace 和流水线 buffer。

这些页面在请求执行期间不可回收。

尤其需要注意：对于 Prefill 和 Decode，目标模型权重在整个推理过程中仍会被重复使用，因此不能将当前活动模型的参数页重新分配给 KV Cache。

### 4.2 Protected Pages

由配置级优化为各非活动模型选择的 launch configuration 所覆盖的页面。

这些页面构成模型的 protected residency floor。正常情况下不参与回收，因为驱逐它们会造成较高的预期 future pipeline stall。

### 4.3 Soft Pages

实际仍驻留，但超出 protected configuration 的参数页。

例如，模型 \(A\) 当前完整驻留 32 层，而规划器只选择保护前 8 层：

- 前 8 层为 protected pages；
- 第 9–32 层仍可继续驻留，但被标记为 soft pages；
- 当 loader 或 ODKV 需要更多显存时，优先回收这些页面；
- 若请求实际内存需求较低，这些页面不会被回收；
- 后续模型再次激活时，只要 soft pages 仍存在，就可直接复用并减少数据传输。

因此：

\[
R_m^{\mathrm{actual}}
\supseteq
R_m^{\mathrm{protected}}.
\]

其中：

- \(R_m^{\mathrm{actual}}\) 表示模型当前真实驻留页面；
- \(R_m^{\mathrm{protected}}\) 表示配置级规划器承诺保护的页面。

---

## 5. 候选 GPU 上的配置优化

当目标模型 \(q\) 的请求到达时，LayerWeave 对每张候选 GPU \(g\) 分别评估。

### 5.1 当前请求的暴露加载停顿

PSE 使用 GPU 的真实页面驻留状态计算：

\[
S(r,q,g).
\]

这里既包括 protected pages，也包括尚未回收的 soft pages。只要页面仍实际驻留，当前请求就可以复用它们。

### 5.2 预计缓存预算

系统根据以下内存需求计算配置级缓存预算：

- 活动模型运行空间；
- 流水线 buffer；
- ODKV 预计需要的 KV Cache；
- 已有活动请求的内存占用；
- 必要安全余量。

记预计可用于保护参数的容量为：

\[
C_g.
\]

### 5.3 Multiple-Choice Knapsack Problem

对于 GPU 上的每个模型 \(m\)，从其配置集合中选择一个 protected configuration：

\[
x_{m,k}\in\{0,1\}.
\]

每个模型选择一个配置，包括 0 层：

\[
\sum_{k\in\mathcal K_m^g}x_{m,k}=1.
\]

受保护配置的总大小不能超过缓存预算：

\[
\sum_m
\sum_{k\in\mathcal K_m^g}
s_{m,k}x_{m,k}
\leq C_g.
\]

目标是最大化重配置后的预期保护价值：

\[
\max
\sum_m
\sum_{k\in\mathcal K_m^g}
v_{m,k}x_{m,k}.
\]

这是一个 **Multiple-Choice Knapsack Problem（MCKP）**：

- 每个模型对应一个 item class；
- 每个 launch configuration 对应一个 item；
- 每个模型选择一个受保护配置；
- 所有模型共享 GPU 的 protected-cache budget；
- 优化目标是最大化未来预期 pipeline-stall reduction。

### 5.4 当前状态约束

对于 GPU 上已有的非目标模型，通常只允许维持或降低 protected configuration。

若模型 \(m\) 当前的保护边界为 \(k_m^0\)，则：

\[
k\leq k_m^0.
\]

对于当前目标模型 \(q\)，本次请求会完成其全部参数的加载与执行，因此请求完成后可选择任意 protected configuration，包括完整模型。

---

## 6. Cache-State Transition Cost

候选 GPU \(g\) 在调度前的受保护缓存价值为：

\[
V_g^{\mathrm{before}}
=
\sum_{m\in\mathcal M_g}
v_{m,k_m^0}.
\]

通过 MCKP 获得最优保护配置后，其价值为：

\[
V_g^{\mathrm{after},*}.
\]

定义缓存状态转移代价：

\[
D_g
=
V_g^{\mathrm{before}}
-
V_g^{\mathrm{after},*}.
\]

该指标表示：

> 为承接当前请求，目标 GPU 需要降低哪些模型的保护边界，以及这些变化会对未来请求造成多少预期暴露加载停顿。

- \(D_g>0\)：当前请求破坏了部分高价值保护状态；
- \(D_g\approx0\)：新旧保护价值基本平衡；
- \(D_g<0\)：当前目标模型更有价值，重配置反而提高了未来缓存收益。

需要注意的是，\(D_g\) 是配置级规划代价。由于配置之外的 soft pages 不会立即回收，实际发生的 cache damage 可能更低。

---

## 7. 联合 Cache–Routing 决策

对于每张候选 GPU，LayerWeave 计算：

1. 当前排队或等待时间；
2. 当前请求暴露的 pipeline stall；
3. 配置级缓存状态转移代价。

最终选择：

\[
g^*
=
\arg\min_g
\left[
T_g^{\mathrm{queue}}
+
S(r,q,g)
+
\lambda D_g
\right].
\]

其中：

- \(T_g^{\mathrm{queue}}\) 表示 GPU 的排队延迟；
- \(S(r,q,g)\) 由 PSE 基于真实驻留状态计算；
- \(D_g\) 由配置级 MCKP 计算；
- \(\lambda\) 控制当前请求时延与未来缓存收益之间的权衡。

配置级策略负责回答：

- 请求应调度到哪张 GPU；
- 每个模型至少应保护多少参数；
- 哪些参数可以进入可回收状态。

但它不要求立即将物理驻留状态收缩到选定配置。

---

## 8. 延迟页面回收

请求被调度到目标 GPU 后，LayerWeave 不立即执行完整的配置转换，而采用 **deferred page reclamation**。

### 8.1 基本流程

1. 根据 MCKP 结果更新各模型的 protected configuration；
2. 将超出保护边界的参数页标记为 soft pages；
3. 在模型加载或 KV Cache 增长真正需要空间时，按需回收 soft pages；
4. 只释放满足当前实际需求所需的页面；
5. 请求结束后，未被回收的 soft pages 继续保留并参与后续复用。

因此：

\[
D_g^{\mathrm{actual}}
\leq
D_g^{\mathrm{planned}}.
\]

其中：

- \(D_g^{\mathrm{planned}}\) 为配置级规划的状态转移代价；
- \(D_g^{\mathrm{actual}}\) 为最终真正回收页面造成的损失。

### 8.2 与 ODKV 的结合

延迟回收与 ODKV 可以自然结合。

请求开始时：

- KV Cache 占用较小；
- 大量 soft parameter pages 可以继续驻留；
- 后续模型请求仍可能复用这些页面。

随着 Decode 推进：

- KV Cache 按需增长；
- ODKV allocator 触发 soft-page reclamation；
- 系统只释放满足实际增长需求的参数页面；
- 若生成提前结束，则无需回收剩余 soft pages。

因此，KV Cache 可以逐步占用机会式保留的参数缓存空间，而无需按照最大序列长度提前清空缓存。

需要强调的是，被 ODKV 回收的是**非活动模型的 soft pages**，而不是当前活动模型正在使用的参数。

---

## 9. Soft Page 回收顺序

当需要释放显存时，系统优先回收 soft pages。

基本顺序可以是：

1. 完全未被任何配置保护的模型参数；
2. 超出各模型 protected prefix 的后部参数；
3. 未来访问概率较低模型的参数；
4. 对 future exposed stall 贡献较小的参数。

可为页面或较大回收单元定义近似价值密度：

\[
\rho_p
=
\frac{
p_m\cdot\Delta S_p
}{
\operatorname{Size}(p)
}.
\]

按 \(\rho_p\) 从低到高回收。

为降低运行时复杂度，不需要为每个底层 CUDA page 实时运行 PSE。可以按照 configuration 间的增量区间，为一组层或 VMM chunk 赋予近似相同的回收优先级。

实际 reclaim unit 可以采用：

- 一层参数；
- 若干连续参数页；
- 固定大小 VMM chunk。

不建议直接使用大量极小页面作为独立回收单元，否则可能增加 CUDA Driver API 开销。

---

## 10. 水位驱动的主动回收

延迟回收不应等同于 allocation failure 后的被动处理，否则可能在关键路径上引入抖动。

LayerWeave 可以维护两级显存水位：

- 当可用显存低于 low watermark 时，启动后台 soft-page reclamation；
- 当可用显存恢复到 high watermark 时，停止回收。

系统还应预留：

- 下一批参数加载所需空间；
- 短期 KV Cache 增长空间；
- 流水线 buffer；
- 必要的 VMM map/unmap 安全余量。

因此，该机制更准确地说是：

> **Deferred but proactive page reclamation**

即策略上延迟回收，但在运行时根据预测需求和显存水位提前执行。

---

## 11. 设计收益

### 11.1 解耦控制粒度与执行粒度

配置级优化采用较粗粒度，以控制 MCKP 的求解开销；物理内存管理采用较细粒度，以提高空间利用率。

即：

> 配置级策略负责全局价值判断，page-level runtime 负责精确空间回收。

### 11.2 避免配置离散化造成物理空间浪费

固定配置不再意味着 GPU 必须立即收缩到相应边界。

实际驻留状态为：

\[
\text{Physical Residency}
=
\text{Protected Configurations}
+
\text{Soft Residual Pages}.
\]

因此，配置间的剩余空间可以继续由 soft pages 占用，不会因为配置粒度较粗而提前闲置。

### 11.3 保留意外复用收益

调度器基于配置级模型进行稳定决策，但未被实际回收的页面仍然可以服务后续请求。

因此，即使某些页面未被规划器保护，也不会立即失去其复用价值。

### 11.4 适应动态 KV Cache 需求

ODKV 根据真实生成长度逐步增长，延迟回收只释放实际需要的空间，从而避免按照保守 KV 上界提前驱逐参数。

---

## 12. 在线执行流程

完整在线流程如下：

1. 收集当前请求的输入长度、batch shape 和目标模型；
2. 获取各候选 GPU 的真实参数驻留状态、保护状态、队列和可用显存；
3. 使用 PSE 预测请求在每张 GPU 上的暴露加载停顿；
4. 根据运行时空间与 ODKV 需求估计每张 GPU 的 protected-cache budget；
5. 为每张 GPU 构造并求解配置级 MCKP；
6. 计算 cache-state transition cost；
7. 综合 queue delay、当前 exposed stall 和未来 transition cost 选择目标 GPU；
8. 将 MCKP 结果设置为新的 protected residency floor；
9. 将配置外参数标记为 soft pages，而非立即回收；
10. 在参数加载和 KV Cache 增长过程中，根据水位和实际需求回收 soft pages；
11. 请求完成后，继续保留所有未被回收的参数页，供未来请求复用。

---

## 13. 核心创新

该设计的关键不在于单独提出一个缓存算法或请求调度算法，而在于将三种机制连接起来：

1. **PSE** 将参数驻留价值统一转换为 exposed pipeline stall；
2. **Configuration-level joint planner** 联合选择目标 GPU 和各模型的保护边界；
3. **Deferred page reclamation** 根据真实运行时需求精确回收配置之外的 soft pages。

最终优化目标为：

\[
\text{Current Exposed Pipeline Stall}
+
\text{Expected Future Pipeline Stall}.
\]

同时，通过配置与物理回收解耦，系统避免了粗粒度 configuration 导致的直接空间浪费：

> LayerWeave uses coarse-grained configurations for low-overhead global planning, while retaining page-level flexibility for high-utilization physical memory management.

该设计将传统的 byte-locality cache 和 affinity routing，转化为统一的关键路径感知缓存规划、请求放置与弹性内存回收机制。

---

## 14. Global Pending Queue 与有界 Placement Wait

候选 GPU 不应只包含当前 idle GPU。若 busy GPU 上已经驻留目标模型，
等待其当前请求完成可能比立即在 cold idle GPU 上加载更快。但请求不应
提前绑定到 per-GPU FIFO；否则两张卡各有一个 pending 后，后续放置只剩
一个有空位的候选 GPU，调度会退化为 first-free。

当前实现将尚未执行的请求保留在 global pending queue。每当至少一张 GPU
空闲时，coordinator 对 pending lookahead 窗口内的每个请求都查询两张 GPU，
并计算：

\[
\operatorname{Score}(r,g)
=
\hat T_g^{\mathrm{queue}}
+
\hat S(r,m,g\mid\hat R_g)
+
\lambda\hat D_g.
\]

其中：

- \(\hat T_g^{\mathrm{queue}}\) 是 running 请求的预计剩余时间；
- \(\hat R_g\) 是 running 请求完成后的 projected residency；
- \(\hat S\) 是请求执行到队首时的预计 service TTFT/exposed stall；
- \(\hat D_g\) 是 projected protection state 上的 configuration transition
  cost。

默认参数为：

```text
routing lookahead = 8
max placement wait = 150 ms
placement hysteresis = 25 ms
per-GPU assigned pending = 0
```

如果 busy GPU 的预测总分至少比最佳 free GPU 低 hysteresis，且预计等待
不超过 wait cap，则该请求暂留在 global queue。coordinator 继续查看窗口中
的其他请求，因此不会因队首请求的 affinity wait 而让 free GPU 无谓空闲。
达到 wait cap 后，请求必须发往当前 free GPU，避免缓存亲和性导致吞吐崩溃。

路由阶段不执行物理回收；请求真正开始前，worker 使用 actual residency
重新运行 MCKP 和 exact-shortage reclamation。worker 完成后将 actual state
与实际 service time 返回 coordinator，用于修正 `predicted_available_at`。

Joint 的负 transition cost 表示新 configuration 提高了未来缓存价值。它仍可
作为当前 placement 的 credit，但默认使用 `transition_weight=0.1`，且 credit
最多为 100 ms，避免未来价值让当前延迟分数变成不受限的负数。

Minimal baseline 使用相同 global-pending/placement-wait 机制，但保持 byte-locality
目标：

\[
\operatorname{Score}_{minimal}(r,g)
=
\hat T_g^{\mathrm{queue}}
+
\frac{\operatorname{MissingWeightBytes}(m,\hat R_g)}
{\operatorname{PCIeBandwidth}_g}.
\]

它不使用 PSE、protected configuration value 或 transition cost。因此
两侧的队列能力一致，A/B 的差异仍然限定为关键路径感知的 placement 和
Joint protected/soft cache planning。

### 14.1 Serial-choice 上界诊断

为区分“策略本身没有收益”和“在线负载下没有 placement 自由度”，实现另设
`serial-choice` 诊断模式。该模式忽略 arrival time，保持 trace 请求顺序，
全系统一次只执行一个请求。每个请求完成后，两张 GPU 都处于 idle，调度器
再使用各自独立且持续演化的 cache residency 对下一请求执行二选一放置。

该模式不使用 busy-GPU queue、lookahead、placement wait 或 hysteresis。
它不是在线吞吐实验，而是 LayerWeave cache/placement 在理想选择自由度下的
性能上界实验。

为避免 greedy baseline 的单卡吸附，Minimal 和 Joint 均可独立启用 seeded
cold-tie random：只有两张卡对当前模型都没有任何 cached page 时随机播种，
已有任意缓存时保持正常 locality/PSE 评分。Joint 的 weighted transition
contribution 同时使用对称限幅，默认 `[-100,+100] ms`，防止 future cache
value 覆盖数百毫秒的当前加载差。

serial-choice 中 route estimate 与正式 plan 之间没有并发状态变化，因此
chosen worker 可复用 estimate 时的 actual idle residency snapshot。该优化
不用于 online 模式。request-shape/configuration estimator curve 也被缓存，
而 demand score 在每次决策时重新应用；这保持策略结果不变并降低重复规划
开销。

### 14.2 Eviction 实现与未来感知上限

物理回收新增 opt-in `LAYERWEAVE_BATCH_UNMAP=1`：对同一 stable arena 中
连续、resident 且未 retained 的页合并 `cuMemUnmap` 区间，成功后逐页归还
extent bookkeeping，driver 不支持时回退逐页。该优化不改变 victim set，
100-request 消融中 eviction apply 从 5.30 降至 4.71 ms/request。

`cache_policy_mode=next-use` 是仅限 serial-choice 的 clairvoyant 上限：

1. active model working set 始终不可回收；
2. inactive resident pages 不设 demand protection floor；
3. 按所属模型在该 GPU 未来请求流中的 next-use position 降序淘汰；
4. 无未来访问视为 infinity，最先淘汰。

为了避免 route/cache coupling，正式比较必须使用 `--routing-replay` 固定
prior demand run 的 request-to-GPU mapping；worker 的 future stream 也按
该 mapping 过滤。固定相同 51/49 路由后，next-use 仅将 mapped pages
从 9351 降至 9206，说明当前固定路由内的 eviction 改进空间有限。若要接近
whole-model DP 的 7615-page 上限，需要联合未来感知 placement/partition，
不能只替换单 GPU victim ordering。

### 14.3 Online 结果

两轮 100-request、trace scale 4、双 GPU online A/B 中，Joint 相比相同全局
pending Minimal：

- throughput 分别提高 2.91% 和 3.10%；
- makespan 分别降低 2.83% 和 3.01%；
- mean service TTFT 分别降低 23.58 和 30.48 ms；
- pooled paired mean gain 27.03 ms，bootstrap 95% CI
  `[5.47, 50.20] ms`；
- exposed load 分别降低 30.19 和 38.41 ms。

paired median 仍接近 0，说明收益来自少数高代价 reload 的规避。后续优化
优先级应是更好的 online placement/partition 预测，而非继续增加 transition
调参或单独微调 victim ordering。

### 14.4 Profile-driven pipeline cache model

`tools/layerpipe/layerweave_pipeline_cache_sim.py` 使用 M4 per-layer compute
regression 和实测 H2D bandwidth，恢复 cold run 的 unique pages by stage，
在相同 pool/KV reservation/trace/routing 下比较：

- Minimal：model-frequency page value，同值 page-LRU，顺序 forward 后前缀
  最老；
- pipeline-aware：条件 greedy 选择使 predicted GPU critical path 降低最多
  的 resident page。

模拟发现旧 JSON reload 未将 `compute` layer key 从字符串转为整数，导致
正式 estimator 忽略全部 per-layer compute。该路径已在 simulator 和
`LayerWeaveSingleGpuCachePolicy` 中修复；旧 GPU A/B 不代表修复后的策略。

固定 Minimal 路由的 1000-request 理论结果：

```text
Minimal exposed load          94.97 ms/request
Pipeline-aware exposed load   84.28 ms/request
Additional H2D                 7.22 ms/request
Critical-path gain            10.69 ms/request (2.80%)
Stage-0 evictions             8380 -> 4223
```

因此 pipeline-aware cache 的正确目标不是最少 H2D，而是允许更多可重叠
suffix H2D，换取更少不可重叠 prefix H2D。当前 Profile 给出的理论收益约
2%--3%；下一步应以修复后的 integer compute keys 重新执行真实 serial-choice
和 online GPU A/B。

五项机制消融（1000 request、672 pages/GPU、固定 Minimal 路由）：

```text
baseline:   cold load + compute serialized       871.02 ms
pipe-only:  cold load/compute pipelined           639.52 ms
reuse-only: Minimal reuse, load/compute serialized 429.17 ms
Minimal:    Minimal reuse + pipeline              381.24 ms
LayerWeave: pipeline-aware cache + pipeline       370.55 ms
```

对应 baseline 加速分别为 0%、26.58%、50.73%、56.23% 和 57.46%。
LayerWeave 相对 Minimal 为 10.69 ms/2.80%。它当前优化的是 retained page
对关键路径的价值，而非 page hit count；事实上 missing pages 为
`52256 -> 54728`，但 exposed load 为 `94.97 -> 84.28 ms`。

结果中的 `five_way_ablation.inspected_request` 提供请求级 timeline；
默认是 LayerWeave 正收益样本的中位案例，也可通过
`--inspect-request-id N` 指定。1000-request 默认选中的 request 61 中，
Minimal 的 29 个缺页全部位于 stage 0，产生 78.90 ms initial stall；
LayerWeave 虽有 31 个缺页，但分别位于 stage 1--31，加载几乎完全被前层
compute 隐藏，因此 predicted total 从 262.50 降至 183.73 ms。

### 14.5 Configuration-MCKP simulator result

模拟器已实现本文第 3--8 节的配置级路径：多 input-shape Profile curve、
`{0,2,4,8,16,32,full}` protected prefix、decay-0.9 online demand、精确
MCKP、inactive configuration 非升级约束，以及仅回收 soft configuration
tiers 的 deferred reclamation。旧 page-level greedy 作为独立消融保留。

固定 Minimal 路由时：

```text
1000 request / 2 GPU
Minimal                  381.24 ms
page-greedy              370.55 ms  (-2.80%)
configuration-MCKP       380.09 ms  (-0.30%)

5000 request / 2 GPU
Minimal                  381.15 ms
page-greedy              368.14 ms  (-3.41%)
configuration-MCKP       381.63 ms  (+0.13%)
```

5000-request MCKP 没有实现预期的总体 hit-rate 提升：missing pages 为
`264378 -> 281933`，H2D 为 `144.50 -> 154.20 ms`，exposed load 为
`96.54 -> 97.02 ms`。这说明完整机制已在模拟器中表达，但当前离散配置和
value/demand 模型仍不足；不能再用 page-greedy 的约 3% 代表配置级联合
规划收益。

### 14.6 Queue-lookahead oracle

固定 Minimal 路由后，模拟器可用 `--lookahead-k` 查看每张 GPU 未来 K 个
真实请求，并使用真实 model/input length 对 configuration gain 求和。仍
resident 的 soft prefix 可以重新保护，缺失页不会被 oracle 预取。由于使用
了尚未 arrival 的 trace，该模式只表示缓存规划上界。

1000-request 扫描：

```text
demand-MCKP   +0.30%
K=1           -0.60%
K=4           +1.19%
K=8           +1.35%   best
K=16          +1.10%
K=32          +0.95%
page-greedy   +2.80%
```

K=8 与 per-GPU reuse-distance P90 约 8--9 相符。真实未来访问和输入长度只将
configuration-MCKP 提高到 1.35%，仍明显低于 page-greedy；因此 EMA 预测
误差不是主要上限，下一步应优先细化 configuration 并修正 configuration
value 对当前 soft residency/cache damage 的表达。

### 14.7 Exact residency-transition result

为修正 configuration value 与实际 cache damage 的语义差异，模拟器新增
stage-granular physical transition planner。它从当前真实 \(R\) 出发，根据
active-model load/KV/pool 得到 exact shortage，并用 lookahead PSE 直接计算
驱逐每个 inactive resident stage group 的边际 future critical-path damage。
规划器只驱逐满足 shortage 所需的最低 damage pages。

1000-request 固定路由扫描：

```text
K=1    378.94 ms   +0.61%   H2D 150.95 ms   missing 55145
K=4    371.76 ms   +2.49%   H2D 140.25 ms   missing 51260
K=8    371.96 ms   +2.44%   H2D 139.63 ms   missing 51031
K=16   372.62 ms   +2.26%   H2D 141.04 ms   missing 51548
K=32   372.94 ms   +2.18%   H2D 141.27 ms   missing 51628
```

K=4 最佳。相较 Minimal 的 381.24 ms/H2D 142.90 ms/missing 52256，
它同时改善关键路径、加载量和命中；旧 page-greedy 为 370.55 ms，但 H2D
150.11 ms、missing 54728。结果支持更新后的核心逻辑：configuration 可用作
候选结构，但最终计划必须根据 exact post-reclamation residency 的未来代价
决定，不能直接最大化抽象 configuration value。

### 14.8 MCKP + Page-greedy

新的 hybrid 保留 configuration-MCKP 作为跨模型 protected-floor budget
allocator，但不再让 configuration tier 决定 soft residency 的物理回收
顺序。实际 shortage 到来时，在 protected floor 之外执行全局 Page-greedy：

\[
V(m,p)=D_K(m)\cdot \Delta CP(m,p)
\]

其中 \(D_K(m)\) 是 future-K demand，\(\Delta CP(m,p)\) 是预计算的
pipeline marginal page value。只回收最低 \(V(m,p)\) 的 inactive soft
pages，并精确满足 shortage。

1000-request 固定 Minimal 路由中，Hybrid K=8 的 critical path 为
`371.25 ms`，相对 Minimal 提升 `2.62%`；exposed load 为 `84.98 ms`，
H2D 为 `139.56 ms`，missing pages 为 `50995`。它达到与 Page-greedy
接近的性能，同时 H2D/miss 低于 Minimal，也略优于高控制开销的
Exact-residency K=4 (`371.76 ms`)。

隔离测量使用 `--policy-suite mckp-page-greedy`，该入口不构造或执行
transition/Exact-residency planner。Hybrid K=8 controller time 为 mean
`0.367 ms`、P95 `0.602 ms`、P99 `0.717 ms`、max `0.878 ms`；读取
per-GPU lookahead queue 的均值仅 `0.00084 ms`。其 critical-path mean 为
`371.25 ms`，P95/P99 分别为 `1269.79/1675.23 ms`，与 Minimal 尾部相同；
当前 workload 的尾部主要由长输入 compute/cold-like 请求决定。

### 14.9 Joint placement 与 configuration 粒度

模拟器新增 `--joint-routing` serial-choice 诊断。每个请求对两张 GPU 的
actual residency 分别执行 MCKP + Page-greedy 候选规划，并使用：

\[
\operatorname{Score}(r,g)
=
S(r,g)
+
\lambda\,
\operatorname{clip}
\left(
\Delta CP_{\text{future-}K}(g), -100, 100
\right)
\]

选择 GPU。候选被选中后直接提交对应 cache snapshot，避免 route estimate
与正式 cache plan 使用不同状态。该诊断保持系统级串行执行，因此不包含
online queue/wait；future-K 使用尚未 arrival 的真实请求，只表示联合
placement/cache 的 oracle 上界。

配置粒度由 `--configuration-step` 控制：

- `0`：原始 `0/2/4/8/16/32/full`；
- `2`：每两层一个 prefix；
- `1`：逐层 prefix。

1000-request、2 GPU、672 pages/GPU、K=8、默认 \(\lambda=0.1\)：

```text
configuration   candidates   joint CP    vs Minimal   MCKP mean   route mean
coarse          52           373.05 ms   +2.15%       0.454 ms    17.202 ms
step=2          158          372.77 ms   +2.22%       0.720 ms    15.409 ms
step=1          308          371.45 ms   +2.57%       1.288 ms    16.703 ms
```

逐层相对 coarse 额外降低 `1.60 ms`，但 MCKP mean 增加 `0.83 ms`；
每两层只额外降低 `0.28 ms`，MCKP mean 增加 `0.27 ms`。逐层候选的模拟
关键路径收益仍大于纯 MCKP CPU 开销增量，但当前 Python serial-choice
route oracle 会对两张卡重复规划并扫描 future-K，约 `17 ms/request`，
不能视为生产 controller 开销。生产实现应缓存 configuration curves，并
共享候选 GPU 的 future request PSE 结果。

transition-weight 消融显示联合 placement 不能退化为只优化当前请求：

```text
step=1, lambda=0.0   381.00 ms   +0.06%
step=1, lambda=0.1   371.45 ms   +2.57%
step=1, lambda=0.2   370.84 ms   +2.73%
```

因此当前 profile/trace 下的最优配置更新为：

```text
joint routing + MCKP/Page-greedy
configuration-step = 1
lookahead-K = 8
transition-weight = 0.2
transition-credit-cap = 100 ms
```

其 exposed load 为 `84.57 ms`，H2D 为 `137.22 ms`，missing pages 为
`50197`；相对 Minimal 的 `381.24 ms`，critical path 降至 `370.84 ms`
（`+2.73%`），也优于固定 Minimal routing 的 coarse K=8 `371.25 ms`
（`+2.62%`）。收益差只有 `0.11` percentage point，因此仍应将固定路由
结果保留为纯 cache-policy 对照。

复现逐层最优配置：

```bash
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
  --output docs/layerweave-joint-cache-sim-1000-step1-lambda02.json
```

### 14.10 Pool-pages matched sweep

容量扫描必须固定请求输入长度。原模拟器使用 `pool_pages` 同时限制请求输入，
直接增大 pool 会放宽长请求 input cap，导致 workload 不匹配。新增
`--request-cap-pool-pages` 后，以下扫描全部使用 672-page workload，只改变
每张 GPU 的实际 weight/KV pool capacity：

```text
pool  Minimal CP  Joint CP  gain       Minimal/Joint exposed  missing pages
672   381.24 ms   370.84 ms  10.40 ms   94.97 / 84.57 ms       52256 / 50197
704   372.64 ms   361.85 ms  10.79 ms   86.37 / 75.58 ms       48853 / 45750
736   362.92 ms   354.09 ms   8.83 ms   76.65 / 67.82 ms       43948 / 41889
768   355.16 ms   347.87 ms   7.29 ms   68.89 / 61.60 ms       38694 / 37185
832   343.82 ms   333.62 ms  10.20 ms   57.55 / 47.34 ms       32734 / 29661
896   332.15 ms   327.33 ms   4.82 ms   45.88 / 41.06 ms       26618 / 25191
```

相对 Minimal 的提升依次为：

```text
672: 2.73%
704: 2.90%
736: 2.43%
768: 2.05%
832: 2.97%  best relative gain
896: 1.45%
```

收益不随容量单调变化，因为路由是离散决策，容量变化会改变每个请求后的两卡
residency 演化。容量较大时 Minimal 自身 exposed load 和 miss 持续下降；
到 896 页时 893/1000 个请求在两策略间 critical path 相同，联合策略的机会
空间明显收缩。832 页仍存在足够 cache conflict，同时 Joint 将 missing pages
额外减少 3073 页，因而得到最大相对收益。各点 Joint MCKP mean 为
`1.16--1.36 ms/request`；当前未优化的 Python serial-choice 两卡候选评估为
`16.4--17.5 ms/request`。

复现单个容量点（替换 `POOL_PAGES`）：

```bash
POOL_PAGES=832

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_pipeline_cache_sim.py \
  --policy-suite mckp-page-greedy \
  --max-requests 1000 \
  --gpus 2 \
  --pool-pages "${POOL_PAGES}" \
  --request-cap-pool-pages 672 \
  --lookahead-k 8 \
  --joint-routing \
  --configuration-step 1 \
  --transition-weight 0.2 \
  --transition-credit-cap-ms 100 \
  --output "docs/layerweave-pool-sweep-${POOL_PAGES}.json"
```

### 14.11 Model-switch-conditioned replay

全请求平均会包含连续同模型 full-hit 请求。它们是真实 workload 的组成部分，
但在当前串行模拟器中不产生 weight loading，会用 hot compute 样本稀释
cache/placement 的 model-switch 收益。新增：

```text
--trace-mode model-switches  # 默认；每个连续同模型 run 只保留首请求
--trace-mode all             # 显式选择全请求 workload average
```

`--max-requests` 始终先限制 source trace 行数，再执行过滤。因此前 1000 条
原始请求在 `model-switches` 模式下变为 704 个请求，保持相同 source window，
而不是继续读取 trace 直到凑满 1000 次切换。

在 672 pages、2 GPU、scale=4、逐层 configuration、K=8、
transition-weight=0.2 下：

```text
metric                    Minimal       Joint         gain
critical path mean        432.46 ms     407.38 ms     25.08 ms / 5.80%
exposed load mean         138.43 ms     113.36 ms     25.08 ms
H2D mean                  218.95 ms     184.92 ms     34.03 ms
missing pages             56396         47594         8802
```

这个结果应命名为 `switch-conditioned service critical path`，用于回答“发生
model activation/switch 时，联合策略减少多少 loading stall”。它不能替代
保留全部请求的 workload-average `2.73%`，否则等价于有选择地删除真实
cache-hit 请求。正式报告应并列给出：

```text
all requests average:          +2.73%
model-switch-conditioned:      +5.80%
```

复现命令：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/layerweave_pipeline_cache_sim.py \
  --policy-suite mckp-page-greedy \
  --max-requests 1000 \
  --trace-mode model-switches \
  --input-scale 4 \
  --gpus 2 \
  --pool-pages 672 \
  --lookahead-k 8 \
  --joint-routing \
  --configuration-step 1 \
  --transition-weight 0.2 \
  --transition-credit-cap-ms 100 \
  --output docs/layerweave-model-switch-only-1000-source-joint.json
```

### 14.12 Matched token-shape trace variants

`tools/layerpipe/build_layerweave_trace_variants.py` 从原 ServeGen trace 为每个
模型分别提取 `(input_tokens, output_tokens)` 经验分布。派生 trace 只改变
model sequence；每次选择模型后，从该模型自己 seed-shuffled 的 shape pool
循环取样，避免把输入长度变化错误归因给 routing/cache。

生成四类 1000-request trace：

- `balanced_iid`：八模型均匀、禁止连续相同；
- `round_robin`：每八请求包含每个模型一次，block 内随机排列；
- `phase_shift`：五个阶段，每阶段三个模型占 70%，热点集合迁移；
- `working_set_shift`：每 100 请求切换一个四模型工作集，集合部分重叠。

统一使用 2 GPU、672 pages、scale=4、逐层 configuration、K=8、
transition-weight=0.2：

```text
trace              mean gain   P90 gain   P95 gain   P99 gain
original switches     5.80%       23.06%      10.85%      0.00%
balanced IID         10.93%        0.43%      -1.36%      0.00%
round robin           8.63%       14.08%      14.66%      0.00%
phase shift          13.09%        0.01%       6.60%      0.00%
working-set shift    10.85%        3.05%     -18.17%      9.52%
```

`phase_shift` 的 mean 从 Minimal `542.19 ms` 降为 `471.23 ms`，
exposed load 从 `270.87` 降为 `199.92 ms`，missing pages 从 `140527`
降为 `108580`。它是当前最大 mean 且 P95 不退化的配置。

`round_robin` 的 mean 从 `703.42` 降为 `642.72 ms`，P90 从 `1424.41`
降为 `1223.91 ms`，P95 从 `1670.79` 降为 `1425.82 ms`。它是当前
P90/P95 最强且 mean 同时改善的配置。

额外组合搜索：

```text
combination                    mean     P90      P95      P99
phase, scale=1, pool=672      +16.51%   -6.20%   -0.09%  -21.60%
round-robin, scale=1, 672      +8.61%   +0.69%  +12.29%   +0.22%
phase, scale=4, pool=704      +10.42%   +0.15%   -3.62%  -16.19%
phase, scale=4, pool=832       +9.11%  +14.16%   -4.99%   -9.71%
```

因此不能把 mean 最大的 `phase + scale=1` 称为综合最优：它通过缩短
compute 放大 mean loading 比例，但改变了路由状态后 P90/P99 明显退化。
当前推荐并列报告：

```text
mean-oriented:  phase-shift, scale=4, pool=672
tail-oriented:  round-robin, scale=4, pool=672
```

所有测试中 P99 经常等于相同的 `1675.23 ms`，因为该分位被长输入 compute
或 cold-like 大模型请求主导。cache placement 可以显著减少 exposed-load
mean/P90/P95，但不能减少 hot compute；P99 无改善是机制边界，不应通过删除
尾部请求隐藏。

生成 trace：

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/build_layerweave_trace_variants.py \
  --source evaluation/traces/servegen_tangram.trace \
  --output-dir evaluation/traces/layerweave_variants \
  --requests 1000 \
  --seed 1234
```

当前两条推荐结果：

```text
docs/layerweave-trace-phase_shift-joint.json
docs/layerweave-trace-round_robin-joint.json
```

### 14.13 Current Joint five-way mean/P90 ablation

将当前 Joint LayerWeave
（逐层 configuration、K=8、transition-weight=0.2）加入传统机制分解。
Baseline/Pipe-only/Reuse-only/Minimal 使用 Minimal routing/cache state；
LayerWeave 使用自己的 joint routing/cache placement，因此最后一级是系统级
联合优化，不是固定路由下只替换 victim policy。

Mean-oriented `phase_shift`：

```text
stage                    mean        P90
baseline serialized      1089.73 ms  2124.22 ms
pipe-only                 846.76 ms  1675.23 ms
reuse-only                654.93 ms  1437.81 ms
Minimal reuse + pipe      542.19 ms  1203.52 ms
Joint LayerWeave          471.23 ms  1203.42 ms
```

Joint LayerWeave 相对各项：

```text
comparison               mean reduction   P90 reduction
vs baseline                 56.76%            43.35%
vs pipe-only                44.35%            28.16%
vs reuse-only               28.05%            16.30%
vs Minimal                  13.09%             0.01%
```

Tail-oriented `round_robin`：

```text
stage                    mean        P90
baseline serialized      1104.23 ms  2124.22 ms
pipe-only                 850.46 ms  1675.23 ms
reuse-only                888.43 ms  1825.83 ms
Minimal reuse + pipe      703.42 ms  1424.41 ms
Joint LayerWeave          642.72 ms  1223.91 ms
```

Joint LayerWeave 相对各项：

```text
comparison               mean reduction   P90 reduction
vs baseline                 41.79%            42.38%
vs pipe-only                24.43%            26.94%
vs reuse-only               27.66%            32.97%
vs Minimal                   8.63%            14.08%
```

对应的完整 ablation JSON：

```text
docs/layerweave-phase-shift-current-joint-five-way.json
docs/layerweave-round-robin-current-joint-five-way.json
```

### 14.14 Aegaeon whole-model lookahead prefetch comparison

模拟器新增 Aegaeon 风格的整模型双缓冲预取。对每个 GPU，scheduler 已知
该 GPU 的下一条请求；当前请求 decode 时，在独立 copy stream 中预取下一
模型。只有以下容量条件成立时才启用预取：

```text
current model pages + current request KV pages + next model pages
    <= pool pages
```

切换时的 loading stall 定义为：

```text
max(0, whole-model H2D - real trace output_tokens * 40 ms/token)
    + prefetched-model GPU relocation
```

GPU relocation 默认用 `864 GB/s` 的可配置有效带宽估计。若双模型容量不足，
不授予 overlap，下一模型直接付出完整 whole-model H2D；若同一 GPU 的相邻
请求模型相同，则 loading stall 为 0。Aegaeon 不继承 LayerWeave 的多模型
页缓存或逐层 H2D/compute pipeline。

注意：默认 `--output-tokens-override 1` 仍用于当前模拟器的 KV 请求口径；
trace 第四列的真实 decode 长度只用于 Aegaeon overlap window。这样可以在
不改变既有 LayerWeave workload 的情况下，单独补上 Aegaeon 的 decode
掩盖机会。

完整运行命令：

```bash
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
  --decode-ms-per-token 40 \
  --aegaeon-gpu-copy-gbps 864 \
  --output docs/layerweave-aegaeon-1000-joint-k8.json
```

默认 `model-switches` trace 模式下，1000 条 source rows 留下 704 条请求。
Aegaeon 回放当前 Joint LayerWeave 的完全相同 routing：

```text
metric                 Aegaeon       Joint LayerWeave   LW reduction
loading stall mean      276.41 ms       113.36 ms          58.99%
loading stall P50        17.63 ms         0.00 ms         100.00%
loading stall P90       652.00 ms       341.96 ms          47.55%
loading stall P95      1195.96 ms       599.12 ms          49.90%
loading stall P99      1658.44 ms      1216.04 ms          26.68%
critical path mean      570.43 ms       407.38 ms          28.58%
```

Aegaeon 的 702 个非首请求中，406 次整模型双缓冲容量可行，158 次被容量
阻挡，另有 138 次同 GPU 相邻请求为同模型；283 次预取的 H2D 被 decode
完全掩盖。即使 H2D 完全掩盖，仍要支付 GPU relocation，所以 Aegaeon P50
不是 0。当前结果说明 LayerWeave 的优势主要来自：

1. 不要求同时容纳两个完整模型；
2. 保留跨模型的页级复用；
3. 只加载缺页并把 suffix H2D 与逐层 compute 重叠；
4. 避免整模型预取区到参数 Buffer 的完整 GPU relocation。

结果文件：

```text
docs/layerweave-aegaeon-1000-joint-k8.json
```
