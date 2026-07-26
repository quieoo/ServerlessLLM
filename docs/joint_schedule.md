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
