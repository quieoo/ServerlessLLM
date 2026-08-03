# Evaluation Plan 

## Evaluation Setup 

- 4*L40 GPU
- 8 模型
- 默认 ServeGen workload；
- Baselines；
    SLLM；
    Pipe-Only；
    Aegaeon；
    Tangram。
- 默认系统参数和指标

## Overall Performance


## Stall-Guided Reuse Policy

### trace影响

6种trace：
- servegen: 默认配置
- cyclic: 每8个请求包含8个模型各一次，块内随机排列
- reuse-low: 8个模型均衡，distinct reuse distance为2
- reuse-high: 8个模型均衡，distinct reuse distance为7
- small-hot: 在servegen基础上，将高频rank映射到小模型
- large-hot: 在servegen基础上，将高频rank映射到大模型

默认segment后端

策略组合(同一个图上的不同线，一共四条线)：
- Cache-bytes路由 + LRU
- Cache-bytes路由 + Suffix-LRU
- Cache-bytes路由 + MCKP-prefix
- MCKP-transition路由 + MCKP-prefix

扫不同的trace配置（一个配置有一张图，一共四张图）：
- reuse distance
- input length
- 模型大小与热点相关性
- GPU数量

## Memory-Backend Trade-offs
三种backend：
- segment
- page-align
- page-pack

汇报 exposed loading的分解：H2D + Segment Compaction + Page map/unmap + Allocation
其他汇报：
- effective resident TG bytes； fragmentation/unreclaimable bytes； 或有效缓存容量占分配容量的比例

测试维度：
- memory pressure
  主要的 end-to-end backend
- page size
  用segment作为参考线
- KV 压力
  通过 batch size 或生成长度控制 KV 增长，报告： 为 KV 请求释放的逻辑容量； 实际成功释放的物理容量； reclamation latency； ODKV TPOT overhead。 这对 Page-Packed 尤其重要，因为共享边界页可能导致“逻辑上淘汰了 TG，但物理上无法释放全部容量”。
- 工作负载局部性和模型切换模式
  把“工作负载局部性和模型切换模式”更准确地改为：  Memory churn / prefix-transition frequency
  例如改变： model switch interval = 1, 4, 16, 64 requests
  或控制每单位时间发生的 TG load/eviction 次数。
  一般的 locality 会同时改变 cache hit、H2D volume 和 replacement decision，难以隔离 backend。Backend 真正关心的是：
    分配和释放频率；
    TG 大小异质性；
    prefix 状态变化频率；
    空洞和共享边界的形成速度。
  最好生成一组固定的逻辑 allocation/reclamation sequence，让三个 backend 执行完全相同的 TG 状态变化。


## Overheads
  - CUDA event/readiness-hook 开销；
    比较 warm Prefill： 原始 vLLM； 加入所有 hook/event，但参数全部 resident。
    结果: 对短prefill (256) 影响大, 整体Prefill时间增加17 ms, 证明了Pipeline类工作对短输入的不适用.当输入长度增加到1k以后,增量降低到6.2ms,整体影响不超过3.8%. 
  - ODKV开销
    通过delayed release, batched allocation, and coarse-grained block management降低申请频率. 虽然会调用MCKP,但是计算时间不超过6ms, 当kv block size增加到32 token/block时, decode时延开销不超过1.4%. 
  - PSE：预测误差
    测了不同情况下PSE预测和实测的误差,在cold, partial-hit, full-hit之下相对误差在2%到6%之间.

