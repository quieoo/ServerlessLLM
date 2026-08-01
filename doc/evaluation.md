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

## Overall Activation Performance
我觉得Overall Activation Performance 主图 还是保留之前SIGMOD版本的那个包含Init-Profile-Prefill-Load的版本，同时只比SLLM， SLLM-C， SLLM-CM，Tangram。这个是为了看到最后Tangram对于不同模型在总TTFT上面的优化。
你说的那个SLLM；Pipe-Only；Aegaeon-inspired；Tangram比较，画出来并不好看，因为他们的前三个阶段都一样。
然后再加一个副图，聚焦于“SLLM；Pipe-Only；Aegaeon-inspired；Tangram”的Exposed Loading Time的比较

```bash
nohup bash doc/1.overall.sh \
  > doc/results/overall/corrected-busyp0p5-req5000_launcher.log \
  2>&1 &
echo $!

```

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

关于backends，我做了两个实验一个是在默认trace下面的exposed loading stall的分解，一个是扫了不同的page size对不同的page类型的影响结果如下。关于breakdown结果，基于page的方法各自使用最优的page-size. Segment的H2D(CPU-GPU数据传输)的时延最小, 因为他拥有最完整的Tenser语义,它传输的数据量最小,分别比Aligned少11%,比compact少10%. Aligned是因为有额外的内部碎片,导致空间利用率,缓存命中率下降,而Compact则由于与tensor语义不对齐,导致过量驱逐,后续需要重复加载更多权重. 另外,得益于PGP算法, Compaction的时间开销不超过5ms/request. 基于Page的方法需要实时调用CUDA VMM API来申请和释放物理空间, 由于page-compact使用的page-size更大,因此开销更低. 关于page-size的结果, 小页面受 map/unmap 主导.随着page size增加, aligned的性能逐渐提高,16MB时候达到最高,但是32MB时候过大的内存碎片导致它无法再运行最大的20B模型. Page-packed没有这个问题,因此他的page size可以继续扩大,在32MB时获得他的最优性能,在这之后,驻留和驱逐粒度变粗，造成过量驱逐及重复 H2D. 除了这两个实验,你觉得还有其他必须要做的实验吗? 
Backend	H2D	Compaction	Map/Unmap	MCKP 

Segment	68.868	2.755	0	6.77

Page-Aligned	80.221	0	10.6	7.29

Page-Packed	80.182	0	1.274	5.55		

Page MiB	Tensor-page exposed	Compact-page exposed

1	195.137	192.856

2	148.495	143.503

4	126.014	123.441

8	115.764	111.265

16	103.438	105.213

32	Infeasible	97.94

64	Infeasible	100.835

128	Infeasible	101.284



## Overheads
  - CUDA event/readiness-hook 开销；
    比较 warm Prefill： 原始 vLLM； 加入所有 hook/event，但参数全部 resident。
    结果: 对短prefill (256) 影响大, 整体Prefill时间增加17 ms, 证明了Pipeline类工作对短输入的不适用.当输入长度增加到1k以后,增量降低到6.2ms,整体影响不超过3.8%. 
  - ODKV开销
    通过delayed release, batched allocation, and coarse-grained block management降低申请频率. 虽然会调用MCKP,但是计算时间不超过6ms, 当kv block size增加到32 token/block时, decode时延开销不超过1.4%. 
  - PSE：预测误差
    测了不同情况下PSE预测和实测的误差,在cold, partial-hit, full-hit之下相对误差在2%到6%之间.




### Exposed Loading Time



### SLO Attainment

#### SLO Scale
```bash
cd /mnt/n0/Tangram/Tangram

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/plot_slo_scale_from_critical_path.py \
  --input-dir doc/results/end2end/corrected-busyp0p5 \
  --base-slo-ms 100 \
  --scale-start 1 \
  --scale-stop 30 \
  --scale-step 0.25 \
  --csv-output \
    doc/results/end2end/corrected-busyp0p5/slo-attainment-vs-scale.csv \
  --output \
    doc/results/end2end/corrected-busyp0p5/slo-attainment-vs-scale.svg

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/plot_slo_scale_from_critical_path.py \
  --input-dir doc/results/end2end/corrected-busyp0p5-req5000 \
  --base-slo-ms 250 \
  --scale-start 1 \
  --scale-stop 13 \
  --scale-step 1 \
  --csv-output \
    doc/results/end2end/corrected-busyp0p5-req5000/slo-attainment-vs-scale.csv \
  --output \
    doc/results/end2end/corrected-busyp0p5-req5000/slo-attainment-vs-scale.svg
```

#### Request Scale

```bash
cd /mnt/n0/Tangram/Tangram

nohup /home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/run_gpu_busy_probability_sweep.py \
  --max-workers 88 \
  --output-dir doc/results/end2end/gpu-busy-sweep \
  --resume \
  > doc/results/end2end/gpu-busy-sweep.log 2>&1 &
```

```bash
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/plot_slo_scale_from_critical_path.py \
  --input-dir doc/results/end2end/gpu-busy-sweep/busyp0p0 \
  --base-slo-ms 250 \
  --scale-start 1 \
  --scale-stop 13 \
  --scale-step 1 \
  --csv-output \
    doc/results/end2end/corrected-busyp0p0-req5000/slo-attainment-vs-scale.csv \
  --output \
    doc/results/end2end/corrected-busyp0p0-req5000/slo-attainment-vs-scale.svg

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/plot_slo_scale_from_critical_path.py \
  --input-dir doc/results/end2end/gpu-busy-sweep/busyp0p5 \
  --base-slo-ms 250 \
  --scale-start 1 \
  --scale-stop 13 \
  --scale-step 1 \
  --csv-output \
    doc/results/end2end/corrected-busyp0p5-req5000/slo-attainment-vs-scale.csv \
  --output \
    doc/results/end2end/corrected-busyp0p5-req5000/slo-attainment-vs-scale.svg

/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/plot_slo_scale_from_critical_path.py \
  --input-dir doc/results/end2end/gpu-busy-sweep/busyp1p0 \
  --base-slo-ms 250 \
  --scale-start 1 \
  --scale-stop 13 \
  --scale-step 1 \
  --csv-output \
    doc/results/end2end/corrected-busyp1p0-req5000/slo-attainment-vs-scale.csv \
  --output \
    doc/results/end2end/corrected-busyp1p0-req5000/slo-attainment-vs-scale.svg
```