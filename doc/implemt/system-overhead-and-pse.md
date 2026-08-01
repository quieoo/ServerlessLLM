# Tangram system overhead 与 PSE accuracy 实现及 smoke test

## 1. 实验范围

本轮只实现两项：

1. CUDA event/readiness-hook warm Prefill overhead；
2. PSE predicted exposed loading stall 与真实 exposed loading stall 的配对统计。

ODKV overhead 不在本轮重复测试，使用已有实验结果。

## 2. Warm Prefill overhead

### 2.1 三条路径

- `native`：原始 vLLM，使用本地 Hugging Face checkpoint，不创建 Tangram VMM pool/controller；
- `vmm-resident`：使用 stable VMM 参数地址，运行前加载并保护全部参数页，然后移除全部 model/layer readiness hooks；
- `layerweave-resident`：同样加载并保护全部参数页，但保留完整 residency query、CUDA event record/wait 和 readiness hooks。

主要归因量为：

```text
readiness hook/event overhead
    = LayerWeave-resident warm Prefill
    - VMM-resident warm Prefill
```

该差值的两侧使用相同的 vLLM、VMM、ODKV、checkpoint 和请求，因此是最干净的
readiness 机制开销。`native` 与两条 Tangram 路径的 KV backend 不同，native
对照仅用于报告完整系统路径差值；如需进一步分解，应结合已有 ODKV overhead
结果，不能把 `VMM-resident - native` 全部归因为 stable VMM。

### 2.2 实现

- `TangramLayerWeaveController.make_fully_resident()`：同步加载并 full-protect
  当前模型全部 required weight pages，加载后再次检查 residency；
- `set_readiness_hooks_enabled(False)`：只允许在 fully-resident setup 之后用于
  overhead ablation，移除 model/layer pre/post hooks；
- `vllm_odkv_trace_bench.py --warm-resident-mode hooks|no-hooks`：要求 trace
  只包含一个模型，防止多个 full-protected 模型超过 pool；
- `vllm_warm_prefill_native_bench.py`：原始 vLLM 单模型 warm Prefill runner；
- `summarize_warm_prefill_overhead.py`：跳过指定 warmup 后，统一报告 mean、P95、
  max 和三项差值。

### 2.3 Smoke test

环境：L40，model 0，input=128，output=1，6 次请求，跳过第 1 次冷启动样本。
输入为 `doc/implemt/system-overhead-smoke-model0.trace`。

| 路径 | post-warmup N | mean Prefill | P95 | max |
|---|---:|---:|---:|---:|
| Native vLLM | 5 | 21.081 ms | 22.159 ms | 22.409 ms |
| VMM-resident/no-hooks | 5 | 30.338 ms | 33.947 ms | 34.957 ms |
| LayerWeave-resident/hooks | 5 | 83.059 ms | 85.066 ms | 85.501 ms |

Smoke 差值：

- readiness hook/event：52.720 ms/request；
- LayerWeave-resident 相对 native：61.978 ms/request。

这只是功能 smoke，不作为论文结果：样本只有 5 个，而且三条路径运行在不同的
空闲 L40 上。正式实验必须在同一张 GPU 上顺序轮转三条路径，并增加重复次数。
不过 smoke 已确认：两条 resident 路径的 `to_load_bytes=0`；no-hooks 路径没有
demand-load stage；hooks 路径完整执行 36 个 decoder-layer readiness stage。

值得注意的是，resident hooks 路径虽然传输字节为 0，仍记录约 17--18 ms 的
stage event interval。这不是 H2D 数据传输，而是逐 stage residency query、
prepare fast path 和 event 操作被当前 `h2d_ms` event interval 包含。正式论文中
应将本实验差值称为 readiness mechanism overhead，而不是传输时间。

结果文件：

- `doc/implemt/system-overhead-smoke-native.json`
- `doc/implemt/system-overhead-smoke-no-hooks.json`
- `doc/implemt/system-overhead-smoke-hooks.json`
- `doc/implemt/system-overhead-smoke-summary.json`

正式运行建议每个模型使用同一 GPU：10 次 warmup + 100 次正式请求。8 个模型
分别生成单模型 trace；不要在同一进程 full-protect 多模型。

## 3. PSE predicted-vs-measured exposed loading stall

### 3.1 配对口径

每次 routing estimate 新增：

```text
predicted_exposed_load_ms
    = PipelineEstimator.predicted_gpu_ready_stall_ms
```

请求真实执行后新增：

```text
measured_exposed_load_ms
    = TangramLayerWeaveController.metrics().exposed_load_ms
pse_exposed_error_ms
    = predicted_exposed_load_ms - measured_exposed_load_ms
```

二者写入同一个 `batch_metric`，由 request/batch ID 一一配对，不再用 service
TTFT、queue time 或总 H2D 代替 exposed loading stall。

`analyze_pse_exposed_accuracy.py` 输出：MAE、median/P95 absolute error、bias、
Pearson correlation 和 MAPE。真实值小于 1 ms 的样本默认不进入 MAPE，避免
接近零的 full-hit 样本使相对误差发散；它们仍进入 MAE、bias 和 correlation。
脚本同时按 model 和 routing 时的 residency class 汇总。

### 3.2 两请求 smoke

使用匹配的 `docs/m4-m5.5-b1-scale4/m4-full-estimator-report.json`，2 GPU、
Joint policy、serial-choice、input scale 4、output 1。两条请求均为 model 2。

| Request | Predicted | Measured | Signed error |
|---:|---:|---:|---:|
| 0 | 635.005 ms | 598.235 ms | +36.769 ms |
| 1 | 0.132 ms | 1.206 ms | -1.074 ms |

Smoke 汇总：MAE 18.922 ms，bias +17.848 ms。两点计算出的 correlation 和 MAPE
没有统计意义，仅用于确认 cold/full-hit 两种记录都能正确落盘并被分析器读取。

### 3.3 Partial-hit 样本

使用 `model 5 -> model 4 -> model 5` 并固定在同一 GPU 的三请求序列构造部分
驻留。第三次访问 model 5（GPT-20B）时：

- required pages：613；cached pages：214；missing pages：399；
- page hit rate：`214 / 613 = 34.91%`；
- PSE predicted exposed loading stall：966.793 ms；
- GPU measured exposed loading stall：941.896 ms；
- signed error：+24.897 ms，即高估 2.64%。

该样本使用 64 MiB page，对应约 13.375 GiB cached、24.938 GiB missing。
原始结果为 `doc/implemt/pse-partial-hit-smoke.json`，汇总为
`doc/implemt/pse-partial-hit-smoke-summary.json`。

结果文件：

- `doc/implemt/pse-exposed-smoke-r2.json`
- `doc/implemt/pse-exposed-smoke-summary.json`
- `doc/implemt/pse-exposed-smoke-summary.csv`

## 4. 验证状态

- Python compile 通过；
- `test_layerweave_joint_policy.py` 和
  `test_layerweave_multi_gpu_scheduler.py`：16 tests passed；
- 三条 warm Prefill 路径均在本地 L40 跑通；
- PSE 两请求真实 GPU 配对链路跑通。

下一步正式实验前，建议先将 overhead runner 扩展成同一 GPU 上按
`native -> no-hooks -> hooks` 多轮交错执行，消除 GPU 间和时间漂移；PSE 则先用
约 50--100 条请求检查误差分布，再决定正式 trace 长度。

## 5. Warm Prefill overhead 正式实验

### 5.1 执行设置

- 4 张 L40 并行；model 0/4、1/5、2/6、3/7 分别固定在 GPU 0/1/2/3；
- 同一个 `model × input length` 的三条路径在同一 GPU 上串行执行；
- 三路径执行顺序随 cell 轮转，减少固定顺序偏差；
- 每条路径 10 次 warmup + 50 次正式样本，output tokens 固定为 1；
- 目标 input lengths 为 128、1024、4096、16384；
- 如果 `input + output` 超过配置中的 L40 safe limit，cell 明确标记为
  `infeasible`，不静默截断；
- 运行器支持按完整 JSON 断点续跑。LLaVA Native fast-tokenizer 兼容问题修复后，
  仅补跑缺失 cell，没有重复已完成的 GPU 工作。

运行器：`doc/implemt/run_system_overhead_formal.py`。总表：
`doc/results/system-overhead-formal/summary.csv`。正式实验共完成 21 个可行 cell，
三路径各获得 1050 个 post-warmup 样本。所有 hooks/no-hooks resident 样本的
`to_load_bytes` 均为 0。

当前 8 个 checkpoint 的 L40 safe input limit 为 1344--12921，因而没有模型能
安全执行 16K。128 和 1K 覆盖全部 8 个模型；4K 覆盖 models 0/2/3/4/7；其余
cell 与全部 16K cell 保留为 `infeasible`。将来替换长上下文模型后，可直接用
同一 runner 补齐 16K。

### 5.2 正式结果

下表中的 delta 为 `LayerWeave-resident - VMM-resident/no-hooks`：

| Model | Input | Native | VMM resident | LW resident | Delta | Delta / VMM |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 33.443 | 28.255 | 58.807 | 30.552 | 108.13% |
| 0 | 1024 | 49.764 | 55.322 | 70.183 | 14.861 | 26.86% |
| 0 | 4096 | 210.836 | 210.387 | 220.577 | 10.190 | 4.84% |
| 1 | 128 | 25.025 | 31.367 | 75.434 | 44.067 | 140.49% |
| 1 | 1024 | 68.113 | 86.526 | 85.572 | -0.954 | -1.10% |
| 2 | 128 | 31.982 | 34.798 | 71.657 | 36.859 | 105.92% |
| 2 | 1024 | 120.067 | 121.624 | 126.632 | 5.008 | 4.12% |
| 2 | 4096 | 491.868 | 503.096 | 509.403 | 6.307 | 1.25% |
| 3 | 128 | 32.919 | 34.472 | 83.981 | 49.509 | 143.62% |
| 3 | 1024 | 108.916 | 110.291 | 114.029 | 3.738 | 3.39% |
| 3 | 4096 | 453.694 | 443.610 | 456.168 | 12.559 | 2.83% |
| 4 | 128 | 55.589 | 57.929 | 142.302 | 84.373 | 145.65% |
| 4 | 1024 | 228.335 | 224.992 | 235.877 | 10.885 | 4.84% |
| 4 | 4096 | 925.915 | 929.899 | 936.843 | 6.944 | 0.75% |
| 5 | 128 | 80.622 | 83.366 | 166.611 | 83.246 | 99.86% |
| 5 | 1024 | 355.812 | 373.021 | 386.538 | 13.517 | 3.62% |
| 6 | 128 | 27.677 | 29.001 | 68.208 | 39.206 | 135.19% |
| 6 | 1024 | 106.161 | 109.121 | 114.142 | 5.022 | 4.60% |
| 7 | 128 | 55.403 | 59.009 | 135.718 | 76.709 | 130.00% |
| 7 | 1024 | 219.774 | 219.775 | 227.331 | 7.556 | 3.44% |
| 7 | 4096 | 918.049 | 918.968 | 910.029 | -8.940 | -0.97% |

按输入长度聚合：

| Input | Models | Delta mean | Delta median | Relative median |
|---:|---:|---:|---:|---:|
| 128 | 8 | 55.565 | 46.788 | 132.59% |
| 1024 | 8 | 7.454 | 6.289 | 3.87% |
| 4096 | 5 | 5.412 | 6.944 | 1.25% |

### 5.3 解释与边界

正式结果支持两个结论：

1. readiness path 的原始工作近似按 layer/page 产生，在短 Prefill 中无法被 GPU
   计算掩盖。128-token 下 delta 与 layer count 的 Pearson correlation 为 0.850，
   与 required page count 为 0.920；样本数只有 8，不能解释为普适线性模型。
2. 输入增长后单层 GPU compute 变长，大部分 CPU hook/residency/prepare 提交成本
   被隐藏。相对增量的中位数从 128-token 的 132.59% 降至 1K 的 3.87% 和 4K
   的 1.25%。

model 1@1K 和 model 7@4K 出现约 -1% 的负 delta。readiness 机制不可能使 fully
resident compute 本身更快，因此这些值应解释为独立进程初始化、GPU frequency
和运行时抖动已经大于真实差值；不能宣称负 overhead。论文中可以将绝对值低于
约 1% 的 cell 描述为 statistically negligible，并展示 mean/P95/max 原始结果。

另一个重要边界是 Native 使用 original vLLM KV，而两个 resident Tangram 路径
使用相同 ODKV。因此本实验中最可信的机制归因是 hooks 与 no-hooks 的差值；
`VMM-resident - Native` 仍混有已有的 ODKV path 差异。
