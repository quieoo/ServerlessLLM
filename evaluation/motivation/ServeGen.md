# Setup
````bash
cd uv_envs/servegen
uv venv --python 3.12
source /mnt/n0/sslm/uv_envs/servegen/.venv/bin/activate

git clone https://github.com/alibaba/ServeGen.git
cd ServeGen
uv pip install -r requirements.txt
uv pip install -e .
````

# quick start (examples)
````bash
source /mnt/n0/sslm/uv_envs/servegen/.venv/bin/activate
cd ServeGen
python basic_usage.py
# 指定模型，指定req/s, 指定duration(可以指定不同时间窗口上的req/s)，生成workload.csv
# # request_id,timestamp,input_tokens,output_tokens

python generate_advanced.py
# 同样指定req/s, duration，但是输出不同。多模态会额外包含模态信息，推理会输出reason_ratio

python generate_custom.py
# 使用自定义client，可以指定CV，Gamma分布参数，input tokens 、 output tokens

python generate_realistic.py
# 指定时间段，按照原始client行为生成workload；有一个bounded_rate, 只指定上限，然后整体收缩，不改变原来的分布

clientpool_example.py
# 测试这批客户端在某个时间点的速率、波动、输入输出长度大致是什么分布

````

# multiple models (废弃)

````bash
cd ServeGen/examples
python multi_model_locality.py
````

```
Generating per-model workloads...
Using 8 logical models out of 8 total
  Pool 0 (lang-small-0 -> m-small): 3142035 requests, view=39600-43200, strategy=bounded(939.19)
  Pool 1 (lang-small-1 -> m-small): 2923299 requests, view=43200-46800, strategy=bounded(861.32)
  Pool 2 (lang-mid-0 -> m-mid): 5953124 requests, view=39600-43200, strategy=bounded(1834.53)
  Pool 3 (lang-mid-1 -> m-mid): 4634802 requests, view=43200-46800, strategy=bounded(1473.85)
  Pool 4 (lang-large-0 -> m-large): 1135756 requests, view=39600-43200, strategy=bounded(356.88)
  Pool 5 (lang-large-1 -> m-large): 713628 requests, view=43200-46800, strategy=bounded(213.95)
  Pool 6 (mm-image-0 -> mm-image): 42563 requests, view=39600-43200, strategy=bounded(12.58)
  Pool 7 (reason-r1-0 -> deepseek-r1): 47761 requests, view=39600-43200, strategy=bounded(16.33)

Merged workload size: 18592968
Saved merged workload to multi_model_workload.csv

Model locality statistics:

Pool 0:
  Samples: 3142034
  Access Interval Counts: {0: 775158, 1: 651629, 2: 611489, 3: 548040, 4: 385321, 5: 159338, 6: 10799, 7: 260}
  CDF: [0.24670579630901512, 0.45409661384950006, 0.6487122672765476, 0.8231343136325069, 0.9457685690224867, 0.9964803054327229, 0.9999172510545716, 1.0]

Pool 1:
  Samples: 2923298
  Access Interval Counts: {0: 797246, 1: 524994, 2: 515059, 3: 500716, 4: 388380, 5: 182520, 6: 14009, 7: 374}
  CDF: [0.27272142627949664, 0.4523110541586934, 0.6285021232867809, 0.799786747707555, 0.9326435416437189, 0.9950798721170404, 0.9998720623077086, 1.0]

Pool 2:
  Samples: 5953123
  Access Interval Counts: {0: 3033952, 1: 1066974, 2: 799948, 3: 586873, 4: 340881, 5: 117342, 6: 6987, 7: 166}
  CDF: [0.5096404021889015, 0.6888696907488725, 0.8232442030846666, 0.9218265774115536, 0.979087447042502, 0.998798445790554, 0.9999721154761962, 1.0]

Pool 3:
  Samples: 4634801
  Access Interval Counts: {0: 2015266, 1: 842732, 2: 695532, 3: 564635, 4: 365866, 5: 141178, 6: 9370, 7: 222}
  CDF: [0.43481176430228613, 0.6166387726247577, 0.7667060570669593, 0.8885311365040268, 0.9674700165120358, 0.9979304397319324, 0.9999521015033871, 1.0]

Pool 4:
  Samples: 1135755
  Access Interval Counts: {0: 118415, 1: 108512, 2: 131815, 3: 174035, 4: 282453, 5: 284321, 6: 34754, 7: 1450}
  CDF: [0.10426104221420993, 0.1998027743659504, 0.315862135759913, 0.4690950072859023, 0.7177868466350578, 0.9681234068967339, 0.9987233162081611, 1.0]

Pool 5:
  Samples: 713627
  Access Interval Counts: {0: 59169, 1: 47035, 2: 57113, 3: 77907, 4: 134070, 5: 288878, 6: 46624, 7: 2831}
  CDF: [0.08291306242616941, 0.1488228444271307, 0.2288548499426171, 0.3380253269565193, 0.5258965818277616, 0.9306990907014449, 0.9960329415787239, 1.0]

Pool 6:
  Samples: 42562
  Access Interval Counts: {0: 6999, 1: 588, 2: 469, 3: 504, 4: 773, 5: 1556, 6: 16674, 7: 14999}
  CDF: [0.164442460410695, 0.17825760067665994, 0.18927681969832244, 0.20111836849772097, 0.2192801090174334, 0.25583854142192564, 0.6475964475353602, 1.0]

Pool 7:
  Samples: 47760
  Access Interval Counts: {0: 5436, 1: 1978, 2: 1543, 3: 1535, 4: 1976, 5: 3249, 6: 17165, 7: 14878}
  CDF: [0.11381909547738693, 0.15523450586264656, 0.18754187604690117, 0.2196817420435511, 0.26105527638190956, 0.3290829145728643, 0.6884840871021776, 1.0]
Saved locality plot to multi_model_locality_cdf.png
```


# request rate over time

````bash
python reqrate_over_time.py
````

```
(servegen) ➜  examples git:(main) ✗ python reqrate_over_time.py
Reconstructed request counts from original trace rates.
This is the closest recovery possible from the released aggregated data; it is not an exact raw-request replay.
Native time window: 600s
Output window size: 600s
Duration: full available range
language/m-small: 2016 windows, time=0-1209000s, total_requests≈1420558786
language/m-mid: 2016 windows, time=0-1209000s, total_requests≈2165253970
language/m-large: 2016 windows, time=0-1209000s, total_requests≈236892826
multimodal/mm-image: 144 windows, time=0-85800s, total_requests≈818610
reason/deepseek-r1: 144 windows, time=0-85800s, total_requests≈1383669
Top-2 access patterns contribute >= 90.00% of requests in 1770/2016 active windows (87.80%).
Saved plot to /mnt/n0/sslm/ServeGen/examples/reqrate_over_time.png
Saved reconstructed counts to /mnt/n0/sslm/ServeGen/examples/reqrate_over_time.csv
```

```
workload,representative_model,kv_bytes_per_token,weight_memory_gib,runtime_essential_memory_gib,mean_gib,p10_gib,p50_gib,p90_gib,p95_gib,p99_gib,p99.9_gib,max_gib,safe_memory_demand_gib
Language-small / Qwen2.5-3B-Instruct,Qwen2.5-3B-Instruct,36864,5.760000,2.000000,0.302045,0.202911,0.285370,0.393249,0.437108,0.530965,2.362043,2.379089,8.153249
Language-mid / Llama-3.1-8B-Instruct,Llama-3.1-8B-Instruct,131072,14.960000,3.000000,1.472091,0.816626,1.302856,2.156213,2.637402,5.018181,8.558814,12.681396,20.116213
Language-large / GPT-20B,GPT-20B,1081344,37.250000,5.000000,21.632168,15.309631,20.632050,29.463336,33.752641,40.842908,46.635975,56.807373,71.713336
Multimodal / Qwen2.5-VL-7B-Instruct,Qwen2.5-VL-7B-Instruct,57344,13.040000,4.000000,1.329913,1.017444,1.292019,1.671445,1.844747,2.303910,2.684554,2.690903,18.711445
Reasoning / DeepSeek-R1-Distill-Qwen-14B,DeepSeek-R1-Distill-Qwen-14B,196608,26.080000,4.000000,8.100773,4.410992,7.567749,12.374707,14.417770,17.414535,20.159998,21.160217,42.454707

```