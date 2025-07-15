# 实验设置
* 单机环境：
    - 4090
    - L40
* 多机环境：
    - 4 * 4090
* 模型选择：
    - 4090：
        "opt1.3b", "opt2.7", "qwen2_3b", "llama2_3b", "llama3_chinese", "yi_9b"
    - L40:
        "opt1.3b", "opt2.7", "qwen2_3b", "llama2_3b", "llama3_chinese", "yi_9b", "opt13b", "Qwen14b"
* Workload
    Azura Serverless trace + Model request，将模型映射到serverless trace
    根据其他工作，在serverless LLM中小模型更多，因此映射过程中小模型被映射到更多的方法
    trace生成的分布选项：gamma分布，CV（变异系数），CV越大数据变异性越强
    CV设置：0.25, 0.5, 1, 2

* SOTA：SLLM-MEDUSA
    在原版的SLLM中实现了CRIU优化VLLM引擎启动时间，设置固定的KV Cache大小（不需要Profile_run）


# G-SLLm Performance
## End-to-End Performance(TTFT)
- Setttings
    - SLLM-MEDUSA
    - G-SLLM
- Models
    - "opt1.3b", "opt2.7", "qwen2_3b", "llama2_3b", "llama3_chinese", "yi_9b", "opt13b", "Qwen14b"
- Fixed Setting
    - CV=1
    - GPU=L40
    - Datasets=GSM8K

测量TTFT Breakdown
TTFT组成
 - Init: CRIU Time + Model Init Time
 - Load
 - KV Cache
 - Prefill

````bash
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv1.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-large.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/gsm8k_tokens.txt --kv_batch_size 1 > l40_cv1.gsm8k.log  2>&1 &
````

## Memory Utilization vs Data Transfer

配置与上相同
开启“DetailedMetrics"的日志输出，打印过程中的实时显存利用率和数据传输量

````bash
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > MemvsTran.R.l40_cv1.gsm8k.log 2>&1 &

nohup ./build/Allocateion -g 43 -m 200 -p 0 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > MemvsTran.WR.l40_cv1.gsm8k.log 2>&1 &

nohup ./build/Allocateion -g 43 -m 200 -p 0 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv1_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-uniform.json  > MemvsTran.WR.l40_cv1.gsm8k.log 2>&1 &

nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv1_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-uniform.json  > MemvsTran.R.l40_cv1.gsm8k.log 2>&1 &

````


# Sensitivity Analysis
## Workload Locality and Datasets

## GPU

````bash
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv1.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-large.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/gsm8k_tokens.txt --kv_batch_size 1 > gpu.l40_cv1.gsm8k.log  2>&1 &


nohup ./build/Allocateion -g 22 -m 200 -p 4 --gpu 1 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv1_large.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-large.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/gsm8k_tokens.txt --kv_batch_size 1 > gpu.4090_cv1.gsm8k.log  2>&1 &
````




# Performance Breakdown
## Loading Strategies

- Fixed Setting
    - CV=1
    - GPU=L40
    - Datasets=GSM8K
- Models
    - "opt1.3b", "opt2.7", "qwen2_3b", "llama2_3b", "llama3_chinese", "yi_9b", "opt13b", "Qwen14b"
- Setttings
    - w/o reuse
    - random + gm
    - cost + gm
    - cost + pbp
````bash
nohup ./run_4090_small.sh > load_strategy.4090.small.log 2>&1 &
nohup ./run_4090_large.sh > load_strategy.4090.large.log 2>&1 &

nohup ./run_l40_small.sh > load_strategy.l40.small.log 2>&1 &
nohup ./run_l40_large.sh > load_strategy.l40.large.log 2>&1 &



````



## On-Demand KV Cache
* 加载时延受KV Cache大小的影响
    - Available Tensor Pool Size 
实验组（w/o od-cache）：根据batch_size大小，调整显存池预留空间
````bash
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > kvsplit.small_pool.batch.na.log 2>&1 &

nohup ./build/Allocateion -g 41.75 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > kvsplit.small_pool.batch.1.log 2>&1 &
nohup ./build/Allocateion -g 40.5 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > kvsplit.small_pool.batch.2.log 2>&1 &
nohup ./build/Allocateion -g 38 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > kvsplit.small_pool.batch.4.log 2>&1 &
nohup ./build/Allocateion -g 33 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > kvsplit.small_pool.batch.8.log 2>&1 &
nohup ./build/Allocateion -g 23 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > kvsplit.small_pool.batch.16.log 2>&1 &
````

控制组(w/ od-cache)：测试在实际的batch下load latency
````bash
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/sharegpt_tokens.txt --kv_batch_size 1 > kvmerge.small_pool.batch.1.log 2>&1 &
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/sharegpt_tokens.txt --kv_batch_size 2 > kvmerge.small_pool.batch.2.log 2>&1 &
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/sharegpt_tokens.txt --kv_batch_size 4 > kvmerge.small_pool.batch.4.log 2>&1 &
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/sharegpt_tokens.txt --kv_batch_size 8 > kvmerge.small_pool.batch.8.log 2>&1 &
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/sharegpt_tokens.txt --kv_batch_size 16 > kvmerge.small_pool.batch.16.log 2>&1 &


````


## Decode Throughput
测量G-SLLM和SLLM-MEDUSA的解码吞吐量
选择某一个特定的模型，如opt6.7B
环境参数：
    - 数据集：ShareGPT/GSM8K
        ````TODO: 不同数据集的平均推理时间，inputs长度，decode长度应该不同. 现在的实现只能设置固定的decode长度，应该设置自动结束````
    - Batch Size: 1/2/4/8/16/32

    ````bash

    ````
## KV Block Size 影响
KV Cache的Block Size会影响Decode性能
参数：
    - Block Size: 8/16/32

# G-SLLM Performance Analysis

## Improved Memory Utilization and Reduced Data Transfer
在不同情况下的内存利用率与数据传输量
显存利用率 vs 数据传输速率
画在同一张图上，瞬时测得，左轴显存利用率（(模型参数+KV Cache大小)/显存总大小），右轴PCIe数据传输速率(仅在装载阶段出现，高度为PCIe带宽，宽度*高度代表数据量)
参数：

    - 时间
    1. 在每次装载模型参数和分配KV前后打印当前显存中的有效数据大小

    ````cpp
    LOG(DetailMetrics)<<"Memory Utilization: "<<pool->second->GetMemoryUtilization();
    ````

    2. 在每次数据传输时，打印传输量

    ````cpp
    LOG(DetailMetrics)<<"CopySize: "<<copy_size;
    ````

    3. 在申请KV Cache时加入时延，模拟decode时间
    ````cpp
    std::this_thread::sleep_for(std::chrono::milliseconds(16*26));
    ````

    4. 执行一个trace文件
    ````bash
    nohup ./build/Allocateion -g 40 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_example.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-uniform.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/gsm8k_tokens.txt --kv_batch_size 1 > mem_transfer.log 2>&1
    ````



## Loading Performance Analysis
分析G-SLLM的加载性能
* 测量模型的平均加载时延和尾时延
参数：
    - 模型
    - 管理策略：w/o reuse, random drop + global merge, cost drop + global merge, g-sllm(cost drop + cost merge)

执行trace文件，同时设置不同的装载策略
````bash
# without reuse
./build/Allocateion -g 20 -m 100 -r guas -s 40 -p 0 --affinity --gpu 2 --regenerate --config configs/4090-small.json
# random drop + global merge
./build/Allocateion -g 20 -m 100 -r guas -s 10 -p 1 -f 0 --affinity --gpu 2 --regenerate  --config configs/4090-small.json
# cost-aware drop + global merge
./build/Allocateion -g 20 -m 100 -r guas -s 10 -p 1 -f 1 --affinity --gpu 2 --config configs/4090-small.json
# cost-aware drop + cost-aware merge(partitioned bin packing)
./build/Allocateion -g 20 -m 100 -r guas -s 40 -p 4 -f 1 --affinity --gpu 2 --config configs/4090-small.json
````






# real-world workloads
测量P99 latency
参数：
    - RPS


