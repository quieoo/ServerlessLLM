# Performance Breakdown

- Original
- + Reuse
+ OD Cache



# Allocation Strategy
````bash
# one script runs all
nohup bash 3-allocation.sh > log/test-3.log 2>&1 &
````

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
cd ServerlessLLM/tools/mock_allocation
nohup ./run_l40_small.1.sh > load_strategy.l40.small.1.log 2>&1 &
nohup ./run_l40_large.1.sh > load_strategy.l40.large.1.log 2>&1 &

````


# On-demand KV Cache

## Per Model Average Load Latency with / without On-Demand KV Cache
````bash
# one script runs all
python benchmark_trace.py --target_cv 0.25 --sllm_model_config_file_path /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/bd-use-l40.json --target_req_file_path ./outputs/breakdown_l40_cv25_large.txt
nohup bash 2-breakdown.sh /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/breakdown_l40_cv25_large.txt > log/test-2.log 2>&1 &
````

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
## On-Demand KV allocate overhead per token

运行不同大小的推理请求，同时查看background_log中记录的申请block的总数以及总时延
将总时延/64, 获得per token allocation time
````bash
sllm-cli deploy --model llama3_chinese_tmp
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=64 --qps=0 --request_length=0 --batch_size=1 --model=llama3_chinese_tmp --n=1

python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=64 --qps=0 --request_length=0 --batch_size=2 --model=llama3_chinese_tmp --n=1

python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=64 --qps=0 --request_length=0 --batch_size=4 --model=llama3_chinese_tmp --n=1

python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=64 --qps=0 --request_length=0 --batch_size=8 --model=llama3_chinese_tmp --n=1

python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=64 --qps=0 --request_length=0 --batch_size=16 --model=llama3_chinese_tmp --n=1

python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=64 --qps=0 --request_length=0 --batch_size=32 --model=llama3_chinese_tmp --n=1

python benchmark_v2.py  --file_path=/mnt/n0/datasets/gsm8k_train.jsonl --type=gsm8k --max_tokens=64 --qps=0 --request_length=0 --batch_size=64 --model=llama3_chinese_tmp --n=1

````