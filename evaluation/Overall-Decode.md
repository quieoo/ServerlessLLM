
## SLLM / SLLM-C / SLLM-CM
1. Startup Clusters
````bash

sllm-store start  --mem-pool-size 48GB
sllm-serve start

sllm-cli deploy --model llama8b
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=50 --qps=0 --request_length=0 --batch_size=16 --model=llama8b --n=1

sllm-cli deploy --model yi9b
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=50 --qps=0 --request_length=0 --batch_size=16 --model=yi9b --n=1

sllm-cli deploy --model opt13b
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=50 --qps=0 --request_length=0 --batch_size=16 --model=opt13b --n=1


sllm-cli deploy --model gpt20b
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=10 --qps=0 --request_length=0 --batch_size=16 --model=gpt20b --n=1

````


## G-SLLM

````bash

# 1. Startup Clusters
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8073
sllm-serve start --enable_storage_aware





sllm-cli deploy --model gpt20b_tmp
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=50 --qps=0 --request_length=0 --batch_size=16 --model=gpt20b_tmp --n=1

sllm-cli deploy --model opt_13b_tmp
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=50 --qps=0 --request_length=0 --batch_size=16 --model=opt_13b_tmp --n=1

sllm-cli deploy --model yi_9b_tmp
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=50 --qps=0 --request_length=0 --batch_size=16 --model=yi_9b_tmp --n=1

sllm-cli deploy --model llama3_chinese_tmp
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=50 --qps=0 --request_length=0 --batch_size=16 --model=llama3_chinese_tmp --n=1


sllm-cli deploy --model qwen2_14b_tmp
python benchmark_v2.py  --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --max_tokens=50 --qps=0 --request_length=0 --batch_size=16 --model=qwen2_14b_tmp --n=1

````