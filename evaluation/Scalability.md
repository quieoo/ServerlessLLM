
## quick start two models
````bash
sllm-cli deploy --config sllm_opt2.7tmp.json 
sllm-cli deploy --config sllm_opt6.7tmp.json
nohup sllm-cli generate opt2.7tmp_input.json >> 2.7.log 2>&1 &
nohup sllm-cli generate opt6.7tmp_input.json >> 2.7.log 2>&1 &

````


## start models with mock criu backend

changing "server.py" to switch reuse mode: 
````bash
load_strategy=0 (w/o reuse)
or
load_strategy=4 (Reuse)
````

start the controller node worker nodes and Reuse Store backend (hear we run four workers on a single machine, so distinguish them by different ports):
````bash
# controller
conda activate sllm-0.6
ray start --head --port=6379 --num-cpus=64 --num-gpus=0 --resources='{"control_node": 1}' --block

# worker-0~7
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=0
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1, "store_port":8073}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_1": 1, "store_port":8074}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=2
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_2": 1, "store_port":8075}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=3
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_3": 1, "store_port":8076}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=4
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_4": 1, "store_port":8077}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=5
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_5": 1, "store_port":8078}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=6
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_6": 1, "store_port":8079}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=7
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_7": 1, "store_port":8080}' --block


# store-0~7
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=0
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8073

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8074

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=2
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8075

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=3
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8076

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=4
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8077

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=5
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8078

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=6
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8079

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=7
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8080

# sllm 
conda activate sllm-0.6
sllm-serve start --enable_storage_aware

````


````bash
conda activate sllm-0.6
cd ServerlessLLM/serverless_scripts/

# batch_size=1, request_length=0, output_length=100, qps=0.4, n=30, req_path=4090_small
sllm-cli deploy --config mock_opt1.3.json
sllm-cli deploy --config mock_opt2.7.json
sllm-cli deploy --config mock_qwen3b.json
sllm-cli deploy --config mock_llama3b.json
sllm-cli deploy --config mock_llama8b.json
sllm-cli deploy --config mock_yi9b.json

sllm-cli deploy --config mock_opt13.json
sllm-cli deploy --config mock_qwen14.json

# simple request
curl http://127.0.0.1:8343/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "llama2_3b_tmp",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name and how are you ?"} 
        ]
    }'


# complex request
# fixed rps but varies gpu_num
python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --max_tokens=50 --batch_size=1 --n=100 --qps=0.4
python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --max_tokens=50 --batch_size=1 --n=200 --qps=1.6
python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --max_tokens=50 --batch_size=1 --n=300 --qps=3.2


python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090-cv1-large.txt --max_tokens=50 --batch_size=1 --n=200 --qps=0.8



python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090-cv1-large.txt --max_tokens=50 --batch_size=1 --n=400 --qps=2.4



# 不同workload的影响
python benchmark_v2.py --file_path=/home/zhchen/zwb/datasets/sharegpt_V3_format.jsonl --type=sharegpt --trace_file_path=/home/zhchen/zwb/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --max_tokens=50 --batch_size=1 --n=400 --qps=2.4


````




## Datasets
测试不同数据集的Prefill时间和decode length
使用Qwen2-3B作为基准模型
测试数据集：
    - shareGPT：a collection of user-shared conversations with ChatGPT
    - GSM8K: GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. 
    - Alpaca: an instruction dataset generated by GPT-3.5 with self-instruct
    - HumanEval: includes 164 programming problems with a function sig- nature, ocstring, body, and several unit tests. 

````bash
python benchmark_v2.py --local_inference=true --file_path=./datasets/haregpt_V3_format.jsonl --type=sharegpt --model ./models/qwen2_3/

python benchmark_v2.py --local_inference=true --file_path=./datasets/gsm8k_train.jsonl --type=gsm8k --model ./models/qwen2_3/

python benchmark_v2.py --local_inference=true --file_path=./datasets/alpaca.parquet --type=alpaca --model ./models/qwen2_3/

python benchmark_v2.py --local_inference=true --file_path=./datasets/humaneval.parquet --type=humaneval --model ./models/qwen2_3/

````