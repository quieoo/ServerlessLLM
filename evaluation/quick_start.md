Here is a quick start guide for running the Tangram demo.

## Prepare CRIU images

0. Setup
````bash
# start a terminal, run the Reuse Store
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=2
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --reuse-store 2 --port 8073

# start another teminal for CRIU service
cd sslm/ServerlessLLM/tools/CRIU/criu_rpc/
sudo su
nohup criu service --address /tmp/criu_service.socket >> criu_service.log 2>&1 &


````

1. Create Dump images
- Before creating images, ensure that VLLM and SLLM libraries are installed using "pip install ." method, not in "editable" mode.
- Modify the model path according to the model being run, and ensure that the "imgs" folder exists under the model path.
- Make sure to see the same GPU as the Store.
````bash
conda activate /mnt/n0/.conda/envs/sllm-worker-0.6
rm -rf /mnt/n0/models/vllm/opt6.7b_tmp/imgs/*
export CUDA_VISIBLE_DEVICES=2
setsid python server.py --socket_addr=/tmp/criu_service.socket --model_path=/mnt/n0/models/vllm/opt6.7b_tmp < /dev/null &> /dev/null

````

Check if the images are successfully dumped
````bash
ls /mnt/n0/models/vllm/opt6.7b_tmp/imgs/
cat /mnt/n0/models/vllm/opt6.7b_tmp/imgs/criu.log
````
If succsess, you will see the following message in the log:
````
Warn  (compel/arch/x86/src/lib/infect.c:418): Will restore 400738 with interrupted system call
````

If failed, the RPC server may be still running. Check and kill it.
````bash
sudo netstat -tulnp | grep ':50051'
sudo kill -9 <pid>
````


2. Test Restore
````bash
cd sslm/ServerlessLLM/tools/CRIU/criu_rpc/
python test_2.py --socket_addr=/tmp/criu_service.socket --model_path=/mnt/n0/models/vllm/opt6.7b_tmp/imgs
````

It should output like:
````
......
Init + Load Time: 0.92 s
Prefill Time: 0.55 s
````

## Get Started

0. Start a Ray cluster locally with 1 controler and 1 worker
````bash
# start two terminals
# run the following command in a directory with a "models" subdirectory which should contain the model parameters, like: "./models/vllm/opt6.7b_tmp/rank_0"

# controller
conda activate sllm-0.6
ray start --head --port=6379 --num-cpus=16 --num-gpus=0 --resources='{"control_node": 1}' --block

# start the worker on another terminal
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=2
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1, "store_port":8073}' --block

````

1. Start the Reuse Store and Scheduler
````bash
# start new terminals, run following commands, respectively

# the GPU ID and port should be consistent with the worker
# Unified Memory Pool Size = Total_Available_GPU_Memory - <memory-pool-size>
# set <chunk-size> to 0 to disable SLLM's default chunking, set <reuse-store> to 2 to enable Tangram
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=2
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --reuse-store 2 --port 8073

cd sslm/ServerlessLLM/tools/CRIU/criu_rpc/
sudo su
nohup criu service --address /tmp/criu_service.socket >> criu_service.log 2>&1 &

conda activate sllm-0.6
sllm-serve start --enable_storage_aware

````


2. Register the model
````bash

conda activate sllm-0.6
cd sslm/ServerlessLLM/serverless_scripts/
sllm-cli deploy --config criu_opt6.7tmp.json
````

Scheduler should output message like:
````
(SllmController pid=1149655, ip=172.17.0.7) 2025-08-25 20:36:04.888 - sllm.serve.controller - INFO - Registering model opt6.7b_tmp, backend criu, backend_config {'pretrained_model_name_or_path': 'opt6.7b_tmp', 'device_map': 'auto', 'torch_dtype': 'float16', 'hf_model_class': 'AutoModelForCausalLM'}, router_config {'node_info': {'0': {'ray_node_id': '44b3b9d679990befd7980839313d712d4bbd0244383addbe9be79a8c', 'address': '127.0.0.1', 'free_gpu': 1.0, 'total_gpu': 1.0, 'store_port': 8073.0}}}, auto_scaling_config {'metric': 'concurrency', 'target': 1, 'min_instances': 0, 'max_instances': 10, 'keep_alive': 0}
````

Reuse Store should output message like:
````
[INFO] Model registered: ./models/vllm/opt6.7b_tmp/rank_0, size: 13317210112
````

3. Request the model

````bash
curl http://127.0.0.1:8343/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "opt6.7b_tmp",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name and how are you ?"} 
        ]
    }'

python benchmark_v2.py --file_path=/mnt/n0/datasets/sharegpt_V3_format.jsonl --type=sharegpt --model=opt6.7b_tmp --n=1 
````

output should be like:
````
Get 200038 prompts
model requests: ['opt6.7b_tmp']
TTFT for each request: 
opt6.7b_tmp 1.33
TTFT mean: 1.3288
TTFT p99: 1.3288
TTFT p95: 1.3288
TTFT p50: 1.3288
````

Note that the first request may be slow because of the tensor miss in GPU Memory Pool, and the subsequent requests should be fast.