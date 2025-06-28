
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


````bash
# batch_size=1, request_length=0, output_length=100, qps=0.4, n=30, req_path=4090_small
sllm-cli deploy --config mock_opt1.3.json
sllm-cli deploy --config mock_opt2.7.json
sllm-cli deploy --config mock_qwen3b.json
sllm-cli deploy --config mock_llama3b.json
sllm-cli deploy --config mock_llama8b.json
sllm-cli deploy --config mock_yi9b.json
python benchmark_v2.py /mnt/n0/datasets/sharegpt_V3_format.jsonl opt6.7b_tmp 1 0 100 3.2 100 /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/build/model_requests.seq

python benchmark_v2.py /mnt/n0/datasets/sharegpt_V3_format.jsonl opt6.7b_tmp 1 0 200 3.2 100 /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/build/model_requests.seq

python benchmark_v2.py /mnt/n0/datasets/sharegpt_V3_format.jsonl opt6.7b_tmp 1 0 200 1.6 100 /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/build/model_requests.seq

````