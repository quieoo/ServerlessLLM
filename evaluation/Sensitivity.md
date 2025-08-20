## Workload Locality and Datasets

````bash
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv1.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-large.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/gsm8k_tokens.txt --kv_batch_size 1 > gpu.l40_cv1.gsm8k.log  2>&1 &


nohup ./build/Allocateion -g 22 -m 200 -p 4 --gpu 1 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv1_large.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-large.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/gsm8k_tokens.txt --kv_batch_size 1 > gpu.4090_cv1.gsm8k.log  2>&1 &
````


## Model Size


````bash
cd serverless_scripts
./generate_trace_files.sh 2.0
./generate_trace_files.sh 1.0
./generate_trace_files.sh 0.5
./generate_trace_files.sh 0.25

````

run the micro benchnmark:
````bash
cd serverless_scripts
nohup ./run_model_size_tests.sh 0.25 > modelpool.l40large_cv0.25.batch.log 2>&1 &
nohup ./run_model_size_tests.sh 0.5 > modelpool.l40large_cv0.5.batch.log 2>&1 &
nohup ./run_model_size_tests.sh 1.0 > modelpool.l40large_cv1.batch.log 2>&1 &
nohup ./run_model_size_tests.sh 2.0 > modelpool.l40large_cv2.batch.log 2>&1 &

````

## Batch Size

````bash
# High CV Case (cv=2.0): 
nohup ./run_batch_sensitivity.sh 2.0 > batch.cv2.4090.large.log 2>&1 &
# Reuse Case (cv=1.0): 
nohup ./run_batch_sensitivity.sh 1.0 > batch.cv1.4090.large.log 2>&1 &

# Low Reuse Case (cv=0.5): 
nohup ./run_batch_sensitivity.sh 0.5 > batch.cv0.5.4090.large.log 2>&1 &

# Without Reuse (cv=0.25): 
nohup ./run_batch_sensitivity.sh 0.25 > batch.cv0.25.4090.large.log 2>&1 &
````


