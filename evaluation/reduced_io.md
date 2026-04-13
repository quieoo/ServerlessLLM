

````bash
export CUDA_VISIBLE_DEVICES=1
conda activate sllm-worker-0.6
cd sslm/ServerlessLLM/tools/mock_allocation/build

# 增加mock_copy参数，只访问元数据，不需要实际的数据移动
./Allocateion -g 43 -m 100 -r guas -s 40 -p 4 -f 1 --affinity --gpu 0 --req_file_path "/mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv0.5.small.txt" --config "/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-small.1.json" --mock_copy

# 多GPU，随机调度
nohup ./Allocateion -g 43 -m 100 -r guas -s 40 -p 4 -f 1 --affinity --req_file_path "/mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv0.5.small.txt" --config "/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-small.1.json" --mock_copy --gpu_num 2 --schedule_policy 0 > log.txt 2>&1 &

# 亲和性感知的调度
nohup ./Allocateion -g 43 -m 100 -r guas -s 40 -p 4 -f 1 --affinity --req_file_path "/mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv0.5.small.txt" --config "/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-small.1.json" --mock_copy --gpu_num 2 --schedule_policy 1 > log.txt 2>&1 &

# model级别的重用
nohup ./Allocateion -g 43 -m 100 -r guas -s 40 -p 4 -f 1 --affinity --req_file_path "/mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv0.5.small.txt" --config "/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-small.1.json" --mock_copy --gpu_num 2 --schedule_policy 0 --reuse_granularity 0 > log.txt 2>&1 &


# 针对GPU-NUM=2的情况，增加模型池大小
cd ../tools/trace/
conda activate sllm-trace
python benchmark_trace.py --target_cv 0.25 --sllm_model_config_file_path ../mock_allocation/configs/L40-large-206g-uniform.json --target_req_file_path ./outputs/L40-large-206g-uniform-0.25.txt
python benchmark_trace.py --target_cv 0.5 --sllm_model_config_file_path ../mock_allocation/configs/L40-large-206g-uniform.json --target_req_file_path ./outputs/L40-large-206g-uniform-0.5.txt
python benchmark_trace.py --target_cv 1 --sllm_model_config_file_path ../mock_allocation/configs/L40-large-206g-uniform.json --target_req_file_path ./outputs/L40-large-206g-uniform-1.txt
python benchmark_trace.py --target_cv 2 --sllm_model_config_file_path ../mock_allocation/configs/L40-large-206g-uniform.json --target_req_file_path ./outputs/L40-large-206g-uniform-2.txt

````