# TTFT with Medusa

````bash
nohup ./build/Allocateion -g 43 -m 200 -p 0 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv1.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-large.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/gsm8k_tokens.txt --kv_batch_size 1 > l40_cv1.gsm8k.log  2>&1 &

nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv1.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-large.json --kv_block_file_path /mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/gsm8k_tokens.txt --kv_batch_size 1 > l40_cv1.gsm8k.log  2>&1 &
````

# Memory utilization
配置与上相同
开启“DetailedMetrics"的日志输出，打印过程中的实时显存利用率和数据传输量

````bash
nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > MemvsTran.R.l40_cv1.gsm8k.log 2>&1 &

nohup ./build/Allocateion -g 43 -m 200 -p 0 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv0.25_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json  > MemvsTran.WR.l40_cv1.gsm8k.log 2>&1 &

nohup ./build/Allocateion -g 43 -m 200 -p 0 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv1_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-uniform.json  > MemvsTran.WR.l40_cv1.gsm8k.log 2>&1 &

nohup ./build/Allocateion -g 43 -m 200 -p 4 --gpu 0 --req_file_path /mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv1_uniform.txt --config /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-uniform.json  > MemvsTran.R.l40_cv1.gsm8k.log 2>&1 &

````
