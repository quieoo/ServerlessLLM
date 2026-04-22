#!/bin/bash

# 检查参数是否提供
if [ $# -ne 1 ]; then
    echo "Usage: $0 <req_file_path>"
    exit 1
fi

REQ_FILE_PATH=$1
CONFIG_PATH="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/bd-use-l40.json"
KV_BLOCK_FILE_PATH="/mnt/n0/sslm/ServerlessLLM/serverless_scripts/datasets_token_length/sharegpt_tokens.txt"

BinaryPath="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/build/Allocateion"

# 确保目录存在
mkdir -p logs

# 实验组（w/o od-cache）
echo "Starting experimental group (w/o od-cache)..."
 $BinaryPath -g 41.75 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH
 $BinaryPath -g 40.5 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH
 $BinaryPath -g 38 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH
 $BinaryPath -g 33 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH
 $BinaryPath -g 23 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH
# 控制组(w/ od-cache)
echo "Starting control group (w/ od-cache)..."
 $BinaryPath -g 43 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH --kv_block_file_path $KV_BLOCK_FILE_PATH --kv_batch_size 1 
 $BinaryPath -g 43 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH --kv_block_file_path $KV_BLOCK_FILE_PATH --kv_batch_size 2 
 $BinaryPath -g 43 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH --kv_block_file_path $KV_BLOCK_FILE_PATH --kv_batch_size 4 
 $BinaryPath -g 43 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH --kv_block_file_path $KV_BLOCK_FILE_PATH --kv_batch_size 8 
 $BinaryPath -g 43 -m 200 -p 4 --gpu 0 --req_file_path $REQ_FILE_PATH --config $CONFIG_PATH --kv_block_file_path $KV_BLOCK_FILE_PATH --kv_batch_size 16
echo "All commands started. Logs are in the 'logs' directory."
