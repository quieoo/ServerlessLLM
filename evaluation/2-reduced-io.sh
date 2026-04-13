#!/bin/bash

GPU_NUM=2

cd ../tools/trace/

python benchmark_trace.py --target_cv 1 --sllm_model_config_file_path ../mock_allocation/configs/L40-large.json --target_req_file_path ./outputs/L40-large-rr-alive-1-worker${GPU_NUM}-cv1.txt --mapping round_robin --hour 24 --rate_scale_factor 0.4  --keep_alive 1 --worker_num $GPU_NUM

cd ../../evaluation

BinaryPath="../tools/mock_allocation/build/Allocateion"
CONFIG_PATH="../tools/mock_allocation/configs/L40-large.json"
REQ_PATH="../tools/trace/outputs/L40-large-rr-alive-1-worker${GPU_NUM}-cv"
KV_BLOCK_FILE_PATH="../serverless_scripts/datasets_token_length/sharegpt_tokens.txt"



# Strategy 1：model-level reuse
# Strategy 2：tensor-level reuse
# Strategy 3：+odkv
# Strategy 4：+affinity
# CV_LIST=(0.25 0.5 1 2)    
CV_LIST=(1)    

REUSE_GRANULARITY=(0 1 1 1)
# USABLE_MEMORY=(23 23 43 43)
USABLE_MEMORY=(20 20 40 40)
BATCH_SIZE=(0 0 16 16)
SCHEDULE_POLICY=(0 0 0 1)


for ((i=0; i<${#CV_LIST[@]}; i++)); do
    echo "====== CV: ${CV_LIST[i]} ======"
    for ((j=0; j<${#REUSE_GRANULARITY[@]}; j++)); do
        echo "====== reuse_granularity: ${REUSE_GRANULARITY[j]} ======"
        echo "====== schedule_policy: ${SCHEDULE_POLICY[j]} ======"
        echo "====== usable_memory: ${USABLE_MEMORY[j]} ======"
        echo "====== batch_size: ${BATCH_SIZE[j]} ======"

        $BinaryPath -g ${USABLE_MEMORY[j]} -m 100 -r guas -s 40 -p 4 \
            --gpu 0 \
            --req_file_path "${REQ_PATH}${CV_LIST[i]}.txt" \
            --config "$CONFIG_PATH" \
            --mock_copy \
            --gpu_num "$GPU_NUM" \
            --kv_block_file_path "$KV_BLOCK_FILE_PATH" \
            --kv_batch_size "${BATCH_SIZE[j]}" \
            --reuse_granularity "${REUSE_GRANULARITY[j]}" \
            --schedule_policy "${SCHEDULE_POLICY[j]}"
        echo "====== End ======"
    done
done

echo "All done"