#!/bin/bash

# cd ../tools/trace/
# # python benchmark_trace.py --target_cv 0.25 --sllm_model_config_file_path ../mock_allocation/configs/L40-large.json --target_req_file_path ./outputs/L40-large-rr-0.25.txt --mapping round_robin
# # python benchmark_trace.py --target_cv 0.5 --sllm_model_config_file_path ../mock_allocation/configs/L40-large.json --target_req_file_path ./outputs/L40-large-rr-0.5.txt --mapping round_robin
# python benchmark_trace.py --target_cv 1 --sllm_model_config_file_path ../mock_allocation/configs/L40-large.json --target_req_file_path ./outputs/L40-large-rr-1.txt --mapping round_robin --days 2
# # python benchmark_trace.py --target_cv 2 --sllm_model_config_file_path ../mock_allocation/configs/L40-large.json --target_req_file_path ./outputs/L40-large-rr-2.txt --mapping round_robin
# cd ../../evaluation

BinaryPath="../tools/mock_allocation/build/Allocateion"
CONFIG_PATH="../tools/mock_allocation/configs/L40-large.json"
REQ_PATH="../tools/trace/outputs/L40-large-rr-"
KV_BLOCK_FILE_PATH="../serverless_scripts/datasets_token_length/sharegpt_tokens.txt"

GPU_NUM=2

CV_LIST=(1)    
REUSE_GRANULARITY=(1)
USABLE_MEMORY=(40)
BATCH_SIZE=(16)
SCHEDULE_POLICY=(1)

ALLOC_POLICY=(1 1 4)
DROP_POLICY=(0 1 1)


for ((i=0; i<${#CV_LIST[@]}; i++)); do
    echo "====== CV: ${CV_LIST[i]} ======"
    for ((j=0; j<${#REUSE_GRANULARITY[@]}; j++)); do
        echo "====== reuse_granularity: ${REUSE_GRANULARITY[j]} ======"
        echo "====== schedule_policy: ${SCHEDULE_POLICY[j]} ======"
        echo "====== usable_memory: ${USABLE_MEMORY[j]} ======"
        echo "====== batch_size: ${BATCH_SIZE[j]} ======"
        for ((k=0; k<${#ALLOC_POLICY[@]}; k++)); do
            echo "====== alloc_policy: ${ALLOC_POLICY[k]} ======"
            echo "====== drop_policy: ${DROP_POLICY[k]} ======"

            $BinaryPath -g ${USABLE_MEMORY[j]} -m 100 -r guas -s 40 -p ${ALLOC_POLICY[k]} -f ${DROP_POLICY[k]} \
                --gpu 0 \
                --req_file_path "${REQ_PATH}${CV_LIST[i]}.txt" \
                --config "$CONFIG_PATH" \
                --gpu_num "$GPU_NUM" \
                --kv_block_file_path "$KV_BLOCK_FILE_PATH" \
                --kv_batch_size "${BATCH_SIZE[j]}" \
                --reuse_granularity "${REUSE_GRANULARITY[j]}" \
                --schedule_policy "${SCHEDULE_POLICY[j]}"
        done
    done
done

echo "All done"