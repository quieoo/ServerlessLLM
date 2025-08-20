#!/bin/bash

# 敏感度分析脚本 - 批量大小测试
# 用法: ./run_batch_sensitivity.sh <cv_value>
# 例如: ./run_batch_sensitivity.sh 2.0

# 检查参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供CV值作为参数"
    echo "用法: $0 <cv_value>"
    echo "例如: $0 2.0"
    exit 1
fi

CV_VALUE=$1

# 验证CV值格式
if ! [[ "$CV_VALUE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "错误: CV值必须是数字"
    exit 1
fi

echo "开始运行CV值为 $CV_VALUE 的批量大小敏感度分析..."

# 基础路径
BASE_PATH="/mnt/n0/sslm/ServerlessLLM"
REQ_FILE_PATH="$BASE_PATH/tools/trace/outputs/l40_cv${CV_VALUE}.txt"
CONFIG_PATH="$BASE_PATH/tools/mock_allocation/configs/L40-large.json"
KV_BLOCK_PATH="$BASE_PATH/serverless_scripts/datasets_token_length/sharegpt_tokens.txt"

# 检查必要文件是否存在
if [ ! -f "$REQ_FILE_PATH" ]; then
    echo "错误: 请求文件不存在: $REQ_FILE_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$KV_BLOCK_PATH" ]; then
    echo "错误: KV块文件不存在: $KV_BLOCK_PATH"
    exit 1
fi

# 批量大小数组
BATCH_SIZES=(1 2 4 8 16 32 64)

# 运行所有批量大小的测试
for batch_size in "${BATCH_SIZES[@]}"; do
    echo "启动批量大小为 $batch_size 的测试..."
    
    ./build/Allocateion \
        -g 43 \
        -m 200 \
        -p 4 \
        --gpu 0 \
        --req_file_path "$REQ_FILE_PATH" \
        --config "$CONFIG_PATH" \
        --kv_block_file_path "$KV_BLOCK_PATH" \
        --kv_batch_size "$batch_size" \
    
    echo "批量大小 $batch_size 的测试已完成"
    
done

echo "所有批量大小测试已启动完成！"