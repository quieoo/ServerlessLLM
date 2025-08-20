#!/bin/bash

# 模型大小测试脚本
# 用法: ./run_model_size_tests.sh <cv_value>
# 例如: ./run_model_size_tests.sh 1.0

# 检查参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供CV值作为参数"
    echo "用法: $0 <cv_value>"
    echo "例如: $0 1.0"
    exit 1
fi

CV_VALUE=$1

# 验证CV值格式
if ! [[ "$CV_VALUE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "错误: CV值必须是数字"
    exit 1
fi

echo "开始运行CV值为 $CV_VALUE 的模型大小测试..."

# 基础路径
BASE_PATH="/mnt/n0/sslm/ServerlessLLM"
MOCK_ALLOCATION_DIR="$BASE_PATH/tools/mock_allocation"
BUILD_DIR="$MOCK_ALLOCATION_DIR/build"
TRACE_DIR="$BASE_PATH/tools/trace/outputs"
CONFIG_DIR="$MOCK_ALLOCATION_DIR/configs"
KV_BLOCK_PATH="$BASE_PATH/serverless_scripts/datasets_token_length/sharegpt_tokens.txt"

# 检查必要文件和目录
if [ ! -f "$BUILD_DIR/Allocateion" ]; then
    echo "错误: Allocateion 可执行文件不存在: $BUILD_DIR/Allocateion"
    exit 1
fi

if [ ! -d "$TRACE_DIR" ]; then
    echo "错误: trace输出目录不存在: $TRACE_DIR"
    exit 1
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "错误: 配置目录不存在: $CONFIG_DIR"
    exit 1
fi

# 模型大小配置
MODEL_SIZES=("45g" "77g" "100g" "153g" "206g")

# 检查所有必要的文件是否存在
echo "检查必要文件..."
for model_size in "${MODEL_SIZES[@]}"; do
    req_file="$TRACE_DIR/l40_cv${CV_VALUE}_large_${model_size}.txt"
    config_file="$CONFIG_DIR/L40-large-${model_size}.json"
    
    if [ ! -f "$req_file" ]; then
        echo "错误: 请求文件不存在: $req_file"
        exit 1
    fi
    
    if [ ! -f "$config_file" ]; then
        echo "错误: 配置文件不存在: $config_file"
        exit 1
    fi
    
    echo "  ✓ ${model_size}: $req_file"
done

echo "所有文件检查完成，开始顺序执行测试..."

# 顺序执行所有模型大小的测试
for model_size in "${MODEL_SIZES[@]}"; do
    req_file="$TRACE_DIR/l40_cv${CV_VALUE}_large_${model_size}.txt"
    config_file="$CONFIG_DIR/L40-large-${model_size}.json"
    log_file="modelpool.l40large_cv${CV_VALUE}.${model_size}.log"
    
    echo ""
    echo "=== 开始测试 ${model_size} 模型 ==="
    echo "请求文件: $req_file"
    echo "配置文件: $config_file"
    echo "日志文件: $log_file"
    
    # 执行命令
    echo "启动 Allocateion 进程..."
    "$BUILD_DIR/Allocateion" \
        -g 43 \
        -m 200 \
        -p 4 \
        --gpu 0 \
        --req_file_path "$req_file" \
        --config "$config_file" \
        --kv_block_file_path "$KV_BLOCK_PATH" \
        --kv_batch_size 16
    
    echo "=== ${model_size} 模型测试结束 ==="
done

echo ""
echo "所有模型大小测试完成！"
echo "日志文件列表:"
for model_size in "${MODEL_SIZES[@]}"; do
    log_file="modelpool.l40large_cv${CV_VALUE}.${model_size}.log"
    if [ -f "$log_file" ]; then
        line_count=$(wc -l < "$log_file")
        echo "  - $log_file ($line_count 行)"
    else
        echo "  - $log_file (未生成)"
    fi
done 