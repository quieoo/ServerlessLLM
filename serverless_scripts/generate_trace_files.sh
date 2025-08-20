#!/bin/bash

# 生成多个trace文件的脚本
# 用法: ./generate_trace_files.sh <cv_value>
# 例如: ./generate_trace_files.sh 1.0

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

echo "开始生成CV值为 $CV_VALUE 的trace文件..."

# 基础路径
BASE_PATH="/mnt/n0/sslm/ServerlessLLM"
CONFIG_DIR="$BASE_PATH/tools/mock_allocation/configs"
TRACE_DIR="$BASE_PATH/tools/trace"
OUTPUT_DIR="$TRACE_DIR/outputs"


# 检查必要目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "错误: 配置目录不存在: $CONFIG_DIR"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "错误: 输出目录不存在: $OUTPUT_DIR"
    exit 1
fi

# 检查benchmark_trace.py是否存在
if [ ! -f "$TRACE_DIR/benchmark_trace.py" ]; then
    echo "错误: benchmark_trace.py 文件不存在"
    exit 1
fi

# 模型大小配置
MODEL_SIZES=("45g" "77g" "100g" "153g" "206g")

# 生成所有模型大小的trace文件
for model_size in "${MODEL_SIZES[@]}"; do
    config_file="$CONFIG_DIR/L40-large-${model_size}.json"
    output_file="$OUTPUT_DIR/l40_cv${CV_VALUE}_large_${model_size}.txt"
    
    echo "正在生成 ${model_size} 模型的trace文件..."
    echo "  配置文件: $config_file"
    echo "  输出文件: $output_file"
    
    # 检查配置文件是否存在
    if [ ! -f "$config_file" ]; then
        echo "  警告: 配置文件不存在，跳过: $config_file"
        continue
    fi
    
    # 运行benchmark_trace.py
    python $TRACE_DIR/benchmark_trace.py \
        --target_cv "$CV_VALUE" \
        --sllm_model_config_file_path "$config_file" \
        --target_req_file_path "$output_file"
    
    # 检查命令是否成功
    if [ $? -eq 0 ]; then
        echo "  ✓ 成功生成: $output_file"
        
        # 显示生成文件的基本信息
        if [ -f "$output_file" ]; then
            line_count=$(wc -l < "$output_file")
            echo "    文件包含 $line_count 行数据"
        fi
    else
        echo "  ✗ 生成失败: $output_file"
    fi
    
    echo ""
done

echo "所有trace文件生成完成！"
echo "生成的文件列表:"
for model_size in "${MODEL_SIZES[@]}"; do
    output_file="$OUTPUT_DIR/l40_cv${CV_VALUE}_large_${model_size}.txt"
    if [ -f "$output_file" ]; then
        line_count=$(wc -l < "$output_file")
        echo "  - $output_file ($line_count 行)"
    else
        echo "  - $output_file (未生成)"
    fi
done 