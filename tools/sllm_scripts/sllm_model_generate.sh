#!/bin/bash

set -e  # 任何命令失败则退出

# 参数校验
if [ $# -eq 0 ]; then
    echo -e "\033[31m错误：必须指定模型名称参数\033[0m"
    echo "用法: $0 <model_name>"
    echo "示例: $0 facebook/opt-2.7b"
    exit 1
fi

MODEL_NAME=$1

# 创建输入文件
cat <<EOF > input.json
{
  "model": "$MODEL_NAME",
  "messages": [
    {
      "role": "user",
      "content": "Please introduce yourself."
    }
  ],
  "temperature": 0.7,
  "max_tokens": 50
}
EOF

# 检查命令是否存在
if ! command -v sllm-cli &> /dev/null; then
    echo -e "\033[31m错误: sllm-cli 未安装或不在 PATH 中\033[0m"
    exit 1
fi

# 执行生成命令
echo -e "\033[32m正在生成 $MODEL_NAME 的响应...\033[0m"
sllm-cli generate input.json