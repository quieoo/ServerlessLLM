#!/bin/bash
set -eo pipefail

# 颜色定义
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
RESET='\033[0m'

# 配置文件参数
CONFIG_FILE="sllm_model_config.json"

# 参数检查
if [ $# -eq 0 ]; then
    echo -e "${RED}错误：必须指定模型名称参数${RESET}"
    echo "用法: $0 <model_name>"
    echo "示例: $0 facebook/opt-2.7b"
    exit 1
fi

# 依赖检查
check_dependency() {
    if ! command -v $1 &>/dev/null; then
        echo -e "${RED}错误：需要安装 $1 工具${RESET}"
        echo -e "请使用: ${YELLOW}sudo apt-get install $1${RESET} 或类似命令安装"
        exit 1
    fi
}

check_dependency jq
check_dependency sllm-cli

# 配置文件验证
[ ! -f "$CONFIG_FILE" ] && echo -e "${RED}错误：配置文件 $CONFIG_FILE 未找到${RESET}" && exit 1

# 动态更新配置
echo -e "${GREEN}正在更新模型配置为: $1${RESET}"
jq --arg model "$1" \
    '.model = $model | 
    .backend_config.pretrained_model_name_or_path = $model' \
    "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && \
mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

# 执行部署命令
echo -e "\n${GREEN}正在启动模型部署...${RESET}"
sllm-cli deploy --config "$CONFIG_FILE"

echo -e "\n${GREEN}部署命令已执行，请检查服务状态${RESET}"