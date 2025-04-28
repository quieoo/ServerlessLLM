from datasets import load_dataset

import json
import os
import subprocess

n = 32

# 加载数据集（假设是一个简单的QA任务数据集）
# 加载GLUE中的SST-2任务（情感分析）
dataset = load_dataset("squad_v2", split=f"train[:{n}]")  # 只加载前n条数据进行测试

# 设置模型参数
model = "facebook/opt-6.7b"

temperature = 0.7
max_tokens = 100*n  # 控制生成的最大tokens数

# 输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 创建一个包含所有prompt的JSON结构
all_prompts = []
for i, item in enumerate(dataset):
    # 读取prompt内容，假设是"question"字段作为prompt
    prompt = item["question"]
    # print(f"Question: {prompt}")
    all_prompts.append({
        "role": "user",
        "content": prompt
    })

# 创建JSON结构
input_data = {
    "model": model,
    "messages": all_prompts,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "max_seq_len":max_tokens
}
print(input_data)

# 输出JSON文件路径
input_file = os.path.join(output_dir, "input_all.json")

# 将数据写入文件
with open(input_file, "w") as f:
    json.dump(input_data, f, indent=2)

# 使用sllm-cli进行推理
print(f"Running sllm-cli on {input_file}...")
result = subprocess.run(["sllm-cli", "generate", input_file], capture_output=True, text=True)

# 输出结果
print("Generated output:", result.stdout)
