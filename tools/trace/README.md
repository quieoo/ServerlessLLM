# Benchmark Trace 工具使用说明

## 概述
`benchmark_trace.py` 是一个用于生成基准测试轨迹的工具，支持通过命令行参数自定义配置。

## 使用方法

### 基本用法
```bash
python benchmark_trace.py
```

### 自定义参数
```bash
python benchmark_trace.py \
    --target_cv 0.5 \
    --target_req_file_path /path/to/output/requests.txt \
    --sllm_model_config_file_path /path/to/model_config.json \
    --trace_name azure_v2 \
    --trace_dir /path/to/trace/data.txt
```

## 参数说明

- `--target_cv`: 目标变异系数 (默认: 0.25)
- `--target_req_file_path`: 生成的请求输出文件路径 (默认: `/mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_large.txt`)
- `--sllm_model_config_file_path`: SLLM模型配置文件路径 (默认: `/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-large.json`)
- `--trace_name`: 轨迹名称 (默认: `azure_v2`)
- `--trace_dir`: 轨迹数据目录路径 (默认: `/mnt/n0/datasets/azura_v2.txt`)

## 示例

### 使用不同的CV值
```bash
python benchmark_trace.py --target_cv 1.0
```

### 指定自定义输出路径
```bash
python benchmark_trace.py \
    --target_req_file_path ./my_requests.txt \
    --sllm_model_config_file_path ./my_model_config.json
```

### 使用不同的轨迹数据
```bash
python benchmark_trace.py \
    --trace_name azure_v1 \
    --trace_dir /path/to/azure_v1_data.txt
```

## 注意事项

1. 确保指定的模型配置文件存在
2. 输出目录会自动创建（如果不存在）
3. 脚本会自动验证文件路径的有效性