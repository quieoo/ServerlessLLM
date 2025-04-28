### 捕捉KV Cache用量
运行测试：

```
conda activate sllm-worker

python main.py --model_path /mnt/n0/models/llama3.18b_intruc_chinese/ --data_path /mnt/n0/datasets/sharegpt_V3_format.jsonl --total_samples 1000 --batch_size 16

python parse_log.py
```
