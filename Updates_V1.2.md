# ReuseStore V1.2
- mock_allocation: Tensor分配策略支持“GlobalMerge”，“Weighted Bipartite Matching + Greedy Merge”，“Partitioned Bin Packing”三种，并进行性能验证
- VLLM Backend：实现Segmented KV Cache更新与计算的核函数，在VLLM中集成AsyncBlockManager维护block_id和physical addr的映射，在初始化LLM Engine时通过访问Backend Store获得的gpu_mem_handle判断是否启用了ReuseStore并使用对应的推理策略，在使用ReuseStore的情况下KV_Cache_initialization被跳过。
- ReuseStore：目前暂时使用Best-Fit Allocation

## 测试
### SLLM Store
1. 启动ray节点
````bash
# 控制节点
conda activate sllm-0.6
export RAY_TMPDIR=/mnt/n0/ray_tmp
ray start --head --port=6379 --num-cpus=16 --num-gpus=0 --resources='{"control_node": 1}' --block

# 工作节点
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/mnt/n0/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
````

2. 启动Backend Store和sllm service
````bash
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
sllm-store start  --mem-pool-size 32GB

conda activate sllm-0.6
export RAY_TMPDIR=/mnt/n0/ray_tmp
sllm-serve start
````


3. 部署模型
````bash
conda activate sllm-0.6
sllm-cli deploy --model facebook/opt-6.7b
````

4. 访问模型
````bash
conda deactivate
python benchmark_v2.py /mnt/n0/datasets/sharegpt_V3_format.jsonl facebook/opt-6.7b 100 0.001
````

预期结果：
````
Get 200038 prompts
Response time: 19.91 seconds
Response: {
  "id": "chatcmpl-1280968c-dd15-4f84-a2bb-851cb193b145",
  "object": "chat.completion",
  "created": 1749003217.6785398,
  "model": "facebook/opt-6.7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "\n\nIt's aimed at people who know what they want to do with their lives, but need a bit more direction. It should also aid people who need to choose a degree in order to find out which vocation suits them best, because there aren't many universities that offer the...\n\naccountants services and business consultancy outsourcing & recruitment...multi-practice search to find the right jobs, funds and equipment providers and provide training and apprenticeships to support their future.\nJob No. J"
      },
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 46,
    "completion_tokens": 100,
    "total_tokens": 146
  }
}
````

### Reuse Store
1. 启动ray节点
````bash
# 控制节点
conda activate sllm-0.6
export RAY_TMPDIR=/mnt/n0/ray_tmp
ray start --head --port=6379 --num-cpus=16 --num-gpus=0 --resources='{"control_node": 1}' --block

# 工作节点
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/mnt/n0/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
````

2. 启动Backend Store和sllm service
````bash
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
sllm-store start  --mem-pool-size 32GB --chunk-size 0B

conda activate sllm-0.6
export RAY_TMPDIR=/mnt/n0/ray_tmp
sllm-serve start
````


3. 部署模型
````bash
conda activate sllm-0.6
sllm-cli deploy --model opt6.7b_tmp
````

4. 访问模型
````bash
conda deactivate
python benchmark_v2.py /mnt/n0/datasets/sharegpt_V3_format.jsonl opt6.7b_tmp 100 0.001
````

预期结果：
````
Get 200038 prompts
Response time: 14.08 seconds
Response: {
  "id": "chatcmpl-8fbf2a6e-ff59-4ca9-b0ac-2f15e5433136",
  "object": "chat.completion",
  "created": 1748950476.975977,
  "model": "opt6.7b_tmp",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "\n\nIt's aimed at people who know what they want to do with their lives, but need a bit more direction. It should also aid people who need to choose a degree in order to find out which vocation suits them best, because there aren't many universities that offer the...\n\naccountants services and business consultancy outsourcing & recruitment...multi-practice search to find the right jobs, funds and equipment providers and provide training and apprenticeships to support their future.\nJob No. J"
      },
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 46,
    "completion_tokens": 100,
    "total_tokens": 146
  }
}
````