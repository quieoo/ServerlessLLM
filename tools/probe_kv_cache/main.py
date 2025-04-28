import argparse
import json
from typing import List
from pathlib import Path
from vllm import LLM, SamplingParams
import time

def load_sharegpt_data(data_path: str, max_samples: int) -> List[str]:
    """加载并处理ShareGPT格式数据集"""
    prompts = []
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    conversation = "\n".join(
                        f"{turn['from']}: {turn['value']}"
                        for turn in data.get("conversations", [])
                    )
                    if conversation:  # 确保对话不为空
                        prompts.append(f"对话开始\n{conversation}\n助理:")
                    
                    # 如果达到样本数限制就提前退出
                    if len(prompts) >= max_samples:
                        break
                        
                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行: {line[:100]}...")  # 只打印前100个字符
                except KeyError as e:
                    print(f"警告: 数据格式不正确, 缺少键: {e}")
    except FileNotFoundError:
        print(f"错误: 未找到文件 {data_path}")
        return []
    
    return prompts

def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径 (本地或HF模型ID)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="ShareGPT数据集路径")
    parser.add_argument("--total_samples", type=int, default=100,
                        help="最大处理样本数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批量推理大小")
    parser.add_argument("--output_path", type=str, default="results.json",
                        help="输出文件路径")
    args = parser.parse_args()

    prompts = load_sharegpt_data(args.data_path, args.total_samples)
    print(f"已加载 {len(prompts)} 条有效样本")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        # max_tokens=512,
        stop=["\n用户:", "<|endoftext|>"]
    )

    start_time= time.time()
    llm = LLM(
        model=args.model_path,
        max_num_seqs=args.batch_size * 2,  # 留出缓冲空间
        # max_num_batched_tokens=4096,
        tensor_parallel_size=1  # 根据GPU数量调整
    )
    end_time= time.time()
    print(f"LLM Engine Init time: {end_time - start_time:.2f} seconds")

    results = []
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i+args.batch_size]
        
        # 执行推理
        outputs = llm.generate(batch_prompts, sampling_params)
    
        for j in range(len(outputs)):
            if outputs[j].metrics.first_token_time is not None and outputs[j].metrics.first_scheduled_time is not None:                
                print(f"prompt length: {len(outputs[j].prompt_token_ids)}, Prefill time: {outputs[j].metrics.first_token_time-outputs[j].metrics.first_scheduled_time}")

        # 收集结果
        for prompt, output in zip(batch_prompts, outputs):
            response = output.outputs[0].text.strip()
            results.append({
                "prompt": prompt,
                "response": response,
                "num_tokens": len(output.outputs[0].token_ids)
            })
        
        print(f"已处理 {min(i+args.batch_size, len(prompts))}/{len(prompts)} 条")

    # 5. 保存结果
    print("推理完成")

if __name__ == "__main__":
    main()