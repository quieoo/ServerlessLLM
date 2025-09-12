import argparse
import json
import os
import subprocess
import time
import requests
import aiohttp
import asyncio
import pandas as pd
import numpy as np

def process_jsonl(file_path, type="sharegpt"):
    values = []
    if type=="sharegpt":
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        conv=data.get("conversations")
                        for c in conv:
                            if c.get("from")=="human":
                                values.append(c.get("value"))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line: {line}")
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
    elif type == "gsm8k":
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        q=data.get("question")
                        # 过滤掉一些无意义的输入(长度小于10)
                        if len(q) > 10:                            
                            values.append(q)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line: {line}")
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
    elif type == "alpaca":
        df=pd.read_parquet(file_path)
        values=df["instruction"].tolist()
        print(values[:10])
    elif type== "humaneval":
        df=pd.read_parquet(file_path)
        values=df["prompt"].tolist()
        print(values[:10])
    return values

def Ask(model, prompt, max_tokens, output_dir):
    input_data={
        "model": model,
        "message":prompt,
        "max_tokens":max_tokens
    }
    input_file = os.path.join(output_dir, "input.json")
    with open(input_file, "w") as f:
        json.dump(input_data, f, indent=2)
    result = subprocess.run(["sllm-cli", "generate", input_file], capture_output=True, text=True)

def Ask_http(model, prompt, max_tokens):
    url = "http://127.0.0.1:8343/v1/chat/completions"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        # print("Response:", json.dumps(result, indent=2))
        return result
    else:
        print(f"Error: {response.status_code}")
        print("Response:", response.text)



async def Ask_http_async(model, prompts, max_tokens):
    start_time=time.time()

    url = "http://127.0.0.1:8343/v1/chat/completions"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
        ],
        "max_tokens": max_tokens
    }
    for prompt in prompts:
        data["messages"].append({"role": "user", "content": prompt})

    headers = {
        "Content-Type": "application/json"
    }
    ttft=[]
    async with aiohttp.ClientSession() as session:
        # print(f"Request time: {time.time()}")
        async with session.post(url, headers=headers, json=data) as response:
            
            if response.status == 200:
                result = await response.json()
                log(f"response: {result}", LOG_LEVEL_INFO)
                for d in result.get('data', []):
                    print(f"{model} {float(d['metrics']['first_token_time'])-start_time:.2f}")
                    ttft.append(float(d['metrics']['first_token_time'])-start_time)
            else:
                print(f"Error: {response.status}, {await response.text()}")
    return ttft


def sync_batched_request(model, prompts, max_tokens):
    url = "http://127.0.0.1:8343/v1/chat/completions"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
        ],
        "max_tokens": max_tokens
    }
    for prompt in prompts:
        data["messages"].append({"role": "user", "content": prompt})

    headers = {
        "Content-Type": "application/json"
    }
    request_time=time.time()
    print(f"request time: {request_time}")  
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_time=time.time()
        print(f"response time: {response_time}")
        first_token_times=[]
        last_token_times=[]
        result = response.json()
        # print("prompt tokens | completion_tokens | metrics")
        for d in result.get('data', []):
            print(f"{d['usage']['prompt_tokens']}  {d['usage']['completion_tokens']} {d['metrics']}")
            # ftt=d['metrics']['first_token_time']
            # ltt=d['metrics']['last_token_time']
            # if ltt-ftt > 1.0:
            #     first_token_times.append(ftt)
            #     last_token_times.append(ltt)
        
        # 统计平均每token时延以及总token吞吐
        decode_times=[]
        for d in result.get('data', []):
            d_time=d['metrics']['last_token_time']-d['metrics']['first_token_time']
            if d_time > 0.5:
                decode_times.append(d_time)
            else:
                # print(f"decode time is 0, which seems impossible")
                pass
        avg_decode_time=sum(decode_times)/len(decode_times)
        throughput=max_tokens*len(decode_times)/avg_decode_time
        print(f"throughput: {throughput:.2f} tokens/second (seems reported last_token_time is wrong, decode time is 0, which seems impossible)")

    else:
        print(f"Error: {response.status_code}, {response.text}")

def single_prompt_request(model, prompt, max_tokens):
    # print(f"sending request: {prompt}")
    url = "http://127.0.0.1:8343/v1/chat/completions"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
        ],
        "max_tokens": max_tokens
    }
    data["messages"].append({"role": "user", "content": prompt})

    headers = {
        "Content-Type": "application/json"
    }
    print(f"request time: {time.time()}")
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        print("prompt tokens | completion_tokens | metrics")
        for d in result.get('data', []):
            print(f"{d['usage']['prompt_tokens']}  {d['usage']['completion_tokens']} {d['metrics']}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

def get_request_length_prompt(prompts, request_length):
    # 拼接prompts成一条，直到满足request_length要求
    combined_prompts = ""
    for prompt in prompts:
        combined_prompts += prompt + "\n"
        if len(combined_prompts) >= request_length:
            break
    return combined_prompts


def local_inference(prompts, type, model):
    from vllm import LLM, SamplingParams, RequestOutput
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_id = tokenizer.eos_token_id

    vllm_engine=LLM(model=model, enforce_eager=True)
    params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,          # 兜底
        stop_token_ids=[eos_id],  # 见下方解释
        skip_special_tokens=True  # 把 </s> 从输出里自动剔除
    )
    print("prompt_id: Prefill Time | Total Tokens")
    for prompt in prompts:
        req_time=time.time()
        request_output_list=vllm_engine.generate(prompt, params)
        for output in request_output_list:
            first_token_time=output.metrics.first_token_time
            prefill_time=first_token_time-req_time
            tokens=len(output.prompt_token_ids)
            for coutput in output.outputs:
                tokens += len(coutput.token_ids)
            print(f"{prefill_time:.4f} | {tokens}")
            # 将tokens写入文件保存
            with open(f"{type}_tokens.txt", "a") as f:
                f.write(f"{tokens}\n")

def batched_local_inference(prompts, batch_size, model, max_tokens=2048, test_round=10):
    from vllm import LLM, SamplingParams, RequestOutput
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_id = tokenizer.eos_token_id

    # 如果模型名称中包含tmp，则使用load_format为serverless_llm
    if "tmp" in model:
        vllm_engine = LLM(model=model, enforce_eager=True, load_format="serverless_llm")
    else:
        vllm_engine = LLM(model=model, enforce_eager=True)
    params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_tokens,
        stop_token_ids=[eos_id],
        skip_special_tokens=True
    )
    
    print(f"Batch inference with batch_size={batch_size}, model={model}")
    print("Batch_id | Prefill Time | Total Tokens | Prompt Tokens | Completion Tokens | Decode Time | Decode Throughput")
    
    # 按批次处理prompts
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    total_batches=min(total_batches, test_round)
    
    # 统计所有批次的总体数据
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_decode_time = 0
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        
        req_time = time.time()
        request_output_list = vllm_engine.generate(batch_prompts, params)
        
        batch_prompt_tokens = 0
        batch_completion_tokens = 0
        batch_prefill_time = 0
        batch_decode_time = 0
        
        for i, output in enumerate(request_output_list):
            first_token_time = output.metrics.first_token_time
            last_token_time = output.metrics.finished_time
            prefill_time = first_token_time - req_time
            decode_time = last_token_time - first_token_time
            
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = 0
            for coutput in output.outputs:
                completion_tokens += len(coutput.token_ids)
            
            batch_prompt_tokens += prompt_tokens
            batch_completion_tokens += completion_tokens
            batch_prefill_time = max(batch_prefill_time, prefill_time)
            batch_decode_time = max(batch_decode_time, decode_time)
        
        # 计算当前批次的decode吞吐量
        if batch_decode_time > 0:
            decode_throughput = batch_completion_tokens / batch_decode_time
        else:
            decode_throughput = 0
            
        print(f"{batch_idx:8d} | {batch_prefill_time:11.4f} | {batch_prompt_tokens + batch_completion_tokens:12d} | {batch_prompt_tokens:12d} | {batch_completion_tokens:16d} | {batch_decode_time:10.4f} | {decode_throughput:16.2f}")
        
        # 累计总体统计
        total_prompt_tokens += batch_prompt_tokens
        total_completion_tokens += batch_completion_tokens
        total_decode_time += batch_decode_time
        
    
    # 打印总体统计
    print("\n" + "="*80)
    print("OVERALL STATISTICS:")
    print(f"Total batches processed: {total_batches}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Total decode time: {total_decode_time:.4f}s")
    if total_decode_time > 0:
        overall_decode_throughput = total_completion_tokens / total_decode_time
        print(f"Overall decode throughput: {overall_decode_throughput:.2f} tokens/second")
    print("="*80)

models=[
    "opt1.3b_tmp",
    "opt2.7_tmp",
    "qwen2_3b_tmp",
    "llama2_3b_tmp",
    "llama3_chinese_tmp",
    "yi_9b_tmp",
    "opt_13b_tmp",
    "qwen2_14b_tmp"
]

LOG_LEVEL_DEBUG = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_ERROR = 3


g_log_level=0
def log(msg, level=0):
    if g_log_level>level:
        print(msg)


async def main():
    parser = argparse.ArgumentParser(description="Process a JSONL file ")
    parser.add_argument('--file_path', type=str, help="Path to the JSONL file")
    parser.add_argument('--type', type=str, help="Type of the JSONL file")
    parser.add_argument('--model', type=str, help="Model name")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch Size")
    parser.add_argument('--request_length', default=0, type=int, help="Request length")
    parser.add_argument('--max_tokens', type=int, default=20, help="Max tokens")
    parser.add_argument('--qps', type=float, default=1, help="Queries sent per second")
    parser.add_argument('--n', type=int, help="Number of prompts/batches to process")
    parser.add_argument('--trace_file_path', type=str, help="Path to the trace file")
    parser.add_argument('--local_inference', type=bool, help="Use local inference")
    parser.add_argument('--batched_inference', type=bool, help="Use batched inference")
    parser.add_argument('--batched_local_inference', type=bool, help="Use batched local inference")
    parser.add_argument('--logl', type=int, default=0, help="Log level")
    args = parser.parse_args()

    global g_log_level
    g_log_level=args.logl

    values = process_jsonl(args.file_path, args.type)
    print(f"Get {len(values)} prompts")

    if args.local_inference:
        local_inference(values, args.type, args.model)
        return
    
    if args.batched_inference:
        batched_local_inference(values, args.batch_size, args.model, args.max_tokens, args.n)
        return

    # 访问trace文件
    model_reqs=[]
    if args.trace_file_path is None:
        model_reqs=[args.model]
    else:
        with open(args.trace_file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            idx=int(line)
            if idx>=0 and idx<len(models):
                model_reqs.append(models[idx])

    print(f"model requests: {model_reqs}")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    round=len(values)//args.batch_size
    round=min(round, args.n)

    if args.qps == 0:
        if args.request_length > 0:
            prompt=get_request_length_prompt(values, args.request_length)
            single_prompt_request(args.model, prompt, args.max_tokens)
            return
        else:
            for i in range(round):
                sync_batched_request(args.model, values[i*args.batch_size:(i+1)*args.batch_size], args.max_tokens)
    else:
        # 使用异步方法
        interval = 1 / args.qps
        tasks = []  # 收集所有任务
        print("TTFT for each request: ")
        ttft_list=[]
        for i in range(round):
            # 创建任务并添加到列表
            task = asyncio.create_task(Ask_http_async(model_reqs[i%len(model_reqs)], values[i*args.batch_size:(i+1)*args.batch_size], args.max_tokens))
            tasks.append(task)
            if i<round-1:
                await asyncio.sleep(interval)
        # 等待所有任务完成后再退出
        await asyncio.gather(*tasks)
        for task in tasks:
            ttft_list.extend(task.result())
        print(f"TTFT mean: {np.mean(ttft_list):.4f}")
        print(f"TTFT p99: {np.percentile(ttft_list, 99):.4f}")
        print(f"TTFT p95: {np.percentile(ttft_list, 95):.4f}")
        print(f"TTFT p50: {np.percentile(ttft_list, 50):.4f}")

if __name__ == "__main__":
    asyncio.run(main())


