import argparse
import json
import os
import subprocess
import time
import requests
import aiohttp
import asyncio
import pandas as pd

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

    async with aiohttp.ClientSession() as session:
        # print(f"Request time: {time.time()}")
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                for d in result.get('data', []):
                    print(f"{model} {d['metrics']['first_token_time']-start_time:.2f}")
            else:
                print(f"Error: {response.status}, {await response.text()}")

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
    print(f"request time: {time.time()}")
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        first_token_times=[]
        last_token_times=[]
        result = response.json()
        print("prompt tokens | completion_tokens | metrics")
        for d in result.get('data', []):
            print(f"{d['usage']['prompt_tokens']}  {d['usage']['completion_tokens']} {d['metrics']}")
            first_token_times.append(d['metrics']['first_token_time'])
            last_token_times.append(d['metrics']['last_token_time'])
        
        # 统计平均每token时延以及总token吞吐
        decode_times=[]
        for d in result.get('data', []):
            decode_times.append(d['metrics']['last_token_time']-d['metrics']['first_token_time'])
        avg_decode_time=sum(decode_times)/len(decode_times)
        throughput=max_tokens*len(prompts)/avg_decode_time
        print(f"throughput: {throughput:.2f} tokens/second")

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
    

models=[
    "opt1.3b_tmp",
    "opt2.7_tmp",
    "qwen2_3b_tmp",
    "llama2_3b_tmp",
    "llama3_chinese_tmp",
    "yi_9b_tmp"
]

async def main():
    parser = argparse.ArgumentParser(description="Process a JSONL file ")
    parser.add_argument('--file_path', type=str, help="Path to the JSONL file")
    parser.add_argument('--type', type=str, help="Type of the JSONL file")
    parser.add_argument('--model', type=str, help="Model name")
    parser.add_argument('--batch_size', type=int, help="Batch Size")
    parser.add_argument('--request_length', type=int, help="Request length")
    parser.add_argument('--max_tokens', type=int, help="Max tokens")
    parser.add_argument('--qps', type=float, help="Queries sent per second")
    parser.add_argument('--n', type=int, help="Number of prompts/batches to process")
    parser.add_argument('--trace_file_path', type=str, help="Path to the trace file")
    parser.add_argument('--local_inference', type=bool, help="Use local inference")
    args = parser.parse_args()

    values = process_jsonl(args.file_path, args.type)
    print(f"Get {len(values)} prompts")

    if args.local_inference:
        local_inference(values, args.type, args.model)
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
        for i in range(round):
            # 创建任务并添加到列表
            task = asyncio.create_task(Ask_http_async(model_reqs[i%len(model_reqs)], values[i*args.batch_size:(i+1)*args.batch_size], args.max_tokens))
            tasks.append(task)
            await asyncio.sleep(interval)
        # 等待所有任务完成后再退出
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())


# python benchmark_v2.py /mnt/n0/datasets/sharegpt_V3_format.jsonl opt6.7b_tmp 100 0.2
