import argparse
import json
import os
import subprocess
import time
import random
import string
import requests
import aiohttp
import asyncio
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class RequestEvent:
    model: str
    arrival_time: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def is_servegen_type(trace_type):
    return trace_type is not None and trace_type.lower() == "servegen"

def process_jsonl(file_path, type="sharegpt"):
    values = []
    if is_servegen_type(type):
        return values
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


def get_model_name_from_config_entry(entry):
    if "model" in entry:
        return entry["model"]
    if "name" in entry:
        return entry["name"]

    path = entry.get("path", "")
    if path:
        normalized_path = os.path.normpath(path)
        basename = os.path.basename(normalized_path)
        if basename.startswith("rank_"):
            return os.path.basename(os.path.dirname(normalized_path))
        return basename

    raise ValueError(f"Cannot infer model name from config entry: {entry}")


def load_model_names(model_config_path):
    with open(model_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    entries = config.get("model_lists", config if isinstance(config, list) else [])
    if not entries:
        raise ValueError(f"No model list found in {model_config_path}")

    entries = sorted(entries, key=lambda entry: entry.get("id", 0))
    return [get_model_name_from_config_entry(entry) for entry in entries]


def default_servegen_model_config_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(script_dir, "..", "configs", "servegen_8_models.json")
    )


def prompt_with_target_tokens(prompts, start_idx, target_tokens):
    if target_tokens is None or target_tokens <= 0:
        return prompts[start_idx % len(prompts)]

    pieces = []
    token_count = 0
    idx = start_idx
    while token_count < target_tokens:
        prompt = prompts[idx % len(prompts)]
        pieces.append(prompt)
        token_count += max(1, len(prompt.split()))
        idx += 1
        if idx - start_idx >= len(prompts) and token_count > 0:
            break
    return "\n".join(pieces)


def random_prompt_from_tokens(input_tokens, seed):
    char_count = max(1, input_tokens or 1)
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + " "
    return "".join(rng.choice(alphabet) for _ in range(char_count))


def build_prompts_for_event(prompts, event, batch_size, batch_idx, use_random_input):
    if use_random_input:
        return [
            random_prompt_from_tokens(
                event.input_tokens,
                seed=(batch_idx * batch_size + item_idx),
            )
            for item_idx in range(batch_size)
        ]

    if event.input_tokens is not None:
        return [
            prompt_with_target_tokens(
                prompts,
                batch_idx * batch_size + item_idx,
                event.input_tokens,
            )
            for item_idx in range(batch_size)
        ]

    return prompts[batch_idx * batch_size:(batch_idx + 1) * batch_size]


def parse_trace_events(trace_file_path, model_names, default_output_tokens):
    events = []
    trace_format = None

    with open(trace_file_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()
            try:
                if len(parts) == 1:
                    model_idx = int(parts[0])
                    event = RequestEvent(
                        model=model_names[model_idx],
                        output_tokens=default_output_tokens,
                    )
                    current_format = "legacy"
                elif len(parts) >= 4:
                    arrival_time = float(parts[0])
                    model_idx = int(parts[1])
                    input_tokens = int(float(parts[2]))
                    output_tokens = int(float(parts[3]))
                    event = RequestEvent(
                        model=model_names[model_idx],
                        arrival_time=arrival_time,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    current_format = "servegen"
                else:
                    raise ValueError("expected either 1 column or at least 4 columns")
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Invalid trace row at {trace_file_path}:{line_no}: {line}. {e}"
                ) from e

            if trace_format is None:
                trace_format = current_format
            elif trace_format != current_format:
                raise ValueError(
                    f"Mixed trace formats are not supported: first format is "
                    f"{trace_format}, but line {line_no} looks like {current_format}"
                )
            events.append(event)

    if not events:
        raise ValueError(f"No valid request events found in {trace_file_path}")

    return events, trace_format


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
    parser.add_argument('--model_config_path', type=str, help="Path to the ServeGen model config file")
    parser.add_argument('--ignore_trace_timestamps', action="store_true", help="Use --qps instead of timestamps in ServeGen trace")
    parser.add_argument('--trace_time_scale', type=float, default=1.0, help="Scale trace timestamps before replay")
    parser.add_argument('--local_inference', type=bool, help="Use local inference")
    parser.add_argument('--batched_inference', type=bool, help="Use batched inference")
    parser.add_argument('--batched_local_inference', type=bool, help="Use batched local inference")
    parser.add_argument('--logl', type=int, default=0, help="Log level")
    args = parser.parse_args()

    global g_log_level
    g_log_level=args.logl

    use_servegen_input = is_servegen_type(args.type)
    if use_servegen_input:
        if args.trace_file_path is None:
            raise ValueError("--trace_file_path is required when --type=ServeGen")
        values = []
        print("Use ServeGen trace input tokens; --file_path is not required")
    else:
        if args.file_path is None:
            raise ValueError("--file_path is required unless --type=ServeGen")
        values = process_jsonl(args.file_path, args.type)
        print(f"Get {len(values)} prompts")

    if args.local_inference:
        local_inference(values, args.type, args.model)
        return
    
    if args.batched_inference:
        batched_local_inference(values, args.batch_size, args.model, args.max_tokens, args.n)
        return

    model_names = models
    trace_format = None
    request_events = []
    if args.trace_file_path is None:
        if args.model is None:
            raise ValueError("--model is required when --trace_file_path is not provided")
        request_events = [
            RequestEvent(model=args.model, output_tokens=args.max_tokens)
        ]
    else:
        with open(args.trace_file_path, "r", encoding="utf-8") as f:
            first_trace_line = next(
                (line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")),
                "",
            )
        first_trace_columns = first_trace_line.replace(",", " ").split()
        if args.model_config_path:
            model_names = load_model_names(args.model_config_path)
        elif use_servegen_input or len(first_trace_columns) >= 4:
            servegen_config_path = default_servegen_model_config_path()
            if os.path.exists(servegen_config_path):
                model_names = load_model_names(servegen_config_path)
            else:
                print(
                    f"Warning: {servegen_config_path} not found; falling back to built-in model list"
                )

        request_events, trace_format = parse_trace_events(
            args.trace_file_path, model_names, args.max_tokens
        )
        if use_servegen_input and trace_format != "servegen":
            raise ValueError("--type=ServeGen requires a 4-column ServeGen trace")

    print(f"trace format: {trace_format or 'none'}")
    print(f"model names: {model_names}")
    print(f"request events: {len(request_events)}")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    if use_servegen_input or trace_format == "servegen":
        round = len(request_events)
    else:
        round=len(values)//args.batch_size
    if args.n is not None:
        round=min(round, args.n)
    if trace_format == "servegen":
        round=min(round, len(request_events))

    if args.qps == 0:
        if args.request_length > 0 and args.trace_file_path is None:
            prompt=get_request_length_prompt(values, args.request_length)
            single_prompt_request(args.model, prompt, args.max_tokens)
            return
        else:
            for i in range(round):
                event = request_events[i % len(request_events)]
                prompts = build_prompts_for_event(
                    values,
                    event,
                    args.batch_size,
                    i,
                    use_servegen_input,
                )
                sync_batched_request(event.model, prompts, event.output_tokens or args.max_tokens)
    else:
        # 使用异步方法
        tasks = []  # 收集所有任务
        print("TTFT for each request: ")
        ttft_list=[]
        trace_start_time = request_events[0].arrival_time
        replay_start_time = time.time()
        for i in range(round):
            event = request_events[i % len(request_events)]
            if (
                trace_format == "servegen"
                and not args.ignore_trace_timestamps
                and event.arrival_time is not None
                and trace_start_time is not None
            ):
                target_elapsed = (event.arrival_time - trace_start_time) * args.trace_time_scale
                sleep_time = replay_start_time + target_elapsed - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            prompts = build_prompts_for_event(
                values,
                event,
                args.batch_size,
                i,
                use_servegen_input,
            )

            max_tokens = event.output_tokens or args.max_tokens
            # 创建任务并添加到列表
            task = asyncio.create_task(Ask_http_async(event.model, prompts, max_tokens))
            tasks.append(task)
            if (
                i < round - 1
                and (trace_format != "servegen" or args.ignore_trace_timestamps)
            ):
                interval = 1 / args.qps
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
