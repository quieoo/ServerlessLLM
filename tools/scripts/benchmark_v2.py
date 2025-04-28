import argparse
import json
import os
import subprocess
import time
import requests
import aiohttp
import asyncio


def process_jsonl(file_path):
    values = []
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

async def Ask_http_async(model, prompt, max_tokens):
    start_time=time.time()
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

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                end_time=time.time()
                print(f"Response time: {end_time-start_time:.2f} seconds")
                # print(f"Response: {json.dumps(result, indent=2)}")
            else:
                print(f"Error: {response.status}, {await response.text()}")

async def main():
    parser = argparse.ArgumentParser(description="Process a JSONL file ")
    parser.add_argument('file_path', type=str, help="Path to the JSONL file")
    parser.add_argument('model', type=str, help="Model name")
    parser.add_argument('max_tokens', type=int, help="Max tokens")
    parser.add_argument('qps', type=float, help="Queries sent per second")
    
    args = parser.parse_args()

    values = process_jsonl(args.file_path)
    print(f"Get {len(values)} prompts")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    interval = 1 / args.qps
    id=0
    while True:
        # Ask(args.model, values[id], args.max_tokens, output_dir)
        # Ask_http(args.model, values[id], args.max_tokens)
        asyncio.create_task(Ask_http_async(args.model, values[id], args.max_tokens))
        id+=1
        if id >= len(values):
            break
        await asyncio.sleep(interval)

if __name__ == "__main__":
    asyncio.run(main())


# python benchmark_v2.py /mnt/n0/datasets/sharegpt_V3_format.jsonl opt6.7b_tmp 100 0.2
