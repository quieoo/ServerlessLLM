import json
import os
import subprocess
import time

# 设置模型参数
model = "facebook/opt-6.7b"
# model = "facebook/opt-1.3b"
# model = "llama3-8b-chinese"

temperature = 0.7
max_tokens = 10  # 控制生成的最大tokens数

# 输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

short_prompts = [
    {"role":"system",   "content": "You are a helpful assistant."},
    {"role":"user",     "content": "What is the weather like today?"}
]

long_prompts = [
    {"role":"system",   "content": "You are a helpful assistant."},
    {"role":"user",     "content": "summarize within 10 word: In recent years, with the rapid development of artificial intelligence and natural language processing technology, various large language models have shown remarkable performance in many fields.From automatic text generation, machine translation to dialogue systems, these models continue to push their limits in understanding and generating natural language.Researchers are constantly exploring new algorithms and network structures, such as Transformer architecture, attention mechanism, and self-supervised learning, so that the model can capture deep semantic information and contextual relationships after training on large-scale data.In practical applications, large language models often need to deal with input texts of varying lengths.For longer input prompts, the model needs to generate a complete context representation at one time, which is called the prefill stage; and during the generation process, the model needs to gradually output new text based on the previous context and internal cache, which isdecode stage.To ensure inference efficiency and generation quality, researchers usually design different optimization strategies for these two stages, such as using CUDA graphs to accelerate the decoding process, designing efficient KV cache mechanisms, and using multimodal information fusion techniques.For example, in scenarios such as news digests, technical document generation, and intelligent customer service, the text input lengths of users may range from hundreds to thousands of characters.For longer text, the model needs to fully understand the entire content of the input in the prefill stage, build a complete semantic representation, and then gradually generate a response in the decode stage.Such a process not only requires the model to have high accuracy when understanding complex text, but also requires the generation process to be efficient enough to meet the needs of real-time interaction.At present, many enterprises and research institutions are exploring ways to further improve the inference speed and resource utilization of models through quantization, pruning and mixed precision training.In addition, with the advent of the big data era, the continuous emergence of massive text information has also prompted large language models to continuously optimize their performance when processing data.In the future, how to reduce the consumption of computing resources while ensuring the quality of generation will become one of the important directions in artificial intelligence research.Major scientific research institutions and enterprises have invested a lot of resources in related experiments and application development, in order to realize the wide application of intelligent services and automated decision-making in various fields.Through continuous technological iteration and model optimization, large language models are expected to play a greater role in many fields such as medical care, finance, and education, and promote digital transformation in all aspects of society.In general, with the continuous progress of technology and the continuous expansion of application scenarios, the performance of large language models in the field of natural language understanding and generation will become more outstanding, providing stronger technical support for the future intelligent society."}
]

def chat(prompt):
    input_data = {
        "model": model,
        "messages": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_seq_len":max_tokens
    }
    print(input_data)
    input_file = os.path.join(output_dir, "input_all.json")

    # 将数据写入文件
    with open(input_file, "w") as f:
        json.dump(input_data, f, indent=2)

    # 使用sllm-cli进行推理
    print(f"Running sllm-cli on {input_file}...")
    start_time = time.time()
    result = subprocess.run(["sllm-cli", "generate", input_file], capture_output=True, text=True)

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")

    # 输出结果
    print("Generated output:", result.stdout)


chat(short_prompts)
# chat(long_prompts)