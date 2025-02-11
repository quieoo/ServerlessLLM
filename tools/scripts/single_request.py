import json
import os
import subprocess

# 设置模型参数
model = "facebook/opt-6.7b"
# model = "facebook/opt-1.3b"
# model = "llama3-8b-chinese"

temperature = 0.7
max_tokens = 100  # 控制生成的最大tokens数

# 输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

short_prompts = [
    {"role":"system",   "content": "You are a helpful assistant."},
    {"role":"user",     "content": "What is the weather like today?"}
]

long_prompts = [
    {"role":"system",   "content": "You are a helpful assistant."},
    {"role":"user",     "content": "近年来,随着人工智能和自然语言处理技术的迅速发展,各种大语言模型在多个领域中展现出了令人瞩目的表现。从自动文本生成、机器翻译到对话系统,这些模型在理解和生成自然语言方面的能力不断突破极限。研究人员不断探索新的算法和网络结构,例如 Transformer 架构、注意力机制以及自监督学习等,使得模型在大规模数据上进行训练后能捕捉到深层次的语义信息和上下文关系。在实际应用中,大语言模型往往需要处理各种长度不一的输入文本。对于较长的输入提示,模型需要一次性生成完整的上下文表示,这一过程被称为 prefill 阶段；而在生成过程中,模型又需要根据之前的上下文和内部缓存逐步输出新的文本,这就是 decode 阶段。为了确保推理效率和生成质量,研究者们通常会针对这两个阶段设计不同的优化策略,比如利用 CUDA 图加速解码过程、设计高效的 KV 缓存机制以及采用多模态信息融合等技。例如,在新闻摘要、技术文档生成以及智能客服等场景中,用户输入的文本长度可能从几百个字符到上千个字符不等。对于较长的文本,模型需要在 prefill 阶段充分理解输入的全部内容,构建一个完整的语义表示,然后在 decode 阶段逐步生成响应。这样的流程不仅要求模型在理解复杂文本时具有较高的精度,同时也要求生成过程足够高效,以满足实时交互的需求。当前,不少企业和研究机构正在探索通过量化、剪枝以及混合精度训练等方法,进一步提升模型的推理速度和资源利用率。此外,随着大数据时代的到来,海量文本信息的不断涌现也促使大语言模型在处理数据时不断优化性能。未来,如何在保证生成质量的同时降低计算资源的消耗,将成为人工智能研究的重要方向之一。各大科研机构和企业纷纷投入大量资源进行相关实验和应用开发,以期在各个领域实现智能化服务和自动化决策的广泛应用。通过不断的技术迭代和模型优化,大语言模型有望在医疗、金融、教育等多个领域发挥更大作用,推动社会各方面的数字化转型。总的来说,随着技术的不断进步和应用场景的不断扩展,大语言模型在自然语言理解与生成领域中的表现将愈发出色,为未来的智能社会提供更强大的技术支撑。"}
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
    result = subprocess.run(["sllm-cli", "generate", input_file], capture_output=True, text=True)

    # 输出结果
    print("Generated output:", result.stdout)


chat(short_prompts)
# chat(long_prompts)