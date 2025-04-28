from vllm import LLM, SamplingParams

def main():
    # 初始化模型（使用 2 个 GPU 张量并行）
    llm = LLM(
        model="/mnt/n0/models/opt13",  # 替换为你的模型路径
        tensor_parallel_size=2,        # 使用 2 个 GPU
        trust_remote_code=True,        # 如果模型需要自定义代码
    )

    # 推理
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
    outputs = llm.generate(["Explain AI in 100 words."], sampling_params)
    print(outputs)

if __name__ == "__main__":
    main()  # 确保主逻辑在 if __name__ == '__main__' 中