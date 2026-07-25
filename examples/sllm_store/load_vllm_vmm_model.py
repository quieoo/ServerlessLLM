#!/usr/bin/env python3
"""Single-GPU smoke/latency test for vLLM 0.5 with in-process Tangram VMM."""

import argparse
import json
import statistics
import time

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/mnt/n0/models/vllm/opt1.3b_tmp"
    )
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=128)
    args = parser.parse_args()

    start = time.perf_counter()
    llm = LLM(
        model=args.model,
        load_format="serverless_llm",
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    load_ms = (time.perf_counter() - start) * 1000.0
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    for _ in range(args.warmup):
        llm.generate([args.prompt], sampling, use_tqdm=False)

    samples = []
    output = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        output = llm.generate([args.prompt], sampling, use_tqdm=False)
        samples.append((time.perf_counter() - start) * 1000.0)
    result = {
        "backend": "vllm_0.5_in_process_vmm",
        "model": args.model,
        "load_ms": load_ms,
        "latency_ms_mean": statistics.mean(samples),
        "latency_ms_median": statistics.median(samples),
        "latency_ms_samples": samples,
        "output": output[0].outputs[0].text,
    }
    print("TANGRAM_VMM_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
