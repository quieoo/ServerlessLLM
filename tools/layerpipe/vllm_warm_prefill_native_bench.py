#!/usr/bin/env python3
"""Small original-vLLM warm-Prefill baseline for the hook overhead study."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def bootstrap_device(argv):
    device = "0"
    for index, value in enumerate(argv):
        if value == "--device" and index + 1 < len(argv):
            device = argv[index + 1]
        elif value.startswith("--device="):
            device = value.split("=", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["USE_GPU"] = "0"


bootstrap_device(sys.argv[1:])
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from layerpipe_bench import (  # noqa: E402
    apply_request_input_limits,
    load_model_input_limits,
    parse_trace,
)
from vllm_odkv_trace_bench import (  # noqa: E402
    load_models,
    make_prompt_tokens,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-path", action="append", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"),
                        default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--truncate-input-to-model-limit",
                        action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requests = parse_trace(args.trace, args.max_requests)
    referenced = sorted({item.model_id for item in requests})
    if len(referenced) != 1:
        raise ValueError("Native warm-Prefill run requires one model")
    model_id = referenced[0]
    for request in requests:
        request.output_tokens = args.output_tokens
    limits = load_model_input_limits(args.config)
    apply_request_input_limits(
        requests, limits, 0, args.truncate_input_to_model_limit)
    configuration = json.loads(args.config.read_text())
    entries = {
        int(item["id"]): item for item in configuration["model_lists"]
    }
    model = entries[model_id]
    packed_metadata = load_models(args.config, args.model_path)[model_id]
    overrides = {
        int(value.split("=", 1)[0]): value.split("=", 1)[1]
        for value in args.model_path
    }
    model_path = overrides.get(
        model_id, model.get("hf_path", model["path"]))

    # Select the unmodified vLLM weight/KV paths.  This process deliberately
    # never constructs TangramVmmPool or TangramLayerWeaveController.
    os.environ["TANGRAM_VMM_IN_PROCESS"] = "0"
    os.environ.pop("TANGRAM_KV_BACKEND", None)
    os.environ.pop("TANGRAM_WEIGHT_LOAD_MODE", None)
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    from vllm import LLM, SamplingParams

    max_model_len = max(
        item.input_tokens + item.output_tokens for item in requests)
    started = time.perf_counter()
    llm = LLM(
        model=model_path,
        load_format="auto",
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=0,
        trust_remote_code=args.trust_remote_code,
        max_num_seqs=1,
        tokenizer_mode=model.get(
            "tokenizer_mode", packed_metadata["tokenizer_mode"]),
        max_model_len=max_model_len,
        **packed_metadata["vision_options"],
    )
    startup_ms = (time.perf_counter() - started) * 1000.0
    worker = llm.llm_engine.model_executor.driver_worker
    vocab_size = int(worker.model_config.get_vocab_size())
    batches = []
    for batch_id, request in enumerate(requests):
        prompt = make_prompt_tokens(request, vocab_size, args.seed)
        sampling = SamplingParams(
            temperature=0,
            max_tokens=request.output_tokens,
            ignore_eos=True,
            detokenize=False,
        )
        wall_started = time.perf_counter()
        output = llm.generate(
            prompt_token_ids=[prompt],
            sampling_params=[sampling],
            use_tqdm=False,
        )[0]
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        metrics = output.metrics
        scheduled = (
            metrics.first_scheduled_time
            if metrics.first_scheduled_time is not None
            else metrics.arrival_time
        )
        batches.append({
            "batch_id": batch_id,
            "request_id": request.request_id,
            "model_id": model_id,
            "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens,
            "prefill_ms": (
                metrics.first_token_time - scheduled) * 1000.0,
            "generate_wall_ms": wall_ms,
        })
    document = {
        "backend": "native_vllm",
        "execution_engine": "vllm",
        "kv_backend": "original_vllm",
        "model_id": model_id,
        "trace": str(args.trace.resolve()),
        "engine_startup_ms": startup_ms,
        "batch_metrics": batches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print("WARM_PREFILL_NATIVE=" + json.dumps({
        "model_id": model_id,
        "requests": len(batches),
        "startup_ms": startup_ms,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
