
## Init Time
### Init with CRIU
- CRIU Restore
    1. CRIU Time: criu restore with socket-RPC. img files saved as files. the time is fixed: 310ms
    2. Model Time: do the rest initialization of model, different model has different time. 
        change the vllm code to report timestamp before and after model_initialize, for example: 
        ````python
        model_executer/models/qwen2.py
        class Qwen2Model(nn.Module):
            def __init__:
                ......
                import xformers
                import time
                print(f"Time begin to Init Qwen2 model layers: {time.time()}")
                self.layers = nn.ModuleList([
                    Qwen2DecoderLayer(config, cache_config, quant_config)
                    for _ in range(config.num_hidden_layers)
                ])        
        ......

        engine/llm_engine.py
        class DefaultModelLoader(BaseModelLoader):
            def load_model:
                with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config,
                                          cache_config)
                print(f"Time begin to load model: {time.time()}")
        ````
        run with, for example qwen14B:
        ````bash
        python server.py --model_path=/mnt/n0/models/qwen2-14/
        ````

- SLLM Init
    step-1: Start the SLLM cluster
    step-2: deploy and request the model

    ````python
    sllm/serve/app_lib.py
        async def inference_handler:
            print(f"Time request received: {time.time()}")
    ````
    ````bash
    sllm-cli deploy --model llama3_chinese_tmp
    curl http://127.0.0.1:8343/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "llama3_chinese_tmp",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name and how are you ?"} 
        ]
    }'

    ````

## Load Time
Measured with mock_allocation

## Profile Time

````python
class LLMEngine:
    def __init__:
        start_kv_init_at = time.time()
        if not self.model_config.embedding_mode:
            self._initialize_kv_caches()
            bg_logger.info("[LLMENGINE Init] 4 Initialize KV Caches")
        end_kv_init_at = time.time()
        print(f"[TTFT BREAKDOWN]: Profile KV Init Time: {end_kv_init_at - start_kv_init_at:.4f}")
````

````bash
python server.py --model_path=/mnt/n0/models/llama3.18b_intruc_chinese/

````

## Prefill Time
Time between "Time finishs LLM Engine Init" and first_token_time
````bash
python server.py --model_path=/mnt/n0/models/llama3.18b_intruc_chinese/
````

The CUDA 