
setup:
````bash
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=4
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --reuse-store 2 --port 8073
````

run server.py
````bash
cd sslm/ServerlessLLM/tools/CRIU/criu_rpc/
conda activate /mnt/n0/.conda/envs/sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=4
nohup python server.py -
-socket_addr=/tmp/criu_service.socket --model_path=/mnt/n0/models/vllm/opt6.7b_tmp --dump=0 > tmp.log 2>&1 &

````
## 'prefix_caching'和输入对推理结果的影响


关闭'prefix_caching'之后，推理结果出错：
````
Processed prompts:   0%| | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s[AsyncBlockManager] Block Mapping: {552: 13317210112}
[AsyncBlockManager] Block Mapping: {552: 13317210112}
[AsyncBlockManager] Block Mapping: {552: 13317210112}
[AsyncBlockManager] Block Mapping: {552: 13317210112}
Processed prompts: 100%|█| 1/1 [00:00<00:00,  1.53it/s, est. speed input: 3.06 toks/s, output: 6.1
[RequestOutput(request_id=0, prompt='hello', prompt_token_ids=[2, 42891], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text=' P P P P', token_ids=[221, 221, 221, 221], cumulative_logprob=-0.38687101006507874, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1757146096.00003, last_token_time=1757146096.00003, first_scheduled_time=1757146096.0251188, first_token_time=1757146096.4711208, time_in_queue=0.025088787078857422, finished_time=1757146096.678075), lora_request=None)]
Processed prompts:   0%| | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s[AsyncBlockManager] Block Mapping: {552: 13317210112}
[AsyncBlockManager] Block Mapping: {552: 13317210112}
[AsyncBlockManager] Block Mapping: {552: 13317210112}
[AsyncBlockManager] Block Mapping: {552: 13317210112}
Processed prompts: 100%|█| 1/1 [00:00<00:00,  8.56it/s, est. speed input: 34.35 toks/s, output: 34
[RequestOutput(request_id=1, prompt='how are you', prompt_token_ids=[2, 9178, 32, 47], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text=' 10 10 10 10', token_ids=[158, 158, 158, 158], cumulative_logprob=0.0, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1757146096.6788507, last_token_time=1757146096.6788507, first_scheduled_time=1757146096.679827, first_token_time=1757146096.7142444, time_in_queue=0.0009763240814208984, finished_time=1757146096.796008), lora_request=None)]
RPC started, monitor on port 50051...
````
seq_0:
    block_id: 552
    slot mapping:
        17664, 17665
        17666
        17667
        17668
seq_0:
    block_id: 552
    slot mapping:
        17664, 17665, 17666, 17667
        17668
        17669
        17670



开启'prefix_caching', 第一个任务正常，第二个任务出错：
````
Processed prompts:   0%| | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s[AsyncBlockManager] Block Mapping: {0: 13317210112}
[AsyncBlockManager] Block Mapping: {0: 13317210112}
[AsyncBlockManager] Block Mapping: {0: 13317210112}
[AsyncBlockManager] Block Mapping: {0: 13317210112}
Processed prompts: 100%|█| 1/1 [00:00<00:00,  1.70it/s, est. speed input: 3.41 toks/s, output: 6.8
[RequestOutput(request_id=0, prompt='hello', prompt_token_ids=[2, 42891], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='. im new to', token_ids=[4, 4356, 92, 7], cumulative_logprob=-8.86873859167099, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1757146191.9332762, last_token_time=1757146191.9332762, first_scheduled_time=1757146191.9486773, first_token_time=1757146192.3490255, time_in_queue=0.015401124954223633, finished_time=1757146192.5345604), lora_request=None)]
Processed prompts:   0%| | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s[AsyncBlockManager] Block Mapping: {1: 13333987328}
[AsyncBlockManager] Block Mapping: {1: 13333987328}
[AsyncBlockManager] Block Mapping: {1: 13333987328}
[AsyncBlockManager] Block Mapping: {1: 13333987328}
Processed prompts: 100%|█| 1/1 [00:00<00:00,  8.64it/s, est. speed input: 34.62 toks/s, output: 34
[RequestOutput(request_id=1, prompt='how are you', prompt_token_ids=[2, 9178, 32, 47], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='itititit', token_ids=[405, 405, 405, 405], cumulative_logprob=0.0, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1757146192.5357928, last_token_time=1757146192.5357928, first_scheduled_time=1757146192.5371277, first_token_time=1757146192.5733092, time_in_queue=0.0013349056243896484, finished_time=1757146192.6523812), lora_request=None)]
RPC started, monitor on port 50051...
````
seq_0:
    block_id: 0
    slot mapping:
        0, 1
        2
        3
        4
seq_0:
    block_id: 1
    slot mapping:
        32, 33, 34, 35
        36
        37
        38


开启'prefix_caching'并且两个任务的输入相同，两个任务都正常：
````
Processed prompts:   0%| | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s[AsyncBlockManager] Block Mapping: {0: 13317210112}
[AsyncBlockManager] Block Mapping: {0: 13317210112}
[AsyncBlockManager] Block Mapping: {0: 13317210112}
[AsyncBlockManager] Block Mapping: {0: 13317210112}
Processed prompts: 100%|█| 1/1 [00:00<00:00,  1.44it/s, est. speed input: 2.89 toks/s, output: 5.7
[RequestOutput(request_id=0, prompt='hello', prompt_token_ids=[2, 42891], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='. im new to', token_ids=[4, 4356, 92, 7], cumulative_logprob=-8.86873859167099, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1757146281.6273053, last_token_time=1757146281.6273053, first_scheduled_time=1757146281.6432364, first_token_time=1757146282.1283793, time_in_queue=0.015931129455566406, finished_time=1757146282.335566), lora_request=None)]
Processed prompts:   0%| | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s[AsyncBlockManager] Block Mapping: {0: 13317210112}
[AsyncBlockManager] Block Mapping: {0: 13317210112}
[AsyncBlockManager] Block Mapping: {0: 13317210112}
[AsyncBlockManager] Block Mapping: {0: 13317210112}
Processed prompts: 100%|█| 1/1 [00:00<00:00, 11.19it/s, est. speed input: 22.40 toks/s, output: 44
[RequestOutput(request_id=1, prompt='hello', prompt_token_ids=[2, 42891], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='  I was wondering', token_ids=[1437, 38, 21, 8020], cumulative_logprob=-11.102068722248077, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1757146282.3367872, last_token_time=1757146282.3367872, first_scheduled_time=1757146282.3380244, first_token_time=1757146282.364067, time_in_queue=0.0012371540069580078, finished_time=1757146282.427104), lora_request=None)]
RPC started, monitor on port 50051...
````

seq_0:
    block_id: 0
    slot mapping:
        0, 1
        2
        3
        4
seq_0:
    block_id: 0
    slot mapping:
        0, 1
        2
        3
        4


## 使用更长的序列

max_tokens: 4 -> 33
prefix_caching: True
input: "hello" 

- 即使输入相同，第二个任务的输出仍旧出错

seq-0:
    block_id: 0, 1
    slot mapping:
        0, 1
        2
        ...
        33
seq-1:
    block_id: 2
    slot mapping:
        64, 65
        66,
        67
        ...
        97

- 在写入KVCache时，从Block_id开始考虑slot mapping作为偏移，相当于偏移了两次：
````cpp
  const int64_t block_idx = slot_idx / block_size;  // logical block id
  const int64_t block_offset = slot_idx % block_size;
  u_int64_t phy_block_offset = block_tables[block_idx];
 
  cache_t* block_key_ptr =
      (cache_t*)((char*)global_memory + phy_block_offset) +
      layer_id * num_heads * (head_size / x) * block_size * x * 2;
  cache_t* block_value_ptr =
      block_key_ptr + num_heads * (head_size / x) * block_size * x;
````
在原版的PagedAttention中，KV Cache是两块完整的连续内存，因此使用slot_idx作为KV块的索引是正确的，但是在我们的实现中，KV块是分散的，因此必须先通过块地址定位到块的起始位置，然后再根据slot_idx在块内偏移（slot_idx必须是块内的偏移量，而不是全局的偏移量）。

最简单的一种方法：slot mapping -= block_id[0] * block_size
原来的slot_idx是全局的，现在转换为当前块表之类的偏移
但是，如果slot_idx不是连续的，那么就会出现错误


另一种方法：
    把(block_id, block_addr)映射表一起传入内核
    在内核中根据slot_idx计算block_id，并获得对应的block_addr

另另一种方法：
    把完整的block mapping表一起传入内核
    最简单
