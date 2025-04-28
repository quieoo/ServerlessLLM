from sllm_store.client import SllmStoreClient
import time 

time_start=time.time()
client = SllmStoreClient("127.0.0.1:8073")
time_client=time.time()

# handle_str = client.get_gpu_pool_handle(0)
# time_get=time.time()
# if(not handle_str):
#     raise ValueError(f"Failed to get gpu pool handle for pool_id {0}")
# print(f"Got handle: {handle_str}")

print(f"Time to create client: {time_client-time_start:.3f}")
# print(f"Time to get handle: {time_get-time_client:.3f}")

num_blocks=client.get_available_blocks_on_gpu("vllm/opt6.7b_tmp/rank_0", 279620, 0)
print(f"Got num_blocks: {num_blocks}")