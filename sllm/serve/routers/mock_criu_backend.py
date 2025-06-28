import logging

import time
from sllm.serve.utils import RPCBackend
import ray
from typing import Dict, Any, List



logger = logging.getLogger("ray")

class MOCKCRIURPCBackend(RPCBackend):
    def __init__(
        self, model, node_id, device_id
    ) -> None:
        self.model=model
        self.node_id=node_id
        self.device_id=device_id
        try:
            self.store_manager = ray.get_actor("store_manager")
        except ValueError:
            logger.error("Store manager not found")
            return

    
    async def init_backend(self) -> None:
        # Restore Engine Time + CUDA Kernerl Init Time
        time.sleep(0.8+0.4)
    
        await self.store_manager.load_model.remote(self.model, self.node_id, self.device_id)
        logger.info(f"Mock CRIU Backend init {self.model} {self.node_id} {self.device_id}")




    async def generate(self, request_data: Dict[str, Any]):

        messages: List[Dict[str, str]] = request_data.get("messages", [])
        # 合并message成一条prompt
        prompt=""
        for message in messages:
            prompt+=message["content"]
        # 异步调用Generate方法

        # Prefill Time
        time.sleep(0.2)

        # construct output
        data_list=[]
        data_list.append({
            "metrics":{
                "first_token_time":time.time()
            }
        })
        
        return {
            "object": "list",
            "data": data_list,
            "model": self.model,
            "usage": {
                "total_prompts": 1,
                "successful_prompts": 1,
            }
        }
    
    async def shutdown(self):
        logger.info("Mock CRIU Backend shutdown")
    
    async def stop(self) -> None:
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        logger.info("CRIU Backend get current tokens")
        return List[List[int]]()

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        # do nothing
        logger.info(f"CRIU Backend resume kv cache {request_datas}")

    async def encode(self, request_data: Dict[str, Any]):
        logger.info(f"CRIU Backend encode")
        return Dict[str, Any]()

