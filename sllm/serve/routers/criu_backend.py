import logging

import grpc
from sllm.serve.routers import worker_rpc_pb2
from sllm.serve.routers import worker_rpc_pb2_grpc
from typing import Any, Dict, List, Optional, Sequence, Union, cast
import time
from sllm.serve.utils import RPCBackend

import sllm_store.proto.storage_pb2 as storage_pb2
import sllm_store.proto.storage_pb2_grpc as storage_pb2_grpc
# import ray



logger = logging.getLogger("ray")
socket_path = "/tmp/criu_service.socket"
images_dir = "/mnt/n0/models/vllm/"
storage_server_port="8073"
llm_engine_server_port="50051"


class CRIURPCBackend(RPCBackend):
    def __init__(
        self, model: str, node_addr:str
    ) -> None:
        self.model=model
        self.node_addr=node_addr
        self.vllm_stub=None


    
    async def init_backend(self) -> None:
        # 连接storage rpc server
        # 组装并发送RestoreEngine请求
        async with grpc.aio.insecure_channel(self.node_addr+":"+storage_server_port) as channel:
            try: 
                stub = storage_pb2_grpc.StorageStub(channel)
                response = await stub.RestoreCRIUEngine(
                    storage_pb2.RestoreCRIUEngineRequest(
                        socket_addr=socket_path,
                        images_dir="vllm/"+self.model,
                    )
                )
                if response.code!=0:
                    logger.error(f"CRIU restore failed {response.code}")
            except grpc.aio.AioRpcError as e:
                logger.error(f"CRIU restore failed {e}")
                return 
        
        
        print("----------criu restore successfully-----------")
        # 尝试连接vllm rpc server
        # 截取storage_node_addr的IP地址，替换端口号
        vllm_rpc_server_addr=self.node_addr+":"+llm_engine_server_port
        # 连接vllm rpc server, 保存stub
        max_try=500
        while True:
            try:
                self.vllm_channel = grpc.aio.insecure_channel(vllm_rpc_server_addr)
                self.vllm_stub = worker_rpc_pb2_grpc.CRIUServiceStub(self.vllm_channel)
                # 异步调用Init方法
                ret = await self.vllm_stub.Init(worker_rpc_pb2.InitRequest(config_path="test_config"))
                break
            except grpc.aio.AioRpcError as e:
                # logger.error(f"CRIU restore failed {e}")
                time.sleep(0.01)
                max_try-=1
                if max_try<=0:
                    logger.error(f"CRIU restore failed {e}")
                    raise e
        print(f"----------criu vllm rpc server connected in {500-max_try} trys-----------")
    async def generate(self, request_data: Dict[str, Any]):

        messages: List[Dict[str, str]] = request_data.get("messages", [])
        # 合并message成一条prompt
        prompt=""
        for message in messages:
            prompt+=message["content"]
        # 异步调用Generate方法
        try: 
            ret = await self.vllm_stub.Run(
                worker_rpc_pb2.RunRequest(
                    task_id=prompt,
                )
            )
        except grpc.aio.AioRpcError as e:
            logger.error(f"CRIU generate failed {e}")
            raise e
        return {
            "object": "list",
            "data": ret.result,
            "model": self.model,
            "usage": {
                "total_prompts": 1,
                "successful_prompts": 1,
            }
        }
    
    async def shutdown(self):
        try:
            await self.vllm_stub.Shutdown(worker_rpc_pb2.ShutdownRequest())
        except grpc.aio.AioRpcError as e:
            logger.error(f"CRIU shutdown failed {e}")
            raise e
        # 显式关闭通道
        if self.vllm_channel:
            await self.vllm_channel.close()
    
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

