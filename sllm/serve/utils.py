# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import ray

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional

sllm_store_base_port=8073

class RPCBackend(ABC):
    @abstractmethod
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    @abstractmethod
    async def init_backend(self) -> None:
        pass

    @abstractmethod
    async def encode(self, request_data: Dict[str, Any]):
        pass

    @abstractmethod
    async def generate(self, request_data: Dict[str, Any]):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    async def get_current_tokens(self) -> List[List[int]]:
        pass

    @abstractmethod
    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        pass


def get_worker_nodes():
    ray_nodes = ray.nodes()
    worker_node_info = {}
    for node in ray_nodes:
        ray_node_id = node.get("NodeID", None)
        assert ray_node_id is not None, "NodeID not found"
        resources = node.get("Resources", {})
        assert resources != {}, "Resources not found"
        node_address = node.get("NodeManagerAddress", None)
        assert (
            node_address is not None and node_address != ""
        ), "NodeManagerAddress not found"
        if resources.get("control_node", 0) > 0:
            continue  # Skip the control node

        for key, value in resources.items():
            if key.startswith("worker_id_"):
                node_id = key.split("_")[-1]
                worker_node_info[node_id] = {
                    "ray_node_id": ray_node_id,
                    "address": node_address,
                    "free_gpu": resources.get("GPU", 0),
                    "total_gpu": resources.get("GPU", 0),
                    "store_port": resources.get("store_port", sllm_store_base_port),
                }
    return worker_node_info


@dataclass
class InstanceStatus:
    instance_id: str
    node_id: str
    num_gpu: int
    concurrency: int

    model_name: Optional[str] = None
    num_current_tokens: Optional[int] = None
    resuming_latency: Optional[float] = None


@dataclass
class InstanceHandle:
    instance_id: str
    max_queue_length: int
    num_gpu: int

    node_id: Optional[str] = None
    backend_instance: Optional[ray.actor.ActorHandle] = None
    ready: bool = False
    concurrency: int = 0

    # criu backend
    criu_backend: Optional[RPCBackend] = None

    lock: asyncio.Lock = asyncio.Lock()

    async def add_requests(self, num_requests: int = 1):
        async with self.lock:
            if not self.ready:
                return False
            if (
                self.concurrency + num_requests > self.max_queue_length
                or self.concurrency + num_requests < 0
            ):
                return False
            self.concurrency += num_requests
            return True

    async def check_request_queue(self):
        async with self.lock:
            return self.concurrency + 1 <= self.max_queue_length

    async def get_status(self):
        async with self.lock:
            return InstanceStatus(
                self.instance_id,
                self.node_id,
                self.num_gpu,
                self.concurrency,
            )
