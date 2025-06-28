import asyncio
import copy
import time
from typing import Mapping, Optional

from sllm.serve.logger import init_logger
from sllm.serve.utils import get_worker_nodes

from .scheduler_utils import SllmScheduler
from collections import defaultdict

import ray

logger = init_logger(__name__)


class ReuseAwareScheduler(SllmScheduler):
    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__()
        self.scheduler_config = scheduler_config

        self.queue_lock = asyncio.Lock()
        self.model_loading_queues = {}

        self.metadata_lock = asyncio.Lock()
        self.model_instance = {}
        self.worker_nodes = {}

        self.loop = asyncio.get_running_loop()

        self.running_lock = asyncio.Lock()
        self.running = False

        # self.static_worker_nodes = None
        self.poll_interval = 0.1

        self.store_manager = None

    async def start(self) -> None:
        async with self.running_lock:
            if self.running:
                logger.error("Reuse Aware scheduler already started")
                return
            self.running = True
        # self.static_worker_nodes = get_worker_nodes() # 假设工作节点不变化，在启动时就获取节点拓扑
        # logger.info(f"Static worker nodes: {self.static_worker_nodes}")
        try:
            self.store_manager = ray.get_actor("store_manager")
        except ValueError:
            logger.error("Store manager not found")
            return
        self.loop_task = self.loop.create_task(self._control_loop())
        logger.info("Reuse Aware scheduler started")

    async def shutdown(self) -> None:
        async with self.running_lock:
            if not self.running:
                logger.error("Reuse Aware scheduler not running")
                return
            self.running = False
        async with self.queue_lock:
            self.model_loading_queues = {}
        if self.loop_task is not None:
            await self.loop_task

    async def allocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ) -> int:
        logger.info(f"Model {model_name} requested, instance_id: {instance_id}, resources: {resources}")
        # TODO: consider other resources
        num_gpus = resources.get("num_gpus", 0)
        async with self.queue_lock:
            if model_name not in self.model_loading_queues:
                self.model_loading_queues[model_name] = []
            allocation_result = self.loop.create_future()
            self.model_loading_queues[model_name].append(
                (time.time(), num_gpus, allocation_result, instance_id)
            )
        # logger.info(f"Model {model_name} added to the loading queue")
        node_id = await allocation_result
        async with self.metadata_lock:
            if model_name not in self.model_instance:
                self.model_instance[model_name] = {}
            self.model_instance[model_name][instance_id] = node_id
        return node_id

    async def deallocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ):
        logger.info(f"Deallocating model {model_name} instance {instance_id}")
        # TODO: consider other resources
        num_gpus = resources.get("num_gpus", 0)
        async with self.metadata_lock:
            if model_name not in self.model_instance:
                logger.error(f"Model {model_name} not found")
                return
            if instance_id not in self.model_instance[model_name]:
                logger.error(f"Instance {instance_id} not found")
                return
            node_id = self.model_instance[model_name].pop(instance_id)
            # logger.info(f"Node {node_id} deallocated {num_gpus} GPUs")
            if node_id not in self.worker_nodes:
                logger.error(f"Node {node_id} not found")
                return
            self.worker_nodes[node_id]["free_gpu"] += num_gpus
        # logger.info(f"Model {model_name} instance {instance_id} deallocated")


    async def get_worker_node_status(self):
        worker_nodes = await self._get_worker_nodes()
        for node_id, node_info in worker_nodes.items():
            logger.info(f"-------------- Node {node_id} free gpu: {node_info['free_gpu']}")

    async def fcfs_schedule(self, loading_requests):
        worker_nodes = await self._get_worker_nodes()
        logger.info(f"Worker nodes: {worker_nodes}")
        update_worker_node=False
        to_remove = defaultdict(list)
        for model_name, idx, request_time, num_gpus, allocation_result in loading_requests: 
            for node_id, node_info in worker_nodes.items():
                if node_info["free_gpu"] >= num_gpus:
                    logger.info(f"Allocated node {node_id} for model {model_name}")
                    to_remove[model_name].append(idx)
                    allocation_result.set_result(node_id)
                    worker_nodes[node_id]["free_gpu"] -= num_gpus
                    update_worker_node=True
                    break
        if update_worker_node:
            await self._update_worker_nodes(worker_nodes)
        return to_remove

    def _get_model_loading_time(
        self,
        model_name: str,
        model_size: int,
        hardware_info: Mapping,
        node_waiting_time: float,
        pinned_memory_pool: Mapping,
    ) -> float:
        latency = 0
        # if model_name not in pinned_memory_pool:
        #     latency += (
        #         node_waiting_time + model_size / hardware_info["disk_bandwidth"]
        #     )
        #     logger.info(
        #         f"Loading model {model_name} will take {latency} seconds"
        #     )
        # else:
        #     latency += model_size / hardware_info["pcie_bandwidth"]
        #     logger.info(
        #         f"Loading model {model_name} will take {latency} seconds"
        #     )
        
        # 假设所有模型都在 pinned_memory_pool 中
        latency += model_size / hardware_info["pcie_bandwidth"]
        # logger.info(f"Loading model {model_name} will take {latency} seconds")
        return latency

    async def cpu_reuse_aware_schedule(self, loading_requests):
        update_worker_node=False
        to_remove = defaultdict(list)

        worker_nodes = await self._get_worker_nodes()
        # logger.info(f"Worker nodes: {worker_nodes}")
        model_info = await self.store_manager.get_model_info.remote()
        # logger.info(f"Model info: {model_info}")
        store_info = await self.store_manager.get_store_info.remote()
        # logger.info(f"Store info: {store_info}")
        hardware_info = (await self.store_manager.get_hardware_info.remote())
        # logger.info(f"Hardware info: {hardware_info}")

        for (model_name, idx, request_time, num_gpus, allocation_result) in loading_requests:
            scheduling_options = []
            for node_id, node_info in worker_nodes.items():
                if node_id not in store_info:
                    logger.error(f"Node {node_id} not found in store info")
                    continue
                free_gpu = node_info["free_gpu"]
                # logger.info(f"Node {node_id} has {free_gpu} free GPUs")
                if free_gpu >= num_gpus:
                    (
                        node_store_info,
                        pinned_memory_pool,
                        node_waiting_time,
                    ) = store_info[node_id]
                    if model_name not in node_store_info:
                        logger.info(f"Model {model_name} not found in node {node_id}")
                        # Note(Yao): Downloading from HuggingFace Hub is
                        # slower than network bandwidth and difficult to estimate.
                        # So we just consider local checkpoints for now.
                        continue
                    latency = self._get_model_loading_time(
                        model_name,
                        model_info[model_name],
                        hardware_info[node_id],
                        node_waiting_time,
                        pinned_memory_pool,
                    )
                    scheduling_options.append((node_id, latency))
            if len(scheduling_options) > 0:
                scheduling_options.sort(key=lambda x: x[1])
                logger.info(f" ============ Sorted scheduling options: {scheduling_options} ============ ")
                node_id=scheduling_options[0][0]
                allocation_result.set_result(node_id)
                to_remove[model_name].append(idx)
                worker_nodes[node_id]["free_gpu"] -= num_gpus
                update_worker_node=True
                # logger.info(f"Allocated node {node_id} for model {model_name}")
            else:
                # logger.info(f"Node {node_id} has no enough free GPUs")
                pass

        if update_worker_node:
            await self._update_worker_nodes(worker_nodes)
        return to_remove
    async def gpu_reuse_aware_schedule(self, loading_requests):
        update_worker_node=False
        to_remove = defaultdict(list)

        # t1=time.time()
        worker_nodes = await self._get_worker_nodes()
        # t2=time.time()
        # logger.info(f"Worker nodes: {worker_nodes} Time cost: {t2-t1}")
        model_info = await self.store_manager.get_model_info.remote()
        # t3=time.time()
        # logger.info(f"Model info: {model_info} Time cost: {t3-t2}")
        store_info = await self.store_manager.get_store_info.remote()
        # t4=time.time()
        # logger.info(f"Store info: {store_info} Time cost: {t4-t3}")
        hardware_info = (await self.store_manager.get_hardware_info.remote())
        # t5=time.time()
        # logger.info(f"Hardware info: {hardware_info} Time cost: {t5-t4}")

        # model_names=[model_name for model_name, _, _, _, _ in loading_requests]
        # model2node_load_size = await self.store_manager.to_load_size_models.remote(model_names)
        # t6=time.time()
        # logger.info(f"Model2node_load_size: {model2node_load_size} Time cost: {t6-t5}")

        for (model_name, idx, request_time, num_gpus, allocation_result, instance_id) in loading_requests:
            scheduling_options = []
            model2node_load_size = await self.store_manager.to_load_size.remote(model_name)
            for node_id, node_info in worker_nodes.items():
                if node_id not in store_info:
                    logger.error(f"Node {node_id} not found in store info")
                    continue
                free_gpu = node_info["free_gpu"]
                # logger.info(f"Node {node_id} has {free_gpu} free GPUs")
                if free_gpu >= num_gpus:
                    (
                        node_store_info,
                        pinned_memory_pool,
                        node_waiting_time,
                    ) = store_info[node_id]
                    if model_name not in node_store_info:
                        logger.info(f"Model {model_name} not found in node {node_id}")
                        # Note(Yao): Downloading from HuggingFace Hub is
                        # slower than network bandwidth and difficult to estimate.
                        # So we just consider local checkpoints for now.
                        continue
                    
                    to_load_size=model2node_load_size[node_id]
                    if to_load_size == -1:
                        logger.error(f"Model {model_name} not found on node {node_id} Store")
                        continue
                    latency = self._get_model_loading_time(
                        model_name,
                        to_load_size,
                        hardware_info[node_id],
                        node_waiting_time,
                        pinned_memory_pool,
                    )
                    scheduling_options.append((node_id, latency))
            if len(scheduling_options) > 0:
                scheduling_options.sort(key=lambda x: x[1])
                logger.info(f" ============ Sorted scheduling options for model {model_name} (node_id latency): {scheduling_options} ============ ")
                node_id=scheduling_options[0][0]
                allocation_result.set_result(node_id)
                to_remove[model_name].append(idx)
                worker_nodes[node_id]["free_gpu"] -= num_gpus
                update_worker_node=True
                logger.info(f"Allocated node {node_id} for model {model_name}")
            else:
                logger.info(f"Node {node_id} has no enough free GPUs")

        if update_worker_node:
            await self._update_worker_nodes(worker_nodes)
        return to_remove

    # TODO: 从当前的loading_requests中选择一个最优的任务调度资源
    # 结合 装载时延+等待时间 综合考虑
    async def gpu_reuse_aware_schedule_global(self, loading_requests):
        pass

    async def _control_loop(self):
        logger.info("Starting control loop")
        while self.running:
            # 收集当前的请求
            loading_requests = []
            async with self.queue_lock:
                for (
                    model_name,
                    loading_queue,
                ) in self.model_loading_queues.items():
                    for idx, (
                        request_time,
                        num_gpus,
                        allocation_result,
                        instance_id,
                    ) in enumerate(loading_queue):
                        loading_requests.append(
                            (
                                model_name,
                                idx,
                                request_time,
                                num_gpus,
                                allocation_result,
                                instance_id,
                            )
                        )
            
            logger.info(f"### Loading Requests {loading_requests}")
            if len(loading_requests) > 0:
                # 将模型请求按照request time排序
                loading_requests.sort(key=lambda x: x[2])
                # to_remove = await self.fcfs_schedule(loading_requests)
                # to_remove = await self.cpu_reuse_aware_schedule(loading_requests)
                to_remove = await self.gpu_reuse_aware_schedule(loading_requests)
                
                async with self.queue_lock:
                    for model_name, indices in to_remove.items():
                        # 降序排序索引，确保从后向前删除
                        for idx in sorted(indices, reverse=True):
                            self.model_loading_queues[model_name].pop(idx)
            await asyncio.sleep(self.poll_interval)


    async def _get_worker_nodes(self):
        worker_nodes = get_worker_nodes()
        async with self.metadata_lock:
            updated_worker_nodes = copy.deepcopy(self.worker_nodes)
        for node_id, node_info in worker_nodes.items():
            if node_id not in updated_worker_nodes:
                updated_worker_nodes[node_id] = copy.deepcopy(node_info)
        async with self.metadata_lock:
            self.worker_nodes = updated_worker_nodes

        return updated_worker_nodes

    # TODO: implement a dedicated class to manage worker nodes
    async def _update_worker_nodes(self, worker_nodes) -> None:
        async with self.metadata_lock:
            updated_worker_nodes = copy.deepcopy(self.worker_nodes)
        for node_id, node_info in worker_nodes.items():
            if node_id not in updated_worker_nodes:
                logger.error(f"Node {node_id} not found")
                continue
            updated_worker_nodes[node_id] = copy.deepcopy(node_info)
        async with self.metadata_lock:
            self.worker_nodes = updated_worker_nodes
        # logger.info(f"Worker nodes updated: {updated_worker_nodes}")