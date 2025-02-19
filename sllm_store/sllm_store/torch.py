# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import json
import os
import time
import uuid
from typing import Dict, Optional, Union


import torch
import hashlib
import shutil


# from accelerate.hooks import add_hook_to_module
from sllm_store._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors,
    save_tensors,
    restore_ptrs_from_store,
)
from sllm_store.client import SllmStoreClient
from sllm_store.device_map_utils import _expand_tensor_name
from sllm_store.logger import init_logger
from sllm_store.utils import (
    calculate_device_memory,
    calculate_tensor_device_offsets,
)

logger = init_logger(__name__)


def _get_uuid():
    return str(uuid.uuid4())


def tensor_hash_fingerprint(tensor: torch.Tensor) -> str:
    tensor_bytes = tensor.numpy().tobytes()
    tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()
    return tensor_hash

def tensor_group_hash_fingerprint(tensor_group: list[torch.Tensor]) -> str:

    tensor_hashes = [tensor_hash_fingerprint(tensor) for tensor in tensor_group]
    tensor_group_hash = hashlib.sha256("".join(tensor_hashes).encode()).hexdigest()
    return tensor_group_hash


def save_dict(
    state_dict: Dict[str, torch.Tensor], model_path: Union[str, os.PathLike]
):
    tensor_names = list(state_dict.keys())
    tensor_data_index = {}
    for name, param in state_dict.items():
        param_storage = param.untyped_storage()
        data_ptr = param_storage.data_ptr()
        size = param_storage.size()
        tensor_data_index[name] = (data_ptr, size)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # save tensors
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

    # create tensor index
    tensor_index = {}
    for name, param in state_dict.items():
        # name: offset, size
        tensor_index[name] = (
            tensor_offsets[name],
            tensor_data_index[name][1],
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
        )

    # save tensor index
    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)


def save_tensor_group_dict(
    state_dict: Dict[str, torch.Tensor], model_path: Union[str, os.PathLike]
):
    
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    tensor_names = list(state_dict.keys())
    tensor_data_index = {}
    for name, param in state_dict.items():
        param_storage = param.untyped_storage()
        data_ptr = param_storage.data_ptr()
        size = param_storage.size()
        tensor_data_index[name] = (data_ptr, size)

    # Save tensors
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

    # Create tensor group index (plain text format)
    dump_index = []
    tensor_group_index = []
    tensor_group_data = []
    current_tensor_group_size = 0
    tensor_group_size = 8 * 1024 * 1024  # 8MB
    # Iterate tensors, add each tensor to the current tensor_group until tensor_group_size is reached
    # Save the tensor_group_index
    for name, param in state_dict.items():
        # Prepare tensor group index data
        tensor_group_index.append(
            (name, tensor_offsets[name], tensor_data_index[name][1])
        )
        tensor_group_data.append(param)
        current_tensor_group_size += tensor_data_index[name][1]
        if current_tensor_group_size >= tensor_group_size:
            # Get tensors from this tensor_group and compute tensor_group fingerprint
            fingerprint = tensor_group_hash_fingerprint(tensor_group_data)
            tensor_group_offset = tensor_group_index[0][1]
            for i, tensor_index in enumerate(tensor_group_index):
                tensor_group_index[i] = (tensor_index[0], tensor_index[1] - tensor_group_offset, *tensor_index[2:])

            dump_index.append(
                (tensor_group_offset, current_tensor_group_size, fingerprint, tensor_group_index)
            )
            tensor_group_index = []
            current_tensor_group_size = 0
            tensor_group_data = []
    # Save the last tensor group if it's not empty
    if tensor_group_index:
        fingerprint = tensor_group_hash_fingerprint(tensor_group_data)
        tensor_group_offset = tensor_group_index[0][1]
        for i, tensor_index in enumerate(tensor_group_index):
            tensor_group_index[i] = (tensor_index[0], tensor_index[1] - tensor_group_offset, *tensor_index[2:])
        dump_index.append(
            (tensor_group_offset, current_tensor_group_size, fingerprint, tensor_group_index)
        )
    

    # Save tensor group index as a readable plain text file
    index_file_path = os.path.join(model_path, "tensor_group_index.txt")
    with open(index_file_path, "w") as f:
        # Write the number of tensor groups
        f.write(f"Number of tensor groups: {len(dump_index)}\n\n")

        # Write each tensor group entry
        for entry in dump_index:
            tensor_group_offset, current_tensor_group_size, fingerprint, group_index = entry
            f.write(f"Group Offset: {tensor_group_offset}\n")
            f.write(f"Group Size: {current_tensor_group_size}\n")
            f.write(f"Fingerprint: {fingerprint}\n")
            f.write("Tensor Group Index:\n")
            for tensor_entry in group_index:
                name, offset, size = tensor_entry
                f.write(f"  Tensor Name: {name}\n")
                f.write(f"  Offset: {offset}\n")
                f.write(f"  Size: {size}\n")
            f.write("\n")
    
    # Save the tensor meta index
    tensor_meta_index = {}
    for name, param in state_dict.items():
        tensor_meta_index[name] = (
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
        )
    with open(os.path.join(model_path, "tensor_meta_index.json"), "w") as f:
        json.dump(tensor_meta_index, f)

    # save the state_dict to file for further verification
    # torch.save(state_dict, os.path.join(model_path, "state_dict.pth"))
    # print("state_dict saved to file")

def load_dict(
    model_path: Union[str, os.PathLike],
    device_map: Dict[str, int],
    storage_path: Optional[str] = None,
):
    # TODO
    if "tmp" in model_path:
        print("Using ReuseStore")
        return load_dict_async(model_path, device_map, storage_path)
    else:
        print("Using SLLMStore")
        replica_uuid, state_dict = load_dict_non_blocking(
            model_path, device_map, storage_path
        )

        client = SllmStoreClient("127.0.0.1:8073")
        client.confirm_model_loaded(model_path, replica_uuid)
        # for k, v in state_dict.items():
        #     print(f"{k} : {torch.sum(torch.abs(v)).item()}")

        return state_dict

def load_dict_async(
    model_path: Optional[Union[str, os.PathLike]],
    device_map: Dict[str, int],
    storage_path: Optional[str] = None,
):
    client = SllmStoreClient("127.0.0.1:8073")
    ret = client.load_into_cpu(model_path)
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into CPU")

    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", "./models")
    with open(
        os.path.join(storage_path, model_path, "tensor_meta_index.json"), "r"
    ) as f:
        tensor_index = json.load(f)
    
    tensor_meta_index = {}
    for name, (shape, stride, dtype) in tensor_index.items():
        tensor_meta_index[name] = (shape, stride, dtype)

    tensor_names = list(tensor_meta_index.keys())
    device_ptrs, tensor_offsets = restore_ptrs_from_store(ret.model_path, tensor_names)
    state_dict=restore_tensors(tensor_meta_index, device_ptrs, tensor_offsets)

    # # read the state_dict from file and verify 
    # state_dict_from_file=torch.load(os.path.join(storage_path, model_path, "state_dict.pth"))
    
    # for k, v in state_dict_from_file.items():
    #     print(f"checking {k}")
        
    #     # 检查 key 是否存在
    #     if k not in state_dict:
    #         raise ValueError(f"Missing key {k} in state_dict")
        
    #     # 将 state_dict 中的张量从 GPU 转移到 CPU 进行比较
    #     if not torch.equal(v, state_dict[k].cpu()):
    #         raise ValueError(f"Mismatch for key {k}")
        
    # print("state_dict is verified")

    # for k, v in state_dict.items():
    #     print(f"{k} : {torch.sum(torch.abs(v)).item()}")

    return state_dict

    


def load_dict_non_blocking(
    model_path: Optional[Union[str, os.PathLike]],
    device_map: Dict[str, int],
    storage_path: Optional[str] = None,
):
    client = SllmStoreClient("127.0.0.1:8073")
    ret = client.load_into_cpu(model_path)
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into CPU")

    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", "./models")
    with open(
        os.path.join(storage_path, model_path, "tensor_index.json"), "r"
    ) as f:
        tensor_index = json.load(f)

    tensor_meta_index = {}
    tensor_data_index = {}
    for name, (offset, size, shape, stride, dtype) in tensor_index.items():
        tensor_meta_index[name] = (shape, stride, dtype)
        tensor_data_index[name] = (offset, size)

    start = time.time()
    expanded_device_map = _expand_tensor_name(
        device_map, list(tensor_index.keys())
    )
    device_memory = calculate_device_memory(
        expanded_device_map, tensor_data_index
    )
    # logger.debug(f"calculate_device_memory {device_memory}")
    cuda_memory_ptrs = allocate_cuda_memory(device_memory)
    # cuda_memory_ptrs = { k: [v] for k,v in cuda_memory_ptrs.items()}
    cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
    device_uuid_map = get_device_uuid_map()
    # logger.debug(f"determine device_uuid_map {device_uuid_map}")
    tensor_device_offsets, tensor_copy_chunks = calculate_tensor_device_offsets(
        expanded_device_map, tensor_data_index
    )
    logger.debug(f"allocate_cuda_memory takes {time.time() - start} seconds")

    replica_uuid = _get_uuid()
    ret = client.load_into_gpu(
        model_path,
        replica_uuid,
        {
            device_uuid_map[device_id]: v
            for device_id, v in tensor_copy_chunks.items()
        },
        {
            device_uuid_map[device_id]: [v]
            for device_id, v in cuda_memory_handles.items()
        },
    )
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into GPU")

    # load model state_dict
    start = time.time()
    state_dict = restore_tensors(
        tensor_meta_index, cuda_memory_ptrs, tensor_device_offsets
    )
    logger.info(f"restore state_dict takes {time.time() - start} seconds")

    return replica_uuid, state_dict
