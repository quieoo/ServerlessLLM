import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    import torch


def _add_local_packages_to_path() -> None:
    """Prefer the repo-local ElasticKV vLLM and sllm_store packages."""
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root / "ElasticKV", repo_root / "sllm_store"):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _install_sllm_torch_import_stub() -> None:
    """Avoid importing a mismatched sllm_store.torch during vLLM imports.

    ElasticKV's vLLM package imports a few sllm_store.torch helpers from its
    top-level modules. The CPU converter does not need those GPU/store helpers,
    and some local builds have a stale sllm_store._C extension that makes the
    full sllm_store.torch import fail before conversion even starts.
    """
    if "sllm_store.torch" in sys.modules:
        return

    stub = types.ModuleType("sllm_store.torch")

    def _unused(*args, **kwargs):
        raise RuntimeError(
            "sllm_store.torch GPU/store helpers are unavailable in the CPU "
            "converter path."
        )

    stub.get_and_open_gpu_pool_handle = _unused
    stub.close_gpu_pool_handle = _unused
    stub.set_store_address = lambda *_args, **_kwargs: None
    sys.modules["sllm_store.torch"] = stub


def _init_single_rank_parallel_state() -> None:
    """Initialize vLLM's TP/PP state for a CPU-only TP=1 conversion."""
    import torch
    import torch.distributed as dist
    from vllm.distributed.parallel_state import (
        initialize_model_parallel,
        model_parallel_is_initialized,
        set_custom_all_reduce,
    )
    from vllm.distributed.parallel_state import init_distributed_environment

    if model_parallel_is_initialized():
        return

    set_custom_all_reduce(False)
    if not dist.is_initialized():
        init_file = TemporaryDirectory()
        _INIT_DIRS.append(init_file)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{init_file.name}/dist_init",
            backend="gloo",
        )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )


def _copy_metadata_files(input_dir: str, output_dir: str) -> None:
    for file_name in os.listdir(input_dir):
        if os.path.splitext(file_name)[1] in (".bin", ".pt", ".safetensors"):
            continue
        src_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(output_dir, file_name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy(src_path, dest_path)


def _str_to_int_tuple(value: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _looks_like_vision_language_model(hf_config) -> bool:
    architectures = getattr(hf_config, "architectures", []) or []
    if any("llava" in architecture.lower() for architecture in architectures):
        return True
    return hasattr(hf_config, "vision_config") and hasattr(hf_config, "text_config")


def _build_vision_language_config(
    model_config,
    image_input_type: Optional[str],
    image_token_id: Optional[int],
    image_input_shape: Optional[str],
    image_feature_size: Optional[int],
    image_processor: Optional[str],
):
    from vllm.config import VisionLanguageConfig

    if image_input_type is None and not _looks_like_vision_language_model(
        model_config.hf_config
    ):
        return None

    hf_config = model_config.hf_config
    vision_config = getattr(hf_config, "vision_config", None)
    if image_input_type is None:
        image_input_type = "pixel_values"

    if image_token_id is None:
        image_token_id = getattr(hf_config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(hf_config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError(
            "Cannot infer image token id. Pass --image_token_id explicitly."
        )

    if image_input_shape is None:
        if vision_config is None:
            raise ValueError(
                "Cannot infer image input shape without vision_config. "
                "Pass --image_input_shape explicitly, for example 1,3,336,336."
            )
        image_size = getattr(vision_config, "image_size", None)
        num_channels = getattr(vision_config, "num_channels", 3)
        if image_size is None:
            raise ValueError(
                "Cannot infer image size from vision_config. "
                "Pass --image_input_shape explicitly, for example 1,3,336,336."
            )
        image_input_shape_tuple = (1, num_channels, image_size, image_size)
    else:
        image_input_shape_tuple = _str_to_int_tuple(image_input_shape)

    if image_feature_size is None:
        if vision_config is None:
            raise ValueError(
                "Cannot infer image feature size without vision_config. "
                "Pass --image_feature_size explicitly."
            )
        image_size = getattr(vision_config, "image_size", None)
        patch_size = getattr(vision_config, "patch_size", None)
        if image_size is None or patch_size is None:
            raise ValueError(
                "Cannot infer image feature size from vision_config. "
                "Pass --image_feature_size explicitly."
            )
        image_feature_size = (image_size // patch_size) ** 2
        if getattr(hf_config, "vision_feature_select_strategy", None) == "full":
            image_feature_size += 1

    if image_processor is None:
        image_processor = model_config.model

    return VisionLanguageConfig(
        image_input_type=VisionLanguageConfig.get_image_input_enum_type(
            image_input_type
        ),
        image_token_id=image_token_id,
        image_input_shape=image_input_shape_tuple,
        image_feature_size=image_feature_size,
        image_processor=image_processor,
        image_processor_revision=model_config.revision,
    )


def _save_state_dict(
    state_dict: "Dict[str, torch.Tensor]",
    output_rank_path: str,
    save_format: str,
    chunk_megabytes: int,
) -> None:
    if save_format == "auto":
        save_format = "tensor_group" if "tmp" in output_rank_path else "tensor_index"

    if save_format == "tensor_group":
        _save_tensor_group_dict(
            state_dict,
            output_rank_path,
            chunk_megabytes=chunk_megabytes,
        )
    elif save_format == "tensor_index":
        _save_dict(state_dict, output_rank_path)
    else:
        raise ValueError(f"Unknown save_format: {save_format}")


def _tensor_hash_fingerprint(tensor: "torch.Tensor") -> str:
    tensor_bytes = tensor.numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def _tensor_group_hash_fingerprint(tensor_group: list["torch.Tensor"]) -> str:
    tensor_hashes = [_tensor_hash_fingerprint(tensor) for tensor in tensor_group]
    return hashlib.sha256("".join(tensor_hashes).encode()).hexdigest()


def _get_tensor_data_index(state_dict: "Dict[str, torch.Tensor]") -> Dict[str, tuple]:
    tensor_data_index = {}
    for name, param in state_dict.items():
        param_storage = param.untyped_storage()
        tensor_data_index[name] = (param_storage.data_ptr(), param_storage.size())
    return tensor_data_index


def _save_dict(
    state_dict: "Dict[str, torch.Tensor]",
    model_path: str,
) -> None:
    from sllm_store._C import save_tensors

    os.makedirs(model_path, exist_ok=True)
    tensor_names = list(state_dict.keys())
    tensor_data_index = _get_tensor_data_index(state_dict)
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

    tensor_index = {}
    for name, param in state_dict.items():
        tensor_index[name] = (
            tensor_offsets[name],
            tensor_data_index[name][1],
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
        )

    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)


def _save_tensor_group_dict(
    state_dict: "Dict[str, torch.Tensor]",
    model_path: str,
    chunk_megabytes: int = 8,
) -> None:
    from sllm_store._C import save_tensors

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)

    tensor_names = list(state_dict.keys())
    tensor_data_index = _get_tensor_data_index(state_dict)
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

    dump_index = []
    tensor_group_index = []
    tensor_group_data = []
    current_tensor_group_size = 0
    tensor_group_size = chunk_megabytes * 1024 * 1024

    for name, param in state_dict.items():
        tensor_group_index.append(
            (name, tensor_offsets[name], tensor_data_index[name][1])
        )
        tensor_group_data.append(param)
        current_tensor_group_size += tensor_data_index[name][1]
        if current_tensor_group_size >= tensor_group_size:
            fingerprint = _tensor_group_hash_fingerprint(tensor_group_data)
            tensor_group_offset = tensor_group_index[0][1]
            for i, tensor_index in enumerate(tensor_group_index):
                tensor_group_index[i] = (
                    tensor_index[0],
                    tensor_index[1] - tensor_group_offset,
                    *tensor_index[2:],
                )
            dump_index.append(
                (
                    tensor_group_offset,
                    current_tensor_group_size,
                    fingerprint,
                    tensor_group_index,
                )
            )
            tensor_group_index = []
            current_tensor_group_size = 0
            tensor_group_data = []

    if tensor_group_index:
        fingerprint = _tensor_group_hash_fingerprint(tensor_group_data)
        tensor_group_offset = tensor_group_index[0][1]
        for i, tensor_index in enumerate(tensor_group_index):
            tensor_group_index[i] = (
                tensor_index[0],
                tensor_index[1] - tensor_group_offset,
                *tensor_index[2:],
            )
        dump_index.append(
            (
                tensor_group_offset,
                current_tensor_group_size,
                fingerprint,
                tensor_group_index,
            )
        )

    with open(os.path.join(model_path, "tensor_group_index.txt"), "w") as f:
        f.write(f"Number of tensor groups: {len(dump_index)}\n\n")
        for tensor_group_offset, group_size, fingerprint, group_index in dump_index:
            f.write(f"Group Offset: {tensor_group_offset}\n")
            f.write(f"Group Size: {group_size}\n")
            f.write(f"Fingerprint: {fingerprint}\n")
            f.write("Tensor Group Index:\n")
            for name, offset, size in group_index:
                f.write(f"  Tensor Name: {name}\n")
                f.write(f"  Offset: {offset}\n")
                f.write(f"  Size: {size}\n")
            f.write("\n")

    tensor_meta_index = {}
    for name, param in state_dict.items():
        tensor_meta_index[name] = (
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
        )
    with open(os.path.join(model_path, "tensor_meta_index.json"), "w") as f:
        json.dump(tensor_meta_index, f)


def convert_hf_to_serverless_rank0(
    model_name: str,
    pretrained_model_name_or_path: str,
    storage_path: str,
    torch_dtype: str,
    load_format: str,
    trust_remote_code: bool,
    save_format: str,
    chunk_megabytes: int,
    overwrite: bool,
    image_input_type: Optional[str],
    image_token_id: Optional[int],
    image_input_shape: Optional[str],
    image_feature_size: Optional[int],
    image_processor: Optional[str],
) -> str:
    import torch
    from vllm.config import (
        CacheConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
    )
    from vllm.model_executor.model_loader.loader import (
        DefaultModelLoader,
        ServerlessLLMLoader,
        _initialize_model,
    )
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype

    _init_single_rank_parallel_state()

    model_path = os.path.abspath(pretrained_model_name_or_path)
    output_model_path = os.path.join(storage_path, model_name)
    output_rank_path = os.path.join(output_model_path, "rank_0")

    if os.path.exists(output_model_path):
        if not overwrite:
            raise FileExistsError(
                f"{output_model_path} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_model_path)
    os.makedirs(output_model_path, exist_ok=True)

    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
        tokenizer_mode="auto",
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
        seed=0,
        max_model_len=1,
        enforce_eager=True,
        skip_tokenizer_init=True,
    )
    load_config = LoadConfig(
        load_format=load_format,
        download_dir=model_path,
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        distributed_executor_backend="mp",
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.01,
        swap_space=0,
        cache_dtype="auto",
        num_gpu_blocks_override=1,
        sliding_window=model_config.get_sliding_window(),
    )
    model_config.verify_with_parallel_config(parallel_config)
    cache_config.verify_with_parallel_config(parallel_config)
    vision_language_config = _build_vision_language_config(
        model_config,
        image_input_type=image_input_type,
        image_token_id=image_token_id,
        image_input_shape=image_input_shape,
        image_feature_size=image_feature_size,
        image_processor=image_processor,
    )

    loader = DefaultModelLoader(load_config)
    with set_default_torch_dtype(model_config.dtype):
        with torch.device("cpu"):
            model = _initialize_model(
                model_config,
                load_config,
                lora_config=None,
                vision_language_config=vision_language_config,
                cache_config=cache_config,
            )

        model.load_weights(
            loader._get_weights_iterator(
                model_config.model,
                model_config.revision,
                fall_back_to_pt=getattr(
                    model,
                    "fall_back_to_pt_during_load",
                    True,
                ),
            )
        )

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)
            if hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading()

    model = model.eval()
    state_dict = ServerlessLLMLoader._filter_subtensors(model.state_dict())
    cpu_state_dict = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in state_dict.items()
    }
    _save_state_dict(
        cpu_state_dict,
        output_rank_path,
        save_format=save_format,
        chunk_megabytes=chunk_megabytes,
    )
    _copy_metadata_files(model_path, output_model_path)

    del cpu_state_dict
    del state_dict
    del model
    gc.collect()

    return output_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a HuggingFace model to Tangram/ServerlessLLM vLLM format "
            "as a single TP=1 rank_0 directory, using CPU model loading."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Output model name under --storage_path.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        required=True,
        help="Local HuggingFace model snapshot path.",
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="./models",
        help="Directory where the converted model directory is written.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=("auto", "float16", "bfloat16", "float32", "half"),
        help="Dtype used when initializing the vLLM model on CPU.",
    )
    parser.add_argument(
        "--load_format",
        type=str,
        default="auto",
        choices=("auto", "safetensors", "pt", "npcache"),
        help="HF checkpoint format to read.",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="auto",
        choices=("auto", "tensor_group", "tensor_index"),
        help=(
            "Tangram output format. auto matches current loader.py behavior: "
            "tensor_group when the output rank path contains 'tmp', otherwise tensor_index."
        ),
    )
    parser.add_argument(
        "--chunk_megabytes",
        type=int,
        default=8,
        help="Chunk size used by tensor_group output.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading custom model code from the HF model config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output model directory if it already exists.",
    )
    parser.add_argument(
        "--image_input_type",
        type=str,
        choices=("pixel_values", "image_features"),
        help=(
            "Vision-language image input type. For Llava this defaults to "
            "pixel_values when omitted."
        ),
    )
    parser.add_argument(
        "--image_token_id",
        type=int,
        help=(
            "Image token id. If omitted, the converter tries image_token_index "
            "or image_token_id from config.json."
        ),
    )
    parser.add_argument(
        "--image_input_shape",
        type=str,
        help=(
            "Comma-separated image input shape, for example 1,3,336,336. "
            "If omitted, Llava-style configs infer it from vision_config."
        ),
    )
    parser.add_argument(
        "--image_feature_size",
        type=int,
        help=(
            "Number of image placeholder/features. If omitted, Llava-style "
            "configs infer (image_size // patch_size) ** 2."
        ),
    )
    parser.add_argument(
        "--image_processor",
        type=str,
        help="Image processor path/name. Defaults to --local_model_path.",
    )
    return parser.parse_args()


_INIT_DIRS = []


def main() -> None:
    _add_local_packages_to_path()
    _install_sllm_torch_import_stub()
    args = parse_args()
    storage_path = os.getenv("STORAGE_PATH", args.storage_path)
    output_path = convert_hf_to_serverless_rank0(
        model_name=args.model_name,
        pretrained_model_name_or_path=args.local_model_path,
        storage_path=storage_path,
        torch_dtype=args.torch_dtype,
        load_format=args.load_format,
        trust_remote_code=args.trust_remote_code,
        save_format=args.save_format,
        chunk_megabytes=args.chunk_megabytes,
        overwrite=args.overwrite,
        image_input_type=args.image_input_type,
        image_token_id=args.image_token_id,
        image_input_shape=args.image_input_shape,
        image_feature_size=args.image_feature_size,
        image_processor=args.image_processor,
    )
    print(f"Saved CPU-converted vLLM ServerlessLLM model to: {output_path}")


if __name__ == "__main__":
    main()
