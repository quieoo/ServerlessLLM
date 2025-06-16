## That Successful Cases

### CRIU Binary + sleep
In Window-1: 
````bash
# 1. sudo needs
sudo su
# 2. run the program
python test_2.py
````
test_2.py: 
````python
print(f"Current PID: {os.getpid()}")
    from vllm import SamplingParams, LLM
    print("Imported libs, init LLM in 5 seconds...")
    time.sleep(5)   
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
    llm = LLM(
        model="/mnt/n0/models/opt6.7",
        enforce_eager=True,
    )
    output=llm.generate(["Explain AI in 100 words."], sampling_params)
    print(output)
````

3. copy the output pid

In Window-2:
````bash
sudo su

# 4. dump with copyed pid
criu dump --images-dir imgs_py_2/ --shell-job -t 1135656
````

In Window-1: 

the program will be killed
````bash
# 5. restore
criu restore --images-dir imgs_py_2/ --shell-job
````

### CRIU Service + RPC

In window-2:
````bash
sudo su
# 1. run the ciru service
criu service --address criu_service.socket
````

In window-1:
````bash
sudo su
# 2. run the python program

python test_2.py criu_service.socket imgs_py_2
````
Program:
````python
def dump_process:
    ...
    # 构造 DUMP 请求
    req = rpc.criu_req()
    req.type = rpc.DUMP
    req.opts.shell_job=True
    req.opts.images_dir_fd = os.open(images_dir, os.O_DIRECTORY)
    ...
def main: 
    print(f"Current PID: {os.getpid()}")
    from vllm import SamplingParams, LLM
    print("Dump with RPC")
    dump_process(args['socket'], args['dir'])
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
    llm = LLM(
        model="/mnt/n0/models/opt6.7",
        enforce_eager=True,
    )
    output=llm.generate(["Explain AI in 100 words."], sampling_params)
    print(output)
````

````bash
# 3. restore
criu restore --images-dir imgs_py_2/ --shell-job
````

### 转储失败似乎和VLLM库相关，当激活原本的环境“conda activate /mnt/n0/.conda/envs/sllm-worker”时，能够成功转储，当激活修改后的环境“conda activate /mnt/n0/.conda/envs/sllm-worker-0.6”转储失败。

在sllm-worker-0.6的环境下，使用“pip install .”重新安装之后能够成功转储
注释掉之前安装的sllm_store.torch.py中的部分库的import，将他们移到调用前。推测使用“pip install .”重新编译sllm_store可能也能解决此问题
````python
# from accelerate.hooks import add_hook_to_module
from sllm_store._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors,
    save_tensors,
    restore_ptrs_from_store,
    open_gpu_memory_handle,
    close_gpu_memory_handle,
)
from sllm_store.client import SllmStoreClient
# from sllm_store.device_map_utils import _expand_tensor_name
from sllm_store.logger import init_logger
# from sllm_store.utils import (
#     calculate_device_memory,
#     calculate_tensor_device_offsets,
# )
````

### Dump During VLLM Engine Init

在vllm中引入dump_process
````python
# 封装转储操作
def dump_process(socket_path, images_dir):
    """
    通过 RPC 触发 CRIU 转储操作
    :param socket_path: CRIU 服务端套接字路径
    :param images_dir: 转储镜像存储目录
    :return: 转储是否成功（True/False）
    """
    try:
        # 连接服务端
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        s.connect(socket_path)

        # 构造 DUMP 请求
        req = rpc.criu_req()
        req.type = rpc.DUMP
        # req.opts.leave_running = True  # 转储后原进程继续运行
        req.opts.shell_job=True
        # req.opts.log_level = 4
        req.opts.images_dir_fd = os.open(images_dir, os.O_DIRECTORY)
        # req.opts.network_lock = rpc.SKIP

        # 发送请求
        s.send(req.SerializeToString())

        # 接收响应
        resp = rpc.criu_resp()
        resp.ParseFromString(s.recv(1024))
        # 验证响应
        if resp.type != rpc.DUMP:
            print("转储失败：意外的响应类型")
            return False
        if not resp.success:
            print("转储失败：CRIU 执行错误")
            return False
        print("转储成功！镜像存储于：{}".format(images_dir))
        if resp.dump.restored:
            print("CRIU恢复进程")
        return True
    except Exception as e:
        print("转储异常：{}".format(str(e)))
        return False
    finally:
        s.close()
        if 'req' in locals():  # 增加存在性检查
            os.close(req.opts.images_dir_fd)

````

测试在初始化过程中哪个位置可以成功转储
1. llm_engine
````
        
        # -------------------------------
        self.model_executor = executor_class(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            vision_language_config=vision_language_config,
            speculative_config=speculative_config,
            load_config=load_config,
        )
````
1.1 gpu_executor
````
        # -------------------------------
        self.driver_worker = self._create_worker()
````

1.1.1 worker
````
        # -------------------------------
        self.model_runner = ModelRunnerClass(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config,
        )
````

1.1.1.1 model_runner
````
        # -------------------------------
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
        )
````