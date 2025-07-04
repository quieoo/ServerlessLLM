## Reduce VLLM Initialization Time with CRIU

### CRIU

Modify the CRIU source code to make sure it could dump anyway even meets unsupported devices (i.e. cuda)

* criu/files-ext.c
````c
int dump_unsupp_fd(struct fd_parms *p, int lfd, char *more, char *info, FdinfoEntry *e)
{
	int ret;

	ret = do_dump_gen_file(p, lfd, &ext_dump_ops, e);
	if (ret == -ENOTSUP)
		pr_err("Can't dump file %d of that type [%o] (%s %s)\n", p->fd, p->stat.st_mode, more, info);
	return 0;
	// if (ret == 0)
	// 	return 0;

	// return -1;
}
````

* criu/files.c
````c
int collect_fd(int pid, FdinfoEntry *e, struct rst_info *rst_info, bool fake)
{
	struct file_desc *fdesc;

	pr_info("Collect fdinfo pid=%d fd=%d id=%#x\n", pid, e->fd, e->id);

	fdesc = find_file_desc(e);
	if (fdesc == NULL) {
		pr_err("No file for fd %d id %#x\n", e->fd, e->id);
        // return -1;
		return 0;
	}

	if (!collect_fd_to(pid, e, rst_info, fdesc, fake, false))
		return -1;

	return 0;
}
````

rebuild and install
````bash
make
sudo make install
````

### VLLM

Update following files to make sure dump save more initailization time and result is correct

* gpu_exector.py
````python
    # CRIUCHECK:
    # 1. delayed model runner init in create_worker
    # 2. move load_model infront of init_device
    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        bg_logger.info("[GPUExecuter Init] 0 Start")
        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")
        
        self.driver_worker = self._create_worker()
        bg_logger.info("[GPUExecuter Init] 1 Create worker")
        
        self.driver_worker.load_model()
        bg_logger.info("[GPUExecuter Init] 3 Load model")

        self.driver_worker.init_device()
        bg_logger.info("[GPUExecuter Init] 2 Init device")

        # CRIUCHECK
        self.driver_worker.delayed_model_runner_init()
````

* worker.py
````python
    # CRIUCHECK
    def delayed_model_runner_init(self):
        self.model_runner.delayed_init_attn_backend()
````

* model_runner.py
````python
 # CRIUCHECK：
    # skip attention backend init in __init__, and add the func delayed_init_attn_backend
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        self.vision_language_config = vision_language_config

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.
        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), self.get_max_block_per_batch()),
            dtype=np.int32)
        # self.attn_backend = get_attn_backend(
        #     self.model_config.get_num_attention_heads(self.parallel_config),
        #     self.model_config.get_head_size(),
        #     self.model_config.get_num_kv_heads(self.parallel_config),
        #     self.model_config.get_sliding_window(),
        #     self.model_config.dtype,
        #     self.kv_cache_dtype,
        #     self.block_size,
        # )

        # Create processor for multi-modal data
        if self.vision_language_config is not None:
            self.multi_modal_input_processor = MULTIMODAL_REGISTRY \
                .create_input_processor(
                    self.model_config,
                    self.vision_language_config,
                )
        else:
            self.multi_modal_input_processor = None

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        # Set if the backend is flashinfer.
        self.flashinfer_workspace_buffer: torch.Tensor
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None

    def delayed_init_attn_backend(self):
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
        )

    def load_model(self) -> None:
        # CRIUCHECK：
        # disable CudaMemoryProfiler
        self.model = get_model(
            model_config=self.model_config,
            device_config=self.device_config,
            load_config=self.load_config,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config,
        )
        
````

* opt.py (used model)

````python

class OPTDecoder(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )

        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
                quant_config=quant_config
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
                quant_config=quant_config
            )
        else:
            self.project_in = None

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        # CRIUCheck:
        # 1. pre-import xformers
        # 2. dump here
        import xformers
        dump_process("/mnt/n0/sslm/ServerlessLLM/tools/CRIU/service/criu_service.socket", "/mnt/n0/models/vllm/opt6.7b_tmp/imgs/")
        # change the socket path and image dir to yours


        self.layers = nn.ModuleList([
            OPTDecoderLayer(config, cache_config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

        total_end = time.time()
````

### How to Run

#### Run With G-SLLM

0. Prepare Worker Node
````bash
cd sslm/ServerlessLLM/tools/CRIU/criu_rpc/
sudo su
nohup criu service --address /tmp/criu_service.socket >> criu_service.log 2>&1 &
````

````bash
conda activate /mnt/n0/.conda/envs/sllm-worker-0.6/
export CUDA_VISIBLE_DEVICES=1
sllm-store start  --mem-pool-size 32GB --chunk-size 0B
````

1. Create Dump images
创建镜像前需要保证VLLM和SLLM这些库都用“pip install ."的方式安装，不能使用”editable"方式。
根据运行的模型修改模型路径，确保模型路径下存在文件夹"imgs"
确保看到和Store一样的GPU
````bash
sudo su
conda activate /mnt/n0/.conda/envs/sllm-worker-0.6
cd criu_rpc
rm -rf /mnt/n0/models/vllm/opt6.7b_tmp/imgs/*
export CUDA_VISIBLE_DEVICES=1   
setsid python server.py --socket_addr=/tmp/criu_service.socket --model_path=/mnt/n0/models/vllm/opt6.7b_tmp < /dev/null &> /dev/null

````

检查镜像创建是否成功：
````bash
ls /mnt/n0/models/vllm/opt6.7b_tmp/imgs/
cat /mnt/n0/models/vllm/opt6.7b_tmp/imgs/criu.log
````
如果成功的话log输出应该是类似：
````
Error (criu/files-ext.c:92): Can't dump file 20 of that type [20666] (chr 195:3)
Error (criu/files-ext.c:92): Can't dump file 21 of that type [20666] (chr 195:4)
Error (criu/files-ext.c:92): Can't dump file 22 of that type [20666] (chr 195:4)
Warn  (compel/arch/x86/src/lib/infect.c:418): Will restore 3495918 with interrupted system call
````

如果转储失败，可以查看并杀死server进程:
````bash
sudo netstat -tulnp | grep ':50051'
sudo kill -9 <pid>
````


2. Test Restore
````bash
python test_2.py --socket_addr=/tmp/criu_service.socket --model_path=/mnt/n0/models/vllm/opt6.7b_tmp/imgs
````

3. Run With G-SLLM
运行SLLM集群：
````bash
conda activate sllm-0.6
export RAY_TMPDIR=/mnt/n0/ray_tmp
ray start --head --port=6379 --num-cpus=16 --num-gpus=0 --resources='{"control_node": 1}' --block
````
````bash
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/mnt/n0/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
````
````bash
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
sllm-store start  --mem-pool-size 32GB --chunk-size 0B
````

````bash
conda activate sllm-0.6
export RAY_TMPDIR=/mnt/n0/ray_tmp
sllm-serve start 
````

部署模型：
````bash
cd ServerlessLLM/serverless_scripts
sllm-cli deploy --config criu_opt6.7tmp.json
````
主要是指定模型以及backend为“criu”

调用模型：
````bash
sllm-cli generate opt6.7tmp_input.jso
````
它应该会创建CRIU backend，并调用推理


## Seen Issues

### CRIU
````
cat /mnt/n0/models/vllm/opt6.7b_tmp/imgs/criu.log 
Error (criu/proc_parse.c:116): handle_device_vma plugin failed: No such file or directory
Error (criu/proc_parse.c:650): Can't handle non-regular mapping on 3432517's map 7ff0ecbc2000
Error (criu/cr-dump.c:1570): Collect mappings (pid: 3432517) failed with -1
Error (criu/cr-dump.c:2111): Dumping FAILED.
````
这个问题主要是转储的时候存在无法保存的状态，比如说在VLLM初始化时已经包含进去GPU设备的一些信息，因此需要将对应的操作移动到转储之前。

### 恢复失败
````
(base) root@zwb:/mnt/n0/models/vllm/opt1.3b_tmp/imgs# cat criu.log 
496686: Error (criu/files.c:857): No file for fd 5 id 0x84
496686: Error (criu/files.c:857): No file for fd 6 id 0x85
496686: Error (criu/files.c:857): No file for fd 7 id 0x86
496686: Error (criu/files.c:857): No file for fd 8 id 0x87
496686: Error (criu/files.c:857): No file for fd 9 id 0x88
496686: Error (criu/files.c:857): No file for fd 10 id 0x89
496686: Error (criu/files.c:857): No file for fd 11 id 0x8a
496686: Error (criu/files.c:857): No file for fd 12 id 0x8b
496686: Error (criu/files.c:857): No file for fd 13 id 0x8c
496686: Error (criu/files.c:857): No file for fd 14 id 0x8d
496686: Error (criu/files.c:857): No file for fd 15 id 0x8e
496686: Error (criu/files.c:857): No file for fd 16 id 0x8f
496686: Error (criu/files.c:857): No file for fd 17 id 0x90
496686: Error (criu/files.c:857): No file for fd 18 id 0x91
496686: Error (criu/files.c:857): No file for fd 19 id 0x92
496686: Error (criu/files.c:857): No file for fd 20 id 0x93
496686: Error (criu/files.c:857): No file for fd 21 id 0x94
496686: Error (criu/files.c:857): No file for fd 22 id 0x95
````
创建转储，执行server.py之前没有设置CUDA_VISIBLE_DEVICES，导致使用的设备不一致



### RPC调用失败
````
(sllm-worker-0.6) sdu@zwb:/mnt/n0/sslm/ServerlessLLM/tools/CRIU/criu_rpc$ python test_2.py --socket_addr=/tmp/criu_service.socket --model_path=/mnt/n0/models/vllm/opt1.3b_tmp/imgs
----- This is a test script for checking if dumpped images can be restored and work
----- Requirements for this Restore scripts: 
 1. A CRIU Service run on a specific socket path with a sudo mode
 2. A python environment with modified VLLM installed (can not be 'editable' installed). Run model should have be configured to accpet Dump request. 
 3. If the model uses ReuseStore, make sure it is running.
连接成功，响应：Already Initialized time: 1751336043.6385107
Traceback (most recent call last):
  File "/mnt/n0/sslm/ServerlessLLM/tools/CRIU/criu_rpc/test_2.py", line 177, in <module>
    measure_resotre_time(args.socket_addr, args.model_path)
  File "/mnt/n0/sslm/ServerlessLLM/tools/CRIU/criu_rpc/test_2.py", line 158, in measure_resotre_time
    run_response = stub.Run(worker_rpc_pb2.RunRequest(task_id="task_123"))
  File "/mnt/n0/.conda/envs/sllm-worker-0.6/lib/python3.10/site-packages/grpc/_channel.py", line 946, in __call__
    return _end_unary_response_blocking(state, call, False, None)
  File "/mnt/n0/.conda/envs/sllm-worker-0.6/lib/python3.10/site-packages/grpc/_channel.py", line 849, in _end_unary_response_blocking
    raise _InactiveRpcError(state)
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.UNKNOWN
        details = "Exception calling application: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
"
        debug_error_string = "UNKNOWN:Error received from peer ipv4:127.0.0.1:50051 {grpc_message:"Exception calling application: CUDA error: an illegal memory access was encountered\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1.\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n", grpc_status:2, created_time:"2025-07-01T10:14:03.66306503+08:00"}"
>
````
