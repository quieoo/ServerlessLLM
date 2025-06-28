## Benchmark_v2

这个脚本被用于测试不同input length和batch_size下的模型性能

参数含义：
* file_path: 数据集路径
* model: 模型名称
* batch_size: 批大小
* request_length: 请求长度
* max_tokens: 最大token数
* qps: 查询速率
* n: 处理的prompt数量

* 单batch，input_length

发送一条查询请求，单批次，输入长度为1000，输出长度为100
````bash
python benchmark_v2.py /mnt/n0/datasets/sharegpt_V3_format.jsonl opt6.7b_tmp 1 1000 100 0 1
````
发送10条查询请求，batch_size为10，不指定输入长度（使用数据集本身的输入），输出长度为100, 
````bash
python benchmark_v2.py /mnt/n0/datasets/sharegpt_V3_format.jsonl opt6.7b_tmp 10 0 100 0 1
````

注意：增大n似乎会导致RestoreStore的错误，猜测可能与未完全释放的GPU Memory Hangle有关。但是保持n=1，可以保证正确执行。

````
(VllmBackend pid=1039070) ERROR 06-18 20:35:52.564 async_llm_engine.py:55] RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx( handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16F, lda, b, CUDA_R_16F, ldb, &fbeta, c, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
`````


