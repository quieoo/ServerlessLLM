#!/usr/bin/python3


import socket, os, sys
import sllm_store.criu_rpc as rpc
import argparse
import time  # 新增：用于计数器延时

import grpc

# import worker_rpc_pb2
# import worker_rpc_pb2_grpc


counter=0

def complex_init():
    time.sleep(10)

def rest_work():
    for i in range(10):
        print("rest_work: ",i)

def restore_process(socket_path, images_dir):
    """
    通过 RPC 触发 CRIU 恢复操作
    :param socket_path: CRIU 服务端套接字路径
    :param images_dir: 转储镜像存储目录
    :return: 恢复是否成功（True/False）
    """
    try:
        # 连接服务端
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        s.connect(socket_path)
        # 构造 RESTORE 请求
        req = rpc.criu_req()
        req.type = rpc.RESTORE
        req.opts.images_dir_fd = os.open(images_dir, os.O_DIRECTORY)
        # req.opts.shell_job=True
        # req.opts.log_level = 4
        # req.opts.network_lock = rpc.SKIP
        # 发送请求
        s.send(req.SerializeToString())
        # 接收响应
        resp = rpc.criu_resp()
        resp.ParseFromString(s.recv(1024))
        # 验证响应
        if resp.type!= rpc.RESTORE:
            print("恢复失败：意外的响应类型")
            return -1
        if not resp.success:
            print("恢复失败：CRIU 执行错误")
            return -1
        # print("恢复成功！")
        return 0
    except Exception as e:
        print("恢复异常：{}".format(str(e)))
        return -1
    finally:
        s.close()
        if'req' in locals():  # 增加存在性检查
            os.close(req.opts.images_dir_fd)


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
        # req.opts.shell_job=True
        # req.opts.log_level = 4
        req.opts.images_dir_fd = os.open(images_dir, os.O_DIRECTORY)
        # req.opts.network_lock = rpc.SKIP
        req.opts.exclude_paths.extend([
            '/dev/nvidia*',  # 排除所有NVIDIA设备
        ])

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
            os._exit(1)
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

# # 替换原有的dump_process调用为：
# import subprocess
# def dump_process_bin(socket_path, images_dir):
#     try:

#         subprocess.run(
#             ["./criu_dump", socket_path, images_dir],
#             check=True,
#             capture_output=True,
#             text=True
#         )
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"CRIU转储失败: {e.stderr}")
#         return False

# def measure_resotre_time():
#     start_time=time.time()
#     restore_process("/mnt/n0/sslm/ServerlessLLM/tools/CRIU/service/criu_service.socket", "/mnt/n0/models/vllm/opt6.7b_tmp/imgs/")
#     # restore_process("/mnt/n0/sslm/ServerlessLLM/tools/CRIU/service/criu_service.socket", "/mnt/ramdisk/imgs")
    

#     # 尝试连接RPC服务端
#     while True:
#         try:
#             channel = grpc.insecure_channel('localhost:50051')
#             stub = worker_rpc_pb2_grpc.CRIUServiceStub(channel)
#             # 简单调用，确保连接正常
#             response = stub.Init(worker_rpc_pb2.InitRequest(config_path="test_config"))
#             # print(f"连接成功，响应：{response.message} time: {time.time()}")
#             break
#         except grpc.RpcError as e:
#             # print(f"连接失败：{e}")
#             # 间隔10ms
#             time.sleep(0.01)

#     # 调用Shutdown
#     run_response = stub.Run(worker_rpc_pb2.RunRequest(task_id="task_123"))
#     print(f"开始时间：{start_time}")
#     print(f"任务结果：{run_response.result}")
#     stub.Shutdown(worker_rpc_pb2.ShutdownRequest())

# # 主程序入口
# if __name__ == "__main__":

#     measure_resotre_time()

#     # 参数解析（保持原有逻辑）
#     # parser = argparse.ArgumentParser(description="Test dump/restore using CRIU RPC")
#     # parser.add_argument('socket', type=str, help="CRIU service socket")
#     # parser.add_argument('dir', type=str, help="Directory where CRIU images should be placed")
#     # args = vars(parser.parse_args())
    
#     # 运行示例业务程序（触发转储）
#     # complex_init()
#     # dump_process(args['socket'], args['dir'])
#     # rest_work()
#     # print(f"Current PID: {os.getpid()}")
#     # from vllm import SamplingParams, LLM
#     # print("Checking Dump with RPC")
#     # # dump_process(args['socket'], args['dir'])
#     # # print("Imported libs, init LLM in 5 seconds...")
#     # # time.sleep(5)   # dump during sleep time   
#     # sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
#     # llm = LLM(
#     #     model="/mnt/n0/models/vllm/opt6.7b_tmp",  # 替换为你的模型路径
#     #     # tensor_parallel_size=2,        # 使用 2 个 GPU
#     #     # trust_remote_code=True,        # 如果模型需要自定义代码
#     #     enforce_eager=True,
#     #     load_format="serverless_llm",
#     # )
#     # output=llm.generate(["Explain AI in 100 words."], sampling_params)
#     # print(output)


#     # import time
#     # # 在导入vllm前添加清理逻辑
#     # import torch
#     # from vllm import (
#     #     AsyncEngineArgs,
#     #     AsyncLLMEngine,
#     #     EmbeddingRequestOutput,
#     #     PoolingParams,
#     #     PromptStrictInputs,
#     #     RequestOutput,
#     #     SamplingParams,
#     # )
#     # # dump_process_bin(args['socket'], args['dir'])
#     # # dump_process(args['socket'], args['dir'])

#     # for i in range(10):
#     #     print("test: ",i)
#     #     time.sleep(0.1)
    