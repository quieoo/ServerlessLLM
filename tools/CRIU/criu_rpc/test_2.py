#!/usr/bin/python3


import socket, os, sys
import criu_rpc as rpc
import argparse
import time  # 新增：用于计数器延时

import grpc
import worker_rpc_pb2
import worker_rpc_pb2_grpc


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
            return False
        if not resp.success:
            print("恢复失败：CRIU 执行错误")
            return False
        # print("恢复成功！")
        return True
    except Exception as e:
        print("恢复异常：{}".format(str(e)))
        return False
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

# 替换原有的dump_process调用为：
import subprocess
def dump_process_bin(socket_path, images_dir):
    try:

        subprocess.run(
            ["./criu_dump", socket_path, images_dir],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"CRIU转储失败: {e.stderr}")
        return False

def measure_resotre_time(socket_addr, model_path):
    
    start_time=time.time()

    print(f"start: {start_time}")
    restore_process(socket_addr, model_path)
    print(f"restored: {time.time()}, takes{time.time()-start_time:.2f} s")
    max_trys=1000
    try_cnt=0
    

    # 尝试连接RPC服务端
    while True:
        try:
            channel = grpc.insecure_channel('localhost:50051')
            stub = worker_rpc_pb2_grpc.CRIUServiceStub(channel)
            # 简单调用，确保连接正常
            response = stub.Init(worker_rpc_pb2.InitRequest(config_path="test_config"))
            print(f"连接成功，响应：{response.message} time: {time.time()}. Time spent {time.time()-start_time:.2f} s")
            break
        except grpc.RpcError as e:
            # print(f"连接失败：{e}")
            # 间隔10ms
            time.sleep(0.01)
            try_cnt+=1
            if try_cnt>max_trys:
                print("连接超时")
                return
    print(f"connected: {time.time()}, takes{time.time()-start_time:.2f} s")
    # 调用Shutdown
    run_response = stub.Run(worker_rpc_pb2.RunRequest(task_id="task_123"))
    print(f"requested: {time.time()}, takes{time.time()-start_time:.2f} s")
    print(f"    result: {run_response.result}")
    stub.Shutdown(worker_rpc_pb2.ShutdownRequest())

# 主程序入口
if __name__ == "__main__":
    print("----- This is a test script for checking if dumpped images can be restored and work")
    print("----- Requirements for this Restore scripts: ")
    print(" 1. A CRIU Service run on a specific socket path with a sudo mode")
    print(" 2. A python environment with modified VLLM installed (can not be 'editable' installed). Run model should have be configured to accpet Dump request. ")
    print(" 3. If the model uses ReuseStore, make sure it is running.")


    parser = argparse.ArgumentParser()
    parser.add_argument("--socket_addr", type=str, default="/tmp/criu.sock", help="CRIU socket path")
    parser.add_argument("--model_path", type=str, default="/mnt/n0/models/vllm/opt6.7b_tmp", help="VLLM model path")
    args = parser.parse_args()

    measure_resotre_time(args.socket_addr, args.model_path)
