#!/usr/bin/python3


import socket, os, sys
import criu_rpc as rpc
import argparse
import time

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
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        s.connect(socket_path)
        req = rpc.criu_req()
        req.type = rpc.RESTORE
        req.opts.images_dir_fd = os.open(images_dir, os.O_DIRECTORY)

        s.send(req.SerializeToString())
        resp = rpc.criu_resp()
        resp.ParseFromString(s.recv(1024))
        if resp.type!= rpc.RESTORE:
            print("Restore failed: unexpected response type")
            return False
        if not resp.success:
            print("Restore failed: CRIU execution error")
            return False
        return True
    except Exception as e:
        print("Restore failed: {}".format(str(e)))
        return False
    finally:
        s.close()
        if 'req' in locals():
            os.close(req.opts.images_dir_fd)


def dump_process(socket_path, images_dir):
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        s.connect(socket_path)

        req = rpc.criu_req()
        req.type = rpc.DUMP
        req.opts.images_dir_fd = os.open(images_dir, os.O_DIRECTORY)
        req.opts.exclude_paths.extend([
            '/dev/nvidia*',
        ])

        s.send(req.SerializeToString())

        resp = rpc.criu_resp()
        resp.ParseFromString(s.recv(1024))
        if resp.type != rpc.DUMP:
            print("Dump failed: unexpected response type")
            return False
        if not resp.success:
            print("Dump failed: CRIU execution error")
            os._exit(1)
            return False
        print("Dump success! Images stored in: {}".format(images_dir))
        if resp.dump.restored:
            print("CRIU restored process")
        return True
    except Exception as e:
        print("Dump failed: {}".format(str(e)))
        return False
    finally:
        s.close()
        if 'req' in locals():
            os.close(req.opts.images_dir_fd)

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
        print(f"CRIU dump failed: {e.stderr}")
        return False

def measure_resotre_time(socket_addr, model_path):
    
    start_time=time.time()

    restore_process(socket_addr, model_path)
    # print(f"restored: {time.time()}, takes{time.time()-start_time:.2f} s")
    max_trys=1000
    try_cnt=0
    while True:
        try:
            channel = grpc.insecure_channel('localhost:50051')
            stub = worker_rpc_pb2_grpc.CRIUServiceStub(channel)
            response = stub.Init(worker_rpc_pb2.InitRequest(config_path="test_config"))
            # print(f"Init Response: {response.success} {response.message} time: {time.time()}. Time spent {time.time()-start_time:.2f} s")
            break
        except grpc.RpcError as e:
            time.sleep(0.01)
            try_cnt+=1
            if try_cnt>max_trys:
                print("Timeout")
                return
    print(f"Init + Load Time: {time.time()-start_time:.2f} s")
    prefill_start_time=time.time()
    run_response = stub.Run(worker_rpc_pb2.RunRequest(task_id="task_123"))
    
    # print(f"Prefill Time: {time.time()-start_time:.2f} s")
    # print(f"    result: {run_response.result}")
    # parse the first token time from the result
    try:
        split_by_equal = run_response.result.split("=")
        if len(split_by_equal) < 19: 
            raise ValueError("Insufficient elements after splitting by equal sign")
        
        split_by_comma = split_by_equal[18].split(",")
        if len(split_by_comma) < 1:
            raise ValueError("The element at index 18 is empty after splitting by comma")
        
        first_token_time = split_by_comma[0]
    except Exception as e:
        print(f"Error parsing first_token_time: {str(e)}")
        print(f"run_response.result: {run_response.result}")
        first_token_time = None
    
    print(f"Prefill Time: {float(first_token_time)-prefill_start_time:.2f} s") if first_token_time else print("TTFT: N/A")
    stub.Shutdown(worker_rpc_pb2.ShutdownRequest())

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
