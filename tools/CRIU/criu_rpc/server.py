import grpc
import time
from concurrent import futures
import worker_rpc_pb2
import worker_rpc_pb2_grpc
import argparse
import sys
import importlib
import ast
from typing import List
import time
import os

def get_vllm_need_modules(model_path):
    before=set(sys.modules.keys())
    import vllm
    sampling_params = vllm.SamplingParams(temperature=0.7, top_p=0.9)
    vllm_engine=vllm.LLM(
        # model="/mnt/n0/models/vllm/opt6.7b_tmp",
        # model="/mnt/n0/models/opt6.7",
        model=model_path,
        enforce_eager=True,
        load_format="serverless_llm",
    )
    after=set(sys.modules.keys())
    sorted_modules = sorted(after-before)
    print(f"vllm need modules: {sorted_modules}")

def parse_modules_from_file(path: str) -> List[str]:
    """从包含 Python 列表字面量的文件中解析模块名列表"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"无法解析文件为 Python 列表：{e}")
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("文件内容必须是字符串列表，例如：['PIL', 'PIL.ExifTags', ...]")
    return data

def to_top_level(mod: str) -> str:
    return mod.split(".", 1)[0]

def import_one(mod: str):
    t0 = time.perf_counter()
    try:
        importlib.import_module(mod)
        dt = (time.perf_counter() - t0) * 1000
        # print(f"[OK]   {mod:<40} {dt:8.2f} ms")
        return True, dt, None
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        # print(f"[FAIL] {mod:<40} {dt:8.2f} ms  ({e})")
        return False, dt, e

def preload_modules(path):
    modules = parse_modules_from_file(path)
    modules = sorted({to_top_level(m) for m in modules})
    start_time=time.perf_counter()
    for mod in modules:
        import_one(mod)
    end_time=time.perf_counter()
    print(f"preload modules time: {end_time-start_time}")


class CRIUServicer(worker_rpc_pb2_grpc.CRIUServiceServicer):
    def __init__(self, model_path):
        self.is_initialized = False
        self.current_task = None
        self.shutdown=False
        import vllm
        self.sampling_params = vllm.SamplingParams(temperature=0.7, top_p=0.9)
        self.vllm_engine=vllm.LLM(
            # model="/mnt/n0/models/vllm/opt6.7b_tmp",
            # model="/mnt/n0/models/opt6.7",
            model=model_path,
            enforce_eager=True,
            load_format="serverless_llm",
        )
        # Debug use
        # test_output=self.vllm_engine.generate(["Introduce yourself"], self.sampling_params)
        # print(test_output)


    def Init(self, request, context):
        # 模拟初始化逻辑（加载配置文件）
        if self.is_initialized:
            return worker_rpc_pb2.InitResponse(success=False, message="Already Initialized")
        
        print(f"Init config: {request.config_path}")
        self.is_initialized = True
        return worker_rpc_pb2.InitResponse(success=True, message="Initialize successfully")

    def Run(self, request, context):
        if not self.is_initialized:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("please call Init first")
            return worker_rpc_pb2.RunResponse()

        self.current_task = request.task_id
        output=self.vllm_engine.generate([request.task_id], self.sampling_params)
        # output="succ"
        # print(f"运行任务：{request.task_id}")
        return worker_rpc_pb2.RunResponse(result=f"task: {request.task_id}, finished: {output}")

    def Shutdown(self, request, context):
        if self.current_task:
            print(f"Terminating")
            self.current_task = None
        self.is_initialized = False
        self.shutdown=True
        return worker_rpc_pb2.ShutdownResponse(success=True)

def serve(model_path):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = CRIUServicer(model_path)  # 创建servicer实例
    worker_rpc_pb2_grpc.add_CRIUServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("RPC started, monitor on port 50051...")
    try:
        while not servicer.shutdown:  # 修改为检查servicer的shutdown标志
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop(0)
    finally:
        server.stop(0)

if __name__ == '__main__':
    print("----- Requirements for this Dumpping scripts: ")
    print(" 1. A CRIU Service run on a specific socket path with a sudo mode")
    print(" 2. A python environment with modified VLLM installed (can not be 'editable' installed). Run model should have be configured to accpet Dump request. ")

    print("----- Run this script will create a VLLM wrapped RPC server and dump the CRIU checkpoint on specific model path 'imgs' dir")

    # 读取命令行参数
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket_addr", type=str, default="/tmp/criu.sock", help="CRIU socket path")
    parser.add_argument("--model_path", type=str, default="/mnt/n0/models/vllm/opt6.7b_tmp", help="VLLM model path")
    args = parser.parse_args()

    # test:
    # get_vllm_need_modules(args.model_path)
    # preload_modules("vllm_need_libs.txt")

    # enable CRIU dump
    os.environ["CRIUDUMP_SOCKET"]=args.socket_addr
    os.environ["CRIUDUMP_MODEL"]=args.model_path
    # force the VLLM to init models in a pure CPU environment, avoid the GPU device map preventing CRIU to dump the process
    os.environ["USE_GPU"]=os.environ.get("CUDA_VISIBLE_DEVICES", 2)
    print(f"USE_GPU: {os.environ['USE_GPU']}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


    serve(args.model_path)
