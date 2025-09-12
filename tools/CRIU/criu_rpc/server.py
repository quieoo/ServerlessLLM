import grpc
import time
from concurrent import futures
import worker_rpc_pb2
import worker_rpc_pb2_grpc
import argparse
import time
import os


class CRIUServicer(worker_rpc_pb2_grpc.CRIUServiceServicer):
    def __init__(self, model_path):
        self.is_initialized = False
        self.current_task = None
        self.shutdown=False
        import vllm
        self.sampling_params = vllm.SamplingParams(temperature=0.7, top_p=0.9, max_tokens=24)
        self.vllm_engine=vllm.LLM(
            model=model_path,
            enforce_eager=True,
            load_format="serverless_llm",
            dtype="float16",
            enable_prefix_caching=True,
            served_model_name=[model_path + "--" + "127.0.0.1:8073"],
        )
        # Debug use
        # test_output=self.vllm_engine.generate(["hello"], self.sampling_params)
        # print(test_output)
        # test_output=self.vllm_engine.generate(["how are you"], self.sampling_params)
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

        prompts=[request.task_id]
        self.current_task = request.task_id
        output=self.vllm_engine.generate(prompts, self.sampling_params)
        # print(output)
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
    parser.add_argument("--dump", type=int, default=1, help="Dump model checkpoint")
    args = parser.parse_args()

    # test:
    # get_vllm_need_modules(args.model_path)
    # preload_modules("vllm_need_libs.txt")

    # enable CRIU dump
    if args.dump==1:
        print(f"CRIU dump enabled, socket path: {args.socket_addr}, model path: {args.model_path}")
        os.environ["CRIUDUMP_SOCKET"]=args.socket_addr
        os.environ["CRIUDUMP_MODEL"]=args.model_path
        # force the VLLM to init models in a pure CPU environment, avoid the GPU device map preventing CRIU to dump the process
        os.environ["USE_GPU"]=os.environ.get("CUDA_VISIBLE_DEVICES", "2")
        print(f"USE_GPU: {os.environ['USE_GPU']}")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    else:
        print(f"CRIU dump disabled")
        for key in os.environ.keys():
            if "CRIUDUMP" in key:
                del os.environ[key]
        


    serve(args.model_path)
