import grpc
import worker_rpc_pb2
import worker_rpc_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = worker_rpc_pb2_grpc.CRIUServiceStub(channel)

        # 初始化
        init_response = stub.Init(worker_rpc_pb2.InitRequest(config_path="/path/to/config"))
        print(f"初始化结果：{init_response.success} - {init_response.message}")

        # 运行任务
        run_response = stub.Run(worker_rpc_pb2.RunRequest(task_id="task_123"))
        print(f"任务结果：{run_response.result}")

        # 关闭服务
        shutdown_response = stub.Shutdown(worker_rpc_pb2.ShutdownRequest())
        print(f"关闭结果：{shutdown_response.success}")

if __name__ == '__main__':

    print("----- This is the script to connect Restored VLLM RPC server and shut it down")

    run()