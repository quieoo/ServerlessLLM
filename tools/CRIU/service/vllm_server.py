from vllm.entrypoints.api_server import main
import argparse

def start_vllm_server():
    # 构建命令行参数
    args = argparse.Namespace(
        model="/mnt/n0/models/vllm/opt6.7b_tmp",  # 模型路径或 HuggingFace ID
        host="0.0.0.0",              # 监听地址
        port=8000,                   # 监听端口
        served_model_name="opt6.7b_tmp"  # 服务别名
    )
    main(args)  # 启动服务

if __name__ == "__main__":
    start_vllm_server()
