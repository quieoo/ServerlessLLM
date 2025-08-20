from trace import Trace, TraceReplay, report_group_stats
from scipy.stats import entropy
import numpy as np
import json
import argparse
import os

#需要： conda activate sllm-trace
# 将Func映射到模型
# 相关工作[Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving]指出在真实的Serverless LLM平台中模型分布：
# Small Model 1B-3B: 43
# Medium Model 4B-8B: 8
# Large Model 9B-30B: 3
# Extra Large Model 31B-70B: 4

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark trace generation tool')
    parser.add_argument('--action', type=str, default="gen_trace", help='Action to perform (gen_trace or trace_locality)')
    parser.add_argument('--target_cv', type=float, default=0.25, 
                       help='Target coefficient of variation (default: 0.25)')
    parser.add_argument('--target_req_file_path', type=str, 
                       default="/mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/4090_cv0.25_large.txt",
                       help='Output file path for generated requests')
    parser.add_argument('--sllm_model_config_file_path', type=str,
                       default="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-large.json",
                       help='Path to SLLM model configuration file')
    parser.add_argument('--trace_name', type=str, default="azure_v2",
                       help='Trace name (default: azure_v2)')
    parser.add_argument('--trace_dir', type=str, 
                       default="/mnt/n0/datasets/azura_v2.txt",
                       help='Trace directory path')
    return parser.parse_args()

# 解析命令行参数
args = parse_arguments()

rate_cv_map={
    0.125 : 1e-3,
    0.25 : 1e-2 * 5,
    0.5 : 1e-4 * 5,
    1:1e-5 * 5,
    2:1e-9 * 1,
    4 : 1e-40,
}

def trace_locality():
    # 使用全局的target_cv参数，但可以在这里覆盖
    local_target_cv = args.target_cv
    sllm_model_config_file_path = args.sllm_model_config_file_path

    # 读取json配置文件获取模型列表
    with open(sllm_model_config_file_path, "r") as f:
        model_config = json.load(f)
    model_dirs = model_config["model_dirs"]
    weighted_mapping = model_config["model_affinity"]
    # 将模型名转换为序列号
    model_ids=[]
    for i in range(len(model_dirs)):
        model_ids.append(i)
    print(f"model_ids: {model_ids}")
    print(f"weighted_mapping: {weighted_mapping}")
    days=7
    hour=24
    trace = Trace(args.trace_name, args.trace_dir)
    model_requests = []
    for day in range(days):
        for h in range(hour):
            start_time = str(day) + "." + str(h) + ".0"
            if h < 23:
                end_time = str(day) + "." + str(h+1) + ".0"
            else:
                end_time = str(day) + "." + "23.60"

            replays = trace.replay(
                                models=model_ids,
                                model_mapping_strategy="weighted",
                                # model_mapping_strategy="round_robin",
                                mapping_params=weighted_mapping,
                                start_time=start_time,
                                end_time=end_time,
                                arrival_distribution="gamma",
                                interval_seconds=60,
                                cv_scale_factor=local_target_cv,
                                time_scale_factor=1.0,
                                replication_factor=1,
                                seed=0)
            # 收集replays中的模型请求
            # 按照时间顺序排列
            for m in replays:
                for req in replays[m].arrivals:
                    model_requests.append((float(req), m))
    
    # 对模型请求进行排序
    model_requests.sort(key=lambda x: x[0])

    # 丢弃时间戳
    model_requests = [x[1] for x in model_requests]

    from id_separate_analyzer import analyze_id_intervals_separately,print_analysis_results
    
    analysis_results = analyze_id_intervals_separately(model_requests)
    print_analysis_results(analysis_results, args.sllm_model_config_file_path, True)

    # # 统计每个模型的访问序号
    # model_request_index = {}
    # for i, m in enumerate(model_requests):
    #     if m not in model_request_index:
    #         model_request_index[m] = []
    #     model_request_index[m].append(i)
    # # print(model_request_index)

    # # 统计每个模型的访问间隔
    # model_request_interval = {}
    # for m in model_request_index:
    #     model_request_interval[m] = []
    #     for i in range(len(model_request_index[m])-1):
    #         model_request_interval[m].append(model_request_index[m][i+1] - model_request_index[m][i]-1)
    # # print(model_request_interval)


    # # 统计每个模型的平均访问间隔
    # for m in model_request_interval:
    #     print(f"{model_dirs[m]}: {np.mean(model_request_interval[m])}")
    
    # # 统计每个模型的各百分位（0-100，间隔为1）访问间隔值，
    # for m in model_request_interval:
    #     # 每个数据输出使用换行隔开
    #     print(f"{model_dirs[m]}:")
    #     for i in range(0, 100, 1):
    #         print(f"{np.percentile(model_request_interval[m], i)}")

        # print(f"{model_dirs[m]}: {np.percentile(model_request_interval[m], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])}")

def sequence_gen():
    trace_name = args.trace_name
    trace_dir = args.trace_dir
    target_cv = args.target_cv
    target_req_file_path = args.target_req_file_path
    sllm_model_config_file_path = args.sllm_model_config_file_path

    # 验证文件路径是否存在
    if not os.path.exists(sllm_model_config_file_path):
        raise FileNotFoundError(f"Model config file not found: {sllm_model_config_file_path}")

    # 确保输出目录存在
    output_dir = os.path.dirname(target_req_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取json配置文件获取模型列表
    with open(sllm_model_config_file_path, "r") as f:
        model_config = json.load(f)
    model_dirs = model_config["model_dirs"]
    weighted_mapping = model_config["model_affinity"]
    # 将模型名转换为序列号
    model_ids=[]
    for i in range(len(model_dirs)):
        model_ids.append(i)
    print(f"model_ids: {model_ids}")
    print(f"weighted_mapping: {weighted_mapping}")
    days=1
    hour=24
    trace = Trace(trace_name, trace_dir)
    model_requests = []
    for day in range(days):
        for h in range(hour):
            start_time = str(day) + "." + str(h) + ".0"
            if h < 23:
                end_time = str(day) + "." + str(h+1) + ".0"
            else:
                end_time = str(day) + "." + "23.60"

            replays = trace.replay(
                                models=model_ids,
                                model_mapping_strategy="weighted",
                                # model_mapping_strategy="round_robin",
                                mapping_params=weighted_mapping,
                                start_time=start_time,
                                end_time=end_time,
                                arrival_distribution="gamma",
                                # interval_seconds=3600,
                                interval_seconds=60,
                                rate_scale_factor=rate_cv_map[target_cv],
                                # rate_scale_factor=1,
                                cv_scale_factor=target_cv,
                                time_scale_factor=1.0,
                                replication_factor=1,
                                seed=0)
            # 收集replays中的模型请求
            # 按照时间顺序排列
            for m in replays:
                for req in replays[m].arrivals:
                    model_requests.append((float(req), m))
    
    # 对模型请求进行排序
    model_requests.sort(key=lambda x: x[0])

    # 丢弃时间戳
    model_requests = [x[1] for x in model_requests]
    if target_cv < 1.0:
        # 处理模型请求，将连续相同的请求合并
        processed_requests = []
        if target_cv >= 0.5:
            # 如果存在连续的相同请求，去除其中的一半
            for i in range(len(model_requests)):
                if i == 0 or model_requests[i] != model_requests[i-1]:
                    processed_requests.append(model_requests[i])
                else:
                    if i % 2 == 0:
                        processed_requests.append(model_requests[i])

        else:
            for i in range(len(model_requests)):
                if i == 0 or model_requests[i] != model_requests[i-1]:
                    processed_requests.append(model_requests[i])
        model_requests=processed_requests
    
    # 统计请求数量
    model_requests_cnt = {}
    total_cnt=0
    for m in model_requests:
        model_name=model_dirs[m]
        if model_name not in model_requests_cnt:
            model_requests_cnt[model_name] = 0
        model_requests_cnt[model_name] += 1
        total_cnt += 1
    # for req in model_requests:
    #     print(req)
    print("total cnt:", total_cnt)
    for model, cnt in model_requests_cnt.items():
        print(f"{model}: {cnt/total_cnt}")

    # 将请求写入文件
    with open(target_req_file_path, "w") as f:
        for req in model_requests:
            f.write(str(req) + "\n")

if __name__ == "__main__":
    print(f"使用参数:")
    print(f"  action: {args.action}")
    print(f"  target_cv: {args.target_cv}")
    print(f"  target_req_file_path: {args.target_req_file_path}")
    print(f"  sllm_model_config_file_path: {args.sllm_model_config_file_path}")
    print(f"  trace_name: {args.trace_name}")
    print(f"  trace_dir: {args.trace_dir}")
    print()
    
    if args.action == "gen_trace":
        sequence_gen()
    elif args.action == "trace_locality":
        trace_locality()
    else:
        raise ValueError(f"Unknown action: {args.action}")