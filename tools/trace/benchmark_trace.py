from trace import Trace, TraceReplay, report_group_stats
from scipy.stats import entropy
import numpy as np
import json
import argparse
import os
from datetime import datetime
from collections import defaultdict


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
    parser.add_argument('--mapping', type=str, default="weighted",
                       help='Mapping strategy (default: weighted)')
    parser.add_argument('--days', type=int, default=0,
                       help='Number of days to generate trace (default: 7)')
    parser.add_argument('--hour', type=int, default=24,
                       help='Number of hour to generate trace (default: 24)')
    parser.add_argument('--keep_alive', type=float, default=0.0,
                       help='Keep alive time (default: 0.0)')
    parser.add_argument('--worker_num', type=int, default=1,
                       help='Number of worker (default: 1)')
    parser.add_argument('--rate_scale_factor', type=float, default=1.0,
                       help='Rate scale factor (default: 1.0)')
    return parser.parse_args()

# 解析命令行参数
args = parse_arguments()

rate_cv_map={
    0.125 : 1e-3,
    0.25 : 1e-2 * 5,
    0.5 : 1e-3 * 9,
    1:1e-4 * 9,
    2:1e-9 * 9,
    4 : 1e-40,
}

def calc_avg_rps(nested_list):
    """
    nested_list: [[timestamp, model_id], ...]
                 timestamp 可以是 datetime 或 Unix 秒/毫秒
    返回 dict：overall_rps / total_seconds / total_requests / model_rps
    """
    if not nested_list:
        return {'overall_rps': 0, 'total_seconds': 0,
                'total_requests': 0, 'model_rps': {}}

    # 统一转成 Unix 秒（float）
    first_ts = nested_list[0][0]
    if isinstance(first_ts, datetime):
        to_sec = lambda ts: ts.timestamp()
    elif isinstance(first_ts, (int, float)) and first_ts > 1e10:
        # 毫秒戳
        to_sec = lambda ts: float(ts) / 1000.0
    else:
        # 已是秒
        to_sec = float

    timestamps = [to_sec(req[0]) for req in nested_list]
    timestamps.sort()

    total_requests = len(nested_list)
    start, end = timestamps[0], timestamps[-1]
    total_seconds = max(end - start, 1.0)          # 防除 0

    overall_rps = total_requests / total_seconds

    # 分模型
    model_cnt = defaultdict(int)
    for _, model_id in nested_list:
        model_cnt[model_id] += 1
    model_rps = {mid: cnt / total_seconds for mid, cnt in model_cnt.items()}

    return {
        'overall_rps': overall_rps,
        'total_seconds': total_seconds,
        'total_requests': total_requests,
        'model_rps': model_rps
    }


def trace_locality():
    trace_name = args.trace_name
    trace_dir = args.trace_dir
    target_cv = args.target_cv
    sllm_model_config_file_path = args.sllm_model_config_file_path

    # 验证文件路径是否存在
    if not os.path.exists(sllm_model_config_file_path):
        raise FileNotFoundError(f"Model config file not found: {sllm_model_config_file_path}")

    
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
    days=args.days
    hour=24
    trace = Trace(trace_name, trace_dir)
    model_requests = []
    start_time = str(0) + "." + str(0) + ".0"
    end_time = str(days) + "." + str(hour-1) + ".0"
    replays = trace.replay(
                                models=model_ids,
                                # model_mapping_strategy="weighted",
                                # model_mapping_strategy="round_robin",
                                model_mapping_strategy=args.mapping,
                                mapping_params=weighted_mapping,
                                start_time=start_time,
                                end_time=end_time,
                                arrival_distribution="gamma",
                                # interval_seconds=3600,
                                interval_seconds=60,
                                # rate_scale_factor=rate_cv_map[target_cv],
                                rate_scale_factor=1,
                                cv_scale_factor=target_cv,
                                time_scale_factor=1.0,
                                replication_factor=1,
                                seed=0)
    for m in replays:
        for req in replays[m].arrivals:
            model_requests.append((float(req), m))

    model_requests.sort(key=lambda x: x[0])
    
    model_requests = [x[1] for x in model_requests]

    # 打印前10个请求
    print(model_requests[:10])

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
    days=args.days
    hour=args.hour
    trace = Trace(trace_name, trace_dir)
    model_requests = []
    start_time = str(0) + "." + str(0) + ".0"
    end_time = str(days) + "." + str(hour-1) + ".0"
    replays = trace.replay(
                                models=model_ids,
                                # model_mapping_strategy="weighted",
                                # model_mapping_strategy="round_robin",
                                model_mapping_strategy=args.mapping,
                                mapping_params=weighted_mapping,
                                start_time=start_time,
                                end_time=end_time,
                                arrival_distribution="gamma",
                                # interval_seconds=3600,
                                interval_seconds=60,
                                # rate_scale_factor=rate_cv_map[target_cv],
                                rate_scale_factor=args.rate_scale_factor,
                                cv_scale_factor=target_cv,
                                time_scale_factor=1.0,
                                replication_factor=1,
                                seed=0)
    for m in replays:
        for req in replays[m].arrivals:
            model_requests.append((float(req), m))
    
    # 对模型请求进行排序
    print(f"model requests len: {len(model_requests)}")
    model_requests.sort(key=lambda x: x[0])

    # 统计请求的RPS
    rps_stats = calc_avg_rps(model_requests)
    print(f"RPS Statistics:")
    print(f"Overall RPS: {rps_stats['overall_rps']:.4f}")
    print(f"Total Seconds: {rps_stats['total_seconds']:.2f}")
    print(f"Total Requests: {rps_stats['total_requests']}")
    print("\nPer Model RPS:")
    print(rps_stats['model_rps'])


    # 清除模型重用的情况
    # 如果两个相邻的请求是相同的模型，且时间间隔小于keep_alive，将第二个请求删除
    keep_alive=args.keep_alive
    container_count=args.worker_num

    # if keep_alive > 0.0:
    #     new_model_requests=[]
    #     for i in range(len(model_requests)):
    #         if i == 0:
    #             new_model_requests.append(model_requests[i])
    #         else:
    #             if model_requests[i][1] != model_requests[i-1][1]:
    #                 new_model_requests.append(model_requests[i])
    #             elif model_requests[i][0] - model_requests[i-1][0] > keep_alive:
    #                 new_model_requests.append(model_requests[i])
    #             # if model_requests[i][1] == model_requests[i-1][1]:
    #             #     print(f"model request interval: {model_requests[i][0] - model_requests[i-1][0]}")
    #     print(f"keep_alive value: {keep_alive}, remove {len(model_requests)-len(new_model_requests)} requests")
    #     model_requests=new_model_requests

    
    if keep_alive > 0.0:
        new_model_requests = []
        
        # 初始化容器状态：每个容器记录最后加载的模型和时间戳
        # [{'model': None, 'last_time': -inf}, ...]
        containers = [{'model': None, 'last_time': float('-inf')} for _ in range(container_count)]
        
        # 统一时间戳转换为秒（兼容datetime、毫秒戳、秒戳）
        first_ts = model_requests[0][0]
        if isinstance(first_ts, datetime):
            to_sec = lambda ts: ts.timestamp()
        elif isinstance(first_ts, (int, float)) and first_ts > 1e10:
            to_sec = lambda ts: float(ts) / 1000.0
        else:
            to_sec = float
        
        for i, req in enumerate(model_requests):
            timestamp = to_sec(req[0])
            model_id = req[1]
            
            # 全局复用检查：遍历所有容器寻找可复用者
            reusable_found = False
            for container in containers:
                if container['model'] == model_id and (timestamp - container['last_time']) <= keep_alive:
                    reusable_found = True
                    break  # 找到即可复用，无需继续检查
            
            if reusable_found:
                # 情况3：可复用 → 删除请求（不追加到new_model_requests）
                continue
            else:
                # 情况1或2：不可复用 → 保留请求
                new_model_requests.append(req)
                
                # 将请求路由到LRU容器（最近使用时间最早的容器）
                # 若多个容器时间相同，默认选择第一个（等价于随机）
                chosen_container = min(containers, key=lambda c: c['last_time'])
                chosen_container['model'] = model_id
                chosen_container['last_time'] = timestamp
        
        print(f"keep_alive={keep_alive}s, containers={container_count}, "
            f"removed {len(model_requests) - len(new_model_requests)} requests")
        model_requests = new_model_requests



    # 丢弃时间戳
    model_requests = [x[1] for x in model_requests]
    # if target_cv < 1.0:
    #     # 处理模型请求，将连续相同的请求合并
    #     processed_requests = []
    #     if target_cv >= 0.5:
    #         # 如果存在连续的相同请求，去除其中的一半
    #         for i in range(len(model_requests)):
    #             if i == 0 or model_requests[i] != model_requests[i-1]:
    #                 processed_requests.append(model_requests[i])
    #             else:
    #                 if i % 2 == 0:
    #                     processed_requests.append(model_requests[i])

    #     else:
    #         for i in range(len(model_requests)):
    #             if i == 0 or model_requests[i] != model_requests[i-1]:
    #                 processed_requests.append(model_requests[i])
    #     model_requests=processed_requests
    
    # 统计请求数量
    req_cnt=[0 for i in range(len(model_dirs))]
    total=0
    for m in model_requests:
        total+=1
        req_cnt[m]=req_cnt[m]+1
    print(f"total req cnt: {total}")
    for i, cnt in enumerate(req_cnt):
        print(f"{i}-{model_dirs[i]} req cnt: {cnt} ({cnt/total})")


    # model_requests_cnt = {}
    # total_cnt=0
    # for m in model_requests:
    #     model_name=model_dirs[m]
    #     if model_name not in model_requests_cnt:
    #         model_requests_cnt[model_name] = 0
    #     model_requests_cnt[model_name] += 1
    #     total_cnt += 1
    # # for req in model_requests:
    # #     print(req)
    # print("total cnt:", total_cnt)
    # for model, cnt in model_requests_cnt.items():
    #     print(f"{model}: {cnt/total_cnt}")

    # 将请求写入文件
    with open(target_req_file_path, "w") as f:
        for req in model_requests:
            f.write(str(req) + "\n")

if __name__ == "__main__":    
    if args.action == "gen_trace":
        sequence_gen()
    elif args.action == "trace_locality":
        trace_locality()
    else:
        raise ValueError(f"Unknown action: {args.action}")