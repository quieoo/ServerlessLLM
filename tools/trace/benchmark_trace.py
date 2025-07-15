from trace import Trace, TraceReplay, report_group_stats
from scipy.stats import entropy
import numpy as np
import json

#需要： conda activate sllm-trace
# 将Func映射到模型
# 相关工作[Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving]指出在真实的Serverless LLM平台中模型分布：
# Small Model 1B-3B: 43
# Medium Model 4B-8B: 8
# Large Model 9B-30B: 3
# Extra Large Model 31B-70B: 4

trace_name = "azure_v2"
# trace_dir = "/mnt/e/projects/projects/dataset/mms_dataset/azure_v2.pkl"

# trace_name = "azure_v1"
# trace_dir = "/mnt/e/projects/projects/dataset/mms_dataset/azure_v1.pkl"
# trace_name="txt"
trace_dir = "/mnt/n0/datasets/azura_v2.txt"
target_cv = 0.5
target_req_file_path="/mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv0.5.large.txt"
sllm_model_config_file_path="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-large.json"
# sllm_model_config_file_path="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-small.json"
# sllm_model_config_file_path="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/sllm_model_config.json"
# sllm_model_config_file_path="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/4090-uniform.json"


rate_cv_map={
    0.125 : 1e-3,
    0.25 : 1e-3,
    0.5 : 1e-4 * 5,
    1:1e-5 * 5,
    2:1e-10 * 5,
    4 : 1e-40,
}

def sequence_gen():
    # 读取json配置文件获取模型列表
    with open(sllm_model_config_file_path, "r") as f:
        model_config = json.load(f)
    model_dirs = model_config["model_dirs"]
    weighted_mapping = model_config["model_affinity"]
    # if target_cv==1:
    #     # 第一个模型的密度等于最后一个模型
    #     weighted_mapping[0]=weighted_mapping[-1]
    # 将模型名转换为序列号
    model_ids=[]
    for i in range(len(model_dirs)):
        model_ids.append(i)
    print(f"model_ids: {model_ids}")
    print(f"weighted_mapping: {weighted_mapping}")
    days=7
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
                                interval_seconds=3600,
                                rate_scale_factor=rate_cv_map[target_cv],
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


# 提供测试例
def sample_trace():
    n_model = 10
    models = [f"gpt{i}" for i in range(n_model)]
    trace = Trace(trace_name, trace_dir)


    def cdf(x):
        x = np.array(x)
        return np.cumsum(np.sort(x)[::-1]) / np.sum(x)


    entropies = {}

    for day in range(14):
        for h in range(24):
            start_time = str(day) + "." + str(h) + ".0"
            if h < 23:
                end_time = str(day) + "." + str(h+1) + ".0"
            else:
                end_time = str(day) + "." + "23.60"


            replays = trace.replay(models,
                                model_mapping_strategy="stripe",
                                start_time=start_time,
                                end_time=end_time,
                                interval_seconds=60,
                                arrival_distribution="exponential",
                                rate_scale_factor=1e-3)
            print(f"-----{start_time} {end_time}-----")
            # 输出replay的内容
            for m in replays:
                print(m, replays[m].arrivals)
            # for m in replays:
            #     replays[m].report_stats()
                # replays[m].visualize(n_interval=1000)
            report_group_stats(list(replays.values()))
            x = [replays[model].arrivals.size for model in replays]
            # print(x)
            entropies[start_time] = entropy(x)
            # print(f"Entropy for {start_time} - {end_time}: {entropy(x)}, top-5 {np.sum(cdf(x)[:5]) / np.sum(cdf(x))}")

    print(entropies)
    print(max(entropies.values()))

    # for day in range(13, 14):
    #     start_time = str(day) + ".0.0"
    #     end_time = str(day+1) + ".0.0"
    #
    #     if day == 13:
    #         end_time = "13.23.60"
    #     # replication_factors = [1, 2, 3]
    #     print(f"Day: {start_time} - {end_time}")
    #     distributions = ["gamma"]
        # for rf in replication_factors:
        # replays = trace.replay_vanilla(models,
        #                                model_mapping_strategy="stripe",
        #                                start_time=start_time,
        #                                end_time=end_time)
        # for m in replays:
        #     replays[m].report_stats()
        #     # replays[m].visualize(n_interval=1000)
        # report_group_stats(list(replays.values()))

        # for distribution in distributions:
        #     replays = trace.replay(models,
        #                            model_mapping_strategy="stripe",
        #                            start_time=start_time,
        #                            end_time=end_time,
        #                            interval_seconds=5400,
        #                            arrival_distribution=distribution)
        #     # for m in replays:
        #     #     replays[m].report_stats()
        #         # replays[m].visualize(n_interval=1000)
        #     report_group_stats(list(replays.values()))
        #     x = [replays[model].arrivals.size for model in replays]
        #     print(x)
        #     print(f"Entropy for {start_time} - {end_time}: {entropy(x)}, CDF: {cdf(x)}")

    # replays = trace.replay(models,
    #                        model_mapping_strategy="stripe",
    #                        start_time="0.0.0",
    #                        end_time="2.0.0",
    #                        interval_seconds=86400 // 2,
    #                        arrival_distribution="gamma")
    # for m in replays:
    #     replays[m].report_stats()
    #     replays[m].visualize()


    # interval_seconds = [600]
    # time_scale_factors = [2.0, 4.0, 8.0]
    # for interval_secs in interval_seconds:
    #     # for distribution in ["exponential", "gamma", "vanilla"]:
    #     for distribution in ["vanilla"]:
    #         for time_scale_factor in time_scale_factors:
    #             replays = trace.replay(models,
    #                                    model_mapping_strategy="stripe",
    #                                    start_time="0.0.0",
    #                                    end_time="1.0.0",
    #                                    arrival_distribution=distribution,
    #                                    interval_seconds=interval_secs,
    #                                    time_scale_factor=time_scale_factor)
    #             for m in replays:
    #                 replays[m].report_stats()
    #                 replays[m].visualize()


if __name__ == "__main__":
    sequence_gen()    
