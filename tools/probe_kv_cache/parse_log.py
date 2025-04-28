def parse_log(log_text):
    lines = log_text.splitlines()
    current_count = -1
    max_counts = []
    
    for line in lines:
        if "Used Blocks:" in line:
            block_num = int(line.split("Used Blocks: ")[1])
            
            # 如果当前数字小于前一个数字,说明开始新的一轮计数
            if block_num <= current_count:
                if current_count >= 0:  # 记录上一轮的最大值
                    max_counts.append(current_count)
                current_count = block_num
            else:
                current_count = block_num
    
    # 添加最后一轮的最大值
    if current_count >= 0:
        max_counts.append(current_count)
        
    return max_counts

# 打印结果
max_counts = parse_log(open('/mnt/n0/background_log.txt').read())
print("每轮Used Blocks的最大值:")
for i, count in enumerate(max_counts, 1):
    print(f"{count}")


# import numpy as np
# import matplotlib.pyplot as plt

# # 绘制CDF图
# def plot_cdf(data):
#     # 计算CDF
#     sorted_data = np.sort(data)
#     yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
#     # 创建图形
#     plt.figure(figsize=(10, 6))
#     plt.plot(sorted_data, yvals, marker='.', linestyle='-')
#     plt.grid(True, alpha=0.3)
    
#     # 设置标签和标题
#     plt.xlabel('Used Blocks')
#     plt.ylabel('CDF')
#     plt.title('Used Blocks CDF')
    
#     # 保存图片
#     plt.savefig('blocks_cdf.png')
#     print("CDF图已保存为 blocks_cdf.png")

# # 调用绘图函数
# plot_cdf(max_counts)