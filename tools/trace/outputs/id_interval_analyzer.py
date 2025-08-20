#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ID间隔分析器
统计每个ID和其上次出现位置之间共有几种其他ID
"""

import sys
import argparse
from typing import List, Dict, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_request_data(file_path: str) -> List[int]:
    """
    从文件中加载请求数据
    
    Args:
        file_path: 请求数据文件路径
        
    Returns:
        请求ID列表
    """
    try:
        with open(file_path, 'r') as f:
            # 读取所有行，去除空白字符，转换为整数
            requests = [int(line.strip()) for line in f if line.strip()]
        return requests
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        sys.exit(1)
    except ValueError as e:
        print(f"错误: 文件格式不正确 - {e}")
        sys.exit(1)

def analyze_id_intervals(requests: List[int]) -> Dict:
    """
    分析每个ID和其上次出现位置之间共有几种其他ID
    
    Args:
        requests: 请求ID列表
        
    Returns:
        包含分析结果的字典
    """
    if not requests:
        return {}
    
    # 记录每个ID最后出现的位置
    last_positions = {}
    
    # 存储每个ID的间隔分析结果
    interval_analysis = []
    
    # 统计所有唯一的ID
    all_ids = set(requests)
    
    for i, current_id in enumerate(requests):
        if current_id in last_positions:
            # 找到上次出现的位置
            last_pos = last_positions[current_id]
            
            # 获取两个位置之间的所有ID
            interval_ids = set(requests[last_pos + 1:i])
            
            # 计算间隔中不同ID的数量
            unique_ids_in_interval = len(interval_ids)
            
            # 记录分析结果
            analysis = {
                'position': i,
                'id': current_id,
                'last_position': last_pos,
                'interval_length': i - last_pos - 1,
                'unique_ids_in_interval': unique_ids_in_interval,
                'ids_in_interval': list(interval_ids)
            }
            interval_analysis.append(analysis)
        else:
            # 第一次出现的ID
            analysis = {
                'position': i,
                'id': current_id,
                'last_position': -1,
                'interval_length': i,
                'unique_ids_in_interval': len(set(requests[:i])),
                'ids_in_interval': list(set(requests[:i]))
            }
            interval_analysis.append(analysis)
        
        # 更新最后出现位置
        last_positions[current_id] = i
    
    # 计算统计信息
    unique_id_counts = [item['unique_ids_in_interval'] for item in interval_analysis]
    
    stats = {
        'total_requests': len(requests),
        'unique_ids': len(all_ids),
        'total_intervals': len(interval_analysis),
        'average_unique_ids': np.mean(unique_id_counts) if unique_id_counts else 0,
        'median_unique_ids': np.median(unique_id_counts) if unique_id_counts else 0,
        'min_unique_ids': min(unique_id_counts) if unique_id_counts else 0,
        'max_unique_ids': max(unique_id_counts) if unique_id_counts else 0,
        'std_unique_ids': np.std(unique_id_counts) if len(unique_id_counts) > 1 else 0,
        'interval_analysis': interval_analysis,
        'unique_id_counts': unique_id_counts
    }
    
    return stats

def print_analysis_results(stats: dict, file_name: str, show_details: bool = False):
    """
    打印分析结果
    
    Args:
        stats: 统计信息字典
        file_name: 文件名
        show_details: 是否显示详细信息
    """
    print(f"\n=== ID间隔分析结果: {file_name} ===")
    print(f"总请求数: {stats['total_requests']}")
    print(f"唯一ID数量: {stats['unique_ids']}")
    print(f"总间隔数: {stats['total_intervals']}")
    print(f"平均间隔中唯一ID数: {stats['average_unique_ids']:.2f}")
    print(f"中位数间隔中唯一ID数: {stats['median_unique_ids']:.2f}")
    print(f"最小间隔中唯一ID数: {stats['min_unique_ids']}")
    print(f"最大间隔中唯一ID数: {stats['max_unique_ids']}")
    print(f"标准差: {stats['std_unique_ids']:.2f}")
    
    # 输出间隔中唯一ID数分布
    # print(f"间隔中唯一ID数分布: {stats['unique_id_counts']}")

    if show_details and stats['interval_analysis']:
        print(f"\n详细分析结果:")
        for i, analysis in enumerate(stats['interval_analysis'][:10]):  # 只显示前10个
            print(f"  位置 {analysis['position']}: ID {analysis['id']} "
                  f"(上次位置: {analysis['last_position']}, "
                  f"间隔长度: {analysis['interval_length']}, "
                  f"间隔中唯一ID数: {analysis['unique_ids_in_interval']})")
            if analysis['ids_in_interval']:
                print(f"    间隔中的ID: {analysis['ids_in_interval']}")
    
    # 显示间隔中唯一ID数的分布
    if stats['unique_id_counts']:
        print(f"\n间隔中唯一ID数分布:")
        unique_counts = sorted(set(stats['unique_id_counts']))
        for count in unique_counts[:10]:  # 只显示前10个不同的计数
            frequency = stats['unique_id_counts'].count(count)
            percentage = (frequency / len(stats['unique_id_counts'])) * 100
            print(f"  {count} 个唯一ID: {frequency} 次 ({percentage:.1f}%)")

def plot_analysis_results(stats: dict, file_name: str, save_plot: bool = False):
    """
    绘制分析结果图表
    
    Args:
        stats: 统计信息字典
        file_name: 文件名
        save_plot: 是否保存图片
    """
    if not stats['unique_id_counts']:
        print("没有足够的数据来绘制图表")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 间隔中唯一ID数的时间序列
    positions = [item['position'] for item in stats['interval_analysis']]
    ax1.plot(positions, stats['unique_id_counts'], 'b-', alpha=0.7)
    ax1.set_xlabel('请求位置')
    ax1.set_ylabel('间隔中唯一ID数')
    ax1.set_title('间隔中唯一ID数时间序列')
    ax1.grid(True, alpha=0.3)
    
    # 间隔中唯一ID数分布直方图
    ax2.hist(stats['unique_id_counts'], bins=min(20, len(set(stats['unique_id_counts']))), 
             alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(stats['average_unique_ids'], color='red', linestyle='--', 
                label=f'平均值: {stats["average_unique_ids"]:.2f}')
    ax2.axvline(stats['median_unique_ids'], color='orange', linestyle='--', 
                label=f'中位数: {stats["median_unique_ids"]:.2f}')
    ax2.set_xlabel('间隔中唯一ID数')
    ax2.set_ylabel('频次')
    ax2.set_title('间隔中唯一ID数分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 间隔长度与唯一ID数的关系
    interval_lengths = [item['interval_length'] for item in stats['interval_analysis']]
    ax3.scatter(interval_lengths, stats['unique_id_counts'], alpha=0.6, color='purple')
    ax3.set_xlabel('间隔长度')
    ax3.set_ylabel('间隔中唯一ID数')
    ax3.set_title('间隔长度 vs 唯一ID数')
    ax3.grid(True, alpha=0.3)
    
    # 每个ID的出现频率
    id_frequencies = defaultdict(int)
    for item in stats['interval_analysis']:
        id_frequencies[item['id']] += 1
    
    ids = list(id_frequencies.keys())
    frequencies = list(id_frequencies.values())
    
    # 只显示前20个最频繁的ID
    if len(ids) > 20:
        sorted_pairs = sorted(zip(ids, frequencies), key=lambda x: x[1], reverse=True)
        ids = [pair[0] for pair in sorted_pairs[:20]]
        frequencies = [pair[1] for pair in sorted_pairs[:20]]
    
    ax4.bar(range(len(ids)), frequencies, alpha=0.7, color='orange')
    ax4.set_xlabel('ID')
    ax4.set_ylabel('出现次数')
    ax4.set_title('ID出现频率 (前20个)')
    ax4.set_xticks(range(len(ids)))
    ax4.set_xticklabels(ids, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plot_filename = f"{file_name.replace('.txt', '')}_id_interval_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {plot_filename}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='分析ID间隔统计')
    parser.add_argument('file_path', help='请求数据文件路径')
    parser.add_argument('--details', action='store_true', help='显示详细分析结果')
    parser.add_argument('--plot', action='store_true', help='显示图表')
    parser.add_argument('--save-plot', action='store_true', help='保存图表到文件')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"正在加载请求数据: {args.file_path}")
    requests = load_request_data(args.file_path)
    print(f"成功加载 {len(requests)} 个请求")
    
    # 分析ID间隔
    stats = analyze_id_intervals(requests)
    
    # 打印结果
    file_name = args.file_path.split('/')[-1]
    print_analysis_results(stats, file_name, args.details)
    
    # 绘制图表
    if args.plot or args.save_plot:
        try:
            plot_analysis_results(stats, file_name, args.save_plot)
        except ImportError:
            print("警告: matplotlib未安装，跳过图表绘制")
            print("请运行: pip install matplotlib 来安装绘图库")

if __name__ == "__main__":
    main() 