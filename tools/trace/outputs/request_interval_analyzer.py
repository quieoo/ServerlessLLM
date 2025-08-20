#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
请求间隔分析器
分析每个模型请求之间的时间间隔，计算平均请求间隔
"""

import sys
import argparse
from typing import List, Tuple
import statistics
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

def calculate_intervals(requests: List[int]) -> List[int]:
    """
    计算每个请求距离上一次相同请求的间隔
    
    Args:
        requests: 请求ID列表
        
    Returns:
        间隔列表（第一个请求的间隔为0）
    """
    if not requests:
        return []
    
    intervals = []  # 第一个请求的间隔为0
    
    for i in range(1, len(requests)):
        found = False
        # 向前查找相同的请求ID
        for j in range(i-1, -1, -1):
            if requests[i] == requests[j]:
                # 统计范围(i,j)内不同请求的种类数量
                unique_requests = set(requests[j+1:i])
                intervals.append(len(unique_requests))
                found = True
                break
        if not found:
            # 如果没找到相同的请求，间隔为当前位置
            intervals.append(i-1)
    
    return intervals

def analyze_intervals(intervals: List[int]) -> dict:
    """
    分析间隔统计信息
    
    Args:
        intervals: 间隔列表
        
    Returns:
        包含统计信息的字典
    """
    if not intervals:
        return {}
    
    # 排除第一个间隔（通常为0）
    non_zero_intervals = intervals[1:] if len(intervals) > 1 else intervals
    
    stats = {
        'total_requests': len(intervals),
        'total_intervals': len(non_zero_intervals),
        'average_interval': statistics.mean(non_zero_intervals) if non_zero_intervals else 0,
        'median_interval': statistics.median(non_zero_intervals) if non_zero_intervals else 0,
        'min_interval': min(non_zero_intervals) if non_zero_intervals else 0,
        'max_interval': max(non_zero_intervals) if non_zero_intervals else 0,
        'std_interval': statistics.stdev(non_zero_intervals) if len(non_zero_intervals) > 1 else 0,
        'intervals': intervals,
        'non_zero_intervals': non_zero_intervals
    }
    
    return stats

def print_statistics(stats: dict, file_name: str):
    """
    打印统计信息
    
    Args:
        stats: 统计信息字典
        file_name: 文件名
    """
    print(f"\n=== 请求间隔分析结果: {file_name} ===")
    print(f"总请求数: {stats['total_requests']}")
    print(f"总间隔数: {stats['total_intervals']}")
    print(f"平均间隔: {stats['average_interval']:.2f}")
    print(f"中位数间隔: {stats['median_interval']:.2f}")
    print(f"最小间隔: {stats['min_interval']}")
    print(f"最大间隔: {stats['max_interval']}")
    print(f"标准差: {stats['std_interval']:.2f}")
    
    # 显示前10个间隔作为示例
    if stats['intervals']:
        print(f"\n前10个间隔: {stats['intervals'][:10]}")
    
    # 显示间隔分布
    if stats['non_zero_intervals']:
        print(f"\n间隔分布统计:")
        unique_intervals = sorted(set(stats['non_zero_intervals']))
        for interval in unique_intervals[:10]:  # 只显示前10个不同的间隔
            count = stats['non_zero_intervals'].count(interval)
            percentage = (count / len(stats['non_zero_intervals'])) * 100
            print(f"  间隔 {interval}: {count} 次 ({percentage:.1f}%)")

def plot_intervals(stats: dict, file_name: str, save_plot: bool = False):
    """
    绘制间隔分布图
    
    Args:
        stats: 统计信息字典
        file_name: 文件名
        save_plot: 是否保存图片
    """
    if not stats['non_zero_intervals']:
        print("没有足够的间隔数据来绘制图表")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 间隔时间序列图
    ax1.plot(range(len(stats['intervals'])), stats['intervals'], 'b-', alpha=0.7)
    ax1.set_xlabel('请求序号')
    ax1.set_ylabel('间隔时间')
    ax1.set_title('请求间隔时间序列')
    ax1.grid(True, alpha=0.3)
    
    # 间隔分布直方图
    ax2.hist(stats['non_zero_intervals'], bins=min(20, len(set(stats['non_zero_intervals']))), 
             alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(stats['average_interval'], color='red', linestyle='--', 
                label=f'平均值: {stats["average_interval"]:.2f}')
    ax2.axvline(stats['median_interval'], color='orange', linestyle='--', 
                label=f'中位数: {stats["median_interval"]:.2f}')
    ax2.set_xlabel('间隔时间')
    ax2.set_ylabel('频次')
    ax2.set_title('间隔时间分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plot_filename = f"{file_name.replace('.txt', '')}_interval_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {plot_filename}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='分析请求间隔统计')
    parser.add_argument('file_path', help='请求数据文件路径')
    parser.add_argument('--plot', action='store_true', help='显示图表')
    parser.add_argument('--save-plot', action='store_true', help='保存图表到文件')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"正在加载请求数据: {args.file_path}")
    requests = load_request_data(args.file_path)
    print(f"成功加载 {len(requests)} 个请求")
    
    # 计算间隔
    intervals = calculate_intervals(requests)
    
    # 分析统计
    stats = analyze_intervals(intervals)
    
    # 打印结果
    file_name = args.file_path.split('/')[-1]
    print_statistics(stats, file_name)
    
    # 绘制图表
    if args.plot or args.save_plot:
        try:
            plot_intervals(stats, file_name, args.save_plot)
        except ImportError:
            print("警告: matplotlib未安装，跳过图表绘制")
            print("请运行: pip install matplotlib 来安装绘图库")

if __name__ == "__main__":
    main() 