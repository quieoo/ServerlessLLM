#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ID分离分析器
按照不同的ID分开统计每个ID和其上次出现位置之间共有几种其他ID
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

def analyze_id_intervals_separately(requests: List[int]) -> Dict:
    """
    按照不同ID分开分析每个ID和其上次出现位置之间共有几种其他ID
    
    Args:
        requests: 请求ID列表
        
    Returns:
        包含每个ID分析结果的字典
    """
    if not requests:
        return {}
    
    # 记录每个ID的所有出现位置
    id_positions = defaultdict(list)
    
    # 记录每个ID的所有出现位置
    for i, current_id in enumerate(requests):
        id_positions[current_id].append(i)
    
    # 为每个ID分析间隔
    id_analysis = {}
    
    for id_val, positions in id_positions.items():
        if len(positions) == 1:
            # 只出现一次的ID
            analysis = {
                'id': id_val,
                'total_occurrences': 1,
                'intervals': [],
                'unique_ids_in_intervals': [],
                'average_unique_ids': 0,
                'median_unique_ids': 0,
                'min_unique_ids': 0,
                'max_unique_ids': 0,
                'std_unique_ids': 0
            }
        else:
            # 多次出现的ID
            intervals = []
            unique_ids_in_intervals = []
            
            for i in range(1, len(positions)):
                last_pos = positions[i-1]
                current_pos = positions[i]
                
                # 获取两个位置之间的所有ID
                interval_ids = set(requests[last_pos + 1:current_pos])
                unique_count = len(interval_ids)
                
                intervals.append({
                    'from_position': last_pos,
                    'to_position': current_pos,
                    'interval_length': current_pos - last_pos - 1,
                    'unique_ids_in_interval': unique_count,
                    'ids_in_interval': list(interval_ids)
                })
                unique_ids_in_intervals.append(unique_count)
            
            # 计算统计信息
            if unique_ids_in_intervals:
                analysis = {
                    'id': id_val,
                    'total_occurrences': len(positions),
                    'intervals': intervals,
                    'unique_ids_in_intervals': unique_ids_in_intervals,
                    'average_unique_ids': np.mean(unique_ids_in_intervals),
                    'median_unique_ids': np.median(unique_ids_in_intervals),
                    'min_unique_ids': min(unique_ids_in_intervals),
                    'max_unique_ids': max(unique_ids_in_intervals),
                    'std_unique_ids': np.std(unique_ids_in_intervals) if len(unique_ids_in_intervals) > 1 else 0
                }
            else:
                analysis = {
                    'id': id_val,
                    'total_occurrences': len(positions),
                    'intervals': [],
                    'unique_ids_in_intervals': [],
                    'average_unique_ids': 0,
                    'median_unique_ids': 0,
                    'min_unique_ids': 0,
                    'max_unique_ids': 0,
                    'std_unique_ids': 0
                }
        
        id_analysis[id_val] = analysis
    
    # 计算总体统计
    all_unique_counts = []
    for analysis in id_analysis.values():
        all_unique_counts.extend(analysis['unique_ids_in_intervals'])
    
    overall_stats = {
        'total_requests': len(requests),
        'unique_ids': len(id_analysis),
        'total_intervals': len(all_unique_counts),
        'average_unique_ids': np.mean(all_unique_counts) if all_unique_counts else 0,
        'median_unique_ids': np.median(all_unique_counts) if all_unique_counts else 0,
        'min_unique_ids': min(all_unique_counts) if all_unique_counts else 0,
        'max_unique_ids': max(all_unique_counts) if all_unique_counts else 0,
        'std_unique_ids': np.std(all_unique_counts) if len(all_unique_counts) > 1 else 0,
        'id_analysis': id_analysis,
        'all_unique_counts': all_unique_counts
    }
    
    return overall_stats

def print_analysis_results(stats: dict, file_name: str, show_details: bool = False):
    """
    打印分析结果
    
    Args:
        stats: 统计信息字典
        file_name: 文件名
        show_details: 是否显示详细信息
    """
    print(f"\n=== ID分离分析结果: {file_name} ===")
    print(f"总请求数: {stats['total_requests']}")
    print(f"唯一ID数量: {stats['unique_ids']}")
    print(f"总间隔数: {stats['total_intervals']}")
    print(f"总体平均间隔中唯一ID数: {stats['average_unique_ids']:.2f}")
    print(f"总体中位数间隔中唯一ID数: {stats['median_unique_ids']:.2f}")
    print(f"总体最小间隔中唯一ID数: {stats['min_unique_ids']}")
    print(f"总体最大间隔中唯一ID数: {stats['max_unique_ids']}")
    print(f"总体标准差: {stats['std_unique_ids']:.2f}")
    
    print(f"\n=== 各ID详细统计 ===")
    
    # 按出现次数排序
    sorted_ids = sorted(stats['id_analysis'].items(), 
                       key=lambda x: x[1]['total_occurrences'], reverse=True)
    
    for id_val, analysis in sorted_ids:
        print(f"\nID {id_val}:")
        print(f"  出现次数: {analysis['total_occurrences']}")
        
        if analysis['total_occurrences'] > 1:
            print(f"  间隔数: {len(analysis['intervals'])}")
            print(f"  平均间隔中唯一ID数: {analysis['average_unique_ids']:.2f}")
            print(f"  中位数间隔中唯一ID数: {analysis['median_unique_ids']:.2f}")
            print(f"  最小间隔中唯一ID数: {analysis['min_unique_ids']}")
            print(f"  最大间隔中唯一ID数: {analysis['max_unique_ids']}")
            print(f"  标准差: {analysis['std_unique_ids']:.2f}")
            # 输出间隔中唯一ID数分布, 比如：0个唯一ID: 10次 (10.0%), 1个唯一ID: 20次 (20.0%), 2个唯一ID: 30次 (30.0%), 3个唯一ID: 40次 (40.0%), 4个唯一ID: 50次 (50.0%)
            unique_id_counts = analysis['unique_ids_in_intervals']
            unique_id_counts_set = set(unique_id_counts)
            for unique_id_count in unique_id_counts_set:
                frequency = unique_id_counts.count(unique_id_count)
                percentage = (frequency / len(unique_id_counts))
                print(f"  {unique_id_count} 个唯一ID: {frequency} 次, 占比: {percentage:.4f}")

        else:
            print(f"  只出现一次，无间隔分析")
    
    # 显示统计摘要
    print(f"\n=== 统计摘要 ===")
    occurrence_counts = [analysis['total_occurrences'] for analysis in stats['id_analysis'].values()]
    print(f"平均每个ID出现次数: {np.mean(occurrence_counts):.2f}")
    print(f"中位数每个ID出现次数: {np.median(occurrence_counts):.2f}")
    print(f"最多出现次数: {max(occurrence_counts)}")
    print(f"最少出现次数: {min(occurrence_counts)}")

def plot_analysis_results(stats: dict, file_name: str, save_plot: bool = False):
    """
    绘制分析结果图表
    
    Args:
        stats: 统计信息字典
        file_name: 文件名
        save_plot: 是否保存图片
    """
    if not stats['all_unique_counts']:
        print("没有足够的数据来绘制图表")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 各ID的平均间隔中唯一ID数
    ids = []
    avg_unique_counts = []
    occurrence_counts = []
    
    for id_val, analysis in stats['id_analysis'].items():
        if analysis['total_occurrences'] > 1:  # 只显示多次出现的ID
            ids.append(id_val)
            avg_unique_counts.append(analysis['average_unique_ids'])
            occurrence_counts.append(analysis['total_occurrences'])
    
    if ids:
        # 按ID排序
        sorted_data = sorted(zip(ids, avg_unique_counts, occurrence_counts))
        ids, avg_unique_counts, occurrence_counts = zip(*sorted_data)
        
        ax1.bar(range(len(ids)), avg_unique_counts, alpha=0.7, color='blue')
        ax1.set_xlabel('ID')
        ax1.set_ylabel('平均间隔中唯一ID数')
        ax1.set_title('各ID平均间隔中唯一ID数')
        ax1.set_xticks(range(len(ids)))
        ax1.set_xticklabels(ids, rotation=45)
        ax1.grid(True, alpha=0.3)
    
    # 总体间隔中唯一ID数分布
    ax2.hist(stats['all_unique_counts'], bins=min(20, len(set(stats['all_unique_counts']))), 
             alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(stats['average_unique_ids'], color='red', linestyle='--', 
                label=f'平均值: {stats["average_unique_ids"]:.2f}')
    ax2.axvline(stats['median_unique_ids'], color='orange', linestyle='--', 
                label=f'中位数: {stats["median_unique_ids"]:.2f}')
    ax2.set_xlabel('间隔中唯一ID数')
    ax2.set_ylabel('频次')
    ax2.set_title('总体间隔中唯一ID数分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ID出现次数与平均间隔中唯一ID数的关系
    if ids:
        ax3.scatter(occurrence_counts, avg_unique_counts, alpha=0.6, color='purple')
        ax3.set_xlabel('ID出现次数')
        ax3.set_ylabel('平均间隔中唯一ID数')
        ax3.set_title('ID出现次数 vs 平均间隔中唯一ID数')
        ax3.grid(True, alpha=0.3)
    
    # ID出现频率分布
    occurrence_counts_all = [analysis['total_occurrences'] for analysis in stats['id_analysis'].values()]
    ax4.hist(occurrence_counts_all, bins=min(15, len(set(occurrence_counts_all))), 
             alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('ID出现次数')
    ax4.set_ylabel('ID数量')
    ax4.set_title('ID出现次数分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plot_filename = f"{file_name.replace('.txt', '')}_id_separate_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {plot_filename}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='按ID分离分析间隔统计')
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
    stats = analyze_id_intervals_separately(requests)
    
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