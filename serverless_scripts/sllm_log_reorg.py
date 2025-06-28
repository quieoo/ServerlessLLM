import re
from datetime import datetime
import argparse


def parse_log_line(line):
    """解析单行日志提取时间戳"""
    timestamp_match = re.search(r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if timestamp_match:
        try:
            return datetime.strptime(timestamp_match.group(), '%m-%d %H:%M:%S'), line
        except ValueError:
            return None, line
    return None, line


def sort_logs(input_file, output_file):
    """主排序函数"""
    logs = []
    
    with open(input_file, 'r') as f:
        for line in f:
            ts, content = parse_log_line(line.strip())
            logs.append((ts, content))

    # 按时间排序，无时间戳的日志放在最后
    sorted_logs = sorted(logs, key=lambda x: x[0] if x[0] else datetime.max)

    with open(output_file, 'w') as f:
        for ts, log in sorted_logs:
            f.write(log + '\n')


def main():
    parser = argparse.ArgumentParser(description='ServerlessLLM 日志重组工具')
    parser.add_argument('-i', '--input', required=True, help='输入日志文件路径')
    parser.add_argument('-o', '--output', required=False, help='输出文件路径')
    parser.add_argument('--inplace', action='store_true', 
                      help='直接修改原文件（自动创建.bak备份）')
    
    args = parser.parse_args()

    if args.inplace:
        import shutil
        shutil.copy2(args.input, args.input + '.log')
        sort_logs(args.input, args.input)
    else:
        sort_logs(args.input, args.output)


if __name__ == '__main__':
    main()