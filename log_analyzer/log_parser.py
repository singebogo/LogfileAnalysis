import re
import os
import mmap
import gc
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import logging


class LogParser:
    def __init__(self, filepath: str):
        """
        初始化日志分析器

        参数:
            filepath: 日志文件路径
        """
        self.filepath = filepath
        self._time_formats = [
            ('%Y-%m-%d %H:%M:%S', r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
            ('%Y-%m-%dT%H:%M:%S%z', r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})'),
            ('%d/%b/%Y:%H:%M:%S %z', r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4})\]'),
            ('%m/%d/%Y %I:%M:%S %p', r'(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M)'),
            ('%Y-%m-%d%H:%M:%S.%f', r'(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}:\d{1,2}.\d{1,10})'),
            ('%b %d %H:%M:%S', r'(\w{3} \d{2} \d{2}:\d{2}:\d{2})'),
            ('%H:%M:%S', r'(\d{2}:\d{2}:\d{2})')
        ]

        # 验证文件存在
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"日志文件不存在: {self.filepath}")

        # 初始化日志
        self.logger = logging.getLogger(__name__)

    def process(self, max_lines: Optional[int] = None,
                sampling_rate: int = 100) -> Dict[str, Union[int, Dict, List]]:
        """
        处理日志文件（支持大文件）

        参数:
            max_lines: 最大处理行数(None表示处理整个文件)
            sampling_rate: 时间统计的采样率(每N行采样一次)

        返回:
            包含分析结果的字典
        """
        # 初始化统计变量
        line_count = 0
        error_count = 0
        warning_count = 0
        time_dist = defaultdict(int)
        error_dist = defaultdict(int)
        sample_errors = []
        first_entry = None
        last_entry = None
        start_time = None
        end_time = None

        # 预编译正则表达式提高效率
        error_pattern = re.compile(r'error:?\s+([\w-]+)', re.IGNORECASE)

        try:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                # 使用内存映射文件提高大文件读取效率
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    for line_bytes in iter(mm.readline, b''):
                        try:
                            line = line_bytes.decode('utf-8').strip()
                        except UnicodeDecodeError:
                            self.logger.warning(f"解码失败的行: {line_bytes[:100]}...")
                            continue

                        if not line:
                            continue

                        # 记录首尾条目
                        if line_count == 0:
                            first_entry = line
                        last_entry = line

                        line_count += 1

                        # 检查是否达到最大行数限制
                        if max_lines and line_count >= max_lines:
                            break

                        # 错误和警告统计
                        lower_line = line.lower()
                        if 'error' in lower_line:
                            error_count += 1

                            # 记录错误样本(最多100条)
                            if len(sample_errors) < 100:
                                sample_errors.append(line)

                            # 错误类型分析
                            match = error_pattern.search(line)
                            error_type = match.group(1) if match else 'unknown'
                            error_dist[error_type] += 1
                        elif 'warning' in lower_line:
                            warning_count += 1

                        # 时间统计(采样处理)
                        if line_count % sampling_rate == 0:
                            timestamp = self._extract_time(line)
                            if timestamp:
                                # 记录首尾时间戳
                                if start_time is None or timestamp < start_time:
                                    start_time = timestamp
                                if end_time is None or timestamp > end_time:
                                    end_time = timestamp

                                # 按小时聚合
                                time_key = timestamp.strftime('%Y-%m-%d %H:00')
                                time_dist[time_key] += sampling_rate  # 按采样率估算

                        # 定期垃圾回收
                        if line_count % 10000 == 0:
                            gc.collect()

            # 如果没有找到时间戳，尝试从首尾条目中提取
            if start_time is None and first_entry:
                start_time = self._extract_time(first_entry)
            if end_time is None and last_entry:
                end_time = self._extract_time(last_entry)

            # 按时间排序时间分布
            sorted_time_dist = dict(sorted(time_dist.items()))

            return {
                'total_entries': line_count,
                'error_count': error_count,
                'warning_count': warning_count,
                'time_distribution': sorted_time_dist,
                'error_distribution': dict(error_dist),
                'sample_errors': sample_errors,
                'first_entry': first_entry,
                'last_entry': last_entry,
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None,
                'sampling_rate': sampling_rate
            }

        except Exception as e:
            self.logger.error(f"处理文件时出错: {str(e)}")
            return {
                'error': f'处理文件时出错: {str(e)}',
                'processed_lines': line_count
            }

    def _extract_time(self, log_entry: str) -> Optional[datetime]:
        """从日志条目中提取时间戳"""
        for pattern, fmt in self._time_formats:
            match = re.search(pattern, log_entry)
            if match:
                try:
                    return datetime.strptime(match.group(1), fmt)
                except ValueError:
                    continue
        return None

    def get_statistics(self) -> Dict[str, Union[int, str]]:
        """获取基本统计信息，确保返回值可序列化"""
        stats = {
            'total_entries': 0,
            'error_count': 0,
            'warning_count': 0,
            'start_time': None,
            'end_time': None
        }

        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                stats['total_entries'] += 1

                lower_line = line.lower()
                if 'error' in lower_line:
                    stats['error_count'] += 1
                elif 'warning' in lower_line:
                    stats['warning_count'] += 1

                if stats['start_time'] is None or stats['end_time'] is None:
                    timestamp = self._extract_time(line)
                    if timestamp:
                        if stats['start_time'] is None or timestamp < stats['start_time']:
                            stats['start_time'] = timestamp.isoformat()
                        if stats['end_time'] is None or timestamp > stats['end_time']:
                            stats['end_time'] = timestamp.isoformat()

        return stats

    def get_error_logs(self, limit: int = 100) -> List[str]:
        """
        获取错误日志样本（优化实现）
        """
        errors = []
        try:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'error' in line.lower():
                        errors.append(line.strip())
                        if len(errors) >= limit:
                            break
        except Exception as e:
            self.logger.error(f"获取错误日志失败: {str(e)}")
        return errors

    def get_logs_by_time(self, interval='hour') -> dict:
        """返回时间分布字典"""
        time_counts = defaultdict(int)
        sample_rate = 100

        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i % sample_rate == 0:
                    timestamp = self._extract_time(line)
                    if timestamp:
                        if interval == 'hour':
                            key = timestamp.strftime('%Y-%m-%d %H:00')
                        elif interval == 'day':
                            key = timestamp.strftime('%Y-%m-%d')
                        else:
                            key = timestamp.strftime('%Y-%m-%d %H:%M')
                        time_counts[key] += sample_rate

        return dict(sorted(time_counts.items()))

    def get_error_distribution(self) -> dict:
        """返回错误分布字典"""
        error_types = defaultdict(int)
        error_pattern = re.compile(r'error:?\s+([\w-]+)', re.IGNORECASE)

        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'error' in line.lower():
                    match = error_pattern.search(line)
                    if match:
                        error_type = match.group(1)
                        error_types[error_type] += 1
                    else:
                        error_types['unknown'] += 1

        return dict(error_types)

    def analyze(self, callback=None) -> Dict[str, Union[int, Dict, List]]:
        """
        完整分析日志文件，支持进度回调

        参数:
            callback: 回调函数，接收(current, total)参数

        返回:
            完整分析结果
        """
        return self.process()

    def validate_time_distribution(self, time_data):
        """验证时间分布数据格式"""
        if not isinstance(time_data, dict):
            raise ValueError("time_distribution must be a dictionary")
        for k, v in time_data.items():
            if not isinstance(k, str) or not isinstance(v, (int, float)):
                raise ValueError("Invalid time_distribution format")
        return True