import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from collections import defaultdict
import base64

from flask import current_app

def get_combined_stats(file_data):
    """计算合并统计信息，安全处理空序列"""
    combined = {
        'total_files': len(file_data),
        'total_entries': sum(d['stats']['total_entries'] for d in file_data),
        'total_errors': sum(d['stats']['error_count'] for d in file_data),
        'total_warnings': sum(d['stats']['warning_count'] for d in file_data),
    }

    # 安全处理时间戳
    start_times = [d['stats']['start_time'] for d in file_data
                   if d['stats']['start_time'] is not None]
    end_times = [d['stats']['end_time'] for d in file_data
                 if d['stats']['end_time'] is not None]

    combined['earliest_time'] = min(start_times) if start_times else None
    combined['latest_time'] = max(end_times) if end_times else None

    return combined


def generate_combined_time_plot(file_data):
    """生成合并时间图表"""
    combined_time_data = defaultdict(int)

    for data in file_data:
        # 确保time_distribution存在且是字典
        time_data = data.get('time_distribution', {})
        if not isinstance(time_data, dict):
            current_app.logger.warning(f"Invalid time_distribution in file data: {data.get('filename', 'unknown')}")
            continue

        for time_slot, count in time_data.items():
            combined_time_data[time_slot] += count

    if not combined_time_data:
        current_app.logger.warning("No valid time distribution data found")
        return generate_empty_plot("No Time Data Available")

    return generate_time_plot(combined_time_data)


def generate_empty_plot(message):
    """生成空图表占位"""
    plt.figure(figsize=(10, 5))
    plt.text(0.5, 0.5, message, ha='center', va='center')
    plt.axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')


def generate_combined_error_plot_from_data(file_data):
    """从已处理的数据生成合并错误图"""
    error_dist = defaultdict(int)

    for data in file_data:
        for error_type, count in data['error_distribution'].items():
            error_dist[error_type] += count

    plt.figure(figsize=(12, 6))
    plt.bar(error_dist.keys(), error_dist.values())
    plt.title('Combined Error Type Distribution')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')


def generate_time_plot(time_data: dict):
    """从时间数据字典生成图表"""
    if not isinstance(time_data, dict):
        raise ValueError("time_data must be a dictionary")

    plt.figure(figsize=(10, 5))
    plt.plot(list(time_data.keys()), list(time_data.values()))
    plt.title('Log Entries Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Entries')
    plt.xticks(rotation=45)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')


def generate_error_plot(error_data: dict):
    """从错误数据字典生成图表"""
    if not isinstance(error_data, dict):
        raise ValueError("error_data must be a dictionary")

    plt.figure(figsize=(10, 5))
    plt.bar(error_data.keys(), error_data.values())
    plt.title('Error Type Distribution')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')


def generate_combined_error_plot(file_data):
    """生成合并错误类型分布图"""
    error_dist = defaultdict(int)

    for data in file_data:
        for error_type, count in data['error_distribution'].items():
            error_dist[error_type] += count

    plt.figure(figsize=(12, 6))
    plt.bar(error_dist.keys(), error_dist.values())
    plt.title('Combined Error Type Distribution')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')


def generate_lightweight_plot(x, y, title, xlabel, ylabel, plot_type='line'):
    """生成轻量级图表，减少数据点"""
    # 数据采样
    if len(x) > app.config['PLOT_DATA_POINTS']:
        step = len(x) // app.config['PLOT_DATA_POINTS']
        x = x[::step]
        y = y[::step]

    plt.figure(figsize=(10, 5))
    if plot_type == 'line':
        plt.plot(x, y)
    else:
        plt.bar(x, y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80)  # 降低DPI
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def down_sample_data(data, max_points=500):
    """减少数据点数量"""
    if len(data) <= max_points:
        return data

    step = len(data) // max_points
    keys = list(data.keys())
    values = list(data.values())

    return {
        keys[i]: sum(values[i:i + step]) for i in range(0, len(keys), step)
    }