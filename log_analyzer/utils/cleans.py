import os
from datetime import datetime, timedelta
from flask import current_app

def safe_max(sequence, default=None):
    """安全获取最大值，处理空序列"""
    try:
        return max(sequence)
    except ValueError:
        return default

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']




def cleanup_files(dir, days=1):
    """清理超过7天的上传文件"""
    now = datetime.now()
    deleted_files = []

    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            if now - file_mtime > timedelta(days=days):
                os.remove(filepath)
                deleted_files.append(filename)

    return deleted_files