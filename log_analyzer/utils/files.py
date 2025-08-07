import json
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
from flask import current_app
import os
import shutil
import glob
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_chunks(temp_dir, user_upload_dir, file_id):
    """
    合并分块文件
    参数:
        temp_dir: 临时目录路径
        file_id: 文件唯一ID
    返回:
        合并后的文件路径
    """
    # 获取原始文件名(如果存在)
    try:
        original_filename = None
        meta_file = os.path.join(temp_dir, 'metadata.json')
        if os.path.exists(meta_file):
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)
                    original_filename = secure_filename(metadata.get('filename'))
            except Exception as e:
                logger.warning(f"读取metadata失败: {str(e)}")

        # 确定输出文件名
        output_filename = original_filename or file_id
        output_path = os.path.join(user_upload_dir, output_filename)
        # 使用UUID作为临时文件名
        import uuid
        temp_output_path = f"{output_path}.tmp-{uuid.uuid4().hex}"


        # 获取并排序分块文件
        chunk_files = sorted(
            [f for f in os.listdir(temp_dir) if f.startswith('chunk_')],
            key=lambda x: int(x.split('_')[1])
        )

        # 合并文件
        try:
            with open(temp_output_path, 'wb') as output_file:
                for chunk_name in chunk_files:
                    chunk_path = os.path.join(temp_dir, chunk_name)
                    with open(chunk_path, 'rb') as chunk_file:
                        while True:
                            data = chunk_file.read(64 * 1024)  # 64KB缓冲
                            if not data:
                                break
                            output_file.write(data)

            # 验证文件大小(如果有记录)
            if os.path.exists(meta_file):
                try:
                    with open(meta_file) as f:
                        metadata = json.load(f)
                        expected_size = metadata.get('filesize')
                        if expected_size:
                            actual_size = os.path.getsize(temp_output_path)
                            if int(expected_size) != actual_size:
                                raise ValueError(
                                    f"文件大小不匹配: 预期 {expected_size}, 实际 {actual_size}"
                                )
                except Exception as e:
                    logger.warning(f"文件大小验证失败: {str(e)}")

            # 重命名临时文件
            safe_replace_file(temp_output_path, output_path)

            # 设置文件权限
            os.chmod(output_path, 0o644)

            # 清理临时文件
            cleanup_temp_files(temp_dir)
            return output_path

        except Exception as e:
            logger.error(f"清理不完整的输出文件 合并失败 {e}")
            raise
        finally:
            # 清理不完整的输出文件
            if os.path.exists(temp_output_path):
                try:
                    os.unlink(temp_output_path)
                except:
                    pass
    except Exception as e:
        # 清理不完整的输出文件
        logger.error(f"合并过程出错: {str(e)}")
        raise
    finally:
        # 清理不完整的输出文件
        if os.path.exists(temp_dir):
            try:
                cleanup_temp_files(temp_dir)
            except:
                pass


def get_report_files(deviceID):
    """获取reports目录下的所有报告文件，并按创建时间排序"""
    report_dir = os.path.join(current_app.config['REPORT_FOLDER'], deviceID)
    reports = []

    try:
        os.makedirs(report_dir, exist_ok=True)
        # 设置目录权限（确保安全）
        os.chmod(report_dir, 0o755)
        os.chmod(report_dir, 0o755)
    except OSError as e:
        logger.error(f"创建临时目录失败: {str(e)}")
        return reports

    for filename in os.listdir(report_dir):
        filepath = os.path.join(report_dir, filename)
        if os.path.isfile(filepath):
            # 获取文件状态信息
            file_stat = os.stat(filepath)

            # 使用创建时间(ctime)或修改时间(mtime)
            # Windows系统通常使用ctime作为创建时间
            # Unix-like系统ctime是状态变更时间，mtime是修改时间
            file_time = file_stat.st_ctime  # 或者使用 st_mtime

            # 转换为可读日期格式
            try:
                file_date = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
                sort_key = file_time  # 用于排序的时间戳
            except:
                file_date = "未知日期"
                sort_key = 0  # 未知日期排在最前或最后

            # 获取文件大小
            size = file_stat.st_size
            size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"

            # 获取文件类型图标
            ext = filename.split('.')[-1].lower()
            icon = {
                'html': 'fa-file-code',
                'pdf': 'fa-file-pdf',
                'doc': 'fa-file-word',
                'docx': 'fa-file-word',
                'xls': 'fa-file-excel',
                'xlsx': 'fa-file-excel',
                'csv': 'fa-file-csv',
                'txt': 'fa-file-alt'
            }.get(ext, 'fa-file')

            reports.append({
                'filename': filename,
                'display_name': filename.replace('_', ' ').replace('.', ' '),
                'date': file_date,  # 显示用日期字符串
                'sort_key': sort_key,  # 排序用时间戳
                'size': size_str,
                'icon': icon,
                'type': ext.upper()
            })

    # 按时间戳降序排序（最新的在前）
    reports.sort(key=lambda x: x['sort_key'], reverse=True)
    return reports


def cleanup_temp_files(temp_dir):
    """安全清理临时文件"""
    try:
        for f in os.listdir(temp_dir):
            try:
                os.unlink(os.path.join(temp_dir, f))
            except Exception as e:
                logger.warning(f"删除临时文件 {f} 失败: {str(e)}")
        os.rmdir(temp_dir)
    except Exception as e:
        logger.error(f"清理临时目录失败: {str(e)}")

def safe_replace_file(src, dst):
    """原子性替换文件"""
    try:
        # 尝试直接替换
        shutil.copy2(src, dst)
        os.unlink(src)
    except OSError:
        try:
            shutil.move(src, dst)  # 类似于 os.replace 但可能有不同的错误处理
        except PermissionError as e:
            print(f"无法移动文件: {e}")
            try:
                shutil.copy2(src, dst)
                os.unlink(src)
            except Exception as e:
                raise RuntimeError(f"无法替换文件 {dst}: {str(e)}")


# 辅助函数 - 安全获取用户目录
def get_user_upload_dir(user_id):
    upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

# 辅助函数 - 安全获取报告目录
def get_user_report_dir(user_id):
    report_dir = os.path.join(current_app.config['REPORT_FOLDER'], user_id)
    os.makedirs(report_dir, exist_ok=True)
    return report_dir