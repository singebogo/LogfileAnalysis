# from attr import attrs
from functools import partial

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm.attributes import flag_modified
from log_analyzer import create_app
import yappi
import logging
import hashlib
import os, uuid
import atexit
from datetime import timedelta
from apscheduler.events import EVENT_JOB_ERROR
from sqlalchemy import create_engine
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.jobstores.memory import MemoryJobStore
from concurrent.futures import ThreadPoolExecutor
from flask import copy_current_request_context
from flask import app, request, render_template, jsonify, session, send_file, Response, redirect, url_for
from werkzeug.security import safe_join
from werkzeug.utils import secure_filename
import sys
from log_analyzer.log_parser import LogParser
from log_analyzer.utils.combines import *
from log_analyzer.utils.files import get_report_files, merge_chunks
from log_analyzer.utils.cleans import cleanup_files, allowed_file
from log_analyzer.models import *
# 预加载模块（避免在请求处理中动态导入）
from log_analyzer.log_analyzer_template import process, save_log_entry

# 创建应用实例
from log_analyzer.utils.combines import down_sample_data, generate_time_plot, \
    generate_error_plot
from log_analyzer.servers import clean_history, get_user_files_from_db, handle_upload_chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = create_app(os.getenv('FLASK_CONFIG') or 'default')
app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'production')  # 回退默认值
app.debug = app.config['FLASK_ENV'] == 'development'

# 为调度器创建独立引擎（避免与Flask-SQLAlchemy冲突）
scheduler_engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])

# 初始化调度器
scheduler = BackgroundScheduler(
    jobstores={"default": MemoryJobStore()},  # ✅ 内存存储，不涉及 pickle
    daemon=True
)


def job_error_listener(event):
    global app
    app.logger.error(f"任务失败: {event.exception}")


def init_scheduler(app):
    """初始化定时任务"""
    if not scheduler.running:
        scheduler.add_job(
            cleanup_stale_uploads,
            'interval',
            minutes=30,  # 每30分钟运行一次
            args=[app]  # 传递app上下文
        )
        scheduler.start()
        scheduler.add_listener(job_error_listener, EVENT_JOB_ERROR)
        app.logger.info("APScheduler 定时任务已启动")


# 应用关闭时关闭调度器
def shutdown_scheduler(exception=None):
    app.logger.error(exception)
    if scheduler.running:
        scheduler.shutdown()
        app.logger.info("APScheduler 已关闭")


def handle_exception(exc_type, exc_value, exc_traceback):
    app.logger.error("发生未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))
    shutdown_scheduler()  # 确保调度器关闭
    sys.__excepthook__(exc_type, exc_value, exc_traceback)  # 调用默认异常处理


# 定时任务清理过期未完成的上传
def cleanup_stale_uploads(app):
    """清理过期上传的任务函数"""
    with app.app_context():
        # 在这里执行清理逻辑
        from log_analyzer.models import FileUpload, db
        from log_analyzer.servers import delete_chunks
        expiry_time = datetime.utcnow() - timedelta(hours=24)
        try:
            stale_uploads = FileUpload.query.filter(
                FileUpload.status == 'uploading',
                FileUpload.created_at < expiry_time
            ).all()

            for upload in stale_uploads:
                delete_chunks(upload.id, upload.user_id)
                db.session.delete(upload)

            db.session.commit()
        except Exception as e:
            app.logger.error(f"删除在途上传记录失败: {e}")


init_scheduler(app)
atexit.register(shutdown_scheduler)
sys.excepthook = handle_exception


@app.route('/')
@app.route('/index')
def index():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', app.config['REPORTS_PER_PAGE'], type=int)
    app.config.update({'REPORTS_PER_PAGE': per_page})
    search_text = request.args.get('search', '').lower()
    filter_type = request.args.get('type', 'all')

    # 优先从 cookie 获取 deviceID，其次从 header 获取
    deviceID = request.cookies.get('device_id', request.headers.get('X-Device-ID', ''))

    # 如果没有 DeviceID，重定向到带有查询参数的版本
    reports = []
    if deviceID:
        reports = get_report_files(deviceID)
        try:
            user = db.session.get(User, int(deviceID))
            if not user:
                new_user = User(id=int(deviceID))
                db.session.add(new_user)
                db.session.commit()
                UserConfig.ensure_user_configs(new_user.id)  # 初始化所有默认配置

        except Exception as e:
            app.logger.error(e)
    # 应用搜索和过滤
    if search_text or filter_type != 'all':
        filtered_reports = []
        for report in reports:
            name_match = not search_text or search_text in report['display_name'].lower()
            type_match = filter_type == 'all' or report['type'] == filter_type
            if name_match and type_match:
                filtered_reports.append(report)
        reports = filtered_reports

    total_reports = len(reports)

    # 分页处理
    start = (page - 1) * per_page
    end = start + per_page
    paginated_reports = reports[start:end]

    total_pages = (total_reports + per_page - 1) // per_page

    return render_template(
        'index.html',
        reports=paginated_reports,
        deviceID=deviceID,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_reports=total_reports
    )


@app.route('/settings')
def settings():
    return render_template('settings.html')


def get_device_id_from_cookie_or_generate(request):
    """
    Get device ID from cookie or generate a new one if not found.

    Args:
        request: Flask request object

    Returns:
        str: Device ID (either from cookie or newly generated)
    """
    # Try to get device ID from cookie
    device_id = request.cookies.get('device_id')

    if device_id:
        return device_id

    # If no cookie exists, generate a new device ID
    # Using a combination of user agent and IP address as the base
    user_agent = request.headers.get('User-Agent', '')
    remote_addr = request.headers.get('X-Forwarded-For', request.remote_addr)
    fingerprint_base = f"{user_agent}-{remote_addr}"

    # Create a SHA256 hash of the fingerprint
    device_id = hashlib.sha256(fingerprint_base.encode('utf-8')).hexdigest()

    return device_id


@app.route('/get_upload_config/<int:user_id>')
def get_upload_config(user_id):
    """动态返回上传配置（包括分片大小）"""
    try:
        chunk_size =  UserConfig.get_user_config(user_id, 'performance', 'batchUploadFiles_chunk_size')
    except Exception as e:
        chunk_size =  app.config['forword_trunk_cache_size']
    config = {
        # 从数据库或配置读取用户特定的分片大小
        'chunk_size': chunk_size,
        'max_retries': 3,
        'allowed_types': ['log', 'txt', 'csv']
    }
    return jsonify(config)

@app.route('/api/defaultconfigs/<int:user_id>')
def default_user_configs(user_id):
    try:
        user = db.session.get(User, user_id)
        if not user:
            new_user = User(id=int(user_id))
            db.session.add(new_user)
            db.session.commit()
            UserConfig.ensure_user_configs(new_user.id)  # 初始化所有默认配置
            return jsonify({'status': 'success', 'data': "默认配置成功"})
        else:
            return jsonify({'status': 'success', 'data': "已配置"})
    except Exception as e:
        return jsonify({'status': 'failed', 'error': e})


@app.route('/api/configs/<int:user_id>')
def get_user_configs(user_id):
    # 检查用户是否存在
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404

    # 获取用户所有配置
    configs = UserConfig.query.filter_by(user_id=int(user_id)).all()

    # 转换为前端需要的格式
    config_groups = {}
    for config in configs:
        if config.config_type not in config_groups:
            config_groups[config.config_type] = {}

        # 根据数据类型解析值
        if config.data_type == 'json':
            value = json.loads(config.value)
        elif config.data_type == 'number':
            value = float(config.value)
        else:
            value = config.value

        config_groups[config.config_type][config.key] = {
            'value': value,
            'dataType': config.data_type,
            'label': config.key.replace('_', ' ').title()
        }

    # 合并默认配置（如果某些配置不存在）
    for config_type, items in DEFAULT_CONFIGS.items():
        if config_type not in config_groups:
            config_groups[config_type] = {}

        for key, default_config in items.items():
            if key not in config_groups[config_type]:
                config_groups[config_type][key] = {
                    'value': json.loads(default_config['value']) if default_config['data_type'] == 'json' else
                    default_config['value'],
                    'dataType': default_config['data_type'],
                    'label': default_config.get('label', key.replace('_', ' ').title()),
                    'isDefault': True  # 标记为默认值
                }

    return jsonify(config_groups)


@app.route('/api/configs/<int:user_id>', methods=['POST'])
def update_configs(user_id):
    data = request.json
    for config_type, items in data.items():
        for key, config in items.items():
            # 确定数据类型
            if isinstance(config['value'], (list, dict)):
                value = json.dumps(config['value'])
                data_type = 'json'
            else:
                value = str(config['value'])
                data_type = config.get('dataType', 'string')

            # 更新或创建配置
            existing = UserConfig.query.filter_by(
                user_id=int(user_id),
                config_type=config_type,
                key=key
            ).first()

            if existing:
                existing.value = value
                existing.data_type = data_type
            else:
                db.session.add(UserConfig(
                    user_id=int(user_id),
                    config_type=config_type,
                    key=key,
                    value=value,
                    data_type=data_type
                ))

    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/delete-file', methods=['POST'])
def delete_file():
    try:
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({"status": "error", "message": "未指定文件名"}), 400

        # 安全处理文件名
        filename = secure_filename(filename)
        upload_dir = os.path.abspath(app.config['REPORT_FOLDER'])
        filepath = os.path.normpath(os.path.join(upload_dir, filename))

        # 安全检查：防止目录遍历
        if not filepath.startswith(upload_dir):
            return jsonify({"status": "error", "message": "非法文件路径"}), 400

        # 调试输出
        app.logger.debug(f"尝试删除文件: {filepath}")
        app.logger.debug(f"上传目录内容: {os.listdir(upload_dir)}")

        if not os.path.exists(filepath):
            return jsonify({
                "status": "error",
                "message": "文件不存在",
                "debug": {
                    "requested_file": filename,
                    "resolved_path": filepath,
                    "existing_files": os.listdir(upload_dir)
                }
            }), 404

        os.remove(filepath)
        return jsonify({
            "status": "success",
            "message": f"文件 {filename} 已成功删除"
        })

    except Exception as e:
        app.logger.error(f"删除文件错误: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"删除文件时出错: {str(e)}"
        }), 500


@app.route('/cleanup-uploads', methods=['GET', 'POST'])
def cleanup_uploads():
    if request.method == 'POST':
        try:
            # 1. 获取或生成用户唯一标识（机器码或session ID）
            user_id = request.headers.get('X-Device-ID')
            user_report_dir = os.path.join(app.config['REPORT_FOLDER'], user_id)
            user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
            cleanup_files(user_report_dir, app.config['CLEAR_TIME'])
            deleted = cleanup_files(user_upload_dir)
            # 清理数据记录
            clean_history()
            return jsonify({
                "status": "success",
                "message": f"成功删除 {len(deleted)} 个文件",
                "deleted_files": deleted
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500

    # 如果是GET请求，返回清理页面
    return render_template('cleanup.html')


@app.route('/upload-stats')
def get_upload_stats():
    try:
        total_files = 0
        old_files = 0
        now = datetime.now()
        user_id = request.headers.get('X-Device-ID')
        user_report_dir = os.path.join(app.config['REPORT_FOLDER'], user_id)
        # user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        for filename in user_report_dir:
            filepath = os.path.join(user_report_dir, filename)
            if os.path.isfile(filepath):
                total_files += 1
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if now - file_mtime > timedelta(days=app.config['CLEAR_TIME']):
                    old_files += 1

        return jsonify({
            "status": "success",
            "total_files": total_files,
            "old_files": old_files
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    """
    处理文件分块上传
    返回:
        - 200: 分块上传成功
        - 400: 客户端错误(缺少参数/无效参数)
        - 413: 文件过大
        - 500: 服务器内部错误
    """
    global batch_id
    try:
        # 1. 验证必要参数
        required_fields = ['fileId', 'chunkIndex', 'totalChunks']
        if not all(field in request.form for field in required_fields):
            return jsonify({
                'status': 'error',
                'message': f'缺少必要参数，需要: {", ".join(required_fields)}'
            }), 400, {'Content-Type': 'application/json'}

        # 1. 获取或生成用户唯一标识（机器码或session ID）
        user_id = request.headers.get('X-Device-ID')

        # 2. 验证文件是否存在
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '未接收到文件数据'
            }), 400, {'Content-Type': 'application/json'}

        # 3. 获取并验证参数
        try:
            file_id = secure_filename(request.form['fileId'])
            upload_record = db.session.get(FileUpload, file_id)

            if not upload_record:
                upload_record = FileUpload(
                    id=file_id,
                    user_id=user_id,
                    filename=request.form.get('fileName'),
                    total_chunks=int(request.form['totalChunks']),
                    received_chunks=[]
                )
                db.session.add(upload_record)

            chunk_index = int(request.form['chunkIndex'])
            if chunk_index not in upload_record.received_chunks:
                upload_record.received_chunks.append(chunk_index)
                flag_modified(upload_record, "received_chunks")  # 关键行
                db.session.commit()

            total_chunks = int(request.form['totalChunks'])
            chunk = request.files['file']

            if chunk_index < 0 or total_chunks <= 0 or chunk_index >= total_chunks:
                return jsonify({
                    'status': 'error',
                    'message': '无效的分块参数'
                }), 400, {'Content-Type': 'application/json'}

        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': f'参数格式错误: {str(e)}'
            }), 400, {'Content-Type': 'application/json'}

        # 4. 初始化session中的上传状态(如果不存在)
        if 'uploads' not in session:
            session['uploads'] = {}
        if 'filenames' not in session:
            session['filenames'] = []

        # 4. 检查文件大小限制 (示例: 最大100MB)
        max_chunk_size = UserConfig.get_user_config(user_id, 'performance', 'max_chunk_size')  # 1MB默认缓冲区
        if chunk.content_length > max_chunk_size:
            return jsonify({
                'status': 'error',
                'message': f'分块大小超过限制({max_chunk_size}字节)'
            }), 413, {'Content-Type': 'application/json'}

        # 5. 创建用户专属上传目录结构
        # 格式: UPLOAD_FOLDER/{user_id}/{file_id}/
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        temp_dir = os.path.join(user_upload_dir, file_id)
        try:
            os.makedirs(temp_dir, exist_ok=True)
            # 设置目录权限（确保安全）
            os.chmod(user_upload_dir, 0o755)
            os.chmod(temp_dir, 0o755)
        except OSError as e:
            logger.error(f"创建临时目录失败: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': '服务器存储错误'
            }), 500, {'Content-Type': 'application/json'}

        # 如果是第一个分块，保存元数据
        if chunk_index == 0:
            metadata = {
                'filename': request.form.get('fileName'),
                'filesize': int(request.form.get('fileSize')),
                'mimetype': request.files['file'].content_type,
                'upload_date': datetime.utcnow().isoformat(),
                'chunk_size': int(
                    request.form.get('chunkSize', UserConfig.get_user_config(user_id, 'performance', 'max_chunk_size')))
            }
            with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)

        # 6. 保存分块文件
        chunk_filename = os.path.join(temp_dir, f'chunk_{chunk_index}')
        try:
            chunk.save(chunk_filename)
            logger.info(f"成功保存分块 {chunk_index}/{total_chunks - 1} 到 {chunk_filename}")
        except IOError as e:
            logger.error(f"保存分块文件失败: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': '文件保存失败'
            }), 500, {'Content-Type': 'application/json'}

        # 7. 如果是最后一块，合并文件
        if chunk_index == total_chunks - 1:
            try:
                # 验证是否所有分块都存在
                existing_chunks = set()
                for f in os.listdir(temp_dir):
                    if f.startswith('chunk_'):
                        try:
                            existing_chunks.add(int(f.split('_')[1]))
                        except (IndexError, ValueError):
                            continue

                expected_chunks = set(range(total_chunks))
                if existing_chunks != expected_chunks:
                    missing = expected_chunks - existing_chunks
                    return jsonify({
                        'status': 'error',
                        'message': f'分块不完整，缺少分块: {missing}'
                    }), 400, {'Content-Type': 'application/json'}

                # 检查是否完成
                if len(upload_record.received_chunks) == upload_record.total_chunks:
                    try:
                        # 合并文件
                        output_filename = merge_chunks(temp_dir, user_upload_dir, file_id)
                        logger.info(f"文件合并完成: {output_filename}")
                        # name = os.path.basename(output_filename)
                        # 检查是否已有batch_id（来自前端或session）
                        batch_id = request.form.get('batchId')
                        handle_upload_chunk(batch_id, user_id)

                        upload_record.status = 'completed'
                        upload_record.final_path = output_filename
                        upload_record.batch_id = batch_id  # 关联上传记录和批次
                        db.session.commit()
                    except Exception as e:
                        logger.error(f"检查是否完成失败: {str(e)}", exc_info=True)
                    return jsonify({
                        'status': 'success',
                        'chunkReceived': chunk_index,
                        'totalChunks': total_chunks,
                        'batchId': batch_id,  # 返回batch_id
                    }), 200, {'Content-Type': 'application/json'}

            except Exception as e:
                logger.error(f"文件合并失败: {str(e)}", exc_info=True)
                session.clear()
                return jsonify({
                    'status': 'error',
                    'message': '文件合并失败'
                }), 500, {'Content-Type': 'application/json'}

        return jsonify({
            'status': 'success',
            'chunkReceived': chunk_index,
            'totalChunks': total_chunks,
            'batchId': None,  # 返回batch_id
        }), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        logger.error(f"上传处理意外错误: {str(e)}", exc_info=True)
        session.clear()
        return jsonify({
            'status': 'error',
            'message': '服务器内部错误'
        }), 500, {'Content-Type': 'application/json'}


@app.route('/upload-status', methods=['GET'])
def upload_status():
    file_id = request.args.get('fileId')
    user_id = request.headers.get('X-Device-ID')
    user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    temp_dir = os.path.join(user_upload_dir, file_id)

    if not os.path.exists(temp_dir):
        return jsonify({'uploadedChunks': []})

    uploaded = [int(f.split('_')[1]) for f in os.listdir(temp_dir)]
    return jsonify({'uploadedChunks': uploaded})


@app.route('/upload_files', methods=['POST'])
def upload_files():
    # 串联上传
    global files
    try:
        session.clear()
        session['filenames'] = []  # 明确初始化filenames列表
        session['analysis_complete'] = False
        session['file_metadata'] = {}  # 使用字典存储更可靠

        # 1. 获取或生成用户唯一标识（机器码或session ID）
        user_id = request.headers.get('X-Device-ID')

        if 'files' not in request.files:
            return jsonify({'status': 'error', 'message': 'No files provided'}), 400

        needfiles = request.files.getlist('files')
        if not needfiles or all(file.filename == '' for file in needfiles):
            return jsonify({'status': 'error', 'message': 'No needfiles selected'}), 400

        file_data = []
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        for file in needfiles:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(user_upload_dir, filename)
                file.save(filepath)

                try:
                    parser = LogParser(filepath)

                    # 获取所有需要的数据，确保都是基本类型
                    stats = parser.get_statistics()
                    error_logs = parser.get_error_logs()
                    time_distribution = parser.get_logs_by_time()  # 获取字典
                    error_distribution = parser.get_error_distribution()  # 获取字典

                    # 生成图表（传递字典而不是parser对象）
                    time_plot = generate_time_plot(time_distribution)
                    error_plot = generate_error_plot(error_distribution)

                    if stats['total_entries'] == 0:
                        app.logger.warning(f"Empty or invalid log file: {filename}")
                        continue

                    file_data.append({
                        'filename': filename,
                        'stats': stats,
                        'error_logs': error_logs,
                        'time_plot': time_plot,
                        'error_plot': error_plot,
                        'time_distribution': time_distribution,  # 存储字典
                        'error_distribution': error_distribution  # 存储字典
                    })
                    session['file_metadata'][filename] = {
                        'time_distribution': parser.get_logs_by_time(),
                        'error_distribution': parser.get_error_distribution(),
                        'timestamp': datetime.now().isoformat()
                    }
                    session['filenames'].append(filename)

                    app.logger.info(f"Stored metadata for {filename}")


                except Exception as e:
                    app.logger.error(f"Error processing {filename}: {str(e)}")
                    continue

        if not session['filenames']:
            return jsonify({'status': 'error', 'message': 'No valid files processed'}), 400

        if not file_data:
            return jsonify({'status': 'error', 'message': 'No valid log files found'}), 400

        # 存储分析结果到session
        # session['file_data'] = file_data
        # session['combined_stats'] = get_combined_stats(file_data)
        session['analysis_complete'] = True

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        session.clear()
        app.logger.error(f"Upload failed: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/reports/<deviceID>/<filename>')
def serve_report(filename, deviceID):
    # 安全验证
    if not filename or '..' in filename or filename.startswith('/'):
        return "非法文件名", 400
    user_id = deviceID
    user_report_dir = os.path.join(app.config['REPORT_FOLDER'], user_id)
    filepath = safe_join(user_report_dir, filename)

    if not os.path.exists(filepath):
        return "文件不存在", 404

    # 获取文件信息
    ext = os.path.splitext(filename)[1][1:].lower()  # 获取不带点的扩展名

    # 可内联显示的文件类型
    inline_types = {
        'pdf': 'application/pdf',
        'html': 'text/html',
        'htm': 'text/html',
        'txt': 'text/plain',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif'
    }

    if ext in inline_types:
        return send_file(
            filepath,
            mimetype=inline_types[ext],
            conditional=True
        )
    else:
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename
        )


@app.route('/results', methods=['POST', 'GET'])
def show_results():
    try:
        required_fields = ['device_id', 'file_ids', 'batch_id']
        if not all(field in request.form for field in required_fields):
            app.logger.warning(f"No {required_fields} found for : {request.form}")
            return redirect(url_for('index'))

        user_id = request.form.get('device_id')
        batchId = request.form.get('batch_id')
        app.logger.info("show_results日志分析...")

        try:
            file_ids = json.loads(request.form['file_ids']).get('file_ids', {})
        except json.JSONDecodeError as e:
            app.logger.error(f"Invalid JSON in file_ids: {e}")
            return redirect(url_for('index'))

        # 从数据库或会话中获取文件列表（避免使用全局变量）
        file_records = get_user_files_from_db(user_id, file_ids)
        if not file_records:
            app.logger.warning(f"No files found for user: {user_id}")
            return redirect(url_for('index'))

        user_report_dir = os.path.join(app.config['REPORT_FOLDER'], user_id)
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)

        try:
            os.makedirs(user_report_dir, exist_ok=True)
            # 设置目录权限（确保安全）
            os.chmod(user_report_dir, 0o755)
        except OSError as e:
            logger.error(f"创建临时目录失败: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': '服务器存储错误'
            }), 500, {'Content-Type': 'application/json'}

        # 验证并收集有效文件
        valid_files = []
        for record in file_records:
            file_path = os.path.join(user_upload_dir, record.filename)
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                app.logger.warning(f"File not found: {record.filename}")

        if not valid_files:
            app.logger.error("No valid files to process")
            return redirect(url_for('index'))

        # 5. 处理文件
        # from log_analyzer.log_analyzer_template import process, save_log_entry
        try:
            result = process(valid_files, user_report_dir, user_id, batch_id)
        except Exception as e:
            app.logger.error(f"Error processing files: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

        # # 创建并启动后台线程
        # @copy_current_request_context
        # def background_task():
        #     with app.app_context():
        #         try:
        #             save_log_entry(valid_files, result['batch_id'], UserConfig.get_user_config(user_id, 'performance', 'batch_size'))
        #         except Exception as e:
        #             app.logger.error(f"存储日志文件失败{e}")
        #         finally:
        #             # 清理app.config['REPORT_FOLDER']
        #             cleanup_files(user_report_dir)
        #
        # # 使用线程池执行
        # executor = ThreadPoolExecutor(max_workers=1)
        # executor.submit(background_task)
        # executor.shutdown(wait=False)

        # 7. 后台任务处理（优化线程池使用）
        def run_background_task(app_context, files, batch_id, user_id, report_dir):
            with app_context:
                try:
                    batch_size = UserConfig.get_user_config(
                        user_id, 'performance', 'batch_size'
                    )
                    save_log_entry(files, batch_id, batch_size)
                except Exception as e:
                    app.logger.error(f"Log save failed: {e}", exc_info=True)
                finally:
                    cleanup_files(report_dir)

        # 使用partial预先绑定参数
        task = partial(
            run_background_task,
            app.app_context(),
            valid_files,
            batchId,
            user_id,
            user_report_dir
        )

        # 使用with管理线程池
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(task)

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        app.logger.error(f"Unexpected error in show_results: {str(e)}")
        return jsonify({'status': 'error', 'message': '内部服务器错误'}), 500
    finally:
        session.clear()
    # try:

    #         parser = LogParser(filepath)
    #         file_data.append({
    #             'filename': filename,
    #             'stats': parser.get_statistics(),
    #             'error_logs': parser.get_error_logs(limit=20),
    #             'time_distribution': file_info['time_distribution'],
    #             'error_distribution': file_info['error_distribution'],
    #             'time_plot': generate_time_plot(file_info['time_distribution']),
    #             'error_plot': generate_error_plot(file_info['error_distribution'])
    #         })
    #     except Exception as e:
    #         app.logger.error(f"Error processing {filename}: {str(e)}")
    #         continue
    #
    # if not file_data:
    #     app.logger.error("No valid files could be processed")
    #     return redirect(url_for('index'))
    #
    # # 生成合并图表
    # combined_stats = get_combined_stats(file_data)
    # combined_time_plot = generate_combined_time_plot(file_data)
    # combined_error_plot = generate_combined_error_plot(file_data)
    #
    # return render_template('results.html',
    #                        file_data=file_data,
    #                        combined_stats=combined_stats,
    #                        combined_time_plot=combined_time_plot,
    #                        combined_error_plot=combined_error_plot)


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API接口，支持多文件分析"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400

    user_id = request.headers.get('X-Device-ID')
    user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    file_data = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(user_upload_dir, filename)
            file.save(filepath)

            parser = LogParser(filepath)
            file_data.append({
                'filename': filename,
                'statistics': parser.get_statistics(),
                'error_logs': parser.get_error_logs(),
                'time_distribution': parser.get_logs_by_time(),
                'error_distribution': parser.get_error_distribution()
            })

    if not file_data:
        return jsonify({'error': 'No valid files uploaded'}), 400

    combined_stats = get_combined_stats(file_data)

    return jsonify({
        'files': file_data,
        'combined_statistics': combined_stats
    })


@app.route('/stream_analyze', methods=['POST'])
def stream_analyze():
    """流式处理大文件分析"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400
    user_id = request.headers.get('X-Device-ID')

    def generate(user_id):
        yield '{"files": ['

        first_file = True
        for file in files:
            if file and allowed_file(file.filename):
                if not first_file:
                    yield ','
                first_file = False

                filename = secure_filename(file.filename)

                user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
                filepath = os.path.join(user_upload_dir, filename)
                file.save(filepath)

                parser = LogParser(filepath)
                result = {
                    'filename': filename,
                    'statistics': parser.get_statistics(),
                    'error_sample': parser.get_error_logs(limit=app.config['MAX_ERRORS_DISPLAY']),
                    'time_distribution': down_sample_data(parser.get_logs_by_time()),
                    'error_distribution': parser.get_error_distribution()
                }

                yield json.dumps(result)

        yield ']}'

    return Response(generate(user_id), mimetype='application/json')


@app.route('/preview-report/<filename>')
def preview_report(filename):
    """预览报告文件（带大小检查）"""
    MAX_PREVIEW_SIZE = app.config['MAX_CONTENT_LENGTH']  # 2MB限制

    # 安全检查
    if not filename or '..' in filename or filename.startswith('/'):
        return jsonify({"status": "error", "message": "非法文件名"}), 400

    user_id = request.headers.get('X-Device-ID')
    report_dir = os.path.join(app.config['REPORT_FOLDER'], user_id)
    filepath = safe_join(report_dir, filename)

    if not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "报告文件不存在"}), 404

    # 检查文件类型
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.html', '.htm']:
        return jsonify({"status": "error", "message": "不支持预览此文件类型"}), 400

    # 检查文件大小
    file_size = os.path.getsize(filepath)
    # 确保所有返回都是JSON格式
    if file_size > MAX_PREVIEW_SIZE:
        return jsonify({
            "status": "oversize",
            "message": "文件过大，请下载查看",
            "size": file_size,
            "deviceID": user_id,
            "max_size": MAX_PREVIEW_SIZE,
            "download_url": url_for('serve_report', filename=filename, deviceID=user_id)
        })

    # 读取并清理HTML内容
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用BeautifulSoup清理HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')

        # # 移除潜在的危险标签
        # for tag in soup(['script', 'iframe', 'frame', 'object', 'embed', 'link']):
        #     tag.decompose()
        #
        # # 移除危险属性
        # for tag in soup():
        #     for attr in ['onclick', 'onload', 'onerror', 'onmouseover']:
        #         if attr in tag.attrs:
        #             del tag[attrs]

        # 添加安全限制的CSS
        style = soup.new_tag('style')
        style.string = """
            iframe, object, embed { display: none !important; }
            body { max-width: 100% !important; overflow-x: hidden !important; }
        """
        soup.head.append(style)

        # 成功返回也要包装成JSON
        return jsonify({
            "status": "success",
            "content": str(soup),  # 清理后的HTML内容
            "filename": filename,
            "deviceID": user_id,
            "download_url": url_for('serve_report', filename=filename, deviceID=user_id)
        })
    except Exception as e:
        app.logger.error(f"预览失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"预览失败: {str(e)}",
            "download_url": url_for('serve_report', filename=filename, deviceID=user_id)
        }), 500


@app.route('/history')
def analysis_history():
    """查看历史分析记录"""
    # Query all batches ordered by start_time (newest first)
    user_id = request.headers.get('X-Device-ID')

    batches = AnalysisBatch.query.order_by(AnalysisBatch.start_time.desc()).all()

    # Convert SQLAlchemy objects to dictionaries
    batches_list = []
    for batch in batches:
        batch_dict = {
            'id': batch.id,
            'user_id': user_id,
            'total_files': batch.total_files,
            'start_time': batch.start_time.isoformat() if batch.start_time else None,
            'end_time': batch.end_time.isoformat() if batch.end_time else None,
            'status': batch.status,
            'batch_id': batch.batch_id,
            'total_lines': batch.total_lines,
            'total_errors': batch.total_errors,
            'total_warnings': batch.total_warnings,
            'report_path': batch.report_path,
        }
        batches_list.append(batch_dict)

    return jsonify({
        'success': True,
        'count': len(batches_list),
        'batches': batches_list
    })


@app.route('/get_log_details')
def get_log_details():
    batch_id = request.args.get('batch_id')
    file_id = request.args.get('file_id')
    show_warnings = request.args.get('show_warnings', 'true') == 'true'
    show_errors = request.args.get('show_errors', 'true') == 'true'

    exceptions, errors = [{}], [{}]
    isAll = False
    """查看特定批次的详细结果"""
    batch = AnalysisBatch.query.filter_by(batch_id=batch_id).first_or_404()

    if not show_errors and not show_warnings:
        isAll = True
    # 获取异常统计
    if show_warnings or isAll:
        exceptions = db.session.query(
            LogException.exception_type,
            LogException.count,
            LogException.first_occurrence,
            LogException.last_occurrence,
            LogException.source_file,
            LogException.example_message,
            LogException.line_number
        ).filter_by(batch_id=batch_id).all()

    if show_errors or isAll:
        # 获取异常统计
        errors = db.session.query(
            LogError.exception_type,
            LogError.message,
            LogError.keyword,
            LogError.line_number,
            LogError.source_file,
            LogError.raw_line,
            LogError.timestamp
        ).filter_by(batch_id=batch_id).all()

    # 获取文件统计
    files = AnalyzedFile.query.filter_by(batch_id=batch_id).all()

    def formatTime(time):
        try:
            if time != '未知时间' and time:
                if '.' in time:
                    occurrence = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')
                else:
                    occurrence = datetime.strptime(time, '%Y-%m-%d %H:%M:%S %f')
            else:
                occurrence = None
        except:
            occurrence = None
        return occurrence

    return jsonify({
        'success': True,
        'batchId': batch.id,
        "exceptions": [{
            "type": e.exception_type,
            "first_occurrence": formatTime(e.first_occurrence),
            "last_occurrence": formatTime(e.last_occurrence),
            "count": e.count,
            "line_number": e.line_number,
            "message": e.example_message,
            "source": e.source_file
        } for e in exceptions],
        'errors': [{
            "timestamp": formatTime(e.timestamp),
            "message": e.message,
            "source": e.source_file,
            "keyword": e.keyword,
            "line_number": e.line_number,
            "raw_line": e.raw_line  # Individual occurrences
        } for e in errors],
        'files': [{
            "batch_id": e.batch_id,
            "file_name": e.file_name,
            "total_lines": e.total_lines,
            "error_count": e.error_count,  # Individual occurrences
            "warning_count": e.warning_count,  # Individual occurrences
            "keyword_count": e.keyword_count,  # Individual occurrences
            "analysis_time": formatTime(e.analysis_time)  # Individual occurrences
        } for e in files],
    })


@app.route('/batch-check-uploads', methods=['POST'])
def batch_check_uploads():
    data = request.get_json()
    user_id = request.headers.get('X-Device-ID')
    file_ids = data.get('file_ids', [])

    results = {}
    uploads = FileUpload.query.filter(
        FileUpload.id.in_(file_ids),
        FileUpload.user_id == user_id
    ).all()

    for upload in uploads:
        results[upload.id] = {
            "status": upload.status,
            "filename": upload.filename,
            "progress": f"{len(upload.received_chunks)}/{upload.total_chunks}"
        }

    # 找出不存在的文件
    missing_ids = set(file_ids) - {u.id for u in uploads}
    for fid in missing_ids:
        results[fid] = {"status": "error", "message": "not_found"}

    return jsonify(results)


@app.route('/api/parse-progress/<batch_id>')
def get_parse_progress(batch_id):
    try:
        batch = AnalysisBatch.query.filter_by(batch_id=batch_id).first_or_404()
        if not batch:
            return jsonify({'status': 'error', 'message': 'Batch not found'}), 404

        # Get all files in this batch
        files = AnalyzedFile.query.filter_by(batch_id=batch_id).all()

        # Calculate overall progress
        total_files = len(files)
        completed_files = sum(1 for f in files if f.status == 'completed')
        progress = 0
        if total_files > 0:
            progress = int((completed_files / total_files) * 100)

        # Prepare file progress data
        file_progress = []
        current_file = None
        for file in files:
            file_progress.append({
                'filename': file.file_name,
                'progress': file.progress,
                'batch_id': file.batch_id,
                'processed_lines': file.processed_lines,
                'status': file.status,
                'error': file.error_message if file.status == 'failed' else None
            })
            if file.status == 'parsing':
                current_file = file.file_name

        return jsonify({
            'status': 'success',
            'progress': progress,
            'processed_files': completed_files,
            'total_files': total_files,
            'current_file': current_file,
            'files': file_progress,
            'batch_status': batch.status
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
