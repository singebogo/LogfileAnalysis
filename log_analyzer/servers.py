import os, shutil
import time, re
from copy import deepcopy
from datetime import timedelta
# from log_analyzer import create_app
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
from random import random

from flask import current_app, Flask, jsonify
from pathlib import Path
from sqlalchemy import delete
from sqlalchemy.orm import scoped_session, sessionmaker
import logging
from threading import Lock
from contextlib import contextmanager

from . import create_app
from .models import *

logger = logging.getLogger(__name__)
# Create a lock for thread-safe database operations
progress_lock = Lock()

# 数据库会话工厂
def create_session_factory(db_engine):
    return scoped_session(sessionmaker(bind=db_engine, expire_on_commit=False))


@contextmanager
def db_session_scope(db_engine):
    """提供事务范围的数据库会话"""
    session_factory = create_session_factory(db_engine)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"数据库操作失败: {str(e)}", exc_info=True)
        raise
    finally:
        session_factory.remove()


@contextmanager
def create_analysis_session(log_files, user_id, batch_id):
    with current_app.app_context():
        """创建或更新分析会话上下文"""
        # 检查是否已存在该 batch_id 的记录
        batch = AnalysisBatch.query.filter_by(batch_id=batch_id).first()
        if batch is None:
            # 如果不存在，创建新记录
            batch = AnalysisBatch(
                batch_id=batch_id,
                start_time=datetime.utcnow(),
                total_files=len(log_files),
                status='running',
                user_id=user_id,
            )
            db.session.add(batch)
        else:
            # 如果已存在，更新记录
            batch.start_time = datetime.utcnow()
            batch.total_files = len(log_files)
            batch.status = 'running'
            batch.user_id = user_id

        db.session.commit()

        try:
            yield batch
            batch.status = 'completed'
            batch.end_time = datetime.utcnow()
        except Exception as e:
            batch.status = 'failed'
            batch.end_time = datetime.utcnow()
            raise
        finally:
            db.session.commit()
            db.session.close()


def save_exceptions_to_db(batch_id, file_results, EXCEPTION_PATTERNS, batch_size):
    """保存异常数据到数据库 file_results['exception_details']"""
    config = deepcopy(current_app.config)

    def insert_exceptions(file):
        app = Flask(__name__)
        app.config.update(config)
        # 重新初始化扩展
        db.init_app(app)

        with app.app_context():
            Session = scoped_session(sessionmaker(bind=db.engine))
            local_session = Session()
            try:
                # 每次循环一个文件
                exceptions_data = file['exception_details']
                count = 0
                exps = EXCEPTION_PATTERNS.values()
                for exp in exps:
                    for detail in exceptions_data[exp]:
                        if not detail:
                            continue

                        '''
                             error_detail = {
                                'source': results['filename'],
                                'line_number': line_number,
                                'time': error_time,
                                'keyword': keyword,
                                'message': line.strip(),
                                'exception': exception_type,
                        }
                        '''
                        try:
                            if detail['time'] != '未知时间':
                                if '.' in detail['time']:
                                    first_occurrence = datetime.strptime(detail['time'], '%Y-%m-%d %H:%M:%S.%f')
                                else:
                                    first_occurrence = datetime.strptime(detail['time'], '%Y-%m-%d %H:%M:%S %f')
                            else:
                                first_occurrence = None
                        except:
                            first_occurrence = None

                        try:
                            if detail['time'] != '未知时间':
                                if '.' in detail['time']:
                                    last_occurrence = datetime.strptime(detail['time'], '%Y-%m-%d %H:%M:%S.%f')
                                else:
                                    last_occurrence = datetime.strptime(detail['time'], '%Y-%m-%d %H:%M:%S %f')
                            else:
                                last_occurrence = None
                        except:
                            last_occurrence = None

                        example = detail['message'][:1000]  # 截取前500字符作为示例

                        exc_record = LogException(
                            batch_id=batch_id,
                            exception_type=detail['exception'],
                            count=len(example),
                            first_occurrence=first_occurrence,
                            last_occurrence=last_occurrence,
                            source_file=detail['source'],
                            example_message=example,
                            line_number=detail['line_number']
                        )
                        local_session.add(exc_record)
                        count += 1
                        # 达到批处理大小时提交
                        if count % batch_size == 0:
                            local_session.commit()
                            local_session.expunge_all()  # 清除已提交对象
                            count = 0

                        # 提交剩余记录
                    if count % batch_size != 0:
                        local_session.commit()

            except Exception as e:
                local_session.rollback()
                logger.error(f"线程 {threading.current_thread().name} insert_exceptions插入失败: {e}")
                raise  # 重新抛出异常以便线程池捕获
            finally:
                Session.remove()

    try:
        with ThreadPoolExecutor(max_workers=len(file_results)) as executor:
            executor.map(lambda f: insert_exceptions(f), file_results)
    except Exception as e:
        logger.error(
            f"线程 {threading.current_thread().name} ThreadPoolExecutor - save_exceptions_to_db: {e}")


def save_errors_to_db(batch_id, file_results, batch_size):
    """保存错误数据到数据库（分批插入优化版）

    Args:
        batch_id: 批次ID
        file_results: 文件结果列表
        batch_size: 每批插入的记录数（默认100）
    """
    config = deepcopy(current_app.config)

    def insert_errors(file):
        app = Flask(__name__)
        app.config.update(config)

        # 重新初始化扩展
        db.init_app(app)
        with app.app_context():
            Session = scoped_session(sessionmaker(bind=db.engine))
            local_session = Session()
            try:
                errors_data = file['error_details']
                count = 0

                for error in errors_data:
                    try:
                        if error['time'] != '未知时间':
                            if '.' in error['time']:
                                error_time = datetime.strptime(error['time'], '%Y-%m-%d %H:%M:%S.%f')
                            else:
                                error_time = datetime.strptime(error['time'], '%Y-%m-%d %H:%M:%S %f')
                        else:
                            error_time = None
                    except:
                        error_time = None

                    error_record = LogError(
                        batch_id=batch_id,
                        timestamp=error_time,
                        source_file=error['source'],
                        line_number=error['line_number'],
                        keyword=error['keyword'],
                        # exception_type=error['exception'],
                        message=error['message'][:2000],
                        raw_line=error['line_number']
                    )
                    local_session.add(error_record)
                    count += 1

                    # 达到批处理大小时提交
                    if count % batch_size == 0:
                        local_session.commit()
                        local_session.expunge_all()  # 清除已提交对象
                        count = 0

                # 提交剩余记录
                if count % batch_size != 0:
                    local_session.commit()

            except Exception as e:
                local_session.rollback()
                logger.error(f"线程 {threading.current_thread().name} insert_errors 插入失败: {e}")
                raise  # 重新抛出异常以便线程池捕获
            finally:
                Session.remove()  # 清理线程局部会话

    try:
        # 限制线程数，每个线程会创建自己的应用上下文
        with ThreadPoolExecutor(max_workers=min(4, len(file_results))) as executor:
            # 将app对象传递给每个线程
            executor.map(lambda f: insert_errors(f), file_results)
    except Exception as e:
        logger.error(f"线程 {threading.current_thread().name} ThreadPoolExecutor - save_errors_to_db: {e}")


def save_file_stats_to_db(batch_id, file_results, REPORT_FOLDER):
    """保存文件统计信息到数据库，包括解析进度信息

    Args:
        batch_id: 批次ID
        file_results: 文件分析结果列表
        REPORT_FOLDER: 报告存储目录
    """
    with current_app.app_context():
        try:
            batch = AnalysisBatch.query.get(batch_id)
            if not batch:
                logger.error(f"批次ID {batch_id} 不存在")
                return

            # 获取已存在的文件记录，避免重复创建
            existing_files = {f.file_name: f for f in AnalyzedFile.query.filter_by(batch_id=batch_id).all()}

            for file in file_results:
                filename = file['filename']

                if filename in existing_files:
                    # 更新现有记录
                    file_record = existing_files[filename]
                    file_record.total_lines = file['stats']['lines']
                    file_record.error_count = file['stats']['errors']
                    file_record.warning_count = file['stats']['warnings']
                    file_record.keyword_count = file['stats']['keywords']
                    file_record.status = 'completed'
                    file_record.progress = 100
                    file_record.end_time = datetime.now()
                else:
                    # 创建新记录
                    file_record = AnalyzedFile(
                        batch_id=batch_id,
                        file_path=os.path.join(REPORT_FOLDER, filename),
                        file_name=filename,
                        total_lines=file['stats']['lines'],
                        error_count=file['stats']['errors'],
                        warning_count=file['stats']['warnings'],
                        keyword_count=file['stats']['keywords'],
                        status='completed',
                        progress=100,
                        start_time=batch.start_time,
                        end_time=datetime.now(),
                        analysis_time=(datetime.now() - batch.start_time).total_seconds()
                    )
                    db.session.add(file_record)

                # 记录关键词统计
                # for keyword, count in file['keyword_counts'].items():
                #     keyword_record = FileKeyword(
                #         file_id=file_record.id,
                #         keyword=keyword,
                #         count=count
                #     )
                #     db.session.add(keyword_record)
                #
                # # 记录异常统计
                # for exception, count in file['exception_counts'].items():
                #     exception_record = FileException(
                #         file_id=file_record.id,
                #         exception_type=exception,
                #         count=count
                #     )
                #     db.session.add(exception_record)

            db.session.commit()
            logger.info(f"成功保存 {len(file_results)} 个文件的统计信息")

        except Exception as e:
            logger.error(f"保存文件统计信息失败: {str(e)}", exc_info=True)
            db.session.rollback()
            raise  # 重新抛出异常以便调用方处理
        finally:
            db.session.close()


def save_log_entry(files: list, batch_id: str, batch_size):
    """保存单条日志到数据库"""
    config = deepcopy(current_app.config)

    def process_file(file):
        # 提前获取数据库引擎配置
        start_time = datetime.now()
        app = Flask(__name__)
        app.config.update(config)
        # 重新初始化扩展
        db.init_app(app)

        with app.app_context():
            file_size = os.path.getsize(file)
            file_name = Path(file).name
            buffer_size = min(max(file_size // 100, 64 * 1024), 8 * 1024 * 1024)
            processed_lines = 0
            try:
                with open(file, 'r', encoding='utf-8', errors='replace',
                          buffering=buffer_size) as f:
                    # 批量收集记录
                    records = []
                    for line_number, line in enumerate(f, 1):
                        if ']' == line.strip():
                            continue
                        records.append({
                            'batch_id': batch_id,
                            'line_number': line_number,
                            'file_name': file_name,
                            'log_content': line.strip()
                        })
                        # 达到批处理大小时提交
                        if len(records) >= batch_size:
                            with db_session_scope(db.engine) as session:
                                session.bulk_insert_mappings(LogEntry, records)
                            processed_lines += len(records)
                            records = []
                            logger.debug(f"已处理 {file_name} 的 {processed_lines} 行")
                    # 提交剩余记录
                    if records:
                        with db_session_scope(db.engine) as session:
                            session.bulk_insert_mappings(LogEntry, records)
                        processed_lines += len(records)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"完成文件 {file_name} 处理, 共 {processed_lines} 行, "
                    f"耗时 {elapsed:.2f} 秒, 平均 {processed_lines / elapsed:.1f} 行/秒"
                )
            except Exception as e:
                logger.error(f"处理文件 {file_name} 失败: {str(e)}", exc_info=True)
                raise

    try:
        with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
            # 将app对象传递给每个线程
            # 每个线程使用独立的数据库引擎
            futures = [
                executor.submit(process_file, file)
                for file in files
            ]

            # 等待所有任务完成
            for future in futures:
                future.result()  # 这里会抛出任何线程中的异常
    except Exception as e:
        logger.error(f"保存日志失败(行{files}): {e}")


def clean_history():
    # 计算两天前的时间

    def background_LogError_task():
        with current_app.app_context():
            while True:
                # Fetch a batch of IDs to delete
                # 分批删除，避免锁表时间过长
                logger.info("清理LogError")
                ids_to_delete = db.session.query(LogError.id) \
                    .filter(LogError.create_time < days_ago) \
                    .limit(batch_size) \
                    .all()

                if not ids_to_delete:
                    break  # No more records to delete

                # Delete the batch
                db.session.execute(
                    delete(LogError)
                        .where(LogError.id.in_([id[0] for id in ids_to_delete]))
                )
                db.session.commit()  # Commit after each batch

    def background_LogException_task():
        with current_app.app_context():
            logger.info("清理LogException")
            while True:
                # Fetch a batch of IDs to delete
                ids_to_delete = db.session.query(LogException.id) \
                    .filter(LogException.create_time < days_ago) \
                    .limit(batch_size) \
                    .all()

                if not ids_to_delete:
                    break  # No more records to delete

                # Delete the batch
                db.session.execute(
                    delete(LogException)
                        .where(LogException.id.in_([id[0] for id in ids_to_delete]))
                )
                db.session.commit()  # Commit after each batch

    def background_LogEntry_task():
        with current_app.app_context():
            logger.info("清理LogEntry")
            while True:
                # Fetch a batch of IDs to delete
                ids_to_delete = db.session.query(LogEntry.id) \
                    .filter(LogEntry.create_time < days_ago) \
                    .limit(batch_size) \
                    .all()

                if not ids_to_delete:
                    break  # No more records to delete

                # Delete the batch
                db.session.execute(
                    delete(LogEntry)
                        .where(LogEntry.id.in_([id[0] for id in ids_to_delete]))
                )
                db.session.commit()  # Commit after each batch

    def background_AnalyzedFile_task():
        with current_app.app_context():
            logger.info("清理AnalyzedFile")
            while True:
                ids_to_delete = db.session.query(AnalyzedFile.id) \
                    .filter(AnalyzedFile.create_time < days_ago) \
                    .limit(batch_size) \
                    .all()

                if not ids_to_delete:
                    break  # No more records to delete

                # Delete the batch
                db.session.execute(
                    delete(AnalyzedFile)
                        .where(AnalyzedFile.id.in_([id[0] for id in ids_to_delete]))
                )
                db.session.commit()  # Commit after each batch

    def background_AnalysisBatch_task():
        with current_app.app_context():
            logger.info("清理AnalysisBatch")
            while True:
                ids_to_delete = db.session.query(AnalysisBatch.id) \
                    .filter(AnalysisBatch.create_time < days_ago) \
                    .limit(batch_size) \
                    .all()

                if not ids_to_delete:
                    break  # No more records to delete

                # Delete the batch
                db.session.execute(
                    delete(AnalysisBatch)
                        .where(AnalysisBatch.id.in_([id[0] for id in ids_to_delete]))
                )
                db.session.commit()  # Commit after each batch

    with current_app.app_context():
        batch_size = current_app.config['DEL_LIMIT']
        try:
            days_ago = datetime.now() - timedelta(days=current_app.config['CLEAR_TIME'])

            # 使用线程池执行
            executor = ThreadPoolExecutor(max_workers=1)
            executor1 = ThreadPoolExecutor(max_workers=1)
            executor2 = ThreadPoolExecutor(max_workers=1)
            executor3 = ThreadPoolExecutor(max_workers=1)
            executor4 = ThreadPoolExecutor(max_workers=1)
            executor.submit(background_LogError_task)
            executor1.submit(background_LogException_task)
            executor2.submit(background_LogEntry_task)
            executor3.submit(background_AnalyzedFile_task)
            executor4.submit(background_AnalysisBatch_task)
            executor.shutdown(wait=False)
            executor2.shutdown(wait=False)
            executor3.shutdown(wait=False)
            executor1.shutdown(wait=False)
            executor4.shutdown(wait=False)

        except Exception as e:
            logger.error("清理历史记录失败{}".format(e))
        finally:
            db.session.close()


def get_config(user_id, config_type, key, default=None):
    """获取指定配置"""
    config = UserConfig.query.filter_by(
        user_id=user_id,
        config_type=config_type,
        key=key,
        is_active=True
    ).first()
    return _parse_value(config.value, config.data_type) if config else default


def set_config(user_id, config_type, key, value):
    """设置配置"""
    data_type = _infer_data_type(value)
    config = UserConfig.query.filter_by(
        user_id=user_id,
        config_type=config_type,
        key=key
    ).first()

    if config:
        config.value = str(value)
        config.data_type = data_type
    else:
        config = UserConfig(
            user_id=user_id,
            config_type=config_type,
            key=key,
            value=str(value),
            data_type=data_type
        )
    db.session.add(config)
    db.session.commit()


def _infer_data_type(value):
    """推断值类型"""
    if isinstance(value, bool):
        return 'boolean'
    elif isinstance(value, (int, float)):
        return 'number'
    elif isinstance(value, (list, dict)):
        return 'json'
    return 'string'


def _parse_value(value, data_type):
    """解析存储的值"""
    if not value:
        return None
    try:
        if data_type == 'number':
            return float(value) if '.' in value else int(value)
        elif data_type == 'boolean':
            return value.lower() == 'true'
        elif data_type == 'json':
            return json.loads(value)
        return value
    except (ValueError, json.JSONDecodeError):
        return value


def delete_chunks(file_id, user_id=None):
    """
    安全删除指定文件的所有分块和临时目录

    参数:
        file_id (str): 文件唯一标识
        user_id (str|None): 用户ID，为None时搜索所有用户目录

    返回:
        tuple: (是否成功, 删除的文件数)
    """
    deleted_count = 0
    upload_dir = Path(current_app.config['UPLOAD_FOLDER'])

    try:
        # 情况1：指定用户目录
        if user_id:
            temp_dir = upload_dir / str(user_id) / str(file_id)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                current_app.logger.info(f"删除用户 {user_id} 的文件 {file_id} 分块")
                return (True, len(list(temp_dir.glob('*'))))

        # 情况2：全局搜索（谨慎使用）
        else:
            for user_dir in upload_dir.iterdir():
                if not user_dir.is_dir():
                    continue

                temp_dir = user_dir / str(file_id)
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    deleted_count += len(list(temp_dir.glob('*')))
                    current_app.logger.info(f"删除 {user_dir.name} 的文件 {file_id} 分块")

            return (True, deleted_count) if deleted_count > 0 else (False, 0)

    except Exception as e:
        current_app.logger.error(f"删除分块失败: {str(e)}", exc_info=True)
        return (False, deleted_count)


# 辅助函数 - 从数据库获取用户文件
def get_user_files_from_db(user_id, file_ids):
    # 这里应该使用数据库查询替代全局变量
    return FileUpload.query.filter(
        FileUpload.id.in_(file_ids),
        FileUpload.user_id == user_id
    ).all()


from sqlalchemy import func, update, case
from sqlalchemy.exc import IntegrityError  # 正确导入方式
from sqlalchemy.exc import OperationalError


#
# MAX_DEADLOCK_RETRIES = 5
# BASE_RETRY_DELAY = 0.1  # seconds


def handle_upload_chunk(batch_id, user_id):
    """
    Handle file chunk upload with proper deadlock handling and transaction management

    Args:
        batch_id: The batch ID for the upload
        user_id: The user ID initiating the upload

    Returns:
        The updated or created AnalysisBatch instance

    Raises:
        Exception: After all retries are exhausted
    """
    MAX_RETRIES = UserConfig.get_user_config(user_id, 'performance', 'max_deadlock_retries')
    BASE_RETRY_DELAY = UserConfig.get_user_config(user_id, 'performance', 'base_retry_delay')

    for attempt in range(MAX_RETRIES):
        try:
            with db.session.begin_nested():  # Use nested transaction
                # First try to update existing record
                stmt = (
                    update(AnalysisBatch)
                        .where(AnalysisBatch.batch_id == batch_id)
                        .values(
                        total_files=AnalysisBatch.total_files + 1,
                        status=case(
                            (AnalysisBatch.status == None, 'uploading'),
                            else_=AnalysisBatch.status
                        )
                    )
                        .execution_options(synchronize_session=False)
                )

                result = db.session.execute(stmt)

                if result.rowcount == 0:  # No record was updated
                    try:
                        # Insert new record
                        batch = AnalysisBatch(
                            batch_id=batch_id,
                            user_id=user_id,
                            status='uploading',
                            total_files=1,
                            start_time=datetime.now()
                        )
                        db.session.add(batch)
                        db.session.flush()  # Immediate execution but don't commit yet
                        return batch
                    except IntegrityError:
                        # Another request created the record, retry the update
                        db.session.rollback()
                        continue

                # If we updated, fetch the current record
                batch = db.session.query(AnalysisBatch).filter_by(batch_id=batch_id).one()
                return batch

        except OperationalError as e:
            if "Deadlock" in str(e) and attempt < MAX_RETRIES - 1:
                # Exponential backoff with jitter
                sleep_time = BASE_RETRY_DELAY * (2 ** attempt) * (0.8 + 0.4 * random())
                logger.warning(
                    f"Deadlock detected handling upload (attempt {attempt + 1}), "
                    f"retrying in {sleep_time:.2f}s"
                )
                time.sleep(sleep_time)
                continue
            logger.error(
                f"Failed to handle upload after {attempt + 1} attempts: {str(e)}",
                exc_info=True
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error handling upload: {str(e)}",
                exc_info=True
            )
            raise

@contextmanager
def scoped_session1():
    """Provide a transactional scope around a series of operations."""
    session = db.session
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def create_analyzed_file(file_path: str, user_id: str, total_lines: int, batch_id) -> AnalyzedFile:
    """Create a new record for file analysis progress"""
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        with progress_lock:  # Ensure thread-safe operation
            with scoped_session1() as session:
                try:
                    analyzed_file = AnalyzedFile(
                        file_name=os.path.basename(file_path),
                        batch_id=batch_id,
                        file_path=file_path,
                        user_id=user_id,
                        total_lines=total_lines,
                        processed_lines=0,
                        status='parsing',
                        start_time=datetime.now(),
                        progress=0
                    )
                    session.add(analyzed_file)
                    session.commit()
                    return analyzed_file.id
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error create analysis : {str(e)}", exc_info=True)
                    raise
                finally:
                    session.close()  # CRITICAL: Always close session
    finally:
        if ctx:
            ctx.pop()

def update_analysis_progress(analyzed_file_or_id: int, processed_lines: int, total_lines: int):
    """
    Update progress in database (thread-safe)

    Args:
        analyzed_file_id: ID of the AnalyzedFile record
        processed_lines: Number of lines processed so far
        total_lines: Total lines in the file
    """
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        with progress_lock:  # Ensure thread-safe operation
            with scoped_session1() as session:
                try:
                    if isinstance(analyzed_file_or_id, int):
                        analyzed_file = session.query(AnalyzedFile).get(analyzed_file_or_id)
                    else:
                        # Re-attach the object to the new session
                        analyzed_file = session.merge(analyzed_file_or_id)

                    if not analyzed_file:
                        logger.warning(f"AnalyzedFile not found")
                        return

                    progress = min(100, int((processed_lines / total_lines) * 100))
                    analyzed_file.processed_lines = processed_lines
                    analyzed_file.progress = progress
                    analyzed_file.last_update = datetime.now()

                    # Expire the object to prevent stale data
                    # session.expire(analyzed_file)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error updating progress: {str(e)}", exc_info=True)
                    raise
                finally:
                    session.close()  # CRITICAL: Always close session
    finally:
        if ctx:
            ctx.pop()

def mark_analysis_complete(analyzed_file_or_id: int):
    """
    Mark file analysis as completed (thread-safe)

    Args:
        analyzed_file_id: ID of the AnalyzedFile record
    """
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        with progress_lock:
            with scoped_session1() as session:
                try:
                    if isinstance(analyzed_file_or_id, int):
                        analyzed_file = session.query(AnalyzedFile).get(analyzed_file_or_id)
                    else:
                        # Re-attach the object to the new session
                        analyzed_file = session.merge(analyzed_file_or_id)

                    if not analyzed_file:
                        logger.warning(f"AnalyzedFile not found")
                        return

                    analyzed_file.status = 'completed'
                    analyzed_file.end_time = datetime.now()
                    analyzed_file.progress = 100
                    # session.expire(analyzed_file)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error marking analysis complete: {str(e)}", exc_info=True)
                    raise
                finally:
                    session.close()  # CRITICAL: Always close session
    finally:
        if ctx:
            ctx.pop()

def mark_analysis_failed(analyzed_file_or_id: int, error_message: str):
    """
    Mark file analysis as failed (thread-safe)

    Args:
        analyzed_file_or_id: ID of the AnalyzedFile record
        error_message: Error message to store
    """
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        with progress_lock:
            with scoped_session1() as session:
                try:
                    if isinstance(analyzed_file_or_id, int):
                        analyzed_file = session.query(AnalyzedFile).get(analyzed_file_or_id)
                    else:
                        # Re-attach the object to the new session
                        analyzed_file = session.merge(analyzed_file_or_id)

                    if not analyzed_file:
                        logger.warning(f"AnalyzedFile not found")
                        return

                    analyzed_file.status = 'failed'
                    analyzed_file.error_message = error_message[:500]  # Limit error message length
                    analyzed_file.end_time = datetime.now()
                    # session.expire(analyzed_file)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error marking analysis failed: {str(e)}", exc_info=True)
                    raise
                finally:
                    session.close()  # CRITICAL: Always close session
    finally:
        if ctx:
            ctx.pop()


def time_regex(userid):
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        TIME_PATTERN = UserConfig.get_user_config(userid, 'time', 'pattern')
        return re.compile(TIME_PATTERN)
    except Exception as e:
        logger.error(f"Error time_regex : {str(e)}", exc_info=True)
        raise
    finally:
        if ctx:
            ctx.pop()


def warning_kwords(user_id):
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        return UserConfig.get_user_config(user_id, 'keyword', 'warning')
    except Exception as e:
        logger.error(f"Error warning_kwords : {str(e)}", exc_info=True)
        raise
    finally:
        if ctx:
            ctx.pop()


def getConfig(user_id, config_type, key):
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        return UserConfig.get_user_config(user_id, config_type, key)
    except Exception as e:
        logger.error(f"Error getConfig: {str(e)}", exc_info=True)
        raise
    finally:
        if ctx:
            ctx.pop()

def err_kwords(user_id):
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        return UserConfig.get_user_config(user_id, 'keyword', 'error')
    except Exception as e:
        logger.error(f"Error err_kwords : {str(e)}", exc_info=True)
        raise
    finally:
        if ctx:
            ctx.pop()

def kwrods(userid):
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        KEYWORDS = UserConfig.get_user_config(userid, 'keyword', 'normal')
        WARNING_KEYWORDS = warning_kwords(userid)
        ERROR_KEYWORDS = err_kwords(userid)
        # EXECEPTION_KEYWORDS = ['EXCEPTION']
        for key in ERROR_KEYWORDS:
            KEYWORDS.append(key)
        for key in WARNING_KEYWORDS:
            KEYWORDS.append(key)
        return KEYWORDS
    except Exception as e:
        logger.error(f"Error kwrods : {str(e)}", exc_info=True)
        raise
    finally:
        if ctx:
            ctx.pop()

def kwds_regex(userid):
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        return {kw: re.compile(re.escape(kw)) for kw in kwrods(userid)}
    except Exception as e:
        logger.error(f"Error kwds_regex : {str(e)}", exc_info=True)
        raise
    finally:
        if ctx:
            ctx.pop()

def exceptions_regex(userid):
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        EXCEPTION_PATTERNS = UserConfig.get_user_config(userid, 'exception', 'patterns')
        return re.compile('|'.join(EXCEPTION_PATTERNS.values()))
    except Exception as e:
        logger.error(f"Error exceptions_regex : {str(e)}", exc_info=True)
        raise
    finally:
        if ctx:
            ctx.pop()


def update_report_path(batch_id: str, report_path: str, user_id):
    """Update report_path for a specific batch_id"""
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None
    try:
        with progress_lock:
            with scoped_session1() as session:
                try:
                    batch = AnalysisBatch.query.filter_by(batch_id=batch_id, user_id=user_id).first()
                    if batch:
                        batch.report_path = report_path
                        db.session.commit()
                    # session.expire(analyzed_file)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to update report_path for batch {batch_id}: {str(e)}")
                    raise
                finally:
                    session.close()  # CRITICAL: Always close session
    finally:
        if ctx:
            ctx.pop()

def getConfigs(userid):
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        TIME_PATTERN = UserConfig.get_user_config(userid, 'time', 'pattern')
        # 日志文件解析配置
        KEYWORDS = UserConfig.get_user_config(userid, 'keyword', 'normal')
        WARNING_KEYWORDS = UserConfig.get_user_config(userid, 'keyword', 'warning')
        ERROR_KEYWORDS = UserConfig.get_user_config(userid, 'keyword', 'error')
        # EXECEPTION_KEYWORDS = ['EXCEPTION']
        for key in ERROR_KEYWORDS:
            KEYWORDS.append(key)
        for key in WARNING_KEYWORDS:
            KEYWORDS.append(key)

        MAX_ERROR_DETAILS = UserConfig.get_user_config(userid, 'performance', 'max_error_details')
        MAX_FILE_ERRORS = UserConfig.get_user_config(userid, 'performance', 'max_file_errors')  # 每个文件保存的最大错误数量
        work = UserConfig.get_user_config(userid, 'performance', 'max_workers')
        MAX_WORKERS = work if os.cpu_count() < work else os.cpu_count()
        MIN_MMAP_SIZE = UserConfig.get_user_config(userid, 'performance', 'min_mmap_size')
        DEFAULT_BUFFER_SIZE = UserConfig.get_user_config(userid, 'performance', 'default_buffer_size')  # 1MB默认缓冲区
        BATCH_SIZE = UserConfig.get_user_config(userid, 'performance', 'batch_size')  # 1MB默认缓冲区
        # 异常类型识别模式
        EXCEPTION_PATTERNS = UserConfig.get_user_config(userid, 'exception', 'patterns')
        return TIME_PATTERN, KEYWORDS, WARNING_KEYWORDS, ERROR_KEYWORDS, MAX_ERROR_DETAILS, MAX_FILE_ERRORS, MAX_WORKERS,\
               MIN_MMAP_SIZE, DEFAULT_BUFFER_SIZE, EXCEPTION_PATTERNS, BATCH_SIZE
    except Exception as e:
        logger.error(f"Error getConfig : {str(e)}", exc_info=True)
        raise
    finally:
        if ctx:
            ctx.pop()