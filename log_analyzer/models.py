import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import LONGTEXT  # 明确导入 MySQL 的 LONGTEXT

db = SQLAlchemy()


class LogEntry(db.Model):
    """
    日志条目模型，对应MySQL中的log_entries表
    """
    __tablename__ = 'log_entries'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment='主键ID')
    file_name = db.Column(db.String(255), nullable=False, comment='日志文件名')
    batch_id = db.Column(db.String(64), nullable=False, comment='批次ID，用于标识同一批日志')
    line_number = db.Column(db.Integer, nullable=False, comment='日志行号')
    log_content = db.Column(LONGTEXT, nullable=False, comment='日志内容')
    create_time = db.Column(db.TIMESTAMP, server_default=func.now(), comment='创建时间')

    # 添加索引
    __table_args__ = (
        Index('idx_batch_id', 'batch_id'),
        Index('idx_file_name', 'file_name'),
        Index('idx_create_time', 'create_time'),
        {'comment': '日志存储表',
         'mysql_engine': 'InnoDB',
         'mysql_charset': 'utf8mb4'}
    )

    def __repr__(self):
        return f"<LogEntry(file='{self.file_name}', line={self.line_number}, level={self.log_level})>"


class AnalysisBatch(db.Model):
    """分析批次表"""
    __tablename__ = 'analysis_batches'

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(64), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    start_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    total_files = db.Column(db.Integer)
    total_lines = db.Column(db.Integer)
    total_errors = db.Column(db.Integer)
    total_warnings = db.Column(db.Integer)
    report_path = db.Column(db.String(512))
    status = db.Column(db.String(32), default='uploading')  # uploading, parsing, completed, failed
    create_time = db.Column(TIMESTAMP, server_default=func.now(), comment='创建时间')
    # 关系
    exceptions = db.relationship('LogException', backref='batch', lazy=True)
    errors = db.relationship('LogError', backref='batch', lazy=True)
    files = db.relationship('AnalyzedFile', backref='batch', lazy=True)

    # 添加索引
    __table_args__ = (
        Index('idx_batch_id', 'batch_id', 'user_id'),
        Index('idx_batch_created', 'start_time'),
        Index('idx_batch_user', 'user_id', 'status'),
        {'comment': '批次表',
         'mysql_engine': 'InnoDB',
         'mysql_charset': 'utf8mb4'}
    )


class LogException(db.Model):
    """异常记录表"""
    __tablename__ = 'log_exceptions'

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(64), db.ForeignKey('analysis_batches.batch_id'), nullable=False)
    exception_type = db.Column(db.String(128), nullable=False)
    count = db.Column(db.Integer, nullable=False)
    line_number = db.Column(db.Integer, nullable=False)
    first_occurrence = db.Column(db.DateTime)
    last_occurrence = db.Column(db.DateTime)
    source_file = db.Column(db.String(512))
    example_message = db.Column(db.Text)
    create_time = db.Column(TIMESTAMP, server_default=func.now(), comment='创建时间')

    # 添加索引
    __table_args__ = (
        Index('idx_batch_id', 'batch_id'),
        {'comment': '异常日志存储表',
         'mysql_engine': 'InnoDB',
         'mysql_charset': 'utf8mb4'}
    )


class LogError(db.Model):
    """错误记录表"""
    __tablename__ = 'log_errors'

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(64), db.ForeignKey('analysis_batches.batch_id'), nullable=False)
    timestamp = db.Column(db.DateTime)
    source_file = db.Column(db.String(512), nullable=False)
    line_number = db.Column(db.Integer)
    keyword = db.Column(db.String(128))
    exception_type = db.Column(db.String(128))
    message = db.Column(db.Text)
    raw_line = db.Column(db.Text)
    create_time = db.Column(TIMESTAMP, server_default=func.now(), comment='创建时间')

    # 添加索引
    __table_args__ = (
        Index('idx_batch_id', 'batch_id'),
        {'comment': '错误日志存储表',
         'mysql_engine': 'InnoDB',
         'mysql_charset': 'utf8mb4'}
    )


class AnalyzedFile(db.Model):
    """已分析文件表"""
    __tablename__ = 'analyzed_files'

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(64), db.ForeignKey('analysis_batches.batch_id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    file_path = db.Column(db.String(512), nullable=False)
    file_name = db.Column(db.String(256), nullable=False)
    total_lines = db.Column(db.Integer)
    error_count = db.Column(db.Integer)
    warning_count = db.Column(db.Integer)
    keyword_count = db.Column(db.Integer)
    analysis_time = db.Column(db.Float)  # 分析耗时(秒)
    create_time = db.Column(TIMESTAMP, server_default=func.now(), comment='创建时间')
    progress = db.Column(db.Integer, nullable=False, default=0)  #   # 0-100
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    processed_lines = db.Column(db.Integer, nullable=False, default=0)
    status = db.Column(db.String(20), nullable=False, default='queued')  # queued/parsing/completed/failed
    last_update = db.Column(db.DateTime, onupdate=datetime.utcnow)

    # 添加索引
    __table_args__ = (
        Index('idx_batch_filename_id', 'batch_id', 'file_name'),
        Index('idx_batch_filename_user_id', 'batch_id', 'file_name', 'user_id'),
        Index('idx_batch_id', 'batch_id'),
        {'comment': '日志结果存储表',
         'mysql_engine': 'InnoDB',
         'mysql_charset': 'utf8mb4'}
    )


DEFAULT_CONFIGS = {
    'time': {
        'pattern': (r'(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}:\d{1,2}.\d{1,10})', 'string')
    },
    'keyword': {
        'normal': ('[]', 'json'),
        'warning': ('["[WARNING]"]', 'json'),
        'error': ('["[ERROR]"]', 'json')
    },
    'performance': {
        'max_error_details': ('500', 'number'),
        'max_file_errors': ('1000', 'number'),
        'max_workers': ('5', 'number'),
        'min_mmap_size': ('104857600', 'number'),  # 100MB
        'default_buffer_size': ('104857600', 'number'),  # 100MB
        'batch_size': ('5000', 'number'),  # 100MB20
        'max_chunk_size': ('104857600', 'number'),
        'max_deadlock_retries': ('5', 'number'),
        'base_retry_delay': ('0.1', 'float'),
        'batchUploadFiles_chunk_size': ('6164480', 'number'),  # 5M
        'byte_cache_size': ('6291456', 'number'),  # 6M
    },
    'exception': {
        'patterns': (json.dumps({
            'NullPointerException': 'NullPointerException',
            'TimeoutException': 'TimeoutException',
            'IOException': 'IOException',
            'SQLException': 'SQLException',
            'IndexOutOfBoundsException': 'IndexOutOfBoundsException',
            'OtherError': 'OtherError'
        }), 'json')
    }
}


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserConfig(db.Model):
    __tablename__ = 'user_configs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    config_type = db.Column(db.String(50), nullable=False)
    key = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Text)
    data_type = db.Column(db.String(20), default='string')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'config_type', 'key', name='uq_user_config'),
    )

    @classmethod
    def ensure_user_exists(cls, user_id):
        """确保用户存在，否则抛出异常"""
        if not User.query.get(user_id):
            raise ValueError(f"用户ID {user_id} 不存在")

    @classmethod
    def ensure_user_configs(cls, user_id):
        """确保用户有完整配置(不存在则创建默认)"""
        for config_type, items in DEFAULT_CONFIGS.items():
            for key, (default_value, data_type) in items.items():
                if not cls.query.filter_by(
                        user_id=user_id,
                        config_type=config_type,
                        key=key
                ).first():
                    db.session.add(cls(
                        user_id=user_id,
                        config_type=config_type,
                        key=key,
                        value=default_value,
                        data_type=data_type
                    ))
        db.session.commit()

    @classmethod
    def get_user_config(cls, user_id, config_type, key):
        """获取配置(自动初始化缺失配置)"""
        cls.ensure_user_exists(user_id)
        config = cls.query.filter_by(
            user_id=user_id,
            config_type=config_type,
            key=key
        ).first()

        if config.data_type == 'number' or config.data_type == 'float' :
            return float(config.value) if '.' in config.value else int(config.value)
        elif config.data_type == 'boolean':
            return config.value.lower() == 'true'
        elif config.data_type == 'json':
            return json.loads(config.value)
        return config.value

    @classmethod
    def set_user_config(cls, user_id, config_type, key, value):
        """更新配置"""
        cls.ensure_user_exists(user_id)
        config = cls.query.filter_by(
            user_id=user_id,
            config_type=config_type,
            key=key
        ).first()

        if isinstance(value, (list, dict)):
            value = json.dumps(value)
            data_type = 'json'
        else:
            value = str(value)
            data_type = 'string'
            if isinstance(value, bool):
                data_type = 'boolean'
            elif isinstance(value, (int, float)):
                data_type = 'number'

        if config:
            config.value = value
            config.data_type = data_type
        else:
            config = cls(
                user_id=user_id,
                config_type=config_type,
                key=key,
                value=value,
                data_type=data_type
            )
        db.session.add(config)
        db.session.commit()


# 在模型中添加上传记录表
class FileUpload(db.Model):
    __tablename__ = 'file_uploads'
    id = db.Column(db.String(64), primary_key=True)  # file_id
    user_id = db.Column(db.String(64), nullable=False)
    filename = db.Column(db.String(255))
    total_chunks = db.Column(db.Integer)
    received_chunks = db.Column(db.JSON)  # 存储已接收的分块索引列表
    status = db.Column(db.String(20), default='uploading')  # uploading/completed/failed
    batch_id = db.Column(db.String(64), db.ForeignKey('analysis_batches.batch_id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 添加索引
    __table_args__ = (
        db.Index('idx_user_batch_id_uploads', 'user_id', 'status', 'batch_id'),
        {'comment': '上传记录表',
         'mysql_engine': 'InnoDB',
         'mysql_charset': 'utf8mb4'}
    )


def init_db():
    db.create_all()
