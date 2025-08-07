# app/extensions.py
from flask_migrate import Migrate
from .config.base import Config
from .models import db, init_db

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 配置连接池

Session = None
engine = None




def configure_extensions(app):
    global Session, engine
    """初始化所有扩展"""
    Config.init_app(app)
    db.init_app(app)
    migrate = Migrate(app, db)  # 关键初始化

    # register_commands(app, db, ActivityLog)

    # 避免循环导入的关键：在初始化后导入模型
    with app.app_context():
        init_db()
        # 可选：扩展配置函数
        engine = create_engine(
            app.config['SQLALCHEMY_DATABASE_URI'],
            pool_size=app.config['POOL_SIZE'],  # 连接池大小
            max_overflow=app.config['MAX_OVERFLOW'],  # 最大溢出连接数
            pool_timeout=app.config['POOL_TIMEOUT'],  # 获取连接超时时间(秒)
            pool_recycle=app.config['POOL_RECYCLE']  # 连接回收时间(秒)
        )
        Session = sessionmaker(bind=engine)
