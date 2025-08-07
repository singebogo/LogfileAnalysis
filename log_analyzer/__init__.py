import os

from flask import Flask

from .extensions import configure_extensions
from .config import config, DevelopmentConfig, base


def create_app(config_name='development'):
    app = Flask(__name__)

    if os.getenv('FLASK_DEBUG') == '1':
        app.config.from_object(DevelopmentConfig)
    else:
        # 加载配置
        app.config.from_object(config[config_name])

    # 初始化扩展
    configure_extensions(app)

    # 注册蓝图
    # from .controllers.auth import auth_bp
    # from .controllers.admin import admin_bp
    # from .controllers.logs import logs_bp
    # from .controllers.email_settings import email_bp
    # app.register_blueprint(auth_bp)
    # app.register_blueprint(admin_bp)
    # app.register_blueprint(logs_bp)
    # app.register_blueprint(email_bp)

    return app