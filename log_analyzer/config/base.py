# config.py
import os
from datetime import timedelta


class Config:
    # 基本配置
    DEBUG = False
    TESTING = False

    # 邮件配置
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.googlemail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')

    LOGIN_FAILURE_WINDOW = 15 * 60  # 统计最近15分钟内的失败尝试
    LOGIN_FAILURE_THRESHOLD = 3  # 3次失败后显示验证码
    LOGIN_LOCKOUT_THRESHOLD = 5  # 5次失败后暂时锁定
    CACHE_DEFAULT_TIMEOUT = 300  # 缓存默认过期时间

    # 分页设置
    POSTS_PER_PAGE = 20

    # 数据库配置
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///log_analysis.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    REPORT_FOLDER = os.path.join(os.getcwd(), 'reports')
    ALLOWED_EXTENSIONS = {'log', 'txt'}
    MAX_CONTENT_LENGTH = 1073741824 * 5  # 5G limit
    CHUNK_SIZE = 100 * 1024 * 1024  # 10MB chunks for processing
    SECRET_KEY = os.getenv('SECRET_KEY') or '@@3xcc3-￥%#%#3de2-$%#e565-432dsfsd'

    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)  # session有效期
    SESSION_TYPE = 'filesystem'  # 使用服务器端session存储
    SESSION_FILE_DIR = './flask_session'  # session文件存储目录
    SESSION_FILE_THRESHOLD = 100  # 最大session文件
    # 新增配置项
    MAX_LOG_LINES_DISPLAY = 1000  # 最大显示日志行数
    MAX_ERRORS_DISPLAY = 200  # 最大显示错误数量
    PLOT_DATA_POINTS = 500  # 图表最大数据点数

    REPORTS_PER_PAGE = 5  # 每页显示的报告数量

    forword_trunk_cache_size = 5 * 1024 * 1024  # 前端缓存






    @staticmethod
    def init_app(app):
        # 确保报告目录存在
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        if not os.path.exists(app.config['REPORT_FOLDER']):
            os.makedirs(app.config['REPORT_FOLDER'])

