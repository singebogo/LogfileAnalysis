from .base import Config


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_ECHO = True
    OTP_TEST_CODE = '123456'  # 开发环境固定验证码

    RECAPTCHA_PUBLIC_KEY = '6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI'  # 站点key
    RECAPTCHA_PRIVATE_KEY = '6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe'  # 密钥
    RECAPTCHA_PARAMETERS = {'theme': 'light'}  # 可选配置
    # 开发环境禁用验证码
    # RECAPTCHA_ENABLED = not os.environ.get('FLASK_DEBUG')
    # 或者基于条件启用
    RECAPTCHA_ENABLED = True  # 始终启用
    SQLALCHEMY_RECORD_QUERIES = True

    DEBUG_TB_INTERCEPT_REDIRECTS = True  # 默认值，可显式声明

    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:123456@127.0.0.1/log_analysis'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    POOL_SIZE = 10  # 连接池大小
    MAX_OVERFLOW = 20  # 最大溢出连接数
    POOL_TIMEOUT = 30  # 获取连接超时时间(秒)
    POOL_RECYCLE= 3600  # 连接回收时间(秒)

    CLEAR_TIME = 2  # 清理报告期限
    DEL_LIMIT = 20000  # 清理数据库条数

    SERVER_NAME = '127.0.0.1'  # e.g., 'example.com' (no http/https)
    PREFERRED_URL_SCHEME = 'http'  # or 'http' in development