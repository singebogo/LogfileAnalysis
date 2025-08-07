# config.py 示例

from dotenv import load_dotenv

from .development import DevelopmentConfig
from .testing import TestingConfig
from .production import ProductionConfig

# 加载环境变量
load_dotenv()

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}