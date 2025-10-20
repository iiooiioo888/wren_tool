"""
Wren Tool - 增強版加密貨幣交易策略回測平台

一個功能完整的加密貨幣交易策略回測平台，支援真實交易環境模擬、
多策略比較、風險管理以及完整的測試框架。
"""

from .config.paths import (
    project_paths,
    PROJECT_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_PATH
)

__version__ = "2.0.0"
__author__ = "Wren Tool Team"
__email__ = "contact@wren-tool.dev"

# 確保基本目錄存在
try:
    project_paths.ensure_directories_exist()
except Exception as e:
    print(f"警告: 初始化專案目錄失敗 - {e}")

def get_version():
    """獲取版本信息"""
    return __version__

def initialize_project():
    """初始化專案目錄結構"""
    project_paths.ensure_directories_exist()
    return project_paths.get_project_info()
