"""
統一路徑配置模組
提供跨平台的路徑解析和管理功能
"""
import os
from pathlib import Path
from typing import Optional


class ProjectPaths:
    """專案路徑配置管理器"""

    def __init__(self):
        # 獲取專案根目錄
        # 從任何模組文件中調用時，都能正確找到專案根目錄
        self.project_root = self._find_project_root()

        # 主要目錄
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "out"
        self.tests_dir = self.project_root / "tests"
        self.scripts_dir = self.project_root / "scripts"
        self.docs_dir = self.project_root / "docs"

        # 預設文件路徑
        self.default_data_path = self.data_dir / "sample_ohlc.csv"
        self.default_output_path = self.output_dir / "poc_results.json"
        self.default_config_path = self.project_root / "config" / "default.yaml"

    def _find_project_root(self) -> Path:
        """找到專案根目錄（含有 wren_tool/ 目錄的位置）"""

        # 從當前檔案位置開始向上查找
        current_path = Path(__file__).resolve()

        # 向上查找直到找到專案根目錄
        for parent in [current_path] + list(current_path.parents):
            if parent.name == "wren_tool" and parent.is_dir():
                # 如果從 wren_tool/ 內部調用，返回父目錄
                return parent.parent
            elif (parent / "data").exists() and (parent / "scripts").exists():
                # 或者檢查是否有典型專案標記
                return parent

        # 兜底策略：假設當前目錄就是專案根
        return Path.cwd()

    def get_data_path(self, filename: Optional[str] = None) -> Path:
        """獲取數據文件路徑"""
        # 優先使用環境變數
        env_path = os.getenv("WREN_DATA_PATH")
        if env_path:
            base_path = Path(env_path)
        else:
            base_path = self.data_dir

        if filename:
            return base_path / filename
        return base_path

    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """獲取輸出文件路徑"""
        # 優先使用環境變數
        env_path = os.getenv("WREN_OUTPUT_PATH")
        if env_path:
            base_path = Path(env_path)
        else:
            base_path = self.output_dir

        if filename:
            return base_path / filename
        return base_path

    def get_config_path(self, filename: str = "default.yaml") -> Path:
        """獲取配置文件路徑"""
        env_path = os.getenv("WREN_CONFIG_PATH")
        if env_path:
            base_path = Path(env_path)
        else:
            base_path = self.project_root / "config"

        return base_path / filename

    def get_test_data_path(self, filename: Optional[str] = None) -> Path:
        """獲取測試數據路徑"""
        base_path = self.tests_dir / "data"

        if filename:
            return base_path / filename
        return base_path

    def ensure_directories_exist(self):
        """確保所需目錄存在"""
        directories = [
            self.data_dir,
            self.output_dir,
            self.tests_dir,
            self.project_root / "config",
            self.project_root / "logs"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_relative_path(self, path_str: str) -> Path:
        """解析相對路徑為絕對路徑"""
        path = Path(path_str)

        if path.is_absolute():
            return path
        else:
            return self.project_root / path

    def get_project_info(self) -> dict:
        """獲取專案路徑資訊"""
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "tests_dir": str(self.tests_dir),
            "scripts_dir": str(self.scripts_dir),
            "docs_dir": str(self.docs_dir),
            "default_data_file": str(self.default_data_path),
            "default_output_file": str(self.default_output_path),
            "default_config_file": str(self.default_config_path)
        }


# 全域路徑管理器實例
project_paths = ProjectPaths()

# 確保基本目錄存在
project_paths.ensure_directories_exist()

# 導出常用路徑
PROJECT_ROOT = project_paths.project_root
DATA_DIR = project_paths.data_dir
OUTPUT_DIR = project_paths.output_dir
TESTS_DIR = project_paths.tests_dir

DEFAULT_DATA_PATH = project_paths.default_data_path
DEFAULT_OUTPUT_PATH = project_paths.default_output_path
DEFAULT_CONFIG_PATH = project_paths.default_config_path
DEFAULT_TEST_DATA_PATH = project_paths.get_test_data_path()


def get_data_file(filename: str) -> Path:
    """便利函數：獲取數據文件路徑"""
    return project_paths.get_data_path(filename)


def get_output_file(filename: str) -> Path:
    """便利函數：獲取輸出文件路徑"""
    return project_paths.get_output_path(filename)


def get_config_file(filename: str = "default.yaml") -> Path:
    """便利函數：獲取配置文件路徑"""
    return project_paths.get_config_path(filename)


def setup_project_directories():
    """設定專案目錄（用於初始化）"""
    project_paths.ensure_directories_exist()
    print(f"✅ 專案目錄初始化完成: {PROJECT_ROOT}")


if __name__ == "__main__":
    # 測試腳本
    print("🗂️  專案路徑配置測試")

    paths = project_paths.get_project_info()
    for key, value in paths.items():
        print(f"  {key}: {value}")

    print("\n📁 目錄狀態檢查:")
    directories_to_check = [
        ("data_dir", DATA_DIR),
        ("output_dir", OUTPUT_DIR),
        ("tests_dir", TESTS_DIR)
    ]

    for name, path in directories_to_check:
        status = "✅ 存在" if path.exists() else "❌ 不存在"
        print(f"  {name}: {status}")
