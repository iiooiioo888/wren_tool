#!/usr/bin/env python
"""
Wren Tool 專案安裝配置
"""

from setuptools import setup, find_packages
from pathlib import Path

# 讀取 README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# 需求依賴
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# 開發需求
dev_requirements = [
    "pytest>=6.0.0",
    "pytest-cov>=2.12.0",
    "black>=21.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "pre-commit>=2.15.0"
]

# 測試需求
test_requirements = [
    "pytest>=6.0.0",
    "pytest-cov>=2.12.0",
    "pytest-xdist>=2.2.0"
]

setup(
    name="wren-tool",
    version="2.0.0",
    author="Wren Tool Team",
    author_email="contact@wren-tool.dev",
    description="增強版加密貨幣交易策略回測平台",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iiooiioo888/wren_tool",
    packages=find_packages(include=["wren_tool", "wren_tool.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="backtesting trading strategy cryptocurrency quantitative finance",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": test_requirements,
        "all": dev_requirements + test_requirements,
    },
    entry_points={
        "console_scripts": [
            "wren-poc=scripts.poc_run:main",
            "wren-test=run_tests:main",
        ],
    },
    include_package_data=True,
    package_data={
        "wren_tool": [
            "data/*.csv",
            "config/*.yaml",
            "config/*.json"
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/iiooiioo888/wren_tool/issues",
        "Source": "https://github.com/iiooiioo888/wren_tool",
        "Documentation": "https://wren-tool.readthedocs.io/",
    },
)
