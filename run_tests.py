#!/usr/bin/env python3
"""
測試運行腳本
運行整個測試套件並生成測試報告
"""
import pytest
import sys
import os
from pathlib import Path
import subprocess
import time

def run_test_suite():
    """運行完整的測試套件"""
    print("🚀 開始運行測試套件...")
    print("=" * 60)

    # 設置測試目錄
    test_dirs = [
        "tests/unit",
        "tests/integration"
    ]

    # 收集所有測試文件
    test_files = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in Path(test_dir).glob("test_*.py"):
                test_files.append(str(file))

    print(f"📋 發現測試文件: {len(test_files)} 個")
    for test_file in test_files:
        print(f"  - {test_file}")

    print("\n" + "=" * 60)

    # 運行測試
    start_time = time.time()

    # 運行pytest
    exit_code = pytest.main([
        "--tb=short",
        "--durations=10",
        "--cov=scripts",
        "--cov=simulation",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "-v"
    ] + test_files)

    end_time = time.time()
    execution_time = end_time - start_time

    print("\n" + "=" * 60)
    print("📊 測試執行完成")
    print(f"⏱️  執行時間: {execution_time:.2f}秒")
    print(f"📁 覆蓋率報告: htmlcov/index.html")

    if exit_code == 0:
        print("✅ 所有測試通過！")
    else:
        print("❌ 部分測試失敗")

    return exit_code

def run_specific_test(test_name: str):
    """運行特定測試"""
    print(f"🎯 運行特定測試: {test_name}")
    print("=" * 60)

    exit_code = pytest.main([
        "--tb=short",
        "--durations=10",
        "-v",
        "-k", test_name
    ])

    return exit_code

def run_smoke_tests():
    """運行煙霧測試（快速驗證核心功能）"""
    print("🔥 運行煙霧測試...")
    print("=" * 60)

    # 運行核心測試
    core_tests = [
        "test_enhanced_poc.py::TestDataLoading::test_load_valid_csv",
        "test_enhanced_poc.py::TestSignalGeneration::test_sma_signal_generation",
        "test_enhanced_poc.py::TestTradingSimulation::test_enhanced_simulation_basic",
        "test_simulators.py::TestDelaySimulator::test_delay_simulator_initialization",
        "test_simulators.py::TestSlippageModel::test_slippage_model_initialization",
    ]

    for test in core_tests:
        print(f"運行: {test}")
        exit_code = pytest.main([
            "--tb=short",
            "-v",
            "-k", test
        ])

        if exit_code != 0:
            print(f"❌ 煙霧測試失敗: {test}")
            return exit_code

    print("✅ 煙霧測試通過！")
    return 0

def main():
    """主函數"""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "smoke":
            # 運行煙霧測試
            exit_code = run_smoke_tests()
        elif command == "specific" and len(sys.argv) > 2:
            # 運行特定測試
            test_name = sys.argv[2]
            exit_code = run_specific_test(test_name)
        else:
            print("❌ 未知命令")
            print("用法:")
            print("  python run_tests.py              # 運行完整測試套件")
            print("  python run_tests.py smoke        # 運行煙霧測試")
            print("  python run_tests.py specific <test_name>  # 運行特定測試")
            return 1
    else:
        # 運行完整測試套件
        exit_code = run_test_suite()

    # 返回退出碼
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
