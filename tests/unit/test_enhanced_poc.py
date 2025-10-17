"""
增強版PoC腳本的單元測試
測試交易模擬、策略生成、結果驗證等功能
"""
import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path

# 導入被測試模組
from scripts.poc_run import (
    load_data, generate_sma_signals, generate_rsi_signals,
    simulate_trades_enhanced, run_poc, calculate_max_drawdown,
    calculate_sharpe_ratio, run_benchmark
)


class TestDataLoading:
    """測試數據加載功能"""

    def test_load_valid_csv(self):
        """測試加載有效的CSV文件"""
        csv_path = "data/sample_ohlc.csv"
        df = load_data(csv_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])
        assert df["timestamp"].is_monotonic_increasing

    def test_load_invalid_csv_missing_columns(self, tmp_path):
        """測試加載缺少必要欄位的CSV文件"""
        invalid_csv = tmp_path / "invalid.csv"
        invalid_csv.write_text("timestamp,close\n2023-01-01,100\n")

        with pytest.raises(ValueError, match="CSV 必須包含欄位"):
            load_data(str(invalid_csv))

    def test_load_empty_csv(self, tmp_path):
        """測試加載空CSV文件"""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("timestamp,open,high,low,close,volume\n")

        df = load_data(str(empty_csv))
        assert len(df) == 0


class TestSignalGeneration:
    """測試信號生成功能"""

    def test_sma_signal_generation(self):
        """測試SMA信號生成"""
        # 創建測試數據
        dates = pd.date_range('2023-01-01', periods=20, freq='1H')
        prices = [100 + i * 0.5 for i in range(20)]  # 上漲趨勢
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 20
        })

        df = generate_sma_signals(test_data)

        # 檢查欄位是否存在
        assert "sma_fast" in df.columns
        assert "sma_slow" in df.columns
        assert "signal" in df.columns
        assert "signal_change" in df.columns

        # 檢查信號值範圍
        assert df["signal"].isin([-1, 0, 1]).all()

    def test_rsi_signal_generation(self):
        """測試RSI信號生成"""
        # 創建測試數據
        dates = pd.date_range('2023-01-01', periods=50, freq='1H')
        prices = []
        for i in range(50):
            if i < 25:
                prices.append(100 + i * 0.2)  # 上漲
            else:
                prices.append(105 - (i-25) * 0.2)  # 下跌

        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 50
        })

        df = generate_rsi_signals(test_data)

        # 檢查欄位是否存在
        assert "rsi" in df.columns
        assert "signal" in df.columns
        assert "signal_change" in df.columns

        # 檢查RSI值範圍
        assert df["rsi"].between(0, 100).all()


class TestTradingSimulation:
    """測試交易模擬功能"""

    def test_enhanced_simulation_basic(self):
        """測試增強版模擬基本功能"""
        # 創建簡單測試數據
        dates = pd.date_range('2023-01-01', periods=20, freq='1H')
        prices = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101,
                 100, 99, 98, 97, 96, 97, 98, 99, 100, 101]

        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 20
        })

        result = simulate_trades_enhanced(
            df=test_data,
            initial_cash=10000.0,
            fee=0.001,
            slippage=0.002,
            delay_ms=100,
            strategy="sma_crossover"
        )

        # 檢查結果結構
        required_keys = [
            "initial_cash", "final_cash", "total_value", "total_trades",
            "total_fees", "total_slippage", "total_delay_cost",
            "trades", "simulation_config"
        ]

        for key in required_keys:
            assert key in result

        # 檢查數值合理性
        assert result["total_value"] >= 0
        assert result["total_trades"] >= 0
        assert result["total_fees"] >= 0

    def test_simulation_with_different_strategies(self):
        """測試不同策略的模擬"""
        dates = pd.date_range('2023-01-01', periods=30, freq='1H')
        prices = [100 + i * 0.1 for i in range(30)]

        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 30
        })

        strategies = ["sma_crossover", "rsi"]

        for strategy in strategies:
            result = simulate_trades_enhanced(
                df=test_data,
                initial_cash=10000.0,
                strategy=strategy
            )

            assert "total_value" in result
            assert result["simulation_config"]["strategy"] == strategy

    def test_simulation_with_zero_cash(self):
        """測試零資金情況"""
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        prices = [100 + i for i in range(10)]

        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 10
        })

        result = simulate_trades_enhanced(
            df=test_data,
            initial_cash=0.0
        )

        assert result["total_value"] == 0
        assert result["total_trades"] == 0


class TestRiskMetrics:
    """測試風險指標計算"""

    def test_max_drawdown_calculation(self):
        """測試最大回撤計算"""
        # 創建測試價格數據
        prices = [100, 110, 105, 115, 108, 120, 118, 125, 122, 130]
        df = pd.DataFrame({'close': prices})

        # 創建假交易數據
        trades = [
            {"side": "buy", "execution_price": 100},
            {"side": "sell", "execution_price": 120}
        ]

        max_dd = calculate_max_drawdown(df, trades, 10000)

        # 最大回撤應該在合理範圍內
        assert 0 <= max_dd <= 1

    def test_sharpe_ratio_calculation(self):
        """測試夏普比率計算"""
        # 創建測試價格數據
        prices = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101]
        df = pd.DataFrame({'close': prices})

        trades = [
            {"side": "buy", "execution_price": 100},
            {"side": "sell", "execution_price": 104}
        ]

        sharpe = calculate_sharpe_ratio(df, trades, 10000)

        # 夏普比率應該是數值
        assert isinstance(sharpe, (int, float))

    def test_metrics_with_no_trades(self):
        """測試無交易時的指標計算"""
        df = pd.DataFrame({'close': [100, 101, 102]})
        trades = []

        max_dd = calculate_max_drawdown(df, trades, 10000)
        sharpe = calculate_sharpe_ratio(df, trades, 10000)

        assert max_dd == 0.0
        assert sharpe == 0.0


class TestIntegration:
    """測試整合功能"""

    def test_run_poc_integration(self, tmp_path):
        """測試完整的PoC運行流程"""
        csv_path = "data/sample_ohlc.csv"
        out_file = tmp_path / "integration_test.json"

        result = run_poc(
            csv_path=csv_path,
            out_path=str(out_file),
            strategy="sma_crossover",
            initial_cash=10000.0,
            verbose=False
        )

        # 檢查結果
        assert "total_value" in result
        assert "total_trades" in result
        assert out_file.exists()

        # 檢查輸出文件內容
        with open(out_file, 'r', encoding='utf-8') as f:
            saved_result = json.load(f)

        assert saved_result["total_value"] == result["total_value"]

    def test_benchmark_functionality(self, tmp_path):
        """測試基準測試功能"""
        csv_path = "data/sample_ohlc.csv"
        output_dir = tmp_path / "benchmark_test"

        strategies = ["sma_crossover", "rsi"]

        results = run_benchmark(
            csv_path=csv_path,
            strategies=strategies,
            output_dir=str(output_dir)
        )

        # 檢查結果結構
        assert isinstance(results, dict)
        assert len(results) == len(strategies)

        for strategy in strategies:
            assert strategy in results
            assert "net_pnl" in results[strategy]
            assert "total_trades" in results[strategy]

        # 檢查輸出文件
        comparison_file = output_dir / "comparison_report.txt"
        assert comparison_file.exists()


class TestEdgeCases:
    """測試邊緣情況"""

    def test_very_small_dataset(self):
        """測試非常小的數據集"""
        dates = pd.date_range('2023-01-01', periods=5, freq='1H')
        prices = [100, 101, 102, 101, 100]

        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 5
        })

        result = simulate_trades_enhanced(test_data, initial_cash=10000.0)

        # 應該能夠處理小數據集而不崩潰
        assert "total_value" in result
        assert result["total_value"] >= 0

    def test_extreme_price_movements(self):
        """測試極端價格波動"""
        dates = pd.date_range('2023-01-01', periods=20, freq='1H')
        # 創建極端波動的價格
        prices = [100]
        for i in range(19):
            change = np.random.choice([-50, -20, -10, 10, 20, 50])
            prices.append(max(1, prices[-1] + change))  # 確保價格不為負

        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [max(p, p + abs(np.random.normal(0, 5))) for p in prices],
            'low': [max(1, min(p, p - abs(np.random.normal(0, 5)))) for p in prices],
            'close': prices,
            'volume': [1000] * 20
        })

        result = simulate_trades_enhanced(test_data, initial_cash=10000.0)

        # 應該能夠處理極端情況
        assert "total_value" in result
        assert result["total_value"] > 0  # 至少應該有初始資金


class TestPerformance:
    """測試性能相關功能"""

    def test_large_dataset_simulation(self):
        """測試大數據集模擬"""
        # 創建較大的測試數據集（1000個數據點）
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        prices = [100 + np.sin(i/10) * 10 + i * 0.01 for i in range(1000)]

        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 1000
        })

        import time
        start_time = time.time()

        result = simulate_trades_enhanced(test_data, initial_cash=10000.0)

        end_time = time.time()
        execution_time = end_time - start_time

        # 應該在合理時間內完成（小於30秒）
        assert execution_time < 30
        assert "total_value" in result


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])
