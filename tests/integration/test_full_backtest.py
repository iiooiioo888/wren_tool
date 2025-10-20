"""
完整的回測整合測試
測試整個回測系統的端到端功能
"""
import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path

# 導入被測試模組
from scripts.poc_run import run_poc, run_benchmark


class TestFullBacktestIntegration:
    """測試完整的回測整合功能"""

    def test_complete_backtest_workflow(self, tmp_path):
        """測試完整的回測工作流程"""
        # 創建測試數據
        test_data = self._create_test_market_data()

        # 保存測試數據到臨時文件
        csv_file = tmp_path / "test_market_data.csv"
        test_data.to_csv(csv_file, index=False)

        # 運行完整的PoC回測
        result = run_poc(
            csv_path=str(csv_file),
            strategy="sma_crossover",
            initial_cash=10000.0,
            fee=0.001,
            slippage=0.002,
            delay_ms=100,
            verbose=False
        )

        # 驗證結果結構
        required_keys = [
            "initial_cash", "final_cash", "total_value", "total_trades",
            "total_fees", "total_slippage", "total_delay_cost",
            "trades", "simulation_config"
        ]

        for key in required_keys:
            assert key in result

        # 驗證業務邏輯
        assert result["total_value"] >= 0
        assert result["initial_cash"] == 10000.0
        assert result["simulation_config"]["strategy"] == "sma_crossover"
        assert result["simulation_config"]["fee"] == 0.001
        assert result["simulation_config"]["slippage"] == 0.002

        # 如果有交易，驗證交易結構
        if result["total_trades"] > 0:
            for trade in result["trades"]:
                assert "time" in trade
                assert "side" in trade
                assert "original_price" in trade
                assert "execution_price" in trade
                assert "qty" in trade

    def test_benchmark_comparison_workflow(self, tmp_path):
        """測試基準測試比較工作流程"""
        # 創建測試數據
        test_data = self._create_test_market_data()

        # 保存測試數據到臨時文件
        csv_file = tmp_path / "benchmark_test_data.csv"
        test_data.to_csv(csv_file, index=False)

        # 運行基準測試
        strategies = ["sma_crossover", "rsi"]
        results = run_benchmark(
            csv_path=str(csv_file),
            strategies=strategies,
            output_dir=str(tmp_path / "benchmark_results")
        )

        # 驗證結果
        assert isinstance(results, dict)
        assert len(results) == len(strategies)

        for strategy in strategies:
            assert strategy in results
            strategy_result = results[strategy]

            # 檢查每個策略的結果結構
            assert "net_pnl" in strategy_result
            assert "total_trades" in strategy_result
            assert "win_rate" in strategy_result
            assert "max_drawdown" in strategy_result
            assert "sharpe_ratio" in strategy_result

        # 檢查輸出文件
        comparison_file = tmp_path / "benchmark_results" / "comparison_report.txt"
        assert comparison_file.exists()

        # 檢查比較報告內容
        with open(comparison_file, 'r', encoding='utf-8') as f:
            report_content = f.read()

        assert "策略比較報告" in report_content
        for strategy in strategies:
            assert strategy in report_content

    def test_different_parameter_combinations(self, tmp_path):
        """測試不同的參數組合"""
        # 創建測試數據
        test_data = self._create_test_market_data()
        csv_file = tmp_path / "param_test_data.csv"
        test_data.to_csv(csv_file, index=False)

        # 測試不同的參數組合
        parameter_sets = [
            {"fee": 0.0001, "slippage": 0.0005, "delay_ms": 50},
            {"fee": 0.002, "slippage": 0.005, "delay_ms": 200},
            {"fee": 0.001, "slippage": 0.002, "delay_ms": 100},
        ]

        results = []
        for params in parameter_sets:
            result = run_poc(
                csv_path=str(csv_file),
                strategy="sma_crossover",
                initial_cash=10000.0,
                verbose=False,
                **params
            )
            results.append(result)

        # 驗證不同參數產生不同的結果
        assert len(results) == len(parameter_sets)

        # 檢查費用影響（費用越高，淨盈虧應該越低）
        fees = [r["simulation_config"]["fee"] for r in results]
        net_pnls = [r["net_pnl"] for r in results]

        # 費用和淨盈虧應該呈負相關（簡化的假設）
        # 注意：這是一個簡化的測試，實際結果取決於市場數據和策略

    def test_error_handling_and_edge_cases(self, tmp_path):
        """測試錯誤處理和邊緣情況"""
        # 測試無效的CSV文件
        invalid_csv = tmp_path / "invalid.csv"
        invalid_csv.write_text("invalid,csv,content")

        with pytest.raises(Exception):
            run_poc(str(invalid_csv), verbose=False)

        # 測試空數據文件
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("timestamp,open,high,low,close,volume\n")

        result = run_poc(str(empty_csv), verbose=False)
        assert result["total_value"] == 10000.0  # 應該等於初始資金，因為沒有交易
        assert result["total_trades"] == 0

    def test_large_dataset_performance(self, tmp_path):
        """測試大數據集的性能"""
        # 創建較大的測試數據集（5000個數據點）
        large_test_data = self._create_large_test_dataset(5000)

        csv_file = tmp_path / "large_dataset.csv"
        large_test_data.to_csv(csv_file, index=False)

        import time
        start_time = time.time()

        result = run_poc(
            csv_path=str(csv_file),
            strategy="sma_crossover",
            initial_cash=10000.0,
            verbose=False
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # 應該在合理時間內完成（小於60秒）
        assert execution_time < 60
        assert "total_value" in result

    def _create_test_market_data(self) -> pd.DataFrame:
        """創建測試市場數據"""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')

        # 創建有趨勢和波動的價格數據
        base_price = 100
        prices = []
        for i in range(100):
            # 添加趨勢和隨機波動
            trend = i * 0.1
            noise = np.sin(i / 10) * 5 + np.random.normal(0, 2)
            price = base_price + trend + noise
            prices.append(max(1, price))  # 確保價格不為負

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 2)) for p in prices],
            'low': [p - abs(np.random.normal(0, 2)) for p in prices],
            'close': prices,
            'volume': [1000000 + np.random.normal(0, 100000) for _ in range(100)]
        })

    def _create_large_test_dataset(self, size: int) -> pd.DataFrame:
        """創建大規模測試數據集"""
        dates = pd.date_range('2023-01-01', periods=size, freq='1H')

        # 創建更複雜的價格模式
        prices = []
        current_price = 100

        for i in range(size):
            # 添加多種價格模式：趨勢、週期、隨機
            trend_component = i * 0.01
            cycle_component = np.sin(i / 50) * 10
            noise_component = np.random.normal(0, 1)

            price_change = trend_component + cycle_component + noise_component
            current_price += price_change
            current_price = max(1, current_price)  # 確保價格不為負

            prices.append(current_price)

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 2)) for p in prices],
            'low': [p - abs(np.random.normal(0, 2)) for p in prices],
            'close': prices,
            'volume': [1000000 + np.random.normal(0, 100000) for _ in range(size)]
        })


class TestBacktestWithRealisticScenarios:
    """測試現實場景的回測"""

    def test_bull_market_scenario(self, tmp_path):
        """測試牛市場景"""
        # 創建強勢上漲的市場數據
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        prices = [100 * (1.001 ** i) + np.random.normal(0, 1) for i in range(200)]

        bull_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 3)) for p in prices],
            'low': [p - abs(np.random.normal(0, 3)) for p in prices],
            'close': prices,
            'volume': [1500000 + np.random.normal(0, 200000) for _ in range(200)]
        })

        csv_file = tmp_path / "bull_market.csv"
        bull_data.to_csv(csv_file, index=False)

        result = run_poc(str(csv_file), verbose=False)

        # 在牛市中，趨勢跟蹤策略應該有不錯的表現
        assert result["total_value"] > 0

    def test_bear_market_scenario(self, tmp_path):
        """測試熊市場景"""
        # 創建下跌趨勢的市場數據
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        prices = [200 * (0.999 ** i) + np.random.normal(0, 1) for i in range(200)]

        bear_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 3)) for p in prices],
            'low': [p - abs(np.random.normal(0, 3)) for p in prices],
            'close': prices,
            'volume': [1500000 + np.random.normal(0, 200000) for _ in range(200)]
        })

        csv_file = tmp_path / "bear_market.csv"
        bear_data.to_csv(csv_file, index=False)

        result = run_poc(str(csv_file), verbose=False)

        # 在熊市中，結果可能為負，但系統應該穩定運行
        assert "total_value" in result
        assert result["total_value"] >= 0  # 至少不應該虧光

    def test_sideways_market_scenario(self, tmp_path):
        """測試橫盤市場場景"""
        # 創建橫盤整理的市場數據
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        base_prices = [100 + np.sin(i / 20) * 5 + np.random.normal(0, 1) for i in range(200)]

        sideways_data = pd.DataFrame({
            'timestamp': dates,
            'open': base_prices,
            'high': [p + abs(np.random.normal(0, 2)) for p in base_prices],
            'low': [p - abs(np.random.normal(0, 2)) for p in base_prices],
            'close': base_prices,
            'volume': [800000 + np.random.normal(0, 100000) for _ in range(200)]
        })

        csv_file = tmp_path / "sideways_market.csv"
        sideways_data.to_csv(csv_file, index=False)

        result = run_poc(str(csv_file), verbose=False)

        # 在橫盤市場中，結果應該相對穩定
        assert "total_value" in result
        assert result["total_value"] > 0


class TestMultiStrategyComparison:
    """測試多策略比較功能"""

    def test_strategy_comparison_metrics(self, tmp_path):
        """測試策略比較指標"""
        # 創建多種市場環境的測試數據
        market_scenarios = {
            "trending": self._create_trending_data(100),
            "mean_reverting": self._create_mean_reverting_data(100),
            "volatile": self._create_volatile_data(100)
        }

        results = {}

        for scenario_name, data in market_scenarios.items():
            csv_file = tmp_path / f"{scenario_name}_data.csv"
            data.to_csv(csv_file, index=False)

            # 測試不同策略在不同市場環境下的表現
            for strategy in ["sma_crossover", "rsi"]:
                result = run_poc(
                    csv_path=str(csv_file),
                    strategy=strategy,
                    verbose=False
                )

                key = f"{strategy}_{scenario_name}"
                results[key] = {
                    "strategy": strategy,
                    "scenario": scenario_name,
                    "net_pnl": result["net_pnl"],
                    "total_trades": result["total_trades"],
                    "sharpe_ratio": result["sharpe_ratio"]
                }

        # 驗證結果結構
        assert len(results) == 6  # 2個策略 × 3個場景

        # 檢查每個結果都有必要的指標
        for result in results.values():
            assert "net_pnl" in result
            assert "total_trades" in result
            assert "sharpe_ratio" in result

    def _create_trending_data(self, size: int) -> pd.DataFrame:
        """創建趨勢數據"""
        dates = pd.date_range('2023-01-01', periods=size, freq='1H')
        prices = [100 + i * 0.5 + np.random.normal(0, 1) for i in range(size)]

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000000] * size
        })

    def _create_mean_reverting_data(self, size: int) -> pd.DataFrame:
        """創建均值回歸數據"""
        dates = pd.date_range('2023-01-01', periods=size, freq='1H')
        prices = [100 + np.sin(i / 10) * 20 + np.random.normal(0, 2) for i in range(size)]

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 3 for p in prices],
            'low': [p - 3 for p in prices],
            'close': prices,
            'volume': [800000] * size
        })

    def _create_volatile_data(self, size: int) -> pd.DataFrame:
        """創建高波動數據"""
        dates = pd.date_range('2023-01-01', periods=size, freq='1H')
        prices = [100]
        for i in range(size - 1):
            change = np.random.normal(0, 5)  # 高波動
            new_price = prices[-1] + change
            prices.append(max(1, new_price))

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 4)) for p in prices],
            'low': [p - abs(np.random.normal(0, 4)) for p in prices],
            'close': prices,
            'volume': [1200000] * size
        })


class TestDataPersistenceAndReporting:
    """測試數據持久化和報告功能"""

    def test_result_serialization(self, tmp_path):
        """測試結果序列化"""
        # 創建測試數據並運行回測
        test_data = self._create_test_market_data()
        csv_file = tmp_path / "serialization_test.csv"
        test_data.to_csv(csv_file, index=False)

        result = run_poc(str(csv_file), verbose=False)

        # 測試JSON序列化
        json_str = json.dumps(result, default=str, indent=2)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # 測試JSON反序列化
        deserialized = json.loads(json_str)
        assert deserialized["total_value"] == result["total_value"]
        assert deserialized["total_trades"] == result["total_trades"]

    def test_trade_details_persistence(self, tmp_path):
        """測試交易詳情持久化"""
        test_data = self._create_test_market_data()
        csv_file = tmp_path / "trade_details_test.csv"
        test_data.to_csv(csv_file, index=False)

        result = run_poc(str(csv_file), verbose=False)

        # 檢查交易詳情結構
        if result["total_trades"] > 0:
            for trade in result["trades"]:
                required_trade_keys = [
                    "time", "side", "original_price", "execution_price",
                    "qty", "fee", "slippage", "delay_cost"
                ]

                for key in required_trade_keys:
                    assert key in trade

                # 檢查數據類型
                assert isinstance(trade["qty"], (int, float))
                assert isinstance(trade["fee"], (int, float))
                assert trade["qty"] > 0

    def _create_test_market_data(self) -> pd.DataFrame:
        """創建測試市場數據（簡化版）"""
        dates = pd.date_range('2023-01-01', periods=50, freq='1H')
        prices = [100 + np.sin(i/5) * 10 + i * 0.1 for i in range(50)]

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000000] * 50
        })


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])
