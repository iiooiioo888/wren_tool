"""
模擬器模組的單元測試
測試延遲模擬器和滑點模型的功能
"""
import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch

# 導入被測試模組
from simulation.execution.delay_simulator import DelaySimulator, DelayConfig, DelayResult
from simulation.costs.slippage_model import SlippageModel, SlippageModelConfig, SlippageResult


class TestDelaySimulator:
    """測試延遲模擬器"""

    def test_delay_simulator_initialization(self):
        """測試延遲模擬器初始化"""
        config = DelayConfig(
            base_network_delay=0.1,
            api_processing_delay=0.05,
            network_jitter=0.02
        )

        simulator = DelaySimulator(config)

        assert simulator.config.base_network_delay == 0.1
        assert simulator.config.api_processing_delay == 0.05
        assert len(simulator.delay_history) == 0

    def test_delay_simulator_default_config(self):
        """測試默認配置"""
        simulator = DelaySimulator()

        assert simulator.config.base_network_delay == 0.1
        assert simulator.config.api_processing_delay == 0.05
        assert simulator.config.network_jitter == 0.05

    def test_simulate_order_delay(self):
        """測試訂單延遲模擬"""
        simulator = DelaySimulator()

        delay_result = simulator.simulate_order_delay(
            order_size=10000,
            order_type="market",
            exchange="binance",
            time_of_day=14,
            network_load=0.5
        )

        # 檢查結果結構
        assert isinstance(delay_result, DelayResult)
        assert delay_result.total_delay > 0
        assert delay_result.network_delay >= 0
        assert delay_result.api_delay >= 0
        assert delay_result.routing_delay >= 0
        assert delay_result.matching_delay >= 0

        # 檢查總延遲是各部分之和
        expected_total = (delay_result.network_delay + delay_result.api_delay +
                         delay_result.routing_delay + delay_result.matching_delay +
                         delay_result.market_data_delay)
        assert abs(delay_result.total_delay - expected_total) < 0.001

    def test_simulate_order_delay_with_different_types(self):
        """測試不同訂單類型的延遲模擬"""
        simulator = DelaySimulator()

        order_types = ["market", "limit", "stop", "iceberg"]

        for order_type in order_types:
            delay_result = simulator.simulate_order_delay(
                order_size=10000,
                order_type=order_type,
                exchange="binance",
                time_of_day=14,
                network_load=0.5
            )

            assert delay_result.total_delay > 0

    def test_simulate_order_delay_with_different_exchanges(self):
        """測試不同交易所的延遲模擬"""
        simulator = DelaySimulator()

        exchanges = ["binance", "coinbase", "kraken", "kucoin", "bybit", "okx"]

        delays = []
        for exchange in exchanges:
            delay_result = simulator.simulate_order_delay(
                order_size=10000,
                order_type="market",
                exchange=exchange,
                time_of_day=14,
                network_load=0.5
            )
            delays.append(delay_result.total_delay)

        # 不同交易所應該有不同的延遲
        assert len(set(delays)) > 1

    def test_simulate_order_execution(self):
        """測試訂單執行模擬"""
        simulator = DelaySimulator()

        execution_result = simulator.simulate_order_execution(
            order_id="test_order_001",
            submit_time=pd.Timestamp.now(),
            order_size=50000,
            order_price=50000,
            order_type="limit",
            exchange="binance",
            time_of_day=10,
            network_load=0.3
        )

        # 檢查結果結構
        assert execution_result.order_id == "test_order_001"
        assert execution_result.total_delay >= 0
        assert execution_result.execution_price > 0
        assert execution_result.fill_quantity > 0
        assert execution_result.status in ["filled", "partial", "cancelled"]

    def test_calculate_expected_delay(self):
        """測試預期延遲計算"""
        simulator = DelaySimulator()

        delay_stats = simulator.calculate_expected_delay(
            order_size=100000,
            order_type="market",
            exchange="binance",
            num_simulations=100
        )

        # 檢查統計結果結構
        expected_keys = [
            "mean_delay", "median_delay", "std_delay", "min_delay",
            "max_delay", "percentile_95", "percentile_99"
        ]

        for key in expected_keys:
            assert key in delay_stats
            assert delay_stats[key] >= 0

        # 檢查統計合理性
        assert delay_stats["min_delay"] <= delay_stats["mean_delay"] <= delay_stats["max_delay"]
        assert delay_stats["percentile_95"] <= delay_stats["percentile_99"]

    def test_delay_history_tracking(self):
        """測試延遲歷史記錄"""
        simulator = DelaySimulator()

        # 執行多次模擬
        for i in range(5):
            simulator.simulate_order_delay(
                order_size=10000,
                order_type="market",
                exchange="binance",
                time_of_day=14,
                network_load=0.5
            )

        # 檢查歷史記錄
        assert len(simulator.delay_history) == 5

        # 檢查歷史限制
        for i in range(100):  # 超過歷史限制
            simulator.simulate_order_delay(
                order_size=10000,
                order_type="market",
                exchange="binance",
                time_of_day=14,
                network_load=0.5
            )

        # 歷史應該被限制在合理範圍內
        assert len(simulator.delay_history) <= 10000

    def test_generate_delay_report(self):
        """測試延遲報告生成"""
        simulator = DelaySimulator()

        # 添加一些測試數據
        for i in range(10):
            simulator.simulate_order_delay(
                order_size=10000,
                order_type="market",
                exchange="binance",
                time_of_day=14,
                network_load=0.5
            )

        report = simulator.generate_delay_report()

        assert isinstance(report, str)
        assert len(report) > 0
        assert "延遲模擬報告" in report
        assert "樣本數量" in report

    def test_export_delay_data(self, tmp_path):
        """測試延遲數據導出"""
        simulator = DelaySimulator()

        # 添加測試數據
        for i in range(5):
            simulator.simulate_order_delay(
                order_size=10000,
                order_type="market",
                exchange="binance",
                time_of_day=14,
                network_load=0.5
            )

        export_file = tmp_path / "delay_data.csv"

        result_file = simulator.export_delay_data(str(export_file))

        assert result_file == str(export_file)
        assert export_file.exists()

        # 檢查導出文件內容
        df = pd.read_csv(export_file)
        assert len(df) > 0
        assert "total_delay" in df.columns


class TestSlippageModel:
    """測試滑點模型"""

    def test_slippage_model_initialization(self):
        """測試滑點模型初始化"""
        config = SlippageModelConfig(
            method="square_root",
            market_impact_factor=0.001,
            volume_participation_rate=0.1
        )

        model = SlippageModel(config)

        assert model.config.method == "square_root"
        assert model.config.market_impact_factor == 0.001
        assert len(model.historical_slippage) == 0

    def test_calculate_slippage_market_order(self):
        """測試市價單滑點計算"""
        model = SlippageModel()

        # 創建測試市場數據
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'symbol': ['BTC'] * 100,
            'close': [50000 + i * 10 for i in range(100)],
            'volume': [1000000] * 100
        })

        slippage_result = model.calculate_slippage(
            symbol="BTC",
            order_size=100000,
            order_price=50000,
            order_type="market",
            market_data=market_data
        )

        # 檢查結果結構
        assert isinstance(slippage_result, SlippageResult)
        assert slippage_result.original_price == 50000
        assert slippage_result.executed_price >= 0
        assert slippage_result.slippage_amount >= 0
        assert slippage_result.slippage_percentage >= 0
        assert slippage_result.market_impact >= 0
        assert slippage_result.estimated_fill_time >= 0
        assert 0 <= slippage_result.confidence <= 1

    def test_calculate_slippage_limit_order(self):
        """測試限價單滑點計算"""
        model = SlippageModel()

        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'symbol': ['BTC'] * 100,
            'close': [50000 + i * 10 for i in range(100)],
            'volume': [1000000] * 100
        })

        slippage_result = model.calculate_slippage(
            symbol="BTC",
            order_size=100000,
            order_price=50000,
            order_type="limit",
            market_data=market_data
        )

        assert isinstance(slippage_result, SlippageResult)
        assert slippage_result.original_price == 50000

    def test_calculate_slippage_iceberg_order(self):
        """測試冰山單滑點計算"""
        model = SlippageModel()

        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'symbol': ['BTC'] * 100,
            'close': [50000 + i * 10 for i in range(100)],
            'volume': [1000000] * 100
        })

        slippage_result = model.calculate_slippage(
            symbol="BTC",
            order_size=100000,
            order_price=50000,
            order_type="iceberg",
            market_data=market_data
        )

        assert isinstance(slippage_result, SlippageResult)

    def test_estimate_optimal_order_size(self):
        """測試最優訂單規模估計"""
        model = SlippageModel()

        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'symbol': ['BTC'] * 100,
            'close': [50000 + i * 10 for i in range(100)],
            'volume': [1000000] * 100
        })

        optimal_size = model.estimate_optimal_order_size(
            symbol="BTC",
            max_slippage_pct=0.01,
            market_data=market_data,
            target_execution_time=3600
        )

        assert optimal_size > 0
        assert isinstance(optimal_size, (int, float))

    def test_simulate_order_execution(self):
        """測試訂單執行模擬"""
        model = SlippageModel()

        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'symbol': ['BTC'] * 100,
            'close': [50000 + i * 10 for i in range(100)],
            'volume': [1000000] * 100
        })

        execution_results = model.simulate_order_execution(
            symbol="BTC",
            order_size=100000,
            order_price=50000,
            order_type="iceberg",
            num_slices=5,
            market_data=market_data
        )

        assert isinstance(execution_results, list)
        assert len(execution_results) == 5  # 應該有5個分割

        for result in execution_results:
            assert isinstance(result, SlippageResult)

    def test_calculate_volume_weighted_slippage(self):
        """測試體積加權滑點計算"""
        model = SlippageModel()

        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'symbol': ['BTC'] * 100,
            'close': [50000 + i * 10 for i in range(100)],
            'volume': [1000000] * 100
        })

        vwap_result = model.calculate_volume_weighted_slippage(
            symbol="BTC",
            total_order_size=1000000,
            market_data=market_data,
            time_window=3600
        )

        # 檢查結果結構
        expected_keys = [
            "avg_slippage", "max_slippage", "min_slippage",
            "weighted_slippage", "num_windows", "window_size"
        ]

        for key in expected_keys:
            assert key in vwap_result

        # 檢查數值合理性
        assert vwap_result["avg_slippage"] >= 0
        assert vwap_result["weighted_slippage"] >= 0
        assert vwap_result["num_windows"] > 0

    def test_generate_slippage_report(self):
        """測試滑點報告生成"""
        model = SlippageModel()

        # 創建測試結果
        test_results = []
        for i in range(10):
            result = model.calculate_slippage(
                symbol="BTC",
                order_size=100000,
                order_price=50000,
                order_type="market"
            )
            test_results.append(result)

        report = model.generate_slippage_report(test_results, "BTC")

        assert isinstance(report, str)
        assert len(report) > 0
        assert "滑點分析報告" in report
        assert "樣本數量" in report

    def test_slippage_model_with_different_methods(self):
        """測試不同滑點計算方法的模型"""
        methods = ["square_root", "linear"]

        for method in methods:
            config = SlippageModelConfig(method=method)
            model = SlippageModel(config)

            result = model.calculate_slippage(
                symbol="BTC",
                order_size=100000,
                order_price=50000,
                order_type="market"
            )

            assert isinstance(result, SlippageResult)
            assert result.slippage_amount >= 0


class TestIntegrationBetweenSimulators:
    """測試模擬器之間的整合"""

    def test_delay_and_slippage_interaction(self):
        """測試延遲和滑點的交互影響"""
        # 這個測試確保兩個模擬器可以一起工作
        try:
            from simulation.execution.delay_simulator import DelaySimulator, DelayConfig
            from simulation.costs.slippage_model import SlippageModel, SlippageModelConfig

            delay_simulator = DelaySimulator()
            slippage_model = SlippageModel()

            # 模擬延遲計算
            delay_result = delay_simulator.simulate_order_delay(
                order_size=10000,
                order_type="market",
                exchange="binance",
                time_of_day=14,
                network_load=0.5
            )

            # 模擬滑點計算
            slippage_result = slippage_model.calculate_slippage(
                symbol="BTC",
                order_size=10000,
                order_price=50000,
                order_type="market"
            )

            # 確保兩個模擬器都能正常工作
            assert delay_result.total_delay >= 0
            assert slippage_result.slippage_amount >= 0

        except ImportError:
            # 如果模擬器不可用，跳過測試
            pytest.skip("模擬器模組不可用")


class TestPerformanceAndStress:
    """測試性能和壓力情況"""

    def test_delay_simulator_performance(self):
        """測試延遲模擬器性能"""
        simulator = DelaySimulator()

        start_time = time.time()

        # 執行大量模擬
        for i in range(1000):
            simulator.simulate_order_delay(
                order_size=10000,
                order_type="market",
                exchange="binance",
                time_of_day=14,
                network_load=0.5
            )

        end_time = time.time()
        execution_time = end_time - start_time

        # 應該在合理時間內完成（小於10秒）
        assert execution_time < 10

    def test_slippage_model_performance(self):
        """測試滑點模型性能"""
        model = SlippageModel()

        # 創建測試數據
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1H'),
            'symbol': ['BTC'] * 1000,
            'close': [50000 + i * 10 for i in range(1000)],
            'volume': [1000000] * 1000
        })

        start_time = time.time()

        # 執行大量滑點計算
        for i in range(100):
            model.calculate_slippage(
                symbol="BTC",
                order_size=100000,
                order_price=50000,
                order_type="market",
                market_data=market_data
            )

        end_time = time.time()
        execution_time = end_time - start_time

        # 應該在合理時間內完成（小於10秒）
        assert execution_time < 10


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])
