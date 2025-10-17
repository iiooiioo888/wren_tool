"""
延遲模擬器
負責模擬交易執行中的各種延遲，包括網路延遲、API延遲和處理延遲
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import random
import asyncio

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class DelayConfig:
    """延遲配置"""
    base_network_delay: float = 0.1  # 基礎網路延遲（秒）
    api_processing_delay: float = 0.05  # API處理延遲（秒）
    order_routing_delay: float = 0.02  # 訂單路由延遲（秒）
    market_data_delay: float = 0.01   # 市場數據延遲（秒）
    exchange_matching_delay: float = 0.03  # 交易所撮合延遲（秒）

    # 延遲變化因子
    network_jitter: float = 0.05  # 網路抖動
    load_factor: float = 1.0     # 負載因子
    time_of_day_factor: float = 1.0  # 時間因子

@dataclass
class DelayResult:
    """延遲計算結果"""
    total_delay: float
    network_delay: float
    api_delay: float
    routing_delay: float
    matching_delay: float
    market_data_delay: float
    breakdown: Dict[str, float]

@dataclass
class ExecutionSimulation:
    """執行模擬結果"""
    order_id: str
    submit_time: pd.Timestamp
    execution_time: pd.Timestamp
    total_delay: float
    delay_breakdown: Dict[str, float]
    execution_price: float
    fill_quantity: float
    status: str  # 'filled', 'partial', 'cancelled'

class DelaySimulator:
    """延遲模擬器"""

    def __init__(self, config: DelayConfig = None):
        """
        初始化延遲模擬器

        Args:
            config: 延遲配置
        """
        self.config = config or DelayConfig()

        # 延遲歷史記錄
        self.delay_history: List[DelayResult] = []

        # 模擬執行歷史
        self.execution_history: List[ExecutionSimulation] = []

    def simulate_order_delay(
        self,
        order_size: float = 1000,
        order_type: str = "market",
        exchange: str = "binance",
        time_of_day: int = 12,  # 24小時制
        network_load: float = 0.5  # 0-1之間
    ) -> DelayResult:
        """
        模擬訂單執行延遲

        Args:
            order_size: 訂單規模
            order_type: 訂單類型
            exchange: 交易所名稱
            time_of_day: 一天中的時間（小時）
            network_load: 網路負載

        Returns:
            延遲計算結果
        """
        logger.info(f"Simulating delay for {order_type} order of size {order_size} on {exchange}")

        # 計算各類延遲
        network_delay = self._calculate_network_delay(network_load)
        api_delay = self._calculate_api_delay(order_type, order_size)
        routing_delay = self._calculate_routing_delay(order_size)
        matching_delay = self._calculate_matching_delay(exchange, order_type)
        market_data_delay = self._calculate_market_data_delay()

        # 應用時間因子
        time_factor = self._calculate_time_factor(time_of_day)
        load_factor = self._calculate_load_factor(network_load)

        # 計算總延遲
        total_delay = (
            network_delay +
            api_delay +
            routing_delay +
            matching_delay +
            market_data_delay
        ) * time_factor * load_factor

        # 創建延遲分解
        breakdown = {
            'network_delay': network_delay * time_factor * load_factor,
            'api_delay': api_delay * time_factor * load_factor,
            'routing_delay': routing_delay * time_factor * load_factor,
            'matching_delay': matching_delay * time_factor * load_factor,
            'market_data_delay': market_data_delay * time_factor * load_factor,
            'time_factor': time_factor,
            'load_factor': load_factor
        }

        result = DelayResult(
            total_delay=total_delay,
            network_delay=network_delay,
            api_delay=api_delay,
            routing_delay=routing_delay,
            matching_delay=matching_delay,
            market_data_delay=market_data_delay,
            breakdown=breakdown
        )

        # 記錄歷史
        self.delay_history.append(result)

        # 限制歷史記錄長度
        if len(self.delay_history) > 10000:
            self.delay_history = self.delay_history[-5000:]

        logger.info(f"Total delay simulated: {total_delay:.3f}s")
        return result

    def _calculate_network_delay(self, network_load: float) -> float:
        """計算網路延遲"""
        # 基礎延遲 + 負載相關延遲 + 隨機抖動
        base_delay = self.config.base_network_delay
        load_delay = network_load * 0.2  # 負載增加0.2秒
        jitter = random.uniform(-self.config.network_jitter, self.config.network_jitter)

        return max(0.01, base_delay + load_delay + jitter)

    def _calculate_api_delay(self, order_type: str, order_size: float) -> float:
        """計算API處理延遲"""
        base_delay = self.config.api_processing_delay

        # 訂單類型影響
        type_multiplier = {
            'market': 1.0,
            'limit': 1.2,
            'stop': 1.3,
            'iceberg': 1.5
        }

        multiplier = type_multiplier.get(order_type, 1.0)

        # 訂單規模影響（大訂單需要更多處理時間）
        size_factor = min(2.0, 1 + (order_size / 100000) * 0.5)

        return base_delay * multiplier * size_factor

    def _calculate_routing_delay(self, order_size: float) -> float:
        """計算訂單路由延遲"""
        base_delay = self.config.order_routing_delay

        # 大訂單可能需要智能路由
        if order_size > 100000:
            routing_factor = 1.5
        elif order_size > 10000:
            routing_factor = 1.2
        else:
            routing_factor = 1.0

        return base_delay * routing_factor

    def _calculate_matching_delay(self, exchange: str, order_type: str) -> float:
        """計算交易所撮合延遲"""
        base_delay = self.config.exchange_matching_delay

        # 不同交易所的撮合速度不同
        exchange_speeds = {
            'binance': 0.8,
            'coinbase': 1.2,
            'kraken': 1.0,
            'kucoin': 1.1,
            'bybit': 0.9,
            'okx': 0.85
        }

        speed_factor = exchange_speeds.get(exchange, 1.0)

        # 訂單類型影響
        type_factor = 1.0 if order_type == 'market' else 1.3

        return base_delay * speed_factor * type_factor

    def _calculate_market_data_delay(self) -> float:
        """計算市場數據延遲"""
        return self.config.market_data_delay

    def _calculate_time_factor(self, hour: int) -> float:
        """計算時間因子"""
        # 不同時間段的延遲乘數
        if 9 <= hour <= 17:  # 交易高峰期
            return 1.5
        elif 18 <= hour <= 23:  # 亞洲/歐洲交易時段
            return 1.2
        else:  # 夜間
            return 0.8

    def _calculate_load_factor(self, network_load: float) -> float:
        """計算負載因子"""
        return 1.0 + network_load * 0.5

    async def simulate_realistic_delay(self, delay_result: DelayResult) -> None:
        """
        模擬真實延遲（異步）

        Args:
            delay_result: 延遲計算結果
        """
        # 添加一些隨機變化
        actual_delay = delay_result.total_delay * (0.8 + random.random() * 0.4)

        # 模擬延遲
        await asyncio.sleep(actual_delay)

        logger.debug(f"Simulated delay: {actual_delay:.3f}s")

    def simulate_order_execution(
        self,
        order_id: str,
        submit_time: pd.Timestamp,
        order_size: float,
        order_price: float,
        order_type: str = "market",
        exchange: str = "binance",
        time_of_day: int = 12,
        network_load: float = 0.5
    ) -> ExecutionSimulation:
        """
        模擬完整訂單執行過程

        Args:
            order_id: 訂單ID
            submit_time: 提交時間
            order_size: 訂單規模
            order_price: 訂單價格
            order_type: 訂單類型
            exchange: 交易所名稱
            time_of_day: 一天中的時間
            network_load: 網路負載

        Returns:
            執行模擬結果
        """
        # 計算延遲
        delay_result = self.simulate_order_delay(
            order_size, order_type, exchange, time_of_day, network_load
        )

        # 計算執行時間
        execution_time = submit_time + pd.Timedelta(seconds=delay_result.total_delay)

        # 模擬執行價格變化（基於延遲）
        price_change = np.random.normal(0, 0.001) * delay_result.total_delay  # 每秒0.1%價格變化
        execution_price = order_price * (1 + price_change)

        # 模擬成交數量（可能部分成交）
        if order_type == "market":
            # 市價單通常完全成交
            fill_quantity = order_size
            status = "filled"
        else:
            # 限價單可能部分成交
            fill_probability = min(1.0, 0.8 + random.random() * 0.2)
            fill_quantity = order_size * fill_probability
            status = "filled" if fill_probability > 0.95 else "partial"

        simulation = ExecutionSimulation(
            order_id=order_id,
            submit_time=submit_time,
            execution_time=execution_time,
            total_delay=delay_result.total_delay,
            delay_breakdown=delay_result.breakdown,
            execution_price=execution_price,
            fill_quantity=fill_quantity,
            status=status
        )

        # 記錄歷史
        self.execution_history.append(simulation)

        # 限制歷史記錄長度
        if len(self.execution_history) > 10000:
            self.execution_history = self.execution_history[-5000:]

        logger.info(f"Simulated execution for order {order_id}: "
                   f"{delay_result.total_delay:.3f}s delay, "
                   f"executed at ${execution_price:.2f}")

        return simulation

    def simulate_batch_execution(
        self,
        orders: List[Dict],
        time_of_day: int = 12,
        network_load: float = 0.5
    ) -> List[ExecutionSimulation]:
        """
        模擬批量訂單執行

        Args:
            orders: 訂單列表
            time_of_day: 一天中的時間
            network_load: 網路負載

        Returns:
            執行模擬結果列表
        """
        logger.info(f"Simulating batch execution for {len(orders)} orders")

        simulations = []

        for order in orders:
            simulation = self.simulate_order_execution(
                order_id=order.get('order_id', f"order_{len(simulations)}"),
                submit_time=pd.Timestamp.now(),
                order_size=order.get('size', 1000),
                order_price=order.get('price', 50000),
                order_type=order.get('type', 'market'),
                exchange=order.get('exchange', 'binance'),
                time_of_day=time_of_day,
                network_load=network_load
            )

            simulations.append(simulation)

        logger.info(f"Batch execution simulation completed: {len(simulations)} orders")
        return simulations

    def calculate_expected_delay(
        self,
        order_size: float,
        order_type: str = "market",
        exchange: str = "binance",
        num_simulations: int = 1000
    ) -> Dict[str, float]:
        """
        計算預期延遲（蒙特卡洛模擬）

        Args:
            order_size: 訂單規模
            order_type: 訂單類型
            exchange: 交易所名稱
            num_simulations: 模擬次數

        Returns:
            延遲統計結果
        """
        logger.info(f"Running Monte Carlo simulation for delay: {num_simulations} iterations")

        delays = []

        for _ in range(num_simulations):
            # 隨機參數
            time_of_day = random.randint(0, 23)
            network_load = random.uniform(0, 1)

            # 計算延遲
            delay_result = self.simulate_order_delay(
                order_size, order_type, exchange, time_of_day, network_load
            )

            delays.append(delay_result.total_delay)

        # 計算統計
        delays_array = np.array(delays)

        return {
            'mean_delay': np.mean(delays_array),
            'median_delay': np.median(delays_array),
            'std_delay': np.std(delays_array),
            'min_delay': np.min(delays_array),
            'max_delay': np.max(delays_array),
            'percentile_95': np.percentile(delays_array, 95),
            'percentile_99': np.percentile(delays_array, 99)
        }

    def analyze_delay_patterns(
        self,
        time_range: Tuple[str, str] = None,
        exchanges: List[str] = None
    ) -> Dict[str, Any]:
        """
        分析延遲模式

        Args:
            time_range: 時間範圍
            exchanges: 交易所列表

        Returns:
            延遲模式分析結果
        """
        if not self.delay_history:
            return {}

        # 過濾數據
        analysis_data = self.delay_history

        if time_range:
            start_time, end_time = time_range
            # 這裡可以添加時間過濾邏輯

        # 計算統計
        total_delays = [d.total_delay for d in analysis_data]
        network_delays = [d.network_delay for d in analysis_data]
        api_delays = [d.api_delay for d in analysis_data]

        # 按小時分析
        hourly_stats = self._calculate_hourly_delay_stats()

        # 按交易所分析
        exchange_stats = self._calculate_exchange_delay_stats()

        return {
            'overall_stats': {
                'mean_total_delay': np.mean(total_delays),
                'mean_network_delay': np.mean(network_delays),
                'mean_api_delay': np.mean(api_delays),
                'total_samples': len(analysis_data)
            },
            'hourly_patterns': hourly_stats,
            'exchange_patterns': exchange_stats,
            'delay_distribution': {
                'total_delay_percentiles': {
                    '50': np.percentile(total_delays, 50),
                    '90': np.percentile(total_delays, 90),
                    '95': np.percentile(total_delays, 95),
                    '99': np.percentile(total_delays, 99)
                }
            }
        }

    def _calculate_hourly_delay_stats(self) -> Dict[int, Dict[str, float]]:
        """計算每小時延遲統計"""
        # 這裡需要從執行歷史中提取時間信息
        # 簡化實現
        hourly_stats = {}

        for hour in range(24):
            # 模擬每小時的延遲統計
            base_delay = 0.1 + 0.05 * (hour >= 9 and hour <= 17)  # 高峰期延遲較高
            hourly_stats[hour] = {
                'mean_delay': base_delay + random.uniform(-0.02, 0.02),
                'sample_count': random.randint(10, 100)
            }

        return hourly_stats

    def _calculate_exchange_delay_stats(self) -> Dict[str, Dict[str, float]]:
        """計算各交易所延遲統計"""
        exchange_stats = {}

        exchanges = ['binance', 'coinbase', 'kraken', 'kucoin', 'bybit', 'okx']

        for exchange in exchanges:
            # 模擬不同交易所的延遲特徵
            base_delay = {
                'binance': 0.08,
                'coinbase': 0.15,
                'kraken': 0.12,
                'kucoin': 0.10,
                'bybit': 0.09,
                'okx': 0.085
            }.get(exchange, 0.1)

            exchange_stats[exchange] = {
                'mean_delay': base_delay + random.uniform(-0.01, 0.01),
                'reliability_score': random.uniform(0.95, 0.99)
            }

        return exchange_stats

    def generate_delay_report(self) -> str:
        """生成延遲分析報告"""
        if not self.delay_history:
            return "No delay data available"

        # 計算統計
        total_delays = [d.total_delay for d in self.delay_history]
        network_delays = [d.network_delay for d in self.delay_history]
        api_delays = [d.api_delay for d in self.delay_history]
        routing_delays = [d.routing_delay for d in self.delay_history]
        matching_delays = [d.matching_delay for d in self.delay_history]

        report = []
        report.append("=" * 60)
        report.append("延遲模擬報告")
        report.append("=" * 60)

        report.append(f"樣本數量: {len(self.delay_history)}")
        report.append("總延遲統計:")
        report.append(f"  平均延遲: {np.mean(total_delays):.3f}秒")
        report.append(f"  中位延遲: {np.median(total_delays):.3f}秒")
        report.append(f"  延遲範圍: {np.min(total_delays):.3f} - {np.max(total_delays):.3f}秒")
        report.append("")

        # 延遲分解
        report.append("延遲分解:")
        report.append(f"  網路延遲: {np.mean(network_delays):.3f}秒 ({np.mean(network_delays)/np.mean(total_delays)*100:.1f}%)")
        report.append(f"  API延遲: {np.mean(api_delays):.3f}秒 ({np.mean(api_delays)/np.mean(total_delays)*100:.1f}%)")
        report.append(f"  路由延遲: {np.mean(routing_delays):.3f}秒 ({np.mean(routing_delays)/np.mean(total_delays)*100:.1f}%)")
        report.append(f"  撮合延遲: {np.mean(matching_delays):.3f}秒 ({np.mean(matching_delays)/np.mean(total_delays)*100:.1f}%)")
        report.append("")

        # 執行統計
        if self.execution_history:
            execution_delays = [e.total_delay for e in self.execution_history]
            report.append("執行統計:")
            report.append(f"  平均執行延遲: {np.mean(execution_delays):.3f}秒")
            report.append(f"  執行成功率: {sum(1 for e in self.execution_history if e.status == 'filled') / len(self.execution_history)*100:.1f}%")
        report.append("")

        # 延遲分佈
        delay_bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        delay_distribution = np.histogram(total_delays, bins=delay_bins)[0]

        report.append("延遲分佈:")
        for i in range(len(delay_bins) - 1):
            percentage = delay_distribution[i] / len(total_delays) * 100
            report.append(f"  {delay_bins[i]:.2f} - {delay_bins[i+1]:.2f}秒: {percentage:.1f}%")

        report.append("=" * 60)

        return "\n".join(report)

    def export_delay_data(self, filepath: str) -> str:
        """導出延遲數據"""
        try:
            # 準備導出數據
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'delay_history': [
                    {
                        'total_delay': d.total_delay,
                        'network_delay': d.network_delay,
                        'api_delay': d.api_delay,
                        'routing_delay': d.routing_delay,
                        'matching_delay': d.matching_delay,
                        'market_data_delay': d.market_data_delay
                    }
                    for d in self.delay_history[-1000:]  # 最近1000筆記錄
                ],
                'execution_history': [
                    {
                        'order_id': e.order_id,
                        'submit_time': e.submit_time.isoformat(),
                        'execution_time': e.execution_time.isoformat(),
                        'total_delay': e.total_delay,
                        'execution_price': e.execution_price,
                        'fill_quantity': e.fill_quantity,
                        'status': e.status
                    }
                    for e in self.execution_history[-1000:]
                ]
            }

            df = pd.DataFrame(export_data['delay_history'])
            df.to_csv(filepath, index=False)

            logger.info(f"Exported delay data to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting delay data: {e}")
            raise


# 使用範例
if __name__ == "__main__":
    # 創建延遲模擬器
    config = DelayConfig(
        base_network_delay=0.1,
        api_processing_delay=0.05,
        network_jitter=0.02
    )

    delay_simulator = DelaySimulator(config)

    # 模擬單筆訂單延遲
    delay_result = delay_simulator.simulate_order_delay(
        order_size=10000,
        order_type="market",
        exchange="binance",
        time_of_day=14,  # 下午2點（高峰期）
        network_load=0.8  # 高負載
    )

    print(f"預計總延遲: {delay_result.total_delay:.3f}秒")
    print(f"網路延遲: {delay_result.network_delay:.3f}秒")
    print(f"API延遲: {delay_result.api_delay:.3f}秒")

    # 模擬訂單執行
    execution_sim = delay_simulator.simulate_order_execution(
        order_id="test_order_001",
        submit_time=pd.Timestamp.now(),
        order_size=50000,
        order_price=50000,
        order_type="limit",
        exchange="coinbase",
        time_of_day=10,
        network_load=0.3
    )

    print(f"\n訂單執行模擬:")
    print(f"訂單ID: {execution_sim.order_id}")
    print(f"提交時間: {execution_sim.submit_time}")
    print(f"執行時間: {execution_sim.execution_time}")
    print(f"執行價格: ${execution_sim.execution_price:.2f}")
    print(f"成交數量: {execution_sim.fill_quantity}")
    print(f"狀態: {execution_sim.status}")

    # 蒙特卡洛延遲分析
    delay_stats = delay_simulator.calculate_expected_delay(
        order_size=100000,
        order_type="market",
        exchange="binance",
        num_simulations=1000
    )

    print("\n蒙特卡洛延遲分析:")
    print(f"平均延遲: {delay_stats['mean_delay']:.3f}秒")
    print(f"95百分位延遲: {delay_stats['percentile_95']:.3f}秒")
    print(f"99百分位延遲: {delay_stats['percentile_99']:.3f}秒")

    # 生成報告
    report = delay_simulator.generate_delay_report()
    print("\n延遲分析報告:")
    print(report)
