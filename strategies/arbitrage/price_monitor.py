"""
跨市場價格監控系統
負責監控不同交易所之間的價格差異，識別套利機會
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import aiohttp
import time

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class ExchangePrice:
    """交易所價格信息"""
    exchange: str
    symbol: str
    price: float
    volume: float
    timestamp: pd.Timestamp
    bid_price: float = 0
    ask_price: float = 0
    spread: float = 0

@dataclass
class ArbitrageOpportunity:
    """套利機會"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    price_diff: float
    price_diff_pct: float
    estimated_profit: float
    estimated_profit_pct: float
    volume_buy: float
    volume_sell: float
    timestamp: pd.Timestamp
    confidence: float

@dataclass
class PriceMonitorConfig:
    """價格監控配置"""
    min_profit_threshold: float = 0.001  # 最小利潤閾值（0.1%）
    max_price_age: int = 30  # 最大價格時效（秒）
    min_volume_threshold: float = 1000  # 最小成交量閾值
    update_interval: int = 5  # 更新間隔（秒）
    max_opportunities: int = 100  # 最大機會數量

class PriceMonitor:
    """價格監控器"""

    def __init__(self, config: PriceMonitorConfig = None):
        """
        初始化價格監控器

        Args:
            config: 監控配置
        """
        self.config = config or PriceMonitorConfig()

        # 價格數據存儲
        self.price_data: Dict[str, Dict[str, ExchangePrice]] = {}  # symbol -> exchange -> price

        # 套利機會歷史
        self.opportunities: List[ArbitrageOpportunity] = []

        # 交易所列表
        self.exchanges = [
            'binance', 'coinbase', 'kraken', 'kucoin', 'bybit',
            'okx', 'huobi', 'gate', 'mexc', 'bitget'
        ]

        # 監控狀態
        self.is_monitoring = False
        self.last_update = {}

    def add_price_data(self, prices: List[ExchangePrice]) -> None:
        """
        添加價格數據

        Args:
            prices: 價格數據列表
        """
        current_time = pd.Timestamp.now()

        for price in prices:
            if price.symbol not in self.price_data:
                self.price_data[price.symbol] = {}

            # 更新價格數據
            self.price_data[price.symbol][price.exchange] = price
            self.last_update[price.exchange] = current_time

        logger.debug(f"Updated prices for {len(prices)} entries")

    def get_latest_prices(self, symbol: str) -> Dict[str, ExchangePrice]:
        """
        獲取最新價格

        Args:
            symbol: 交易對符號

        Returns:
            各交易所最新價格字典
        """
        if symbol not in self.price_data:
            return {}

        current_time = pd.Timestamp.now()
        latest_prices = {}

        for exchange, price in self.price_data[symbol].items():
            # 檢查價格時效
            age_seconds = (current_time - price.timestamp).total_seconds()

            if age_seconds <= self.config.max_price_age:
                latest_prices[exchange] = price
            else:
                logger.warning(f"Price data too old for {exchange}: {age_seconds}s")

        return latest_prices

    def find_arbitrage_opportunities(
        self,
        symbols: List[str] = None,
        min_profit_threshold: float = None
    ) -> List[ArbitrageOpportunity]:
        """
        尋找套利機會

        Args:
            symbols: 要檢查的交易對列表
            min_profit_threshold: 最小利潤閾值

        Returns:
            套利機會列表
        """
        if symbols is None:
            symbols = list(self.price_data.keys())

        if min_profit_threshold is None:
            min_profit_threshold = self.config.min_profit_threshold

        opportunities = []
        current_time = pd.Timestamp.now()

        for symbol in symbols:
            symbol_opportunities = self._find_symbol_opportunities(
                symbol, min_profit_threshold, current_time
            )

            opportunities.extend(symbol_opportunities)

        # 按潛在利潤排序
        opportunities.sort(key=lambda x: x.estimated_profit_pct, reverse=True)

        # 限制數量
        opportunities = opportunities[:self.config.max_opportunities]

        # 添加到歷史記錄
        self.opportunities.extend(opportunities)

        # 限制歷史記錄長度
        if len(self.opportunities) > 1000:
            self.opportunities = self.opportunities[-500:]

        logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        return opportunities

    def _find_symbol_opportunities(
        self,
        symbol: str,
        min_profit_threshold: float,
        current_time: pd.Timestamp
    ) -> List[ArbitrageOpportunity]:
        """為單個交易對尋找套利機會"""
        opportunities = []

        # 獲取最新價格
        latest_prices = self.get_latest_prices(symbol)

        if len(latest_prices) < 2:
            return opportunities

        exchanges = list(latest_prices.keys())

        # 檢查所有交易所配對
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                exchange1 = exchanges[i]
                exchange2 = exchanges[j]

                price1 = latest_prices[exchange1]
                price2 = latest_prices[exchange2]

                # 檢查成交量閾值
                if (price1.volume < self.config.min_volume_threshold or
                    price2.volume < self.config.min_volume_threshold):
                    continue

                # 計算價格差異
                if price1.price > price2.price:
                    # 在exchange2買入，在exchange1賣出
                    buy_exchange = exchange2
                    sell_exchange = exchange1
                    buy_price = price2.price
                    sell_price = price1.price
                else:
                    # 在exchange1買入，在exchange2賣出
                    buy_exchange = exchange1
                    sell_exchange = exchange2
                    buy_price = price1.price
                    sell_price = price2.price

                price_diff = sell_price - buy_price
                price_diff_pct = price_diff / buy_price

                # 檢查利潤閾值
                if price_diff_pct < min_profit_threshold:
                    continue

                # 估計利潤（考慮手續費）
                estimated_profit = self._estimate_arbitrage_profit(
                    buy_price, sell_price, price1.volume, price2.volume
                )

                estimated_profit_pct = estimated_profit / buy_price

                # 計算信心度
                confidence = self._calculate_opportunity_confidence(
                    price1, price2, price_diff_pct
                )

                opportunity = ArbitrageOpportunity(
                    symbol=symbol,
                    buy_exchange=buy_exchange,
                    sell_exchange=sell_exchange,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    price_diff=price_diff,
                    price_diff_pct=price_diff_pct,
                    estimated_profit=estimated_profit,
                    estimated_profit_pct=estimated_profit_pct,
                    volume_buy=price2.volume if buy_exchange == exchange2 else price1.volume,
                    volume_sell=price1.volume if sell_exchange == exchange1 else price2.volume,
                    timestamp=current_time,
                    confidence=confidence
                )

                opportunities.append(opportunity)

        return opportunities

    def _estimate_arbitrage_profit(
        self,
        buy_price: float,
        sell_price: float,
        volume_buy: float,
        volume_sell: float
    ) -> float:
        """估計套利利潤"""
        # 假設交易規模為兩個交易所最小成交量的50%
        trade_size = min(volume_buy, volume_sell) * 0.5

        if trade_size <= 0:
            return 0

        # 計算毛利潤
        gross_profit = (sell_price - buy_price) * trade_size

        # 估計交易成本（簡化：假設0.1%總成本）
        trading_cost = (buy_price + sell_price) * trade_size * 0.001

        # 估計轉帳成本（簡化：假設10美元固定成本）
        transfer_cost = 10

        net_profit = gross_profit - trading_cost - transfer_cost

        return max(0, net_profit)

    def _calculate_opportunity_confidence(
        self,
        price1: ExchangePrice,
        price2: ExchangePrice,
        price_diff_pct: float
    ) -> float:
        """計算機會信心度"""
        # 基於價格時效、成交量和價差大小計算信心度

        # 價格時效分數（越新越好）
        current_time = pd.Timestamp.now()
        age1 = (current_time - price1.timestamp).total_seconds()
        age2 = (current_time - price2.timestamp).total_seconds()

        age_score1 = max(0, 1 - age1 / self.config.max_price_age)
        age_score2 = max(0, 1 - age2 / self.config.max_price_age)
        age_score = (age_score1 + age_score2) / 2

        # 成交量分數（越大越好）
        avg_volume = (price1.volume + price2.volume) / 2
        volume_score = min(1.0, avg_volume / 100000)  # 假設10萬為滿分

        # 價差分數（適中為佳，太大可能有風險）
        if price_diff_pct <= 0.005:  # 0.5%以下
            diff_score = price_diff_pct / 0.005
        elif price_diff_pct <= 0.02:  # 0.5%-2%之間
            diff_score = 1.0
        else:  # 2%以上
            diff_score = max(0.5, 1 - (price_diff_pct - 0.02) / 0.03)

        # 加權綜合分數
        confidence = (
            age_score * 0.4 +
            volume_score * 0.3 +
            diff_score * 0.3
        )

        return confidence

    async def fetch_prices_async(
        self,
        symbols: List[str],
        exchanges: List[str] = None
    ) -> List[ExchangePrice]:
        """
        異步獲取價格數據

        Args:
            symbols: 交易對列表
            exchanges: 交易所列表

        Returns:
            價格數據列表
        """
        if exchanges is None:
            exchanges = self.exchanges

        prices = []

        # 創建異步任務
        tasks = []
        for symbol in symbols:
            for exchange in exchanges:
                task = self._fetch_single_price_async(symbol, exchange)
                tasks.append((symbol, exchange, task))

        # 執行所有任務
        results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)

        # 處理結果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                symbol, exchange, _ = tasks[i]
                logger.error(f"Error fetching price for {symbol} on {exchange}: {result}")
                continue

            if result:
                prices.append(result)

        logger.info(f"Fetched {len(prices)} price updates")
        return prices

    async def _fetch_single_price_async(
        self,
        symbol: str,
        exchange: str
    ) -> Optional[ExchangePrice]:
        """異步獲取單個價格"""
        try:
            # 這裡實現實際的API調用
            # 由於這是模擬，我們創建假數據

            # 模擬API延遲
            await asyncio.sleep(0.1)

            # 模擬價格數據
            base_price = self._get_base_price(symbol)
            exchange_spread = np.random.uniform(0.0005, 0.002)  # 0.05%-0.2%價差
            noise = np.random.normal(0, base_price * 0.001)  # 0.1%隨機波動

            price = base_price + noise

            # 模擬買賣價
            mid_price = price
            half_spread = mid_price * exchange_spread / 2

            bid_price = mid_price - half_spread
            ask_price = mid_price + half_spread

            # 模擬成交量
            volume = np.random.uniform(10000, 1000000)

            return ExchangePrice(
                exchange=exchange,
                symbol=symbol,
                price=mid_price,
                volume=volume,
                timestamp=pd.Timestamp.now(),
                bid_price=bid_price,
                ask_price=ask_price,
                spread=exchange_spread
            )

        except Exception as e:
            logger.error(f"Error in async price fetch for {symbol} on {exchange}: {e}")
            return None

    def _get_base_price(self, symbol: str) -> float:
        """獲取基準價格（簡化模擬）"""
        # 模擬不同交易對的基準價格
        base_prices = {
            'BTC': 50000,
            'ETH': 3000,
            'BNB': 300,
            'ADA': 0.5,
            'SOL': 100,
            'DOT': 10,
            'AVAX': 50,
            'MATIC': 1.0
        }

        return base_prices.get(symbol, 100)

    def start_monitoring(
        self,
        symbols: List[str],
        exchanges: List[str] = None,
        callback: Optional[callable] = None
    ) -> None:
        """
        開始價格監控

        Args:
            symbols: 要監控的交易對列表
            exchanges: 要監控的交易所列表
            callback: 發現機會時的回調函數
        """
        if exchanges is None:
            exchanges = self.exchanges

        self.is_monitoring = True

        def monitoring_loop():
            while self.is_monitoring:
                try:
                    # 獲取最新價格
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    prices = loop.run_until_complete(
                        self.fetch_prices_async(symbols, exchanges)
                    )

                    if prices:
                        self.add_price_data(prices)

                        # 尋找套利機會
                        opportunities = self.find_arbitrage_opportunities(symbols)

                        if opportunities and callback:
                            callback(opportunities)

                    loop.close()

                    # 等待下一個更新間隔
                    time.sleep(self.config.update_interval)

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(self.config.update_interval)

        # 啟動監控線程
        import threading
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

        logger.info(f"Started price monitoring for {len(symbols)} symbols on {len(exchanges)} exchanges")

    def stop_monitoring(self) -> None:
        """停止價格監控"""
        self.is_monitoring = False
        logger.info("Stopped price monitoring")

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """獲取監控統計"""
        return {
            'is_monitoring': self.is_monitoring,
            'tracked_symbols': len(self.price_data),
            'total_opportunities': len(self.opportunities),
            'last_update': self.last_update,
            'config': {
                'min_profit_threshold': self.config.min_profit_threshold,
                'update_interval': self.config.update_interval,
                'max_price_age': self.config.max_price_age
            }
        }

    def generate_monitoring_report(self) -> str:
        """生成監控報告"""
        stats = self.get_monitoring_stats()

        report = []
        report.append("=" * 60)
        report.append("價格監控報告")
        report.append("=" * 60)

        report.append(f"監控狀態: {'運行中' if stats['is_monitoring'] else '已停止'}")
        report.append(f"追蹤交易對數量: {stats['tracked_symbols']}")
        report.append(f"發現機會總數: {stats['total_opportunities']}")
        report.append("")

        # 配置信息
        config = stats['config']
        report.append("監控配置:")
        report.append(f"  最小利潤閾值: {config['min_profit_threshold']".2%"}")
        report.append(f"  更新間隔: {config['update_interval']}秒")
        report.append(f"  最大價格時效: {config['max_price_age']}秒")
        report.append("")

        # 最近機會
        if self.opportunities:
            report.append("最近套利機會:")
            recent_opportunities = self.opportunities[-10:]  # 最近10個機會

            for opp in recent_opportunities:
                report.append(f"  {opp.symbol}: {opp.buy_exchange} -> {opp.sell_exchange}")
                report.append(f"    價差: {opp.price_diff_pct".2%"}")
                report.append(f"    預估利潤: {opp.estimated_profit_pct".2%"}")
                report.append(f"    信心度: {opp.confidence".2f"}")
        else:
            report.append("尚未發現套利機會")

        report.append("=" * 60)

        return "\n".join(report)


# 使用範例
if __name__ == "__main__":
    # 創建價格監控器
    config = PriceMonitorConfig(
        min_profit_threshold=0.001,  # 0.1%
        update_interval=2,  # 2秒更新
        min_volume_threshold=10000
    )

    monitor = PriceMonitor(config)

    # 創建範例價格數據
    current_time = pd.Timestamp.now()

    sample_prices = [
        ExchangePrice('binance', 'BTC', 50000, 1000000, current_time, 49990, 50010, 0.0004),
        ExchangePrice('coinbase', 'BTC', 50100, 500000, current_time, 50090, 50110, 0.0004),
        ExchangePrice('kraken', 'BTC', 50050, 200000, current_time, 50040, 50060, 0.0004),
        ExchangePrice('binance', 'ETH', 3000, 500000, current_time, 2995, 3005, 0.003),
        ExchangePrice('coinbase', 'ETH', 3010, 300000, current_time, 3005, 3015, 0.003),
    ]

    # 添加價格數據
    monitor.add_price_data(sample_prices)

    # 尋找套利機會
    opportunities = monitor.find_arbitrage_opportunities(['BTC', 'ETH'])

    print(f"發現 {len(opportunities)} 個套利機會:")

    for opp in opportunities:
        print(f"{opp.symbol}: {opp.buy_exchange} (${opp.buy_price:.2f}) -> "
              f"{opp.sell_exchange} (${opp.sell_price:.2f})")
        print(f"  價差: {opp.price_diff_pct:.2%}")
        print(f"  預估利潤: ${opp.estimated_profit:.2f} ({opp.estimated_profit_pct:.2%})")
        print(f"  信心度: {opp.confidence:.2f}")
        print()

    # 生成報告
    report = monitor.generate_monitoring_report()
    print("監控報告:")
    print(report)
