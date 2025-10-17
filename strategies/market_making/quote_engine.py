"""
報價計算引擎
負責做市策略中的報價計算和買賣價差優化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class QuoteConfig:
    """報價配置"""
    base_spread: float = 0.001  # 基礎價差（0.1%）
    min_spread: float = 0.0005  # 最小價差
    max_spread: float = 0.01    # 最大價差
    spread_multiplier: float = 1.0  # 價差乘數
    inventory_skew_factor: float = 0.1  # 庫存傾斜因子
    volatility_adjustment: bool = True  # 是否根據波動率調整價差
    volume_adjustment: bool = True     # 是否根據成交量調整價差
    urgency_factor: float = 0.5        # 緊急程度因子

@dataclass
class Quote:
    """報價"""
    timestamp: pd.Timestamp
    symbol: str
    bid_price: float
    ask_price: float
    bid_quantity: float
    ask_quantity: float
    spread: float
    mid_price: float
    inventory_level: float  # 庫存水平（-1到1之間）

@dataclass
class QuoteResult:
    """報價結果"""
    quotes: List[Quote]
    total_quotes: int
    avg_spread: float
    max_spread: float
    min_spread: float
    performance_metrics: Dict[str, float]

class QuoteEngine:
    """報價引擎"""

    def __init__(self, config: QuoteConfig = None):
        """
        初始化報價引擎

        Args:
            config: 報價配置
        """
        self.config = config or QuoteConfig()
        self.inventory_history = {}  # 追蹤庫存歷史

    def calculate_quotes(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        inventory_level: float = 0.0,
        start_date: str = None,
        end_date: str = None
    ) -> QuoteResult:
        """
        計算報價序列

        Args:
            market_data: 市場數據DataFrame
            symbol: 交易對符號
            inventory_level: 庫存水平（-1到1之間）
            start_date: 開始日期
            end_date: 結束日期

        Returns:
            報價結果
        """
        logger.info(f"Calculating quotes for {symbol}, inventory_level: {inventory_level}")

        # 準備數據
        symbol_data = market_data[market_data['symbol'] == symbol].copy()

        if symbol_data.empty:
            logger.error(f"No market data found for {symbol}")
            return QuoteResult([], 0, 0, 0, 0, {})

        symbol_data = symbol_data.sort_values('timestamp')

        # 過濾時間範圍
        if start_date:
            symbol_data = symbol_data[symbol_data['timestamp'] >= start_date]
        if end_date:
            symbol_data = symbol_data[symbol_data['timestamp'] <= end_date]

        if symbol_data.empty:
            logger.warning(f"No data in specified time range for {symbol}")
            return QuoteResult([], 0, 0, 0, 0, {})

        # 計算報價
        quotes = []
        spreads = []

        for idx, row in symbol_data.iterrows():
            timestamp = row['timestamp']
            mid_price = row['close']  # 使用收盤價作為中間價

            # 計算動態價差
            spread = self._calculate_dynamic_spread(
                mid_price=mid_price,
                inventory_level=inventory_level,
                row=row
            )

            # 計算買賣價
            bid_price, ask_price = self._calculate_bid_ask(
                mid_price=mid_price,
                spread=spread,
                inventory_level=inventory_level
            )

            # 計算報價數量
            bid_quantity, ask_quantity = self._calculate_quote_quantities(
                mid_price=mid_price,
                spread=spread,
                inventory_level=inventory_level
            )

            # 創建報價對象
            quote = Quote(
                timestamp=timestamp,
                symbol=symbol,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_quantity=bid_quantity,
                ask_quantity=ask_quantity,
                spread=spread,
                mid_price=mid_price,
                inventory_level=inventory_level
            )

            quotes.append(quote)
            spreads.append(spread)

        # 計算統計指標
        if spreads:
            avg_spread = np.mean(spreads)
            max_spread = np.max(spreads)
            min_spread = np.min(spreads)
        else:
            avg_spread = max_spread = min_spread = 0

        # 計算性能指標
        performance_metrics = self._calculate_quote_performance(quotes)

        logger.info(f"Generated {len(quotes)} quotes for {symbol}")
        logger.info(f"Spread stats: avg={avg_spread:.4f}, min={min_spread:.4f}, max={max_spread:.4f}")

        return QuoteResult(
            quotes=quotes,
            total_quotes=len(quotes),
            avg_spread=avg_spread,
            max_spread=max_spread,
            min_spread=min_spread,
            performance_metrics=performance_metrics
        )

    def _calculate_dynamic_spread(
        self,
        mid_price: float,
        inventory_level: float,
        row: pd.Series
    ) -> float:
        """計算動態價差"""
        # 基礎價差
        spread = self.config.base_spread * self.config.spread_multiplier

        # 庫存傾斜調整
        inventory_adjustment = self.config.inventory_skew_factor * abs(inventory_level)
        spread += inventory_adjustment

        # 波動率調整
        if self.config.volatility_adjustment:
            volatility = self._calculate_volatility_adjustment(row)
            spread *= (1 + volatility)

        # 成交量調整
        if self.config.volume_adjustment:
            volume_adjustment = self._calculate_volume_adjustment(row)
            spread *= volume_adjustment

        # 緊急程度調整
        urgency_adjustment = self.config.urgency_factor * abs(inventory_level)
        spread += urgency_adjustment

        # 確保價差在合理範圍內
        spread = max(self.config.min_spread, min(self.config.max_spread, spread))

        return spread

    def _calculate_volatility_adjustment(self, row: pd.Series) -> float:
        """計算波動率調整因子"""
        try:
            # 簡化的波動率估計（基於價格變化）
            price_change = abs(row['close'] - row['open']) / row['open'] if row['open'] > 0 else 0

            # 如果價格變化大，增加價差
            if price_change > 0.02:  # 2%以上的價格變化
                return 0.5
            elif price_change > 0.01:  # 1%以上的價格變化
                return 0.2
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_volume_adjustment(self, row: pd.Series) -> float:
        """計算成交量調整因子"""
        try:
            volume = row.get('volume', 0)

            # 基於成交量調整價差（成交量大時可以縮小價差）
            if volume > 1000000:  # 高成交量
                return 0.8
            elif volume > 100000:  # 中等成交量
                return 0.9
            else:  # 低成交量
                return 1.2

        except Exception:
            return 1.0

    def _calculate_bid_ask(
        self,
        mid_price: float,
        spread: float,
        inventory_level: float
    ) -> Tuple[float, float]:
        """計算買賣價"""
        half_spread = spread / 2

        # 根據庫存水平調整報價傾斜
        if inventory_level > 0:
            # 庫存為正，傾向於賣出，壓低賣價
            bid_price = mid_price - half_spread * (1 - inventory_level * 0.5)
            ask_price = mid_price + half_spread * (1 + inventory_level * 0.5)
        elif inventory_level < 0:
            # 庫存為負，傾向於買入，抬高買價
            bid_price = mid_price - half_spread * (1 + abs(inventory_level) * 0.5)
            ask_price = mid_price + half_spread * (1 - abs(inventory_level) * 0.5)
        else:
            # 庫存平衡
            bid_price = mid_price - half_spread
            ask_price = mid_price + half_spread

        return bid_price, ask_price

    def _calculate_quote_quantities(
        self,
        mid_price: float,
        spread: float,
        inventory_level: float
    ) -> Tuple[float, float]:
        """計算報價數量"""
        # 基礎數量（簡化為固定值）
        base_quantity = 1000 / mid_price  # 價值約1000美元

        # 根據價差調整數量（價差越大，數量越小）
        spread_factor = max(0.1, 1 - spread * 100)
        adjusted_quantity = base_quantity * spread_factor

        # 根據庫存水平調整數量
        if abs(inventory_level) > 0.5:
            # 庫存極端時，減少報價數量
            inventory_factor = 1 - abs(inventory_level) * 0.5
            adjusted_quantity *= inventory_factor

        # 買賣數量可能略有不同
        bid_quantity = adjusted_quantity * (1 + inventory_level * 0.2)
        ask_quantity = adjusted_quantity * (1 - inventory_level * 0.2)

        return max(0, bid_quantity), max(0, ask_quantity)

    def _calculate_quote_performance(self, quotes: List[Quote]) -> Dict[str, float]:
        """計算報價性能指標"""
        if not quotes:
            return {}

        spreads = [quote.spread for quote in quotes]
        bid_prices = [quote.bid_price for quote in quotes]
        ask_prices = [quote.ask_price for quote in quotes]

        # 價差統計
        avg_spread = np.mean(spreads)
        spread_volatility = np.std(spreads)

        # 報價範圍
        price_range = max(bid_prices) - min(bid_prices)

        # 估計交易成本（簡化）
        estimated_cost = avg_spread * 0.5  # 假設成交在價差中間

        # 估計潛在收益
        potential_profit = avg_spread * min(1000, sum(q.bid_quantity + q.ask_quantity for q in quotes) / len(quotes))

        return {
            'avg_spread': avg_spread,
            'spread_volatility': spread_volatility,
            'price_range': price_range,
            'estimated_cost': estimated_cost,
            'potential_profit': potential_profit,
            'quote_count': len(quotes)
        }

    def optimize_quote_parameters(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        parameter_ranges: Dict[str, List[float]] = None
    ) -> Dict[str, Any]:
        """
        優化報價參數

        Args:
            market_data: 市場數據
            symbol: 交易對符號
            parameter_ranges: 參數範圍

        Returns:
            優化結果
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'base_spread': [0.0005, 0.001, 0.002, 0.005],
                'inventory_skew_factor': [0.05, 0.1, 0.2, 0.3],
                'urgency_factor': [0.2, 0.5, 0.8, 1.0]
            }

        logger.info(f"Optimizing quote parameters for {symbol}")

        best_performance = None
        best_params = None
        all_results = []

        # 網格搜尋
        from itertools import product

        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        for param_combination in product(*param_values):
            # 創建參數組合
            params = dict(zip(param_names, param_combination))

            # 創建臨時配置
            temp_config = QuoteConfig(
                base_spread=params['base_spread'],
                inventory_skew_factor=params.get('inventory_skew_factor', self.config.inventory_skew_factor),
                urgency_factor=params.get('urgency_factor', self.config.urgency_factor)
            )

            # 生成報價
            temp_engine = QuoteEngine(temp_config)
            result = temp_engine.calculate_quotes(market_data, symbol)

            if result.quotes:
                # 計算性能分數（綜合考慮價差和穩定性）
                performance_score = self._calculate_quote_score(result)

                result_entry = {
                    'parameters': params,
                    'avg_spread': result.avg_spread,
                    'spread_volatility': result.performance_metrics.get('spread_volatility', 0),
                    'potential_profit': result.performance_metrics.get('potential_profit', 0),
                    'score': performance_score
                }

                all_results.append(result_entry)

                # 更新最佳參數
                if best_performance is None or performance_score > best_performance:
                    best_performance = performance_score
                    best_params = params.copy()

        # 排序結果
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)

        logger.info(f"Quote parameter optimization completed. Best score: {best_performance}")

        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'all_results': sorted_results[:10],  # 返回前10個結果
            'total_combinations': len(list(product(*param_values)))
        }

    def _calculate_quote_score(self, result: QuoteResult) -> float:
        """計算報價綜合分數"""
        # 分數組成：
        # 1. 價差適中（不太大也不太小）
        # 2. 價差穩定性（波動小）
        # 3. 潛在收益

        avg_spread = result.avg_spread
        spread_volatility = result.performance_metrics.get('spread_volatility', 0)
        potential_profit = result.performance_metrics.get('potential_profit', 0)

        # 標準化價差分數（最優價差約0.001-0.002）
        if avg_spread <= 0.001:
            spread_score = avg_spread / 0.001  # 太小了
        elif avg_spread <= 0.002:
            spread_score = 1.0  # 最優範圍
        else:
            spread_score = max(0.5, 1 - (avg_spread - 0.002) / 0.003)  # 太大扣分

        # 穩定性分數（波動越小越好）
        stability_score = max(0, 1 - spread_volatility / avg_spread) if avg_spread > 0 else 0

        # 收益分數（標準化）
        profit_score = min(1.0, potential_profit / 1000)  # 假設1000為滿分

        # 加權綜合分數
        total_score = (
            spread_score * 0.4 +
            stability_score * 0.3 +
            profit_score * 0.3
        )

        return total_score

    def simulate_amm_quotes(
        self,
        price_data: pd.DataFrame,
        symbol: str,
        liquidity_usd: float = 100000,
        fee_rate: float = 0.003,
        price_range_factor: float = 1.5
    ) -> QuoteResult:
        """
        模擬AMM報價（類似Uniswap V3）

        Args:
            price_data: 價格數據
            symbol: 交易對符號
            liquidity_usd: 流動性價值（美元）
            fee_rate: 手續費率
            price_range_factor: 價格範圍因子

        Returns:
            AMM報價結果
        """
        logger.info(f"Simulating AMM quotes for {symbol}")

        symbol_data = price_data[price_data['symbol'] == symbol].copy()

        if symbol_data.empty:
            return QuoteResult([], 0, 0, 0, 0, {})

        quotes = []

        for idx, row in symbol_data.iterrows():
            timestamp = row['timestamp']
            current_price = row['close']

            # 計算價格範圍
            price_range = current_price * price_range_factor
            price_min = current_price / price_range_factor
            price_max = current_price * price_range_factor

            # 計算AMM報價（簡化版本）
            # 在AMM中，報價基於恆定乘積公式 x*y = k
            sqrt_price = np.sqrt(current_price)
            liquidity_x = liquidity_usd / (2 * sqrt_price)  # 簡化計算

            # 計算買賣價（基於當前價格和流動性）
            price_impact = 0.001  # 假設1%的價格影響
            bid_price = current_price * (1 - price_impact)
            ask_price = current_price * (1 + price_impact)

            # 計算報價數量
            bid_quantity = liquidity_x * sqrt_price
            ask_quantity = liquidity_x / sqrt_price

            # 調整價差（考慮手續費）
            effective_spread = (ask_price - bid_price) / current_price + fee_rate

            quote = Quote(
                timestamp=timestamp,
                symbol=symbol,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_quantity=bid_quantity,
                ask_quantity=ask_quantity,
                spread=effective_spread,
                mid_price=current_price,
                inventory_level=0  # AMM沒有庫存概念
            )

            quotes.append(quote)

        # 計算統計
        if quotes:
            spreads = [q.spread for q in quotes]
            avg_spread = np.mean(spreads)
            max_spread = np.max(spreads)
            min_spread = np.min(spreads)
        else:
            avg_spread = max_spread = min_spread = 0

        performance_metrics = self._calculate_quote_performance(quotes)

        logger.info(f"Generated {len(quotes)} AMM quotes for {symbol}")

        return QuoteResult(
            quotes=quotes,
            total_quotes=len(quotes),
            avg_spread=avg_spread,
            max_spread=max_spread,
            min_spread=min_spread,
            performance_metrics=performance_metrics
        )

    def calculate_impermanent_loss(
        self,
        quotes: List[Quote],
        initial_price: float,
        current_price: float
    ) -> float:
        """
        計算無常損失

        Args:
            quotes: 報價序列
            initial_price: 初始價格
            current_price: 當前價格

        Returns:
            無常損失百分比
        """
        if not quotes:
            return 0.0

        try:
            # 簡化的無常損失計算
            # 無常損失 = 2*sqrt(r) / (1+r) - 1，其中r為價格比率
            price_ratio = current_price / initial_price

            if price_ratio <= 0:
                return 0.0

            impermanent_loss = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1

            return abs(impermanent_loss)  # 返回絕對值

        except Exception as e:
            logger.error(f"Error calculating impermanent loss: {e}")
            return 0.0

    def generate_quote_report(self, result: QuoteResult, symbol: str = None) -> str:
        """生成報價報告"""
        report = []
        report.append("=" * 60)
        report.append("報價計算報告")
        report.append("=" * 60)

        if symbol:
            report.append(f"交易對: {symbol}")

        report.append(f"總報價數: {result.total_quotes}")
        report.append(f"平均價差: {result.avg_spread".4f"} ({result.avg_spread*100".2f"}%)")
        report.append(f"價差範圍: {result.min_spread".4f"} - {result.max_spread".4f"}")
        report.append("")

        if result.performance_metrics:
            report.append("性能指標:")
            for metric, value in result.performance_metrics.items():
                if isinstance(value, float):
                    if 'profit' in metric.lower():
                        report.append(f"  {metric}: ${value".2f"}")
                    else:
                        report.append(f"  {metric}: {value".4f"}")
                else:
                    report.append(f"  {metric}: {value}")
            report.append("")

        # 顯示最近的報價
        if result.quotes:
            report.append("最近報價:")
            recent_quotes = result.quotes[-5:]  # 顯示最近5個
            for quote in recent_quotes:
                report.append(f"  {quote.timestamp.strftime('%Y-%m-%d %H:%M')}: "
                            f"Bid: ${quote.bid_price".2f"}, Ask: ${quote.ask_price".2f"}, "
                            f"Spread: {quote.spread*100".2f"}%")

        report.append("=" * 60)

        return "\n".join(report)


# 使用範例
if __name__ == "__main__":
    # 創建報價引擎
    config = QuoteConfig(
        base_spread=0.001,
        inventory_skew_factor=0.1,
        volatility_adjustment=True
    )

    engine = QuoteEngine(config)

    # 創建範例市場數據
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')

    # 創建價格序列（帶有趨勢和波動）
    base_price = 50000
    trend = np.linspace(0, 0.1, 100)  # 10%上漲趨勢
    noise = np.random.randn(100) * 0.02  # 2%隨機波動

    prices = base_price * (1 + trend + noise)

    sample_data = []
    for i, date in enumerate(dates):
        sample_data.append({
            'timestamp': date,
            'symbol': 'BTC',
            'open': prices[i] * 0.999,
            'high': prices[i] * 1.002,
            'low': prices[i] * 0.998,
            'close': prices[i],
            'volume': np.random.randint(100000, 1000000)
        })

    df = pd.DataFrame(sample_data)

    # 計算報價
    result = engine.calculate_quotes(df, 'BTC', inventory_level=0.2)

    print(f"生成的報價數量: {result.total_quotes}")
    print(f"平均價差: {result.avg_spread".4f"}")

    # 生成報告
    report = engine.generate_quote_report(result, 'BTC')
    print("\n報價報告:")
    print(report)

    # 模擬AMM報價
    amm_result = engine.simulate_amm_quotes(df, 'BTC', liquidity_usd=100000)
    print(f"\nAMM報價數量: {amm_result.total_quotes}")
    print(f"AMM平均價差: {amm_result.avg_spread".4f"}")

    # 計算無常損失
    initial_price = prices[0]
    final_price = prices[-1]
    il = engine.calculate_impermanent_loss(result.quotes, initial_price, final_price)
    print(f"無常損失: {il".2%"}")
