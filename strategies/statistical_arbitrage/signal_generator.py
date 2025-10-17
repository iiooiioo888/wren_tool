"""
信號生成系統
負責統計套利策略中的信號生成，包含Z-score計算和進出場條件判斷
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
class SignalConfig:
    """信號生成配置"""
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    stop_loss_zscore: float = 3.0
    lookback_window: int = 20
    volatility_window: int = 20
    min_holding_period: int = 1
    max_position_size: float = 1.0

@dataclass
class TradingSignal:
    """交易信號"""
    timestamp: pd.Timestamp
    symbol1: str
    symbol2: str
    signal_type: str  # 'long', 'short', 'close', 'stop_loss'
    zscore: float
    price1: float
    price2: float
    hedge_ratio: float
    spread: float
    position_size: float
    confidence: float

@dataclass
class SignalResult:
    """信號生成結果"""
    signals: List[TradingSignal]
    total_signals: int
    long_signals: int
    short_signals: int
    close_signals: int
    stop_loss_signals: int
    performance_metrics: Dict[str, float]

class SignalGenerator:
    """信號生成器"""

    def __init__(self, config: SignalConfig = None):
        """
        初始化信號生成器

        Args:
            config: 信號生成配置
        """
        self.config = config or SignalConfig()
        self.position_tracker = {}  # 追蹤當前持倉

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        pair_info: Dict[str, Any],
        start_date: str = None,
        end_date: str = None
    ) -> SignalResult:
        """
        生成交易信號

        Args:
            price_data: 價格數據DataFrame
            pair_info: 配對信息（包含hedge_ratio等）
            start_date: 開始日期
            end_date: 結束日期

        Returns:
            信號生成結果
        """
        logger.info(f"Generating signals for pair: {pair_info.get('symbol1', 'N/A')} - {pair_info.get('symbol2', 'N/A')}")

        # 準備數據
        symbol1 = pair_info['symbol1']
        symbol2 = pair_info['symbol2']
        hedge_ratio = pair_info['hedge_ratio']

        # 獲取價格序列
        prices1 = self._get_price_series(price_data, symbol1)
        prices2 = self._get_price_series(price_data, symbol2)

        if len(prices1) != len(prices2):
            logger.error("Price series length mismatch")
            return SignalResult([], 0, 0, 0, 0, 0, {})

        # 計算價差序列
        spread = prices2 - hedge_ratio * prices1

        # 計算Z-score
        zscores = self._calculate_zscore(spread)

        # 生成信號序列
        signals = []
        position = 0  # 當前持倉（1: 多頭, -1: 空頭, 0: 無倉位）
        entry_time = None

        for i in range(self.config.lookback_window, len(zscores)):
            current_zscore = zscores.iloc[i]
            current_time = prices1.index[i]
            current_price1 = prices1.iloc[i]
            current_price2 = prices2.iloc[i]
            current_spread = spread.iloc[i]

            # 檢查時間範圍
            if start_date and current_time < pd.to_datetime(start_date):
                continue
            if end_date and current_time > pd.to_datetime(end_date):
                break

            # 檢查最小持有期
            if (entry_time and
                (current_time - entry_time).days < self.config.min_holding_period and
                position != 0):
                continue

            # 生成信號
            signal = self._generate_single_signal(
                current_zscore, position, current_time, symbol1, symbol2,
                current_price1, current_price2, hedge_ratio, current_spread
            )

            if signal:
                signals.append(signal)

                # 更新持倉狀態
                if signal.signal_type in ['long', 'short']:
                    position = 1 if signal.signal_type == 'long' else -1
                    entry_time = current_time
                elif signal.signal_type in ['close', 'stop_loss']:
                    position = 0
                    entry_time = None

        # 計算統計信息
        signal_stats = self._calculate_signal_statistics(signals)

        logger.info(f"Generated {len(signals)} signals: {signal_stats}")

        return SignalResult(
            signals=signals,
            total_signals=len(signals),
            long_signals=signal_stats['long_count'],
            short_signals=signal_stats['short_count'],
            close_signals=signal_stats['close_count'],
            stop_loss_signals=signal_stats['stop_loss_count'],
            performance_metrics=self._calculate_performance_metrics(signals)
        )

    def _get_price_series(self, price_data: pd.DataFrame, symbol: str) -> pd.Series:
        """獲取價格序列並設定時間索引"""
        symbol_data = price_data[price_data['symbol'] == symbol].copy()

        if symbol_data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        symbol_data = symbol_data.sort_values('timestamp')
        symbol_data = symbol_data.set_index('timestamp')['close']

        return symbol_data

    def _calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """計算Z-score"""
        # 使用滾動窗口計算均值和標準差
        rolling_mean = spread.rolling(window=self.config.lookback_window).mean()
        rolling_std = spread.rolling(window=self.config.lookback_window).std()

        # 避免除零錯誤
        rolling_std = rolling_std.replace(0, np.nan)

        # 計算Z-score
        zscore = (spread - rolling_mean) / rolling_std

        return zscore

    def _generate_single_signal(
        self,
        zscore: float,
        current_position: int,
        timestamp: pd.Timestamp,
        symbol1: str,
        symbol2: str,
        price1: float,
        price2: float,
        hedge_ratio: float,
        spread: float
    ) -> Optional[TradingSignal]:
        """生成單個信號"""
        # 止損檢查（優先級最高）
        if abs(zscore) >= self.config.stop_loss_zscore:
            if current_position == 1:  # 多頭止損
                return TradingSignal(
                    timestamp=timestamp,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    signal_type='stop_loss',
                    zscore=zscore,
                    price1=price1,
                    price2=price2,
                    hedge_ratio=hedge_ratio,
                    spread=spread,
                    position_size=abs(current_position),
                    confidence=self._calculate_confidence(abs(zscore), 'stop_loss')
                )
            elif current_position == -1:  # 空頭止損
                return TradingSignal(
                    timestamp=timestamp,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    signal_type='stop_loss',
                    zscore=zscore,
                    price1=price1,
                    price2=price2,
                    hedge_ratio=hedge_ratio,
                    spread=spread,
                    position_size=abs(current_position),
                    confidence=self._calculate_confidence(abs(zscore), 'stop_loss')
                )

        # 平倉信號
        if current_position == 1 and zscore <= self.config.exit_zscore:
            return TradingSignal(
                timestamp=timestamp,
                symbol1=symbol1,
                symbol2=symbol2,
                signal_type='close',
                zscore=zscore,
                price1=price1,
                price2=price2,
                hedge_ratio=hedge_ratio,
                spread=spread,
                position_size=1,
                confidence=self._calculate_confidence(abs(zscore), 'close')
            )
        elif current_position == -1 and zscore >= -self.config.exit_zscore:
            return TradingSignal(
                timestamp=timestamp,
                symbol1=symbol1,
                symbol2=symbol2,
                signal_type='close',
                zscore=zscore,
                price1=price1,
                price2=price2,
                hedge_ratio=hedge_ratio,
                spread=spread,
                position_size=1,
                confidence=self._calculate_confidence(abs(zscore), 'close')
            )

        # 開倉信號（無持倉時）
        if current_position == 0:
            if zscore >= self.config.entry_zscore:  # 開空頭
                return TradingSignal(
                    timestamp=timestamp,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    signal_type='short',
                    zscore=zscore,
                    price1=price1,
                    price2=price2,
                    hedge_ratio=hedge_ratio,
                    spread=spread,
                    position_size=self.config.max_position_size,
                    confidence=self._calculate_confidence(zscore, 'entry')
                )
            elif zscore <= -self.config.entry_zscore:  # 開多頭
                return TradingSignal(
                    timestamp=timestamp,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    signal_type='long',
                    zscore=zscore,
                    price1=price1,
                    price2=price2,
                    hedge_ratio=hedge_ratio,
                    spread=spread,
                    position_size=self.config.max_position_size,
                    confidence=self._calculate_confidence(abs(zscore), 'entry')
                )

        return None

    def _calculate_confidence(self, zscore: float, signal_type: str) -> float:
        """計算信號信心度"""
        if signal_type == 'entry':
            # 進場信號：基於Z-score絕對值，越極端信心越高
            return min(abs(zscore) / self.config.entry_zscore, 1.0)
        elif signal_type == 'close':
            # 平倉信號：基於Z-score接近0的程度
            return max(1 - abs(zscore) / self.config.exit_zscore, 0.0)
        elif signal_type == 'stop_loss':
            # 止損信號：信心較低，因為通常表示策略失效
            return 0.3
        else:
            return 0.5

    def _calculate_signal_statistics(self, signals: List[TradingSignal]) -> Dict[str, int]:
        """計算信號統計"""
        stats = {
            'long_count': 0,
            'short_count': 0,
            'close_count': 0,
            'stop_loss_count': 0
        }

        for signal in signals:
            if signal.signal_type == 'long':
                stats['long_count'] += 1
            elif signal.signal_type == 'short':
                stats['short_count'] += 1
            elif signal.signal_type == 'close':
                stats['close_count'] += 1
            elif signal.signal_type == 'stop_loss':
                stats['stop_loss_count'] += 1

        return stats

    def _calculate_performance_metrics(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """計算信號性能指標"""
        if not signals:
            return {}

        # 計算信號頻率
        if len(signals) > 1:
            time_diffs = []
            for i in range(1, len(signals)):
                diff = (signals[i].timestamp - signals[i-1].timestamp).total_seconds() / (24 * 3600)  # 天數
                time_diffs.append(diff)

            avg_signal_interval = np.mean(time_diffs) if time_diffs else 0
            signal_frequency = 1 / avg_signal_interval if avg_signal_interval > 0 else 0
        else:
            avg_signal_interval = 0
            signal_frequency = 0

        # 計算信號強度統計
        zscores = [abs(signal.zscore) for signal in signals]
        avg_signal_strength = np.mean(zscores) if zscores else 0
        max_signal_strength = np.max(zscores) if zscores else 0

        # 計算信心度統計
        confidences = [signal.confidence for signal in signals]
        avg_confidence = np.mean(confidences) if confidences else 0

        return {
            'avg_signal_interval_days': avg_signal_interval,
            'signal_frequency_per_day': signal_frequency,
            'avg_signal_strength': avg_signal_strength,
            'max_signal_strength': max_signal_strength,
            'avg_confidence': avg_confidence,
            'total_signals': len(signals)
        }

    def backtest_signals(
        self,
        signals: List[TradingSignal],
        price_data: pd.DataFrame,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001
    ) -> Dict[str, Any]:
        """
        回測信號表現

        Args:
            signals: 交易信號列表
            price_data: 價格數據
            initial_capital: 初始資金
            transaction_cost: 交易成本

        Returns:
            回測結果
        """
        logger.info(f"Backtesting {len(signals)} signals")

        # 準備價格數據查詢
        price_dict = {}
        for _, row in price_data.iterrows():
            symbol = row['symbol']
            timestamp = row['timestamp']
            if symbol not in price_dict:
                price_dict[symbol] = {}
            price_dict[symbol][timestamp] = row['close']

        # 模擬交易
        capital = initial_capital
        position1 = 0  # 第一個資產持倉
        position2 = 0  # 第二個資產持倉
        trades = []
        portfolio_values = []

        current_prices = {'price1': 0, 'price2': 0}

        for signal in signals:
            # 獲取當前價格
            try:
                current_prices['price1'] = price_dict[signal.symbol1][signal.timestamp]
                current_prices['price2'] = price_dict[signal.symbol2][signal.timestamp]
            except KeyError:
                logger.warning(f"Price data missing for {signal.timestamp}")
                continue

            # 計算交易價值
            trade_value = capital * signal.position_size

            if signal.signal_type == 'long':
                # 買入第二個資產，賣出第一個資產
                qty2 = trade_value / current_prices['price2']
                qty1 = qty2 * signal.hedge_ratio

                # 扣除交易成本
                cost = (qty1 * current_prices['price1'] + qty2 * current_prices['price2']) * transaction_cost

                position2 += qty2
                position1 -= qty1  # 賣空第一個資產
                capital -= cost

                trades.append({
                    'timestamp': signal.timestamp,
                    'type': 'long_entry',
                    'price1': current_prices['price1'],
                    'price2': current_prices['price2'],
                    'qty1': qty1,
                    'qty2': qty2,
                    'cost': cost
                })

            elif signal.signal_type == 'short':
                # 賣出第二個資產，買入第一個資產
                qty1 = trade_value / current_prices['price1']
                qty2 = qty1 / signal.hedge_ratio

                # 扣除交易成本
                cost = (qty1 * current_prices['price1'] + qty2 * current_prices['price2']) * transaction_cost

                position1 += qty1
                position2 -= qty2  # 賣空第二個資產
                capital -= cost

                trades.append({
                    'timestamp': signal.timestamp,
                    'type': 'short_entry',
                    'price1': current_prices['price1'],
                    'price2': current_prices['price2'],
                    'qty1': qty1,
                    'qty2': qty2,
                    'cost': cost
                })

            elif signal.signal_type in ['close', 'stop_loss']:
                # 平倉
                if position1 != 0 or position2 != 0:
                    # 計算平倉價值
                    close_value1 = position1 * current_prices['price1']
                    close_value2 = position2 * current_prices['price2']

                    # 扣除交易成本
                    cost = (abs(position1) * current_prices['price1'] + abs(position2) * current_prices['price2']) * transaction_cost

                    capital += close_value1 + close_value2 - cost

                    trades.append({
                        'timestamp': signal.timestamp,
                        'type': 'close',
                        'price1': current_prices['price1'],
                        'price2': current_prices['price2'],
                        'qty1': position1,
                        'qty2': position2,
                        'value1': close_value1,
                        'value2': close_value2,
                        'cost': cost
                    })

                    position1 = 0
                    position2 = 0

            # 計算當前投資組合價值
            portfolio_value = capital + position1 * current_prices['price1'] + position2 * current_prices['price2']
            portfolio_values.append({
                'timestamp': signal.timestamp,
                'portfolio_value': portfolio_value,
                'capital': capital,
                'position_value': position1 * current_prices['price1'] + position2 * current_prices['price2']
            })

        # 計算最終價值
        final_value = portfolio_values[-1]['portfolio_value'] if portfolio_values else initial_capital

        # 計算回報率
        total_return = (final_value - initial_capital) / initial_capital

        # 計算年化回報率
        if signals:
            days = (signals[-1].timestamp - signals[0].timestamp).days
            annual_return = total_return * (365 / max(days, 1))
        else:
            annual_return = 0

        # 計算夏普比率（簡化版本）
        if portfolio_values:
            returns = []
            for i in range(1, len(portfolio_values)):
                daily_return = (portfolio_values[i]['portfolio_value'] - portfolio_values[i-1]['portfolio_value']) / portfolio_values[i-1]['portfolio_value']
                returns.append(daily_return)

            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (avg_return / std_return) * np.sqrt(365) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'portfolio_values': portfolio_values,
            'trades': trades
        }

    def optimize_signal_parameters(
        self,
        price_data: pd.DataFrame,
        pair_info: Dict[str, Any],
        parameter_ranges: Dict[str, List[float]] = None
    ) -> Dict[str, Any]:
        """
        優化信號參數

        Args:
            price_data: 價格數據
            pair_info: 配對信息
            parameter_ranges: 參數範圍

        Returns:
            優化結果
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'entry_zscore': [1.5, 2.0, 2.5, 3.0],
                'exit_zscore': [0.2, 0.5, 0.8, 1.0],
                'lookback_window': [15, 20, 25, 30]
            }

        logger.info("Optimizing signal parameters")

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
            temp_config = SignalConfig(
                entry_zscore=params['entry_zscore'],
                exit_zscore=params.get('exit_zscore', self.config.exit_zscore),
                lookback_window=params.get('lookback_window', self.config.lookback_window)
            )

            # 生成信號
            temp_generator = SignalGenerator(temp_config)
            result = temp_generator.generate_signals(price_data, pair_info)

            if result.signals:
                # 回測性能
                backtest_result = temp_generator.backtest_signals(
                    result.signals, price_data
                )

                performance_score = backtest_result['sharpe_ratio']

                result_entry = {
                    'parameters': params,
                    'signal_count': result.total_signals,
                    'sharpe_ratio': backtest_result['sharpe_ratio'],
                    'total_return': backtest_result['total_return'],
                    'max_drawdown': self._calculate_max_drawdown(backtest_result['portfolio_values'])
                }

                all_results.append(result_entry)

                # 更新最佳參數
                if best_performance is None or performance_score > best_performance:
                    best_performance = performance_score
                    best_params = params.copy()

        # 排序結果
        sorted_results = sorted(all_results, key=lambda x: x['sharpe_ratio'], reverse=True)

        logger.info(f"Parameter optimization completed. Best Sharpe: {best_performance}")

        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'all_results': sorted_results[:10],  # 返回前10個結果
            'total_combinations': len(list(product(*param_values)))
        }

    def _calculate_max_drawdown(self, portfolio_values: List[Dict]) -> float:
        """計算最大回撤"""
        if not portfolio_values:
            return 0.0

        values = [pv['portfolio_value'] for pv in portfolio_values]
        peak = values[0]
        max_drawdown = 0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def generate_signal_report(self, result: SignalResult) -> str:
        """生成信號報告"""
        report = []
        report.append("=" * 60)
        report.append("信號生成報告")
        report.append("=" * 60)
        report.append(f"總信號數: {result.total_signals}")
        report.append(f"多頭信號: {result.long_signals}")
        report.append(f"空頭信號: {result.short_signals}")
        report.append(f"平倉信號: {result.close_signals}")
        report.append(f"止損信號: {result.stop_loss_signals}")
        report.append("")

        if result.performance_metrics:
            report.append("性能指標:")
            for metric, value in result.performance_metrics.items():
                if isinstance(value, float):
                    report.append(f"  {metric}: {value".4f"}")
                else:
                    report.append(f"  {metric}: {value}")
            report.append("")

        # 顯示最近的信號
        if result.signals:
            report.append("最近信號:")
            recent_signals = result.signals[-5:]  # 顯示最近5個
            for signal in recent_signals:
                report.append(f"  {signal.timestamp.strftime('%Y-%m-%d %H:%M')}: "
                            f"{signal.signal_type.upper()} (Z-score: {signal.zscore".2f"})")

        report.append("=" * 60)

        return "\n".join(report)


# 使用範例
if __name__ == "__main__":
    # 創建信號生成器
    config = SignalConfig(
        entry_zscore=2.0,
        exit_zscore=0.5,
        lookback_window=20
    )

    generator = SignalGenerator(config)

    # 創建範例數據
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    # 創建相關的價格序列
    price1 = 100 + np.cumsum(np.random.randn(100) * 0.01)
    price2 = 50 + 0.5 * price1 + np.cumsum(np.random.randn(100) * 0.005)

    sample_data = []
    for i, date in enumerate(dates):
        sample_data.extend([
            {'timestamp': date, 'symbol': 'BTC', 'close': price1[i]},
            {'timestamp': date, 'symbol': 'ETH', 'close': price2[i]}
        ])

    df = pd.DataFrame(sample_data)

    # 配對信息
    pair_info = {
        'symbol1': 'BTC',
        'symbol2': 'ETH',
        'hedge_ratio': 0.5
    }

    # 生成信號
    result = generator.generate_signals(df, pair_info)

    print(f"生成的信號數量: {result.total_signals}")
    print(f"多頭信號: {result.long_signals}")
    print(f"空頭信號: {result.short_signals}")

    # 生成報告
    report = generator.generate_signal_report(result)
    print("\n信號報告:")
    print(report)

    # 回測信號
    if result.signals:
        backtest_result = generator.backtest_signals(result.signals, df)
        print("
回測結果:")
        print(f"總回報: {backtest_result['total_return']".2%"}")
        print(f"年化回報: {backtest_result['annual_return']".2%"}")
        print(f"夏普比率: {backtest_result['sharpe_ratio']".2f"}")
