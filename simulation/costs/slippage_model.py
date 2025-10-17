"""
滑點模擬器
負責模擬大額訂單對市場價格的影響，基於歷史數據和訂單規模
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
class SlippageModelConfig:
    """滑點模型配置"""
    method: str = "square_root"  # 滑點計算方法
    market_impact_factor: float = 0.001  # 市場影響因子
    volume_participation_rate: float = 0.1  # 成交量參與率
    price_volatility_factor: float = 1.0  # 價格波動率因子
    order_book_depth_factor: float = 1.0  # 訂單簿深度因子

@dataclass
class SlippageResult:
    """滑點計算結果"""
    original_price: float
    executed_price: float
    slippage_amount: float
    slippage_percentage: float
    market_impact: float
    estimated_fill_time: float
    confidence: float

class SlippageModel:
    """滑點模擬器"""

    def __init__(self, config: SlippageModelConfig = None):
        """
        初始化滑點模擬器

        Args:
            config: 滑點模型配置
        """
        self.config = config or SlippageModelConfig()

        # 歷史滑點數據（用於模型訓練）
        self.historical_slippage = {}

    def calculate_slippage(
        self,
        symbol: str,
        order_size: float,
        order_price: float,
        order_type: str = "market",  # market, limit, iceberg
        market_data: pd.DataFrame = None,
        order_book_data: Dict[str, Any] = None
    ) -> SlippageResult:
        """
        計算訂單滑點

        Args:
            symbol: 交易對符號
            order_size: 訂單規模
            order_price: 訂單價格（限價單）或市場價格（市價單）
            order_type: 訂單類型
            market_data: 市場數據
            order_book_data: 訂單簿數據

        Returns:
            滑點計算結果
        """
        logger.info(f"Calculating slippage for {symbol} order: {order_size} @ {order_price}")

        # 獲取市場統計數據
        market_stats = self._get_market_statistics(symbol, market_data)

        if order_type == "limit":
            # 限價單滑點計算
            slippage_result = self._calculate_limit_order_slippage(
                order_size, order_price, market_stats, order_book_data
            )
        elif order_type == "iceberg":
            # 冰山單滑點計算
            slippage_result = self._calculate_iceberg_order_slippage(
                order_size, order_price, market_stats, order_book_data
            )
        else:
            # 市價單滑點計算（默認）
            slippage_result = self._calculate_market_order_slippage(
                order_size, order_price, market_stats, order_book_data
            )

        logger.info(f"Slippage calculated: {slippage_result.slippage_percentage:.4f} ({slippage_result.slippage_percentage*100:.2f}%)")
        return slippage_result

    def _get_market_statistics(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, float]:
        """獲取市場統計數據"""
        if market_data is None or symbol not in market_data['symbol'].values:
            # 返回默認統計數據
            return {
                'avg_volume_24h': 1000000,
                'price_volatility': 0.02,
                'order_book_depth': 100000,
                'spread': 0.001
            }

        symbol_data = market_data[market_data['symbol'] == symbol]

        if symbol_data.empty:
            return {
                'avg_volume_24h': 1000000,
                'price_volatility': 0.02,
                'order_book_depth': 100000,
                'spread': 0.001
            }

        # 計算統計指標
        volumes = symbol_data['volume'].dropna()
        prices = symbol_data['close'].dropna()

        avg_volume = volumes.mean() if len(volumes) > 0 else 1000000

        # 計算價格波動率（簡化）
        price_volatility = prices.pct_change().std() if len(prices) > 1 else 0.02

        # 估計訂單簿深度（簡化）
        order_book_depth = avg_volume * 0.1  # 假設深度為日成交量的10%

        # 估計價差
        spread = price_volatility * 0.1  # 簡化估計

        return {
            'avg_volume_24h': avg_volume,
            'price_volatility': price_volatility,
            'order_book_depth': order_book_depth,
            'spread': spread
        }

    def _calculate_market_order_slippage(
        self,
        order_size: float,
        order_price: float,
        market_stats: Dict[str, float],
        order_book_data: Dict[str, Any]
    ) -> SlippageResult:
        """計算市價單滑點"""
        # 使用平方根模型計算市場影響
        if self.config.method == "square_root":
            # 平方根模型：市場影響 = k * sqrt(訂單規模 / 平均成交量)
            participation_rate = order_size / market_stats['avg_volume_24h']
            market_impact = self.config.market_impact_factor * np.sqrt(participation_rate)

        elif self.config.method == "linear":
            # 線性模型：市場影響 = k * (訂單規模 / 訂單簿深度)
            participation_rate = order_size / market_stats['order_book_depth']
            market_impact = self.config.market_impact_factor * participation_rate

        else:
            # 默認平方根模型
            participation_rate = order_size / market_stats['avg_volume_24h']
            market_impact = self.config.market_impact_factor * np.sqrt(participation_rate)

        # 考慮價格波動率調整
        volatility_adjustment = 1 + market_stats['price_volatility'] * self.config.price_volatility_factor
        market_impact *= volatility_adjustment

        # 計算執行價格（假設買單推高價格，賣單壓低價格）
        # 這裡假設是買單，實際應該根據訂單方向決定
        executed_price = order_price * (1 + market_impact)

        # 計算滑點
        slippage_amount = executed_price - order_price
        slippage_percentage = slippage_amount / order_price

        # 估計成交時間（簡化）
        estimated_fill_time = max(1, order_size / market_stats['avg_volume_24h'] * 86400)  # 秒

        # 計算信心度
        confidence = self._calculate_slippage_confidence(
            order_size, market_stats, market_impact
        )

        return SlippageResult(
            original_price=order_price,
            executed_price=executed_price,
            slippage_amount=slippage_amount,
            slippage_percentage=slippage_percentage,
            market_impact=market_impact,
            estimated_fill_time=estimated_fill_time,
            confidence=confidence
        )

    def _calculate_limit_order_slippage(
        self,
        order_size: float,
        order_price: float,
        market_stats: Dict[str, float],
        order_book_data: Dict[str, Any]
    ) -> SlippageResult:
        """計算限價單滑點"""
        # 限價單的滑點主要來自於無法完全成交的部分
        # 這裡簡化處理為市場影響的一半

        participation_rate = order_size / market_stats['avg_volume_24h']
        market_impact = self.config.market_impact_factor * np.sqrt(participation_rate) * 0.5

        # 對於限價單，滑點較小但成交概率較低
        executed_price = order_price * (1 + market_impact * 0.3)

        slippage_amount = executed_price - order_price
        slippage_percentage = slippage_amount / order_price

        # 限價單成交時間通常較長
        estimated_fill_time = order_size / market_stats['avg_volume_24h'] * 86400 * 2

        confidence = self._calculate_slippage_confidence(
            order_size, market_stats, market_impact
        ) * 0.8  # 限價單信心度較低

        return SlippageResult(
            original_price=order_price,
            executed_price=executed_price,
            slippage_amount=slippage_amount,
            slippage_percentage=slippage_percentage,
            market_impact=market_impact,
            estimated_fill_time=estimated_fill_time,
            confidence=confidence
        )

    def _calculate_iceberg_order_slippage(
        self,
        order_size: float,
        order_price: float,
        market_stats: Dict[str, float],
        order_book_data: Dict[str, Any]
    ) -> SlippageResult:
        """計算冰山單滑點"""
        # 冰山單將大訂單分成小塊執行，滑點較小但執行時間較長

        # 假設分成10塊執行
        num_slices = min(10, max(2, int(order_size / 1000)))
        slice_size = order_size / num_slices

        # 每塊的市場影響較小
        slice_participation = slice_size / market_stats['avg_volume_24h']
        slice_market_impact = self.config.market_impact_factor * np.sqrt(slice_participation) * 0.3

        # 總市場影響（考慮重複影響）
        total_market_impact = slice_market_impact * (1 + (num_slices - 1) * 0.2)  # 後續訂單影響遞減

        executed_price = order_price * (1 + total_market_impact)

        slippage_amount = executed_price - order_price
        slippage_percentage = slippage_amount / order_price

        # 冰山單執行時間較長
        estimated_fill_time = order_size / market_stats['avg_volume_24h'] * 86400 * num_slices

        confidence = self._calculate_slippage_confidence(
            order_size, market_stats, total_market_impact
        ) * 0.9  # 冰山單信心度較高

        return SlippageResult(
            original_price=order_price,
            executed_price=executed_price,
            slippage_amount=slippage_amount,
            slippage_percentage=slippage_percentage,
            market_impact=total_market_impact,
            estimated_fill_time=estimated_fill_time,
            confidence=confidence
        )

    def _calculate_slippage_confidence(
        self,
        order_size: float,
        market_stats: Dict[str, float],
        market_impact: float
    ) -> float:
        """計算滑點估計信心度"""
        # 基於訂單規模、市場流動性和估計影響計算信心度

        # 訂單規模分數（越小信心越高）
        size_score = max(0, 1 - order_size / market_stats['avg_volume_24h'])

        # 市場流動性分數（成交量越大信心越高）
        liquidity_score = min(1.0, market_stats['avg_volume_24h'] / 1000000)

        # 影響大小分數（影響越小信心越高）
        impact_score = max(0, 1 - market_impact * 100)

        # 加權綜合分數
        confidence = (
            size_score * 0.4 +
            liquidity_score * 0.4 +
            impact_score * 0.2
        )

        return confidence

    def estimate_optimal_order_size(
        self,
        symbol: str,
        max_slippage_pct: float = 0.01,  # 最大可接受滑點1%
        market_data: pd.DataFrame = None,
        target_execution_time: float = 3600  # 目標執行時間（秒）
    ) -> float:
        """
        估計最優訂單規模

        Args:
            symbol: 交易對符號
            max_slippage_pct: 最大可接受滑點百分比
            market_data: 市場數據
            target_execution_time: 目標執行時間（秒）

        Returns:
            最優訂單規模
        """
        market_stats = self._get_market_statistics(symbol, market_data)

        # 基於滑點約束計算最優規模
        if self.config.method == "square_root":
            # 平方根模型：滑點 = k * sqrt(size / volume)
            # => size = (max_slippage / k)^2 * volume
            max_market_impact = max_slippage_pct
            optimal_size = (max_market_impact / self.config.market_impact_factor) ** 2 * market_stats['avg_volume_24h']

        elif self.config.method == "linear":
            # 線性模型：滑點 = k * (size / depth)
            # => size = (max_slippage / k) * depth
            max_market_impact = max_slippage_pct
            optimal_size = (max_market_impact / self.config.market_impact_factor) * market_stats['order_book_depth']

        else:
            # 默認平方根模型
            max_market_impact = max_slippage_pct
            optimal_size = (max_market_impact / self.config.market_impact_factor) ** 2 * market_stats['avg_volume_24h']

        # 考慮執行時間約束
        if target_execution_time > 0:
            max_size_by_time = market_stats['avg_volume_24h'] * (target_execution_time / 86400)  # 按日成交量比例
            optimal_size = min(optimal_size, max_size_by_time)

        # 確保規模合理
        optimal_size = max(0, min(optimal_size, market_stats['avg_volume_24h'] * 0.1))  # 不超過日成交量的10%

        logger.info(f"Optimal order size for {symbol}: {optimal_size:.2f} (max slippage: {max_slippage_pct:.2%})")
        return optimal_size

    def simulate_order_execution(
        self,
        symbol: str,
        order_size: float,
        order_price: float,
        order_type: str = "market",
        num_slices: int = 1,
        market_data: pd.DataFrame = None
    ) -> List[SlippageResult]:
        """
        模擬訂單執行過程

        Args:
            symbol: 交易對符號
            order_size: 訂單規模
            order_price: 訂單價格
            order_type: 訂單類型
            num_slices: 分割數量（用於冰山單）
            market_data: 市場數據

        Returns:
            執行過程中的滑點結果列表
        """
        if num_slices <= 1:
            # 單一訂單執行
            return [self.calculate_slippage(symbol, order_size, order_price, order_type, market_data)]

        # 分割訂單執行（冰山單）
        slice_size = order_size / num_slices
        execution_results = []

        for i in range(num_slices):
            # 模擬每個分割的執行
            # 假設市場影響會隨著時間遞減（市場吸收影響）
            time_decay = 1 - (i * 0.1)  # 每個分割影響減少10%

            # 調整市場影響因子
            original_factor = self.config.market_impact_factor
            self.config.market_impact_factor *= time_decay

            try:
                slice_result = self.calculate_slippage(
                    symbol, slice_size, order_price, order_type, market_data
                )
                execution_results.append(slice_result)
            finally:
                # 恢復原始因子
                self.config.market_impact_factor = original_factor

        return execution_results

    def calculate_volume_weighted_slippage(
        self,
        symbol: str,
        total_order_size: float,
        market_data: pd.DataFrame,
        time_window: int = 3600  # 時間窗口（秒）
    ) -> Dict[str, float]:
        """
        計算體積加權平均滑點

        Args:
            symbol: 交易對符號
            total_order_size: 總訂單規模
            market_data: 市場數據
            time_window: 時間窗口（秒）

        Returns:
            體積加權滑點統計
        """
        market_stats = self._get_market_statistics(symbol, market_data)

        # 將總訂單分成多個時間窗口執行
        num_windows = max(1, int(total_order_size / market_stats['avg_volume_24h'] * (86400 / time_window)))

        window_size = total_order_size / num_windows
        window_slippages = []

        for i in range(num_windows):
            # 計算每個時間窗口的滑點
            window_result = self.calculate_slippage(
                symbol, window_size, market_stats.get('current_price', 100),
                market_data=market_data
            )

            window_slippages.append(window_result.slippage_percentage)

        # 計算加權統計
        if window_slippages:
            avg_slippage = np.mean(window_slippages)
            max_slippage = np.max(window_slippages)
            min_slippage = np.min(window_slippages)

            # 體積加權（每個窗口的權重為訂單規模）
            weighted_slippage = sum(
                slippage * window_size for slippage, window_size in zip(window_slippages, [window_size] * len(window_slippages))
            ) / total_order_size
        else:
            avg_slippage = max_slippage = min_slippage = weighted_slippage = 0

        return {
            'avg_slippage': avg_slippage,
            'max_slippage': max_slippage,
            'min_slippage': min_slippage,
            'weighted_slippage': weighted_slippage,
            'num_windows': num_windows,
            'window_size': window_size
        }

    def train_slippage_model(
        self,
        historical_data: pd.DataFrame,
        symbols: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        訓練滑點模型參數

        Args:
            historical_data: 歷史交易數據
            symbols: 訓練的交易對列表

        Returns:
            訓練後的模型參數
        """
        if symbols is None:
            symbols = historical_data['symbol'].unique().tolist()

        logger.info(f"Training slippage model for {len(symbols)} symbols")

        model_parameters = {}

        for symbol in symbols:
            try:
                symbol_data = historical_data[historical_data['symbol'] == symbol]

                if len(symbol_data) < 50:  # 需要足夠數據
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue

                # 提取特徵和標籤
                features, labels = self._extract_slippage_features(symbol_data)

                if len(features) < 20:  # 需要足夠樣本
                    continue

                # 訓練線性回歸模型
                parameters = self._train_linear_model(features, labels)

                model_parameters[symbol] = parameters

                logger.info(f"Trained model for {symbol}: impact_factor={parameters.get('impact_factor', 0):.6f}")

            except Exception as e:
                logger.error(f"Error training model for {symbol}: {e}")
                continue

        return model_parameters

    def _extract_slippage_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """提取滑點預測特徵"""
        features_list = []
        labels = []

        for i in range(50, len(data)):  # 需要50個觀測值作為特徵
            # 計算特徵
            window_data = data.iloc[i-50:i]

            # 訂單規模相關特徵（這裡簡化為成交量變化）
            volume_change = window_data['volume'].pct_change().mean()
            volume_volatility = window_data['volume'].std() / window_data['volume'].mean()

            # 價格相關特徵
            price_change = window_data['close'].pct_change().mean()
            price_volatility = window_data['close'].pct_change().std()

            # 市場深度特徵（簡化）
            spread_estimate = price_volatility * 0.1

            features = [
                volume_change,
                volume_volatility,
                price_change,
                price_volatility,
                spread_estimate
            ]

            features_list.append(features)

            # 標籤：下一個時段的價格變化（作為滑點代理）
            next_price_change = (data.iloc[i]['close'] - data.iloc[i-1]['close']) / data.iloc[i-1]['close']
            labels.append(abs(next_price_change))  # 使用絕對值作為滑點估計

        features_df = pd.DataFrame(features_list, columns=[
            'volume_change', 'volume_volatility', 'price_change',
            'price_volatility', 'spread_estimate'
        ])

        return features_df, pd.Series(labels)

    def _train_linear_model(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
        """訓練線性回歸模型"""
        try:
            # 簡化的線性回歸訓練
            X = features.values
            y = labels.values

            # 添加常數項
            X = np.column_stack([np.ones(X.shape[0]), X])

            # 計算線性回歸係數
            coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

            # 轉換為參數字典
            param_names = ['intercept'] + list(features.columns)
            parameters = dict(zip(param_names, coefficients))

            # 計算市場影響因子（基於係數估計）
            impact_factor = abs(parameters.get('volume_volatility', 0)) * 0.001

            return {
                'impact_factor': impact_factor,
                'coefficients': parameters,
                'r_squared': self._calculate_r_squared(X @ coefficients, y)
            }

        except Exception as e:
            logger.error(f"Error in linear model training: {e}")
            return {'impact_factor': self.config.market_impact_factor}

    def _calculate_r_squared(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """計算R平方值"""
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        if ss_tot == 0:
            return 0

        return 1 - (ss_res / ss_tot)

    def generate_slippage_report(
        self,
        slippage_results: List[SlippageResult],
        symbol: str = None
    ) -> str:
        """生成滑點分析報告"""
        if not slippage_results:
            return "No slippage data available"

        report = []
        report.append("=" * 60)
        report.append("滑點分析報告")
        report.append("=" * 60)

        if symbol:
            report.append(f"交易對: {symbol}")

        # 統計數據
        slippages = [r.slippage_percentage for r in slippage_results]
        market_impacts = [r.market_impact for r in slippage_results]
        fill_times = [r.estimated_fill_time for r in slippage_results]
        confidences = [r.confidence for r in slippage_results]

        report.append(f"樣本數量: {len(slippage_results)}")
        report.append(f"平均滑點: {np.mean(slippages):.4f} ({np.mean(slippages)*100:.2f}%)")
        report.append(f"滑點範圍: {np.min(slippages):.4f} - {np.max(slippages):.4f}")
        report.append(f"平均市場影響: {np.mean(market_impacts):.4f}")
        report.append(f"平均預計成交時間: {np.mean(fill_times):.0f}秒")
        report.append(f"平均信心度: {np.mean(confidences):.2f}")
        report.append("")

        # 滑點分佈
        slippage_bins = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 1.0]
        slippage_distribution = np.histogram(slippages, bins=slippage_bins)[0]

        report.append("滑點分佈:")
        for i in range(len(slippage_bins) - 1):
            percentage = slippage_distribution[i] / len(slippages) * 100
            report.append(f"  {slippage_bins[i]:.3f} - {slippage_bins[i+1]:.3f}: {percentage:.1f}%")

        report.append("=" * 60)

        return "\n".join(report)


# 使用範例
if __name__ == "__main__":
    # 創建滑點模擬器
    config = SlippageModelConfig(
        method="square_root",
        market_impact_factor=0.001,
        volume_participation_rate=0.1
    )

    slippage_model = SlippageModel(config)

    # 創建範例市場數據
    sample_data = {
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
        'symbol': ['BTC'] * 100,
        'open': [50000 + i * 10 for i in range(100)],
        'high': [50100 + i * 10 for i in range(100)],
        'low': [49900 + i * 10 for i in range(100)],
        'close': [50050 + i * 10 for i in range(100)],
        'volume': [1000000 + i * 10000 for i in range(100)]
    }

    market_df = pd.DataFrame(sample_data)

    # 計算單筆訂單滑點
    slippage_result = slippage_model.calculate_slippage(
        symbol='BTC',
        order_size=100000,  # 1億美元訂單
        order_price=50000,
        order_type="market",
        market_data=market_df
    )

    print(f"預計執行價格: ${slippage_result.executed_price:.2f}")
    print(f"滑點金額: ${slippage_result.slippage_amount:.2f}")
    print(f"滑點百分比: {slippage_result.slippage_percentage:.4f}")
    print(f"市場影響: {slippage_result.market_impact:.4f}")
    print(f"預計成交時間: {slippage_result.estimated_fill_time:.0f}秒")
    print(f"信心度: {slippage_result.confidence:.2f}")

    # 計算最優訂單規模
    optimal_size = slippage_model.estimate_optimal_order_size(
        symbol='BTC',
        max_slippage_pct=0.01,  # 最大1%滑點
        market_data=market_df
    )

    print(f"\n最優訂單規模: ${optimal_size:,.0f}")

    # 模擬訂單執行過程
    execution_results = slippage_model.simulate_order_execution(
        symbol='BTC',
        order_size=100000,
        order_price=50000,
        order_type="iceberg",
        num_slices=5,
        market_data=market_df
    )

    print(f"\n冰山單執行模擬 ({len(execution_results)}個分割):")
    for i, result in enumerate(execution_results):
        print(f"  分割 {i+1}: 滑點 {result.slippage_percentage:.4f}")

    # 生成報告
    report = slippage_model.generate_slippage_report(execution_results, 'BTC')
    print("\n滑點分析報告:")
    print(report)
