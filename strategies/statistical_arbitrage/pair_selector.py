"""
配對選擇演算法
實現統計套利策略中的配對選擇，包含相關性分析和共整合檢定
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from scipy import stats

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class PairCandidate:
    """配對候選"""
    symbol1: str
    symbol2: str
    correlation: float
    cointegration_pvalue: float
    half_life: float
    hedge_ratio: float
    score: float
    price_ratio: float
    volume_ratio: float
    sector_similarity: float

@dataclass
class PairSelectionResult:
    """配對選擇結果"""
    selected_pairs: List[PairCandidate]
    total_candidates: int
    selection_criteria: Dict[str, Any]
    execution_time: float

class PairSelector:
    """配對選擇器"""

    def __init__(
        self,
        min_correlation: float = 0.7,
        max_cointegration_pvalue: float = 0.05,
        min_half_life: int = 1,
        max_half_life: int = 252,
        min_data_points: int = 252
    ):
        """
        初始化配對選擇器

        Args:
            min_correlation: 最小相關性係數
            max_cointegration_pvalue: 最大共整合檢定p值
            min_half_life: 最小半衰期（天）
            max_half_life: 最大半衰期（天）
            min_data_points: 最少數據點數
        """
        self.min_correlation = min_correlation
        self.max_cointegration_pvalue = max_cointegration_pvalue
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_data_points = min_data_points

    def select_pairs(
        self,
        price_data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        method: str = "correlation_cointegration",
        top_n: int = 20
    ) -> PairSelectionResult:
        """
        選擇交易配對

        Args:
            price_data: 價格數據DataFrame
            symbols: 候選交易對列表（如果為None，使用所有可用符號）
            method: 選擇方法
            top_n: 返回最優配對數量

        Returns:
            配對選擇結果
        """
        start_time = pd.Timestamp.now()

        logger.info(f"Starting pair selection with method: {method}")

        # 準備數據
        if symbols is None:
            symbols = price_data['symbol'].unique().tolist()

        logger.info(f"Processing {len(symbols)} symbols for pair selection")

        # 生成所有可能的配對組合
        all_pairs = self._generate_symbol_pairs(symbols)

        if not all_pairs:
            logger.warning("No valid pairs to analyze")
            return PairSelectionResult(
                selected_pairs=[],
                total_candidates=0,
                selection_criteria={},
                execution_time=(pd.Timestamp.now() - start_time).total_seconds()
            )

        # 計算每個配對的指標
        pair_candidates = []

        for symbol1, symbol2 in all_pairs:
            try:
                # 獲取價格序列
                prices1 = self._get_price_series(price_data, symbol1)
                prices2 = self._get_price_series(price_data, symbol2)

                if len(prices1) < self.min_data_points or len(prices2) < self.min_data_points:
                    continue

                # 計算配對指標
                candidate = self._analyze_pair(prices1, prices2, symbol1, symbol2)

                if candidate:
                    pair_candidates.append(candidate)

            except Exception as e:
                logger.error(f"Error analyzing pair {symbol1}-{symbol2}: {e}")
                continue

        # 根據方法篩選和排序
        if method == "correlation_cointegration":
            selected_pairs = self._filter_by_correlation_cointegration(pair_candidates)
        elif method == "distance_approach":
            selected_pairs = self._filter_by_distance_approach(pair_candidates)
        elif method == "score_based":
            selected_pairs = self._filter_by_composite_score(pair_candidates)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        # 返回前N個最優配對
        selected_pairs = selected_pairs[:top_n]

        execution_time = (pd.Timestamp.now() - start_time).total_seconds()

        logger.info(f"Pair selection completed: {len(selected_pairs)} pairs selected from {len(pair_candidates)} candidates")

        return PairSelectionResult(
            selected_pairs=selected_pairs,
            total_candidates=len(pair_candidates),
            selection_criteria={
                'method': method,
                'min_correlation': self.min_correlation,
                'max_cointegration_pvalue': self.max_cointegration_pvalue,
                'min_half_life': self.min_half_life,
                'max_half_life': self.max_half_life,
                'min_data_points': self.min_data_points
            },
            execution_time=execution_time
        )

    def _generate_symbol_pairs(self, symbols: List[str]) -> List[Tuple[str, str]]:
        """生成所有可能的配對組合"""
        pairs = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pairs.append((symbols[i], symbols[j]))

        logger.info(f"Generated {len(pairs)} possible pairs from {len(symbols)} symbols")
        return pairs

    def _get_price_series(self, price_data: pd.DataFrame, symbol: str) -> pd.Series:
        """獲取指定符號的價格序列"""
        symbol_data = price_data[price_data['symbol'] == symbol]

        if symbol_data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        # 確保時間戳排序
        symbol_data = symbol_data.sort_values('timestamp')

        return symbol_data['close'].reset_index(drop=True)

    def _analyze_pair(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        symbol1: str,
        symbol2: str
    ) -> Optional[PairCandidate]:
        """分析單個配對"""
        try:
            # 1. 計算相關性
            correlation, _ = stats.pearsonr(prices1, prices2)

            if abs(correlation) < self.min_correlation:
                return None

            # 2. 共整合檢定
            coint_result = coint(prices1, prices2)
            coint_pvalue = coint_result[1]

            if coint_pvalue > self.max_cointegration_pvalue:
                return None

            # 3. 計算對沖比率（線性回歸）
            hedge_ratio = self._calculate_hedge_ratio(prices1, prices2)

            # 4. 計算價差序列
            spread = prices2 - hedge_ratio * prices1

            # 5. 計算半衰期（均值回歸速度）
            half_life = self._calculate_half_life(spread)

            if half_life < self.min_half_life or half_life > self.max_half_life:
                return None

            # 6. 計算其他指標
            price_ratio = prices2.iloc[-1] / prices1.iloc[-1]
            volume_ratio = 1.0  # 簡化處理，實際應使用成交量數據

            # 7. 計算部門相似性（簡化為價格相關性的絕對值）
            sector_similarity = abs(correlation)

            # 8. 計算綜合得分
            score = self._calculate_pair_score(
                correlation, coint_pvalue, half_life, sector_similarity
            )

            return PairCandidate(
                symbol1=symbol1,
                symbol2=symbol2,
                correlation=correlation,
                cointegration_pvalue=coint_pvalue,
                half_life=half_life,
                hedge_ratio=hedge_ratio,
                score=score,
                price_ratio=price_ratio,
                volume_ratio=volume_ratio,
                sector_similarity=sector_similarity
            )

        except Exception as e:
            logger.error(f"Error in pair analysis for {symbol1}-{symbol2}: {e}")
            return None

    def _calculate_hedge_ratio(self, prices1: pd.Series, prices2: pd.Series) -> float:
        """計算對沖比率"""
        # 使用線性回歸計算對沖比率
        X = sm.add_constant(prices1)
        model = sm.OLS(prices2, X)
        results = model.fit()

        hedge_ratio = results.params.iloc[1]  # 排除常數項

        return hedge_ratio

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """計算半衰期"""
        # 計算價差的一階差分
        spread_diff = spread.diff().dropna()

        # 確保沒有NaN值
        valid_data = spread_diff.dropna()
        if len(valid_data) < 2:
            return float('inf')

        # 擬合一階自回歸模型：Δspread_t = α + β * spread_{t-1} + ε_t
        spread_lag = spread.shift(1).dropna()

        # 確保長度匹配
        min_length = min(len(valid_data), len(spread_lag))
        spread_diff_clean = valid_data.iloc[:min_length]
        spread_lag_clean = spread_lag.iloc[:min_length]

        if len(spread_diff_clean) < 10:  # 需要足夠數據點
            return float('inf')

        # 擬合線性回歸
        X = sm.add_constant(spread_lag_clean)
        model = sm.OLS(spread_diff_clean, X)
        results = model.fit()

        # 計算半衰期
        beta = results.params.iloc[1]  # 自回歸係數

        if beta >= 1 or beta <= -1:
            return float('inf')  # 非平穩序列

        half_life = -np.log(2) / np.log(1 + beta)  # 轉換為天數單位

        return half_life

    def _calculate_pair_score(
        self,
        correlation: float,
        coint_pvalue: float,
        half_life: float,
        sector_similarity: float
    ) -> float:
        """計算配對綜合得分"""
        # 標準化各個指標為0-1分數
        correlation_score = abs(correlation)  # 相關性絕對值

        # 共整合p值越小越好（轉換為分數）
        coint_score = 1 - coint_pvalue

        # 半衰期適中為佳（使用高斯函數）
        optimal_half_life = (self.min_half_life + self.max_half_life) / 2
        half_life_score = np.exp(-0.5 * ((half_life - optimal_half_life) / 50) ** 2)

        # 部門相似性分數
        similarity_score = sector_similarity

        # 加權綜合得分
        weights = {
            'correlation': 0.3,
            'cointegration': 0.3,
            'half_life': 0.25,
            'similarity': 0.15
        }

        total_score = (
            correlation_score * weights['correlation'] +
            coint_score * weights['cointegration'] +
            half_life_score * weights['half_life'] +
            similarity_score * weights['similarity']
        )

        return total_score

    def _filter_by_correlation_cointegration(
        self,
        candidates: List[PairCandidate]
    ) -> List[PairCandidate]:
        """基於相關性和共整合的篩選方法"""
        # 按綜合得分排序
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

        logger.info(f"Selected {len(sorted_candidates)} pairs by correlation-cointegration method")
        return sorted_candidates

    def _filter_by_distance_approach(
        self,
        candidates: List[PairCandidate]
    ) -> List[PairCandidate]:
        """基於距離的方法（簡化實現）"""
        # 標準化指標
        normalized_candidates = []

        for candidate in candidates:
            # 標準化相關性
            corr_norm = abs(candidate.correlation)

            # 標準化共整合p值（越小越好）
            coint_norm = 1 - candidate.cointegration_pvalue

            # 標準化半衰期（適中為佳）
            optimal_half_life = (self.min_half_life + self.max_half_life) / 2
            half_life_norm = 1 - abs(candidate.half_life - optimal_half_life) / optimal_half_life

            # 計算歐幾里得距離（距離越小越好）
            distance = np.sqrt(
                (1 - corr_norm) ** 2 +
                (1 - coint_norm) ** 2 +
                (1 - half_life_norm) ** 2
            )

            candidate.score = 1 - distance  # 轉換為分數
            normalized_candidates.append(candidate)

        # 按距離分數排序
        sorted_candidates = sorted(normalized_candidates, key=lambda x: x.score, reverse=True)

        logger.info(f"Selected {len(sorted_candidates)} pairs by distance approach")
        return sorted_candidates

    def _filter_by_composite_score(
        self,
        candidates: List[PairCandidate]
    ) -> List[PairCandidate]:
        """基於綜合得分的篩選方法"""
        # 已經在_analyze_pair中計算了綜合得分，直接排序即可
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

        logger.info(f"Selected {len(sorted_candidates)} pairs by composite score")
        return sorted_candidates

    def analyze_pair_detailed(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        symbol1: str = "Asset1",
        symbol2: str = "Asset2"
    ) -> Dict[str, Any]:
        """
        詳細分析配對

        Args:
            prices1: 第一個資產價格序列
            prices2: 第二個資產價格序列
            symbol1: 第一個資產名稱
            symbol2: 第二個資產名稱

        Returns:
            詳細分析結果
        """
        try:
            # 基本統計
            basic_stats = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'data_points': len(prices1),
                'price1_stats': {
                    'mean': prices1.mean(),
                    'std': prices1.std(),
                    'min': prices1.min(),
                    'max': prices1.max()
                },
                'price2_stats': {
                    'mean': prices2.mean(),
                    'std': prices2.std(),
                    'min': prices2.min(),
                    'max': prices2.max()
                }
            }

            # 相關性分析
            correlation, corr_pvalue = stats.pearsonr(prices1, prices2)

            # 共整合檢定
            coint_t_stat, coint_pvalue, critical_values = coint(prices1, prices2)

            # 對沖比率
            hedge_ratio = self._calculate_hedge_ratio(prices1, prices2)

            # 價差統計
            spread = prices2 - hedge_ratio * prices1
            spread_stats = {
                'mean': spread.mean(),
                'std': spread.std(),
                'min': spread.min(),
                'max': spread.max(),
                'adf_pvalue': adfuller(spread)[1]  # ADF檢定p值
            }

            # 半衰期
            half_life = self._calculate_half_life(spread)

            # 交易成本估計（簡化）
            price_volatility = np.std(np.log(prices1 / prices1.shift(1)).dropna())
            trading_costs = {
                'estimated_spread_cost': price_volatility * 0.001,  # 假設0.1%交易成本
                'half_life_days': half_life,
                'annual_turnover': 252 / half_life if half_life > 0 else 0
            }

            return {
                'basic_stats': basic_stats,
                'correlation': {
                    'coefficient': correlation,
                    'p_value': corr_pvalue
                },
                'cointegration': {
                    't_statistic': coint_t_stat,
                    'p_value': coint_pvalue,
                    'critical_values': critical_values
                },
                'hedge_ratio': hedge_ratio,
                'spread_stats': spread_stats,
                'trading_costs': trading_costs,
                'recommendation': self._generate_recommendation(
                    correlation, coint_pvalue, half_life
                )
            }

        except Exception as e:
            logger.error(f"Error in detailed pair analysis: {e}")
            return {}

    def _generate_recommendation(
        self,
        correlation: float,
        coint_pvalue: float,
        half_life: float
    ) -> str:
        """生成配對交易建議"""
        recommendations = []

        if abs(correlation) < self.min_correlation:
            recommendations.append(f"相關性過低 ({correlation:.".3f"), 不適合配對交易")

        if coint_pvalue > self.max_cointegration_pvalue:
            recommendations.append(f"共整合檢定失敗 (p值={coint_pvalue:.".3f"), 價差可能非平穩")

        if half_life < self.min_half_life:
            recommendations.append(f"半衰期過短 ({half_life:.1f}天), 可能過於頻繁交易")

        if half_life > self.max_half_life:
            recommendations.append(f"半衰期過長 ({half_life:.".1f"天), 可能回歸速度過慢")

        if not recommendations:
            recommendations.append("配對滿足基本條件，適合進行統計套利")

        return "; ".join(recommendations)

    def filter_by_sector_similarity(
        self,
        candidates: List[PairCandidate],
        min_similarity: float = 0.5
    ) -> List[PairCandidate]:
        """
        按部門相似性過濾配對

        Args:
            candidates: 配對候選列表
            min_similarity: 最小相似性閾值

        Returns:
            過濾後的配對列表
        """
        filtered = [c for c in candidates if c.sector_similarity >= min_similarity]

        logger.info(f"Filtered {len(candidates)} pairs by sector similarity "
                   f"({min_similarity}), {len(filtered)} pairs remaining")

        return filtered

    def filter_by_half_life_range(
        self,
        candidates: List[PairCandidate],
        min_half_life: float = None,
        max_half_life: float = None
    ) -> List[PairCandidate]:
        """
        按半衰期範圍過濾配對

        Args:
            candidates: 配對候選列表
            min_half_life: 最小半衰期
            max_half_life: 最大半衰期

        Returns:
            過濾後的配對列表
        """
        if min_half_life is None:
            min_half_life = self.min_half_life
        if max_half_life is None:
            max_half_life = self.max_half_life

        filtered = [
            c for c in candidates
            if min_half_life <= c.half_life <= max_half_life
        ]

        logger.info(f"Filtered {len(candidates)} pairs by half-life range "
                   f"({min_half_life}-{max_half_life}), {len(filtered)} pairs remaining")

        return filtered

    def generate_pair_report(self, pair: PairCandidate) -> str:
        """生成配對報告"""
        report = []
        report.append("=" * 60)
        report.append(f"配對分析報告: {pair.symbol1} - {pair.symbol2}")
        report.append("=" * 60)
        report.append(f"相關性係數: {pair.correlation".4f"}")
        report.append(f"共整合檢定p值: {pair.cointegration_pvalue".4f"}")
        report.append(f"半衰期: {pair.half_life".1f"} 天")
        report.append(f"對沖比率: {pair.hedge_ratio".4f"}")
        report.append(f"綜合得分: {pair.score".4f"}")
        report.append(f"價格比率: {pair.price_ratio".4f"}")
        report.append(f"部門相似性: {pair.sector_similarity".4f"}")
        report.append("=" * 60)

        return "\n".join(report)


# 使用範例
if __name__ == "__main__":
    # 創建配對選擇器
    selector = PairSelector(
        min_correlation=0.7,
        max_cointegration_pvalue=0.05,
        min_half_life=5,
        max_half_life=100
    )

    # 創建範例價格數據
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')

    # 創建相關的價格序列
    price1 = 100 + np.cumsum(np.random.randn(252) * 0.01)
    price2 = 50 + 0.5 * price1 + np.cumsum(np.random.randn(252) * 0.005)  # 相關性約0.8

    # 添加一些無關的價格序列
    price3 = 200 + np.cumsum(np.random.randn(252) * 0.02)  # 低相關性

    sample_data = []
    for i, date in enumerate(dates):
        sample_data.extend([
            {'timestamp': date, 'symbol': 'BTC', 'close': price1[i]},
            {'timestamp': date, 'symbol': 'ETH', 'close': price2[i]},
            {'timestamp': date, 'symbol': 'LTC', 'close': price3[i]}
        ])

    df = pd.DataFrame(sample_data)

    # 執行配對選擇
    result = selector.select_pairs(df, method="correlation_cointegration", top_n=5)

    print(f"選擇了 {len(result.selected_pairs)} 個配對:")
    for i, pair in enumerate(result.selected_pairs, 1):
        print(f"{i}. {pair.symbol1} - {pair.symbol2}")
        print(f"   相關性: {pair.correlation".3f"}")
        print(f"   半衰期: {pair.half_life".1f"} 天")
        print(f"   得分: {pair.score".3f"}")
        print()

    # 詳細分析最佳配對
    if result.selected_pairs:
        best_pair = result.selected_pairs[0]
        detailed_analysis = selector.analyze_pair_detailed(
            price1, price2, best_pair.symbol1, best_pair.symbol2
        )

        print("最佳配對詳細分析:")
        for key, value in detailed_analysis.items():
            print(f"{key}: {value}")
