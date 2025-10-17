"""
高效數據查詢接口
提供優化的數據查詢功能，支援複雜查詢條件和高效數據檢索
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import logging
import time
import re
from dataclasses import dataclass
import pickle

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class QueryFilter:
    """查詢過濾條件"""
    column: str
    operator: str  # 'eq', 'gt', 'lt', 'gte', 'lte', 'in', 'between', 'like'
    value: Any
    logical_op: str = 'AND'  # 'AND', 'OR'

@dataclass
class QueryOptions:
    """查詢選項"""
    sort_by: Optional[str] = None
    sort_order: str = 'asc'  # 'asc', 'desc'
    limit: Optional[int] = None
    offset: int = 0
    group_by: Optional[str] = None
    aggregations: Optional[Dict[str, str]] = None  # column -> agg_function

@dataclass
class QueryResult:
    """查詢結果"""
    data: pd.DataFrame
    total_count: int
    filtered_count: int
    execution_time: float
    metadata: Dict[str, Any]

class DataQueryEngine:
    """高效數據查詢引擎"""

    def __init__(self, storage_manager):
        """
        初始化查詢引擎

        Args:
            storage_manager: 存儲管理器實例
        """
        self.storage_manager = storage_manager
        self.query_cache = {}
        self.cache_ttl = 300  # 緩存時間（秒）

    def query(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1h",
        filters: Optional[List[QueryFilter]] = None,
        options: Optional[QueryOptions] = None,
        use_cache: bool = True
    ) -> QueryResult:
        """
        執行數據查詢

        Args:
            symbols: 交易對符號列表
            start_date: 開始日期
            end_date: 結束日期
            timeframe: 時間週期
            filters: 查詢過濾條件
            options: 查詢選項
            use_cache: 是否使用緩存

        Returns:
            查詢結果
        """
        start_time = time.time()

        # 生成查詢緩存鍵
        cache_key = self._generate_cache_key(symbols, start_date, end_date, timeframe, filters, options)

        # 檢查緩存
        if use_cache and cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                logger.info(f"Cache hit for query: {cache_key}")
                return cache_entry['result']

        try:
            # 執行查詢
            data = self._execute_query(symbols, start_date, end_date, timeframe, filters, options)

            # 應用查詢選項
            if options:
                data = self._apply_query_options(data, options)

            # 創建結果對象
            execution_time = time.time() - start_time
            result = QueryResult(
                data=data,
                total_count=len(data),
                filtered_count=len(data),
                execution_time=execution_time,
                metadata={
                    'symbols': symbols,
                    'timeframe': timeframe,
                    'start_date': start_date,
                    'end_date': end_date,
                    'filters_applied': len(filters) if filters else 0,
                    'cache_used': False
                }
            )

            # 緩存結果
            if use_cache:
                self.query_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }

            logger.info(f"Query executed: {len(data)} records in {execution_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def _generate_cache_key(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str,
        filters: List[QueryFilter],
        options: QueryOptions
    ) -> str:
        """生成查詢緩存鍵"""
        # 創建確定性的緩存鍵
        key_parts = [
            '_'.join(sorted(symbols)),
            str(start_date),
            str(end_date),
            timeframe
        ]

        if filters:
            filter_str = '_'.join([
                f"{f.column}_{f.operator}_{str(f.value)}" for f in sorted(filters, key=lambda x: x.column)
            ])
            key_parts.append(filter_str)

        if options:
            options_str = f"{options.sort_by}_{options.sort_order}_{options.limit}_{options.offset}"
            key_parts.append(options_str)

        cache_key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()
        return cache_key

    def _execute_query(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str,
        filters: List[QueryFilter],
        options: QueryOptions
    ) -> pd.DataFrame:
        """執行實際查詢"""
        # 使用存儲管理器查詢基礎數據
        query = DataQuery(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )

        df = self.storage_manager.query_data(query)

        if df.empty:
            return df

        # 應用過濾條件
        if filters:
            df = self._apply_filters(df, filters)

        return df

    def _apply_filters(self, df: pd.DataFrame, filters: List[QueryFilter]) -> pd.DataFrame:
        """應用過濾條件"""
        filtered_df = df.copy()

        for filter_condition in filters:
            try:
                if filter_condition.operator == 'eq':
                    mask = filtered_df[filter_condition.column] == filter_condition.value
                elif filter_condition.operator == 'gt':
                    mask = filtered_df[filter_condition.column] > filter_condition.value
                elif filter_condition.operator == 'lt':
                    mask = filtered_df[filter_condition.column] < filter_condition.value
                elif filter_condition.operator == 'gte':
                    mask = filtered_df[filter_condition.column] >= filter_condition.value
                elif filter_condition.operator == 'lte':
                    mask = filtered_df[filter_condition.column] <= filter_condition.value
                elif filter_condition.operator == 'in':
                    mask = filtered_df[filter_condition.column].isin(filter_condition.value)
                elif filter_condition.operator == 'between':
                    if isinstance(filter_condition.value, (list, tuple)) and len(filter_condition.value) == 2:
                        mask = (filtered_df[filter_condition.column] >= filter_condition.value[0]) & \
                               (filtered_df[filter_condition.column] <= filter_condition.value[1])
                    else:
                        logger.warning(f"Invalid between value for {filter_condition.column}: {filter_condition.value}")
                        continue
                elif filter_condition.operator == 'like':
                    mask = filtered_df[filter_condition.column].astype(str).str.contains(str(filter_condition.value), na=False)
                else:
                    logger.warning(f"Unsupported operator: {filter_condition.operator}")
                    continue

                # 應用邏輯運算子
                if filter_condition.logical_op == 'AND':
                    filtered_df = filtered_df[mask]
                elif filter_condition.logical_op == 'OR':
                    filtered_df = pd.concat([filtered_df, df[mask]], ignore_index=True).drop_duplicates()

            except Exception as e:
                logger.error(f"Error applying filter {filter_condition.column} {filter_condition.operator}: {e}")
                continue

        return filtered_df

    def _apply_query_options(self, df: pd.DataFrame, options: QueryOptions) -> pd.DataFrame:
        """應用查詢選項"""
        result_df = df.copy()

        # 排序
        if options.sort_by and options.sort_by in result_df.columns:
            ascending = options.sort_order.lower() == 'asc'
            result_df = result_df.sort_values(options.sort_by, ascending=ascending)

        # 分頁
        if options.offset > 0:
            result_df = result_df.iloc[options.offset:]

        if options.limit:
            result_df = result_df.head(options.limit)

        # 分組聚合
        if options.group_by and options.aggregations:
            result_df = self._apply_grouping(result_df, options)

        return result_df

    def _apply_grouping(self, df: pd.DataFrame, options: QueryOptions) -> pd.DataFrame:
        """應用分組聚合"""
        try:
            grouped = df.groupby(options.group_by)

            # 應用聚合函數
            agg_dict = {}
            for column, func in options.aggregations.items():
                if column in df.columns:
                    agg_dict[column] = func

            if agg_dict:
                result_df = grouped.agg(agg_dict).reset_index()
                logger.info(f"Applied grouping by {options.group_by} with aggregations: {agg_dict}")
                return result_df

        except Exception as e:
            logger.error(f"Error applying grouping: {e}")

        return df

    def get_aggregated_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        group_by: str = 'symbol',
        aggregations: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        獲取聚合數據

        Args:
            symbols: 交易對符號列表
            start_date: 開始日期
            end_date: 結束日期
            group_by: 分組欄位
            aggregations: 聚合函數字典

        Returns:
            聚合後的DataFrame
        """
        if aggregations is None:
            aggregations = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }

        # 查詢基礎數據
        result = self.query(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            options=QueryOptions(
                group_by=group_by,
                aggregations=aggregations
            )
        )

        return result.data

    def get_price_statistics(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        獲取價格統計信息

        Args:
            symbols: 交易對符號列表
            start_date: 開始日期
            end_date: 結束日期
            timeframe: 時間週期

        Returns:
            價格統計DataFrame
        """
        # 查詢數據
        result = self.query(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )

        if result.data.empty:
            return pd.DataFrame()

        df = result.data.copy()

        # 計算統計指標
        stats_data = []
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol]

            if symbol_data.empty:
                continue

            price_stats = {
                'symbol': symbol,
                'start_date': symbol_data['timestamp'].min().strftime('%Y-%m-%d'),
                'end_date': symbol_data['timestamp'].max().strftime('%Y-%m-%d'),
                'record_count': len(symbol_data),
                'price_change': (symbol_data['close'].iloc[-1] - symbol_data['close'].iloc[0]) / symbol_data['close'].iloc[0] if len(symbol_data) > 1 else 0,
                'price_volatility': symbol_data['close'].pct_change().std(),
                'avg_volume': symbol_data['volume'].mean(),
                'max_price': symbol_data['high'].max(),
                'min_price': symbol_data['low'].min(),
                'avg_price': symbol_data['close'].mean()
            }

            stats_data.append(price_stats)

        return pd.DataFrame(stats_data)

    def search_time_series(
        self,
        symbols: List[str],
        pattern: str = "peak|valley|trend",
        start_date: str = None,
        end_date: str = None,
        sensitivity: float = 0.02
    ) -> Dict[str, List[Dict]]:
        """
        搜尋時間序列模式

        Args:
            symbols: 交易對符號列表
            pattern: 模式類型（peak, valley, trend）
            start_date: 開始日期
            end_date: 結束日期
            sensitivity: 敏感度（價格變動閾值）

        Returns:
            檢測到的模式字典
        """
        # 查詢數據
        result = self.query(symbols=symbols, start_date=start_date, end_date=end_date)

        if result.data.empty:
            return {}

        df = result.data.copy()
        patterns_found = {}

        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol]

            if len(symbol_data) < 10:  # 需要足夠數據點
                continue

            symbol_patterns = []

            if 'peak' in pattern.lower():
                peaks = self._detect_peaks(symbol_data, sensitivity)
                symbol_patterns.extend(peaks)

            if 'valley' in pattern.lower():
                valleys = self._detect_valleys(symbol_data, sensitivity)
                symbol_patterns.extend(valleys)

            if 'trend' in pattern.lower():
                trends = self._detect_trends(symbol_data, sensitivity)
                symbol_patterns.extend(trends)

            if symbol_patterns:
                patterns_found[symbol] = symbol_patterns

        return patterns_found

    def _detect_peaks(self, df: pd.DataFrame, sensitivity: float) -> List[Dict]:
        """檢測價格峰值"""
        peaks = []

        for i in range(1, len(df) - 1):
            current_price = df.iloc[i]['close']
            prev_price = df.iloc[i-1]['close']
            next_price = df.iloc[i+1]['close']

            # 簡單的峰值檢測：當前價格高於前後價格且變動幅度超過閾值
            if current_price > prev_price and current_price > next_price:
                price_change = abs(current_price - prev_price) / prev_price
                if price_change >= sensitivity:
                    peaks.append({
                        'type': 'peak',
                        'timestamp': df.iloc[i]['timestamp'],
                        'price': current_price,
                        'change_pct': price_change
                    })

        return peaks

    def _detect_valleys(self, df: pd.DataFrame, sensitivity: float) -> List[Dict]:
        """檢測價格谷值"""
        valleys = []

        for i in range(1, len(df) - 1):
            current_price = df.iloc[i]['close']
            prev_price = df.iloc[i-1]['close']
            next_price = df.iloc[i+1]['close']

            # 簡單的谷值檢測：當前價格低於前後價格且變動幅度超過閾值
            if current_price < prev_price and current_price < next_price:
                price_change = abs(current_price - prev_price) / prev_price
                if price_change >= sensitivity:
                    valleys.append({
                        'type': 'valley',
                        'timestamp': df.iloc[i]['timestamp'],
                        'price': current_price,
                        'change_pct': price_change
                    })

        return valleys

    def _detect_trends(self, df: pd.DataFrame, sensitivity: float) -> List[Dict]:
        """檢測趨勢"""
        trends = []

        # 計算移動平均線
        df_copy = df.copy()
        df_copy['ma_short'] = df_copy['close'].rolling(window=5, min_periods=3).mean()
        df_copy['ma_long'] = df_copy['close'].rolling(window=20, min_periods=10).mean()

        # 檢測趨勢變化
        for i in range(10, len(df_copy)):
            current_short = df_copy.iloc[i]['ma_short']
            current_long = df_copy.iloc[i]['ma_long']
            prev_short = df_copy.iloc[i-1]['ma_short']
            prev_long = df_copy.iloc[i-1]['ma_long']

            # 檢測金叉（短期線上穿長期線）
            if prev_short <= prev_long and current_short > current_long:
                trends.append({
                    'type': 'bullish_crossover',
                    'timestamp': df_copy.iloc[i]['timestamp'],
                    'price': df_copy.iloc[i]['close']
                })

            # 檢測死叉（短期線下穿長期線）
            elif prev_short >= prev_long and current_short < current_long:
                trends.append({
                    'type': 'bearish_crossover',
                    'timestamp': df_copy.iloc[i]['timestamp'],
                    'price': df_copy.iloc[i]['close']
                })

        return trends

    def get_data_summary(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        獲取數據摘要統計

        Args:
            symbols: 交易對符號列表
            start_date: 開始日期
            end_date: 結束日期

        Returns:
            數據摘要統計DataFrame
        """
        # 查詢數據
        result = self.query(symbols=symbols, start_date=start_date, end_date=end_date)

        if result.data.empty:
            return pd.DataFrame()

        df = result.data.copy()

        # 按符號計算統計
        summary_data = []
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol]

            if symbol_data.empty:
                continue

            summary = {
                'symbol': symbol,
                'total_records': len(symbol_data),
                'date_range': f"{symbol_data['timestamp'].min().strftime('%Y-%m-%d')} to {symbol_data['timestamp'].max().strftime('%Y-%m-%d')}",
                'price_range': f"{symbol_data['low'].min():.2f} - {symbol_data['high'].max():.2f}",
                'avg_price': symbol_data['close'].mean(),
                'price_std': symbol_data['close'].std(),
                'total_volume': symbol_data['volume'].sum(),
                'avg_volume': symbol_data['volume'].mean(),
                'data_completeness': self._calculate_completeness(symbol_data)
            }

            summary_data.append(summary)

        return pd.DataFrame(summary_data)

    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """計算數據完整性"""
        if len(df) < 2:
            return 1.0

        # 計算時間間隔
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else 3600

        # 計算缺失數據點數量
        total_expected_points = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / expected_interval
        completeness = len(df) / max(total_expected_points, 1)

        return min(completeness, 1.0)

    def export_query_result(
        self,
        result: QueryResult,
        filepath: str,
        format: str = "csv"
    ) -> str:
        """
        導出查詢結果

        Args:
            result: 查詢結果
            filepath: 導出檔案路徑
            format: 導出格式（csv, json, parquet）

        Returns:
            實際導出檔案路徑
        """
        filepath = Path(filepath)
        df = result.data.copy()

        try:
            if format.lower() == 'csv':
                df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                # 轉換時間戳為字符串
                df_export = df.copy()
                if 'timestamp' in df_export.columns:
                    df_export['timestamp'] = df_export['timestamp'].astype(str)
                df_export.to_json(filepath, orient='records', indent=2)
            elif format.lower() == 'parquet':
                df.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Exported query result to {filepath} ({format} format)")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting query result: {e}")
            raise

    def clear_cache(self):
        """清除查詢緩存"""
        self.query_cache.clear()
        logger.info("Query cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        return {
            'cache_size': len(self.query_cache),
            'cache_ttl': self.cache_ttl
        }


# 便捷查詢函數
def quick_query(
    storage_manager,
    symbols: List[str],
    start_date: str = None,
    end_date: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    快速查詢便捷函數

    Args:
        storage_manager: 存儲管理器實例
        symbols: 交易對符號列表
        start_date: 開始日期
        end_date: 結束日期
        **kwargs: 其他查詢參數

    Returns:
        查詢結果DataFrame
    """
    engine = DataQueryEngine(storage_manager)

    result = engine.query(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )

    return result.data


# 使用範例
if __name__ == "__main__":
    # 假設已有存儲管理器實例
    # storage_manager = StorageManager()

    # 創建查詢引擎
    # query_engine = DataQueryEngine(storage_manager)

    # 執行查詢範例
    print("Data query engine ready!")

    # 查詢範例：
    # result = query_engine.query(
    #     symbols=["BTC", "ETH"],
    #     start_date="2023-01-01",
    #     end_date="2023-12-31",
    #     filters=[
    #         QueryFilter(column="volume", operator="gt", value=1000000)
    #     ],
    #     options=QueryOptions(
    #         sort_by="timestamp",
    #         limit=1000
    #     )
    # )

    # 獲取價格統計
    # stats = query_engine.get_price_statistics(
    #     symbols=["BTC", "ETH"],
    #     start_date="2023-01-01",
    #     end_date="2023-12-31"
    # )

    # 搜尋模式
    # patterns = query_engine.search_time_series(
    #     symbols=["BTC"],
    #     pattern="peak",
    #     start_date="2023-01-01",
    #     end_date="2023-12-31"
    # )
