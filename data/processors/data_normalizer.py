"""
數據標準化處理模組
負責將來自不同數據源的數據轉換為統一格式
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import logging
import re

# 設定日誌
logger = logging.getLogger(__name__)

class DataNormalizer:
    """數據標準化處理器"""

    def __init__(self):
        """初始化標準化處理器"""
        self.column_mapping = {
            # 時間戳欄位名稱變體
            'timestamp': ['timestamp', 'time', 'datetime', 'date', 'ts'],
            'open': ['open', 'open_price', 'opening_price'],
            'high': ['high', 'high_price', 'highest_price', 'max_price'],
            'low': ['low', 'low_price', 'lowest_price', 'min_price'],
            'close': ['close', 'close_price', 'closing_price', 'last_price', 'price'],
            'volume': ['volume', 'vol', 'amount', 'quantity', 'size']
        }

        # 支援的時間格式
        self.time_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y%m%d %H:%M:%S',
            '%Y-%m-%d',
            '%Y/%m/%d %H:%M:%S',
        ]

    def normalize_ohlcv_dataframe(
        self,
        df: pd.DataFrame,
        source: str = "unknown"
    ) -> pd.DataFrame:
        """
        標準化OHLCV DataFrame

        Args:
            df: 原始DataFrame
            source: 數據源名稱

        Returns:
            標準化後的DataFrame
        """
        if df.empty:
            logger.warning(f"Empty DataFrame from {source}")
            return self._create_empty_ohlcv_df()

        logger.info(f"Normalizing OHLCV data from {source}: {len(df)} records")

        # 創建標準化副本
        normalized_df = df.copy()

        # 1. 標準化欄位名稱
        normalized_df = self._normalize_column_names(normalized_df)

        # 2. 標準化時間戳
        if 'timestamp' in normalized_df.columns:
            normalized_df = self._normalize_timestamps(normalized_df)

        # 3. 標準化數值欄位
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in normalized_df.columns:
                normalized_df = self._normalize_numeric_column(normalized_df, col)

        # 4. 添加元數據欄位
        normalized_df = self._add_metadata_columns(normalized_df, source)

        # 5. 最終驗證和清理
        normalized_df = self._finalize_normalization(normalized_df)

        logger.info(f"Normalization complete for {source}: {len(normalized_df)} records")
        return normalized_df

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準化欄位名稱"""
        normalized_df = df.copy()
        column_rename_map = {}

        # 為每個標準欄位尋找匹配的原始欄位
        for standard_col, variants in self.column_mapping.items():
            for variant in variants:
                if variant in normalized_df.columns and variant != standard_col:
                    column_rename_map[variant] = standard_col

        # 執行欄位名稱重新命名
        if column_rename_map:
            normalized_df = normalized_df.rename(columns=column_rename_map)
            logger.info(f"Renamed columns: {column_rename_map}")

        return normalized_df

    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準化時間戳格式"""
        normalized_df = df.copy()

        # 如果已經是datetime類型，檢查時區
        if pd.api.types.is_datetime64_any_dtype(normalized_df['timestamp']):
            # 確保是UTC時區
            if normalized_df['timestamp'].dt.tz is None:
                # 假設是UTC時間
                normalized_df['timestamp'] = normalized_df['timestamp'].dt.tz_localize('UTC')
        else:
            # 需要轉換為datetime
            normalized_df = self._convert_to_datetime(normalized_df)

        # 確保時間戳按升序排列
        normalized_df = normalized_df.sort_values('timestamp').reset_index(drop=True)

        return normalized_df

    def _convert_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """將時間戳欄位轉換為datetime格式"""
        normalized_df = df.copy()

        # 嘗試不同的轉換方法
        for col in ['timestamp']:
            if col not in normalized_df.columns:
                continue

            original_col = normalized_df[col].copy()

            # 方法1：直接轉換（pandas自動推斷）
            try:
                converted = pd.to_datetime(normalized_df[col], infer_datetime_format=True)
                if converted is not None:
                    normalized_df[col] = converted
                    logger.info(f"Successfully converted {col} using automatic inference")
                    continue
            except Exception as e:
                logger.debug(f"Automatic conversion failed for {col}: {e}")

            # 方法2：嘗試指定的時間格式
            for time_format in self.time_formats:
                try:
                    converted = pd.to_datetime(normalized_df[col], format=time_format)
                    if converted is not None:
                        normalized_df[col] = converted
                        logger.info(f"Successfully converted {col} using format: {time_format}")
                        break
                except Exception as e:
                    logger.debug(f"Format {time_format} failed for {col}: {e}")
                    continue

            # 方法3：處理Unix時間戳（毫秒或秒）
            if normalized_df[col].dtype in ['int64', 'float64']:
                try:
                    # 嘗試毫秒時間戳
                    if normalized_df[col].max() > 1e10:  # 毫秒時間戳通常很大
                        converted = pd.to_datetime(normalized_df[col], unit='ms')
                    else:  # 秒時間戳
                        converted = pd.to_datetime(normalized_df[col], unit='s')

                    normalized_df[col] = converted
                    logger.info(f"Successfully converted {col} from Unix timestamp")
                except Exception as e:
                    logger.debug(f"Unix timestamp conversion failed for {col}: {e}")

            # 如果所有方法都失敗，記錄錯誤但保留原始數據
            if not pd.api.types.is_datetime64_any_dtype(normalized_df[col]):
                logger.error(f"Failed to convert {col} to datetime, keeping original format")
                # 嘗試轉換為字符串以避免後續錯誤
                normalized_df[col] = normalized_df[col].astype(str)

        return normalized_df

    def _normalize_numeric_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """標準化數值欄位"""
        normalized_df = df.copy()

        if column not in normalized_df.columns:
            return normalized_df

        # 轉換為數值類型
        try:
            normalized_df[column] = pd.to_numeric(normalized_df[column], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting {column} to numeric: {e}")
            return normalized_df

        # 處理極端值（無窮大、無窮小）
        if np.inf in normalized_df[column].values or -np.inf in normalized_df[column].values:
            logger.warning(f"Found infinite values in {column}, replacing with NaN")
            normalized_df[column] = normalized_df[column].replace([np.inf, -np.inf], np.nan)

        # 處理負數值（對於價格和交易量）
        if column in ['open', 'high', 'low', 'close']:
            negative_mask = normalized_df[column] < 0
            if negative_mask.any():
                logger.warning(f"Found negative prices in {column}, replacing with NaN")
                normalized_df.loc[negative_mask, column] = np.nan

        elif column == 'volume':
            negative_mask = normalized_df[column] < 0
            if negative_mask.any():
                logger.warning(f"Found negative volume in {column}, replacing with NaN")
                normalized_df.loc[negative_mask, column] = np.nan

        return normalized_df

    def _add_metadata_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """添加元數據欄位"""
        normalized_df = df.copy()

        # 添加數據源信息
        normalized_df['data_source'] = source

        # 添加標準化時間戳
        normalized_df['normalized_at'] = pd.Timestamp.now(tz='UTC')

        # 添加數據品質標記（簡化版本）
        normalized_df['is_normalized'] = True

        return normalized_df

    def _finalize_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """最終驗證和清理"""
        normalized_df = df.copy()

        # 確保必要的欄位存在
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in normalized_df.columns]

        if missing_columns:
            logger.error(f"Missing required columns after normalization: {missing_columns}")
            # 為缺失欄位創建NaN欄位
            for col in missing_columns:
                normalized_df[col] = np.nan

        # 移除包含太多NaN的行（超過50%欄位為NaN）
        nan_threshold = len(normalized_df.columns) * 0.5
        original_len = len(normalized_df)

        normalized_df = normalized_df.dropna(thresh=nan_threshold)

        if len(normalized_df) < original_len:
            removed_rows = original_len - len(normalized_df)
            logger.info(f"Removed {removed_rows} rows with too many NaN values")

        # 最終排序和重設索引
        if 'timestamp' in normalized_df.columns:
            normalized_df = normalized_df.sort_values('timestamp').reset_index(drop=True)

        return normalized_df

    def _create_empty_ohlcv_df(self) -> pd.DataFrame:
        """創建空的標準化OHLCV DataFrame"""
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'data_source', 'normalized_at', 'is_normalized'
        ]
        return pd.DataFrame(columns=columns)

    def merge_multiple_sources(
        self,
        dataframes: List[pd.DataFrame],
        sources: List[str] = None,
        merge_strategy: str = "combine"
    ) -> pd.DataFrame:
        """
        合併來自多個數據源的數據

        Args:
            dataframes: DataFrame列表
            sources: 數據源名稱列表
            merge_strategy: 合併策略（combine, prioritize, validate）

        Returns:
            合併後的DataFrame
        """
        if not dataframes:
            return self._create_empty_ohlcv_df()

        if len(dataframes) == 1:
            return dataframes[0]

        logger.info(f"Merging {len(dataframes)} data sources with strategy: {merge_strategy}")

        if merge_strategy == "combine":
            return self._merge_combine_strategy(dataframes, sources)
        elif merge_strategy == "prioritize":
            return self._merge_prioritize_strategy(dataframes, sources)
        elif merge_strategy == "validate":
            return self._merge_validate_strategy(dataframes, sources)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    def _merge_combine_strategy(
        self,
        dataframes: List[pd.DataFrame],
        sources: List[str]
    ) -> pd.DataFrame:
        """合併策略：簡單合併所有數據"""
        merged_df = pd.concat(dataframes, ignore_index=True)

        # 移除重複記錄（基於時間戳）
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=['timestamp'])
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

        return merged_df

    def _merge_prioritize_strategy(
        self,
        dataframes: List[pd.DataFrame],
        sources: List[str]
    ) -> pd.DataFrame:
        """優先策略：按數據源優先級合併"""
        if not sources:
            sources = [f"source_{i}" for i in range(len(dataframes))]

        # 定義優先級（CoinGecko優先於CryptoCompare）
        priority_order = ['coingecko', 'cryptocompare', 'unknown']
        prioritized_dfs = sorted(
            zip(dataframes, sources),
            key=lambda x: priority_order.index(x[1].lower() if x[1] else 'unknown')
        )

        result_df = prioritized_dfs[0][0]  # 最高優先級的數據

        for df, source in prioritized_dfs[1:]:
            # 合併時保留優先數據源的值
            combined = result_df.combine_first(df)
            result_df = combined

        return result_df

    def _merge_validate_strategy(
        self,
        dataframes: List[pd.DataFrame],
        sources: List[str]
    ) -> pd.DataFrame:
        """驗證策略：比較數據並選擇最可靠的值"""
        if len(dataframes) != 2:
            raise ValueError("Validate strategy requires exactly 2 dataframes")

        df1, df2 = dataframes
        source1, source2 = sources if sources else ['source1', 'source2']

        # 合併數據進行比較
        merged = pd.merge(df1, df2, on='timestamp', how='outer', suffixes=('_1', '_2'))

        # 選擇邏輯：如果兩個源都有數據，選擇絕對差異較小的
        for col in ['open', 'high', 'low', 'close', 'volume']:
            col1 = f'{col}_1'
            col2 = f'{col}_2'

            if col1 in merged.columns and col2 in merged.columns:
                # 計算價格差異百分比
                mask = merged[col1].notna() & merged[col2].notna()
                if mask.any():
                    diff_pct = abs(merged.loc[mask, col1] - merged.loc[mask, col2]) / merged.loc[mask, col1]

                    # 如果差異太大（>1%），記錄警告
                    large_diff = diff_pct > 0.01
                    if large_diff.any():
                        logger.warning(f"Large price differences found between {source1} and {source2}")

                # 選擇策略：優先選擇非NaN值，如果都有值則選擇平均值
                merged[col] = np.where(
                    merged[col1].notna() & merged[col2].notna(),
                    (merged[col1] + merged[col2]) / 2,
                    merged[col1].fillna(merged[col2])
                )

        # 清理臨時欄位
        cols_to_drop = [col for col in merged.columns if col.endswith('_1') or col.endswith('_2')]
        merged = merged.drop(columns=cols_to_drop)

        return merged

    def detect_timeframe(self, df: pd.DataFrame) -> str:
        """
        自動檢測數據時間週期

        Args:
            df: 數據DataFrame

        Returns:
            檢測到的時間週期（1m, 5m, 15m, 1h, 4h, 1d等）
        """
        if 'timestamp' not in df.columns or len(df) < 2:
            return "unknown"

        try:
            time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()

            if time_diffs.empty:
                return "unknown"

            # 計算最常見的時間間隔（秒）
            mode_seconds = time_diffs.mode().iloc[0]

            # 轉換為標準時間週期格式
            timeframe_map = {
                60: "1m",
                300: "5m",
                900: "15m",
                1800: "30m",
                3600: "1h",
                7200: "2h",
                14400: "4h",
                21600: "6h",
                43200: "12h",
                86400: "1d",
                604800: "1w"
            }

            # 找到最接近的時間週期
            closest_timeframe = "1h"  # 默認值
            min_diff = float('inf')

            for seconds, timeframe in timeframe_map.items():
                diff = abs(mode_seconds - seconds)
                if diff < min_diff:
                    min_diff = diff
                    closest_timeframe = timeframe

            logger.info(f"Detected timeframe: {closest_timeframe} (mode: {mode_seconds}s)")
            return closest_timeframe

        except Exception as e:
            logger.error(f"Error detecting timeframe: {e}")
            return "unknown"

    def resample_to_timeframe(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
        aggregation_method: str = "ohlc"
    ) -> pd.DataFrame:
        """
        重新採樣到目標時間週期

        Args:
            df: 原始數據
            target_timeframe: 目標時間週期（1m, 5m, 1h等）
            aggregation_method: 聚合方法（ohlc, mean, sum等）

        Returns:
            重新採樣後的DataFrame
        """
        if df.empty or 'timestamp' not in df.columns:
            return df

        try:
            # 設定時間頻率映射
            freq_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '1H',
                '2h': '2H',
                '4h': '4H',
                '6h': '6H',
                '12h': '12H',
                '1d': '1D',
                '1w': '1W'
            }

            freq = freq_map.get(target_timeframe)
            if not freq:
                logger.error(f"Unsupported timeframe: {target_timeframe}")
                return df

            # 設定為時間序列
            df_timeindexed = df.set_index('timestamp')

            if aggregation_method == "ohlc":
                # OHLC聚合
                resampled = df_timeindexed.resample(freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
            elif aggregation_method == "mean":
                # 均值聚合
                resampled = df_timeindexed.resample(freq).mean()
            elif aggregation_method == "sum":
                # 總和聚合
                resampled = df_timeindexed.resample(freq).sum()
            else:
                logger.error(f"Unsupported aggregation method: {aggregation_method}")
                return df

            # 重設索引並清理
            resampled = resampled.reset_index()
            resampled = resampled.dropna()

            logger.info(f"Resampled data to {target_timeframe}: {len(resampled)} records")
            return resampled

        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return df


# 使用範例
if __name__ == "__main__":
    # 創建標準化處理器
    normalizer = DataNormalizer()

    # 創建測試數據（模擬來自不同源的數據）
    test_data = {
        'time': pd.date_range('2023-01-01', periods=5, freq='1H'),
        'open_price': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [99, 100, 101, 102, 103],
        'close': [104, 105, 106, 107, 108],
        'vol': [1000, 1100, 1200, 1300, 1400]
    }

    df = pd.DataFrame(test_data)
    print("原始數據:")
    print(df.head())
    print(f"欄位名稱: {list(df.columns)}")

    # 標準化處理
    normalized_df = normalizer.normalize_ohlcv_dataframe(df, "test_source")
    print("\n標準化後數據:")
    print(normalized_df.head())
    print(f"欄位名稱: {list(normalized_df.columns)}")

    # 檢測時間週期
    timeframe = normalizer.detect_timeframe(normalized_df)
    print(f"\n檢測到的時間週期: {timeframe}")
