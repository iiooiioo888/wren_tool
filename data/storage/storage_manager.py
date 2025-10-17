"""
數據存儲管理器
負責設計和實現分層存儲架構，支援原始數據、處理數據和特徵數據的存儲
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import pickle
import logging
from dataclasses import dataclass, asdict
import hashlib
import os

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class StorageMetadata:
    """存儲元數據"""
    source: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    record_count: int
    file_size: int  # bytes
    created_at: str
    checksum: str
    compression: str = "none"
    version: str = "1.0"

@dataclass
class DataQuery:
    """數據查詢參數"""
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: str = "1h"
    sources: Optional[List[str]] = None
    columns: Optional[List[str]] = None

class StorageManager:
    """數據存儲管理器"""

    def __init__(self, base_path: str = "data/storage"):
        """
        初始化存儲管理器

        Args:
            base_path: 存儲基礎路徑
        """
        self.base_path = Path(base_path)
        self.layers = {
            'raw': self.base_path / 'raw',
            'processed': self.base_path / 'processed',
            'features': self.base_path / 'features',
            'cache': self.base_path / 'cache',
            'metadata': self.base_path / 'metadata'
        }

        # 創建目錄結構
        self._create_directories()

        # 載入現有的元數據索引
        self.metadata_index = self._load_metadata_index()

    def _create_directories(self):
        """創建目錄結構"""
        for layer_path in self.layers.values():
            layer_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {layer_path}")

    def _load_metadata_index(self) -> Dict[str, StorageMetadata]:
        """載入元數據索引"""
        index_file = self.layers['metadata'] / 'index.json'
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 將字典轉換回StorageMetadata對象
                    index = {}
                    for key, metadata_dict in data.items():
                        index[key] = StorageMetadata(**metadata_dict)
                    logger.info(f"Loaded metadata index with {len(index)} entries")
                    return index
            except Exception as e:
                logger.error(f"Error loading metadata index: {e}")

        return {}

    def _save_metadata_index(self):
        """保存元數據索引"""
        index_file = self.layers['metadata'] / 'index.json'
        try:
            # 將StorageMetadata對象轉換為字典
            data = {key: asdict(metadata) for key, metadata in self.metadata_index.items()}
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved metadata index with {len(self.metadata_index)} entries")
        except Exception as e:
            logger.error(f"Error saving metadata index: {e}")

    def _generate_file_key(self, source: str, symbol: str, timeframe: str, date: str) -> str:
        """生成檔案鍵值"""
        # 創建唯一鍵值用於識別檔案
        key_data = f"{source}_{symbol}_{timeframe}_{date}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]

    def _calculate_checksum(self, data: Union[pd.DataFrame, Dict, str]) -> str:
        """計算數據校驗和"""
        if isinstance(data, pd.DataFrame):
            # 對於DataFrame，使用內容摘要
            content = str(data.values.tobytes()) + str(data.columns.tolist())
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)

        return hashlib.sha256(content.encode()).hexdigest()

    def store_raw_data(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        存儲原始數據

        Args:
            df: 原始數據DataFrame
            source: 數據源名稱
            symbol: 交易對符號
            metadata: 額外元數據

        Returns:
            檔案ID
        """
        if df.empty:
            raise ValueError("Cannot store empty DataFrame")

        # 生成檔案信息
        start_date = df['timestamp'].min().strftime('%Y-%m-%d')
        end_date = df['timestamp'].max().strftime('%Y-%m-%d')
        timeframe = "1h"  # 預設時間週期
        file_key = self._generate_file_key(source, symbol, timeframe, start_date)

        # 創建檔案路徑
        filename = f"{source}_{symbol}_{start_date}_{file_key}.csv"
        filepath = self.layers['raw'] / filename

        # 保存數據
        df.to_csv(filepath, index=False)

        # 創建元數據
        file_size = filepath.stat().st_size
        checksum = self._calculate_checksum(df)

        storage_metadata = StorageMetadata(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            record_count=len(df),
            file_size=file_size,
            created_at=datetime.now().isoformat(),
            checksum=checksum
        )

        # 添加到索引
        index_key = f"raw_{source}_{symbol}_{file_key}"
        self.metadata_index[index_key] = storage_metadata

        # 保存元數據索引
        self._save_metadata_index()

        logger.info(f"Stored raw data: {filepath} ({len(df)} records)")
        return index_key

    def store_processed_data(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        processing_type: str = "normalized",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        存儲處理後的數據

        Args:
            df: 處理後的數據DataFrame
            source: 數據源名稱
            symbol: 交易對符號
            processing_type: 處理類型
            metadata: 額外元數據

        Returns:
            檔案ID
        """
        if df.empty:
            raise ValueError("Cannot store empty DataFrame")

        # 生成檔案信息
        start_date = df['timestamp'].min().strftime('%Y-%m-%d')
        end_date = df['timestamp'].max().strftime('%Y-%m-%d')
        timeframe = "1h"
        file_key = self._generate_file_key(source, symbol, timeframe, start_date)

        # 創建檔案路徑
        filename = f"{processing_type}_{source}_{symbol}_{start_date}_{file_key}.parquet"
        filepath = self.layers['processed'] / filename

        # 保存為Parquet格式（更高效）
        df.to_parquet(filepath, index=False)

        # 創建元數據
        file_size = filepath.stat().st_size
        checksum = self._calculate_checksum(df)

        storage_metadata = StorageMetadata(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            record_count=len(df),
            file_size=file_size,
            created_at=datetime.now().isoformat(),
            checksum=checksum,
            compression="parquet"
        )

        # 添加到索引
        index_key = f"processed_{processing_type}_{source}_{symbol}_{file_key}"
        self.metadata_index[index_key] = storage_metadata

        # 保存元數據索引
        self._save_metadata_index()

        logger.info(f"Stored processed data: {filepath} ({len(df)} records)")
        return index_key

    def store_feature_data(
        self,
        features: Dict[str, Any],
        source: str,
        symbol: str,
        feature_type: str = "technical",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        存儲特徵數據

        Args:
            features: 特徵數據字典
            source: 數據源名稱
            symbol: 交易對符號
            feature_type: 特徵類型
            metadata: 額外元數據

        Returns:
            檔案ID
        """
        # 生成檔案信息
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_key = self._generate_file_key(source, symbol, feature_type, timestamp)

        # 創建檔案路徑
        filename = f"{feature_type}_{source}_{symbol}_{timestamp}_{file_key}.pkl"
        filepath = self.layers['features'] / filename

        # 保存特徵數據
        with open(filepath, 'wb') as f:
            pickle.dump(features, f)

        # 創建元數據
        file_size = filepath.stat().st_size
        checksum = self._calculate_checksum(features)

        storage_metadata = StorageMetadata(
            source=source,
            symbol=symbol,
            timeframe=feature_type,
            start_date=timestamp,
            end_date=timestamp,
            record_count=len(features) if isinstance(features, dict) else 1,
            file_size=file_size,
            created_at=datetime.now().isoformat(),
            checksum=checksum,
            compression="pickle"
        )

        # 添加到索引
        index_key = f"features_{feature_type}_{source}_{symbol}_{file_key}"
        self.metadata_index[index_key] = storage_metadata

        # 保存元數據索引
        self._save_metadata_index()

        logger.info(f"Stored feature data: {filepath}")
        return index_key

    def query_data(self, query: DataQuery) -> pd.DataFrame:
        """
        查詢數據

        Args:
            query: 查詢參數

        Returns:
            查詢結果DataFrame
        """
        logger.info(f"Querying data: symbols={query.symbols}, timeframe={query.timeframe}")

        all_data = []

        for symbol in query.symbols:
            # 尋找相關的元數據
            relevant_metadata = self._find_relevant_metadata(symbol, query)

            for metadata in relevant_metadata:
                try:
                    # 載入數據
                    if metadata.source.startswith('raw'):
                        df = self._load_raw_data(metadata)
                    elif metadata.source.startswith('processed'):
                        df = self._load_processed_data(metadata)
                    else:
                        continue

                    # 應用時間範圍過濾
                    if query.start_date:
                        df = df[df['timestamp'] >= query.start_date]
                    if query.end_date:
                        df = df[df['timestamp'] <= query.end_date]

                    # 應用欄位過濾
                    if query.columns:
                        available_columns = [col for col in query.columns if col in df.columns]
                        df = df[['timestamp'] + available_columns]

                    if not df.empty:
                        df['symbol'] = symbol  # 添加符號欄位
                        all_data.append(df)

                except Exception as e:
                    logger.error(f"Error loading data for {symbol} from {metadata.source}: {e}")
                    continue

        if all_data:
            # 合併所有數據
            result_df = pd.concat(all_data, ignore_index=True)
            result_df = result_df.sort_values('timestamp').reset_index(drop=True)
            logger.info(f"Query returned {len(result_df)} records")
            return result_df
        else:
            logger.warning("No data found for query")
            return pd.DataFrame()

    def _find_relevant_metadata(
        self,
        symbol: str,
        query: DataQuery
    ) -> List[StorageMetadata]:
        """尋找相關的元數據"""
        relevant = []

        for key, metadata in self.metadata_index.items():
            # 檢查符號匹配
            if metadata.symbol != symbol:
                continue

            # 檢查數據源過濾
            if query.sources and metadata.source not in query.sources:
                continue

            # 檢查時間範圍重疊（簡化檢查）
            # 這裡可以實現更複雜的時間範圍檢查邏輯

            relevant.append(metadata)

        return relevant

    def _load_raw_data(self, metadata: StorageMetadata) -> pd.DataFrame:
        """載入原始數據"""
        # 根據元數據中的信息構造檔案路徑
        filename = f"{metadata.source}_{metadata.symbol}_{metadata.start_date}_{metadata.checksum[:12]}.csv"
        filepath = self.layers['raw'] / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")

        df = pd.read_csv(filepath)
        return df

    def _load_processed_data(self, metadata: StorageMetadata) -> pd.DataFrame:
        """載入處理後的數據"""
        # 根據元數據中的信息構造檔案路徑
        filename = f"{metadata.source}_{metadata.symbol}_{metadata.start_date}_{metadata.checksum[:12]}.parquet"
        filepath = self.layers['processed'] / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Processed data file not found: {filepath}")

        df = pd.read_parquet(filepath)
        return df

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        獲取存儲統計信息

        Returns:
            存儲統計字典
        """
        stats = {
            'total_files': 0,
            'total_size': 0,
            'layers': {},
            'sources': {},
            'symbols': set(),
            'timeframes': set()
        }

        for layer_name, layer_path in self.layers.items():
            if layer_path.exists():
                files = list(layer_path.glob('*'))
                total_size = sum(f.stat().st_size for f in files if f.is_file())

                stats['layers'][layer_name] = {
                    'file_count': len(files),
                    'total_size': total_size,
                    'files': [f.name for f in files[:10]]  # 只顯示前10個檔案
                }

                stats['total_files'] += len(files)
                stats['total_size'] += total_size

        # 從元數據收集統計
        for metadata in self.metadata_index.values():
            if metadata.source not in stats['sources']:
                stats['sources'][metadata.source] = 0
            stats['sources'][metadata.source] += 1
            stats['symbols'].add(metadata.symbol)
            stats['timeframes'].add(metadata.timeframe)

        stats['symbols'] = list(stats['symbols'])
        stats['timeframes'] = list(stats['timeframes'])

        return stats

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        清理舊數據

        Args:
            days_to_keep: 保留天數

        Returns:
            刪除的檔案數量
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0

        # 清理暫存檔案
        cache_files = list(self.layers['cache'].glob('*'))
        for file_path in cache_files:
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting cache file {file_path}: {e}")

        logger.info(f"Cleaned up {deleted_count} old cache files")
        return deleted_count

    def verify_data_integrity(self) -> Dict[str, bool]:
        """
        驗證數據完整性

        Returns:
            每個檔案的完整性檢查結果
        """
        results = {}

        for key, metadata in self.metadata_index.items():
            try:
                # 重新計算校驗和
                if metadata.source.startswith('raw'):
                    df = self._load_raw_data(metadata)
                    current_checksum = self._calculate_checksum(df)
                elif metadata.source.startswith('processed'):
                    df = self._load_processed_data(metadata)
                    current_checksum = self._calculate_checksum(df)
                else:
                    # 特徵數據
                    continue

                # 比較校驗和
                is_valid = current_checksum == metadata.checksum
                results[key] = is_valid

                if not is_valid:
                    logger.warning(f"Checksum mismatch for {key}")

            except Exception as e:
                logger.error(f"Error verifying integrity for {key}: {e}")
                results[key] = False

        return results

    def export_metadata(self, filepath: str = None) -> str:
        """
        導出元數據

        Args:
            filepath: 導出檔案路徑（如果為None，則生成預設路徑）

        Returns:
            導出檔案路徑
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.layers['metadata'] / f'metadata_export_{timestamp}.json'

        try:
            # 將元數據轉換為可序列化格式
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_entries': len(self.metadata_index),
                'metadata': {key: asdict(metadata) for key, metadata in self.metadata_index.items()}
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported metadata to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting metadata: {e}")
            raise

    def list_available_data(self) -> pd.DataFrame:
        """
        列出所有可用數據

        Returns:
            包含所有可用數據信息的DataFrame
        """
        if not self.metadata_index:
            return pd.DataFrame()

        # 將元數據轉換為DataFrame
        metadata_list = []
        for key, metadata in self.metadata_index.items():
            metadata_dict = asdict(metadata)
            metadata_dict['index_key'] = key
            metadata_list.append(metadata_dict)

        df = pd.DataFrame(metadata_list)

        # 重新排列欄位順序
        columns = ['index_key', 'source', 'symbol', 'timeframe', 'start_date', 'end_date',
                  'record_count', 'file_size', 'created_at', 'checksum']

        available_columns = [col for col in columns if col in df.columns]
        return df[available_columns].sort_values('created_at', ascending=False)


# 使用範例
if __name__ == "__main__":
    # 創建存儲管理器
    manager = StorageManager()

    # 創建範例數據
    sample_data = {
        'timestamp': pd.date_range('2023-01-01', periods=10, freq='1H'),
        'open': [100 + i for i in range(10)],
        'high': [105 + i for i in range(10)],
        'low': [99 + i for i in range(10)],
        'close': [104 + i for i in range(10)],
        'volume': [1000 + i * 100 for i in range(10)]
    }

    df = pd.DataFrame(sample_data)

    # 存儲原始數據
    raw_id = manager.store_raw_data(df, "test_source", "BTC")
    print(f"Stored raw data with ID: {raw_id}")

    # 存儲處理後數據
    processed_id = manager.store_processed_data(df, "test_source", "BTC", "normalized")
    print(f"Stored processed data with ID: {processed_id}")

    # 查詢數據
    query = DataQuery(
        symbols=["BTC"],
        start_date="2023-01-01",
        end_date="2023-01-02"
    )

    result = manager.query_data(query)
    print(f"Query returned {len(result)} records")

    # 顯示存儲統計
    stats = manager.get_storage_stats()
    print(f"\nStorage stats: {stats['total_files']} files, {stats['total_size']} bytes")

    # 列出可用數據
    available = manager.list_available_data()
    print(f"\nAvailable data:\n{available}")
