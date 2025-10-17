"""
本地緩存系統
提供高效的數據緩存機制，減少重複計算和數據載入時間
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import pickle
import logging
import hashlib
import time
import threading
from dataclasses import dataclass, asdict
import os
import shutil

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """緩存條目"""
    key: str
    data_hash: str
    file_path: str
    created_at: str
    last_accessed: str
    access_count: int
    size_bytes: int
    ttl_seconds: int
    metadata: Dict[str, Any]

    def is_expired(self) -> bool:
        """檢查是否過期"""
        created_time = datetime.fromisoformat(self.created_at)
        expiry_time = created_time + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time

    def update_access(self):
        """更新訪問統計"""
        self.last_accessed = datetime.now().isoformat()
        self.access_count += 1

class CacheManager:
    """緩存管理器"""

    def __init__(
        self,
        cache_dir: str = "data/storage/cache",
        max_size_gb: float = 1.0,
        default_ttl: int = 3600
    ):
        """
        初始化緩存管理器

        Args:
            cache_dir: 緩存目錄路徑
            max_size_gb: 最大緩存大小（GB）
            default_ttl: 預設緩存時間（秒）
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.default_ttl = default_ttl

        # 創建緩存目錄
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 緩存索引檔案
        self.index_file = self.cache_dir / 'cache_index.json'

        # 載入現有緩存索引
        self.cache_index = self._load_cache_index()

        # 鎖定機制確保線程安全
        self.lock = threading.Lock()

        # 啟動定期清理線程
        self.cleanup_thread = None
        self.start_cleanup_thread()

        logger.info(f"Cache manager initialized: {cache_dir}, max_size={max_size_gb}GB")

    def _load_cache_index(self) -> Dict[str, CacheEntry]:
        """載入緩存索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 將字典轉換為CacheEntry對象
                index = {}
                for key, entry_dict in data.items():
                    index[key] = CacheEntry(**entry_dict)

                logger.info(f"Loaded cache index with {len(index)} entries")
                return index

            except Exception as e:
                logger.error(f"Error loading cache index: {e}")

        return {}

    def _save_cache_index(self):
        """保存緩存索引"""
        try:
            # 將CacheEntry對象轉換為字典
            data = {key: asdict(entry) for key, entry in self.cache_index.items()}

            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving cache index: {e}")

    def _generate_cache_key(self, data: Any, cache_type: str = "data") -> str:
        """生成緩存鍵"""
        # 創建數據內容的哈希值作為鍵
        if isinstance(data, pd.DataFrame):
            # 對於DataFrame，使用內容和欄位名稱生成哈希
            content = str(data.values.tobytes()) + str(sorted(data.columns.tolist()))
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        elif isinstance(data, str):
            content = data
        else:
            content = str(data)

        # 添加類型前綴
        full_content = f"{cache_type}:{content}"
        return hashlib.sha256(full_content.encode()).hexdigest()

    def _get_cache_filepath(self, cache_key: str) -> Path:
        """獲取緩存檔案路徑"""
        # 使用前綴組織檔案結構
        prefix = cache_key[:2]
        cache_subdir = self.cache_dir / prefix
        cache_subdir.mkdir(exist_ok=True)

        return cache_subdir / f"{cache_key}.pkl"

    def _get_current_cache_size(self) -> int:
        """獲取當前緩存總大小"""
        total_size = 0

        for entry in self.cache_index.values():
            if Path(entry.file_path).exists():
                try:
                    total_size += entry.size_bytes
                except Exception:
                    pass

        return total_size

    def _cleanup_expired_entries(self) -> int:
        """清理過期條目"""
        expired_keys = []
        current_time = datetime.now()

        for key, entry in self.cache_index.items():
            if entry.is_expired():
                expired_keys.append(key)

        # 刪除過期檔案和索引
        for key in expired_keys:
            try:
                entry = self.cache_index[key]
                cache_file = Path(entry.file_path)

                if cache_file.exists():
                    cache_file.unlink()
                    logger.debug(f"Deleted expired cache file: {cache_file}")

                del self.cache_index[key]

            except Exception as e:
                logger.error(f"Error deleting expired cache entry {key}: {e}")

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            self._save_cache_index()

        return len(expired_keys)

    def _evict_lru_entries(self, target_bytes: int) -> int:
        """驅逐最少使用的條目（LRU）"""
        if not self.cache_index:
            return 0

        # 按最後訪問時間和訪問次數排序
        entries_by_lru = sorted(
            self.cache_index.values(),
            key=lambda x: (x.last_accessed, x.access_count)
        )

        evicted_count = 0
        freed_bytes = 0

        for entry in entries_by_lru:
            if freed_bytes >= target_bytes:
                break

            try:
                # 刪除檔案
                cache_file = Path(entry.file_path)
                if cache_file.exists():
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    freed_bytes += file_size
                    evicted_count += 1

                # 從索引中移除
                del self.cache_index[entry.key]

            except Exception as e:
                logger.error(f"Error evicting cache entry {entry.key}: {e}")

        if evicted_count > 0:
            logger.info(f"Evicted {evicted_count} LRU cache entries, freed {freed_bytes} bytes")
            self._save_cache_index()

        return evicted_count

    def _ensure_cache_space(self, required_bytes: int):
        """確保有足夠的緩存空間"""
        current_size = self._get_current_cache_size()

        if current_size + required_bytes > self.max_size_bytes:
            # 需要釋放空間
            space_to_free = (current_size + required_bytes) - self.max_size_bytes

            # 先清理過期條目
            freed_by_expiry = 0
            expired_count = self._cleanup_expired_entries()

            if expired_count > 0:
                current_size = self._get_current_cache_size()
                freed_by_expiry = current_size

            # 如果還需要空間，使用LRU驅逐
            remaining_to_free = space_to_free - freed_by_expiry
            if remaining_to_free > 0:
                self._evict_lru_entries(remaining_to_free)

    def set(
        self,
        key: str,
        data: Any,
        ttl_seconds: int = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        設定緩存

        Args:
            key: 緩存鍵
            data: 要緩存的數據
            ttl_seconds: 緩存時間（秒）
            metadata: 元數據

        Returns:
            是否成功設定緩存
        """
        with self.lock:
            try:
                if ttl_seconds is None:
                    ttl_seconds = self.default_ttl

                if metadata is None:
                    metadata = {}

                # 序列化數據以計算大小
                temp_file = self.cache_dir / f"temp_{key}.pkl"
                with open(temp_file, 'wb') as f:
                    pickle.dump(data, f)

                file_size = temp_file.stat().st_size

                # 確保有足夠空間
                self._ensure_cache_space(file_size)

                # 生成最終緩存鍵和檔案路徑
                cache_key = hashlib.sha256(key.encode()).hexdigest()
                cache_filepath = self._get_cache_filepath(cache_key)

                # 移動檔案到最終位置
                shutil.move(str(temp_file), cache_filepath)

                # 創建緩存條目
                cache_entry = CacheEntry(
                    key=cache_key,
                    data_hash=self._generate_cache_key(data),
                    file_path=str(cache_filepath),
                    created_at=datetime.now().isoformat(),
                    last_accessed=datetime.now().isoformat(),
                    access_count=0,
                    size_bytes=file_size,
                    ttl_seconds=ttl_seconds,
                    metadata=metadata
                )

                # 添加到索引
                self.cache_index[cache_key] = cache_entry
                self._save_cache_index()

                logger.debug(f"Cached data with key: {cache_key} ({file_size} bytes)")
                return True

            except Exception as e:
                logger.error(f"Error setting cache for key {key}: {e}")
                # 清理臨時檔案
                if temp_file.exists():
                    temp_file.unlink()
                return False

    def get(self, key: str) -> Any:
        """
        獲取緩存數據

        Args:
            key: 緩存鍵

        Returns:
            緩存的數據，如果不存在或過期則返回None
        """
        with self.lock:
            cache_key = hashlib.sha256(key.encode()).hexdigest()

            # 檢查緩存索引
            if cache_key not in self.cache_index:
                return None

            entry = self.cache_index[cache_key]

            # 檢查是否過期
            if entry.is_expired():
                logger.debug(f"Cache entry expired: {cache_key}")
                self.delete(cache_key)
                return None

            # 檢查檔案是否存在
            cache_file = Path(entry.file_path)
            if not cache_file.exists():
                logger.warning(f"Cache file missing: {cache_file}")
                del self.cache_index[cache_key]
                self._save_cache_index()
                return None

            try:
                # 載入數據
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)

                # 更新訪問統計
                entry.update_access()
                self._save_cache_index()

                logger.debug(f"Cache hit for key: {cache_key}")
                return data

            except Exception as e:
                logger.error(f"Error loading cache for key {cache_key}: {e}")
                self.delete(cache_key)
                return None

    def delete(self, key: str) -> bool:
        """
        刪除緩存條目

        Args:
            key: 緩存鍵

        Returns:
            是否成功刪除
        """
        with self.lock:
            cache_key = hashlib.sha256(key.encode()).hexdigest()

            if cache_key not in self.cache_index:
                return False

            try:
                entry = self.cache_index[cache_key]

                # 刪除檔案
                cache_file = Path(entry.file_path)
                if cache_file.exists():
                    cache_file.unlink()

                # 從索引中移除
                del self.cache_index[cache_key]
                self._save_cache_index()

                logger.debug(f"Deleted cache entry: {cache_key}")
                return True

            except Exception as e:
                logger.error(f"Error deleting cache entry {cache_key}: {e}")
                return False

    def exists(self, key: str) -> bool:
        """
        檢查緩存是否存在且有效

        Args:
            key: 緩存鍵

        Returns:
            是否存在且有效
        """
        with self.lock:
            cache_key = hashlib.sha256(key.encode()).hexdigest()

            if cache_key not in self.cache_index:
                return False

            entry = self.cache_index[cache_key]

            # 檢查是否過期
            if entry.is_expired():
                return False

            # 檢查檔案是否存在
            return Path(entry.file_path).exists()

    def clear(self):
        """清除所有緩存"""
        with self.lock:
            try:
                # 刪除所有緩存檔案
                for entry in self.cache_index.values():
                    cache_file = Path(entry.file_path)
                    if cache_file.exists():
                        cache_file.unlink()

                # 清空索引
                self.cache_index.clear()
                self._save_cache_index()

                logger.info("All cache cleared")

            except Exception as e:
                logger.error(f"Error clearing cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.cache_index.values())
            active_entries = sum(1 for entry in self.cache_index.values() if not entry.is_expired())

            # 計算命中率（基於訪問次數）
            total_accesses = sum(entry.access_count for entry in self.cache_index.values())

            return {
                'total_entries': len(self.cache_index),
                'active_entries': active_entries,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_bytes': self.max_size_bytes,
                'max_size_gb': self.max_size_bytes / (1024 * 1024 * 1024),
                'utilization_pct': (total_size / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0,
                'total_accesses': total_accesses,
                'cache_dir': str(self.cache_dir)
            }

    def start_cleanup_thread(self, interval_seconds: int = 300):
        """啟動定期清理線程"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return

        def cleanup_worker():
            while True:
                try:
                    time.sleep(interval_seconds)
                    self._cleanup_expired_entries()
                except Exception as e:
                    logger.error(f"Error in cache cleanup thread: {e}")

        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info(f"Started cache cleanup thread (interval: {interval_seconds}s)")

    def stop_cleanup_thread(self):
        """停止定期清理線程"""
        if self.cleanup_thread:
            # 線程會自動結束，因為設置為daemon=True
            self.cleanup_thread = None
            logger.info("Stopped cache cleanup thread")

    def export_cache_index(self, filepath: str = None) -> str:
        """導出緩存索引"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.cache_dir / f'cache_index_export_{timestamp}.json'

        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'stats': self.get_stats(),
                'entries': {key: asdict(entry) for key, entry in self.cache_index.items()}
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported cache index to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting cache index: {e}")
            raise

    def import_cache_index(self, filepath: str) -> int:
        """導入緩存索引"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            imported_count = 0
            for key, entry_dict in data.get('entries', {}).items():
                try:
                    entry = CacheEntry(**entry_dict)
                    self.cache_index[key] = entry
                    imported_count += 1
                except Exception as e:
                    logger.error(f"Error importing cache entry {key}: {e}")

            self._save_cache_index()
            logger.info(f"Imported {imported_count} cache entries from {filepath}")
            return imported_count

        except Exception as e:
            logger.error(f"Error importing cache index: {e}")
            return 0


class DataFrameCache:
    """DataFrame專用緩存"""

    def __init__(self, cache_manager: CacheManager):
        """
        初始化DataFrame緩存

        Args:
            cache_manager: 緩存管理器實例
        """
        self.cache_manager = cache_manager

    def cache_dataframe(
        self,
        df: pd.DataFrame,
        cache_key: str,
        ttl_seconds: int = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        緩存DataFrame

        Args:
            df: 要緩存的DataFrame
            cache_key: 緩存鍵
            ttl_seconds: 緩存時間
            metadata: 元數據

        Returns:
            是否成功緩存
        """
        if metadata is None:
            metadata = {}

        # 添加DataFrame特定元數據
        metadata.update({
            'type': 'dataframe',
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage': int(df.memory_usage(deep=True).sum())
        })

        return self.cache_manager.set(cache_key, df, ttl_seconds, metadata)

    def get_dataframe(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        獲取緩存的DataFrame

        Args:
            cache_key: 緩存鍵

        Returns:
            緩存的DataFrame，如果不存在則返回None
        """
        return self.cache_manager.get(cache_key)

    def cache_query_result(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        df: pd.DataFrame,
        ttl_seconds: int = 1800
    ) -> str:
        """
        緩存查詢結果

        Args:
            symbols: 交易對符號列表
            start_date: 開始日期
            end_date: 結束日期
            df: 查詢結果DataFrame
            ttl_seconds: 緩存時間

        Returns:
            緩存鍵
        """
        # 創建查詢特定的緩存鍵
        query_key = f"query_{'_'.join(sorted(symbols))}_{start_date}_{end_date}"
        cache_key = hashlib.sha256(query_key.encode()).hexdigest()

        success = self.cache_dataframe(df, cache_key, ttl_seconds, {
            'query_symbols': symbols,
            'query_start_date': start_date,
            'query_end_date': end_date
        })

        return cache_key if success else None

    def get_query_result(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        獲取緩存的查詢結果

        Args:
            symbols: 交易對符號列表
            start_date: 開始日期
            end_date: 結束日期

        Returns:
            緩存的查詢結果，如果不存在則返回None
        """
        query_key = f"query_{'_'.join(sorted(symbols))}_{start_date}_{end_date}"
        cache_key = hashlib.sha256(query_key.encode()).hexdigest()

        return self.get_dataframe(cache_key)


# 使用範例
if __name__ == "__main__":
    # 創建緩存管理器
    cache_manager = CacheManager(max_size_gb=0.1, default_ttl=3600)  # 100MB緩存

    # 創建範例數據
    sample_df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
        'price': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

    # 緩存數據
    cache_key = "test_dataframe"
    success = cache_manager.set(cache_key, sample_df, ttl_seconds=1800)

    if success:
        print("✅ Data cached successfully")

        # 獲取緩存數據
        cached_df = cache_manager.get(cache_key)
        if cached_df is not None:
            print(f"✅ Retrieved cached data: {len(cached_df)} records")
        else:
            print("❌ Failed to retrieve cached data")

        # 顯示統計
        stats = cache_manager.get_stats()
        print(f"📊 Cache stats: {stats['total_size_mb']:.2f}MB used, "
              f"{stats['active_entries']} active entries")

    # 清理過期條目
    expired_count = cache_manager._cleanup_expired_entries()
    print(f"🧹 Cleaned up {expired_count} expired entries")
