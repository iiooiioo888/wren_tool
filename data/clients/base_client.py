"""
數據獲取接口基類
定義統一的數據獲取接口，支援多個數據源的切換
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import pandas as pd
from datetime import datetime
import logging

# 設定日誌
logger = logging.getLogger(__name__)

class BaseDataClient(ABC):
    """數據客戶端基類"""

    def __init__(self, name: str):
        """
        初始化基類

        Args:
            name: 客戶端名稱
        """
        self.name = name
        self.is_connected = False

    @abstractmethod
    def test_connection(self) -> bool:
        """
        測試與數據源的連接

        Returns:
            連接是否成功
        """
        pass

    @abstractmethod
    def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        **kwargs
    ) -> pd.DataFrame:
        """
        獲取歷史OHLCV數據

        Args:
            symbol: 交易對符號（例如：BTC, ETH）
            timeframe: 時間週期（1m, 5m, 1h, 1d等）
            limit: 數據點數量
            **kwargs: 其他參數

        Returns:
            標準化的OHLCV DataFrame
        """
        pass

    @abstractmethod
    def get_multiple_symbols_history(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        start_date: str = None,
        end_date: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        獲取多個交易對的歷史數據

        Args:
            symbols: 交易對符號列表
            timeframe: 時間週期
            start_date: 開始日期（格式：YYYY-MM-DD）
            end_date: 結束日期（格式：YYYY-MM-DD）
            **kwargs: 其他參數

        Returns:
            合併後的OHLCV DataFrame
        """
        pass

    def get_available_symbols(self) -> List[str]:
        """
        獲取可用交易對列表

        Returns:
            交易對符號列表
        """
        try:
            return self._get_available_symbols()
        except Exception as e:
            logger.error(f"Error getting available symbols from {self.name}: {e}")
            return []

    @abstractmethod
    def _get_available_symbols(self) -> List[str]:
        """
        實作獲取可用交易對列表（子類實作）

        Returns:
            交易對符號列表
        """
        pass

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        獲取交易對信息

        Args:
            symbol: 交易對符號

        Returns:
            交易對詳細信息
        """
        try:
            return self._get_symbol_info(symbol)
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol} from {self.name}: {e}")
            return None

    @abstractmethod
    def _get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        實作獲取交易對信息（子類實作）

        Args:
            symbol: 交易對符號

        Returns:
            交易對詳細信息
        """
        pass

    def normalize_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        標準化OHLCV數據格式

        Args:
            df: 原始數據DataFrame

        Returns:
            標準化後的DataFrame，包含以下欄位：
            - timestamp: 時間戳（datetime）
            - open: 開盤價（float）
            - high: 最高價（float）
            - low: 最低價（float）
            - close: 收盤價（float）
            - volume: 交易量（float）
            - symbol: 交易對符號（str，可選）
        """
        if df.empty:
            return df

        # 確保必要的欄位存在
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        # 標準化時間戳格式
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                logger.error(f"Error converting timestamp column: {e}")
                raise

        # 確保數值欄位類型正確
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 移除包含NaN的行
        df = df.dropna(subset=numeric_columns)

        # 排序並重設索引
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Normalized OHLCV data: {len(df)} records")
        return df

    def validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """
        驗證OHLCV數據的有效性

        Args:
            df: 要驗證的DataFrame

        Returns:
            數據是否有效
        """
        if df.empty:
            logger.warning("DataFrame is empty")
            return False

        # 檢查必要欄位
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # 檢查價格邏輯關係
        price_issues = 0
        if 'high' in df.columns and 'low' in df.columns:
            price_issues += (df['high'] < df['low']).sum()

        if 'high' in df.columns and 'close' in df.columns:
            price_issues += (df['high'] < df['close']).sum()

        if 'low' in df.columns and 'close' in df.columns:
            price_issues += (df['low'] > df['close']).sum()

        if price_issues > 0:
            logger.warning(f"Found {price_issues} price logic issues")

        # 檢查負數值
        negative_values = 0
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                negative_values += negative_count
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}")

        # 檢查時間戳順序
        if len(df) > 1:
            timestamp_issues = (df['timestamp'].diff().dt.total_seconds() < 0).sum()
            if timestamp_issues > 0:
                logger.warning(f"Found {timestamp_issues} timestamp order issues")

        # 數據被視為有效，除非有嚴重問題
        is_valid = price_issues == 0 and negative_values == 0
        logger.info(f"Data validation result: {'Valid' if is_valid else 'Invalid'} "
                   f"(Price issues: {price_issues}, Negative values: {negative_values})")

        return is_valid

    def __str__(self) -> str:
        """字串表示"""
        return f"{self.name}Client(connected={self.is_connected})"

    def __repr__(self) -> str:
        """詳細字串表示"""
        return f"{self.__class__.__name__}(name='{self.name}', connected={self.is_connected})"


class DataClientManager:
    """數據客戶端管理器"""

    def __init__(self):
        """初始化管理器"""
        self.clients: Dict[str, BaseDataClient] = {}
        self.active_client: Optional[str] = None

    def register_client(self, client: BaseDataClient) -> None:
        """
        註冊數據客戶端

        Args:
            client: 要註冊的客戶端實例
        """
        self.clients[client.name.lower()] = client
        logger.info(f"Registered data client: {client.name}")

    def set_active_client(self, client_name: str) -> bool:
        """
        設定活躍客戶端

        Args:
            client_name: 客戶端名稱

        Returns:
            設定是否成功
        """
        if client_name.lower() in self.clients:
            self.active_client = client_name.lower()
            logger.info(f"Set active client to: {client_name}")
            return True
        else:
            logger.error(f"Client '{client_name}' not found")
            return False

    def get_active_client(self) -> Optional[BaseDataClient]:
        """
        獲取活躍客戶端

        Returns:
            活躍客戶端實例，如果沒有設定則返回None
        """
        if self.active_client and self.active_client in self.clients:
            return self.clients[self.active_client]
        return None

    def get_client(self, client_name: str) -> Optional[BaseDataClient]:
        """
        獲取指定客戶端

        Args:
            client_name: 客戶端名稱

        Returns:
            客戶端實例，如果不存在則返回None
        """
        return self.clients.get(client_name.lower())

    def list_clients(self) -> List[str]:
        """
        列出所有註冊的客戶端

        Returns:
            客戶端名稱列表
        """
        return list(self.clients.keys())

    def test_all_connections(self) -> Dict[str, bool]:
        """
        測試所有客戶端的連接

        Returns:
            每個客戶端的連接測試結果
        """
        results = {}
        for name, client in self.clients.items():
            try:
                results[name] = client.test_connection()
            except Exception as e:
                logger.error(f"Error testing connection for {name}: {e}")
                results[name] = False
        return results

    def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        client_name: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        通過指定客戶端獲取歷史數據

        Args:
            symbol: 交易對符號
            timeframe: 時間週期
            limit: 數據點數量
            client_name: 客戶端名稱（如果不指定，使用活躍客戶端）
            **kwargs: 其他參數

        Returns:
            OHLCV DataFrame
        """
        client = self.get_client(client_name) if client_name else self.get_active_client()

        if not client:
            raise ValueError("No active client available")

        try:
            df = client.get_historical_ohlcv(symbol, timeframe, limit, **kwargs)
            df = client.normalize_ohlcv_data(df)

            if not client.validate_ohlcv_data(df):
                logger.warning(f"Data validation failed for {symbol} from {client.name}")

            return df

        except Exception as e:
            logger.error(f"Error getting data from {client.name}: {e}")
            raise

    def get_multiple_symbols_history(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        start_date: str = None,
        end_date: str = None,
        client_name: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        通過指定客戶端獲取多個交易對的歷史數據

        Args:
            symbols: 交易對符號列表
            timeframe: 時間週期
            start_date: 開始日期
            end_date: 結束日期
            client_name: 客戶端名稱
            **kwargs: 其他參數

        Returns:
            合併後的OHLCV DataFrame
        """
        client = self.get_client(client_name) if client_name else self.get_active_client()

        if not client:
            raise ValueError("No active client available")

        try:
            df = client.get_multiple_symbols_history(
                symbols, timeframe, start_date, end_date, **kwargs
            )

            if not df.empty:
                df = client.normalize_ohlcv_data(df)

                if not client.validate_ohlcv_data(df):
                    logger.warning(f"Data validation failed for symbols {symbols} from {client.name}")

            return df

        except Exception as e:
            logger.error(f"Error getting multiple symbols data from {client.name}: {e}")
            raise


# 使用範例
if __name__ == "__main__":
    # 創建客戶端管理器
    manager = DataClientManager()

    # 這裡會註冊具體的客戶端實例（CoinGecko, CryptoCompare等）
    # manager.register_client(CoinGeckoClient())
    # manager.register_client(CryptoCompareClient())

    print("Base data client framework ready!")
    print(f"Available clients: {manager.list_clients()}")
