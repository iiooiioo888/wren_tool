"""
CryptoCompare API 客戶端
作為CoinGecko的備用數據源，負責獲取加密貨幣歷史價格和交易量數據
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoCompareClient:
    """CryptoCompare API 客戶端"""

    BASE_URL = "https://min-api.cryptocompare.com/data"

    def __init__(self, api_key: str = None, requests_per_second: int = 1):
        """
        初始化客戶端

        Args:
            api_key: CryptoCompare API密鑰（可選，免費用戶可不提供）
            requests_per_second: 每秒請求次數限制（免費方案建議1次/秒）
        """
        self.api_key = api_key
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
        self.session = requests.Session()

        # 設定請求頭
        headers = {
            'User-Agent': 'wren-backtesting-platform/1.0'
        }
        if api_key:
            headers['authorization'] = f'Apikey {api_key}'

        self.session.headers.update(headers)

    def _rate_limit_wait(self):
        """實作速率限制等待"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        # 計算需要的等待時間
        min_interval = 1.0 / self.requests_per_second
        if time_since_last_request < min_interval:
            wait_time = min_interval - time_since_last_request
            logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        發出API請求（帶速率限制）

        Args:
            endpoint: API端點
            params: 請求參數

        Returns:
            API回應數據
        """
        self._rate_limit_wait()

        url = f"{self.BASE_URL}{endpoint}"
        try:
            logger.info(f"Making request to: {url}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_historical_daily(
        self,
        fsym: str,
        tsym: str = "USD",
        limit: int = 30,
        aggregate: int = 1,
        toTs: int = None
    ) -> pd.DataFrame:
        """
        獲取歷史日線數據

        Args:
            fsym: 來源貨幣符號（例如：BTC, ETH）
            tsym: 目標貨幣符號（默認USD）
            limit: 返回數據點數量
            aggregate: 數據聚合粒度（天）
            toTs: 結束時間戳（默認當前時間）

        Returns:
            包含OHLCV數據的DataFrame
        """
        endpoint = "/histoday"
        params = {
            'fsym': fsym,
            'tsym': tsym,
            'limit': limit,
            'aggregate': aggregate
        }

        if toTs:
            params['toTs'] = toTs

        try:
            data = self._make_request(endpoint, params)

            if 'Data' not in data:
                logger.warning(f"No data found for {fsym}/{tsym}")
                return pd.DataFrame()

            # 轉換為DataFrame
            df_data = []
            for item in data['Data']:
                df_data.append({
                    'timestamp': pd.to_datetime(item['time'], unit='s'),
                    'open': item['open'],
                    'high': item['high'],
                    'low': item['low'],
                    'close': item['close'],
                    'volume': item['volumefrom']
                })

            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Retrieved {len(df)} daily data points for {fsym}/{tsym}")
            return df

        except Exception as e:
            logger.error(f"Error getting historical daily data for {fsym}: {e}")
            return pd.DataFrame()

    def get_historical_hourly(
        self,
        fsym: str,
        tsym: str = "USD",
        limit: int = 24,
        toTs: int = None
    ) -> pd.DataFrame:
        """
        獲取歷史小時線數據

        Args:
            fsym: 來源貨幣符號
            tsym: 目標貨幣符號
            limit: 返回數據點數量
            toTs: 結束時間戳

        Returns:
            包含OHLCV數據的DataFrame
        """
        endpoint = "/histohour"
        params = {
            'fsym': fsym,
            'tsym': tsym,
            'limit': limit
        }

        if toTs:
            params['toTs'] = toTs

        try:
            data = self._make_request(endpoint, params)

            if 'Data' not in data:
                logger.warning(f"No hourly data found for {fsym}/{tsym}")
                return pd.DataFrame()

            # 轉換為DataFrame
            df_data = []
            for item in data['Data']:
                df_data.append({
                    'timestamp': pd.to_datetime(item['time'], unit='s'),
                    'open': item['open'],
                    'high': item['high'],
                    'low': item['low'],
                    'close': item['close'],
                    'volume': item['volumefrom']
                })

            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Retrieved {len(df)} hourly data points for {fsym}/{tsym}")
            return df

        except Exception as e:
            logger.error(f"Error getting historical hourly data for {fsym}: {e}")
            return pd.DataFrame()

    def get_multiple_coins_history(
        self,
        coin_symbols: List[str],
        tsym: str = "USD",
        days: int = 30
    ) -> pd.DataFrame:
        """
        獲取多個貨幣的歷史數據

        Args:
            coin_symbols: 貨幣符號列表（例如：['BTC', 'ETH']）
            tsym: 目標貨幣符號
            days: 數據天數

        Returns:
            合併後的DataFrame，包含所有貨幣的價格數據
        """
        all_data = []

        for symbol in coin_symbols:
            try:
                # 計算小時數（每小時一個數據點）
                hours = days * 24

                # 獲取小時數據
                df = self.get_historical_hourly(symbol, tsym, limit=hours)

                if not df.empty:
                    df['coin_symbol'] = symbol
                    all_data.append(df)

                # 在請求之間添加延遲
                time.sleep(1.5)  # CryptoCompare免費方案限制較嚴格

            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
                continue

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Retrieved data for {len(all_data)} coins, total {len(combined_df)} records")
            return combined_df
        else:
            logger.warning("No data retrieved for any coin")
            return pd.DataFrame()

    def get_current_price(self, fsyms: List[str], tsyms: List[str] = None) -> Dict:
        """
        獲取當前價格

        Args:
            fsyms: 來源貨幣符號列表
            tsyms: 目標貨幣符號列表（默認['USD'])

        Returns:
            當前價格數據
        """
        if tsyms is None:
            tsyms = ['USD']

        endpoint = "/price"
        params = {
            'fsym': ','.join(fsyms),
            'tsyms': ','.join(tsyms)
        }

        return self._make_request(endpoint, params)

    def get_coin_list(self) -> List[Dict]:
        """
        獲取支援的貨幣列表

        Returns:
            貨幣列表信息
        """
        endpoint = "/all/coinlist"
        return self._make_request(endpoint)

    def test_connection(self) -> bool:
        """
        測試API連接是否正常

        Returns:
            連接是否成功
        """
        try:
            # 嘗試獲取比特幣當前價格作為連接測試
            data = self.get_current_price(['BTC'])
            if 'BTC' in data and 'USD' in data['BTC']:
                logger.info("CryptoCompare API connection test successful")
                return True
            else:
                logger.error("CryptoCompare API returned unexpected data format")
                return False
        except Exception as e:
            logger.error(f"CryptoCompare API connection test failed: {e}")
            return False


# 使用範例
if __name__ == "__main__":
    client = CryptoCompareClient()

    # 測試連接
    if client.test_connection():
        print("✅ CryptoCompare API連接正常")

        # 獲取比特幣小時數據
        btc_data = client.get_historical_hourly("BTC", limit=24)
        print(f"比特幣小時數據形狀: {btc_data.shape}")
        print(btc_data.head())

        # 獲取多貨幣數據
        coins_data = client.get_multiple_coins_history(
            ["BTC", "ETH"],
            days=7
        )
        print(f"多貨幣數據形狀: {coins_data.shape}")

        # 獲取當前價格
        current_prices = client.get_current_price(["BTC", "ETH"])
        print(f"當前價格: {current_prices}")
    else:
        print("❌ CryptoCompare API連接失敗")
