"""
CoinGecko API 客戶端
負責獲取加密貨幣歷史價格和交易量數據
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

class CoinGeckoClient:
    """CoinGecko API 客戶端"""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, requests_per_minute: int = 10):
        """
        初始化客戶端

        Args:
            requests_per_minute: 每分鐘請求次數限制（免費方案建議不要超過10）
        """
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0
        self.session = requests.Session()
        # 設定User-Agent以避免被阻擋
        self.session.headers.update({
            'User-Agent': 'wren-backtesting-platform/1.0'
        })

    def _rate_limit_wait(self):
        """實作速率限制等待"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        # 計算需要的等待時間
        min_interval = 60.0 / self.requests_per_minute
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

    def get_coin_list(self) -> List[Dict]:
        """
        獲取所有支援的加密貨幣列表

        Returns:
            貨幣列表，每個貨幣包含id, symbol, name等信息
        """
        endpoint = "/coins/list"
        return self._make_request(endpoint)

    def get_coin_history(
        self,
        coin_id: str,
        date: str,
        localization: str = "false"
    ) -> Dict:
        """
        獲取指定貨幣在指定日期的歷史數據

        Args:
            coin_id: 貨幣ID（例如：bitcoin, ethereum）
            date: 日期（格式：dd-mm-yyyy）
            localization: 是否本地化

        Returns:
            該貨幣在指定日期的詳細信息
        """
        endpoint = f"/coins/{coin_id}/history"
        params = {
            'date': date,
            'localization': localization
        }
        return self._make_request(endpoint, params)

    def get_coin_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 30,
        interval: str = "daily"
    ) -> Dict:
        """
        獲取貨幣市場圖表數據（價格、交易量等）

        Args:
            coin_id: 貨幣ID
            vs_currency: 對應貨幣（默認USD）
            days: 數據天數
            interval: 數據間隔（daily, hourly等）

        Returns:
            包含價格、市場上限、總交易量等的時間序列數據
        """
        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': interval
        }
        return self._make_request(endpoint, params)

    def get_historical_ohlcv(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 30
    ) -> pd.DataFrame:
        """
        獲取歷史OHLCV數據並轉換為DataFrame

        Args:
            coin_id: 貨幣ID
            vs_currency: 對應貨幣
            days: 數據天數

        Returns:
            包含時間戳、開盤價、最高價、最低價、收盤價、交易量的DataFrame
        """
        try:
            data = self.get_coin_market_chart(coin_id, vs_currency, days)

            # 提取價格數據
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])

            if not prices:
                logger.warning(f"No price data found for {coin_id}")
                return pd.DataFrame()

            # 轉換為DataFrame
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0

                df_data.append({
                    'timestamp': pd.to_datetime(timestamp, unit='ms'),
                    'open': price,  # CoinGecko的市場圖表API主要提供收盤價
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                })

            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Retrieved {len(df)} data points for {coin_id}")
            return df

        except Exception as e:
            logger.error(f"Error getting historical data for {coin_id}: {e}")
            return pd.DataFrame()

    def get_multiple_coins_history(
        self,
        coin_ids: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        獲取多個貨幣在指定時間範圍內的歷史數據

        Args:
            coin_ids: 貨幣ID列表
            start_date: 開始日期（格式：dd-mm-yyyy）
            end_date: 結束日期（格式：dd-mm-yyyy）

        Returns:
            合併後的DataFrame，包含所有貨幣的價格數據
        """
        all_data = []

        for coin_id in coin_ids:
            try:
                # 計算日期範圍內的天數
                start = datetime.strptime(start_date, "%d-%m-%Y")
                end = datetime.strptime(end_date, "%d-%m-%Y")
                days = (end - start).days

                if days <= 0:
                    logger.warning(f"Invalid date range: {start_date} to {end_date}")
                    continue

                # 獲取數據
                df = self.get_historical_ohlcv(coin_id, days=min(days, 365))

                if not df.empty:
                    df['coin_id'] = coin_id
                    all_data.append(df)

                # 在請求之間添加小延遲
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error getting data for {coin_id}: {e}")
                continue

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Retrieved data for {len(all_data)} coins, total {len(combined_df)} records")
            return combined_df
        else:
            logger.warning("No data retrieved for any coin")
            return pd.DataFrame()

    def test_connection(self) -> bool:
        """
        測試API連接是否正常

        Returns:
            連接是否成功
        """
        try:
            # 嘗試獲取ping端點（如果有的話）或簡單的請求
            endpoint = "/ping"
            self._make_request(endpoint)
            logger.info("CoinGecko API connection test successful")
            return True
        except Exception as e:
            logger.error(f"CoinGecko API connection test failed: {e}")
            return False


# 使用範例
if __name__ == "__main__":
    client = CoinGeckoClient()

    # 測試連接
    if client.test_connection():
        print("✅ CoinGecko API連接正常")

        # 獲取比特幣歷史數據
        btc_data = client.get_historical_ohlcv("bitcoin", days=7)
        print(f"比特幣數據形狀: {btc_data.shape}")
        print(btc_data.head())

        # 獲取多貨幣數據
        coins_data = client.get_multiple_coins_history(
            ["bitcoin", "ethereum"],
            "01-10-2025",
            "16-10-2025"
        )
        print(f"多貨幣數據形狀: {coins_data.shape}")
    else:
        print("❌ CoinGecko API連接失敗")
