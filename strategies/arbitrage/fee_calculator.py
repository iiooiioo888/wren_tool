"""
手續費計算模組
負責計算各交易所的手續費率和交易成本
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class FeeStructure:
    """手續費結構"""
    exchange: str
    maker_fee: float  # 掛單手續費率
    taker_fee: float  # 吃單手續費率
    withdrawal_fee: Dict[str, float]  # 提幣手續費（幣種 -> 費率）
    min_withdrawal: Dict[str, float]  # 最小提幣量
    deposit_fee: Dict[str, float]     # 充值手續費
    trading_volume_tiers: Dict[str, Dict[str, float]]  # 交易量等級費率

@dataclass
class TradeCost:
    """交易成本"""
    exchange: str
    symbol: str
    trade_type: str  # 'buy', 'sell', 'maker', 'taker'
    quantity: float
    price: float
    trading_fee: float
    withdrawal_fee: float
    network_fee: float
    total_cost: float
    net_proceeds: float

@dataclass
class ArbitrageCost:
    """套利成本"""
    buy_exchange: str
    sell_exchange: str
    symbol: str
    quantity: float
    buy_price: float
    sell_price: float
    trading_fee_buy: float
    trading_fee_sell: float
    withdrawal_fee: float
    transfer_cost: float
    total_cost: float
    gross_profit: float
    net_profit: float
    profit_margin: float

class FeeCalculator:
    """手續費計算器"""

    def __init__(self):
        """初始化手續費計算器"""
        # 各交易所的手續費結構（基於2024年數據，可能會變化）
        self.fee_structures = self._initialize_fee_structures()

        # 網路轉帳費用估計（美元）
        self.network_fees = {
            'BTC': 5.0,
            'ETH': 10.0,
            'BNB': 0.1,
            'ADA': 0.2,
            'SOL': 0.01,
            'DOT': 0.1,
            'AVAX': 0.1,
            'MATIC': 0.1,
            'USDC': 5.0,
            'USDT': 5.0
        }

    def _initialize_fee_structures(self) -> Dict[str, FeeStructure]:
        """初始化各交易所手續費結構"""
        return {
            'binance': FeeStructure(
                exchange='binance',
                maker_fee=0.001,  # 0.1%
                taker_fee=0.001,  # 0.1%
                withdrawal_fee={
                    'BTC': 0.0005,
                    'ETH': 0.005,
                    'USDT': 1.0,
                    'USDC': 1.0
                },
                min_withdrawal={
                    'BTC': 0.001,
                    'ETH': 0.01,
                    'USDT': 10,
                    'USDC': 10
                },
                deposit_fee={},
                trading_volume_tiers={
                    'vip0': {'maker': 0.001, 'taker': 0.001},
                    'vip1': {'maker': 0.0009, 'taker': 0.001},
                    'vip2': {'maker': 0.0008, 'taker': 0.001}
                }
            ),

            'coinbase': FeeStructure(
                exchange='coinbase',
                maker_fee=0.005,  # 0.5%
                taker_fee=0.005,  # 0.5%
                withdrawal_fee={
                    'BTC': 0.0001,
                    'ETH': 0.005,
                    'USDT': 5.0,
                    'USDC': 5.0
                },
                min_withdrawal={
                    'BTC': 0.0001,
                    'ETH': 0.005,
                    'USDT': 5,
                    'USDC': 5
                },
                deposit_fee={},
                trading_volume_tiers={}
            ),

            'kraken': FeeStructure(
                exchange='kraken',
                maker_fee=0.0016,  # 0.16%
                taker_fee=0.0026,  # 0.26%
                withdrawal_fee={
                    'BTC': 0.0001,
                    'ETH': 0.005,
                    'USDT': 5.0,
                    'USDC': 5.0
                },
                min_withdrawal={
                    'BTC': 0.0005,
                    'ETH': 0.01,
                    'USDT': 5,
                    'USDC': 5
                },
                deposit_fee={},
                trading_volume_tiers={}
            ),

            'kucoin': FeeStructure(
                exchange='kucoin',
                maker_fee=0.001,  # 0.1%
                taker_fee=0.001,  # 0.1%
                withdrawal_fee={
                    'BTC': 0.0005,
                    'ETH': 0.005,
                    'USDT': 2.0,
                    'USDC': 2.0
                },
                min_withdrawal={
                    'BTC': 0.0005,
                    'ETH': 0.01,
                    'USDT': 10,
                    'USDC': 10
                },
                deposit_fee={},
                trading_volume_tiers={}
            ),

            'bybit': FeeStructure(
                exchange='bybit',
                maker_fee=0.001,  # 0.1%
                taker_fee=0.001,  # 0.1%
                withdrawal_fee={
                    'BTC': 0.0005,
                    'ETH': 0.005,
                    'USDT': 3.0,
                    'USDC': 3.0
                },
                min_withdrawal={
                    'BTC': 0.0005,
                    'ETH': 0.01,
                    'USDT': 10,
                    'USDC': 10
                },
                deposit_fee={},
                trading_volume_tiers={}
            ),

            'okx': FeeStructure(
                exchange='okx',
                maker_fee=0.0008,  # 0.08%
                taker_fee=0.0015,  # 0.15%
                withdrawal_fee={
                    'BTC': 0.0002,
                    'ETH': 0.001,
                    'USDT': 2.0,
                    'USDC': 2.0
                },
                min_withdrawal={
                    'BTC': 0.0002,
                    'ETH': 0.001,
                    'USDT': 2,
                    'USDC': 2
                },
                deposit_fee={},
                trading_volume_tiers={}
            )
        }

    def calculate_trading_fee(
        self,
        exchange: str,
        symbol: str,
        quantity: float,
        price: float,
        is_maker: bool = False,
        trading_volume_30d: float = 0
    ) -> float:
        """
        計算交易手續費

        Args:
            exchange: 交易所名稱
            symbol: 交易對符號
            quantity: 交易數量
            price: 交易價格
            is_maker: 是否為掛單
            trading_volume_30d: 30天交易量（美元）

        Returns:
            手續費金額
        """
        if exchange not in self.fee_structures:
            logger.warning(f"Unknown exchange: {exchange}, using default fees")
            fee_rate = 0.001  # 默認0.1%
        else:
            fee_structure = self.fee_structures[exchange]

            # 根據交易量選擇費率等級
            fee_rate = self._get_fee_rate_by_volume(fee_structure, trading_volume_30d, is_maker)

        # 計算手續費金額
        trade_value = quantity * price
        fee_amount = trade_value * fee_rate

        logger.debug(f"Trading fee for {exchange} {symbol}: ${fee_amount:.4f} (rate: {fee_rate:.4f})")
        return fee_amount

    def _get_fee_rate_by_volume(
        self,
        fee_structure: FeeStructure,
        trading_volume_30d: float,
        is_maker: bool
    ) -> float:
        """根據交易量獲取費率"""
        fee_type = 'maker' if is_maker else 'taker'

        # 如果沒有分級費率，直接返回基礎費率
        if not fee_structure.trading_volume_tiers:
            return fee_structure.maker_fee if is_maker else fee_structure.taker_fee

        # 簡化的費率等級判斷（實際應該根據具體規則）
        if trading_volume_30d > 10000000:  # 1000萬美元以上
            tier = 'vip2'
        elif trading_volume_30d > 1000000:  # 100萬美元以上
            tier = 'vip1'
        else:
            tier = 'vip0'

        if tier in fee_structure.trading_volume_tiers:
            return fee_structure.trading_volume_tiers[tier][fee_type]

        # 默認費率
        return fee_structure.maker_fee if is_maker else fee_structure.taker_fee

    def calculate_withdrawal_fee(
        self,
        exchange: str,
        currency: str,
        amount: float
    ) -> float:
        """
        計算提幣手續費

        Args:
            exchange: 交易所名稱
            currency: 幣種
            amount: 提幣數量

        Returns:
            提幣手續費金額
        """
        if exchange not in self.fee_structures:
            return self.network_fees.get(currency, 1.0)  # 使用網路費用作為默認

        fee_structure = self.fee_structures[exchange]

        if currency in fee_structure.withdrawal_fee:
            fee_rate = fee_structure.withdrawal_fee[currency]

            # 檢查是否為固定費用還是比例費用
            if fee_rate < 1:  # 假設小於1的是比例費用
                fee_amount = amount * fee_rate
            else:  # 大於等於1的是固定費用
                fee_amount = fee_rate

            # 檢查最小提幣量
            if currency in fee_structure.min_withdrawal:
                min_amount = fee_structure.min_withdrawal[currency]
                if amount < min_amount:
                    logger.warning(f"Withdrawal amount {amount} below minimum {min_amount} for {currency}")
                    return 0  # 不收取手續費但無法提幣

            return fee_amount

        # 如果沒有特定費率，使用網路費用估計
        return self.network_fees.get(currency, 1.0)

    def calculate_arbitrage_cost(
        self,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        quantity: float,
        buy_price: float,
        sell_price: float,
        trading_volume_30d: Dict[str, float] = None
    ) -> ArbitrageCost:
        """
        計算套利總成本

        Args:
            buy_exchange: 買入交易所
            sell_exchange: 賣出交易所
            symbol: 交易對符號
            quantity: 交易數量
            buy_price: 買入價格
            sell_price: 賣出價格
            trading_volume_30d: 各交易所30天交易量

        Returns:
            套利成本分析
        """
        if trading_volume_30d is None:
            trading_volume_30d = {}

        # 計算交易手續費
        trading_fee_buy = self.calculate_trading_fee(
            buy_exchange, symbol, quantity, buy_price,
            trading_volume_30d=trading_volume_30d.get(buy_exchange, 0)
        )

        trading_fee_sell = self.calculate_trading_fee(
            sell_exchange, symbol, quantity, sell_price,
            trading_volume_30d=trading_volume_30d.get(sell_exchange, 0)
        )

        # 計算提幣手續費（假設需要將資金轉移）
        withdrawal_fee = self.calculate_withdrawal_fee(sell_exchange, symbol, quantity)

        # 估計轉帳成本（網路費用）
        transfer_cost = self.network_fees.get(symbol, 5.0)

        # 計算總成本
        total_cost = trading_fee_buy + trading_fee_sell + withdrawal_fee + transfer_cost

        # 計算毛利潤
        gross_profit = (sell_price - buy_price) * quantity

        # 計算淨利潤
        net_profit = gross_profit - total_cost

        # 計算利潤率
        profit_margin = net_profit / (buy_price * quantity) if buy_price * quantity > 0 else 0

        return ArbitrageCost(
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            symbol=symbol,
            quantity=quantity,
            buy_price=buy_price,
            sell_price=sell_price,
            trading_fee_buy=trading_fee_buy,
            trading_fee_sell=trading_fee_sell,
            withdrawal_fee=withdrawal_fee,
            transfer_cost=transfer_cost,
            total_cost=total_cost,
            gross_profit=gross_profit,
            net_profit=net_profit,
            profit_margin=profit_margin
        )

    def calculate_optimal_trade_size(
        self,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        buy_price: float,
        sell_price: float,
        max_cost_ratio: float = 0.5  # 最大成本佔比
    ) -> float:
        """
        計算最優交易規模

        Args:
            buy_exchange: 買入交易所
            sell_exchange: 賣出交易所
            symbol: 交易對符號
            buy_price: 買入價格
            sell_price: 賣出價格
            max_cost_ratio: 最大成本佔比

        Returns:
            最優交易規模
        """
        # 計算單位成本
        unit_gross_profit = sell_price - buy_price

        if unit_gross_profit <= 0:
            return 0

        # 估計單位成本（簡化）
        unit_trading_cost = (buy_price + sell_price) * 0.001  # 假設0.1%交易成本
        unit_transfer_cost = self.network_fees.get(symbol, 5.0) / 1000  # 假設交易規模為1000單位

        unit_total_cost = unit_trading_cost + unit_transfer_cost

        # 計算最大可接受的成本
        max_unit_cost = unit_gross_profit * max_cost_ratio

        if unit_total_cost >= max_unit_cost:
            return 0  # 無利可圖

        # 計算最優規模（簡化計算）
        # 實際應該考慮更多因素，如價格影響等
        optimal_quantity = max_unit_cost / unit_total_cost * 1000  # 假設基準規模為1000

        return optimal_quantity

    def compare_exchange_costs(
        self,
        exchanges: List[str],
        symbol: str,
        quantity: float,
        price: float,
        trading_volume_30d: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        比較各交易所交易成本

        Args:
            exchanges: 交易所列表
            symbol: 交易對符號
            quantity: 交易數量
            price: 交易價格
            trading_volume_30d: 各交易所30天交易量

        Returns:
            成本比較DataFrame
        """
        if trading_volume_30d is None:
            trading_volume_30d = {}

        cost_data = []

        for exchange in exchanges:
            # 計算作為maker和taker的成本
            for is_maker in [True, False]:
                trading_fee = self.calculate_trading_fee(
                    exchange, symbol, quantity, price, is_maker,
                    trading_volume_30d.get(exchange, 0)
                )

                withdrawal_fee = self.calculate_withdrawal_fee(exchange, symbol, quantity)

                total_cost = trading_fee + withdrawal_fee

                cost_data.append({
                    'exchange': exchange,
                    'order_type': 'maker' if is_maker else 'taker',
                    'trading_fee': trading_fee,
                    'withdrawal_fee': withdrawal_fee,
                    'total_cost': total_cost,
                    'cost_per_unit': total_cost / quantity if quantity > 0 else 0,
                    'trading_volume_30d': trading_volume_30d.get(exchange, 0)
                })

        return pd.DataFrame(cost_data)

    def calculate_triangular_arbitrage_cost(
        self,
        exchange: str,
        path: List[str],  # 三角套利路徑，如 ['USDT', 'BTC', 'ETH', 'USDT']
        amounts: List[float],  # 各步交易金額
        prices: Dict[str, float],  # 各交易對價格
        trading_volume_30d: float = 0
    ) -> Dict[str, float]:
        """
        計算三角套利成本

        Args:
            exchange: 交易所名稱
            path: 套利路徑
            amounts: 各步交易金額
            prices: 各交易對價格
            trading_volume_30d: 30天交易量

        Returns:
            三角套利成本分析
        """
        if len(path) < 3 or len(amounts) != len(path) - 1:
            raise ValueError("Invalid triangular arbitrage path")

        total_fees = 0
        step_costs = []

        for i in range(len(path) - 1):
            token_from = path[i]
            token_to = path[i + 1]
            amount = amounts[i]

            # 構造交易對符號
            if token_from == 'USDT' or token_to == 'USDT':
                symbol = f"{token_from}{token_to}" if token_from != 'USDT' else f"{token_to}{token_from}"
            else:
                symbol = f"{token_from}{token_to}"

            # 獲取價格
            price = prices.get(symbol, prices.get(f"{token_to}{token_from}", 0))
            if price == 0:
                continue

            # 計算手續費
            trading_fee = self.calculate_trading_fee(
                exchange, symbol, amount, price, trading_volume_30d=trading_volume_30d
            )

            total_fees += trading_fee
            step_costs.append({
                'step': f"{token_from} -> {token_to}",
                'amount': amount,
                'price': price,
                'fee': trading_fee
            })

        # 計算最終價值
        initial_amount = amounts[0]
        final_amount = amounts[-1]

        gross_return = (final_amount - initial_amount) / initial_amount if initial_amount > 0 else 0
        net_return = gross_return - (total_fees / (initial_amount * prices.get(path[0], 1)))

        return {
            'total_fees': total_fees,
            'gross_return': gross_return,
            'net_return': net_return,
            'step_costs': step_costs,
            'profitable': net_return > 0
        }

    def update_fee_structure(self, exchange: str, fee_structure: FeeStructure) -> None:
        """
        更新交易所手續費結構

        Args:
            exchange: 交易所名稱
            fee_structure: 新的手續費結構
        """
        self.fee_structures[exchange] = fee_structure
        logger.info(f"Updated fee structure for {exchange}")

    def get_fee_summary(self, exchange: str) -> Dict[str, Any]:
        """
        獲取交易所手續費摘要

        Args:
            exchange: 交易所名稱

        Returns:
            手續費摘要信息
        """
        if exchange not in self.fee_structures:
            return {}

        fee_structure = self.fee_structures[exchange]

        return {
            'exchange': exchange,
            'maker_fee': fee_structure.maker_fee,
            'taker_fee': fee_structure.taker_fee,
            'withdrawal_fees': fee_structure.withdrawal_fee,
            'trading_volume_tiers': fee_structure.trading_volume_tiers,
            'supports_volume_discounts': len(fee_structure.trading_volume_tiers) > 0
        }

    def estimate_annual_costs(
        self,
        exchange: str,
        annual_volume: float,
        avg_trade_size: float,
        withdrawal_frequency: int = 12
    ) -> Dict[str, float]:
        """
        估計年交易成本

        Args:
            exchange: 交易所名稱
            annual_volume: 年交易量（美元）
            avg_trade_size: 平均交易規模（美元）
            withdrawal_frequency: 年提幣次數

        Returns:
            年成本估計
        """
        if exchange not in self.fee_structures:
            return {}

        # 計算交易手續費（假設50% maker, 50% taker）
        maker_fee_rate = self.fee_structures[exchange].maker_fee
        taker_fee_rate = self.fee_structures[exchange].taker_fee

        trading_fee_rate = (maker_fee_rate + taker_fee_rate) / 2
        annual_trading_fees = annual_volume * trading_fee_rate

        # 計算提幣費用
        # 假設主要提幣BTC和USDT
        btc_withdrawal_fee = self.fee_structures[exchange].withdrawal_fee.get('BTC', 0.0005) * 50000  # 假設BTC價格5萬
        usdt_withdrawal_fee = self.fee_structures[exchange].withdrawal_fee.get('USDT', 5.0)

        annual_withdrawal_fees = (btc_withdrawal_fee + usdt_withdrawal_fee) * withdrawal_frequency

        # 總成本
        total_annual_cost = annual_trading_fees + annual_withdrawal_fees

        return {
            'annual_trading_fees': annual_trading_fees,
            'annual_withdrawal_fees': annual_withdrawal_fees,
            'total_annual_cost': total_annual_cost,
            'cost_per_trade': total_annual_cost / (annual_volume / avg_trade_size) if annual_volume > 0 else 0,
            'fee_rate_effective': total_annual_cost / annual_volume if annual_volume > 0 else 0
        }


# 使用範例
if __name__ == "__main__":
    # 創建手續費計算器
    fee_calculator = FeeCalculator()

    # 計算單筆交易手續費
    trading_fee = fee_calculator.calculate_trading_fee(
        exchange='binance',
        symbol='BTCUSDT',
        quantity=1.0,
        price=50000,
        is_maker=True
    )

    print(f"Binance掛單手續費: ${trading_fee:.2f}")

    # 計算提幣手續費
    withdrawal_fee = fee_calculator.calculate_withdrawal_fee(
        exchange='binance',
        currency='BTC',
        amount=0.5
    )

    print(f"BTC提幣手續費: {withdrawal_fee:.6f} BTC")

    # 計算套利成本
    arbitrage_cost = fee_calculator.calculate_arbitrage_cost(
        buy_exchange='coinbase',
        sell_exchange='binance',
        symbol='BTC',
        quantity=0.1,
        buy_price=50000,
        sell_price=50100
    )

    print("套利成本分析:")
    print(f"  總成本: ${arbitrage_cost.total_cost:.2f}")
    print(f"  毛利潤: ${arbitrage_cost.gross_profit:.2f}")
    print(f"  淨利潤: ${arbitrage_cost.net_profit:.2f}")
    print(f"  利潤率: {arbitrage_cost.profit_margin:.2%}")

    # 比較交易所成本
    cost_comparison = fee_calculator.compare_exchange_costs(
        exchanges=['binance', 'coinbase', 'kraken'],
        symbol='ETHUSDT',
        quantity=10,
        price=3000
    )

    print("\n交易所成本比較:")
    print(cost_comparison)

    # 計算最優交易規模
    optimal_size = fee_calculator.calculate_optimal_trade_size(
        buy_exchange='coinbase',
        sell_exchange='binance',
        symbol='BTC',
        buy_price=50000,
        sell_price=50100
    )

    print(f"\n最優交易規模: {optimal_size:.4f} BTC")

    # 獲取手續費摘要
    fee_summary = fee_calculator.get_fee_summary('binance')
    print(f"\nBinance手續費摘要: {fee_summary}")
