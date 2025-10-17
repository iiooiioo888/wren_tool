"""
AMM模擬器
負責模擬Uniswap V3風格的自動做市商行為，包含集中流動性和價格發現機制
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
class LiquidityPosition:
    """流動性頭寸"""
    pool_id: str
    token0: str
    token1: str
    liquidity: float
    price_lower: float
    price_upper: float
    current_price: float
    fee_growth0: float
    fee_growth1: float
    tokens_owed0: float
    tokens_owed1: float
    created_at: pd.Timestamp

@dataclass
class AMMPool:
    """AMM資金池"""
    pool_id: str
    token0: str
    token1: str
    fee_rate: float
    current_price: float
    sqrt_price: float
    liquidity: float
    total_fees0: float
    total_fees1: float
    positions: List[LiquidityPosition]

@dataclass
class AMMSimulationResult:
    """AMM模擬結果"""
    final_pools: Dict[str, AMMPool]
    price_history: List[Dict]
    volume_history: List[Dict]
    impermanent_loss: Dict[str, float]
    total_fees_earned: Dict[str, float]
    performance_metrics: Dict[str, float]

class AMMSimulator:
    """AMM模擬器"""

    def __init__(
        self,
        fee_rates: List[float] = [0.0005, 0.003, 0.001],  # 常見手續費率
        price_impact_factor: float = 0.001,
        max_positions_per_pool: int = 100
    ):
        """
        初始化AMM模擬器

        Args:
            fee_rates: 支援的手續費率列表
            price_impact_factor: 價格影響因子
            max_positions_per_pool: 每個資金池最大頭寸數量
        """
        self.fee_rates = fee_rates
        self.price_impact_factor = price_impact_factor
        self.max_positions_per_pool = max_positions_per_pool

        # 活躍資金池
        self.pools: Dict[str, AMMPool] = {}

        # 價格歷史
        self.price_history: List[Dict] = []

        # 成交量歷史
        self.volume_history: List[Dict] = []

    def create_pool(
        self,
        token0: str,
        token1: str,
        initial_price: float,
        fee_rate: float = 0.003
    ) -> str:
        """
        創建AMM資金池

        Args:
            token0: 第一個token符號
            token1: 第二個token符號
            initial_price: 初始價格（token1/token0）
            fee_rate: 手續費率

        Returns:
            資金池ID
        """
        pool_id = f"{token0}_{token1}_{fee_rate}"

        # 計算平方根價格
        sqrt_price = np.sqrt(initial_price)

        pool = AMMPool(
            pool_id=pool_id,
            token0=token0,
            token1=token1,
            fee_rate=fee_rate,
            current_price=initial_price,
            sqrt_price=sqrt_price,
            liquidity=0,
            total_fees0=0,
            total_fees1=0,
            positions=[]
        )

        self.pools[pool_id] = pool
        logger.info(f"Created AMM pool: {pool_id} with initial price {initial_price}")

        return pool_id

    def add_liquidity(
        self,
        pool_id: str,
        price_lower: float,
        price_upper: float,
        liquidity_usd: float,
        current_price: float = None
    ) -> LiquidityPosition:
        """
        添加流動性頭寸

        Args:
            pool_id: 資金池ID
            price_lower: 價格下限
            price_upper: 價格上限
            liquidity_usd: 流動性價值（美元）
            current_price: 當前價格（如果為None，使用資金池當前價格）

        Returns:
            創建的流動性頭寸
        """
        if pool_id not in self.pools:
            raise ValueError(f"Pool {pool_id} not found")

        pool = self.pools[pool_id]

        if current_price is None:
            current_price = pool.current_price

        # 檢查價格範圍是否有效
        if price_lower >= price_upper:
            raise ValueError("Price lower must be less than price upper")

        if current_price < price_lower or current_price > price_upper:
            logger.warning(f"Current price {current_price} is outside position range [{price_lower}, {price_upper}]")

        # 計算頭寸中的token數量（簡化計算）
        liquidity = self._calculate_liquidity_from_usd(
            liquidity_usd, price_lower, price_upper, current_price
        )

        # 創建頭寸
        position = LiquidityPosition(
            pool_id=pool_id,
            token0=pool.token0,
            token1=pool.token1,
            liquidity=liquidity,
            price_lower=price_lower,
            price_upper=price_upper,
            current_price=current_price,
            fee_growth0=0,
            fee_growth1=0,
            tokens_owed0=0,
            tokens_owed1=0,
            created_at=pd.Timestamp.now()
        )

        # 添加到資金池
        pool.positions.append(position)
        pool.liquidity += liquidity

        logger.info(f"Added liquidity to {pool_id}: ${liquidity_usd:,.2f} "
                   f"at range [{price_lower:.2f}, {price_upper:.2f}]")

        return position

    def _calculate_liquidity_from_usd(
        self,
        liquidity_usd: float,
        price_lower: float,
        price_upper: float,
        current_price: float
    ) -> float:
        """從美元價值計算流動性數量"""
        try:
            # 簡化的流動性計算
            # 實際AMM流動性計算更複雜，涉及價格範圍和當前價格位置

            if current_price <= price_lower:
                # 當前價格低於範圍，只提供token0
                amount0 = liquidity_usd / price_lower
                amount1 = 0
            elif current_price >= price_upper:
                # 當前價格高於範圍，只提供token1
                amount1 = liquidity_usd / current_price
                amount0 = 0
            else:
                # 當前價格在範圍內，提供兩個token
                sqrt_price = np.sqrt(current_price)
                sqrt_lower = np.sqrt(price_lower)
                sqrt_upper = np.sqrt(price_upper)

                # 計算L = sqrt(x*y)
                amount0 = liquidity_usd / (2 * sqrt_price)
                amount1 = liquidity_usd / 2

            # 簡化的流動性估計
            estimated_liquidity = min(amount0, amount1) if amount0 > 0 and amount1 > 0 else max(amount0, amount1)

            return estimated_liquidity

        except Exception:
            return liquidity_usd  # 簡化回退

    def simulate_swap(
        self,
        pool_id: str,
        token_in: str,
        amount_in: float,
        current_timestamp: pd.Timestamp = None
    ) -> Dict[str, Any]:
        """
        模擬交換交易

        Args:
            pool_id: 資金池ID
            token_in: 輸入token
            amount_in: 輸入數量
            current_timestamp: 當前時間戳

        Returns:
            交換結果
        """
        if pool_id not in self.pools:
            raise ValueError(f"Pool {pool_id} not found")

        pool = self.pools[pool_id]
        current_time = current_timestamp or pd.Timestamp.now()

        # 計算交換結果
        amount_out, price_impact, fee_amount = self._calculate_swap_output(
            pool, token_in, amount_in
        )

        # 更新資金池狀態
        if token_in == pool.token0:
            pool.total_fees0 += fee_amount
        else:
            pool.total_fees1 += fee_amount

        # 更新價格（簡化）
        if amount_out > 0:
            new_price = self._update_pool_price(pool, token_in, amount_in, amount_out)
            pool.current_price = new_price
            pool.sqrt_price = np.sqrt(new_price)

        # 記錄價格歷史
        self.price_history.append({
            'timestamp': current_time,
            'pool_id': pool_id,
            'price': pool.current_price,
            'token0': pool.token0,
            'token1': pool.token1
        })

        # 記錄成交量歷史
        self.volume_history.append({
            'timestamp': current_time,
            'pool_id': pool_id,
            'token_in': token_in,
            'amount_in': amount_in,
            'amount_out': amount_out,
            'fee_amount': fee_amount,
            'price_impact': price_impact
        })

        logger.info(f"Swap in {pool_id}: {amount_in:.6f} {token_in} -> {amount_out:.6f} {pool.token1 if token_in == pool.token0 else pool.token0}")

        return {
            'amount_out': amount_out,
            'price_impact': price_impact,
            'fee_amount': fee_amount,
            'new_price': pool.current_price,
            'execution_price': amount_out / amount_in if amount_in > 0 else 0
        }

    def _calculate_swap_output(
        self,
        pool: AMMPool,
        token_in: str,
        amount_in: float
    ) -> Tuple[float, float, float]:
        """計算交換輸出"""
        try:
            # 簡化的AMM交換計算
            # 實際Uniswap V3計算更複雜，涉及tick和流動性分佈

            if pool.liquidity <= 0:
                return 0, 0, 0

            # 計算價格影響
            k = pool.liquidity ** 2  # 簡化的恆定乘積

            if token_in == pool.token0:
                # 用token0買token1
                reserve0 = pool.liquidity / pool.sqrt_price
                reserve1 = pool.liquidity * pool.sqrt_price

                # 簡化的交換計算
                amount_out = amount_in * (reserve1 / (reserve0 + amount_in))
                price_impact = amount_in / (reserve0 + amount_in)

            else:
                # 用token1買token0
                reserve0 = pool.liquidity / pool.sqrt_price
                reserve1 = pool.liquidity * pool.sqrt_price

                # 簡化的交換計算
                amount_out = amount_in * (reserve0 / (reserve1 + amount_in))
                price_impact = amount_in / (reserve1 + amount_in)

            # 計算手續費
            fee_amount = amount_in * pool.fee_rate

            return amount_out, price_impact, fee_amount

        except Exception as e:
            logger.error(f"Error calculating swap output: {e}")
            return 0, 0, 0

    def _update_pool_price(self, pool: AMMPool, token_in: str, amount_in: float, amount_out: float) -> float:
        """更新資金池價格"""
        try:
            # 簡化的價格更新
            if token_in == pool.token0:
                # 賣出token0，價格上漲
                price_change = amount_in / (pool.liquidity * 2)
                new_price = pool.current_price * (1 + price_change)
            else:
                # 賣出token1，價格下跌
                price_change = amount_in / (pool.liquidity * 2)
                new_price = pool.current_price * (1 - price_change)

            return max(new_price, 0.000001)  # 確保價格不為負

        except Exception:
            return pool.current_price

    def simulate_market_impact(
        self,
        pool_id: str,
        trade_size_usd: float,
        current_price: float,
        direction: str = "buy"
    ) -> Dict[str, float]:
        """
        模擬市場影響

        Args:
            pool_id: 資金池ID
            trade_size_usd: 交易規模（美元）
            current_price: 當前價格
            direction: 交易方向（buy或sell）

        Returns:
            市場影響分析結果
        """
        if pool_id not in self.pools:
            raise ValueError(f"Pool {pool_id} not found")

        pool = self.pools[pool_id]

        # 計算價格影響
        if direction == "buy":
            # 買單：價格上漲
            price_impact = self.price_impact_factor * (trade_size_usd / pool.liquidity)
            effective_price = current_price * (1 + price_impact)
            slippage = price_impact
        else:
            # 賣單：價格下跌
            price_impact = self.price_impact_factor * (trade_size_usd / pool.liquidity)
            effective_price = current_price * (1 - price_impact)
            slippage = -price_impact

        # 計算手續費影響
        fee_impact = pool.fee_rate

        # 計算總成本
        total_cost = abs(slippage) + fee_impact

        return {
            'price_impact': price_impact,
            'slippage': slippage,
            'fee_impact': fee_impact,
            'total_cost': total_cost,
            'effective_price': effective_price,
            'execution_price': effective_price  # 考慮手續費後的執行價格
        }

    def calculate_position_pnl(
        self,
        position: LiquidityPosition,
        current_price: float
    ) -> Dict[str, float]:
        """
        計算頭寸損益

        Args:
            position: 流動性頭寸
            current_price: 當前價格

        Returns:
            損益計算結果
        """
        try:
            # 計算頭寸當前價值
            current_value = self._calculate_position_value(position, current_price)

            # 計算無常損失（簡化）
            price_ratio = current_price / position.current_price
            if price_ratio > 0:
                il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
            else:
                il = 0

            # 計算手續費收入（簡化估計）
            fee_earned = (position.fee_growth0 + position.fee_growth1) * 0.5

            # 計算總損益
            total_pnl = current_value - position.liquidity + fee_earned
            pnl_percentage = total_pnl / position.liquidity if position.liquidity > 0 else 0

            return {
                'current_value': current_value,
                'impermanent_loss': il,
                'fee_earned': fee_earned,
                'total_pnl': total_pnl,
                'pnl_percentage': pnl_percentage,
                'hodl_return': (current_price - position.current_price) / position.current_price
            }

        except Exception as e:
            logger.error(f"Error calculating position PnL: {e}")
            return {}

    def _calculate_position_value(
        self,
        position: LiquidityPosition,
        current_price: float
    ) -> float:
        """計算頭寸當前價值"""
        try:
            # 簡化的頭寸價值計算
            if current_price < position.price_lower:
                # 只持有token0
                amount0 = position.liquidity * np.sqrt(position.price_lower / current_price)
                amount1 = 0
            elif current_price > position.price_upper:
                # 只持有token1
                amount0 = 0
                amount1 = position.liquidity * np.sqrt(current_price / position.price_upper)
            else:
                # 持有兩個token
                amount0 = position.liquidity * np.sqrt(position.price_lower / current_price)
                amount1 = position.liquidity * np.sqrt(current_price / position.price_upper)

            # 計算總價值（假設兩個token都是穩定幣或有價格）
            # 實際應該查詢真實價格
            value0 = amount0 * position.price_lower  # 簡化：使用範圍下限作為價格
            value1 = amount1 * position.price_upper  # 簡化：使用範圍上限作為價格

            return value0 + value1

        except Exception:
            return 0.0

    def run_simulation(
        self,
        price_data: pd.DataFrame,
        trading_activity: List[Dict],
        initial_liquidity: Dict[str, float] = None
    ) -> AMMSimulationResult:
        """
        運行AMM模擬

        Args:
            price_data: 價格數據
            trading_activity: 交易活動列表
            initial_liquidity: 初始流動性設置

        Returns:
            模擬結果
        """
        logger.info("Starting AMM simulation")

        # 初始化資金池
        if initial_liquidity:
            for pool_config, liquidity in initial_liquidity.items():
                if isinstance(pool_config, dict):
                    pool_id = self.create_pool(
                        pool_config['token0'],
                        pool_config['token1'],
                        pool_config['initial_price'],
                        pool_config.get('fee_rate', 0.003)
                    )

                    # 添加初始流動性
                    self.add_liquidity(
                        pool_id,
                        pool_config['price_lower'],
                        pool_config['price_upper'],
                        liquidity
                    )

        # 運行模擬
        for activity in trading_activity:
            try:
                timestamp = activity['timestamp']
                pool_id = activity['pool_id']
                action = activity['action']

                if action == 'swap':
                    self.simulate_swap(
                        pool_id,
                        activity['token_in'],
                        activity['amount_in'],
                        timestamp
                    )

                elif action == 'add_liquidity':
                    self.add_liquidity(
                        pool_id,
                        activity['price_lower'],
                        activity['price_upper'],
                        activity['liquidity_usd']
                    )

            except Exception as e:
                logger.error(f"Error in simulation step: {e}")
                continue

        # 計算最終結果
        final_pools = self.pools.copy()

        # 計算無常損失
        impermanent_loss = {}
        total_fees_earned = {}

        for pool_id, pool in final_pools.items():
            pool_fees = pool.total_fees0 + pool.total_fees1
            total_fees_earned[pool_id] = pool_fees

            # 計算每個頭寸的無常損失
            for position in pool.positions:
                pos_pnl = self.calculate_position_pnl(position, pool.current_price)
                if pos_pnl:
                    il_key = f"{pool_id}_{position.created_at.strftime('%Y%m%d_%H%M%S')}"
                    impermanent_loss[il_key] = pos_pnl.get('impermanent_loss', 0)

        # 計算性能指標
        performance_metrics = self._calculate_simulation_performance()

        logger.info("AMM simulation completed")

        return AMMSimulationResult(
            final_pools=final_pools,
            price_history=self.price_history,
            volume_history=self.volume_history,
            impermanent_loss=impermanent_loss,
            total_fees_earned=total_fees_earned,
            performance_metrics=performance_metrics
        )

    def _calculate_simulation_performance(self) -> Dict[str, float]:
        """計算模擬性能指標"""
        if not self.price_history:
            return {}

        # 價格變化統計
        prices = [entry['price'] for entry in self.price_history]
        price_returns = np.diff(prices) / prices[:-1]

        # 成交量統計
        volumes = [entry['amount_in'] for entry in self.volume_history]

        # 價格波動率
        price_volatility = np.std(price_returns) * np.sqrt(365) if len(price_returns) > 0 else 0

        # 總成交量
        total_volume = sum(volumes) if volumes else 0

        # 手續費收入統計
        total_fees = sum(
            pool.total_fees0 + pool.total_fees1
            for pool in self.pools.values()
        )

        return {
            'price_volatility': price_volatility,
            'total_volume': total_volume,
            'total_fees_earned': total_fees,
            'active_pools': len(self.pools),
            'total_positions': sum(len(pool.positions) for pool in self.pools.values()),
            'avg_liquidity': np.mean([pool.liquidity for pool in self.pools.values()]) if self.pools else 0
        }

    def generate_amm_report(self) -> str:
        """生成AMM模擬報告"""
        report = []
        report.append("=" * 60)
        report.append("AMM模擬報告")
        report.append("=" * 60)

        # 資金池統計
        report.append(f"活躍資金池數量: {len(self.pools)}")
        report.append("資金池詳情:")

        for pool_id, pool in self.pools.items():
            report.append(f"  {pool_id}:")
            report.append(f"    當前價格: ${pool.current_price".4f"}")
            report.append(f"    總流動性: {pool.liquidity".6f"}")
            report.append(f"    頭寸數量: {len(pool.positions)}")
            report.append(f"    累計手續費: ${pool.total_fees0 + pool.total_fees1".6f"}")
        report.append("")

        # 價格歷史統計
        if self.price_history:
            prices = [entry['price'] for entry in self.price_history]
            report.append("價格統計:")
            report.append(f"  價格範圍: ${min(prices)".4f"} - ${max(prices)".4f"}")
            report.append(f"  平均價格: ${np.mean(prices)".4f"}")
            report.append(f"  價格波動率: {np.std(prices)".4f"}")
        report.append("")

        # 成交量統計
        if self.volume_history:
            volumes = [entry['amount_in'] for entry in self.volume_history]
            report.append("成交量統計:")
            report.append(f"  總成交量: {sum(volumes)".2f"}")
            report.append(f"  平均成交量: {np.mean(volumes)".2f"}")
            report.append(f"  交易次數: {len(volumes)}")
        report.append("")

        # 性能指標
        if self.price_history:
            perf_metrics = self._calculate_simulation_performance()
            report.append("性能指標:")
            for metric, value in perf_metrics.items():
                if isinstance(value, float):
                    if 'fees' in metric.lower() or 'volume' in metric.lower():
                        report.append(f"  {metric}: ${value".2f"}")
                    else:
                        report.append(f"  {metric}: {value".4f"}")
                else:
                    report.append(f"  {metric}: {value}")

        report.append("=" * 60)

        return "\n".join(report)


# 使用範例
if __name__ == "__main__":
    # 創建AMM模擬器
    simulator = AMMSimulator(fee_rates=[0.003, 0.001])

    # 創建資金池
    pool_id = simulator.create_pool('ETH', 'USDC', 3000, 0.003)
    print(f"創建資金池: {pool_id}")

    # 添加流動性
    position = simulator.add_liquidity(
        pool_id,
        price_lower=2500,
        price_upper=3500,
        liquidity_usd=100000
    )

    print(f"添加流動性頭寸: ${position.liquidity:.2f}")

    # 模擬交換交易
    swap_result = simulator.simulate_swap(
        pool_id,
        'ETH',
        10,  # 賣出10個ETH
        pd.Timestamp.now()
    )

    print(f"交換結果: 獲得 {swap_result['amount_out']:.2f} USDC")
    print(f"價格影響: {swap_result['price_impact']:.4f}")
    print(f"手續費: {swap_result['fee_amount']:.4f}")

    # 模擬市場影響
    market_impact = simulator.simulate_market_impact(
        pool_id,
        10000,  # 1萬美元交易
        3000,
        "buy"
    )

    print(f"市場影響分析: {market_impact}")

    # 生成報告
    report = simulator.generate_amm_report()
    print("\nAMM模擬報告:")
    print(report)
