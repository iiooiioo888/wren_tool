"""
庫存管理系統
負責做市策略中的頭寸追蹤、無常損失計算和風險管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """持倉信息"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: pd.Timestamp
    last_update: pd.Timestamp

@dataclass
class InventoryState:
    """庫存狀態"""
    timestamp: pd.Timestamp
    total_value: float
    positions: Dict[str, Position]
    net_exposure: float  # 淨敞口（-1到1之間）
    concentration_risk: float  # 集中度風險
    liquidity_score: float  # 流動性評分

@dataclass
class ImpermanentLoss:
    """無常損失計算結果"""
    initial_value: float
    current_value: float
    hodl_return: float  # 如果持有不變的回報
    amm_return: float    # AMM實際回報
    il_amount: float     # 無常損失金額
    il_percentage: float # 無常損失百分比

class InventoryManager:
    """庫存管理器"""

    def __init__(
        self,
        max_position_size: float = 100000,  # 單個頭寸最大價值
        max_portfolio_risk: float = 0.8,     # 投資組合最大風險敞口
        rebalance_threshold: float = 0.1     # 再平衡閾值
    ):
        """
        初始化庫存管理器

        Args:
            max_position_size: 單個頭寸最大價值
            max_portfolio_risk: 投資組合最大風險敞口
            rebalance_threshold: 再平衡閾值
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.rebalance_threshold = rebalance_threshold

        # 當前持倉
        self.positions: Dict[str, Position] = {}

        # 交易歷史
        self.trade_history: List[Dict] = []

        # 庫存歷史記錄
        self.inventory_history: List[InventoryState] = []

    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: pd.Timestamp,
        trade_type: str = "trade"
    ) -> Position:
        """
        更新持倉

        Args:
            symbol: 交易對符號
            quantity: 交易數量（正數買入，負數賣出）
            price: 交易價格
            timestamp: 交易時間
            trade_type: 交易類型

        Returns:
            更新後的持倉信息
        """
        current_time = timestamp

        if symbol in self.positions:
            # 更新現有持倉
            position = self.positions[symbol]
            old_quantity = position.quantity
            old_avg_price = position.avg_price

            # 計算新的平均價格
            if quantity > 0:  # 買入
                total_cost = old_quantity * old_avg_price + quantity * price
                new_quantity = old_quantity + quantity
                new_avg_price = total_cost / new_quantity if new_quantity != 0 else 0
            else:  # 賣出
                new_quantity = old_quantity + quantity
                if new_quantity == 0:
                    new_avg_price = 0
                else:
                    new_avg_price = old_avg_price

            position.quantity = new_quantity
            position.avg_price = new_avg_price
            position.last_update = current_time

            # 計算未實現損益
            if new_quantity != 0:
                position.market_value = new_quantity * price
                position.unrealized_pnl = (price - new_avg_price) * new_quantity
            else:
                position.market_value = 0
                position.unrealized_pnl = 0

            # 記錄實現損益（如果有平倉）
            if old_quantity * new_quantity < 0 or new_quantity == 0:
                realized_pnl = (price - old_avg_price) * quantity
                position.realized_pnl += realized_pnl

        else:
            # 創建新持倉
            if quantity != 0:
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    entry_time=current_time,
                    last_update=current_time
                )
                self.positions[symbol] = position

        # 記錄交易
        self.trade_history.append({
            'timestamp': current_time,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'trade_type': trade_type,
            'position_before': symbol in self.positions and quantity != 0
        })

        logger.info(f"Updated position for {symbol}: qty={position.quantity}, avg_price={position.avg_price:.2f}")
        return position

    def get_inventory_state(self, current_prices: Dict[str, float]) -> InventoryState:
        """
        獲取當前庫存狀態

        Args:
            current_prices: 當前市場價格字典

        Returns:
            庫存狀態
        """
        current_time = pd.Timestamp.now()

        # 計算總價值和各頭寸價值
        total_value = 0
        position_values = {}

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                position.last_update = current_time

                position_values[symbol] = position.market_value
                total_value += position.market_value

        # 計算淨敞口（簡化為最大單一頭寸佔比）
        if total_value > 0:
            max_exposure = max(abs(value) for value in position_values.values()) / total_value
            net_exposure = max_exposure
        else:
            net_exposure = 0

        # 計算集中度風險（基於頭寸分佈）
        if position_values:
            # 使用赫芬達爾指數計算集中度
            weights = [abs(value) / total_value for value in position_values.values()]
            concentration_risk = sum(w ** 2 for w in weights)
        else:
            concentration_risk = 0

        # 計算流動性評分（簡化）
        liquidity_score = self._calculate_liquidity_score(position_values, total_value)

        inventory_state = InventoryState(
            timestamp=current_time,
            total_value=total_value,
            positions=self.positions.copy(),
            net_exposure=net_exposure,
            concentration_risk=concentration_risk,
            liquidity_score=liquidity_score
        )

        # 記錄歷史
        self.inventory_history.append(inventory_state)

        # 限制歷史記錄長度
        if len(self.inventory_history) > 1000:
            self.inventory_history = self.inventory_history[-500:]

        return inventory_state

    def _calculate_liquidity_score(self, position_values: Dict[str, float], total_value: float) -> float:
        """計算流動性評分"""
        if not position_values or total_value == 0:
            return 1.0

        # 簡化的流動性評分：基於頭寸分佈和規模
        # 假設較小的頭寸有更好的流動性
        avg_position_size = np.mean(list(position_values.values()))
        size_score = min(1.0, avg_position_size / self.max_position_size)

        # 頭寸數量越多，流動性越好（假設可以更快調整）
        diversity_score = min(1.0, len(position_values) / 10)

        return (size_score + diversity_score) / 2

    def check_rebalance_needed(self, current_state: InventoryState) -> Dict[str, Any]:
        """
        檢查是否需要再平衡

        Args:
            current_state: 當前庫存狀態

        Returns:
            再平衡建議
        """
        recommendations = {
            'needs_rebalance': False,
            'reasons': [],
            'suggested_actions': []
        }

        # 檢查風險敞口
        if abs(current_state.net_exposure) > self.max_portfolio_risk:
            recommendations['needs_rebalance'] = True
            recommendations['reasons'].append(f"Risk exposure too high: {current_state.net_exposure:.2f}")
            recommendations['suggested_actions'].append("Reduce position sizes")

        # 檢查集中度風險
        if current_state.concentration_risk > 0.5:
            recommendations['needs_rebalance'] = True
            recommendations['reasons'].append(f"Concentration risk too high: {current_state.concentration_risk:.2f}")
            recommendations['suggested_actions'].append("Diversify positions")

        # 檢查流動性
        if current_state.liquidity_score < 0.3:
            recommendations['needs_rebalance'] = True
            recommendations['reasons'].append(f"Liquidity score too low: {current_state.liquidity_score:.2f}")
            recommendations['suggested_actions'].append("Reduce position sizes for better liquidity")

        # 檢查單個頭寸大小
        for symbol, position in current_state.positions.items():
            if abs(position.market_value) > self.max_position_size:
                recommendations['needs_rebalance'] = True
                recommendations['reasons'].append(f"Position too large for {symbol}: ${position.market_value:,.0f}")
                recommendations['suggested_actions'].append(f"Reduce {symbol} position")

        return recommendations

    def calculate_impermanent_loss(
        self,
        amm_positions: Dict[str, Dict],
        current_prices: Dict[str, float],
        initial_prices: Dict[str, float] = None
    ) -> Dict[str, ImpermanentLoss]:
        """
        計算無常損失

        Args:
            amm_positions: AMM頭寸信息
            current_prices: 當前價格
            initial_prices: 初始價格（如果沒有則使用當前頭寸的平均價格）

        Returns:
            各頭寸的無常損失計算結果
        """
        il_results = {}

        for pool_id, position_info in amm_positions.items():
            try:
                # 獲取頭寸信息
                token0_symbol = position_info.get('token0', 'TOKEN0')
                token1_symbol = position_info.get('token1', 'TOKEN1')
                liquidity = position_info.get('liquidity', 0)
                fee_growth = position_info.get('fee_growth', 0)

                # 獲取價格
                price0 = current_prices.get(token0_symbol, 0)
                price1 = current_prices.get(token1_symbol, 0)

                if price0 <= 0 or price1 <= 0:
                    continue

                # 計算當前價值
                current_value = self._calculate_amm_position_value(
                    liquidity, price0, price1, position_info
                )

                # 計算HODL價值（如果持有等額資產）
                if initial_prices:
                    initial_price0 = initial_prices.get(token0_symbol, price0)
                    initial_price1 = initial_prices.get(token1_symbol, price1)
                else:
                    # 使用頭寸平均價格作為初始價格
                    initial_price0 = position_info.get('avg_price0', price0)
                    initial_price1 = position_info.get('avg_price1', price1)

                hodl_value = self._calculate_hodl_value(
                    liquidity, initial_price0, initial_price1, position_info
                )

                # 計算AMM實際回報（包含手續費）
                amm_return = (current_value - hodl_value) / hodl_value if hodl_value > 0 else 0

                # 計算無常損失
                price_ratio = (price0 * price1) / (initial_price0 * initial_price1)

                if price_ratio > 0:
                    il_percentage = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
                    il_amount = hodl_value * abs(il_percentage)
                else:
                    il_percentage = 0
                    il_amount = 0

                il_results[pool_id] = ImpermanentLoss(
                    initial_value=hodl_value,
                    current_value=current_value,
                    hodl_return=(hodl_value - hodl_value) / hodl_value if hodl_value > 0 else 0,
                    amm_return=amm_return,
                    il_amount=il_amount,
                    il_percentage=il_percentage
                )

            except Exception as e:
                logger.error(f"Error calculating IL for {pool_id}: {e}")
                continue

        return il_results

    def _calculate_amm_position_value(
        self,
        liquidity: float,
        price0: float,
        price1: float,
        position_info: Dict
    ) -> float:
        """計算AMM頭寸當前價值"""
        try:
            # 簡化的AMM價值計算
            # 實際AMM價值取決於價格範圍和流動性分佈

            # 假設頭寸在當前價格附近
            sqrt_price = np.sqrt(price0 / price1)
            amount0 = liquidity * sqrt_price
            amount1 = liquidity / sqrt_price

            total_value = amount0 * price0 + amount1 * price1

            return total_value

        except Exception:
            return 0.0

    def _calculate_hodl_value(
        self,
        liquidity: float,
        price0: float,
        price1: float,
        position_info: Dict
    ) -> float:
        """計算HODL價值（作為比較基準）"""
        try:
            # 假設初始時等價值分配
            total_liquidity_value = liquidity  # 假設初始價值為1單位

            # 計算初始時的token數量（假設50/50分配）
            initial_sqrt_price = np.sqrt(price0 / price1)
            initial_amount0 = total_liquidity_value / (2 * initial_sqrt_price)
            initial_amount1 = total_liquidity_value / 2

            # 計算當前價值
            current_value = initial_amount0 * price0 + initial_amount1 * price1

            return current_value

        except Exception:
            return 0.0

    def generate_inventory_report(self) -> str:
        """生成庫存報告"""
        if not self.inventory_history:
            return "No inventory history available"

        latest_state = self.inventory_history[-1]

        report = []
        report.append("=" * 60)
        report.append("庫存管理報告")
        report.append("=" * 60)
        report.append(f"報告時間: {latest_state.timestamp}")
        report.append(f"總價值: ${latest_state.total_value:,.2f}")
        report.append(f"淨敞口: {latest_state.net_exposure".2%"}")
        report.append(f"集中度風險: {latest_state.concentration_risk".4f"}")
        report.append(f"流動性評分: {latest_state.liquidity_score".4f"}")
        report.append("")

        # 持倉詳情
        if latest_state.positions:
            report.append("持倉詳情:")
            for symbol, position in latest_state.positions.items():
                report.append(f"  {symbol}:")
                report.append(f"    數量: {position.quantity".6f"}")
                report.append(f"    平均價格: ${position.avg_price".2f"}")
                report.append(f"    市場價值: ${position.market_value",.2f"}")
                report.append(f"    未實現損益: ${position.unrealized_pnl",.2f"}")
                report.append(f"    實現損益: ${position.realized_pnl",.2f"}")
            report.append("")

        # 交易統計
        if self.trade_history:
            recent_trades = self.trade_history[-10:]  # 最近10筆交易
            report.append("最近交易:")
            for trade in recent_trades:
                report.append(f"  {trade['timestamp'].strftime('%Y-%m-%d %H:%M')}: "
                            f"{trade['symbol']} {trade['quantity']".6f"} @ ${trade['price']".2f"}")
            report.append("")

        # 風險評估
        risk_assessment = self._assess_portfolio_risk(latest_state)
        report.append("風險評估:")
        for risk_type, risk_level in risk_assessment.items():
            report.append(f"  {risk_type}: {risk_level}")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def _assess_portfolio_risk(self, state: InventoryState) -> Dict[str, str]:
        """評估投資組合風險"""
        risk_levels = {}

        # 敞口風險
        if state.net_exposure > 0.8:
            risk_levels['敞口風險'] = '高'
        elif state.net_exposure > 0.5:
            risk_levels['敞口風險'] = '中'
        else:
            risk_levels['敞口風險'] = '低'

        # 集中度風險
        if state.concentration_risk > 0.7:
            risk_levels['集中度風險'] = '高'
        elif state.concentration_risk > 0.4:
            risk_levels['集中度風險'] = '中'
        else:
            risk_levels['集中度風險'] = '低'

        # 流動性風險
        if state.liquidity_score < 0.3:
            risk_levels['流動性風險'] = '高'
        elif state.liquidity_score < 0.6:
            risk_levels['流動性風險'] = '中'
        else:
            risk_levels['流動性風險'] = '低'

        return risk_levels

    def get_position_summary(self) -> pd.DataFrame:
        """獲取持倉摘要"""
        if not self.positions:
            return pd.DataFrame()

        summary_data = []
        for symbol, position in self.positions.items():
            summary_data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'holding_period': (pd.Timestamp.now() - position.entry_time).days,
                'return_pct': (position.unrealized_pnl / (position.quantity * position.avg_price)) * 100 if position.quantity * position.avg_price != 0 else 0
            })

        return pd.DataFrame(summary_data)

    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """計算投資組合指標"""
        if not self.inventory_history:
            return {}

        # 獲取最新狀態
        latest = self.inventory_history[-1]

        # 計算回報率（與初始價值比較）
        if len(self.inventory_history) > 1:
            initial_value = self.inventory_history[0].total_value
            current_value = latest.total_value

            if initial_value > 0:
                total_return = (current_value - initial_value) / initial_value
            else:
                total_return = 0
        else:
            total_return = 0

        # 計算波動率（簡化）
        if len(self.inventory_history) > 10:
            values = [state.total_value for state in self.inventory_history[-20:]]
            returns = np.diff(values) / values[:-1]
            volatility = np.std(returns) * np.sqrt(365)  # 年化波動率
        else:
            volatility = 0

        # 計算最大回撤
        max_drawdown = self._calculate_max_drawdown()

        # 計算夏普比率（簡化）
        if volatility > 0:
            sharpe_ratio = total_return / volatility
        else:
            sharpe_ratio = 0

        return {
            'total_value': latest.total_value,
            'total_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'position_count': len(latest.positions),
            'net_exposure': latest.net_exposure,
            'concentration_risk': latest.concentration_risk
        }

    def _calculate_max_drawdown(self) -> float:
        """計算最大回撤"""
        if len(self.inventory_history) < 2:
            return 0.0

        values = [state.total_value for state in self.inventory_history]
        peak = values[0]
        max_drawdown = 0

        for value in values:
            if value > peak:
                peak = value
            if peak > 0:
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return max_drawdown

    def export_inventory_history(self, filepath: str) -> str:
        """導出庫存歷史"""
        try:
            # 準備導出數據
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'positions': {symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'entry_time': pos.entry_time.isoformat(),
                    'last_update': pos.last_update.isoformat()
                } for symbol, pos in self.positions.items()},
                'trade_history': self.trade_history[-100:],  # 最近100筆交易
                'inventory_snapshots': [
                    {
                        'timestamp': state.timestamp.isoformat(),
                        'total_value': state.total_value,
                        'net_exposure': state.net_exposure,
                        'concentration_risk': state.concentration_risk,
                        'liquidity_score': state.liquidity_score,
                        'position_count': len(state.positions)
                    }
                    for state in self.inventory_history[-50:]  # 最近50個快照
                ]
            }

            df = pd.DataFrame(export_data['inventory_snapshots'])
            df.to_csv(filepath, index=False)

            logger.info(f"Exported inventory history to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting inventory history: {e}")
            raise


# 使用範例
if __name__ == "__main__":
    # 創建庫存管理器
    inventory_manager = InventoryManager(
        max_position_size=50000,
        max_portfolio_risk=0.7,
        rebalance_threshold=0.1
    )

    # 模擬交易
    current_time = pd.Timestamp.now()

    # 買入BTC
    btc_position = inventory_manager.update_position(
        symbol='BTC',
        quantity=1.0,
        price=50000,
        timestamp=current_time,
        trade_type='buy'
    )

    print(f"BTC頭寸: 數量={btc_position.quantity}, 平均價格=${btc_position.avg_price:.2f}")

    # 買入ETH
    eth_position = inventory_manager.update_position(
        symbol='ETH',
        quantity=10.0,
        price=3000,
        timestamp=current_time,
        trade_type='buy'
    )

    print(f"ETH頭寸: 數量={eth_position.quantity}, 平均價格=${eth_position.avg_price:.2f}")

    # 獲取當前價格（模擬）
    current_prices = {'BTC': 51000, 'ETH': 3100}

    # 獲取庫存狀態
    inventory_state = inventory_manager.get_inventory_state(current_prices)

    print(f"總價值: ${inventory_state.total_value:,.2f}")
    print(f"淨敞口: {inventory_state.net_exposure:.2%}")
    print(f"集中度風險: {inventory_state.concentration_risk:.4f}")

    # 生成報告
    report = inventory_manager.generate_inventory_report()
    print("\n庫存報告:")
    print(report)

    # 計算投資組合指標
    metrics = inventory_manager.calculate_portfolio_metrics()
    print("
投資組合指標:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
