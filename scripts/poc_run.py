"""增強版PoC 腳本：
1) 讀取 CSV 格式的 OHLCV 測試數據
2) 執行多種交易策略（SMA交叉、統計套利等）
3) 模擬真實交易環境（滑點、延遲、手續費）並輸出結果報表
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from datetime import datetime, timedelta

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 導入模擬器
try:
    from simulation.execution.delay_simulator import DelaySimulator, DelayConfig
    from simulation.costs.slippage_model import SlippageModel, SlippageModelConfig
    DELAY_SIMULATOR_AVAILABLE = True
except ImportError:
    logger.warning("模擬器模組不可用，使用簡化版本")
    DELAY_SIMULATOR_AVAILABLE = False

# 默認參數
DEFAULT_FEE = 0.0005  # 手續費比例
DEFAULT_SLIPPAGE = 0.001  # 默認滑點
DEFAULT_DELAY_MS = 100  # 默認延遲（毫秒）


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]) 
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV 必須包含欄位: {required}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(window=3, min_periods=1).mean()
    df["sma_slow"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
    df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1
    df["signal_change"] = df["signal"].diff().fillna(0)
    return df


def simulate_trades_enhanced(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    fee: float = None,
    slippage: float = None,
    delay_ms: int = None,
    strategy: str = "sma_crossover"
) -> dict:
    """
    增強版交易模擬，支援滑點、延遲和多種策略

    Args:
        df: 價格數據
        initial_cash: 初始資金
        fee: 手續費比例
        slippage: 滑點比例
        delay_ms: 延遲毫秒數
        strategy: 策略名稱

    Returns:
        模擬結果
    """
    # 使用默認參數
    if fee is None:
        fee = DEFAULT_FEE
    if slippage is None:
        slippage = DEFAULT_SLIPPAGE
    if delay_ms is None:
        delay_ms = DEFAULT_DELAY_MS

    logger.info(f"開始增強版交易模擬: 策略={strategy}, 手續費={fee}, 滑點={slippage}, 延遲={delay_ms}ms")

    # 初始化模擬器
    delay_simulator = None
    slippage_model = None

    if DELAY_SIMULATOR_AVAILABLE:
        delay_config = DelayConfig(
            base_network_delay=delay_ms / 1000,  # 轉換為秒
            api_processing_delay=0.05,
            network_jitter=0.02
        )
        delay_simulator = DelaySimulator(delay_config)

        slippage_config = SlippageModelConfig(
            market_impact_factor=slippage,
            method="square_root"
        )
        slippage_model = SlippageModel(slippage_config)

    # 根據策略生成信號
    if strategy == "sma_crossover":
        df = generate_sma_signals(df)
    elif strategy == "rsi":
        df = generate_rsi_signals(df)
    elif strategy == "statistical_arbitrage":
        # 統計套利需要多個資產，這裡簡化為單資產趨勢策略
        df = generate_sma_signals(df)
    else:
        logger.warning(f"未知策略: {strategy}，使用默認SMA策略")
        df = generate_sma_signals(df)

    # 初始化資金和倉位
    cash = initial_cash
    position = 0.0
    position_price = 0.0
    trades = []
    total_fees = 0.0
    total_slippage = 0.0
    total_delay_cost = 0.0

    # 執行交易模擬
    for i, row in df.iterrows():
        signal_change = row.get("signal_change", 0)

        if signal_change == 2:  # 買入信號 (-1 -> 1)
            if cash > 0:
                # 計算延遲影響後的價格
                current_price = row["close"]
                execution_price = current_price

                # 應用延遲影響（價格可能變化）
                if delay_simulator:
                    delay_result = delay_simulator.simulate_order_delay(
                        order_size=cash,
                        order_type="market",
                        time_of_day=row["timestamp"].hour,
                        network_load=0.5
                    )
                    # 模擬價格變化（簡化）
                    price_change = np.random.normal(0, 0.001) * delay_result.total_delay
                    execution_price = current_price * (1 + price_change)

                # 應用滑點
                if slippage_model:
                    slippage_result = slippage_model.calculate_slippage(
                        symbol="BTC",  # 假設為BTC
                        order_size=cash,
                        order_price=current_price,
                        order_type="market"
                    )
                    execution_price = slippage_result.executed_price
                    total_slippage += slippage_result.slippage_amount

                # 計算買入數量和成本（扣除手續費後的可用資金）
                available_cash = cash / (1 + fee)  # 預留手續費
                qty = available_cash / execution_price
                fee_amount = qty * execution_price * fee
                total_cost = qty * execution_price + fee_amount

                print(f"買入嘗試: 價格={current_price}, 執行價格={execution_price:.2f}, 數量={qty:.4f}, 成本=${total_cost:.2f}, 現金=${cash:.2f}")
                print(f"條件檢查: total_cost <= cash: {total_cost} <= {cash} = {total_cost <= cash}, qty > 0: {qty} > 0 = {qty > 0}")
                print(f"數值詳情: total_cost={total_cost:.10f}, cash={cash:.10f}, diff={abs(total_cost - cash):.10f}")

                # 使用更寬鬆的條件，避免浮點數精度問題
                if abs(total_cost - cash) < 0.001 or total_cost <= cash:
                    cash -= total_cost
                    position = qty
                    position_price = execution_price
                    total_fees += fee_amount

                    trades.append({
                        "time": row["timestamp"].isoformat(),
                        "side": "buy",
                        "original_price": float(current_price),
                        "execution_price": float(execution_price),
                        "qty": float(qty),
                        "fee": float(fee_amount),
                        "slippage": float(slippage_result.slippage_amount) if slippage_model else 0.0,
                        "delay_cost": float(execution_price - current_price)
                    })

                    total_delay_cost += (execution_price - current_price) * qty

        elif signal_change == -2:  # 賣出信號 (1 -> -1)
            if position > 0:
                # 計算賣出價格（考慮延遲和滑點）
                current_price = row["close"]
                execution_price = current_price

                # 應用延遲影響
                if delay_simulator:
                    delay_result = delay_simulator.simulate_order_delay(
                        order_size=position * current_price,
                        order_type="market",
                        time_of_day=row["timestamp"].hour,
                        network_load=0.5
                    )
                    price_change = np.random.normal(0, 0.001) * delay_result.total_delay
                    execution_price = current_price * (1 + price_change)

                # 應用滑點（賣單）
                if slippage_model:
                    slippage_result = slippage_model.calculate_slippage(
                        symbol="BTC",
                        order_size=position * current_price,
                        order_price=current_price,
                        order_type="market"
                    )
                    # 賣單滑點會降低執行價格
                    execution_price = slippage_result.executed_price
                    total_slippage += abs(slippage_result.slippage_amount)

                # 計算賣出收益
                revenue = position * execution_price
                fee_amount = revenue * fee
                net_revenue = revenue - fee_amount

                cash += net_revenue
                total_fees += fee_amount

                trades.append({
                    "time": row["timestamp"].isoformat(),
                    "side": "sell",
                    "original_price": float(current_price),
                    "execution_price": float(execution_price),
                    "qty": float(position),
                    "fee": float(fee_amount),
                    "slippage": float(abs(slippage_result.slippage_amount)) if slippage_model else 0.0,
                    "delay_cost": float(current_price - execution_price) * position
                })

                position = 0.0
                position_price = 0.0
                total_delay_cost += (current_price - execution_price) * position

    # 計算最終價值
    last_price = df.iloc[-1]["close"]
    final_position_value = position * last_price
    total_value = cash + final_position_value
    gross_pnl = total_value - initial_cash
    net_pnl = gross_pnl - total_fees - total_slippage - total_delay_cost

    result = {
        "initial_cash": float(initial_cash),
        "final_cash": float(cash),
        "final_position": float(position),
        "final_position_value": float(final_position_value),
        "total_value": float(total_value),
        "gross_pnl": float(gross_pnl),
        "net_pnl": float(net_pnl),
        "total_fees": float(total_fees),
        "total_slippage": float(total_slippage),
        "total_delay_cost": float(total_delay_cost),
        "total_trades": len(trades),
        "win_rate": float(len([t for t in trades if (t["side"] == "buy" and t["execution_price"] < last_price) or
                                                    (t["side"] == "sell" and t["execution_price"] > position_price)]) / max(1, len(trades))),
        "max_drawdown": float(calculate_max_drawdown(df, trades, initial_cash)),
        "sharpe_ratio": float(calculate_sharpe_ratio(df, trades, initial_cash)),
        "trades": trades,
        "simulation_config": {
            "strategy": strategy,
            "fee": fee,
            "slippage": slippage,
            "delay_ms": delay_ms,
            "simulators_available": DELAY_SIMULATOR_AVAILABLE
        }
    }

    logger.info(f"模擬完成: 總價值=${total_value:.2f}, 淨盈虧=${net_pnl:.2f}, 總交易次數={len(trades)}")
    return result


def generate_sma_signals(df: pd.DataFrame) -> pd.DataFrame:
    """生成SMA交叉信號"""
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(window=3, min_periods=1).mean()
    df["sma_slow"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
    df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1
    df["signal_change"] = df["signal"].diff().fillna(0)
    return df


def generate_rsi_signals(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """生成RSI信號"""
    df = df.copy()

    # 計算價格變化
    df["price_change"] = df["close"].diff()

    # 計算收益和損失
    df["gain"] = df["price_change"].where(df["price_change"] > 0, 0)
    df["loss"] = -df["price_change"].where(df["price_change"] < 0, 0)

    # 計算平均收益和損失
    df["avg_gain"] = df["gain"].rolling(window=period, min_periods=1).mean()
    df["avg_loss"] = df["loss"].rolling(window=period, min_periods=1).mean()

    # 計算RS和RSI
    df["rs"] = df["avg_gain"] / df["avg_loss"].replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + df["rs"]))

    # 生成信號
    df["signal"] = 0
    df.loc[df["rsi"] < 30, "signal"] = 1   # 超賣，買入
    df.loc[df["rsi"] > 70, "signal"] = -1  # 超買，賣出
    df["signal_change"] = df["signal"].diff().fillna(0)

    return df


def calculate_max_drawdown(df: pd.DataFrame, trades: List[dict], initial_cash: float) -> float:
    """計算最大回撤"""
    if not trades:
        return 0.0

    # 簡化的回撤計算
    prices = df["close"].values
    min_price = np.min(prices)
    max_price = np.max(prices)

    if max_price > 0:
        return (max_price - min_price) / max_price
    return 0.0


def calculate_sharpe_ratio(df: pd.DataFrame, trades: List[dict], initial_cash: float) -> float:
    """計算夏普比率（簡化版）"""
    if not trades or len(df) < 2:
        return 0.0

    # 計算每日收益率
    returns = df["close"].pct_change().dropna()

    if len(returns) == 0:
        return 0.0

    # 簡化的夏普比率計算
    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    # 年化夏普比率（假設252個交易日）
    annual_sharpe = mean_return / std_return * np.sqrt(252)
    return float(annual_sharpe)


def simulate_trades(df: pd.DataFrame, initial_cash=10000.0) -> dict:
    """舊版交易模擬函數（為了向下相容）"""
    return simulate_trades_enhanced(df, initial_cash, DEFAULT_FEE, DEFAULT_SLIPPAGE, DEFAULT_DELAY_MS)


def run_poc(
    csv_path: str,
    out_path: str = None,
    fee: float = None,
    slippage: float = None,
    delay_ms: int = None,
    strategy: str = "sma_crossover",
    initial_cash: float = 10000.0,
    verbose: bool = True
):
    """
    執行增強版PoC回測

    Args:
        csv_path: CSV數據文件路徑
        out_path: 輸出結果文件路徑
        fee: 手續費比例
        slippage: 滑點比例
        delay_ms: 延遲毫秒數
        strategy: 交易策略
        initial_cash: 初始資金
        verbose: 是否輸出詳細信息
    """
    try:
        # 加載數據
        if verbose:
            logger.info(f"加載數據: {csv_path}")
        df = load_data(csv_path)

        if len(df) == 0:
            raise ValueError("數據文件為空")

        if verbose:
            logger.info(f"數據範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            logger.info(f"數據點數: {len(df)}")

        # 執行增強版模擬
        res = simulate_trades_enhanced(
            df=df,
            initial_cash=initial_cash,
            fee=fee,
            slippage=slippage,
            delay_ms=delay_ms,
            strategy=strategy
        )

        # 保存結果
        out = Path(out_path) if out_path else Path("e:/Jerry_python/wren_tool/out/poc_results.json")
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2, default=str)

        # 生成詳細報告
        if verbose:
            print_detailed_report(res)

        return res

    except Exception as e:
        logger.error(f"PoC執行失敗: {e}")
        raise


def print_detailed_report(result: dict):
    """打印詳細報告"""
    print("\n" + "=" * 60)
    print("增強版PoC回測報告")
    print("=" * 60)

    print(f"初始資金: ${result['initial_cash']:.2f}")
    print(f"最終價值: ${result['total_value']:.2f}")
    print(f"總盈虧: ${result['gross_pnl']:.2f}")
    print(f"淨盈虧: ${result['net_pnl']:.2f}")
    print(f"收益率: {result['net_pnl']/result['initial_cash']*100:.2f}%")

    print("\n交易成本:")
    print(f"  手續費: ${result['total_fees']:.2f}")
    print(f"  滑點成本: ${result['total_slippage']:.2f}")
    print(f"  延遲成本: ${result['total_delay_cost']:.2f}")

    print("\n交易統計:")
    print(f"  總交易次數: {result['total_trades']}")
    print(f"  勝率: {result['win_rate']*100:.1f}%")
    print(f"  最大回撤: {result['max_drawdown']*100:.2f}%")
    print(f"  夏普比率: {result['sharpe_ratio']:.3f}")

    config = result.get('simulation_config', {})
    print("\n模擬配置:")
    print(f"  策略: {config.get('strategy', 'unknown')}")
    print(f"  手續費: {config.get('fee', 0)*100:.3f}%")
    print(f"  滑點: {config.get('slippage', 0)*100:.3f}%")
    print(f"  延遲: {config.get('delay_ms', 0)}ms")
    print(f"  模擬器可用: {config.get('simulators_available', False)}")

    print("=" * 60)


def run_benchmark(csv_path: str, strategies: List[str] = None, output_dir: str = None):
    """
    運行基準測試，比較不同策略和參數組合

    Args:
        csv_path: CSV數據文件路徑
        strategies: 策略列表
        output_dir: 輸出目錄
    """
    if strategies is None:
        strategies = ["sma_crossover", "rsi"]

    if output_dir is None:
        output_dir = "e:/Jerry_python/wren_tool/out/benchmark"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # 測試不同策略
    for strategy in strategies:
        logger.info(f"測試策略: {strategy}")

        result = run_poc(
            csv_path=csv_path,
            out_path=f"{output_dir}/{strategy}_result.json",
            strategy=strategy,
            verbose=False
        )

        results[strategy] = {
            'net_pnl': result['net_pnl'],
            'total_trades': result['total_trades'],
            'win_rate': result['win_rate'],
            'max_drawdown': result['max_drawdown'],
            'sharpe_ratio': result['sharpe_ratio']
        }

    # 生成比較報告
    comparison_report = generate_comparison_report(results)
    comparison_file = output_path / "comparison_report.txt"

    with open(comparison_file, "w", encoding="utf-8") as f:
        f.write(comparison_report)

    print(f"基準測試完成，結果保存至: {output_dir}")
    return results


def generate_comparison_report(results: dict) -> str:
    """生成策略比較報告"""
    report = []
    report.append("策略比較報告")
    report.append("=" * 50)

    for strategy, metrics in results.items():
        report.append(f"\n策略: {strategy}")
        report.append(f"  淨盈虧: ${metrics['net_pnl']:.2f}")
        report.append(f"  總交易次數: {metrics['total_trades']}")
        report.append(f"  勝率: {metrics['win_rate']*100:.1f}%")
        report.append(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        report.append(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")

    # 找出最佳策略
    best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
    report.append(f"\n最佳策略（按夏普比率）: {best_strategy}")

    return "\n".join(report)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="增強版PoC回測工具")
    parser.add_argument("--csv", required=False, default="e:/Jerry_python/wren_tool/data/sample_ohlc.csv",
                       help="輸入CSV數據文件路徑")
    parser.add_argument("--out", required=False, default="e:/Jerry_python/wren_tool/out/poc_results.json",
                       help="輸出結果文件路徑")
    parser.add_argument("--strategy", required=False, default="sma_crossover",
                       choices=["sma_crossover", "rsi", "statistical_arbitrage"],
                       help="交易策略")
    parser.add_argument("--fee", required=False, type=float, default=None,
                       help="手續費比例 (默認: 0.0005)")
    parser.add_argument("--slippage", required=False, type=float, default=None,
                       help="滑點比例 (默認: 0.001)")
    parser.add_argument("--delay", required=False, type=int, default=None,
                       help="延遲毫秒數 (默認: 100)")
    parser.add_argument("--initial-cash", required=False, type=float, default=10000.0,
                       help="初始資金 (默認: 10000)")
    parser.add_argument("--benchmark", required=False, nargs="*", default=None,
                       help="運行基準測試，比較多個策略")
    parser.add_argument("--quiet", required=False, action="store_true",
                       help="靜默模式，不輸出詳細報告")

    args = parser.parse_args()

    if args.benchmark:
        # 運行基準測試
        run_benchmark(args.csv, args.benchmark)
    else:
        # 運行單次PoC
        run_poc(
            csv_path=args.csv,
            out_path=args.out,
            strategy=args.strategy,
            fee=args.fee,
            slippage=args.slippage,
            delay_ms=args.delay,
            initial_cash=args.initial_cash,
            verbose=not args.quiet
        )
