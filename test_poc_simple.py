#!/usr/bin/env python3
"""簡單的PoC測試腳本"""

import pandas as pd
import numpy as np

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV 必須包含欄位: {required}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

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

def simulate_trades_simple(df: pd.DataFrame, initial_cash: float = 10000.0) -> dict:
    """簡化的交易模擬"""
    df = generate_sma_signals(df)

    cash = initial_cash
    position = 0.0
    trades = []

    print(f"開始模擬，數據點數: {len(df)}")

    for i, row in df.iterrows():
        signal_change = row.get("signal_change", 0)

        if signal_change == 2:  # 買入信號
            if cash > 0:
                current_price = row["close"]
                qty = cash / current_price
                cost = qty * current_price

                print(f"買入: 價格={current_price}, 數量={qty:.4f}, 成本=${cost:.2f}")

                if cost <= cash:
                    cash -= cost
                    position += qty
                    trades.append({
                        "time": row["timestamp"].isoformat(),
                        "side": "buy",
                        "price": float(current_price),
                        "qty": float(qty)
                    })

        elif signal_change == -2:  # 賣出信號
            if position > 0:
                current_price = row["close"]
                revenue = position * current_price

                print(f"賣出: 價格={current_price}, 數量={position:.4f}, 收益=${revenue:.2f}")

                cash += revenue
                trades.append({
                    "time": row["timestamp"].isoformat(),
                    "side": "sell",
                    "price": float(current_price),
                    "qty": float(position)
                })
                position = 0.0

    # 計算最終價值
    last_price = df.iloc[-1]["close"]
    final_position_value = position * last_price
    total_value = cash + final_position_value

    result = {
        "initial_cash": float(initial_cash),
        "final_cash": float(cash),
        "final_position": float(position),
        "total_value": float(total_value),
        "total_trades": len(trades),
        "trades": trades
    }

    return result

def main():
    df = load_data("data/sample_ohlc.csv")
    result = simulate_trades_simple(df)

    print("\n模擬結果:")
    print(f"初始資金: ${result['initial_cash']}")
    print(f"最終價值: ${result['total_value']}")
    print(f"總交易次數: {result['total_trades']}")

if __name__ == "__main__":
    main()
