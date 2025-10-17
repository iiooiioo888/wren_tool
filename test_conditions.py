#!/usr/bin/env python3
"""測試條件檢查問題"""

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

def test_conditions():
    df = load_data("data/sample_ohlc.csv")
    df = generate_sma_signals(df)

    cash = 10000.0
    fee = 0.0005

    print(f"開始條件測試，現金=${cash}")

    for i, row in df.iterrows():
        signal_change = row.get("signal_change", 0)

        if signal_change == 2:  # 買入信號
            current_price = row["close"]
            execution_price = current_price

            # 計算買入數量和成本（扣除手續費後的可用資金）
            available_cash = cash / (1 + fee)  # 預留手續費
            qty = available_cash / execution_price
            fee_amount = qty * execution_price * fee
            total_cost = qty * execution_price + fee_amount

            print(f"第{i}行買入測試:")
            print(f"  價格={current_price}, 執行價格={execution_price}")
            print(f"  可用資金=${available_cash:.6f}")
            print(f"  數量={qty:.6f}")
            print(f"  手續費=${fee_amount:.6f}")
            print(f"  總成本=${total_cost:.6f}")
            print(f"  現金=${cash:.6f}")
            print(f"  條件1 (total_cost <= cash): {total_cost} <= {cash} = {total_cost <= cash}")
            print(f"  條件2 (qty > 0): {qty} > 0 = {qty > 0}")
            print(f"  條件3 (total_cost < cash * 1.0001): {total_cost} < {cash * 1.0001} = {total_cost < cash * 1.0001}")

            # 測試不同條件
            condition1 = total_cost <= cash and qty > 0
            condition2 = total_cost < cash * 1.0001 and qty > 0

            print(f"  原條件結果: {condition1}")
            print(f"  寬鬆條件結果: {condition2}")

            if condition2:
                print("  -> 交易會被執行")
            else:
                print("  -> 交易不會被執行")
            print()

def main():
    test_conditions()

if __name__ == "__main__":
    main()
