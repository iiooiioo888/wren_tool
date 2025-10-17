#!/usr/bin/env python3
"""調試腳本：檢查信號生成邏輯"""

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

def main():
    # 加載數據
    df = load_data("data/sample_ohlc.csv")

    print("原始數據:")
    print(df.head(10))
    print()

    # 生成信號
    df_with_signals = generate_sma_signals(df)

    print("信號數據:")
    print(df_with_signals[['timestamp', 'close', 'sma_fast', 'sma_slow', 'signal', 'signal_change']].head(15))
    print()

    # 檢查信號變化
    signal_changes = df_with_signals[df_with_signals['signal_change'] != 0]
    print(f"信號變化點數: {len(signal_changes)}")
    if len(signal_changes) > 0:
        print("信號變化詳情:")
        print(signal_changes[['timestamp', 'close', 'sma_fast', 'sma_slow', 'signal', 'signal_change']])
    else:
        print("沒有發現信號變化")

    # 檢查SMA計算是否正確
    print("\n檢查SMA計算:")
    for i in range(min(10, len(df))):
        if i >= 2:  # 至少需要3個點計算SMA快線
            sma_fast_val = df.iloc[i-2:i+1]['close'].mean()
            print(f"第{i}行: 價格={df.iloc[i]['close']}, SMA快線(3)={sma_fast_val}")

        if i >= 4:  # 至少需要5個點計算SMA慢線
            sma_slow_val = df.iloc[i-4:i+1]['close'].mean()
            print(f"第{i}行: 價格={df.iloc[i]['close']}, SMA慢線(5)={sma_slow_val}")

if __name__ == "__main__":
    main()
