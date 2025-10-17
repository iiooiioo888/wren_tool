"""Pairs trading PoC
- 讀取 CSV (timestamp, price_a, price_b)
- 計算 hedge ratio via OLS (regress price_a on price_b)
- 計算spread = price_a - beta * price_b，使用 rolling mean/std 計算 zscore
- 當 zscore > entry_z -> 做空 spread (short A, long B);
  當 zscore < -entry_z -> 做多 spread (long A, short B)
- 退出條件 zscore 回歸到 0 或達到 stop_z
- 簡化假設：以現金分配給多/空各半，按當日 close 價成交
"""
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import json


def load_pair_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]) 
    required = ["timestamp", "price_a", "price_b"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV 必須包含欄位: {required}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def hedge_ratio_ols(y, x):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    beta = model.params[1]
    return float(beta)


def compute_zscore(df: pd.DataFrame, beta: float, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df['spread'] = df['price_a'] - beta * df['price_b']
    df['spread_mean'] = df['spread'].rolling(window=window, min_periods=1).mean()
    df['spread_std'] = df['spread'].rolling(window=window, min_periods=1).std().replace(0, 1e-8)
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    return df


def simulate_pairs(df: pd.DataFrame, entry_z=2.0, exit_z=0.2, fee=0.0005, slippage=0.0, initial_cash=10000.0, max_holding_bars: int = None, stop_z: float = None):
    # hedge ratio estimated on full sample for PoC simplicity
    beta = hedge_ratio_ols(df['price_a'], df['price_b'])
    df = compute_zscore(df, beta)

    cash = initial_cash
    pos_a = 0.0
    pos_b = 0.0
    trades = []
    in_trade = False

    holding_bars = 0
    for i, row in df.iterrows():
        z = row['zscore']
        price_a = row['price_a']
        price_b = row['price_b']
        if not in_trade:
            if z > entry_z:
                # short spread: short A, long B
                # allocate half cash to each leg
                cash_per_leg = cash / 2
                qty_a = cash_per_leg / (price_a * (1 + slippage))
                qty_b = cash_per_leg / (price_b * (1 + slippage))
                # for short A, we simulate borrowed sell: receive proceeds minus fee
                proceeds_a = qty_a * price_a * (1 - fee)
                cost_b = qty_b * price_b * (1 + fee)
                cash = cash + proceeds_a - cost_b
                pos_a = -qty_a
                pos_b = qty_b
                trades.append({'time': str(row['timestamp']), 'side': 'enter_short_spread', 'beta': beta, 'qty_a': -qty_a, 'qty_b': qty_b})
                in_trade = True
                holding_bars = 0
            elif z < -entry_z:
                # long spread: long A, short B
                cash_per_leg = cash / 2
                qty_a = cash_per_leg / (price_a * (1 + slippage))
                qty_b = cash_per_leg / (price_b * (1 + slippage))
                cost_a = qty_a * price_a * (1 + fee)
                proceeds_b = qty_b * price_b * (1 - fee)
                cash = cash - cost_a + proceeds_b
                pos_a = qty_a
                pos_b = -qty_b
                trades.append({'time': str(row['timestamp']), 'side': 'enter_long_spread', 'beta': beta, 'qty_a': qty_a, 'qty_b': -qty_b})
                in_trade = True
                holding_bars = 0
        else:
            # check exit
            holding_bars += 1
            # stoploss if specified
            if stop_z is not None and abs(z) > stop_z:
                # force exit at loss
                value_a = pos_a * price_a * (1 - fee if pos_a > 0 else 1 - fee)
                value_b = pos_b * price_b * (1 - fee if pos_b > 0 else 1 - fee)
                cash += value_a + value_b
                trades.append({'time': str(row['timestamp']), 'side': 'stop_loss_exit', 'price_a': price_a, 'price_b': price_b, 'pos_a': pos_a, 'pos_b': pos_b, 'z': z})
                pos_a = 0
                pos_b = 0
                in_trade = False
                holding_bars = 0
                continue
            if max_holding_bars is not None and holding_bars >= max_holding_bars:
                # force exit due to max holding
                value_a = pos_a * price_a * (1 - fee if pos_a > 0 else 1 - fee)
                value_b = pos_b * price_b * (1 - fee if pos_b > 0 else 1 - fee)
                cash += value_a + value_b
                trades.append({'time': str(row['timestamp']), 'side': 'max_holding_exit', 'price_a': price_a, 'price_b': price_b, 'pos_a': pos_a, 'pos_b': pos_b, 'holding_bars': holding_bars})
                pos_a = 0
                pos_b = 0
                in_trade = False
                holding_bars = 0
                continue
            if abs(z) < exit_z:
                # exit positions at current prices
                value_a = pos_a * price_a * (1 - fee if pos_a > 0 else 1 - fee)
                value_b = pos_b * price_b * (1 - fee if pos_b > 0 else 1 - fee)
                cash += value_a + value_b
                trades.append({'time': str(row['timestamp']), 'side': 'exit', 'price_a': price_a, 'price_b': price_b, 'pos_a': pos_a, 'pos_b': pos_b})
                pos_a = 0
                pos_b = 0
                in_trade = False
    # mark-to-market at last prices
    last = df.iloc[-1]
    mtm = cash + pos_a * last['price_a'] + pos_b * last['price_b']
    result = {
        'beta': beta,
        'cash': float(cash),
        'pos_a': float(pos_a),
        'pos_b': float(pos_b),
        'mtm': float(mtm),
        'pnl': float(mtm - initial_cash),
        'trades': trades,
    }
    return result


def run_pairs_poc(csv_path: str, out_path: str = None, entry_z=2.0, exit_z=0.2, fee=0.0005, slippage=0.0, initial_cash=10000.0, max_holding_bars: int = None, stop_z: float = None):
    df = load_pair_data(csv_path)
    res = simulate_pairs(df, entry_z=entry_z, exit_z=exit_z, fee=fee, slippage=slippage, initial_cash=initial_cash, max_holding_bars=max_holding_bars, stop_z=stop_z)
    out = Path(out_path) if out_path else Path('e:/Jerry_python/wren_tool/out/pairs_poc_results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"Pairs PoC finished. mtm={res['mtm']}, pnl={res['pnl']}")
    return res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='e:/Jerry_python/wren_tool/data/sample_pair.csv')
    parser.add_argument('--out', default='e:/Jerry_python/wren_tool/out/pairs_poc_results.json')
    parser.add_argument('--entry-z', type=float, default=2.0)
    parser.add_argument('--exit-z', type=float, default=0.2)
    parser.add_argument('--fee', type=float, default=0.0005)
    parser.add_argument('--slippage', type=float, default=0.0)
    parser.add_argument('--initial-cash', type=float, default=10000.0)
    parser.add_argument('--max-holding-bars', type=int, default=None)
    parser.add_argument('--stop-z', type=float, default=None)
    args = parser.parse_args()
    run_pairs_poc(args.csv, args.out, entry_z=args.entry_z, exit_z=args.exit_z, fee=args.fee, slippage=args.slippage, initial_cash=args.initial_cash, max_holding_bars=args.max_holding_bars, stop_z=args.stop_z)
