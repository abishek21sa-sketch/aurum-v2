import numpy as np
import pandas as pd
from src.agents.backtester import fetch_price_data, SP500_UNIVERSE

def main():
    print("Fetching data...")
    close, volume = fetch_price_data(SP500_UNIVERSE, "2018-01-01", "2026-06-29")
    print(f"Shape: {close.shape}\n")

    # 1. Check for NaNs remaining per ticker
    print("── NaN counts per ticker ──────────────")
    nan_counts = close.isna().sum()
    bad_nan = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if len(bad_nan) > 0:
        print(bad_nan)
    else:
        print("No NaNs remain after ffill.")

    # 2. Check for near-zero or negative prices (data errors)
    print("\n── Min price per ticker (flag anything < $1) ──")
    min_prices = close.min()
    suspicious = min_prices[min_prices < 1.0].sort_values()
    if len(suspicious) > 0:
        print(suspicious)
    else:
        print("No suspiciously low prices.")

    # 3. Check for extreme single-day returns (splits, bad ticks)
    print("\n── Extreme daily returns (>50% single day move) ──")
    daily_returns = close.pct_change()
    extreme = daily_returns[(daily_returns.abs() > 0.5)]
    extreme_flat = extreme.stack()
    if len(extreme_flat) > 0:
        print(extreme_flat.sort_values(ascending=False))
    else:
        print("No extreme single-day moves found.")

    # 4. Check the actual holding-period returns used in backtest
    print("\n── Extreme 21-day forward returns (>200% or <-90%) ──")
    fwd_returns = close.pct_change(21)
    extreme_fwd = fwd_returns[(fwd_returns > 2.0) | (fwd_returns < -0.9)]
    extreme_fwd_flat = extreme_fwd.stack()
    if len(extreme_fwd_flat) > 0:
        print(extreme_fwd_flat.sort_values())
    else:
        print("No extreme 21-day returns found.")

    # 5. Ticker-level date ranges (catch IPOs / delistings mid-window)
    print("\n── First and last valid date per ticker ──")
    for ticker in close.columns:
        series = close[ticker].dropna()
        if len(series) == 0:
            print(f"  {ticker}: NO DATA AT ALL")
            continue
        first, last = series.index[0], series.index[-1]
        if first > pd.Timestamp("2018-02-01") or last < pd.Timestamp("2026-06-01"):
            print(f"  {ticker}: {first.date()} → {last.date()}  (partial coverage)")

if __name__ == "__main__":
    main()