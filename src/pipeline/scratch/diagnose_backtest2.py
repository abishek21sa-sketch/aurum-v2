import numpy as np
import pandas as pd
from src.agents.backtester import fetch_price_data, build_composite_signal, SP500_UNIVERSE
from src.models.hypothesis import Hypothesis
from src.core.database import SessionLocal

def main():
    db = SessionLocal()
    hyp = db.query(Hypothesis).order_by(Hypothesis.hypothesis_number.desc()).first()
    print(f"Using Hypothesis #{hyp.hypothesis_number}: {hyp.title}\n")

    close, volume = fetch_price_data(SP500_UNIVERSE, "2018-01-01", "2026-06-29")
    composite = build_composite_signal(close, volume, hyp.signal_components)

    n_long = max(1, int(len(composite) * 0.2))
    long_tickers = composite.nlargest(n_long).index.tolist()
    print(f"Top quintile tickers: {long_tickers}\n")

    available = [t for t in long_tickers if t in close.columns]
    portfolio_close = close[available]

    fwd_returns = portfolio_close.pct_change(21).dropna()
    print(f"Forward returns shape: {fwd_returns.shape}")
    print(f"Forward returns describe:\n{fwd_returns.describe()}\n")

    portfolio_returns = fwd_returns.mean(axis=1)
    print(f"Portfolio returns min: {portfolio_returns.min()}")
    print(f"Portfolio returns max: {portfolio_returns.max()}")
    print(f"Worst 5 portfolio return days:\n{portfolio_returns.nsmallest(5)}\n")

    cumulative = (1 + portfolio_returns).cumprod()
    print(f"Cumulative min: {cumulative.min()}")
    print(f"Cumulative at worst point:\n{cumulative.nsmallest(5)}")

    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    print(f"\nWorst drawdown: {drawdowns.min()}")
    print(f"Worst drawdown date: {drawdowns.idxmin()}")
    print(f"Cumulative value at that date: {cumulative.loc[drawdowns.idxmin()]}")
    print(f"Rolling max at that date: {rolling_max.loc[drawdowns.idxmin()]}")

if __name__ == "__main__":
    main()