from src.core.database import SessionLocal
from src.agents.backtester import (
    fetch_price_data, build_composite_signal, run_backtest, SP500_UNIVERSE
)
from src.models.hypothesis import Hypothesis

def main():
    db = SessionLocal()
    try:
        hyp = db.query(Hypothesis).filter_by(hypothesis_number=4).first()
        print(f"Backtesting Hypothesis #{hyp.hypothesis_number} with REAL EDGAR earnings data\n")

        start, end = "2018-01-01", "2026-06-30"
        close, volume = fetch_price_data(SP500_UNIVERSE, start, end)

        print("Building composite signal with real earnings data (this will take ~10-15s for EDGAR calls)...")
        composite_real = build_composite_signal(close, volume, hyp.signal_components, use_real_earnings_data=True)

        print("Running backtest with real earnings signal...")
        results_real = run_backtest(composite_real, close, holding_days=hyp.expected_holding_days or 21, backtest_start=start)

        print("\n── REAL EDGAR DATA RESULTS ─────────────")
        for k, v in results_real.items():
            print(f"  {k}: {v}")

        print("\nBuilding composite signal with OLD price-volume proxy for comparison...")
        composite_proxy = build_composite_signal(close, volume, hyp.signal_components, use_real_earnings_data=False)
        results_proxy = run_backtest(composite_proxy, close, holding_days=hyp.expected_holding_days or 21, backtest_start=start)

        print("\n── OLD PROXY RESULTS (for comparison) ──")
        for k, v in results_proxy.items():
            print(f"  {k}: {v}")

        print(f"\n── COMPARISON ──")
        print(f"  Sharpe — real: {results_real['sharpe_ratio']}, proxy: {results_proxy['sharpe_ratio']}")
        print(f"  Max DD — real: {results_real['max_drawdown']}, proxy: {results_proxy['max_drawdown']}")

    finally:
        db.close()

if __name__ == "__main__":
    main()