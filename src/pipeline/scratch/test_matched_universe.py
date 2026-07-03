from src.agents.backtester import (
    fetch_price_data, build_composite_signal, run_backtest
)
from src.core.database import SessionLocal
from src.models.hypothesis import Hypothesis

DROPPED_TICKERS = {"BRK-B", "GS", "V"}

SP500_47 = [t for t in [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","LLY",
    "UNH","XOM","AVGO","MA","PG","HD","COST","JNJ","MRK",
    "ABBV","BAC","WMT","NFLX","CRM","AMD","PEP","KO","TMO","ORCL",
    "ACN","MCD","LIN","CSCO","DHR","NKE","TXN","NEE","PM","QCOM",
    "BMY","UPS","MS","INTC","RTX","AMGN","HON","LOW","IBM"
] if t not in DROPPED_TICKERS]

def main():
    db = SessionLocal()
    try:
        hyp = db.query(Hypothesis).filter_by(hypothesis_number=4).first()
        start, end = "2018-01-01", "2026-06-30"

        print(f"Fetching price data for matched 47-ticker universe...")
        close, volume = fetch_price_data(SP500_47, start, end)
        print(f"  Shape: {close.shape}\n")

        print("Running PROXY signal on matched 47-ticker universe...")
        composite_proxy_47 = build_composite_signal(
            close, volume, hyp.signal_components, use_real_earnings_data=False
        )
        results_proxy_47 = run_backtest(
            composite_proxy_47, close,
            holding_days=hyp.expected_holding_days or 21,
            backtest_start=start
        )

        print("Running REAL EDGAR signal on matched 47-ticker universe...")
        composite_real_47 = build_composite_signal(
            close, volume, hyp.signal_components, use_real_earnings_data=True
        )
        results_real_47 = run_backtest(
            composite_real_47, close,
            holding_days=hyp.expected_holding_days or 21,
            backtest_start=start
        )

        print(f"\n{'='*65}")
        print(f"JUDGE'S REQUIRED TEST — MATCHED UNIVERSE (47 tickers)")
        print(f"{'='*65}")
        print(f"\n{'Metric':<25} {'Proxy 47':>12} {'EDGAR 47':>12} {'Diff':>10}")
        print(f"{'-'*60}")
        for k in ["sharpe_ratio", "sortino_ratio", "calmar_ratio",
                   "max_drawdown", "annualized_return", "win_rate", "oos_sharpe"]:
            p = results_proxy_47.get(k, 0)
            r = results_real_47.get(k, 0)
            diff = r - p
            print(f"  {k:<23} {p:>12.4f} {r:>12.4f} {diff:>+10.4f}")

        print(f"\nVERDICT CRITERIA:")
        oos_diff = results_real_47.get("oos_sharpe", 0) - results_proxy_47.get("oos_sharpe", 0)
        win_diff = results_real_47.get("win_rate", 0) - results_proxy_47.get("win_rate", 0)
        print(f"  OOS Sharpe difference: {oos_diff:+.4f} "
              f"({'material EDGAR advantage' if oos_diff > 0.15 else 'artifact confirmed' if oos_diff < 0.05 else 'ambiguous'})")
        print(f"  Win rate difference:   {win_diff:+.4f} "
              f"({'signal ranks better' if win_diff > 0.02 else 'no ranking improvement'})")

    finally:
        db.close()

if __name__ == "__main__":
    main()