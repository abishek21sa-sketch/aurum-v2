from src.core.database import SessionLocal
from src.agents.backtester import (
    fetch_price_data, fetch_vix_data, build_composite_signal,
    run_backtest, run_backtest_with_circuit_breaker, SP500_UNIVERSE
)
from src.models.hypothesis import Hypothesis

def main():
    db = SessionLocal()
    try:
        hyp = db.query(Hypothesis).filter_by(hypothesis_number=7).first()
        print(f"Comparing WITH vs WITHOUT circuit breaker — Hypothesis #{hyp.hypothesis_number}\n")

        start, end = "2018-01-01", "2026-06-30"
        close, volume = fetch_price_data(SP500_UNIVERSE, start, end)
        vix = fetch_vix_data(start, end)
        composite = build_composite_signal(close, volume, hyp.signal_components)

        print("── WITHOUT circuit breaker (scheduled rebalance only) ──")
        baseline = run_backtest(composite, close, holding_days=10, backtest_start=start)
        for k, v in baseline.items():
            print(f"  {k}: {v}")

        print("\n── WITH circuit breaker (VIX>=18 forces early exit) ──")
        protected = run_backtest_with_circuit_breaker(
            composite, close, vix, holding_days=10,
            vix_exit_threshold=18.0, backtest_start=start
        )
        for k, v in protected.items():
            print(f"  {k}: {v}")

        print(f"\n── COMPARISON ──")
        print(f"  Max drawdown — baseline: {baseline['max_drawdown']:.4f}, protected: {protected['max_drawdown']:.4f}")
        print(f"  Sharpe       — baseline: {baseline['sharpe_ratio']:.4f}, protected: {protected['sharpe_ratio']:.4f}")
        print(f"  Circuit breaker fired {protected['circuit_breaker_triggers']} times "
              f"out of {protected['total_periods']} periods "
              f"({protected['circuit_breaker_trigger_rate']*100:.1f}%)")

    finally:
        db.close()

if __name__ == "__main__":
    main()