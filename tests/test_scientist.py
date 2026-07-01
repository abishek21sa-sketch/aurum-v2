from src.core.database import SessionLocal
from src.agents.research_scientist import generate_hypothesis, print_hypothesis
from src.agents.backtester import run_hypothesis_backtest
from src.models.hypothesis import Hypothesis

def main():
    db = SessionLocal()
    try:
        # Generate a new hypothesis
        observations = {
            "momentum_signal": "strong positive price momentum across large cap equities, 12-1 month",
            "volatility_regime": "low volatility, VIX around 14, compressing further",
            "earnings_trend": "positive earnings revisions in technology and industrials",
            "macro_regime": "expansion, GDP growth above trend",
            "rate_environment": "stable, Fed on hold",
            "institutional_flow": "consistent buying in growth factors over past 4 weeks"
        }

        print("Generating hypothesis...")
        hypothesis = generate_hypothesis(db, observations)
        print_hypothesis(hypothesis)

        # Run backtest on it
        print(f"Running backtest on Hypothesis #{hypothesis.hypothesis_number}...")
        results = run_hypothesis_backtest(db, hypothesis)

        print("\n── Backtest Results ──────────────────────")
        for k, v in results.items():
            print(f"  {k:<28}: {v}")

        # Verify DB state
        db.refresh(hypothesis)
        print(f"\n── DB State ──────────────────────────────")
        print(f"  Hypothesis status : {hypothesis.status}")
        print(f"  Sharpe ratio      : {hypothesis.sharpe_ratio}")
        print(f"  Max drawdown      : {hypothesis.max_drawdown}")

    finally:
        db.close()

if __name__ == "__main__":
    main()