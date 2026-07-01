from src.core.database import SessionLocal
from src.agents.backtester import run_hypothesis_backtest
from src.models.hypothesis import Hypothesis

def main():
    db = SessionLocal()
    try:
        hyp = db.query(Hypothesis).filter_by(hypothesis_number=3).first()
        if not hyp:
            print("Hypothesis #3 not found.")
            return

        print(f"Re-running backtest for Hypothesis #{hyp.hypothesis_number}: {hyp.title}")
        results = run_hypothesis_backtest(db, hyp)

        print("\n── Corrected Backtest Results ──────────────")
        for k, v in results.items():
            print(f"  {k:<28}: {v}")

    finally:
        db.close()

if __name__ == "__main__":
    main()