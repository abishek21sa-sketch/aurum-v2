from src.core.database import SessionLocal
from src.agents.alpha_registrar import register_all_eligible

def main():
    db = SessionLocal()
    try:
        print("Scanning all hypotheses for registry eligibility...\n")
        # Use require_paper_trading=False first to see what's validated but not yet deployed
        registered = register_all_eligible(db, require_paper_trading=False)

        print(f"\n{'='*50}")
        print(f"Total registered: {len(registered)}")
        print(f"{'='*50}")
        for a in registered:
            print(f"  {a.signal_name[:50]:<50} Sharpe: {a.sharpe_ratio}")

    finally:
        db.close()

if __name__ == "__main__":
    main()