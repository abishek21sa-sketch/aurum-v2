from src.core.database import SessionLocal
from src.agents.debate_engine import run_debate, print_debate
from src.models.hypothesis import Hypothesis

def main():
    db = SessionLocal()
    try:
        for hyp_number in [7, 8]:
            hyp = db.query(Hypothesis).filter_by(hypothesis_number=hyp_number).first()
            print(f"\n\n{'#'*70}")
            print(f"# DEBATING HYPOTHESIS #{hyp_number}: {hyp.title}")
            print(f"{'#'*70}")
            debate_record = run_debate(db, hyp)
            print_debate(debate_record)

    finally:
        db.close()

if __name__ == "__main__":
    main()