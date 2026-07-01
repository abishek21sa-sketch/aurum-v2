from src.core.database import SessionLocal
from src.agents.debate_engine import run_debate, print_debate
from src.models.hypothesis import Hypothesis

def main():
    db = SessionLocal()
    try:
        hyp = db.query(Hypothesis).filter_by(hypothesis_number=4).first()
        if not hyp:
            print("Hypothesis #4 not found.")
            return

        print(f"Running committee debate for Hypothesis #{hyp.hypothesis_number}: {hyp.title}\n")
        debate_record = run_debate(db, hyp)
        print_debate(debate_record)

    finally:
        db.close()

if __name__ == "__main__":
    main()