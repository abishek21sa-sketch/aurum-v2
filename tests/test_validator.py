from src.core.database import SessionLocal
from src.agents.statistical_validator import run_statistical_validation, print_validation
from src.models.hypothesis import Hypothesis

def main():
    db = SessionLocal()
    try:
        # Validate hypothesis #4 (the clean one)
        hyp = db.query(Hypothesis).filter_by(hypothesis_number=4).first()
        if not hyp:
            print("Hypothesis #4 not found.")
            return

        print(f"Running statistical validation for Hypothesis #{hyp.hypothesis_number}: {hyp.title}")
        review = run_statistical_validation(db, hyp)
        print_validation(review)

    finally:
        db.close()

if __name__ == "__main__":
    main()