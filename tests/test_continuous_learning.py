from src.core.database import SessionLocal
from src.agents.continuous_learning import simulate_paper_trading, print_learning_report
from src.models.hypothesis import Hypothesis

def main():
    db = SessionLocal()
    try:
        hyp = db.query(Hypothesis).filter_by(hypothesis_number=4).first()
        print(f"Running continuous learning simulation for Hypothesis #{hyp.hypothesis_number}\n")

        report = simulate_paper_trading(db, hyp, simulation_days=756)
        if "error" in report:
            print(f"Error: {report['error']}")
            return

        print_learning_report(report)

    finally:
        db.close()

if __name__ == "__main__":
    main()