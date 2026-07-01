from src.core.database import SessionLocal
from src.agents.continuous_learning import run_full_registry_evaluation

def main():
    db = SessionLocal()
    try:
        reports = run_full_registry_evaluation(db)
        print(f"\nTotal evaluated: {len(reports)}")
    finally:
        db.close()

if __name__ == "__main__":
    main()