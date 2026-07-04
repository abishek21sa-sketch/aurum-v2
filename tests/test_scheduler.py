from src.core.database import SessionLocal
from src.agents.research_scheduler import run_scheduler, print_schedule

def main():
    db = SessionLocal()
    try:
        print("Running Research Scheduler...")
        result = run_scheduler(db)
        print_schedule(result)
    finally:
        db.close()

if __name__ == "__main__":
    main()