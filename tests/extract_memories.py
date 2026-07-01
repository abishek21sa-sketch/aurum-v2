from src.core.database import SessionLocal
from src.agents.memory_extractor import extract_memory_from_debate, print_memory
from src.models.hypothesis import Hypothesis
from src.models.governance import GovernanceRecord

def main():
    db = SessionLocal()
    try:
        for hyp_number in [3, 4]:
            hyp = db.query(Hypothesis).filter_by(hypothesis_number=hyp_number).first()
            gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hyp.id).first()

            print(f"Extracting memory from Hypothesis #{hyp_number} debate...")
            memory = extract_memory_from_debate(db, hyp, gov)
            if memory:
                print_memory(memory)

    finally:
        db.close()

if __name__ == "__main__":
    main()