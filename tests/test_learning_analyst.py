from src.core.database import SessionLocal
from src.agents.learning_analyst import run_full_registry_analysis

def main():
    db = SessionLocal()
    try:
        print("Running Learning Analyst across full registry...\n")
        results = run_full_registry_analysis(db)

        print(f"\n{'='*60}")
        print(f"LEARNING ANALYST — FULL REGISTRY ANALYSIS")
        print(f"{'='*60}")

        for r in results:
            print(f"\nH#{r['hypothesis_number']}: {r['title'][:50]}")
            print(f"  Root cause: [{r['root_cause_category']}] confidence={r['confidence']}")
            print(f"  Diagnosis: {r['root_cause'][:120]}")
            print(f"  Action: {r['recommended_action']} — {r['action_rationale'][:100]}")
            if r.get("memory_written"):
                print(f"  ✅ New memory written")
            if r.get("improvement_suggestions"):
                print(f"  Improvements:")
                for s in r["improvement_suggestions"][:2]:
                    print(f"    - {s[:80]}")

    finally:
        db.close()

if __name__ == "__main__":
    main()