from src.core.database import SessionLocal
from src.agents.research_memory_service import ResearchMemoryService

def main():
    db = SessionLocal()
    try:
        service = ResearchMemoryService(db)

        print("Testing constraint compilation for momentum + low vol hypothesis...\n")
        constraints = service.get_applicable_constraints(
            signal_types=["price_momentum", "earnings_revision", "institutional_flow_proxy"],
            conditions={"macro_regime": "expansion", "vix_range": "<15", "rate_environment": "stable"},
            holding_days=21
        )

        print(f"Applicable constraints found: {len(constraints)}")
        for c in constraints:
            print(f"\n[{c['failure_mode']}]")
            print(f"  Why it applies: {c['applicability_reasoning']}")
            print(f"  Constraint: {c['constraint']}")
            print(f"  Verification: {c['verification_check']}")

        if constraints:
            print("\n\nFormatted for Research Scientist:")
            print("─"*50)
            print(service.format_constraints_for_scientist(constraints))

    finally:
        db.close()

if __name__ == "__main__":
    main()