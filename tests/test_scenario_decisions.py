from src.core.database import SessionLocal
from src.agents.scenario_decision_engine import run_scenario_with_decisions

def main():
    db = SessionLocal()
    try:
        print("Running scenario decisions (flash_crash + fed_hike_100bps)...\n")
        results = run_scenario_with_decisions(
            db, scenario_keys=["flash_crash", "fed_hike_100bps"]
        )
        for r in results:
            d = r["decision"]
            icon = {"HOLD": "🟢", "REDUCE": "🟡", "RETIRE": "🔴",
                    "RESEARCH": "🔵", "RECALIBRATE": "🟠", "HEDGE": "🟡"}.get(
                d.get("decision", ""), "⚪"
            )
            print(f"{icon} H#{r['hypothesis_number']} | {r['scenario_name'][:30]:<30} "
                  f"→ {d.get('decision'):<12} "
                  f"(confidence {d.get('confidence', 0):.0%})")
            print(f"   {d.get('rationale', '')[:100]}")
            if d.get("decision") == "RESEARCH" and d.get("research_trigger"):
                rt = d["research_trigger"]
                print(f"   → New angle: {rt.get('hypothesis_angle', '')[:80]}")
    finally:
        db.close()

if __name__ == "__main__":
    main()