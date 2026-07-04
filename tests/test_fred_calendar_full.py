from src.data.fred_client import get_upcoming_releases, calendar_to_risk_context

def main():
    print("Fetching economic calendar (next 30 days)...\n")
    events = get_upcoming_releases(days_ahead=30)

    print(f"Upcoming events: {len(events)}\n")
    for e in events:
        importance_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(
            e["importance"], "⚪"
        )
        print(f"  {importance_icon} {e['date']} ({e['days_until']}d) — "
              f"{e['name']} [{e['signal']}]")

    print("\nRisk context:")
    ctx = calendar_to_risk_context(events)
    print(f"  Near-term risk: {ctx['near_term_risk']}")
    print(f"  Summary: {ctx['summary']}")
    if ctx["critical_events_7d"]:
        print(f"  Critical events this week:")
        for e in ctx["critical_events_7d"]:
            print(f"    - {e['name']} on {e['date']}")

if __name__ == "__main__":
    main()