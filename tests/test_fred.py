from src.data.fred_client import get_macro_snapshot, macro_snapshot_to_observations
import json

def main():
    print("Fetching FRED macro snapshot...\n")
    snapshot = get_macro_snapshot()

    if "error" in snapshot:
        print(f"Error: {snapshot['error']}")
        return

    print(f"Timestamp: {snapshot['timestamp']}")
    print(f"\n── Regime Classification ──────────────────")
    print(f"  Macro regime     : {snapshot['regime']}")
    print(f"  Rate environment : {snapshot['rate_environment']}")
    print(f"  Inflation        : {snapshot['inflation_pressure']}")
    print(f"  Yield curve      : {snapshot['yield_curve']}")
    print(f"  VIX regime       : {snapshot['vix_regime']}")

    print(f"\n── Key Levels ─────────────────────────────")
    for k, v in snapshot["key_levels"].items():
        if v is not None:
            print(f"  {k:<25}: {v}")

    print(f"\n── As Research Scientist Observations ─────")
    obs = macro_snapshot_to_observations(snapshot)
    for k, v in obs.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()