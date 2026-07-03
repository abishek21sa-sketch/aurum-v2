from src.core.database import SessionLocal
from src.agents.portfolio_lab import run_portfolio_lab, SCENARIOS

def main():
    db = SessionLocal()
    try:
        # Run two scenarios first to test
        print("Running Portfolio Lab stress tests...\n")
        results = run_portfolio_lab(db, scenario_keys=["fed_hike_100bps", "flash_crash"])

        for alpha_result in results:
            print(f"\n{'='*60}")
            print(f"ALPHA: {alpha_result['alpha_name']}")
            print(f"Backtest Sharpe: {alpha_result['backtest_sharpe']}")
            print(f"{'='*60}")

            for scenario_key, scenario_result in alpha_result["scenarios"].items():
                print(f"\n── {scenario_result['scenario_name']} ──")
                print(f"  Portfolio impact: {scenario_result['estimated_portfolio_impact']*100:.1f}%")
                print(f"  Circuit breaker: {'FIRES' if scenario_result['circuit_breaker_fires'] else 'does not fire'}")
                print(f"  Crowding amplifier: {scenario_result['crowding_amplifier']:.2f}x")
                print(f"\n  Analysis:")
                print(f"  {scenario_result['explanation']}")

    finally:
        db.close()

if __name__ == "__main__":
    main()