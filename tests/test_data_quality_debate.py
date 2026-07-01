from src.agents.data_quality_debate import debate_data_quality_finding, print_finding_debate

def main():
    real_results = {
        "sharpe_ratio": 1.3974, "sortino_ratio": 3.5593, "calmar_ratio": 1.7768,
        "max_drawdown": -0.153, "annualized_return": 0.2719, "win_rate": 0.7,
        "oos_sharpe": 2.3283, "n_long_positions": 9
    }
    proxy_results = {
        "sharpe_ratio": 1.1321, "sortino_ratio": 3.047, "calmar_ratio": 1.1266,
        "max_drawdown": -0.2291, "annualized_return": 0.2581, "win_rate": 0.7,
        "oos_sharpe": 1.4477, "n_long_positions": 10
    }

    print("Running data quality validation debate...")
    result = debate_data_quality_finding(
        real_results, proxy_results,
        dropped_tickers=["BRK-B", "GS", "V"],
        real_n=47, proxy_n=50
    )
    print_finding_debate(result)

if __name__ == "__main__":
    main()