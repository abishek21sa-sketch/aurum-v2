from src.agents.backtester import fetch_price_data, SP500_UNIVERSE
from src.data.edgar_client import fetch_company_facts, extract_eps_history, compute_eps_surprise_proxy

def main():
    close, volume = fetch_price_data(SP500_UNIVERSE[:10], "2024-01-01", "2026-06-30")
    tickers = close.columns.tolist()
    print("Tickers in this slice:", tickers)
    print()

    for t in tickers:
        facts = fetch_company_facts(t)
        if not facts:
            print(f"{t}: NO FACTS RETRIEVED (likely missing/incorrect CIK mapping)")
            continue
        hist = extract_eps_history(facts)
        surprise = compute_eps_surprise_proxy(hist)
        if "error" in surprise:
            print(f"{t}: {surprise['error']} (n_quarters={surprise.get('n_quarters', '?')})")
        else:
            print(f"{t}: OK — eps_growth_yoy_proxy={surprise['eps_growth_yoy_proxy']}")

if __name__ == "__main__":
    main()