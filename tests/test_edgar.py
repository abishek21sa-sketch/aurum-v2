import time
from src.data.edgar_client import fetch_company_facts, extract_eps_history, compute_eps_surprise_proxy

def main():
    tickers = ["AAPL", "NVDA", "JPM"]

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"EDGAR DATA — {ticker}")
        print(f"{'='*50}")

        facts = fetch_company_facts(ticker)
        if not facts:
            print(f"  No data retrieved for {ticker}")
            continue

        eps_history = extract_eps_history(facts)
        print(f"  EPS records found: {len(eps_history)}")

        quarterly_only = [r for r in eps_history if r["form"] == "10-Q"]
        if quarterly_only:
            print(f"  Most recent 4 quarterly filings:")
            for r in quarterly_only[-4:]:
                print(f"    {r['filed_date']} ({r['fiscal_period']}): EPS = {r['value']}")

        surprise = compute_eps_surprise_proxy(eps_history)
        print(f"\n  EPS trend signal:")
        for k, v in surprise.items():
            print(f"    {k}: {v}")

        time.sleep(0.5)  # SEC rate limit courtesy — max 10 req/sec, we're well under

if __name__ == "__main__":
    main()