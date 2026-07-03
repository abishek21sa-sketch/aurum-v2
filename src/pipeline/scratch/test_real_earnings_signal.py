from src.agents.backtester import fetch_price_data, compute_real_earnings_revision, SP500_UNIVERSE

def main():
    print("Fetching price data (for ticker universe only)...")
    close, volume = fetch_price_data(SP500_UNIVERSE[:10], "2024-01-01", "2026-06-30")

    print(f"\nFetching real EDGAR earnings signal for {len(close.columns)} tickers...")
    signal = compute_real_earnings_revision(close.columns.tolist())

    print(f"\nReal earnings revision signal ({len(signal)} tickers returned):")
    print(signal.sort_values(ascending=False))

if __name__ == "__main__":
    main()