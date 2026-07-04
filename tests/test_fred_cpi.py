from src.data.fred_client import get_recent_values

def main():
    trend = get_recent_values("CPIAUCSL", 13)
    print(f"CPI observations: {len(trend)}")
    for t in trend:
        print(f"  {t['date']}: {t['value']}")

if __name__ == "__main__":
    main()