from src.data.relational_knowledge_store import get_knowledge_store

def main():
    ks = get_knowledge_store()

    print("=== ENTITY LOOKUP ===")
    nvda = ks.get_entity("NVDA")
    print(f"NVDA: {nvda.name} | {nvda.sector} | {nvda.industry}")

    print("\n=== SECTOR PEERS ===")
    peers = ks.get_peers("NVDA")
    print(f"NVDA peers: {[p.ticker for p in peers]}")

    print("\n=== ETF EXPOSURE ===")
    etfs = ks.get_etf_exposure("NVDA")
    for e in etfs:
        print(f"  {e['etf']}: {e['n_constituents']} constituents")

    print("\n=== MACRO SENSITIVITIES ===")
    macro = ks.get_macro_sensitivities("NVDA")
    print(f"NVDA macro sensitivities: {macro}")

    print("\n=== SECTOR CONCENTRATION (top momentum names) ===")
    top = ["NVDA", "AMD", "META", "MSFT", "AAPL", "GOOGL", "AMZN"]
    concentration = ks.get_sector_concentration(top)
    for sector, info in concentration.items():
        flag = " ⚠️ HIGH" if info["concentration_flag"] else ""
        print(f"  {sector}: {info['count']} ({info['pct']*100:.0f}%){flag}")

    print("\n=== ETF CROWDING (top momentum names) ===")
    crowding = ks.get_common_etf_exposure(top)
    for etf, info in sorted(crowding.items(), key=lambda x: -x[1]["overlap_count"])[:5]:
        print(f"  {etf}: {info['overlap_count']}/{len(top)} names — {info['concentration_risk']} risk")

    print("\n=== NATURAL LANGUAGE QUERY ===")
    result = ks.query("What ETFs does NVDA appear in?")
    print(result)

if __name__ == "__main__":
    main()