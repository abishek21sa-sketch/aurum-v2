import json
from sqlalchemy import text
from sqlalchemy.orm import Session
from src.data.knowledge_store import KnowledgeStore, Entity, Relationship

# ── Static knowledge base for our 50-ticker SP500 universe ───────
# GICS sector classification + macro sensitivity tags
# This is the relational "brain" — starts static, queryable, upgradeable later

ENTITY_DATA = {
    "AAPL":  ("Apple Inc",               "Information Technology", "Technology Hardware",    "large", ["rates", "consumer_spending", "china_trade"]),
    "MSFT":  ("Microsoft Corp",          "Information Technology", "Software",               "large", ["rates", "enterprise_spend", "cloud_growth"]),
    "NVDA":  ("NVIDIA Corp",             "Information Technology", "Semiconductors",         "large", ["rates", "ai_capex", "china_trade", "datacenter"]),
    "AMZN":  ("Amazon.com Inc",          "Consumer Discretionary","E-Commerce",             "large", ["consumer_spending", "cloud_growth", "rates"]),
    "GOOGL": ("Alphabet Inc",            "Communication Services","Internet Media",         "large", ["digital_advertising", "rates", "ai_capex"]),
    "META":  ("Meta Platforms",          "Communication Services","Social Media",           "large", ["digital_advertising", "rates", "regulation"]),
    "TSLA":  ("Tesla Inc",               "Consumer Discretionary","Electric Vehicles",      "large", ["rates", "energy_prices", "china_trade", "consumer_spending"]),
    "BRK-B": ("Berkshire Hathaway",      "Financials",            "Diversified Financials", "large", ["rates", "credit_spreads", "insurance_cycle"]),
    "JPM":   ("JPMorgan Chase",          "Financials",            "Banks",                  "large", ["rates", "credit_spreads", "gdp_growth"]),
    "LLY":   ("Eli Lilly",               "Health Care",           "Pharmaceuticals",        "large", ["drug_pricing", "fda_policy", "healthcare_reform"]),
    "V":     ("Visa Inc",                "Financials",            "Payment Networks",       "large", ["consumer_spending", "rates", "digital_payments"]),
    "UNH":   ("UnitedHealth Group",      "Health Care",           "Managed Care",           "large", ["healthcare_reform", "drug_pricing", "gdp_growth"]),
    "XOM":   ("Exxon Mobil",             "Energy",                "Integrated Oil",         "large", ["oil_prices", "energy_transition", "geopolitics"]),
    "AVGO":  ("Broadcom Inc",            "Information Technology","Semiconductors",         "large", ["ai_capex", "datacenter", "china_trade"]),
    "MA":    ("Mastercard Inc",          "Financials",            "Payment Networks",       "large", ["consumer_spending", "rates", "digital_payments"]),
    "PG":    ("Procter & Gamble",        "Consumer Staples",      "Household Products",     "large", ["consumer_spending", "inflation", "fx_rates"]),
    "HD":    ("Home Depot",              "Consumer Discretionary","Home Improvement",       "large", ["rates", "housing_market", "consumer_spending"]),
    "COST":  ("Costco Wholesale",        "Consumer Staples",      "Discount Retail",        "large", ["consumer_spending", "inflation"]),
    "JNJ":   ("Johnson & Johnson",       "Health Care",           "Diversified Healthcare", "large", ["healthcare_reform", "drug_pricing", "litigation"]),
    "MRK":   ("Merck & Co",              "Health Care",           "Pharmaceuticals",        "large", ["drug_pricing", "fda_policy", "healthcare_reform"]),
    "ABBV":  ("AbbVie Inc",              "Health Care",           "Biopharmaceuticals",     "large", ["drug_pricing", "fda_policy", "patent_cliff"]),
    "BAC":   ("Bank of America",         "Financials",            "Banks",                  "large", ["rates", "credit_spreads", "gdp_growth"]),
    "WMT":   ("Walmart Inc",             "Consumer Staples",      "General Merchandise",    "large", ["consumer_spending", "inflation", "china_trade"]),
    "NFLX":  ("Netflix Inc",             "Communication Services","Streaming",              "large", ["consumer_spending", "rates", "content_costs"]),
    "CRM":   ("Salesforce Inc",          "Information Technology","Enterprise Software",    "large", ["enterprise_spend", "rates", "ai_capex"]),
    "AMD":   ("Advanced Micro Devices",  "Information Technology","Semiconductors",         "large", ["ai_capex", "datacenter", "china_trade"]),
    "PEP":   ("PepsiCo Inc",             "Consumer Staples",      "Beverages",              "large", ["consumer_spending", "inflation", "fx_rates"]),
    "KO":    ("Coca-Cola Co",            "Consumer Staples",      "Beverages",              "large", ["consumer_spending", "inflation", "fx_rates"]),
    "TMO":   ("Thermo Fisher Scientific","Health Care",           "Life Sciences Tools",    "large", ["healthcare_reform", "biotech_funding", "rates"]),
    "ORCL":  ("Oracle Corp",             "Information Technology","Enterprise Software",    "large", ["enterprise_spend", "cloud_growth", "ai_capex"]),
    "ACN":   ("Accenture PLC",           "Information Technology","IT Services",            "large", ["enterprise_spend", "rates", "ai_capex"]),
    "MCD":   ("McDonald's Corp",         "Consumer Discretionary","Restaurants",            "large", ["consumer_spending", "inflation", "fx_rates"]),
    "LIN":   ("Linde PLC",               "Materials",             "Industrial Gases",       "large", ["energy_prices", "industrial_production", "rates"]),
    "CSCO":  ("Cisco Systems",           "Information Technology","Networking Equipment",   "large", ["enterprise_spend", "datacenter", "rates"]),
    "DHR":   ("Danaher Corp",            "Health Care",           "Life Sciences Tools",    "large", ["healthcare_reform", "biotech_funding", "rates"]),
    "NKE":   ("Nike Inc",                "Consumer Discretionary","Apparel",                "large", ["consumer_spending", "china_trade", "fx_rates"]),
    "TXN":   ("Texas Instruments",       "Information Technology","Semiconductors",         "large", ["industrial_production", "auto_cycle", "rates"]),
    "NEE":   ("NextEra Energy",          "Utilities",             "Electric Utilities",     "large", ["rates", "energy_transition", "regulation"]),
    "PM":    ("Philip Morris Intl",      "Consumer Staples",      "Tobacco",                "large", ["regulation", "fx_rates", "consumer_spending"]),
    "QCOM":  ("Qualcomm Inc",            "Information Technology","Semiconductors",         "large", ["ai_capex", "china_trade", "mobile_cycle"]),
    "BMY":   ("Bristol-Myers Squibb",    "Health Care",           "Pharmaceuticals",        "large", ["drug_pricing", "fda_policy", "patent_cliff"]),
    "UPS":   ("United Parcel Service",   "Industrials",           "Air Freight",            "large", ["consumer_spending", "industrial_production", "energy_prices"]),
    "MS":    ("Morgan Stanley",          "Financials",            "Investment Banking",     "large", ["rates", "credit_spreads", "capital_markets"]),
    "INTC":  ("Intel Corp",              "Information Technology","Semiconductors",         "large", ["ai_capex", "datacenter", "china_trade"]),
    "RTX":   ("RTX Corp",                "Industrials",           "Aerospace & Defense",   "large", ["defense_spending", "geopolitics", "supply_chain"]),
    "AMGN":  ("Amgen Inc",               "Health Care",           "Biopharmaceuticals",    "large", ["drug_pricing", "fda_policy", "patent_cliff"]),
    "HON":   ("Honeywell Intl",          "Industrials",           "Industrial Conglomerate","large", ["industrial_production", "energy_prices", "rates"]),
    "LOW":   ("Lowe's Companies",        "Consumer Discretionary","Home Improvement",      "large", ["rates", "housing_market", "consumer_spending"]),
    "IBM":   ("IBM Corp",                "Information Technology","IT Services",            "large", ["enterprise_spend", "cloud_growth", "rates"]),
    "GS":    ("Goldman Sachs",           "Financials",            "Investment Banking",     "large", ["rates", "credit_spreads", "capital_markets"]),
}

# ETF constituency map — which major ETFs hold which tickers
ETF_CONSTITUENCIES = {
    "QQQ":  ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","CSCO","AMD","QCOM","INTC","NFLX","CRM","ORCL","ACN"],
    "XLK":  ["AAPL","MSFT","NVDA","AVGO","AMD","QCOM","INTC","CRM","ORCL","ACN","IBM","CSCO","TXN","HON"],
    "XLF":  ["JPM","BAC","V","MA","GS","MS","BRK-B","UNH","AXP"],
    "XLV":  ["LLY","UNH","JNJ","MRK","ABBV","TMO","DHR","BMY","AMGN"],
    "XLC":  ["GOOGL","META","NFLX","DIS"],
    "XLY":  ["AMZN","TSLA","HD","MCD","NKE","LOW"],
    "XLP":  ["PG","KO","PEP","WMT","COST","PM"],
    "XLE":  ["XOM","CVX"],
    "XLI":  ["RTX","HON","UPS","LIN"],
    "XLU":  ["NEE"],
    "SMH":  ["NVDA","AVGO","AMD","QCOM","INTC","TXN"],
    "SPY":  list(ENTITY_DATA.keys()),
    "IVV":  list(ENTITY_DATA.keys()),
}

class RelationalKnowledgeStore(KnowledgeStore):
    """
    Tier 1 implementation: static relational knowledge.
    All data lives in Python dicts, queryable without a DB.
    Drop-in replaceable with GraphKnowledgeStore (Neo4j) later
    by implementing the same KnowledgeStore interface.
    """

    def get_entity(self, ticker: str) -> Entity | None:
        data = ENTITY_DATA.get(ticker)
        if not data:
            return None
        name, sector, industry, cap_tier, _ = data
        return Entity(
            ticker=ticker, name=name, sector=sector,
            industry=industry, market_cap_tier=cap_tier
        )

    def get_peers(self, ticker: str) -> list[Entity]:
        entity = self.get_entity(ticker)
        if not entity:
            return []
        return [
            self.get_entity(t)
            for t in ENTITY_DATA
            if t != ticker and ENTITY_DATA[t][1] == entity.sector
        ]

    def get_etf_exposure(self, ticker: str) -> list[dict]:
        results = []
        for etf, constituents in ETF_CONSTITUENCIES.items():
            if ticker in constituents:
                results.append({
                    "etf": etf,
                    "n_constituents": len(constituents),
                    "concentration": round(1 / len(constituents), 4)
                })
        return results

    def get_sector_tickers(self, sector: str) -> list[str]:
        return [t for t, data in ENTITY_DATA.items() if data[1] == sector]

    def get_macro_sensitivities(self, ticker: str) -> list[str]:
        data = ENTITY_DATA.get(ticker)
        if not data:
            return []
        return data[4]

    def get_common_etf_exposure(self, tickers: list[str]) -> dict:
        """
        For a list of tickers (e.g. a portfolio's top holdings),
        returns which ETFs hold ALL of them — identifying crowding risk.
        """
        etf_sets = {}
        for ticker in tickers:
            for entry in self.get_etf_exposure(ticker):
                etf = entry["etf"]
                if etf not in etf_sets:
                    etf_sets[etf] = set()
                etf_sets[etf].add(ticker)

        return {
            etf: {
                "tickers_held": list(held),
                "overlap_count": len(held),
                "concentration_risk": "high" if len(held) >= len(tickers) * 0.7 else "medium" if len(held) >= 0.4 * len(tickers) else "low"
            }
            for etf, held in etf_sets.items()
            if len(held) >= 2
        }

    def get_sector_concentration(self, tickers: list[str]) -> dict:
        """
        For a portfolio's top holdings, return sector breakdown
        and flag any sector exceeding 40% concentration.
        """
        sectors = {}
        for t in tickers:
            entity = self.get_entity(t)
            if entity:
                sectors[entity.sector] = sectors.get(entity.sector, 0) + 1

        total = len(tickers)
        return {
            sector: {
                "count": count,
                "pct": round(count / total, 3),
                "concentration_flag": count / total > 0.4
            }
            for sector, count in sorted(sectors.items(), key=lambda x: -x[1])
        }

    def query(self, natural_language: str) -> str:
        """
        Simple keyword-based query dispatcher.
        Upgraded to LLM-based routing when Neo4j is available.
        """
        q = natural_language.lower()

        if "etf" in q or "exposure" in q:
            tickers = [t for t in ENTITY_DATA if t.lower() in q]
            if tickers:
                results = []
                for t in tickers:
                    etfs = self.get_etf_exposure(t)
                    results.append(f"{t}: {[e['etf'] for e in etfs]}")
                return "\n".join(results)

        if "sector" in q:
            sectors = list(set(data[1] for data in ENTITY_DATA.values()))
            matching = [s for s in sectors if s.lower().replace(" ", "_") in q.replace(" ", "_")]
            if matching:
                tickers = self.get_sector_tickers(matching[0])
                return f"{matching[0]}: {tickers}"

        if "peer" in q or "peers" in q:
            tickers = [t for t in ENTITY_DATA if t in natural_language.upper()]
            if tickers:
                peers = self.get_peers(tickers[0])
                return f"Peers of {tickers[0]}: {[p.ticker for p in peers]}"

        if "macro" in q or "sensitive" in q or "sensitivity" in q:
            tickers = [t for t in ENTITY_DATA if t in natural_language.upper()]
            if tickers:
                sens = self.get_macro_sensitivities(tickers[0])
                return f"{tickers[0]} macro sensitivities: {sens}"

        return "Query not resolved. Available: etf exposure, sector peers, macro sensitivities."

# ── Singleton for use across the system ──────────────────────────
_store = None

def get_knowledge_store() -> RelationalKnowledgeStore:
    global _store
    if _store is None:
        _store = RelationalKnowledgeStore()
    return _store