import requests
import time
from datetime import datetime, timedelta

SEC_USER_AGENT = "AURUM-V2 Research Platform research@aurum-v2.local"
BASE_URL = "https://data.sec.gov"

HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

# Static ticker -> CIK mapping for our universe (SEC requires zero-padded 10-digit CIK)
# Pulled once from https://www.sec.gov/files/company_tickers.json and cached here
# for the 50 tickers in our existing SP500_UNIVERSE.
TICKER_TO_CIK = {
    "AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810",
    "AMZN": "0001018724", "GOOGL": "0001652044", "META": "0001326801",
    "TSLA": "0001318605", "JPM": "0000019617", "LLY": "0000059478",
    "BRK-B": "0001067983",
    "V": "0001403161", "UNH": "0000731766", "XOM": "0000034088",
    "AVGO": "0001730168", "MA": "0001141391", "PG": "0000080424",
    "HD": "0000354950", "COST": "0000909832", "JNJ": "0000200406",
    "MRK": "0000310158", "ABBV": "0001551152", "BAC": "0000070858",
    "WMT": "0000104169", "NFLX": "0001065280", "CRM": "0001108524",
    "AMD": "0000002488", "PEP": "0000077476", "KO": "0000021344",
    "TMO": "0000097745", "ORCL": "0001341439", "ACN": "0001467373",
    "MCD": "0000063908", "LIN": "0001707925", "CSCO": "0000858877",
    "DHR": "0000313616", "NKE": "0000320187", "TXN": "0000097476",
    "NEE": "0000753308", "PM": "0001413329", "QCOM": "0000804328",
    "BMY": "0000014272", "UPS": "0001090727", "MS": "0000895421",
    "INTC": "0000050863", "RTX": "0000101829", "AMGN": "0000318154",
    "HON": "0000773840", "LOW": "0000060667", "IBM": "0000051143",
    "GS": "0000886982",
}

# Tickers with confirmed structural XBRL reporting gaps — excluded from
# earnings-based signals with documented reasons rather than silent drops.
KNOWN_EDGAR_GAPS = {
    "BRK-B": "dual-class share structure; EarningsPerShareDiluted tag not consistently populated",
    "V": "insufficient quarterly EPS history in EDGAR XBRL — likely non-standard fiscal year or tag variant"
}

def get_cik(ticker: str) -> str | None:
    return TICKER_TO_CIK.get(ticker)

def fetch_company_facts(ticker: str) -> dict | None:
    """
    Fetches all XBRL company facts (EPS, revenue, etc.) for a ticker from
    SEC EDGAR's structured data API. Returns the raw JSON or None on failure.
    Tickers in KNOWN_EDGAR_GAPS are skipped immediately without an API call.
    """
    if ticker in KNOWN_EDGAR_GAPS:
        return None
    cik = get_cik(ticker)
    if not cik:
        return None
    
    url = f"{BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  EDGAR fetch failed for {ticker}: HTTP {resp.status_code}")
            return None
    except requests.RequestException as e:
        print(f"  EDGAR fetch error for {ticker}: {e}")
        return None

def extract_eps_history(facts: dict) -> list[dict]:
    """
    Extracts genuinely single-quarter EPS (diluted) history from company facts JSON.

    Uniquely identifies each quarter by (fiscal_year, fiscal_period) rather than
    end_date alone, since SEC filings routinely include prior-year comparative
    figures under the same end_date/span but a different fy — keying on end_date
    caused duplicate and out-of-order entries. When the same (fy, fp) appears
    multiple times (restatements, amendments), the most recently filed value wins.
    """
    if not facts:
        return []

    try:
        eps_data = facts["facts"]["us-gaap"]["EarningsPerShareDiluted"]["units"]["USD/shares"]
    except KeyError:
        return []

    from datetime import datetime as _dt

    candidates = {}
    for r in eps_data:
        if r.get("form") not in ("10-Q", "10-K"):
            continue
        start = r.get("start")
        end = r.get("end")
        fy = r.get("fy")
        fp = r.get("fp")
        if not start or not end or fy is None or not fp:
            continue

        try:
            span_days = (_dt.fromisoformat(end) - _dt.fromisoformat(start)).days
        except ValueError:
            continue

        is_quarterly = 75 <= span_days <= 100
        is_annual = 350 <= span_days <= 380
        if not (is_quarterly or is_annual):
            continue

        # Unique key: fiscal year + fiscal period (e.g. (2026, 'Q2')), NOT end_date,
        # since comparative/restated figures share end_date but differ in fy.
        key = (fy, fp, r.get("form"))
        if key not in candidates or r["filed"] > candidates[key]["filed_date"]:
            candidates[key] = {
                "end_date": end,
                "start_date": start,
                "filed_date": r["filed"],
                "value": r["val"],
                "fiscal_year": fy,
                "fiscal_period": fp,
                "form": r.get("form"),
                "span_days": span_days
            }

    records = list(candidates.values())
    # Sort chronologically by actual period end date, not filed date —
    # filed_date can lag for late filers and doesn't reflect period order.
    records.sort(key=lambda r: r["end_date"])
    return records

def compute_eps_surprise_proxy(eps_history: list[dict], as_of_date: str = None) -> dict:
    """
    Computes a simple EPS trend signal using ONLY quarterly (10-Q) records —
    annual 10-K figures are excluded from the trailing average since mixing
    a ~4x-larger annual EPS into a quarterly average corrupts the trend signal.
    The 10-K's annual EPS can still be cross-checked separately if needed, but
    it must never share an averaging window with quarterly figures.
    """
    quarterly = [r for r in eps_history if r["form"] == "10-Q"]

    if as_of_date:
        quarterly = [r for r in quarterly if r["filed_date"] <= as_of_date]

    if len(quarterly) < 5:
        return {"error": "insufficient quarterly EPS history", "n_quarters": len(quarterly)}

    recent = quarterly[-4:]
    prior = quarterly[-8:-4] if len(quarterly) >= 8 else quarterly[:-4]

    if not prior:
        return {"error": "insufficient prior-period history"}

    recent_avg = sum(r["value"] for r in recent) / len(recent)
    prior_avg = sum(r["value"] for r in prior) / len(prior)

    if prior_avg == 0:
        return {"error": "zero prior EPS, cannot compute growth"}

    eps_growth = (recent_avg - prior_avg) / abs(prior_avg)

    latest_qoq = None
    if len(quarterly) >= 2:
        latest = quarterly[-1]["value"]
        prev = quarterly[-2]["value"]
        if prev != 0:
            latest_qoq = (latest - prev) / abs(prev)

    return {
        "trailing_4q_avg_eps": round(recent_avg, 4),
        "prior_4q_avg_eps": round(prior_avg, 4),
        "eps_growth_yoy_proxy": round(eps_growth, 4),
        "latest_qoq_change": round(latest_qoq, 4) if latest_qoq is not None else None,
        "n_quarters_used": len(quarterly),
        "most_recent_filed_date": quarterly[-1]["filed_date"]
    }