import os
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# ── Series we track ───────────────────────────────────────────────
SERIES = {
    "FEDFUNDS":  "Federal Funds Rate (monthly avg)",
    "CPIAUCSL":  "CPI All Items (monthly, seasonally adjusted)",
    "UNRATE":    "Unemployment Rate (monthly)",
    "T10Y2Y":    "10Y-2Y Treasury Spread (daily)",
    "DGS10":     "10-Year Treasury Yield (daily)",
    "DGS2":      "2-Year Treasury Yield (daily)",
    "DCOILWTICO":"WTI Crude Oil Price (daily)",
    "VIXCLS":    "VIX Close (daily)",
}

def _fetch(endpoint: str, params: dict) -> dict | None:
    params["api_key"] = FRED_API_KEY
    params["file_type"] = "json"
    try:
        r = requests.get(f"{FRED_BASE_URL}/{endpoint}",
                        params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        print(f"  FRED {endpoint} failed: HTTP {r.status_code}")
        return None
    except requests.RequestException as e:
        print(f"  FRED request error: {e}")
        return None

def get_latest_value(series_id: str) -> dict | None:
    """Fetch the most recent observation for a series."""
    data = _fetch("series/observations", {
        "series_id": series_id,
        "sort_order": "desc",
        "limit": 1,
        "observation_start": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    })
    if not data or not data.get("observations"):
        return None
    obs = data["observations"][0]
    if obs["value"] == ".":
        return None
    return {
        "series_id": series_id,
        "value": float(obs["value"]),
        "date": obs["date"],
        "description": SERIES.get(series_id, series_id)
    }

def get_recent_values(series_id: str, n: int = 6) -> list[dict]:
    """Fetch the last N observations for trend analysis."""
    # Use 18 months lookback to ensure enough history for monthly series
    lookback_days = 548 if n >= 12 else 365
    data = _fetch("series/observations", {
        "series_id": series_id,
        "sort_order": "desc",
        "limit": n,
        "observation_start": (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    })
    if not data or not data.get("observations"):
        return []
    results = []
    for obs in data["observations"]:
        if obs["value"] != ".":
            results.append({
                "date": obs["date"],
                "value": float(obs["value"])
            })
    return list(reversed(results))  # chronological order

def get_macro_snapshot() -> dict:
    """
    Fetches current values for all tracked macro series and
    computes a structured regime classification.
    """
    print("  Fetching FRED macro data...")
    snapshot = {}

    # Fetch latest values
    for series_id in SERIES:
        val = get_latest_value(series_id)
        if val:
            snapshot[series_id] = val
        time.sleep(0.1)  # rate limit courtesy

    if not snapshot:
        return {"error": "No FRED data retrieved"}

    # Compute regime signals from raw data
    fed_funds = snapshot.get("FEDFUNDS", {}).get("value")
    cpi = snapshot.get("CPIAUCSL", {}).get("value")
    unrate = snapshot.get("UNRATE", {}).get("value")
    spread = snapshot.get("T10Y2Y", {}).get("value")
    dgs10 = snapshot.get("DGS10", {}).get("value")
    vix = snapshot.get("VIXCLS", {}).get("value")
    oil = snapshot.get("DCOILWTICO", {}).get("value")

    # Rate environment — compare to 6-month trend
    fedfunds_trend = get_recent_values("FEDFUNDS", 6)
    if len(fedfunds_trend) >= 2:
        rate_change = fedfunds_trend[-1]["value"] - fedfunds_trend[0]["value"]
        if rate_change > 0.25:
            rate_env = "rising"
        elif rate_change < -0.25:
            rate_env = "falling"
        else:
            rate_env = "stable"
    else:
        rate_env = "unknown"

    # CPI trend — YoY not directly available, use 6-month trend
    cpi_trend = get_recent_values("CPIAUCSL", 13)  # 13 months for YoY
    if len(cpi_trend) >= 6:
        # Use whatever span we have — annualize if < 12 months
        months_span = len(cpi_trend) - 1
        raw_change = (cpi_trend[-1]["value"] / cpi_trend[0]["value"] - 1)
        yoy_cpi = raw_change * (12 / months_span) * 100
        if yoy_cpi > 4.0:
            inflation = "high"
        elif yoy_cpi > 2.5:
            inflation = "moderate"
        elif yoy_cpi > 0:
            inflation = "low"
        else:
            inflation = "deflating"
    else:
        yoy_cpi = None
        inflation = "unknown"

    # Yield curve
    if spread is not None:
        if spread > 0.5:
            yield_curve = "normal"
        elif spread > 0:
            yield_curve = "flat"
        else:
            yield_curve = "inverted"
    else:
        yield_curve = "unknown"

    # VIX regime
    if vix is not None:
        if vix < 15:
            vix_regime = "low"
        elif vix < 25:
            vix_regime = "elevated"
        else:
            vix_regime = "stressed"
    else:
        vix_regime = "unknown"

    # Macro regime — simplified classification
    if unrate is not None and unrate < 4.5 and inflation in ["low", "moderate"]:
        if rate_env == "falling" or rate_env == "stable":
            regime = "expansion"
        else:
            regime = "late_cycle"
    elif unrate is not None and unrate > 5.5:
        regime = "contraction"
    else:
        regime = "transition"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "raw": {k: v for k, v in snapshot.items()},
        "regime": regime,
        "rate_environment": rate_env,
        "inflation_pressure": inflation,
        "yield_curve": yield_curve,
        "vix_regime": vix_regime,
        "key_levels": {
            "fed_funds_rate": fed_funds,
            "unemployment_rate": unrate,
            "ten_year_yield": dgs10,
            "yield_spread_10y2y": spread,
            "vix": vix,
            "oil_price": oil,
            "cpi_yoy_pct": round(yoy_cpi, 2) if yoy_cpi else None
        }
    }

def macro_snapshot_to_observations(snapshot: dict,
                                    include_calendar: bool = True) -> dict:
    """
    Converts a FRED macro snapshot into the observation dict format
    that the Research Scientist expects — replacing manual text strings
    with real computed macro context.
    """
    if "error" in snapshot:
        return {}

    kl = snapshot.get("key_levels", {})
    regime = snapshot.get("regime", "unknown")
    rate_env = snapshot.get("rate_environment", "unknown")
    inflation = snapshot.get("inflation_pressure", "unknown")
    yield_curve = snapshot.get("yield_curve", "unknown")
    vix = kl.get("vix")
    fed = kl.get("fed_funds_rate")
    spread = kl.get("yield_spread_10y2y")
    unrate = kl.get("unemployment_rate")
    oil = kl.get("oil_price")

    obs = {
        "macro_regime": (
            f"{regime}, unemployment {unrate}%, "
            f"yield curve {yield_curve} (10Y-2Y spread {spread:.2f}%)"
            if unrate and spread else regime
        ),
        "rate_environment": (
            f"{rate_env}, fed funds {fed}%, "
            f"10Y yield {kl.get('ten_year_yield')}%"
            if fed else rate_env
        ),
        "inflation_signal": (
            f"{inflation} inflation, CPI YoY {kl.get('cpi_yoy_pct')}%"
            if kl.get("cpi_yoy_pct") else inflation
        ),
        "volatility_regime": (
            f"{'low' if vix and vix < 15 else 'elevated' if vix and vix < 25 else 'stressed'} "
            f"volatility, VIX {vix:.1f}"
            if vix else "unknown volatility"
        ),
        "energy_signal": (
            f"WTI crude ${oil:.1f}/bbl"
            if oil else "oil data unavailable"
        ),
    }

    # Add economic calendar context
    if include_calendar:
        try:
            events = get_upcoming_releases(days_ahead=14)
            ctx = calendar_to_risk_context(events)
            risk = ctx["near_term_risk"]
            critical = ctx["critical_events_7d"]
            if critical:
                event_names = ", ".join(
                    f"{e['name']} ({e['days_until']}d)" for e in critical
                )
                obs["event_risk"] = (
                    f"ELEVATED — {event_names} within 7 days. "
                    f"Consider shorter holding periods or deferred entry."
                )
            elif events:
                next_e = events[0]
                obs["event_risk"] = (
                    f"{risk} — next major event: "
                    f"{next_e['name']} in {next_e['days_until']} days"
                )
            else:
                obs["event_risk"] = "low — no major events in next 14 days"
        except Exception as e:
            obs["event_risk"] = "calendar unavailable"

    return obs

# ── Economic Calendar ─────────────────────────────────────────────

# FRED release IDs for key macro events
CALENDAR_RELEASES = {
    10:  {"name": "CPI", "importance": "high",
          "signal": "inflation", "typical_vix_impact": "medium"},
    50:  {"name": "NFP (Jobs Report)", "importance": "high",
          "signal": "employment", "typical_vix_impact": "high"},
    175: {"name": "GDP", "importance": "high",
          "signal": "growth", "typical_vix_impact": "medium"},
    180: {"name": "FOMC Statement", "importance": "critical",
          "signal": "rates", "typical_vix_impact": "high"},
    46:  {"name": "PPI", "importance": "medium",
          "signal": "inflation_leading", "typical_vix_impact": "low"},
    9:   {"name": "Retail Sales", "importance": "medium",
          "signal": "consumer", "typical_vix_impact": "low"},
}

# Known FOMC meeting dates for 2026 (FRED ID 180 returns too many dates)
# Source: Federal Reserve published schedule
FOMC_MEETING_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06",
    "2026-06-17", "2026-07-29", "2026-09-16",
    "2026-10-28", "2026-12-16"
]

def get_upcoming_releases(days_ahead: int = 30) -> list[dict]:
    """
    Fetch upcoming economic releases within the next N days.
    Returns a list of events sorted by date, with market impact metadata.
    """
    from datetime import datetime, timedelta

    today = datetime.now()
    end_date = today + timedelta(days=days_ahead)
    today_str = today.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    events = []

    for release_id, meta in CALENDAR_RELEASES.items():
        # Skip FOMC from API — use our hardcoded schedule instead
        if release_id == 180:
            continue

        data = _fetch("release/dates", {
            "release_id": release_id,
            "realtime_start": today_str,
            "realtime_end": end_str,
            "sort_order": "asc",
            "include_release_dates_with_no_data": "true",
            "limit": 5
        })

        if data and data.get("release_dates"):
            for entry in data["release_dates"]:
                date_str = entry["date"]
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                days_until = (event_date - today).days
                if 0 <= days_until <= days_ahead:
                    events.append({
                        "date": date_str,
                        "days_until": days_until,
                        "name": meta["name"],
                        "importance": meta["importance"],
                        "signal": meta["signal"],
                        "typical_vix_impact": meta["typical_vix_impact"],
                        "release_id": release_id
                    })
        time.sleep(0.1)

    # Add FOMC meeting dates from hardcoded schedule
    for date_str in FOMC_MEETING_DATES_2026:
        event_date = datetime.strptime(date_str, "%Y-%m-%d")
        days_until = (event_date - today).days
        if 0 <= days_until <= days_ahead:
            events.append({
                "date": date_str,
                "days_until": days_until,
                "name": "FOMC Meeting Decision",
                "importance": "critical",
                "signal": "rates",
                "typical_vix_impact": "high",
                "release_id": 180
            })

    return sorted(events, key=lambda x: x["date"])

def calendar_to_risk_context(events: list[dict]) -> dict:
    """
    Converts upcoming events into a structured risk context
    that the Research Scientist and Portfolio Lab can use.
    """
    if not events:
        return {
            "near_term_risk": "low",
            "next_event": None,
            "critical_events_7d": [],
            "summary": "No major macro events in the next 30 days."
        }

    critical_7d = [e for e in events if e["days_until"] <= 7
                   and e["importance"] in ["high", "critical"]]
    next_event = events[0] if events else None

    # Risk level
    if any(e["importance"] == "critical" and e["days_until"] <= 3 for e in events):
        risk_level = "critical"
    elif critical_7d:
        risk_level = "elevated"
    elif events:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Summary
    if critical_7d:
        names = ", ".join(e["name"] for e in critical_7d)
        summary = (f"{len(critical_7d)} high-impact event(s) within 7 days: {names}. "
                  f"New hypothesis deployment and position entry should account for "
                  f"event risk — consider reducing holding period or deferring entry.")
    else:
        next_name = next_event["name"] if next_event else "unknown"
        next_days = next_event["days_until"] if next_event else "?"
        summary = (f"Next major event: {next_name} in {next_days} day(s). "
                  f"Macro calendar risk is manageable.")

    return {
        "near_term_risk": risk_level,
        "next_event": next_event,
        "critical_events_7d": critical_7d,
        "all_events": events,
        "summary": summary
    }