import anthropic
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.core.config import ANTHROPIC_API_KEY
from src.models.alpha_registry import AlphaSignal
from src.models.hypothesis import Hypothesis
from src.data.relational_knowledge_store import get_knowledge_store
from src.agents.backtester import fetch_price_data, SP500_UNIVERSE

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Scenario definitions ──────────────────────────────────────────
SCENARIOS = {
    "fed_hike_100bps": {
        "name": "Fed +100bps surprise hike",
        "description": "Fed raises rates 100bps in a single meeting, well above expectations",
        "historical_analogs": ["2022-06-15", "1994-11-15"],
        "factor_shocks": {
            "rates": -0.15,           # rate-sensitive names down ~15%
            "growth": -0.10,          # growth factor repricing
            "value": +0.05,           # value outperforms
            "momentum": -0.12,        # momentum crash risk (high-beta unwind)
            "low_vol": -0.03,         # defensive but still hit
        },
        "sector_impacts": {
            "Information Technology": -0.14,
            "Consumer Discretionary": -0.12,
            "Communication Services": -0.10,
            "Financials": +0.04,
            "Utilities": -0.08,
            "Energy": +0.02,
            "Health Care": -0.04,
            "Consumer Staples": -0.02,
            "Industrials": -0.06,
        },
        "vix_spike": 12.0,
        "duration_days": 21,
    },
    "nvda_minus_40pct": {
        "name": "NVDA -40% idiosyncratic shock",
        "description": "NVDA drops 40% on earnings miss / export controls / competitive threat",
        "historical_analogs": ["2022-09-01"],
        "factor_shocks": {
            "ai_capex": -0.25,
            "momentum": -0.15,        # momentum names unwind as AI thesis questioned
            "earnings_revision": -0.10,
            "low_vol": -0.05,
        },
        "sector_impacts": {
            "Information Technology": -0.12,  # AI/semi contagion
            "Communication Services": -0.06,  # AI spend questioned
            "Consumer Discretionary": -0.02,
            "Financials": 0.0,
            "Health Care": +0.01,
            "Consumer Staples": +0.02,
        },
        "vix_spike": 8.0,
        "duration_days": 10,
        "idiosyncratic_ticker": "NVDA",
        "idiosyncratic_shock": -0.40,
    },
    "flash_crash": {
        "name": "Flash crash (-10% intraday)",
        "description": "Rapid 10% intraday decline, partial recovery, elevated vol for weeks",
        "historical_analogs": ["2010-05-06", "2015-08-24"],
        "factor_shocks": {
            "momentum": -0.20,        # worst factor in flash crashes
            "low_vol": -0.08,
            "liquidity": -0.15,
            "rates": -0.02,
        },
        "sector_impacts": {
            "Information Technology": -0.12,
            "Consumer Discretionary": -0.10,
            "Communication Services": -0.09,
            "Financials": -0.08,
            "Health Care": -0.05,
            "Consumer Staples": -0.04,
            "Utilities": -0.03,
            "Industrials": -0.08,
            "Energy": -0.06,
        },
        "vix_spike": 20.0,
        "duration_days": 5,
    },
    "regional_banking_collapse": {
        "name": "Regional banking collapse",
        "description": "SVB-style contagion: regional banks fail, credit tightens, flight to quality",
        "historical_analogs": ["2023-03-10"],
        "factor_shocks": {
            "credit_spreads": -0.12,
            "rates": +0.03,           # flight to treasuries
            "momentum": -0.08,
            "earnings_revision": -0.06,
        },
        "sector_impacts": {
            "Financials": -0.18,
            "Real Estate": -0.12,
            "Consumer Discretionary": -0.06,
            "Industrials": -0.05,
            "Information Technology": -0.04,
            "Health Care": +0.02,
            "Consumer Staples": +0.03,
            "Utilities": +0.02,
        },
        "vix_spike": 15.0,
        "duration_days": 14,
    },
    "pandemic_shock": {
        "name": "Pandemic-style shock",
        "description": "Rapid 30% market decline over 4 weeks, all sectors hit, then recovery",
        "historical_analogs": ["2020-02-20"],
        "factor_shocks": {
            "momentum": -0.30,
            "rates": +0.05,           # flight to treasuries
            "low_vol": -0.20,
            "earnings_revision": -0.25,
            "consumer_spending": -0.20,
        },
        "sector_impacts": {
            "Information Technology": -0.18,
            "Consumer Discretionary": -0.28,
            "Communication Services": -0.15,
            "Financials": -0.30,
            "Energy": -0.40,
            "Industrials": -0.30,
            "Health Care": -0.10,
            "Consumer Staples": -0.08,
            "Utilities": -0.12,
        },
        "vix_spike": 35.0,
        "duration_days": 30,
    },
}

def estimate_alpha_scenario_impact(
    alpha: AlphaSignal,
    hypothesis: Hypothesis,
    scenario: dict
) -> dict:
    """
    Estimates the impact of a stress scenario on a given alpha signal,
    using the Knowledge Graph for sector/macro exposure and the scenario's
    factor shock map.
    """
    ks = get_knowledge_store()

    # Get the signal's implied top holdings based on sector bias in signal components
    # We use the universe and infer tech-heavy momentum bias from signal type
    signal_factors = [c["factor"] for c in (hypothesis.signal_components or [])]

    # Estimate sector exposure from signal type
    # Momentum + low vol in expansion → tech-heavy, growth-biased
    has_momentum = any("momentum" in f for f in signal_factors)
    has_earnings = any("earnings" in f or "revision" in f for f in signal_factors)
    has_low_vol = any("vol" in f or "volatility" in f for f in signal_factors)

    # Build implied portfolio sector weights
    if has_momentum and has_earnings:
        implied_sectors = {
            "Information Technology": 0.45,
            "Communication Services": 0.20,
            "Consumer Discretionary": 0.15,
            "Health Care": 0.10,
            "Financials": 0.10,
        }
    elif has_momentum:
        implied_sectors = {
            "Information Technology": 0.50,
            "Communication Services": 0.20,
            "Consumer Discretionary": 0.15,
            "Financials": 0.15,
        }
    else:
        implied_sectors = {sector: 1/len(scenario["sector_impacts"])
                          for sector in scenario["sector_impacts"]}

    # Compute weighted portfolio impact
    sector_impacts = scenario.get("sector_impacts", {})
    portfolio_impact = sum(
        weight * sector_impacts.get(sector, -0.05)
        for sector, weight in implied_sectors.items()
    )

    # Factor shock adjustment
    factor_shocks = scenario.get("factor_shocks", {})
    factor_hit = 0.0
    if has_momentum and "momentum" in factor_shocks:
        factor_hit += 0.6 * factor_shocks["momentum"]
    if has_earnings and "earnings_revision" in factor_shocks:
        factor_hit += 0.3 * factor_shocks["earnings_revision"]
    if has_low_vol and "low_vol" in factor_shocks:
        factor_hit += 0.3 * factor_shocks["low_vol"]

    total_impact = portfolio_impact * 0.6 + factor_hit * 0.4

    # VIX circuit-breaker effectiveness
    vix_spike = scenario.get("vix_spike", 0)
    circuit_breaker_fires = vix_spike >= 18.0
    if circuit_breaker_fires and any("circuit_breaker" in f or "vix" in f.lower()
                                      for f in signal_factors):
        circuit_breaker_note = f"VIX spike of {vix_spike} points would trigger circuit-breaker — positions liquidated, limiting downside"
        total_impact *= 0.4  # circuit breaker reduces loss
    else:
        circuit_breaker_note = f"VIX spike of {vix_spike} points — no circuit-breaker in this signal"

    # ETF crowding amplification
    top_tickers = ["NVDA", "AMD", "META", "MSFT", "AAPL"]
    crowding = ks.get_common_etf_exposure(top_tickers)
    high_crowding_etfs = [etf for etf, info in crowding.items()
                          if info["concentration_risk"] == "high"]
    crowding_amplifier = 1.0 + (0.1 * len(high_crowding_etfs) / 3)

    # Idiosyncratic shock (e.g. NVDA -40%)
    idiosyncratic_note = ""
    if "idiosyncratic_ticker" in scenario:
        ticker = scenario["idiosyncratic_ticker"]
        shock = scenario["idiosyncratic_shock"]
        etf_exposure = ks.get_etf_exposure(ticker)
        etf_names = [e["etf"] for e in etf_exposure]
        idiosyncratic_note = (f"{ticker} {shock*100:.0f}% shock propagates through "
                             f"{', '.join(etf_names[:3])} ETF holdings")

    final_impact = total_impact * crowding_amplifier

    return {
        "scenario_name": scenario["name"],
        "estimated_portfolio_impact": round(final_impact, 4),
        "sector_contribution": round(portfolio_impact, 4),
        "factor_contribution": round(factor_hit, 4),
        "crowding_amplifier": round(crowding_amplifier, 4),
        "vix_spike": vix_spike,
        "circuit_breaker_fires": circuit_breaker_fires,
        "circuit_breaker_note": circuit_breaker_note,
        "idiosyncratic_note": idiosyncratic_note,
        "implied_sectors": implied_sectors,
        "duration_days": scenario.get("duration_days", 21),
    }

EXPLAINER_SYSTEM = """You are AURUM's Portfolio Lab analyst. Given a stress scenario impact
analysis for a specific alpha signal, explain:
1. The mechanism by which this scenario affects the signal (2 sentences)
2. Whether the signal's design provides any protection (1 sentence)
3. The recommended response if this scenario materializes (2 sentences)

Be specific, cite the actual numbers provided, and reference the signal's actual components.
Keep total response under 200 words. Write in plain prose, no headers or bullets."""

def explain_scenario_impact(
    alpha: AlphaSignal,
    hypothesis: Hypothesis,
    scenario_key: str,
    impact: dict
) -> str:
    scenario = SCENARIOS[scenario_key]
    context = f"""Alpha signal: {alpha.signal_name}
Signal components: {[c['factor'] for c in (hypothesis.signal_components or [])]}
Backtest Sharpe: {alpha.sharpe_ratio} | Max DD: {alpha.max_drawdown}

Scenario: {scenario['name']}
Description: {scenario['description']}
Historical analogs: {scenario['historical_analogs']}

Impact analysis:
- Estimated portfolio impact: {impact['estimated_portfolio_impact']*100:.1f}%
- Sector contribution: {impact['sector_contribution']*100:.1f}%
- Factor contribution: {impact['factor_contribution']*100:.1f}%
- VIX spike: +{impact['vix_spike']} points
- Circuit breaker fires: {impact['circuit_breaker_fires']}
- {impact['circuit_breaker_note']}
{f"- {impact['idiosyncratic_note']}" if impact['idiosyncratic_note'] else ""}
- Crowding amplifier: {impact['crowding_amplifier']:.2f}x (high ETF overlap)
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=EXPLAINER_SYSTEM,
        messages=[{"role": "user", "content": context}]
    )
    return response.content[0].text.strip()

def run_portfolio_lab(db: Session, scenario_keys: list[str] = None) -> list[dict]:
    """Run all registered alphas through selected scenarios."""
    if scenario_keys is None:
        scenario_keys = list(SCENARIOS.keys())

    alphas = db.query(AlphaSignal).filter_by(is_active=True).all()
    if not alphas:
        return []

    results = []
    for alpha in alphas:
        hyp = db.query(Hypothesis).filter_by(id=alpha.hypothesis_id).first()
        if not hyp:
            continue

        alpha_results = {
            "alpha_name": alpha.signal_name,
            "hypothesis_number": hyp.hypothesis_number,
            "backtest_sharpe": alpha.sharpe_ratio,
            "scenarios": {}
        }

        for scenario_key in scenario_keys:
            scenario = SCENARIOS[scenario_key]
            impact = estimate_alpha_scenario_impact(alpha, hyp, scenario)
            explanation = explain_scenario_impact(alpha, hyp, scenario_key, impact)
            alpha_results["scenarios"][scenario_key] = {
                **impact,
                "explanation": explanation
            }

        results.append(alpha_results)

    return results