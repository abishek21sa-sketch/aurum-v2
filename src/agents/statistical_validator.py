import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.models.hypothesis import Hypothesis
from src.models.governance import GovernanceRecord, GovernanceStage
from src.agents.backtester import (
    fetch_price_data, build_composite_signal, SP500_UNIVERSE
)

def compute_return_series(hypothesis: Hypothesis, close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """Rebuild the actual non-overlapping period return series for statistical testing."""
    composite = build_composite_signal(close, volume, hypothesis.signal_components)

    n_long = max(1, int(len(composite) * 0.2))
    long_tickers = composite.nlargest(n_long).index.tolist()
    available = [t for t in long_tickers if t in close.columns]
    portfolio_close = close[available]

    holding_days = hypothesis.expected_holding_days or 21
    rebalance_idx = portfolio_close.index[::holding_days]
    rebalance_close = portfolio_close.loc[rebalance_idx]
    fwd_returns = rebalance_close.pct_change().dropna()

    return fwd_returns.mean(axis=1)

def t_test_against_zero(returns: pd.Series) -> dict:
    """One-sample t-test: are mean returns statistically different from zero?"""
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    return {
        "t_stat": float(round(t_stat, 4)),
        "p_value": float(round(p_value, 4)),
        "significant_at_5pct": bool(p_value < 0.05),
        "n_observations": int(len(returns))
    }

def sharpe_significance(returns: pd.Series, periods_per_year: float) -> dict:
    """
    Statistical significance of the Sharpe ratio itself, using the
    standard asymptotic approximation (Lo, 2002 simplified form).
    """
    n = len(returns)
    sharpe_period = returns.mean() / returns.std() if returns.std() > 0 else 0
    sharpe_annual = sharpe_period * np.sqrt(periods_per_year)

    # Standard error of Sharpe ratio estimate
    se_sharpe = np.sqrt((1 + 0.5 * sharpe_period**2) / n)
    sharpe_t_stat = sharpe_period / se_sharpe if se_sharpe > 0 else 0
    sharpe_p_value = 2 * (1 - stats.norm.cdf(abs(sharpe_t_stat)))

    return {
        "annualized_sharpe": float(round(sharpe_annual, 4)),
        "sharpe_t_stat": float(round(sharpe_t_stat, 4)),
        "sharpe_p_value": float(round(sharpe_p_value, 4)),
        "sharpe_significant_at_5pct": bool(sharpe_p_value < 0.05)
    }

def stability_check(returns: pd.Series, n_splits: int = 3) -> dict:
    """
    Split the return series into N consecutive chunks and check
    whether the Sharpe ratio is consistent across chunks (not driven
    by one lucky period).
    """
    if len(returns) < n_splits * 4:
        return {"error": "insufficient observations for stability check", "passed": False}

    chunks = np.array_split(returns, n_splits)
    chunk_sharpes = []
    for chunk in chunks:
        if chunk.std() > 0:
            chunk_sharpes.append(float(chunk.mean() / chunk.std()))
        else:
            chunk_sharpes.append(0.0)

    positive_chunks = sum(1 for s in chunk_sharpes if s > 0)
    stability_ratio = positive_chunks / n_splits

    return {
        "chunk_sharpes": [round(s, 4) for s in chunk_sharpes],
        "positive_periods": positive_chunks,
        "total_periods": n_splits,
        "stability_ratio": round(stability_ratio, 4),
        "passed": stability_ratio >= (2/3)  # at least 2 of 3 periods positive
    }

def sample_size_check(returns: pd.Series, min_observations: int = 30) -> dict:
    """Flag if there simply aren't enough independent observations to trust the result."""
    n = len(returns)
    return {
        "n_observations": n,
        "minimum_required": min_observations,
        "passed": n >= min_observations,
        "note": "Fewer than 30 non-overlapping holding periods makes Sharpe/Sortino unreliable" if n < min_observations else "Adequate sample size"
    }

def run_statistical_validation(db: Session, hypothesis: Hypothesis) -> dict:
    print(f"  Rebuilding return series for statistical tests...")

    close, volume = fetch_price_data(SP500_UNIVERSE, "2018-01-01",
                                       datetime.now().strftime("%Y-%m-%d"))
    returns = compute_return_series(hypothesis, close, volume)

    holding_days = hypothesis.expected_holding_days or 21
    periods_per_year = 252 / holding_days

    print(f"  Running t-test against zero...")
    t_test = t_test_against_zero(returns)

    print(f"  Testing Sharpe ratio significance...")
    sharpe_sig = sharpe_significance(returns, periods_per_year)

    print(f"  Checking stability across sub-periods...")
    stability = stability_check(returns)

    print(f"  Checking sample size adequacy...")
    sample_size = sample_size_check(returns)

    # Overall pass/fail
    checks_passed = [
        t_test["significant_at_5pct"],
        sharpe_sig["sharpe_significant_at_5pct"],
        stability.get("passed", False),
        sample_size["passed"]
    ]
    overall_passed = sum(checks_passed) >= 3  # require 3 of 4 checks to pass

    review = {
        "t_test": t_test,
        "sharpe_significance": sharpe_sig,
        "stability": stability,
        "sample_size": sample_size,
        "checks_passed": sum(checks_passed),
        "checks_total": 4,
        "overall_passed": overall_passed,
        "reviewed_at": datetime.now(timezone.utc).isoformat()
    }

    # Write to governance record
    gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hypothesis.id).first()
    if gov:
        gov.statistical_review = review
        notes = (f"Statistical review {'PASSED' if overall_passed else 'FAILED'} "
                 f"({sum(checks_passed)}/4 checks). "
                 f"t-stat={t_test['t_stat']}, p={t_test['p_value']}, "
                 f"Sharpe p={sharpe_sig['sharpe_p_value']}")

        if overall_passed:
            gov.advance_stage(GovernanceStage.RISK_REVIEW, notes=notes)
        else:
            notes += " — Held at statistical review, did not advance."
            gov.stage_history = (gov.stage_history or []) + [{
                "from_stage": gov.current_stage,
                "to_stage": gov.current_stage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "notes": notes
            }]
        db.commit()

    return review

def print_validation(review: dict):
    print(f"\n{'='*60}")
    print(f"STATISTICAL VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Overall: {'PASSED' if review['overall_passed'] else 'FAILED'} "
          f"({review['checks_passed']}/{review['checks_total']} checks)")

    print(f"\n1. T-test against zero:")
    print(f"   t-stat: {review['t_test']['t_stat']}, p-value: {review['t_test']['p_value']}")
    print(f"   Significant: {review['t_test']['significant_at_5pct']}")

    print(f"\n2. Sharpe ratio significance:")
    print(f"   Annualized Sharpe: {review['sharpe_significance']['annualized_sharpe']}")
    print(f"   t-stat: {review['sharpe_significance']['sharpe_t_stat']}, "
          f"p-value: {review['sharpe_significance']['sharpe_p_value']}")
    print(f"   Significant: {review['sharpe_significance']['sharpe_significant_at_5pct']}")

    print(f"\n3. Stability across sub-periods:")
    if "error" not in review['stability']:
        print(f"   Chunk Sharpes: {review['stability']['chunk_sharpes']}")
        print(f"   Positive periods: {review['stability']['positive_periods']}/{review['stability']['total_periods']}")
        print(f"   Passed: {review['stability']['passed']}")
    else:
        print(f"   {review['stability']['error']}")

    print(f"\n4. Sample size:")
    print(f"   N observations: {review['sample_size']['n_observations']} "
          f"(min required: {review['sample_size']['minimum_required']})")
    print(f"   Passed: {review['sample_size']['passed']}")
    print(f"   {review['sample_size']['note']}")
    print(f"{'='*60}\n")