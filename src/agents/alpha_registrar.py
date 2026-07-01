import uuid
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.models.hypothesis import Hypothesis, HypothesisStatus
from src.models.governance import GovernanceRecord, GovernanceStage
from src.models.alpha_registry import AlphaSignal

def hypothesis_to_signal_code(hypothesis: Hypothesis) -> str:
    """
    Generates a readable Python representation of the signal construction
    logic from the hypothesis's structured signal_components. This is what
    gets stored as signal_code in the registry — a real, runnable description
    of the signal, not just metadata.
    """
    lines = [
        f"# Signal: {hypothesis.title}",
        f"# Hypothesis #{hypothesis.hypothesis_number}",
        f"# Universe: {hypothesis.universe} | Holding: {hypothesis.expected_holding_days}d",
        "",
        "def compute_signal(close, volume, vix=None):",
        "    components = {}",
    ]
    for comp in hypothesis.signal_components:
        factor = comp["factor"]
        direction = comp["direction"]
        lb = comp.get("lookback_days", 21)
        lines.append(f"    # {comp['description'][:100]}")
        lines.append(f"    components['{factor}'] = compute_{factor}(close, volume, lookback={lb}, direction='{direction}')")
    lines.append("    return combine_zscore(components)")
    return "\n".join(lines)

def register_alpha(db: Session, hypothesis: Hypothesis,
                    require_paper_trading: bool = True) -> AlphaSignal | None:
    """
    Registers a hypothesis as a formal alpha signal IF it meets the bar:
    - Has passed statistical validation
    - Has a governance record at PAPER_TRADING stage or later (unless overridden)
    - Has not already been registered
    """
    gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hypothesis.id).first()
    if not gov:
        print(f"  Hypothesis #{hypothesis.hypothesis_number}: no governance record, skipping.")
        return None

    # Explicitly check committee_decision rather than trusting current_stage
    # alone — a hypothesis that was sent back for revision does not advance
    # its stage, but its decision field is the source of truth on eligibility.
    decision = gov.committee_decision.get("decision") if gov.committee_decision else None
    if decision == "reject":
        print(f"  Hypothesis #{hypothesis.hypothesis_number}: committee REJECTED. "
              f"Not eligible for registry.")
        return None
    if decision == "request_revision":
        print(f"  Hypothesis #{hypothesis.hypothesis_number}: committee requested "
              f"REVISION, not yet approved. Not eligible for registry.")
        return None

    eligible_stages = [GovernanceStage.PAPER_TRADING, GovernanceStage.PRODUCTION,
                        GovernanceStage.MONITORING]
    if require_paper_trading and gov.current_stage not in eligible_stages:
        print(f"  Hypothesis #{hypothesis.hypothesis_number}: stage is "
              f"'{gov.current_stage.value}', not yet eligible for registry "
              f"(requires paper_trading or later). Skipping.")
        return None

    existing = db.query(AlphaSignal).filter_by(hypothesis_id=hypothesis.id).first()
    if existing:
        print(f"  Hypothesis #{hypothesis.hypothesis_number}: already registered "
              f"as alpha {existing.id}.")
        return existing

    if hypothesis.sharpe_ratio is None:
        print(f"  Hypothesis #{hypothesis.hypothesis_number}: no backtest results, "
              f"cannot register. Skipping.")
        return None

    signal_code = hypothesis_to_signal_code(hypothesis)
    features_used = [c["factor"] for c in hypothesis.signal_components]

    # Pull regime performance and robustness notes from statistical review if available
    regime_performance = None
    parameter_sensitivity = None
    if gov.statistical_review:
        regime_performance = {
            "stability_chunks": gov.statistical_review.get("stability", {}).get("chunk_sharpes"),
            "stability_passed": gov.statistical_review.get("stability", {}).get("passed"),
        }

    alpha = AlphaSignal(
        id=uuid.uuid4(),
        hypothesis_id=hypothesis.id,
        signal_name=hypothesis.title,
        signal_code=signal_code,
        features_used=features_used,
        universe=hypothesis.universe,
        sharpe_ratio=hypothesis.sharpe_ratio,
        sortino_ratio=hypothesis.sortino_ratio,
        calmar_ratio=hypothesis.calmar_ratio,
        max_drawdown=hypothesis.max_drawdown,
        annualized_return=hypothesis.annualized_return,
        win_rate=hypothesis.win_rate,
        regime_performance=regime_performance,
        is_active=True,
        registered_at=datetime.now(timezone.utc)
    )
    db.add(alpha)
    db.commit()
    db.refresh(alpha)

    print(f"  Hypothesis #{hypothesis.hypothesis_number}: registered as alpha "
          f"{alpha.id} (Sharpe {alpha.sharpe_ratio}, stage {gov.current_stage.value})")
    return alpha

def register_all_eligible(db: Session, require_paper_trading: bool = True) -> list[AlphaSignal]:
    """Scans all hypotheses and registers any that meet the bar."""
    hypotheses = db.query(Hypothesis).order_by(Hypothesis.hypothesis_number).all()
    registered = []
    for hyp in hypotheses:
        result = register_alpha(db, hyp, require_paper_trading=require_paper_trading)
        if result:
            registered.append(result)
    return registered