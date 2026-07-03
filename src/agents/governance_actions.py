from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.models.hypothesis import Hypothesis, HypothesisStatus
from src.models.governance import GovernanceRecord, GovernanceStage

# ── Stage transition rules ────────────────────────────────────────
# Each entry: current_stage -> list of allowed next stages
ALLOWED_TRANSITIONS = {
    GovernanceStage.IDEA: [GovernanceStage.EXPERIMENT],
    GovernanceStage.EXPERIMENT: [GovernanceStage.STATISTICAL_REVIEW],
    GovernanceStage.STATISTICAL_REVIEW: [GovernanceStage.RISK_REVIEW],
    GovernanceStage.RISK_REVIEW: [GovernanceStage.COMMITTEE],
    GovernanceStage.COMMITTEE: [
        GovernanceStage.PAPER_TRADING,
        GovernanceStage.RETIRED
    ],
    GovernanceStage.PAPER_TRADING: [
        GovernanceStage.PRODUCTION,
        GovernanceStage.RETIRED
    ],
    GovernanceStage.PRODUCTION: [
        GovernanceStage.MONITORING,
        GovernanceStage.RETIRED
    ],
    GovernanceStage.MONITORING: [GovernanceStage.RETIRED],
    GovernanceStage.RETIRED: [],
}

# ── Prerequisite checks ───────────────────────────────────────────
def check_prerequisites(hypothesis: Hypothesis,
                         gov: GovernanceRecord,
                         target_stage: GovernanceStage) -> tuple[bool, str]:
    """
    Returns (ok, reason) — if ok is False, reason explains what's missing.
    """
    if target_stage == GovernanceStage.STATISTICAL_REVIEW:
        if hypothesis.sharpe_ratio is None:
            return False, "Backtest results required before statistical review. Run the backtester first."

    elif target_stage == GovernanceStage.RISK_REVIEW:
        if not gov.statistical_review:
            return False, "Statistical review must be completed before risk review. Run the statistical validator first."

    elif target_stage == GovernanceStage.COMMITTEE:
        if not gov.debate_record:
            return False, "Committee debate must be run before committee stage. Run the debate engine first."

    elif target_stage == GovernanceStage.PAPER_TRADING:
        if not gov.committee_decision:
            return False, "Committee decision required before paper trading."
        if gov.committee_decision.get("decision") not in ("approve_paper_trading",):
            return False, f"Committee decision was '{gov.committee_decision.get('decision')}', not approved for paper trading."

    elif target_stage == GovernanceStage.PRODUCTION:
        from sqlalchemy import text
        # Check learning reports exist via raw query since we don't have the db here
        # This will be checked at call site instead
        pass

    return True, ""

def advance_stage(db: Session,
                   hypothesis: Hypothesis,
                   target_stage: GovernanceStage,
                   notes: str = "",
                   actor: str = "dashboard") -> tuple[bool, str]:
    """
    Attempts a stage transition. Returns (success, message).
    Enforces allowed transitions and prerequisites.
    """
    gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hypothesis.id).first()
    if not gov:
        return False, "No governance record found for this hypothesis."

    current = gov.current_stage

    # Check transition is allowed
    allowed = ALLOWED_TRANSITIONS.get(current, [])
    if target_stage not in allowed and target_stage != GovernanceStage.RETIRED:
        return False, (f"Cannot advance from {current.value} to {target_stage.value}. "
                       f"Allowed next stages: {[s.value for s in allowed]}")

    # Check prerequisites
    ok, reason = check_prerequisites(hypothesis, gov, target_stage)
    if not ok:
        return False, reason

    # Special handling for paper trading production check
    if target_stage == GovernanceStage.PRODUCTION:
        from sqlalchemy import text
        count = db.execute(
            text("SELECT COUNT(*) FROM learning_reports WHERE hypothesis_id = :hid"),
            {"hid": str(hypothesis.id)}
        ).scalar()
        if count == 0:
            return False, "At least one continuous learning evaluation required before production."

    # Execute the transition
    gov.advance_stage(target_stage, notes=f"[{actor}] {notes}".strip())

    # Sync hypothesis status
    status_map = {
        GovernanceStage.PAPER_TRADING: HypothesisStatus.VALIDATED,
        GovernanceStage.PRODUCTION: HypothesisStatus.VALIDATED,
        GovernanceStage.RETIRED: HypothesisStatus.FAILED,
    }
    if target_stage in status_map:
        hypothesis.status = status_map[target_stage]

    # Set timestamps
    if target_stage == GovernanceStage.PAPER_TRADING:
        if not gov.paper_trading_start:
            gov.paper_trading_start = datetime.now(timezone.utc)
    elif target_stage == GovernanceStage.RETIRED:
        gov.retirement_date = datetime.now(timezone.utc)

    db.commit()
    return True, f"Successfully advanced to {target_stage.value}."

def retire_hypothesis(db: Session,
                       hypothesis: Hypothesis,
                       reason: str,
                       actor: str = "dashboard") -> tuple[bool, str]:
    """Retire a hypothesis from any active stage with a documented reason."""
    gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hypothesis.id).first()
    if not gov:
        return False, "No governance record found."
    if gov.current_stage == GovernanceStage.RETIRED:
        return False, "Hypothesis is already retired."

    gov.failure_reason = reason
    gov.retirement_date = datetime.now(timezone.utc)
    gov.advance_stage(GovernanceStage.RETIRED,
                       notes=f"[{actor}] Retired: {reason}")
    hypothesis.status = HypothesisStatus.FAILED
    db.commit()
    return True, "Hypothesis retired."

def get_allowed_actions(hypothesis: Hypothesis,
                         gov: GovernanceRecord) -> list[dict]:
    """
    Returns the list of actions available from the current stage,
    with prerequisite check results so the UI can show why an action
    is blocked rather than just hiding it.
    """
    current = gov.current_stage
    actions = []

    for target in ALLOWED_TRANSITIONS.get(current, []):
        ok, reason = check_prerequisites(hypothesis, gov, target)
        actions.append({
            "label": f"Advance to {target.value.replace('_', ' ').title()}",
            "target_stage": target,
            "enabled": ok,
            "blocked_reason": reason if not ok else None
        })

    # Retire is always available from active stages
    if current != GovernanceStage.RETIRED:
        actions.append({
            "label": "Retire hypothesis",
            "target_stage": GovernanceStage.RETIRED,
            "enabled": True,
            "blocked_reason": None
        })

    return actions