import anthropic
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.core.config import ANTHROPIC_API_KEY
from src.models.hypothesis import Hypothesis, HypothesisStatus
from src.models.governance import GovernanceRecord, GovernanceStage
from src.models.alpha_registry import AlphaSignal
from src.models.experiment_queue import ExperimentJob, JobStatus, JobPriority

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

DECISION_SYSTEM = """You are AURUM's Portfolio Lab Decision Engine. Given a stress scenario
impact analysis for a registered alpha, produce a structured research decision.

Decisions:
- HOLD: impact is within acceptable bounds, no action required
- REDUCE: recommend reducing position sizing (specify by how much)
- HEDGE: recommend adding a hedging mechanism (specify what kind)
- RECALIBRATE: signal needs parameter adjustment (specify what to change)
- RETIRE: impact exceeds drawdown tolerance, retire the signal
- RESEARCH: spawn a new research cycle with scenario-aware constraints

Use RETIRE only if estimated portfolio impact exceeds -20% OR circuit breaker
completely fails to protect AND max drawdown would be catastrophic.

Use RESEARCH when the scenario reveals a structural gap that a new hypothesis
could address — e.g. a scenario shows momentum fails in rate spikes, so
generate a rate-hedged momentum hypothesis.

Return ONLY valid JSON, no markdown fences:
{
  "decision": "HOLD | REDUCE | HEDGE | RECALIBRATE | RETIRE | RESEARCH",
  "confidence": <float 0-1>,
  "rationale": "2-3 sentences citing specific impact numbers",
  "action_detail": "specific action — e.g. 'reduce position to 60% of normal size' or 'add rate duration hedge'",
  "research_trigger": {
    "hypothesis_angle": "what new hypothesis to generate, or null if not RESEARCH",
    "scenario_constraint": "what constraint the scenario imposes on the new hypothesis",
    "priority": "high | medium | low"
  } | null
}"""

def make_scenario_decision(
    alpha: AlphaSignal,
    hypothesis: Hypothesis,
    scenario_key: str,
    scenario: dict,
    impact: dict,
    explanation: str
) -> dict:
    """
    Given a scenario impact analysis, produce a structured research decision.
    """
    context = f"""Alpha: {alpha.signal_name}
Signal components: {[c['factor'] for c in (hypothesis.signal_components or [])]}
Backtest Sharpe: {alpha.sharpe_ratio} | Backtest Max DD: {alpha.max_drawdown}
Paper trading Sharpe: {alpha.paper_trading_sharpe}
Decay flagged: {alpha.decay_flag}

Scenario: {scenario['name']}
Description: {scenario['description']}

Impact analysis:
- Estimated portfolio impact: {impact['estimated_portfolio_impact']*100:.1f}%
- Circuit breaker fires: {impact['circuit_breaker_fires']}
- Crowding amplifier: {impact['crowding_amplifier']:.2f}x
- VIX spike: +{impact['vix_spike']} points

AI explanation:
{explanation}

Decision criteria:
- Acceptable single-scenario impact: > -15%
- Circuit breaker should fire for VIX spikes > 18
- Crowding amplifier > 1.2x = elevated synchronous risk
- If paper trading Sharpe already degraded AND scenario impact > -10%: consider RETIRE
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=DECISION_SYSTEM,
        messages=[{"role": "user", "content": context}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)

def apply_scenario_decision(
    db: Session,
    alpha: AlphaSignal,
    hypothesis: Hypothesis,
    decision: dict,
    scenario_key: str,
    scenario_name: str
) -> dict:
    """
    Applies an approved scenario decision to the governance system.
    Returns a summary of what was written.
    """
    gov = db.query(GovernanceRecord).filter_by(
        hypothesis_id=hypothesis.id
    ).first()

    action = decision["decision"]
    notes = (
        f"[Portfolio Lab] Scenario: {scenario_name} | "
        f"Decision: {action} | "
        f"Rationale: {decision['rationale'][:150]} | "
        f"Action: {decision.get('action_detail', 'N/A')[:100]}"
    )

    applied = {
        "action": action,
        "hypothesis_number": hypothesis.hypothesis_number,
        "scenario": scenario_name,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if action == "RETIRE":
        # Retire the alpha and hypothesis
        if gov:
            gov.advance_stage(GovernanceStage.RETIRED, notes=notes)
            gov.failure_reason = decision["rationale"]
            gov.retirement_date = datetime.now(timezone.utc)
        alpha.is_active = False
        alpha.retirement_recommended = True
        hypothesis.status = HypothesisStatus.FAILED
        applied["result"] = "Alpha retired, hypothesis marked failed"

    elif action in ["REDUCE", "HEDGE", "RECALIBRATE"]:
        # Flag in governance as needing attention
        if gov:
            gov.stage_history = (gov.stage_history or []) + [{
                "from_stage": gov.current_stage,
                "to_stage": gov.current_stage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "notes": notes
            }]
        alpha.decay_flag = True
        applied["result"] = f"Flagged for {action}: {decision.get('action_detail', '')}"

    elif action == "RESEARCH":
        # Add to experiment queue with scenario context
        rt = decision.get("research_trigger", {})
        if rt and rt.get("hypothesis_angle"):
            # Create a placeholder for the new research direction
            priority_map = {
                "high": JobPriority.HIGH,
                "medium": JobPriority.NORMAL,
                "low": JobPriority.DEFERRED
            }
            # Write scenario context to governance as a research note
            if gov:
                gov.stage_history = (gov.stage_history or []) + [{
                    "from_stage": gov.current_stage,
                    "to_stage": gov.current_stage,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": (
                        f"[Portfolio Lab → Research] Scenario {scenario_name} triggered "
                        f"new research direction: {rt['hypothesis_angle']}. "
                        f"Constraint: {rt.get('scenario_constraint', '')}. "
                        f"Priority: {rt.get('priority', 'medium')}"
                    )
                }]
            applied["result"] = f"Research spawned: {rt['hypothesis_angle']}"
            applied["research_angle"] = rt["hypothesis_angle"]
            applied["research_constraint"] = rt.get("scenario_constraint", "")
            applied["research_priority"] = rt.get("priority", "medium")

    elif action == "HOLD":
        # Log but no action
        if gov:
            gov.stage_history = (gov.stage_history or []) + [{
                "from_stage": gov.current_stage,
                "to_stage": gov.current_stage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "notes": f"[Portfolio Lab] {scenario_name}: HOLD — {decision['rationale'][:150]}"
            }]
        applied["result"] = "No action — impact within acceptable bounds"

    db.commit()
    return applied

def run_scenario_with_decisions(
    db: Session,
    scenario_keys: list[str] = None
) -> list[dict]:
    """
    Runs all registered alphas through selected scenarios and
    produces decision recommendations (not yet applied).
    """
    from src.agents.portfolio_lab import (
        SCENARIOS, estimate_alpha_scenario_impact, explain_scenario_impact
    )

    if scenario_keys is None:
        scenario_keys = list(SCENARIOS.keys())

    alphas = db.query(AlphaSignal).filter_by(is_active=True).all()
    results = []

    for alpha in alphas:
        hyp = db.query(Hypothesis).filter_by(id=alpha.hypothesis_id).first()
        if not hyp:
            continue

        for scenario_key in scenario_keys:
            scenario = SCENARIOS[scenario_key]
            impact = estimate_alpha_scenario_impact(alpha, hyp, scenario)
            explanation = explain_scenario_impact(alpha, hyp, scenario_key, impact)
            decision = make_scenario_decision(
                alpha, hyp, scenario_key, scenario, impact, explanation
            )

            results.append({
                "alpha_id": str(alpha.id),
                "hypothesis_number": hyp.hypothesis_number,
                "alpha_name": alpha.signal_name,
                "scenario_key": scenario_key,
                "scenario_name": scenario["name"],
                "impact": impact,
                "explanation": explanation,
                "decision": decision,
                "applied": False  # pending approval
            })

    return results