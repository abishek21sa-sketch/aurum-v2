import anthropic
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.core.config import ANTHROPIC_API_KEY
from src.models.hypothesis import Hypothesis
from src.models.governance import GovernanceRecord
from src.models.research_memory import ResearchMemory
from src.models.alpha_registry import AlphaSignal

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

ANALYST_SYSTEM = """You are the AURUM Learning Analyst. Given a paper-trading
performance report for a registered alpha signal, you reason about WHY the
signal performed differently from its backtest baseline.

You produce:
1. A root cause diagnosis — what mechanistically explains the performance difference
2. A confidence rating — how certain you are of the diagnosis
3. A recommended action — CONTINUE | IMPROVE | RETIRE | RECALIBRATE
4. A new research constraint — if this finding is novel and strong enough to
   warrant updating the research memory corpus

Be specific. Cite the actual numbers. Distinguish between:
- Expected degradation (circuit-breaker cost is expected and acceptable)
- Regime-driven degradation (strategy worked, market changed)
- Signal decay (the factor edge is eroding)
- Structural flaw (the hypothesis design was wrong from the start)

Return ONLY valid JSON, no markdown fences:
{
  "root_cause": "precise mechanistic explanation of performance difference",
  "root_cause_category": "circuit_breaker_cost | regime_change | signal_decay | structural_flaw | expected_variance | unknown",
  "confidence": <float 0-1>,
  "recommended_action": "CONTINUE | IMPROVE | RETIRE | RECALIBRATE",
  "action_rationale": "1-2 sentences on why this action",
  "write_memory": true | false,
  "memory_failure_mode": "failure mode category if write_memory=true, else null",
  "memory_lesson": "specific lesson for future hypothesis generation, or null",
  "memory_constraint": {
    "applies_when": "...",
    "required_action": "..."
  } | null,
  "affected_signal_types": ["list of affected factors"] | null,
  "improvement_suggestions": ["specific changes that could improve the signal"] | null
}"""

def analyze_learning_report(
    db: Session,
    hypothesis: Hypothesis,
    report: dict
) -> dict:
    """
    Reasons over a paper-trading report to produce a diagnosis
    and optionally write a new Research Memory.
    """
    gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hypothesis.id).first()
    alpha = db.query(AlphaSignal).filter_by(hypothesis_id=hypothesis.id).first()

    # Get existing memories to avoid writing duplicates
    existing_memories = db.query(ResearchMemory).all()
    existing_failure_modes = [m.failure_mode for m in existing_memories]
    existing_lessons_preview = [m.lesson[:80] for m in existing_memories]

    context = f"""Alpha signal: {hypothesis.title}
Signal components: {json.dumps([c['factor'] for c in (hypothesis.signal_components or [])])}
Conditions: {json.dumps(hypothesis.conditions)}
Holding period: {hypothesis.expected_holding_days} days

Backtest baseline:
  Sharpe: {hypothesis.sharpe_ratio}
  Sortino: {hypothesis.sortino_ratio}
  Max DD: {hypothesis.max_drawdown}
  Win rate: {hypothesis.win_rate}

Paper trading results ({report.get('simulation_window_days', '?')}d window):
  Paper Sharpe: {report.get('paper_sharpe')}
  Paper Max DD: {report.get('paper_max_drawdown')}
  Paper Win Rate: {report.get('paper_win_rate')}
  Sharpe degradation: {report.get('sharpe_degradation')} ({report.get('sharpe_degradation', 0)*100:.1f}%)
  VIX max observed: {report.get('vix_max_observed')}
  VIX breach occurred: {report.get('vix_breach_occurred')}
  Circuit breaker triggers: {report.get('circuit_breaker_triggers')}
  Recommended action: {report.get('recommended_action')}

Committee debate summary (if available):
  Decision: {gov.committee_decision.get('decision') if gov and gov.committee_decision else 'unknown'}
  Bear objection: {gov.debate_record.get('bear', {}).get('strongest_objection', 'N/A')[:200] if gov and gov.debate_record else 'N/A'}

Existing research memory failure modes (avoid duplicating):
{json.dumps(existing_failure_modes)}

Sample of existing lessons (for context, avoid near-duplicates):
{json.dumps(existing_lessons_preview)}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=ANALYST_SYSTEM,
        messages=[{"role": "user", "content": context}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    analysis = json.loads(raw)

    # Write memory if the analyst recommends it and the finding is novel
    memory_written = None
    if analysis.get("write_memory") and analysis.get("memory_lesson"):
        # Check it's not too similar to existing memories
        new_lesson = analysis["memory_lesson"].lower()
        is_duplicate = any(
            len(set(new_lesson.split()) & set(existing.lower().split())) /
            max(len(set(new_lesson.split())), 1) > 0.6
            for existing in existing_lessons_preview
        )

        if not is_duplicate:
            memory = ResearchMemory(
                id=uuid.uuid4(),
                source_hypothesis_id=hypothesis.id,
                source_hypothesis_number=hypothesis.hypothesis_number,
                failure_mode=analysis.get("memory_failure_mode", "other"),
                conditions=hypothesis.conditions or {},
                lesson=analysis["memory_lesson"],
                structured_constraint=analysis.get("memory_constraint") or {
                    "applies_when": "see lesson",
                    "required_action": "see lesson"
                },
                affected_signal_types=analysis.get("affected_signal_types", []),
                affected_features=[],
                created_by="learning_analyst"
            )
            db.add(memory)
            db.commit()
            db.refresh(memory)
            memory_written = str(memory.id)
            analysis["memory_id"] = memory_written
        else:
            analysis["memory_skipped"] = "too similar to existing memory"

    # Update alpha registry with analyst recommendation
    if alpha:
        if analysis["recommended_action"] == "RETIRE":
            alpha.retirement_recommended = True
        elif analysis["recommended_action"] in ["IMPROVE", "RECALIBRATE"]:
            alpha.decay_flag = True
        alpha.last_evaluated_at = datetime.now(timezone.utc)
        db.commit()

    return analysis

def run_full_registry_analysis(db: Session) -> list[dict]:
    """
    Runs the learning analyst on every alpha that has a learning report.
    Returns structured analyses with optional new memories written.
    """
    from src.agents.continuous_learning import simulate_paper_trading

    alphas = db.query(AlphaSignal).filter_by(is_active=True).all()
    results = []

    for alpha in alphas:
        hyp = db.query(Hypothesis).filter_by(id=alpha.hypothesis_id).first()
        if not hyp:
            continue

        # Get latest learning report
        lr = db.execute(text(
            "SELECT paper_sharpe, paper_max_drawdown, paper_win_rate, "
            "sharpe_degradation, vix_max_observed, vix_breach_occurred, "
            "circuit_breaker_triggers, recommended_action, simulation_window_days "
            "FROM learning_reports WHERE hypothesis_id = :hid "
            "ORDER BY evaluated_at DESC LIMIT 1"
        ), {"hid": str(hyp.id)}).fetchone()

        if not lr:
            print(f"  H#{hyp.hypothesis_number}: no learning report, running simulation first...")
            report = simulate_paper_trading(db, hyp)
            if "error" in report:
                print(f"  H#{hyp.hypothesis_number}: simulation failed — {report['error']}")
                continue
        else:
            report = {
                "paper_sharpe": lr.paper_sharpe,
                "paper_max_drawdown": lr.paper_max_drawdown,
                "paper_win_rate": lr.paper_win_rate,
                "sharpe_degradation": lr.sharpe_degradation,
                "vix_max_observed": lr.vix_max_observed,
                "vix_breach_occurred": lr.vix_breach_occurred,
                "circuit_breaker_triggers": lr.circuit_breaker_triggers,
                "recommended_action": lr.recommended_action,
                "simulation_window_days": lr.simulation_window_days
            }

        print(f"  Analyzing H#{hyp.hypothesis_number}...")
        analysis = analyze_learning_report(db, hyp, report)

        result = {
            "hypothesis_number": hyp.hypothesis_number,
            "title": hyp.title,
            "root_cause": analysis["root_cause"],
            "root_cause_category": analysis["root_cause_category"],
            "confidence": analysis["confidence"],
            "recommended_action": analysis["recommended_action"],
            "action_rationale": analysis["action_rationale"],
            "memory_written": analysis.get("memory_id"),
            "memory_skipped": analysis.get("memory_skipped"),
            "improvement_suggestions": analysis.get("improvement_suggestions", [])
        }
        results.append(result)

        print(f"    Root cause: [{analysis['root_cause_category']}] "
              f"(confidence {analysis['confidence']})")
        print(f"    Action: {analysis['recommended_action']}")
        if analysis.get("memory_id"):
            print(f"    Memory written: {analysis['memory_id']}")
        elif analysis.get("memory_skipped"):
            print(f"    Memory skipped: {analysis['memory_skipped']}")

    return results