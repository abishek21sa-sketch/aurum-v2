import anthropic
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.core.config import ANTHROPIC_API_KEY
from src.models.hypothesis import Hypothesis
from src.models.experiment_queue import ExperimentJob, JobStatus, JobPriority
from src.models.governance import GovernanceRecord, GovernanceStage
from src.models.alpha_registry import AlphaSignal
from src.models.research_memory import ResearchMemory

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Scoring weights ───────────────────────────────────────────────
# Higher = more likely to be prioritized for immediate execution
SCORE_WEIGHTS = {
    "has_strong_backtest": 0.25,      # Sharpe > 1.0
    "passed_stat_validation": 0.20,   # 4/4 checks
    "memory_compliance_high": 0.15,   # compliance > 80%
    "novel_signal_components": 0.15,  # not a near-duplicate
    "addresses_known_failure": 0.10,  # resolves a research memory
    "short_holding_period": 0.05,     # faster to validate
    "low_compute_cost": 0.05,         # fewer signal components
    "not_stuck_in_stage": 0.05,       # not been sitting too long
}

SIMILARITY_THRESHOLD = 0.7  # above this → consider merge/cancel

SCHEDULER_SYSTEM = """You are the AURUM Research Scheduler. Given a list of pending
experiments and the current research corpus, make scheduling decisions for each experiment.

For each experiment you must decide ONE of:
- RUN: execute immediately (high value, novel, passes constraints)
- DELAY: defer to next cycle (lower value but worth keeping)
- CANCEL: remove from queue (duplicate, superseded, or violates constraints)
- MERGE: consolidate with another experiment (near-duplicate, combine signals)

Return ONLY valid JSON, no markdown fences:
{
  "decisions": [
    {
      "hypothesis_number": <int>,
      "action": "RUN | DELAY | CANCEL | MERGE",
      "priority_score": <float 0-1>,
      "reasoning": "1-2 sentences",
      "merge_with": <int or null>,
      "merge_rationale": "why merge, or null"
    }
  ],
  "summary": "1-2 sentences on overall scheduling strategy this cycle"
}"""

def compute_priority_score(
    hypothesis: Hypothesis,
    gov: GovernanceRecord,
    all_hypotheses: list[Hypothesis],
    memories: list[ResearchMemory]
) -> float:
    """
    Compute a 0-1 priority score for a hypothesis based on
    evidence quality, novelty, and research value.
    """
    score = 0.0

    # Strong backtest
    if hypothesis.sharpe_ratio and hypothesis.sharpe_ratio > 1.0:
        score += SCORE_WEIGHTS["has_strong_backtest"]

    # Passed statistical validation
    if gov.statistical_review and gov.statistical_review.get("overall_passed"):
        score += SCORE_WEIGHTS["passed_stat_validation"]

    # Memory compliance
    compliance_entries = [
        e for e in (gov.stage_history or [])
        if "compliance" in e.get("notes", "").lower()
    ]
    if compliance_entries:
        latest = compliance_entries[-1].get("notes", "")
        if "COMPLIANCE_WARNING" not in latest:
            score += SCORE_WEIGHTS["memory_compliance_high"]
    else:
        # No compliance issues recorded
        score += SCORE_WEIGHTS["memory_compliance_high"] * 0.5

    # Novel signal components (not just a parameter variant)
    own_factors = set(c["factor"] for c in (hypothesis.signal_components or []))
    duplicate_count = 0
    for other in all_hypotheses:
        if other.id == hypothesis.id:
            continue
        other_factors = set(c["factor"] for c in (other.signal_components or []))
        overlap = len(own_factors & other_factors) / max(len(own_factors), 1)
        if overlap >= SIMILARITY_THRESHOLD:
            duplicate_count += 1
    if duplicate_count == 0:
        score += SCORE_WEIGHTS["novel_signal_components"]
    elif duplicate_count == 1:
        score += SCORE_WEIGHTS["novel_signal_components"] * 0.5

    # Addresses a known failure mode from research memory
    memory_failure_modes = set(m.failure_mode for m in memories)
    hypothesis_text = f"{hypothesis.title} {hypothesis.thesis}".lower()
    if any(mode in hypothesis_text for mode in
           ["timing_mismatch", "circuit_breaker", "selection_bias", "overfitting"]):
        score += SCORE_WEIGHTS["addresses_known_failure"]

    # Short holding period (faster feedback loop)
    holding = hypothesis.expected_holding_days or 21
    if holding <= 10:
        score += SCORE_WEIGHTS["short_holding_period"]
    elif holding <= 15:
        score += SCORE_WEIGHTS["short_holding_period"] * 0.5

    # Fewer signals = lower compute cost
    n_signals = len(hypothesis.signal_components or [])
    if n_signals <= 3:
        score += SCORE_WEIGHTS["low_compute_cost"]

    # Not stuck (stage recently changed)
    recent_activity = False
    for entry in reversed(gov.stage_history or []):
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if (datetime.now(timezone.utc) - ts).days <= 1:
                recent_activity = True
                break
        except Exception:
            pass
    if recent_activity:
        score += SCORE_WEIGHTS["not_stuck_in_stage"]

    return round(min(score, 1.0), 4)

def detect_near_duplicates(
    hypotheses: list[Hypothesis]
) -> dict[int, list[int]]:
    """
    For each hypothesis, find others with high signal component overlap.
    Returns {hypothesis_number: [similar_hypothesis_numbers]}
    """
    duplicates = {}
    for h in hypotheses:
        own_factors = set(c["factor"] for c in (h.signal_components or []))
        similar = []
        for other in hypotheses:
            if other.id == h.id:
                continue
            other_factors = set(c["factor"] for c in (other.signal_components or []))
            if not own_factors or not other_factors:
                continue
            overlap = len(own_factors & other_factors) / max(
                len(own_factors | other_factors), 1
            )
            if overlap >= SIMILARITY_THRESHOLD:
                similar.append(other.hypothesis_number)
        if similar:
            duplicates[h.hypothesis_number] = similar
    return duplicates

def run_scheduler(db: Session) -> dict:
    """
    Main scheduler entry point. Evaluates all pending experiments
    and assigns RUN/DELAY/CANCEL/MERGE decisions with priority scores.
    """
    # Get all pending jobs
    pending_jobs = db.query(ExperimentJob).filter(
        ExperimentJob.status.in_([JobStatus.PENDING, JobStatus.DELAYED])
    ).all()

    if not pending_jobs:
        return {"decisions": [], "summary": "No pending experiments in queue.",
                "total_pending": 0}

    # Load full context
    all_hypotheses = db.query(Hypothesis).all()
    memories = db.query(ResearchMemory).all()
    duplicates = detect_near_duplicates(all_hypotheses)

    # Build experiment summaries for the LLM
    experiment_summaries = []
    score_map = {}

    for job in pending_jobs:
        hyp = db.query(Hypothesis).filter_by(id=job.hypothesis_id).first()
        if not hyp:
            continue
        gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hyp.id).first()
        if not gov:
            continue

        priority_score = compute_priority_score(hyp, gov, all_hypotheses, memories)
        score_map[hyp.hypothesis_number] = priority_score

        similar = duplicates.get(hyp.hypothesis_number, [])
        alpha = db.query(AlphaSignal).filter_by(hypothesis_id=hyp.id).first()

        experiment_summaries.append({
            "hypothesis_number": hyp.hypothesis_number,
            "title": hyp.title,
            "stage": gov.current_stage.value,
            "signal_factors": [c["factor"] for c in (hyp.signal_components or [])],
            "holding_days": hyp.expected_holding_days,
            "sharpe": hyp.sharpe_ratio,
            "has_backtest": hyp.sharpe_ratio is not None,
            "has_stat_review": bool(gov.statistical_review),
            "has_debate": bool(gov.debate_record),
            "committee_decision": gov.committee_decision.get("decision")
                if gov.committee_decision else None,
            "compliance_warning": any(
                "COMPLIANCE_WARNING" in e.get("notes", "")
                for e in (gov.stage_history or [])
            ),
            "is_registered_alpha": alpha is not None,
            "similar_to": similar,
            "computed_priority_score": priority_score
        })

    # Memory context
    memory_context = [
        {"failure_mode": m.failure_mode, "lesson": m.lesson[:100]}
        for m in memories
    ]

    # Registered alphas context
    registered = db.query(AlphaSignal).filter_by(is_active=True).all()
    registered_factors = []
    for a in registered:
        reg_hyp = db.query(Hypothesis).filter_by(id=a.hypothesis_id).first()
        if reg_hyp:
            registered_factors.append({
                "hypothesis_number": reg_hyp.hypothesis_number,
                "factors": [c["factor"] for c in (reg_hyp.signal_components or [])],
                "sharpe": a.sharpe_ratio
            })

    prompt = f"""Pending experiments ({len(experiment_summaries)}):
{json.dumps(experiment_summaries, indent=2)}

Current research memories ({len(memory_context)}):
{json.dumps(memory_context, indent=2)}

Currently registered alphas (avoid duplicating these):
{json.dumps(registered_factors, indent=2)}

Make scheduling decisions for each pending experiment.
Prioritize: novel signals, those addressing known failure modes, those with
strong backtests. Merge or cancel near-duplicates of registered alphas.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SCHEDULER_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    decisions = json.loads(raw)

    # Apply decisions to DB
    applied = []
    for decision in decisions.get("decisions", []):
        hyp_num = decision["hypothesis_number"]
        action = decision["action"]
        new_score = decision.get("priority_score", 0.5)

        # Find the job for this hypothesis
        hyp = next(
            (h for h in all_hypotheses if h.hypothesis_number == hyp_num), None
        )
        if not hyp:
            continue

        job = db.query(ExperimentJob).filter_by(
            hypothesis_id=hyp.id
        ).first()
        if not job:
            continue

        # Apply action
        if action == "RUN":
            job.status = JobStatus.PENDING
            job.priority = JobPriority.HIGH
            job.priority_score = new_score
            job.scheduler_notes = {
                "action": "RUN",
                "reasoning": decision.get("reasoning"),
                "scheduled_at": datetime.now(timezone.utc).isoformat()
            }
            job.scheduled_at = datetime.now(timezone.utc)

        elif action == "DELAY":
            job.status = JobStatus.DELAYED
            job.priority = JobPriority.DEFERRED
            job.priority_score = new_score
            job.scheduler_notes = {
                "action": "DELAY",
                "reasoning": decision.get("reasoning")
            }

        elif action == "CANCEL":
            job.status = JobStatus.CANCELLED
            job.priority_score = 0.0
            job.scheduler_notes = {
                "action": "CANCEL",
                "reasoning": decision.get("reasoning")
            }

        elif action == "MERGE":
            merge_with = decision.get("merge_with")
            job.status = JobStatus.MERGED
            job.priority_score = new_score
            job.scheduler_notes = {
                "action": "MERGE",
                "merge_with_hypothesis": merge_with,
                "reasoning": decision.get("merge_rationale")
            }

        applied.append({
            "hypothesis_number": hyp_num,
            "action": action,
            "priority_score": new_score,
            "reasoning": decision.get("reasoning", "")
        })

    db.commit()

    return {
        "decisions": applied,
        "summary": decisions.get("summary", ""),
        "total_pending": len(pending_jobs),
        "total_decided": len(applied)
    }

def print_schedule(result: dict):
    print(f"\n{'='*60}")
    print(f"RESEARCH SCHEDULER — CYCLE RESULTS")
    print(f"{'='*60}")
    print(f"Pending: {result['total_pending']} | Decided: {result['total_decided']}")
    print(f"\nStrategy: {result['summary']}\n")

    for d in result["decisions"]:
        action_icon = {
            "RUN": "🟢", "DELAY": "🟡",
            "CANCEL": "🔴", "MERGE": "🔵"
        }.get(d["action"], "⚪")
        print(f"{action_icon} H#{d['hypothesis_number']} → {d['action']} "
              f"(score: {d['priority_score']:.2f})")
        print(f"   {d['reasoning'][:100]}")
    print(f"{'='*60}\n")