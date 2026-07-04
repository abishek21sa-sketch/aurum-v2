"""
AURUM V2 — Autonomous Research Cycle
=====================================
One command that runs the complete research lifecycle:

  observe → generate → backtest → validate → debate → register (or retire + memory)

Usage:
  python -m src.pipeline.research_cycle
  python -m src.pipeline.research_cycle --observations custom_observations.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from src.core.database import SessionLocal
from src.models.hypothesis import Hypothesis, HypothesisStatus
from src.models.governance import GovernanceRecord, GovernanceStage
from src.models.alpha_registry import AlphaSignal

from src.agents.research_scientist import generate_hypothesis
from src.agents.backtester import run_hypothesis_backtest
from src.agents.statistical_validator import run_statistical_validation
from src.agents.debate_engine import run_debate
from src.agents.memory_extractor import extract_memory_from_debate
from src.agents.alpha_registrar import register_alpha

from src.core.config import ANTHROPIC_API_KEY

# ── Default market observations ───────────────────────────────────
DEFAULT_OBSERVATIONS = {
    "momentum_signal": "strong positive price momentum across large cap equities, 12-1 month",
    "volatility_regime": "low volatility, VIX around 14, compressing further",
    "earnings_trend": "positive earnings revisions in technology and industrials",
    "macro_regime": "expansion, GDP growth above trend",
    "rate_environment": "stable, Fed on hold",
    "institutional_flow": "consistent buying in growth factors over past 4 weeks"
}

# ── Pipeline state tracker ────────────────────────────────────────
class PipelineResult:
    def __init__(self):
        self.hypothesis_number = None
        self.stages_completed = []
        self.stage_results = {}
        self.final_outcome = None
        self.started_at = datetime.now(timezone.utc)
        self.errors = []

    def record(self, stage: str, result: dict):
        self.stages_completed.append(stage)
        self.stage_results[stage] = result

    def fail(self, stage: str, reason: str):
        self.errors.append({"stage": stage, "reason": reason})
        self.final_outcome = f"FAILED at {stage}: {reason}"

    def elapsed(self) -> str:
        delta = datetime.now(timezone.utc) - self.started_at
        return f"{delta.seconds}s"

def _header(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def _step(n: int, total: int, label: str):
    print(f"\n[{n}/{total}] {label}")
    print(f"{'─'*50}")

def _ok(msg: str):
    print(f"  ✅ {msg}")

def _warn(msg: str):
    print(f"  ⚠️  {msg}")

def _fail(msg: str):
    print(f"  ❌ {msg}")

def print_provenance(result: PipelineResult, db: Session):
    """Print the full DB state after the pipeline completes."""
    _header("PIPELINE PROVENANCE SUMMARY")

    if not result.hypothesis_number:
        print("  No hypothesis generated.")
        return

    hyp = db.query(Hypothesis).filter_by(
        hypothesis_number=result.hypothesis_number
    ).first()
    gov = db.query(GovernanceRecord).filter_by(
        hypothesis_id=hyp.id
    ).first() if hyp else None

    print(f"\n  Hypothesis #{result.hypothesis_number}: {hyp.title if hyp else '?'}")
    print(f"  Status    : {hyp.status.value if hyp else '?'}")
    print(f"  Stage     : {gov.current_stage.value if gov else '?'}")
    print(f"  Elapsed   : {result.elapsed()}")

    print(f"\n  Stages completed:")
    for stage in result.stages_completed:
        r = result.stage_results.get(stage, {})
        if stage == "generate":
            print(f"    ✅ generate     → H#{result.hypothesis_number} "
                  f"({len(hyp.signal_components)} signals, "
                  f"{hyp.expected_holding_days}d holding)")
        elif stage == "backtest":
            print(f"    ✅ backtest     → Sharpe {r.get('sharpe_ratio')}, "
                  f"MaxDD {r.get('max_drawdown')}, "
                  f"OOS Sharpe {r.get('oos_sharpe')}")
        elif stage == "validate":
            sr = r.get("statistical_review", {})
            passed = sr.get("checks_passed", "?")
            total = sr.get("checks_total", "?")
            print(f"    ✅ validate     → {passed}/{total} checks passed, "
                  f"t-stat={sr.get('t_test', {}).get('t_stat', '?')}, "
                  f"p={sr.get('t_test', {}).get('p_value', '?')}")
        elif stage == "debate":
            decision = r.get("decision", "?")
            confidence = r.get("confidence", "?")
            print(f"    ✅ debate       → {decision.upper()} "
                  f"(confidence {confidence})")
        elif stage == "register":
            print(f"    ✅ register     → Alpha ID {r.get('alpha_id', '?')}")
        elif stage == "memory":
            print(f"    ✅ memory       → failure_mode={r.get('failure_mode', '?')}, "
                  f"affects={r.get('affected_signal_types', [])}")
        elif stage == "retire":
            print(f"    ✅ retire       → {r.get('reason', '?')[:80]}")

    if result.errors:
        print(f"\n  Errors:")
        for e in result.errors:
            print(f"    ❌ {e['stage']}: {e['reason']}")

    print(f"\n  Governance timeline:")
    if gov and gov.stage_history:
        for entry in gov.stage_history:
            ts = entry.get("timestamp", "")[:19]
            stage = entry.get("to_stage", "?")
            notes = entry.get("notes", "")[:80]
            print(f"    {ts} → {stage}: {notes}")

    print(f"\n  Final outcome: {result.final_outcome or 'UNKNOWN'}")

def run_pipeline(
    observations: dict,
    db: Session,
    use_real_earnings: bool = True,
    skip_debate: bool = False
) -> PipelineResult:

    result = PipelineResult()
    total_steps = 6

    # Ensure clean session state before starting
    try:
        db.rollback()
    except Exception:
        pass

    _header("AURUM V2 — AUTONOMOUS RESEARCH CYCLE")
    print(f"  Started: {result.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Observations: {list(observations.keys())}")

    # ── Step 1: Generate hypothesis ───────────────────────────────
    _step(1, total_steps, "Research Scientist — generating hypothesis")
    try:
        hyp = generate_hypothesis(db, observations)
        result.hypothesis_number = hyp.hypothesis_number
        result.record("generate", {
            "hypothesis_number": hyp.hypothesis_number,
            "title": hyp.title
        })
        _ok(f"H#{hyp.hypothesis_number}: {hyp.title}")
        _ok(f"{len(hyp.signal_components)} signal components, "
            f"{hyp.expected_holding_days}d holding, {hyp.universe}")

        # Verify memory compliance immediately after generation
        # Verify memory compliance and gate on minimum threshold
        applicable = []
        compliance = {"verified": True, "overall_compliance": 1.0, "checks": []}
        try:
            from src.agents.research_memory_service import ResearchMemoryService
            service = ResearchMemoryService(db)
            obs_text = " ".join(str(v) for v in observations.values()).lower()
            inferred_signals = []
            if "momentum" in obs_text: inferred_signals.append("price_momentum")
            if "earnings" in obs_text: inferred_signals.append("earnings_revision")
            if "volatil" in obs_text: inferred_signals.append("volatility")
            if "institutional" in obs_text or "flow" in obs_text:
                inferred_signals.append("institutional_flow_proxy")
            inferred_conditions = {}
            if "expansion" in obs_text: inferred_conditions["macro_regime"] = "expansion"
            if "vix" in obs_text or "volatil" in obs_text:
                inferred_conditions["vix_range"] = "<15"
            if "stable" in obs_text: inferred_conditions["rate_environment"] = "stable"

            applicable = service.get_applicable_constraints(
                inferred_signals, inferred_conditions, hyp.expected_holding_days or 21
            )
            if applicable:
                compliance = service.verify_hypothesis_compliance(hyp, applicable)
                result.record("memory_compliance", compliance)
                score = compliance["overall_compliance"]
                n = compliance["n_constraints_checked"]

                if compliance["verified"]:
                    _ok(f"Memory compliance: {score:.0%} across {n} constraint(s)")
                else:
                    _warn(f"Memory compliance: {score:.0%} ({n} constraints) — attempting regeneration")

                    # ── Compliance gate: regenerate once with violation context ──
                    if score < 0.80:
                        violations = [
                            f"[{c['failure_mode']}]: {c.get('violation', '')}"
                            for c in compliance["checks"] if not c["compliant"]
                        ]
                        violation_prompt = (
                            "COMPLIANCE VIOLATIONS from previous generation attempt:\n" +
                            "\n".join(f"  - {v}" for v in violations) +
                            "\n\nThese violations MUST be structurally resolved in this attempt. "
                            "Do not just mention the issue — add concrete architectural elements "
                            "(e.g. a documented filing lag if required, an explicit regime-transition "
                            "test section, a specific threshold with backtested trigger frequency)."
                        )

                        _warn("Regenerating with violation context...")
                        # Delete non-compliant hypothesis cleanly
                        # Order matters: delete children before parent
                        from src.models.experiment_queue import ExperimentJob
                        job_old = db.query(ExperimentJob).filter_by(
                            hypothesis_id=hyp.id
                        ).first()
                        if job_old:
                            db.delete(job_old)
                        gov_old = db.query(GovernanceRecord).filter_by(
                            hypothesis_id=hyp.id
                        ).first()
                        if gov_old:
                            db.delete(gov_old)
                        db.delete(hyp)
                        try:
                            db.commit()
                        except Exception as del_err:
                            db.rollback()
                            _warn(f"Could not delete original hypothesis: {del_err}")
                            # Continue without deletion — regeneration will
                            # still create a new hypothesis with a new number

                        # Regenerate with enhanced prompt
                        from src.agents.research_scientist import (
                            get_next_hypothesis_number, retrieve_relevant_memories,
                            build_observation_prompt, SYSTEM_PROMPT
                        )
                        import anthropic as _anthropic
                        import uuid as _uuid
                        from src.models.experiment_queue import ExperimentJob, JobStatus, JobPriority

                        _client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                        hyp_number = get_next_hypothesis_number(db)
                        memories_passive = retrieve_relevant_memories(
                            db, list(observations.keys()), observations
                        )
                        base_prompt = build_observation_prompt(
                            observations, memories_passive,
                            service.format_constraints_for_scientist(applicable)
                        )
                        enhanced_prompt = base_prompt + "\n\n" + violation_prompt

                        response = _client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=4000,
                            system=SYSTEM_PROMPT,
                            messages=[{"role": "user", "content": enhanced_prompt}]
                        )
                        raw = response.content[0].text.strip()
                        if raw.startswith("```"):
                            raw = raw.split("```")[1]
                            if raw.startswith("json"):
                                raw = raw[4:]
                            raw = raw.strip()

                        import json as _json
                        data = _json.loads(raw)

                        from src.models.hypothesis import HypothesisStatus
                        from src.models.governance import GovernanceStage
                        import datetime as _dt

                        hyp = Hypothesis(
                            id=_uuid.uuid4(),
                            hypothesis_number=hyp_number,
                            title=data["title"],
                            thesis=data["thesis"],
                            signal_components=data["signal_components"],
                            conditions=data.get("conditions", {}),
                            expected_holding_days=data.get("expected_holding_days"),
                            universe=data.get("universe", "SP500"),
                            status=HypothesisStatus.DRAFT,
                            generated_by="research_scientist_regen"
                        )
                        db.add(hyp)
                        db.flush()

                        gov_new = GovernanceRecord(
                            id=_uuid.uuid4(),
                            hypothesis_id=hyp.id,
                            current_stage=GovernanceStage.IDEA,
                            stage_history=[{
                                "from_stage": None,
                                "to_stage": GovernanceStage.IDEA,
                                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                                "notes": f"Hypothesis #{hyp_number} regenerated after compliance failure. Violations: {violations}"
                            }]
                        )
                        db.add(gov_new)

                        job_new = ExperimentJob(
                            id=_uuid.uuid4(),
                            hypothesis_id=hyp.id,
                            status=JobStatus.PENDING,
                            priority=JobPriority.NORMAL,
                            priority_score=0.5
                        )
                        db.add(job_new)
                        db.commit()
                        db.refresh(hyp)

                        result.hypothesis_number = hyp.hypothesis_number
                        _ok(f"Regenerated: H#{hyp.hypothesis_number}: {hyp.title}")

                        # Re-verify compliance on regenerated hypothesis
                        compliance_2 = service.verify_hypothesis_compliance(hyp, applicable)
                        score_2 = compliance_2["overall_compliance"]
                        result.record("memory_compliance", compliance_2)

                        if compliance_2["verified"]:
                            _ok(f"Post-regen compliance: {score_2:.0%} ✅")
                        else:
                            _warn(f"Post-regen compliance: {score_2:.0%} — flagging in governance")
                            # Flag in governance but don't stop — committee will see it
                            gov_new = db.query(GovernanceRecord).filter_by(
                                hypothesis_id=hyp.id
                            ).first()
                            if gov_new:
                                gov_new.stage_history = (gov_new.stage_history or []) + [{
                                    "from_stage": GovernanceStage.IDEA,
                                    "to_stage": GovernanceStage.IDEA,
                                    "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                                    "notes": f"COMPLIANCE_WARNING: {score_2:.0%} after regeneration. "
                                             f"Violations: {[c['failure_mode'] for c in compliance_2['checks'] if not c['compliant']]}"
                                }]
                                db.commit()

        except Exception as e:
            db.rollback()  # Reset session to clean state before continuing
            _warn(f"Compliance check error (session rolled back): {str(e)[:120]}")
            # Continue pipeline with original hypothesis — compliance gate
            # failure is non-fatal; committee will see the governance flag

    except Exception as e:
        result.fail("generate", str(e))
        _fail(str(e))
        return result

    # ── Step 2: Backtest ──────────────────────────────────────────
    _step(2, total_steps, "Backtester — running signal on historical data")
    print(f"  Mode: {'real EDGAR earnings data' if use_real_earnings else 'price-volume proxy'}")
    try:
        # Patch use_real_earnings into the backtester call
        from src.agents import backtester as bt_module
        original_build = bt_module.build_composite_signal

        def patched_build(close, volume, signal_components, use_real_earnings_data=True):
            return original_build(close, volume, signal_components,
                                  use_real_earnings_data=use_real_earnings)

        bt_module.build_composite_signal = patched_build
        backtest_results = run_hypothesis_backtest(db, hyp)
        bt_module.build_composite_signal = original_build  # restore

        if "error" in backtest_results:
            result.fail("backtest", backtest_results["error"])
            _fail(backtest_results["error"])
            return result

        result.record("backtest", backtest_results)
        db.refresh(hyp)
        _ok(f"Sharpe {backtest_results['sharpe_ratio']}, "
            f"Sortino {backtest_results['sortino_ratio']}, "
            f"MaxDD {backtest_results['max_drawdown']}, "
            f"WinRate {backtest_results['win_rate']}")
        _ok(f"OOS Sharpe {backtest_results['oos_sharpe']}")
    except Exception as e:
        result.fail("backtest", str(e))
        _fail(str(e))
        return result

    # ── Step 3: Statistical validation ───────────────────────────
    _step(3, total_steps, "Statistical Validator — 4-gate significance test")
    try:
        review = run_statistical_validation(db, hyp)
        db.refresh(hyp)

        passed = review.get("checks_passed", 0)
        total_checks = review.get("checks_total", 4)
        overall = review.get("overall_passed", False)

        result.record("validate", {"statistical_review": review})

        if overall:
            _ok(f"{passed}/{total_checks} checks passed — advancing to risk review")
        else:
            _warn(f"{passed}/{total_checks} checks passed — hypothesis held at statistical review")
            _warn("Writing research memory and stopping pipeline")

            # Write memory for statistical failure
            gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hyp.id).first()
            if gov:
                memory = extract_memory_from_debate(db, hyp, gov)
                if memory:
                    result.record("memory", {
                        "failure_mode": memory.failure_mode,
                        "affected_signal_types": memory.affected_signal_types
                    })
                    _ok(f"Research memory written: {memory.failure_mode}")

            result.final_outcome = f"STOPPED — statistical validation failed ({passed}/{total_checks} checks)"
            return result

    except Exception as e:
        result.fail("validate", str(e))
        _fail(str(e))
        return result

    # ── Step 4: Debate ────────────────────────────────────────────
    _step(4, total_steps, "Debate Engine — Bull / Bear / Risk / Judge")
    if skip_debate:
        _warn("Debate skipped (--skip-debate flag)")
    else:
        try:
            gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hyp.id).first()
            debate_record = run_debate(db, hyp)
            db.refresh(hyp)

            judge = debate_record.get("judge", {})
            decision = judge.get("decision", "unknown")
            confidence = judge.get("confidence", 0)

            result.record("debate", {
                "decision": decision,
                "confidence": confidence,
                "justification": judge.get("justification", "")[:150]
            })

            _ok(f"Bull confidence: {debate_record['bull'].get('confidence')}")
            _ok(f"Bear confidence: {debate_record['bear'].get('confidence')}")
            _ok(f"Committee: {decision.upper()} (judge confidence {confidence})")

            # Handle non-approval outcomes
            if decision == "reject":
                _warn("Committee REJECTED — retiring hypothesis and writing memory")
                gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hyp.id).first()
                memory = extract_memory_from_debate(db, hyp, gov)
                if memory:
                    result.record("memory", {
                        "failure_mode": memory.failure_mode,
                        "affected_signal_types": memory.affected_signal_types
                    })
                    _ok(f"Research memory written: {memory.failure_mode}")
                result.record("retire", {"reason": judge.get("justification", "")[:150]})
                result.final_outcome = "RETIRED — committee rejected"
                return result

            elif decision == "request_revision":
                _warn("Committee requested REVISION — writing memory and flagging")
                gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hyp.id).first()
                memory = extract_memory_from_debate(db, hyp, gov)
                if memory:
                    result.record("memory", {
                        "failure_mode": memory.failure_mode,
                        "affected_signal_types": memory.affected_signal_types
                    })
                    _ok(f"Research memory written: {memory.failure_mode}")
                result.final_outcome = "REVISION REQUESTED — see governance record"
                return result

        except Exception as e:
            result.fail("debate", str(e))
            _fail(str(e))
            return result

    # ── Step 5: Alpha registry ────────────────────────────────────
    _step(5, total_steps, "Alpha Registrar — registering validated signal")
    try:
        alpha = register_alpha(db, hyp, require_paper_trading=True)
        if alpha:
            result.record("register", {"alpha_id": str(alpha.id)})
            _ok(f"Registered: {alpha.signal_name}")
            _ok(f"Sharpe {alpha.sharpe_ratio} | MaxDD {alpha.max_drawdown}")
        else:
            _warn("Not yet eligible for registry (requires paper trading approval)")
            _warn("Hypothesis approved by committee — awaiting paper trading deployment")
    except Exception as e:
        result.fail("register", str(e))
        _fail(str(e))

    # ── Step 6: Memory extraction (from approved debate) ─────────
    _step(6, total_steps, "Memory Extractor — capturing lessons from debate")
    try:
        gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hyp.id).first()
        if gov and gov.debate_record:
            memory = extract_memory_from_debate(db, hyp, gov)
            if memory:
                result.record("memory", {
                    "failure_mode": memory.failure_mode,
                    "affected_signal_types": memory.affected_signal_types
                })
                _ok(f"Memory written: {memory.failure_mode}")
                _ok(f"Affects: {memory.affected_signal_types}")
            else:
                _warn("No distinct lesson extracted from this debate")
        else:
            _warn("No debate record available for memory extraction")
    except Exception as e:
        _warn(f"Memory extraction failed (non-critical): {e}")

    result.final_outcome = "COMPLETE — hypothesis approved, awaiting paper trading"

    # Run scheduler at end of each cycle to re-prioritize the queue
    try:
        from src.agents.research_scheduler import run_scheduler
        print(f"\n  Running scheduler to re-prioritize experiment queue...")
        schedule_result = run_scheduler(db)
        print(f"  Scheduler: {schedule_result['summary'][:100]}")
        result.record("scheduler", {
            "decisions": len(schedule_result["decisions"]),
            "summary": schedule_result["summary"]
        })
    except Exception as e:
        _warn(f"Scheduler skipped: {e}")

    return result

def main():
    parser = argparse.ArgumentParser(description="AURUM V2 Autonomous Research Cycle")
    parser.add_argument("--observations", type=str, default=None,
                        help="Path to JSON file with market observations")
    parser.add_argument("--skip-debate", action="store_true",
                        help="Skip debate engine (faster, for testing)")
    args = parser.parse_args()

    # Load observations
    if args.observations:
        with open(args.observations) as f:
            observations = json.load(f)
        print(f"Loaded observations from {args.observations}")
    else:
        observations = DEFAULT_OBSERVATIONS

    db = SessionLocal()
    try:
        result = run_pipeline(
            observations=observations,
            db=db,
            use_real_earnings=True,
            skip_debate=args.skip_debate
        )
        print_provenance(result, db)

        # Exit code: 0 for success, 1 for failure
        if result.errors:
            sys.exit(1)

    finally:
        db.close()

if __name__ == "__main__":
    main()