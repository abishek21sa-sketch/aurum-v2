import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.orm import Session
from src.models.hypothesis import Hypothesis
from src.models.governance import GovernanceRecord, GovernanceStage
from src.models.alpha_registry import AlphaSignal
from src.agents.backtester import (
    fetch_price_data, fetch_vix_data, build_composite_signal,
    run_backtest_with_circuit_breaker, SP500_UNIVERSE
)

def simulate_paper_trading(db: Session, hypothesis: Hypothesis,
                            simulation_days: int = 756) -> dict:
    gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hypothesis.id).first()
    if not gov or gov.current_stage != GovernanceStage.PAPER_TRADING:
        return {"error": f"hypothesis not in paper trading stage (current: {gov.current_stage.value if gov else 'no record'})"}

    holding_days = hypothesis.expected_holding_days or 21
    min_required_days = holding_days * 14
    effective_window = max(simulation_days, min_required_days)

    print(f"  Fetching data for paper trading simulation...")
    end = datetime.now().strftime("%Y-%m-%d")
    close, volume = fetch_price_data(SP500_UNIVERSE, "2018-01-01", end)
    vix = fetch_vix_data("2018-01-01", end)

    composite = build_composite_signal(close, volume, hypothesis.signal_components)
    if composite is None or len(composite) == 0:
        return {"error": "composite signal is empty"}

    effective_window = min(effective_window, len(close))
    recent_close = close.iloc[-effective_window:]
    recent_vix = vix.iloc[-effective_window:]

    print(f"  Running walk-forward simulation ({effective_window} days)...")
    results = run_backtest_with_circuit_breaker(
        composite, recent_close, recent_vix,
        holding_days=holding_days,
        vix_exit_threshold=18.0,
        backtest_start=recent_close.index[0].strftime("%Y-%m-%d")
    )

    if "error" in results:
        return results

    vix_max = float(recent_vix.max())
    vix_breach = vix_max >= 18.0
    decay_flag = (
        hypothesis.sharpe_ratio is not None and
        results["sharpe_ratio"] < hypothesis.sharpe_ratio * 0.5
    )
    retirement_recommended = (
        results["sharpe_ratio"] < 0 or
        results["max_drawdown"] < -0.40
    )

    # Determine recommended action
    if retirement_recommended:
        action = "RETIRE"
    elif decay_flag:
        action = "IMPROVE"
    else:
        action = "CONTINUE"

    report = {
        "simulation_window_days": effective_window,
        "simulation_start": recent_close.index[0].strftime("%Y-%m-%d"),
        "simulation_end": recent_close.index[-1].strftime("%Y-%m-%d"),
        "paper_sharpe": results["sharpe_ratio"],
        "paper_max_drawdown": results["max_drawdown"],
        "paper_win_rate": results["win_rate"],
        "original_backtest_sharpe": hypothesis.sharpe_ratio,
        "sharpe_degradation": float(round(
            (hypothesis.sharpe_ratio - results["sharpe_ratio"]) / hypothesis.sharpe_ratio, 4
        )) if hypothesis.sharpe_ratio else None,
        "vix_max_observed": round(vix_max, 2),
        "vix_breach_occurred": vix_breach,
        "circuit_breaker_triggers": results.get("circuit_breaker_triggers", 0),
        "decay_flag": decay_flag,
        "retirement_recommended": retirement_recommended,
        "recommended_action": action,
        "evaluated_at": datetime.now(timezone.utc).isoformat()
    }

    # Write to learning_reports table
    alpha = db.query(AlphaSignal).filter_by(hypothesis_id=hypothesis.id).first()
    if alpha:
        alpha.paper_trading_sharpe = results["sharpe_ratio"]
        alpha.decay_flag = decay_flag
        alpha.retirement_recommended = retirement_recommended
        alpha.last_evaluated_at = datetime.now(timezone.utc)

    db.execute(text("""
        INSERT INTO learning_reports (
            alpha_id, hypothesis_id, hypothesis_number,
            simulation_window_days, simulation_start, simulation_end,
            paper_sharpe, paper_max_drawdown, paper_win_rate, original_sharpe,
            sharpe_degradation, vix_max_observed, vix_breach_occurred,
            circuit_breaker_triggers, decay_flag, retirement_recommended, recommended_action
        ) VALUES (
            :alpha_id, :hypothesis_id, :hypothesis_number,
            :simulation_window_days, :simulation_start, :simulation_end,
            :paper_sharpe, :paper_max_drawdown, :paper_win_rate, :original_sharpe,
            :sharpe_degradation, :vix_max_observed, :vix_breach_occurred,
            :circuit_breaker_triggers, :decay_flag, :retirement_recommended, :recommended_action
        )
    """), {
        "alpha_id": str(alpha.id) if alpha else None,
        "hypothesis_id": str(hypothesis.id),
        "hypothesis_number": hypothesis.hypothesis_number,
        "simulation_window_days": effective_window,
        "simulation_start": report["simulation_start"],
        "simulation_end": report["simulation_end"],
        "paper_sharpe": report["paper_sharpe"],
        "paper_max_drawdown": report["paper_max_drawdown"],
        "paper_win_rate": report["paper_win_rate"],
        "original_sharpe": report["original_backtest_sharpe"],
        "sharpe_degradation": report["sharpe_degradation"],
        "vix_max_observed": report["vix_max_observed"],
        "vix_breach_occurred": report["vix_breach_occurred"],
        "circuit_breaker_triggers": report["circuit_breaker_triggers"],
        "decay_flag": report["decay_flag"],
        "retirement_recommended": report["retirement_recommended"],
        "recommended_action": report["recommended_action"]
    })

    # Update governance with monitoring note
    notes = (f"Paper trading eval ({effective_window}d): "
             f"Sharpe {results['sharpe_ratio']} vs backtest {hypothesis.sharpe_ratio}, "
             f"MaxDD {results['max_drawdown']}, VIX max {vix_max:.1f} "
             f"({'BREACHED' if vix_breach else 'no breach'}), "
             f"breaker fired {results.get('circuit_breaker_triggers', 0)}x. "
             f"Action: {action}")

    gov.stage_history = (gov.stage_history or []) + [{
        "from_stage": gov.current_stage,
        "to_stage": gov.current_stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": notes
    }]
    db.commit()

    return report

def run_full_registry_evaluation(db: Session) -> list[dict]:
    """
    V2.7 core: evaluates every active alpha in the registry.
    This is the function that gets called on a schedule.
    """
    alphas = db.query(AlphaSignal).filter_by(is_active=True).all()
    if not alphas:
        print("No active alphas in registry.")
        return []

    print(f"\nV2.7 Continuous Learning — evaluating {len(alphas)} active alpha(s)\n")
    reports = []

    for alpha in alphas:
        hyp = db.query(Hypothesis).filter_by(id=alpha.hypothesis_id).first()
        if not hyp:
            continue

        print(f"{'='*55}")
        print(f"Hypothesis #{hyp.hypothesis_number}: {hyp.title[:50]}")
        report = simulate_paper_trading(db, hyp)

        if "error" in report:
            print(f"  Skipped: {report['error']}")
            continue

        reports.append({"hypothesis_number": hyp.hypothesis_number, **report})
        print(f"  Sharpe: {report['paper_sharpe']} (backtest: {report['original_backtest_sharpe']})")
        print(f"  Max DD: {report['paper_max_drawdown']}")
        print(f"  Decay: {report['decay_flag']} | Action: {report['recommended_action']}")

    print(f"\n{'='*55}")
    print(f"REGISTRY SUMMARY")
    print(f"{'='*55}")
    for r in reports:
        flag = "⚠️ " if r["decay_flag"] else "✅ "
        print(f"  {flag}H#{r['hypothesis_number']}: {r['recommended_action']} "
              f"(Sharpe {r['paper_sharpe']} vs {r['original_backtest_sharpe']})")

    return reports

def print_learning_report(report: dict):
    print(f"\n{'='*65}")
    print(f"CONTINUOUS LEARNING REPORT")
    print(f"{'='*65}")
    print(f"Window: {report['simulation_start']} → {report['simulation_end']}")
    print(f"Paper Sharpe       : {report['paper_sharpe']}")
    print(f"Original backtest  : {report['original_backtest_sharpe']}")
    if report.get('sharpe_degradation') is not None:
        print(f"Sharpe degradation : {report['sharpe_degradation']*100:.1f}%")
    print(f"Paper max drawdown : {report['paper_max_drawdown']}")
    print(f"Paper win rate     : {report['paper_win_rate']}")
    print(f"VIX max observed   : {report['vix_max_observed']}")
    print(f"VIX breach (>=18)  : {report['vix_breach_occurred']}")
    print(f"Circuit breaker    : {report['circuit_breaker_triggers']}x")
    print(f"Decay flag         : {report['decay_flag']}")
    print(f">>> ACTION: {report['recommended_action']} <<<")
    print(f"{'='*65}\n")