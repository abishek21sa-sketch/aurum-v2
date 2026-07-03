import anthropic
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.core.config import ANTHROPIC_API_KEY
from src.models.hypothesis import Hypothesis
from src.models.governance import GovernanceRecord, GovernanceStage
from src.data.relational_knowledge_store import get_knowledge_store

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def _build_context(hypothesis: Hypothesis, gov: GovernanceRecord) -> str:
    ks = get_knowledge_store()

    # Build knowledge context for this hypothesis
    # Extract tickers from universe and get real relationship data
    universe_tickers = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
                        "JPM","BAC","GS","LLY","UNH","AMD","QCOM","INTC"]

    sector_concentration = ks.get_sector_concentration(universe_tickers)
    etf_crowding = ks.get_common_etf_exposure(universe_tickers[:7])

    # Get macro sensitivities for signal components
    signal_factors = [c["factor"] for c in (hypothesis.signal_components or [])]
    momentum_tickers = ["NVDA", "AMD", "META", "MSFT", "AAPL"]
    macro_context = {}
    for t in momentum_tickers:
        macro_context[t] = ks.get_macro_sensitivities(t)

    knowledge_context = f"""
Knowledge Graph Context:
- Universe sector concentration: {json.dumps({k: f"{v['pct']*100:.0f}%" for k, v in sector_concentration.items()})}
- Top ETF crowding (momentum names): {', '.join([f"{etf}:holds {info['overlap_count']}/7 names" for etf, info in list(etf_crowding.items())[:3]])}
- Macro sensitivities (key names): {json.dumps({t: macros[:3] for t, macros in macro_context.items()})}
"""

    return f"""Hypothesis #{hypothesis.hypothesis_number}: {hypothesis.title}

Thesis: {hypothesis.thesis}

Signal components: {json.dumps(hypothesis.signal_components, indent=2)}

Conditions: {json.dumps(hypothesis.conditions, indent=2)}

Backtest results:
- Sharpe ratio: {hypothesis.sharpe_ratio}
- Sortino ratio: {hypothesis.sortino_ratio}
- Calmar ratio: {hypothesis.calmar_ratio}
- Max drawdown: {hypothesis.max_drawdown}
- Annualized return: {hypothesis.annualized_return}
- Win rate: {hypothesis.win_rate}
- Universe: {hypothesis.universe}
- Holding period: {hypothesis.expected_holding_days} days

Statistical review: {json.dumps(gov.statistical_review, indent=2) if gov.statistical_review else "Not available"}
{knowledge_context}
"""

def _call_claude(system: str, user: str, max_tokens: int = 1000) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    if response.stop_reason == "max_tokens":
        print(f"  WARNING: response truncated at max_tokens={max_tokens}")
    return response.content[0].text.strip()

def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw

BULL_SYSTEM = """You are the Bull Agent on AURUM's investment committee. Your job is to make
the strongest possible case FOR deploying this hypothesis to paper trading.

You are not a cheerleader — you are a skilled advocate building a rigorous case. Use the
backtest and statistical data provided. Address why this signal should work going forward,
not just why it worked historically.

Return ONLY valid JSON in this exact structure, no markdown fences:
{
  "position": "deploy",
  "thesis_points": ["point 1", "point 2", "point 3"],
  "strongest_argument": "the single most compelling reason to deploy this",
  "confidence": <float 0-1>
}"""

BEAR_SYSTEM = """You are the Bear Agent on AURUM's investment committee. Your job is to find
the strongest possible case AGAINST deploying this hypothesis.

You will be given the hypothesis, its backtest data, and the Bull Agent's thesis. Your
rebuttal must specifically target the Bull's weakest points — do not just list generic risks.
Look for: overfitting signs, regime dependency, statistical fragility, crowding risk, and
whether the historical edge is likely to persist.

Return ONLY valid JSON in this exact structure, no markdown fences:
{
  "position": "reject",
  "rebuttal_points": ["point 1", "point 2", "point 3"],
  "strongest_objection": "the single most damaging objection, specifically targeting the bull's weakest claim",
  "confidence": <float 0-1>
}"""

RISK_SYSTEM = """You are the Risk Agent on AURUM's investment committee. You do not argue
bull or bear — you provide an independent risk assessment.

Evaluate: portfolio concentration risk, tail risk (using max drawdown and the signal's
implied beta to known crash scenarios), liquidity risk in the named universe, and whether
the position sizing implied by this strategy (long top quintile) creates correlated exposure.

Return ONLY valid JSON in this exact structure, no markdown fences:
{
  "risk_flags": ["flag 1", "flag 2"],
  "concentration_risk": "low | medium | high",
  "tail_risk_assessment": "1-2 sentence assessment",
  "recommended_position_limit": "e.g. 'max 8% of portfolio NAV'",
  "overall_risk_rating": "low | medium | high"
}"""

JUDGE_SYSTEM = """You are the Judge on AURUM's investment committee. You have read the Bull
thesis, Bear rebuttal, and Risk assessment for this hypothesis. Your job is to produce a final
committee decision.

CRITICAL RULE: Your decision is only valid if it explicitly addresses the Bear's strongest
objection. You cannot simply side with the Bull because the numbers look good — you must
explain why the strongest bear objection does or does not invalidate the thesis.

Return ONLY valid JSON in this exact structure, no markdown fences:
{
  "decision": "approve_paper_trading | reject | request_revision",
  "justification": "2-4 sentences that explicitly engage with the bear's strongest objection",
  "addressed_bear_objection": "restate the specific bear objection you are responding to",
  "conditions_for_approval": ["any conditions, e.g. position limits from risk agent"],
  "confidence": <float 0-1>
}"""

def run_debate(db: Session, hypothesis: Hypothesis) -> dict:
    gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hypothesis.id).first()
    if not gov:
        raise ValueError(f"No governance record found for hypothesis {hypothesis.id}")

    context = _build_context(hypothesis, gov)

    def _safe_parse(raw: str, agent_name: str) -> dict:
        cleaned = _strip_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{agent_name} returned invalid/truncated JSON: {e}\n"
                f"--- RAW RESPONSE START ---\n{raw}\n--- RAW RESPONSE END ---"
            )

    print("  Bull Agent building thesis...")
    bull_raw = _call_claude(BULL_SYSTEM, context, max_tokens=2000)
    bull = _safe_parse(bull_raw, "Bull Agent")

    print("  Bear Agent building rebuttal...")
    bear_context = context + f"\n\nBull Agent's thesis:\n{json.dumps(bull, indent=2)}"
    bear_raw = _call_claude(BEAR_SYSTEM, bear_context, max_tokens=2000)
    bear = _safe_parse(bear_raw, "Bear Agent")

    print("  Risk Agent assessing portfolio risk...")
    risk_raw = _call_claude(RISK_SYSTEM, context, max_tokens=2000)
    risk = _safe_parse(risk_raw, "Risk Agent")

    print("  Judge synthesizing committee decision...")
    judge_context = (
        context
        + f"\n\nBull Agent's thesis:\n{json.dumps(bull, indent=2)}"
        + f"\n\nBear Agent's rebuttal:\n{json.dumps(bear, indent=2)}"
        + f"\n\nRisk Agent's assessment:\n{json.dumps(risk, indent=2)}"
    )
    judge_raw = _call_claude(JUDGE_SYSTEM, judge_context, max_tokens=2000)
    judge = _safe_parse(judge_raw, "Judge Agent")

    debate_record = {
        "bull": bull,
        "bear": bear,
        "risk": risk,
        "judge": judge,
        "debated_at": datetime.now(timezone.utc).isoformat()
    }

    # Write to governance
    gov.debate_record = debate_record
    gov.committee_decision = {
        "decision": judge["decision"],
        "justification": judge["justification"],
        "date": datetime.now(timezone.utc).isoformat(),
        "members": ["bull_agent", "bear_agent", "risk_agent", "judge_agent"]
    }

    decision = judge["decision"]
    if decision == "approve_paper_trading":
        gov.advance_stage(
            GovernanceStage.PAPER_TRADING,
            notes=f"Committee approved. Judge confidence: {judge['confidence']}. "
                  f"Addressed bear objection: {judge['addressed_bear_objection'][:100]}"
        )
        gov.paper_trading_start = datetime.now(timezone.utc)
    elif decision == "reject":
        gov.advance_stage(
            GovernanceStage.RETIRED,
            notes=f"Committee rejected. Reason: {judge['justification'][:200]}"
        )
        gov.retirement_date = datetime.now(timezone.utc)
        gov.failure_reason = judge['justification']
    else:  # request_revision
        # current_stage stays at its existing value intentionally — a revision
        # request does not advance governance — but we record this explicitly
        # as a NEEDS_REVISION flag so eligibility checks elsewhere (e.g. alpha
        # registry) can distinguish "stuck because never debated" from
        # "actively rejected pending revision."
        gov.stage_history = (gov.stage_history or []) + [{
            "from_stage": gov.current_stage,
            "to_stage": gov.current_stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": f"Committee requested revision: {judge['justification'][:200]}"
        }]
        gov.failure_reason = f"REVISION REQUESTED: {judge['justification']}"

    db.commit()
    db.refresh(gov)

    return debate_record

def print_debate(debate_record: dict):
    print(f"\n{'='*70}")
    print(f"INVESTMENT COMMITTEE DEBATE")
    print(f"{'='*70}")

    print(f"\n── BULL CASE ──────────────────────────────")
    for p in debate_record["bull"]["thesis_points"]:
        print(f"  • {p}")
    print(f"\n  Strongest argument: {debate_record['bull']['strongest_argument']}")
    print(f"  Confidence: {debate_record['bull']['confidence']}")

    print(f"\n── BEAR CASE ───────────────────────────────")
    for p in debate_record["bear"]["rebuttal_points"]:
        print(f"  • {p}")
    print(f"\n  Strongest objection: {debate_record['bear']['strongest_objection']}")
    print(f"  Confidence: {debate_record['bear']['confidence']}")

    print(f"\n── RISK ASSESSMENT ─────────────────────────")
    for f in debate_record["risk"]["risk_flags"]:
        print(f"  • {f}")
    print(f"\n  Concentration risk: {debate_record['risk']['concentration_risk']}")
    print(f"  Tail risk: {debate_record['risk']['tail_risk_assessment']}")
    print(f"  Position limit: {debate_record['risk']['recommended_position_limit']}")
    print(f"  Overall rating: {debate_record['risk']['overall_risk_rating']}")

    print(f"\n── COMMITTEE DECISION ──────────────────────")
    j = debate_record["judge"]
    print(f"  Decision: {j['decision'].upper()}")
    print(f"  Justification: {j['justification']}")
    print(f"  Addressed bear objection: {j['addressed_bear_objection']}")
    if j.get("conditions_for_approval"):
        print(f"  Conditions: {', '.join(j['conditions_for_approval'])}")
    print(f"  Confidence: {j['confidence']}")
    print(f"{'='*70}\n")