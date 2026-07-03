import anthropic
import json
from sqlalchemy.orm import Session
from src.core.config import ANTHROPIC_API_KEY
from src.models.hypothesis import Hypothesis
from src.models.governance import GovernanceRecord
from src.models.research_memory import ResearchMemory
from src.models.alpha_registry import AlphaSignal
from src.data.relational_knowledge_store import get_knowledge_store
from sqlalchemy import text

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def build_research_corpus(db: Session) -> str:
    """
    Serializes the entire research corpus into a structured text block
    that Claude can reason over. Keeps it dense but readable.
    """
    sections = []

    # Hypotheses
    hypotheses = db.query(Hypothesis).order_by(Hypothesis.hypothesis_number).all()
    sections.append("=== HYPOTHESES ===")
    for h in hypotheses:
        gov = db.query(GovernanceRecord).filter_by(hypothesis_id=h.id).first()
        stage = gov.current_stage.value if gov else "unknown"
        decision = gov.committee_decision.get("decision") if gov and gov.committee_decision else None

        entry = [
            f"H#{h.hypothesis_number}: {h.title}",
            f"  Status: {stage}" + (f" | Committee: {decision}" if decision else ""),
            f"  Thesis: {h.thesis}",
            f"  Signals: {', '.join(c['factor'] for c in (h.signal_components or []))}",
            f"  Conditions: {json.dumps(h.conditions)}",
            f"  Holding: {h.expected_holding_days}d | Universe: {h.universe}",
        ]
        if h.sharpe_ratio is not None:
            entry.append(f"  Backtest: Sharpe={h.sharpe_ratio}, MaxDD={h.max_drawdown}, "
                        f"WinRate={h.win_rate}, AnnReturn={h.annualized_return}")

        if gov and gov.debate_record:
            judge = gov.debate_record.get("judge", {})
            bear = gov.debate_record.get("bear", {})
            entry.append(f"  Bear objection: {bear.get('strongest_objection', '')[:200]}")
            entry.append(f"  Judge decision: {judge.get('decision')} — {judge.get('justification', '')[:200]}")

        sections.append("\n".join(entry))

    # Research memories
    memories = db.query(ResearchMemory).order_by(ResearchMemory.created_at).all()
    sections.append("\n=== RESEARCH MEMORIES ===")
    for m in memories:
        sections.append(
            f"Memory from H#{m.source_hypothesis_number} [{m.failure_mode}]:\n"
            f"  Lesson: {m.lesson}\n"
            f"  Affects: {', '.join(m.affected_signal_types)}\n"
            f"  Influenced: {m.influenced_hypothesis_numbers}"
        )

    # Alpha registry
    alphas = db.query(AlphaSignal).all()
    sections.append("\n=== ALPHA REGISTRY ===")
    for a in alphas:
        hyp = db.query(Hypothesis).filter_by(id=a.hypothesis_id).first()
        paper_note = f" | Paper Sharpe: {a.paper_trading_sharpe}" if a.paper_trading_sharpe else ""
        decay_note = " | ⚠️ DECAY FLAGGED" if a.decay_flag else ""
        sections.append(
            f"Alpha from H#{hyp.hypothesis_number if hyp else '?'}: {a.signal_name}\n"
            f"  Sharpe={a.sharpe_ratio}, MaxDD={a.max_drawdown}, WinRate={a.win_rate}"
            f"{paper_note}{decay_note}"
        )

    # Learning reports summary
    reports = db.execute(text(
        "SELECT hypothesis_number, paper_sharpe, paper_max_drawdown, "
        "recommended_action, evaluated_at "
        "FROM learning_reports ORDER BY evaluated_at DESC LIMIT 10"
    )).fetchall()
    if reports:
        sections.append("\n=== RECENT LEARNING REPORTS ===")
        for r in reports:
            sections.append(
                f"H#{r.hypothesis_number}: Paper Sharpe={r.paper_sharpe}, "
                f"MaxDD={r.paper_max_drawdown}, Action={r.recommended_action}, "
                f"Evaluated={str(r.evaluated_at)[:10]}"
            )

    # Knowledge graph context
    ks = get_knowledge_store()
    sections.append("\n=== KNOWLEDGE GRAPH ===")

    # Sector map
    sectors = {}
    for ticker in ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","JPM","BAC","GS",
                   "LLY","UNH","XOM","AMD","QCOM","INTC"]:
        e = ks.get_entity(ticker)
        if e:
            sectors.setdefault(e.sector, []).append(ticker)

    for sector, tickers in sectors.items():
        sections.append(f"Sector {sector}: {', '.join(tickers)}")

    # ETF crowding for top momentum names
    top_names = ["NVDA", "AMD", "META", "MSFT", "AAPL"]
    crowding = ks.get_common_etf_exposure(top_names)
    sections.append(f"ETF overlap (top momentum names {top_names}):")
    for etf, info in list(crowding.items())[:5]:
        sections.append(f"  {etf}: holds {info['overlap_count']}/{len(top_names)} names "
                       f"({info['concentration_risk']} concentration risk)")

    return "\n\n".join(sections)

COPILOT_SYSTEM = """You are the AURUM Research Copilot — an AI assistant with full knowledge
of AURUM V2's hypothesis corpus, research memories, alpha registry, and learning reports.

You answer questions about the research system's findings, hypotheses, signals, and lessons.
You cite specific hypothesis numbers (H#N), memory failure modes, and alpha names when relevant.
You are precise, direct, and honest about what the data shows versus what is uncertain.

You can answer questions like:
- "Which hypotheses failed because of regime conditioning?"
- "What did the debate engine find wrong with H#7?"
- "Which alphas are degrading?"
- "What lessons have we learned about circuit-breakers?"
- "Find strategies that survived the VIX spike"
- "What should we test next?"

Return your answer as plain text with specific citations. Keep responses under 300 words.
If the corpus doesn't contain enough information to answer confidently, say so directly.
"""

def query_copilot(db: Session, question: str) -> str:
    corpus = build_research_corpus(db)

    user_prompt = f"""Research corpus:

{corpus}

Question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=COPILOT_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text.strip()