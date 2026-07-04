import anthropic
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.core.config import ANTHROPIC_API_KEY
from src.models.hypothesis import Hypothesis, HypothesisStatus
from src.models.governance import GovernanceRecord, GovernanceStage
from src.models.experiment_queue import ExperimentJob, JobStatus, JobPriority
from src.models.research_memory import ResearchMemory
from src.data.relational_knowledge_store import get_knowledge_store

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are the Autonomous Research Scientist for AURUM V2, an AI-native quantitative research firm.

Your job is to generate rigorous, testable investment hypotheses based on market observations.

Each hypothesis must follow this exact JSON structure:
{
  "title": "short descriptive title (max 10 words)",
  "thesis": "1-2 sentence investment thesis in plain English",
  "signal_components": [
    {
      "factor": "factor name (e.g. momentum, low_volatility, earnings_revision)",
      "direction": "positive or negative",
      "lookback_days": integer,
      "description": "how this factor is constructed"
    }
  ],
  "conditions": {
    "macro_regime": "any | bull | bear | sideways | recession | expansion",
    "vix_range": "any | <15 | 15-25 | >25 | >30",
    "rate_environment": "any | rising | falling | stable"
  },
  "expected_holding_days": integer,
  "universe": "SP500 | Russell2000 | SP500_sectors | broad_market",
  "rationale": "2-3 sentences on why this combination should generate alpha",
  "risk_factors": ["list of key risks that could invalidate this hypothesis"]
}

Rules:
- Be specific and testable. Vague hypotheses cannot be backtested.
- Each signal_component must map to a computable feature from price, volume, or fundamental data.
- Conditions define when this signal is expected to work — be precise.
- Risk factors must be genuine threats, not generic disclaimers.
- If research memory is provided showing past failures, you MUST adapt the hypothesis to address
  those specific failure modes. This is not optional. If a memory warns that a VIX<15 gate cannot
  serve as a timely exit signal for holding periods over 15 days, either (a) shorten the holding
  period below that threshold, (b) add an explicit stop-loss/circuit-breaker signal component
  that fires independently of the rebalance cycle, or (c) explicitly document in the rationale
  why this hypothesis is structurally different from the failure case. Silently repeating a
  known failure pattern is a critical error.
- Return ONLY valid JSON. No preamble, no explanation, no markdown fences.
"""

def get_next_hypothesis_number(db: Session) -> int:
    result = db.query(Hypothesis).order_by(Hypothesis.hypothesis_number.desc()).first()
    return (result.hypothesis_number + 1) if result else 1

def retrieve_relevant_memories(db: Session, signal_types: list[str], conditions: dict) -> list[dict]:
    """
    Retrieve research memories relevant to the current observation set.

    First pass: cheap substring match against affected_signal_types and lesson text,
    using both the raw observation keys/values and common factor synonyms. This catches
    the common cases without an extra API call. Falls back to returning all memories
    (capped) if the corpus is small, since at low volume it's cheap to just show everything
    and let the generation prompt itself reason about relevance.
    """
    memories = db.query(ResearchMemory).all()
    if not memories:
        return []

    # Build a flat text blob of everything we know about this observation set
    obs_text = " ".join(signal_types) + " " + " ".join(str(v) for v in conditions.values())
    obs_text = obs_text.lower()

    # Common factor synonym expansion so 'momentum_signal' matches 'momentum' etc.
    synonym_map = {
        "momentum": ["momentum", "trend", "price_momentum"],
        "earnings_revision": ["earnings", "revision", "eps"],
        "institutional_flow_proxy": ["institutional", "flow", "accumulation", "volume"],
        "volatility": ["volatility", "vol", "vix"],
    }

    relevant = []
    for m in memories:
        affected = [s.lower() for s in (m.affected_signal_types or [])]
        match = False

        # Direct substring match against affected signal types
        for factor in affected:
            if factor in obs_text:
                match = True
                break
            # Synonym expansion match
            for canonical, synonyms in synonym_map.items():
                if factor == canonical or factor in synonyms:
                    if any(syn in obs_text for syn in synonyms):
                        match = True
                        break
            if match:
                break

        # Also match on regime conditions overlap (e.g. both mention vix, expansion)
        if not match and m.conditions:
            mem_conditions_text = " ".join(str(v) for v in m.conditions.values()).lower()
            condition_words = [w for w in mem_conditions_text.split() if len(w) > 3]
            if any(w in obs_text for w in condition_words):
                match = True

        if match:
            relevant.append({
                "lesson": m.lesson,
                "conditions": m.conditions,
                "failure_mode": m.failure_mode,
                "structured_constraint": m.structured_constraint,
                "affected_signal_types": m.affected_signal_types
            })

    # If the corpus is small (<=10 memories) and nothing matched, surface everything anyway —
    # cheap insurance against false negatives while the memory base is young.
    if not relevant and len(memories) <= 10:
        relevant = [{
            "lesson": m.lesson,
            "conditions": m.conditions,
            "failure_mode": m.failure_mode,
            "structured_constraint": m.structured_constraint,
            "affected_signal_types": m.affected_signal_types
        } for m in memories]

    return relevant[:5]

def build_observation_prompt(observations: dict,
                              memories: list[dict],
                              active_constraints: str = "") -> str:
    ks = get_knowledge_store()

    top_tech = ks.get_sector_tickers("Information Technology")
    top_fin = ks.get_sector_tickers("Financials")
    nvda_etfs = [e["etf"] for e in ks.get_etf_exposure("NVDA")]
    momentum_concentration = ks.get_sector_concentration(
        ["NVDA", "AMD", "META", "MSFT", "AAPL", "GOOGL", "AMZN"]
    )

    knowledge_context = f"""
Knowledge Graph Context (use when reasoning about sector concentration and crowding):
- Information Technology tickers in universe: {top_tech}
- Financials tickers in universe: {top_fin}
- NVDA ETF memberships: {nvda_etfs}
- Top momentum names sector concentration: {json.dumps({k: f"{v['pct']*100:.0f}%" for k, v in momentum_concentration.items()})}
- Warning: top momentum names are 57% Information Technology — high concentration risk
"""

    prompt = f"""Current market observations:
{json.dumps(observations, indent=2)}

{knowledge_context}
"""

    # Active constraints take priority over passive memories
    if active_constraints:
        prompt += f"""
{active_constraints}

"""
    elif memories:
        prompt += f"""Research memory — past failures relevant to these conditions:
{json.dumps(memories, indent=2)}

Apply these lessons when constructing the hypothesis. If a past failure directly applies,
adjust the signal components or conditions to avoid repeating it.

"""

    prompt += "Generate one high-quality investment hypothesis based on these observations."
    return prompt

def generate_hypothesis(db: Session, observations: dict,
                         use_memory_service: bool = True) -> Hypothesis:
    signal_hints = list(observations.keys())

    # Use the reasoning service if available, fall back to passive retrieval
    active_constraints = ""
    memories = []

    if use_memory_service:
        try:
            from src.agents.research_memory_service import ResearchMemoryService
            service = ResearchMemoryService(db)

            # Infer signal types from observation keys for constraint compilation
            inferred_signals = []
            obs_text = " ".join(str(v) for v in observations.values()).lower()
            if "momentum" in obs_text: inferred_signals.append("price_momentum")
            if "earnings" in obs_text or "revision" in obs_text:
                inferred_signals.append("earnings_revision")
            if "volatil" in obs_text or "vix" in obs_text:
                inferred_signals.append("volatility")
            if "institutional" in obs_text or "flow" in obs_text:
                inferred_signals.append("institutional_flow_proxy")

            # Infer conditions
            inferred_conditions = {}
            if "expansion" in obs_text: inferred_conditions["macro_regime"] = "expansion"
            if "low vol" in obs_text or "vix" in obs_text:
                inferred_conditions["vix_range"] = "<15"
            if "stable" in obs_text: inferred_conditions["rate_environment"] = "stable"

            applicable = service.get_applicable_constraints(
                signal_types=inferred_signals,
                conditions=inferred_conditions,
                holding_days=21
            )

            if applicable:
                active_constraints = service.format_constraints_for_scientist(applicable)
                print(f"  Memory Service: {len(applicable)} active constraint(s) compiled")
                for c in applicable:
                    print(f"    [{c['failure_mode']}]: {c['constraint'][:80]}")
            else:
                print("  Memory Service: no applicable constraints found")

        except Exception as e:
            print(f"  Memory Service unavailable, using passive retrieval: {e}")
            memories = retrieve_relevant_memories(db, signal_hints, observations)
    else:
        memories = retrieve_relevant_memories(db, signal_hints, observations)

    user_prompt = build_observation_prompt(observations, memories, active_constraints)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )
    if response.stop_reason == "max_tokens":
        print("  WARNING: hypothesis generation truncated at max_tokens=3000")

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # 3. Parse response — with truncation recovery
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Response was likely truncated mid-JSON by max_tokens limit.
        # Attempt recovery: truncate at the last complete top-level field
        # by finding the last complete closing of a major array/object.
        try:
            # Find the last position where risk_factors or rationale closed cleanly
            # Try progressively shorter truncation points
            recovery_raw = raw
            for closing in ['",\n  "risk_factors"', '",\n  "rationale"',
                            '",\n  "universe"', '",\n  "expected_holding_days"']:
                idx = recovery_raw.rfind(closing)
                if idx > 0:
                    # Truncate before this field and close the JSON
                    truncated = recovery_raw[:idx] + '"\n}'
                    try:
                        data = json.loads(truncated)
                        print(f"  WARNING: JSON recovered via truncation "
                              f"(dropped fields after {closing.strip()})")
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                raise ValueError(
                    f"Research Scientist returned invalid JSON "
                    f"(truncation recovery failed): {e}\nRaw: {raw[:500]}"
                )
        except Exception:
            raise ValueError(
                f"Research Scientist returned invalid JSON: {e}\nRaw: {raw[:500]}"
            )

    # 4. Create Hypothesis record
    hypothesis_number = get_next_hypothesis_number(db)

    hypothesis = Hypothesis(
        id=uuid.uuid4(),
        hypothesis_number=hypothesis_number,
        title=data["title"],
        thesis=data["thesis"],
        signal_components=data["signal_components"],
        conditions=data.get("conditions", {}),
        expected_holding_days=data.get("expected_holding_days"),
        universe=data.get("universe", "SP500"),
        status=HypothesisStatus.DRAFT,
        generated_by="research_scientist"
    )
    db.add(hypothesis)
    db.flush()  # get the UUID before creating related records

    # 5. Create Governance record immediately
    governance = GovernanceRecord(
        id=uuid.uuid4(),
        hypothesis_id=hypothesis.id,
        current_stage=GovernanceStage.IDEA,
        stage_history=[{
            "from_stage": None,
            "to_stage": GovernanceStage.IDEA,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": f"Hypothesis #{hypothesis_number} generated from market observations. "
                     f"Memories applied: {len(memories)}"
        }]
    )
    db.add(governance)

    # 6. Add to experiment queue
    job = ExperimentJob(
        id=uuid.uuid4(),
        hypothesis_id=hypothesis.id,
        status=JobStatus.PENDING,
        priority=JobPriority.NORMAL,
        priority_score=0.5
    )
    db.add(job)

    db.commit()
    db.refresh(hypothesis)

    return hypothesis

def print_hypothesis(h: Hypothesis):
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS #{h.hypothesis_number}")
    print(f"{'='*60}")
    print(f"Title     : {h.title}")
    print(f"Status    : {h.status}")
    print(f"Universe  : {h.universe}")
    print(f"Holding   : {h.expected_holding_days} days")
    print(f"\nThesis:\n{h.thesis}")
    print(f"\nSignal Components:")
    for i, s in enumerate(h.signal_components, 1):
        print(f"  {i}. {s['factor']} ({s['direction']}, {s['lookback_days']}d) — {s['description']}")
    print(f"\nConditions: {json.dumps(h.conditions, indent=2)}")
    print(f"\nID: {h.id}")
    print(f"{'='*60}\n")