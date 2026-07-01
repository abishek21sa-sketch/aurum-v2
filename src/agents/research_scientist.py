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

def build_observation_prompt(observations: dict, memories: list[dict]) -> str:
    prompt = f"""Current market observations:
{json.dumps(observations, indent=2)}

"""
    if memories:
        prompt += f"""Research memory — past failures relevant to these conditions:
{json.dumps(memories, indent=2)}

Apply these lessons when constructing the hypothesis. If a past failure directly applies,
adjust the signal components or conditions to avoid repeating it.

"""
    prompt += "Generate one high-quality investment hypothesis based on these observations."
    return prompt

def generate_hypothesis(db: Session, observations: dict) -> Hypothesis:
    """
    Core generation loop.
    observations: dict of current market signals, e.g.:
    {
        "momentum_signal": "strong positive across large caps",
        "volatility_regime": "low (VIX ~14)",
        "earnings_trend": "positive revisions in tech sector",
        "macro_regime": "expansion",
        "rate_environment": "stable"
    }
    """

    # 1. Pull relevant research memories
    signal_hints = list(observations.keys())
    memories = retrieve_relevant_memories(db, signal_hints, observations)

    # 2. Build prompt and call Claude
    user_prompt = build_observation_prompt(observations, memories)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )
    if response.stop_reason == "max_tokens":
        print("  WARNING: hypothesis generation truncated at max_tokens=2000")

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # 3. Parse response
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Research Scientist returned invalid JSON: {e}\nRaw: {raw}")

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