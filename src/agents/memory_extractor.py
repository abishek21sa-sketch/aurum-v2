import anthropic
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.core.config import ANTHROPIC_API_KEY
from src.models.hypothesis import Hypothesis
from src.models.governance import GovernanceRecord
from src.models.research_memory import ResearchMemory

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

EXTRACTOR_SYSTEM = """You convert investment committee debate records into structured research
memory entries. Your job is to extract the core lesson from a debate so future hypothesis
generation can avoid repeating the same structural mistake.

You will be given a hypothesis, its signal components, and a full debate record (bull, bear,
risk, judge). Extract ONE structured memory entry capturing the most important lesson —
usually from the bear's strongest objection or the judge's conditions, but only if the
objection was genuinely validated by the judge's decision (not dismissed).

Return ONLY valid JSON in this exact structure, no markdown fences:
{
  "failure_mode": "short_snake_case_category",
  "conditions": {"regime": "...", "vix_range": "...", "rate_env": "..."},
  "lesson": "1-2 sentence human-readable lesson",
  "structured_constraint": {
    "applies_when": "description of when this constraint triggers",
    "required_action": "what the research scientist should do differently"
  },
  "affected_signal_types": ["factor names this applies to, e.g. momentum, earnings_revision"],
  "affected_features": ["specific feature patterns to flag, if any"]
}

failure_mode should be one of: regime_mismatch | timing_mismatch | overfitting | factor_crowding |
missing_validation | selection_bias | other (use sparingly)
"""

def extract_memory_from_debate(db: Session, hypothesis: Hypothesis, gov: GovernanceRecord) -> ResearchMemory | None:
    if not gov.debate_record:
        print(f"  No debate record for Hypothesis #{hypothesis.hypothesis_number}, skipping.")
        return None

    context = f"""Hypothesis #{hypothesis.hypothesis_number}: {hypothesis.title}

Signal components: {json.dumps(hypothesis.signal_components, indent=2)}
Conditions: {json.dumps(hypothesis.conditions, indent=2)}

Full debate record:
{json.dumps(gov.debate_record, indent=2)}

Committee decision: {gov.committee_decision.get('decision') if gov.committee_decision else 'unknown'}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=EXTRACTOR_SYSTEM,
        messages=[{"role": "user", "content": context}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    memory = ResearchMemory(
        id=uuid.uuid4(),
        source_hypothesis_id=hypothesis.id,
        source_hypothesis_number=hypothesis.hypothesis_number,
        failure_mode=data["failure_mode"],
        conditions=data["conditions"],
        lesson=data["lesson"],
        structured_constraint=data["structured_constraint"],
        affected_signal_types=data["affected_signal_types"],
        affected_features=data.get("affected_features", []),
        created_by="memory_extractor"
    )
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory

def print_memory(memory: ResearchMemory):
    print(f"\n{'='*60}")
    print(f"RESEARCH MEMORY — from Hypothesis #{memory.source_hypothesis_number}")
    print(f"{'='*60}")
    print(f"Failure mode: {memory.failure_mode}")
    print(f"Conditions: {json.dumps(memory.conditions, indent=2)}")
    print(f"\nLesson:\n{memory.lesson}")
    print(f"\nStructured constraint:\n{json.dumps(memory.structured_constraint, indent=2)}")
    print(f"\nAffected signal types: {memory.affected_signal_types}")
    print(f"{'='*60}\n")