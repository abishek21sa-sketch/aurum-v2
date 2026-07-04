import anthropic
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.core.config import ANTHROPIC_API_KEY
from src.models.research_memory import ResearchMemory
from src.models.hypothesis import Hypothesis

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Constraint Compiler ───────────────────────────────────────────

COMPILER_SYSTEM = """You are the AURUM Research Memory Compiler.

Given a research memory (a documented past failure) and a new hypothesis being considered,
you determine:
1. Whether this memory applies to the new hypothesis
2. If it applies, what specific structural constraint it imposes on the hypothesis design
3. What verification check would confirm the constraint was honored

Return ONLY valid JSON, no markdown fences:
{
  "applies": true | false,
  "applicability_reasoning": "1-2 sentences on why this memory does or does not apply",
  "constraint": "specific actionable constraint on hypothesis design, or null if not applicable",
  "verification_check": "what to look for in the generated hypothesis to confirm compliance, or null"
}"""

def compile_constraint(memory: ResearchMemory, hypothesis_spec: dict) -> dict:
    """
    Given a memory and a hypothesis being planned (not yet generated),
    determine whether the memory applies and what constraint it imposes.
    """
    context = f"""Research memory:
Failure mode: {memory.failure_mode}
Lesson: {memory.lesson}
Structured constraint: {json.dumps(memory.structured_constraint)}
Affected signal types: {memory.affected_signal_types}
Conditions when failure occurred: {json.dumps(memory.conditions)}

New hypothesis being planned:
Signal types proposed: {hypothesis_spec.get('signal_types', [])}
Conditions proposed: {json.dumps(hypothesis_spec.get('conditions', {}))}
Holding period proposed: {hypothesis_spec.get('holding_days')} days
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=COMPILER_SYSTEM,
        messages=[{"role": "user", "content": context}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Truncate at last complete field if JSON is malformed
        # Common cause: long constraint strings with unescaped apostrophes
        import re
        # Try to extract just the applies field to at least get a binary decision
        applies_match = re.search(r'"applies"\s*:\s*(true|false)', raw)
        if applies_match:
            applies = applies_match.group(1) == "true"
            return {
                "applies": applies,
                "applicability_reasoning": "constraint text truncated due to parse error",
                "constraint": None,
                "verification_check": None
            }
        return {"applies": False, "applicability_reasoning": "parse error", "constraint": None, "verification_check": None}

# ── Compliance Verifier ───────────────────────────────────────────

VERIFIER_SYSTEM = """You are the AURUM Research Memory Compliance Verifier.

Given a memory constraint that was supposed to be applied to a hypothesis,
and the actual generated hypothesis, determine whether the constraint was honored.

Be strict. "Mentioned in passing" is not compliance.
The hypothesis must structurally differ from the failure pattern in a verifiable way.

Return ONLY valid JSON, no markdown fences:
{
  "compliant": true | false,
  "compliance_evidence": "specific text or structural feature in the hypothesis that shows compliance, or null",
  "violation_description": "what was missed or inadequately addressed, or null if compliant",
  "compliance_score": <float 0-1>
}"""

def verify_compliance(
    memory: ResearchMemory,
    constraint: str,
    verification_check: str,
    hypothesis: Hypothesis
) -> dict:
    """
    After hypothesis generation, verify whether the constraint was honored.
    """
    context = f"""Memory constraint that should have been applied:
{constraint}

Verification check to perform:
{verification_check}

Generated hypothesis:
Title: {hypothesis.title}
Thesis: {hypothesis.thesis}
Signal components: {json.dumps(hypothesis.signal_components, indent=2)}
Conditions: {json.dumps(hypothesis.conditions)}
Holding period: {hypothesis.expected_holding_days} days
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=VERIFIER_SYSTEM,
        messages=[{"role": "user", "content": context}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)

# ── Memory Service ────────────────────────────────────────────────

class ResearchMemoryService:
    """
    Upgrades Research Memory from passive storage to active reasoning.

    Instead of:
      retrieve memories → pass to scientist → hope they apply

    Does:
      retrieve memories → compile applicable constraints → scientist generates
      → verify compliance → update memory confidence scores
    """

    def __init__(self, db: Session):
        self.db = db

    def get_applicable_constraints(
        self,
        signal_types: list[str],
        conditions: dict,
        holding_days: int = 21
    ) -> list[dict]:
        """
        Retrieve all memories, compile constraints for applicable ones,
        return only the ones that actually apply with their specific constraints.
        """
        memories = self.db.query(ResearchMemory).filter(
            ResearchMemory.confidence_score > 0.3
        ).order_by(ResearchMemory.created_at).all()

        if not memories:
            return []

        hypothesis_spec = {
            "signal_types": signal_types,
            "conditions": conditions,
            "holding_days": holding_days
        }

        applicable = []
        for memory in memories:
            # Quick keyword pre-filter before paying for LLM call
            affected = [s.lower() for s in (memory.affected_signal_types or [])]
            signal_lower = [s.lower() for s in signal_types]
            conditions_text = " ".join(str(v) for v in conditions.values()).lower()

            # Check any overlap in signal types or conditions
            has_overlap = (
                any(a in " ".join(signal_lower) for a in affected) or
                any(s in " ".join(affected) for s in signal_lower) or
                any(word in conditions_text
                    for word in ["vix", "expansion", "momentum", "vol", "rate"]
                    if word in (memory.lesson or "").lower())
            )

            if not has_overlap and len(memories) > 5:
                # Skip LLM call for clearly irrelevant memories
                # when corpus is large enough to be selective
                continue

            compiled = compile_constraint(memory, hypothesis_spec)
            if compiled.get("applies"):
                applicable.append({
                    "memory_id": str(memory.id),
                    "failure_mode": memory.failure_mode,
                    "lesson": memory.lesson,
                    "constraint": compiled["constraint"],
                    "verification_check": compiled["verification_check"],
                    "applicability_reasoning": compiled["applicability_reasoning"],
                    "confidence_score": memory.confidence_score
                })

        return applicable

    def verify_hypothesis_compliance(
        self,
        hypothesis: Hypothesis,
        applied_constraints: list[dict]
    ) -> dict:
        """
        After generation, check each applied constraint was actually honored.
        Updates memory confidence scores based on compliance.
        """
        if not applied_constraints:
            return {"verified": True, "checks": [], "overall_compliance": 1.0}

        checks = []
        total_score = 0.0

        for constraint_info in applied_constraints:
            if not constraint_info.get("constraint"):
                continue

            result = verify_compliance(
                memory=None,  # not needed for verification
                constraint=constraint_info["constraint"],
                verification_check=constraint_info.get("verification_check", ""),
                hypothesis=hypothesis
            )

            checks.append({
                "failure_mode": constraint_info["failure_mode"],
                "constraint": constraint_info["constraint"],
                "compliant": result["compliant"],
                "compliance_score": result["compliance_score"],
                "evidence": result.get("compliance_evidence"),
                "violation": result.get("violation_description")
            })

            total_score += result["compliance_score"]

            # Update memory confidence based on compliance
            memory = self.db.query(ResearchMemory).filter_by(
                id=constraint_info["memory_id"]
            ).first()
            if memory:
                if result["compliant"]:
                    memory.times_validated = (memory.times_validated or 0) + 1
                    # Slight confidence boost — constraint was applied correctly
                    memory.confidence_score = min(1.0,
                        (memory.confidence_score or 1.0) * 1.02)
                else:
                    memory.times_overridden = (memory.times_overridden or 0) + 1
                    # Confidence decay — constraint was not honored
                    memory.confidence_score = max(0.1,
                        (memory.confidence_score or 1.0) * 0.95)
                memory.updated_at = datetime.now(timezone.utc)

        self.db.commit()

        overall = total_score / len(checks) if checks else 1.0
        return {
            "verified": overall >= 0.7,
            "checks": checks,
            "overall_compliance": round(overall, 3),
            "n_constraints_checked": len(checks)
        }

    def format_constraints_for_scientist(
        self,
        applicable_constraints: list[dict]
    ) -> str:
        """
        Format applicable constraints into a clear, actionable prompt block
        for the Research Scientist — stronger than just "here are the memories."
        """
        if not applicable_constraints:
            return ""

        lines = [
            "ACTIVE RESEARCH CONSTRAINTS (from Research Memory — these are MANDATORY):",
            ""
        ]
        for i, c in enumerate(applicable_constraints, 1):
            lines.append(f"Constraint {i} [{c['failure_mode']}]:")
            lines.append(f"  Why it applies: {c['applicability_reasoning']}")
            lines.append(f"  Required: {c['constraint']}")
            lines.append(f"  Verification: {c['verification_check']}")
            lines.append("")

        lines.append(
            "CRITICAL: Each constraint above MUST be structurally honored in the hypothesis design. "
            "Generic acknowledgment is insufficient — the hypothesis architecture must demonstrably "
            "differ from the documented failure pattern."
        )
        return "\n".join(lines)