from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.models.hypothesis import Hypothesis, HypothesisStatus
from src.models.governance import GovernanceRecord, GovernanceStage
from src.models.alpha_registry import AlphaSignal

# ── Next action map ───────────────────────────────────────────────
# For each stage, what needs to happen next and who should do it
STAGE_NEXT_ACTIONS = {
    GovernanceStage.IDEA: {
        "action": "Run backtest",
        "actor": "backtester",
        "command": "python -m tests.test_scientist  # then run backtester",
        "blocking": True
    },
    GovernanceStage.EXPERIMENT: {
        "action": "Run statistical validation",
        "actor": "statistical_validator",
        "command": "python -m tests.test_validator",
        "blocking": True
    },
    GovernanceStage.STATISTICAL_REVIEW: {
        "action": "Run committee debate",
        "actor": "debate_engine",
        "command": "python -m tests.run_more_debates",
        "blocking": True
    },
    GovernanceStage.RISK_REVIEW: {
        "action": "Awaiting committee decision",
        "actor": "committee",
        "command": "python -m tests.run_more_debates",
        "blocking": True
    },
    GovernanceStage.COMMITTEE: {
        "action": "Deploy to paper trading",
        "actor": "dashboard",
        "command": "Use dashboard → Hypothesis Detail → Governance actions",
        "blocking": False
    },
    GovernanceStage.PAPER_TRADING: {
        "action": "Run continuous learning evaluation",
        "actor": "continuous_learning",
        "command": "python -m tests.test_v27_full_registry",
        "blocking": False
    },
    GovernanceStage.PRODUCTION: {
        "action": "Monitor for decay",
        "actor": "continuous_learning",
        "command": "python -m tests.test_v27_full_registry",
        "blocking": False
    },
    GovernanceStage.MONITORING: {
        "action": "Review decay signals",
        "actor": "dashboard",
        "command": "Use dashboard → Alpha Registry → Latest learning report",
        "blocking": False
    },
    GovernanceStage.RETIRED: {
        "action": "No action required",
        "actor": "none",
        "command": "",
        "blocking": False
    },
}

class ResearchTicket:
    """
    A lightweight work item wrapping a hypothesis + governance record.
    Not a DB model — computed on demand from existing records.
    """
    def __init__(self, hypothesis: Hypothesis, gov: GovernanceRecord):
        self.hypothesis = hypothesis
        self.gov = gov
        self.id = str(hypothesis.id)
        self.number = hypothesis.hypothesis_number
        self.title = hypothesis.title
        self.stage = gov.current_stage
        self.assigned_to = getattr(gov, 'assigned_to', None) or "unassigned"
        self.next_action_info = STAGE_NEXT_ACTIONS.get(gov.current_stage, {})
        self.next_action = self.next_action_info.get("action", "Unknown")
        self.actor = self.next_action_info.get("actor", "unknown")
        self.is_blocking = self.next_action_info.get("blocking", True)
        self.command = self.next_action_info.get("command", "")
        self.has_debate = bool(gov.debate_record)
        self.has_backtest = hypothesis.sharpe_ratio is not None
        self.has_stat_review = bool(gov.statistical_review)
        self.committee_decision = (
            gov.committee_decision.get("decision")
            if gov.committee_decision else None
        )
        self.compliance_warning = any(
            "COMPLIANCE_WARNING" in (e.get("notes", ""))
            for e in (gov.stage_history or [])
        )
        self.decay_flag = False
        self.days_in_stage = self._days_in_current_stage()
        self.priority = getattr(gov, 'priority_score', 0.5) or 0.5

    def _days_in_current_stage(self) -> int:
        """How many days has this hypothesis been in its current stage."""
        history = self.gov.stage_history or []
        for entry in reversed(history):
            if entry.get("to_stage") == self.stage.value:
                try:
                    ts = datetime.fromisoformat(entry["timestamp"])
                    return (datetime.now(timezone.utc) - ts).days
                except Exception:
                    return 0
        return 0

class WorkflowEngine:
    def __init__(self, db: Session):
        self.db = db

    def get_all_tickets(self) -> list[ResearchTicket]:
        hypotheses = self.db.query(Hypothesis).order_by(
            Hypothesis.hypothesis_number
        ).all()
        tickets = []
        for h in hypotheses:
            gov = self.db.query(GovernanceRecord).filter_by(
                hypothesis_id=h.id
            ).first()
            if gov:
                ticket = ResearchTicket(h, gov)
                # Check decay from alpha registry
                alpha = self.db.query(AlphaSignal).filter_by(
                    hypothesis_id=h.id
                ).first()
                if alpha:
                    ticket.decay_flag = alpha.decay_flag or False
                tickets.append(ticket)
        return tickets

    def get_experiment_board(self) -> dict[str, list[ResearchTicket]]:
        """All tickets grouped by stage — the kanban view."""
        tickets = self.get_all_tickets()
        board = {stage.value: [] for stage in GovernanceStage}
        for ticket in tickets:
            stage_key = ticket.stage.value
            if stage_key in board:
                board[stage_key].append(ticket)
        # Remove empty stages
        return {k: v for k, v in board.items() if v}

    def get_approval_queue(self) -> list[ResearchTicket]:
        """
        Tickets that are blocked and need a decision to advance.
        Sorted by days_in_stage descending (oldest first).
        """
        tickets = self.get_all_tickets()
        blocked = [
            t for t in tickets
            if t.is_blocking
            and t.stage != GovernanceStage.RETIRED
            and t.stage != GovernanceStage.PRODUCTION
        ]
        return sorted(blocked, key=lambda t: -t.days_in_stage)

    def get_paper_trading_queue(self) -> list[ResearchTicket]:
        """Active paper-trading hypotheses with learning report data."""
        tickets = self.get_all_tickets()
        return [t for t in tickets if t.stage == GovernanceStage.PAPER_TRADING]

    def get_retirement_queue(self) -> list[ResearchTicket]:
        """Signals flagged for decay or with revision requests outstanding."""
        tickets = self.get_all_tickets()
        retirement_candidates = []
        for t in tickets:
            # Decay-flagged alphas
            if t.decay_flag:
                retirement_candidates.append(t)
                continue
            # Stuck in revision-requested state
            if (t.stage not in [GovernanceStage.RETIRED, GovernanceStage.PRODUCTION]
                    and t.committee_decision == "request_revision"
                    and t.days_in_stage > 7):
                retirement_candidates.append(t)
        return retirement_candidates

    def get_needs_attention(self) -> list[dict]:
        """
        High-level summary of what needs human attention right now.
        The equivalent of a morning briefing for the research team.
        """
        tickets = self.get_all_tickets()
        items = []

        # Compliance warnings not yet reviewed
        compliance_warnings = [
            t for t in tickets if t.compliance_warning
            and t.stage not in [GovernanceStage.RETIRED]
        ]
        if compliance_warnings:
            items.append({
                "priority": "high",
                "type": "compliance_warning",
                "message": f"{len(compliance_warnings)} hypothesis/hypotheses have unresolved compliance warnings",
                "tickets": [t.number for t in compliance_warnings]
            })

        # Decay-flagged alphas
        decaying = [t for t in tickets if t.decay_flag]
        if decaying:
            items.append({
                "priority": "high",
                "type": "decay",
                "message": f"{len(decaying)} registered alpha(s) showing decay — consider IMPROVE or RETIRE",
                "tickets": [t.number for t in decaying]
            })

        # Stuck tickets (>3 days in blocking stage)
        stuck = [
            t for t in tickets
            if t.is_blocking and t.days_in_stage > 3
            and t.stage != GovernanceStage.RETIRED
        ]
        if stuck:
            items.append({
                "priority": "medium",
                "type": "stuck",
                "message": f"{len(stuck)} ticket(s) stuck in blocking stages for >3 days",
                "tickets": [t.number for t in stuck]
            })

        # Pending approval queue
        approval = self.get_approval_queue()
        if approval:
            items.append({
                "priority": "medium",
                "type": "approval_needed",
                "message": f"{len(approval)} ticket(s) waiting for next pipeline step",
                "tickets": [t.number for t in approval]
            })

        return items