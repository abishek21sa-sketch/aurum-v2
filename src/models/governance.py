from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base
from datetime import datetime, timezone
import uuid
import enum

class GovernanceStage(str, enum.Enum):
    IDEA = "idea"
    EXPERIMENT = "experiment"
    STATISTICAL_REVIEW = "statistical_review"
    RISK_REVIEW = "risk_review"
    COMMITTEE = "committee"
    PAPER_TRADING = "paper_trading"
    PRODUCTION = "production"
    MONITORING = "monitoring"
    RETIRED = "retired"

class GovernanceRecord(Base):
    __tablename__ = "governance_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hypothesis_id = Column(UUID(as_uuid=True), ForeignKey("hypotheses.id"), unique=True, nullable=False)

    # Lifecycle tracking
    current_stage = Column(Enum(GovernanceStage), default=GovernanceStage.IDEA, nullable=False)
    stage_history = Column(JSON, default=list)         # [{stage, timestamp, notes}]

    # Review records
    statistical_review = Column(JSON, nullable=True)   # {p_value, t_stat, passed, notes}
    risk_review = Column(JSON, nullable=True)          # {var, cvar, correlation, passed, notes}
    committee_decision = Column(JSON, nullable=True)   # {decision, justification, date, members}

    # Debate record (V2.6)
    debate_record = Column(JSON, nullable=True)        # {bull_thesis, bear_rebuttal, specialist_inputs, resolution}

    # Deployment
    paper_trading_start = Column(DateTime, nullable=True)
    production_start = Column(DateTime, nullable=True)
    retirement_date = Column(DateTime, nullable=True)

    # Post-mortem
    failure_reason = Column(Text, nullable=True)
    lessons_learned = Column(Text, nullable=True)
    retirement_notes = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    hypothesis = relationship("Hypothesis", back_populates="governance_record")

    def advance_stage(self, new_stage: GovernanceStage, notes: str = ""):
        entry = {
            "from_stage": self.current_stage,
            "to_stage": new_stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes
        }
        if self.stage_history is None:
            self.stage_history = []
        self.stage_history = self.stage_history + [entry]
        self.current_stage = new_stage
        self.updated_at = datetime.now(timezone.utc)