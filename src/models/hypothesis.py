from sqlalchemy import Column, String, Text, Float, Integer, DateTime, JSON, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base
from datetime import datetime, timezone
import uuid
import enum

class HypothesisStatus(str, enum.Enum):
    DRAFT = "draft"
    QUEUED = "queued"
    BACKTESTING = "backtesting"
    VALIDATED = "validated"
    DEBATE = "debate"
    COMMITTEE = "committee"
    PAPER_TRADING = "paper_trading"
    PRODUCTION = "production"
    RETIRED = "retired"
    FAILED = "failed"

class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hypothesis_number = Column(Integer, unique=True, nullable=False)

    # Core content
    title = Column(String(255), nullable=False)
    thesis = Column(Text, nullable=False)
    signal_components = Column(JSON, nullable=False)   # list of signal descriptors
    conditions = Column(JSON, nullable=True)           # {"regime": "...", "vix": "..."}
    expected_holding_days = Column(Integer, nullable=True)
    universe = Column(String(100), nullable=True)      # e.g. "SP500", "Russell2000"

    # Status
    status = Column(Enum(HypothesisStatus), default=HypothesisStatus.DRAFT, nullable=False)

    # Backtest results (populated after V2.4)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    annualized_return = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    backtest_period_start = Column(DateTime, nullable=True)
    backtest_period_end = Column(DateTime, nullable=True)

    # Metadata
    generated_by = Column(String(50), default="research_scientist")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    governance_record = relationship("GovernanceRecord", back_populates="hypothesis", uselist=False)
    experiment_job = relationship("ExperimentJob", back_populates="hypothesis", uselist=False)