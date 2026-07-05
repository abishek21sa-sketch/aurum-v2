from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from src.core.database import Base
import uuid

class LearningReport(Base):
    __tablename__ = "learning_reports"

    id = Column(UUID(as_uuid=True), primary_key=True,
                server_default="gen_random_uuid()")
    alpha_id = Column(UUID(as_uuid=True), nullable=True)
    hypothesis_id = Column(UUID(as_uuid=True), nullable=False)
    hypothesis_number = Column(Integer, nullable=False)
    simulation_window_days = Column(Integer, nullable=False)
    simulation_start = Column(DateTime, nullable=True)
    simulation_end = Column(DateTime, nullable=True)
    paper_sharpe = Column(Float, nullable=True)
    paper_max_drawdown = Column(Float, nullable=True)
    paper_win_rate = Column(Float, nullable=True)
    original_sharpe = Column(Float, nullable=True)
    sharpe_degradation = Column(Float, nullable=True)
    vix_max_observed = Column(Float, nullable=True)
    vix_breach_occurred = Column(Boolean, default=False)
    circuit_breaker_triggers = Column(Integer, nullable=True)
    decay_flag = Column(Boolean, default=False)
    retirement_recommended = Column(Boolean, default=False)
    recommended_action = Column(String(20), nullable=True)
    evaluated_at = Column(DateTime(timezone=True),
                          server_default="NOW()")