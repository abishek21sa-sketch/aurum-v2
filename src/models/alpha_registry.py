from sqlalchemy import Column, String, Text, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from src.core.database import Base
from datetime import datetime, timezone
import uuid

class AlphaSignal(Base):
    __tablename__ = "alpha_registry"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hypothesis_id = Column(UUID(as_uuid=True), ForeignKey("hypotheses.id"), nullable=False)

    # Signal identity
    signal_name = Column(String(255), nullable=False)
    signal_code = Column(Text, nullable=False)           # actual Python feature engineering code
    features_used = Column(JSON, nullable=False)
    universe = Column(String(100), nullable=False)

    # Validated performance
    sharpe_ratio = Column(Float, nullable=False)
    sortino_ratio = Column(Float, nullable=False)
    calmar_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    annualized_return = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    information_coefficient = Column(Float, nullable=True)

    # Robustness
    oos_sharpe = Column(Float, nullable=True)            # out-of-sample Sharpe
    regime_performance = Column(JSON, nullable=True)     # {bull: sharpe, bear: sharpe, sideways: sharpe}
    parameter_sensitivity = Column(JSON, nullable=True)  # did it survive perturbation?

    # Live tracking (updated by V2.7)
    is_active = Column(Boolean, default=True)
    paper_trading_sharpe = Column(Float, nullable=True)
    decay_flag = Column(Boolean, default=False)
    last_evaluated_at = Column(DateTime, nullable=True)
    retirement_recommended = Column(Boolean, default=False)

    # Metadata
    registered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))