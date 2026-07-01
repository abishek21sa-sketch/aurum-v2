from sqlalchemy import Column, String, Text, DateTime, JSON, Float, Integer
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from src.core.database import Base
from datetime import datetime, timezone
import uuid

class ResearchMemory(Base):
    __tablename__ = "research_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # What failed and why
    source_hypothesis_id = Column(UUID(as_uuid=True), nullable=False)
    source_hypothesis_number = Column(Integer, nullable=False)
    failure_mode = Column(String(100), nullable=False)   # regime_mismatch | data_leakage | overfitting | factor_crowding | ...

    # Conditions under which it failed
    conditions = Column(JSON, nullable=False)            # {"regime": "recession", "vix_range": ">30", "rate_env": "rising"}

    # The lesson
    lesson = Column(Text, nullable=False)                # human-readable constraint
    structured_constraint = Column(JSON, nullable=False) # machine-readable version for the scientist

    # What signal types this applies to
    affected_signal_types = Column(JSON, nullable=False) # ["momentum", "low_vol"] etc.
    affected_features = Column(JSON, nullable=True)      # specific features to avoid/adjust

    # Confidence in this memory
    confidence_score = Column(Float, default=1.0)        # degrades if similar signals later succeed
    times_validated = Column(Integer, default=0)         # how many times this lesson prevented a bad hypothesis
    times_overridden = Column(Integer, default=0)        # how many times it was ignored and signal still worked

    # Source
    # Source
    created_by = Column(String(50), default="governance_layer")
    influenced_hypothesis_numbers = Column(JSON, default=list)  # which later hypotheses applied this memory
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))