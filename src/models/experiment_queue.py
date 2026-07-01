from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base
from datetime import datetime, timezone
import uuid
import enum

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    MERGED = "merged"

class JobPriority(str, enum.Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    DEFERRED = "deferred"

class ExperimentJob(Base):
    __tablename__ = "experiment_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hypothesis_id = Column(UUID(as_uuid=True), ForeignKey("hypotheses.id"), unique=True, nullable=False)

    # Scheduling
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    priority = Column(Enum(JobPriority), default=JobPriority.NORMAL, nullable=False)
    priority_score = Column(Float, default=0.5)          # 0-1, computed by scheduler

    # Scheduler reasoning
    scheduler_notes = Column(JSON, nullable=True)        # {action: "delay", reason: "similar to #14 running"}
    merged_into_job_id = Column(UUID(as_uuid=True), nullable=True)

    # Estimated cost
    estimated_backtest_seconds = Column(Integer, nullable=True)
    estimated_api_calls = Column(Integer, nullable=True)

    # Timing
    scheduled_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    hypothesis = relationship("Hypothesis", back_populates="experiment_job")