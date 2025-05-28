import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, JSON, Enum
from sqlalchemy.dialects.postgresql import UUID
from app.db.session import engine
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    input = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(JSON, nullable=True)


# (Only for dev) create table if not exists
async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
