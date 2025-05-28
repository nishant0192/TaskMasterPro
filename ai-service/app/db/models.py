# ai-service/app/db/models.py
from datetime import datetime
from sqlmodel import SQLModel, Field
from typing import Optional

class TrainingData(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    task_id: Optional[str] = Field(default=None, index=True)
    task_description: Optional[str] = Field(default=None)
    initial_priority: Optional[int] = Field(default=None)
    actual_completion_time: Optional[int] = Field(default=None)
    was_deadline_met: Optional[bool] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

class TaskPrediction(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    task_id: str = Field(index=True)
    user_id: Optional[str] = Field(default=None, index=True)
    model_version: Optional[str] = Field(default=None)
    predicted_priority: Optional[int] = Field(default=None)
    predicted_completion_time: Optional[int] = Field(default=None)
    recommended_due_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

class BatchJob(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    job_type: str
    status: str = Field(default="PENDING")
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    details: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
