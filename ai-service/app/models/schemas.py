# app/models/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum

# Enums
class TaskStatus(str, Enum):
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    CANCELLED = "CANCELLED"

class TaskPriority(int, Enum):
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class JobType(str, Enum):
    PREDICTION_UPDATE = "PREDICTION_UPDATE"
    MODEL_TRAINING = "MODEL_TRAINING"
    INSIGHT_GENERATION = "INSIGHT_GENERATION"
    BATCH_PRIORITIZATION = "BATCH_PRIORITIZATION"

class TimePeriod(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

# Base Models
class TaskBase(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.TODO
    priority: Optional[TaskPriority] = None
    due_date: Optional[datetime] = None
    created_at: datetime
    estimated_duration: Optional[int] = None  # in minutes
    category: Optional[str] = None
    tags: List[str] = []

class CalendarEvent(BaseModel):
    id: str
    title: str
    start_time: datetime
    end_time: datetime
    is_busy: bool = True
    location: Optional[str] = None

class UserPreferences(BaseModel):
    work_hours_start: int = Field(9, ge=0, le=23)  # 9 AM
    work_hours_end: int = Field(17, ge=0, le=23)   # 5 PM
    work_days: List[int] = Field([1, 2, 3, 4, 5])  # Monday to Friday
    timezone: str = "UTC"
    break_duration: int = Field(15, ge=5, le=60)   # minutes
    max_work_hours_per_day: int = Field(8, ge=1, le=16)
    productivity_hours: List[int] = Field([9, 10, 14, 15])  # Most productive hours

class TimeRange(BaseModel):
    start_date: date
    end_date: date

# Request Models
class TaskPrioritizationRequest(BaseModel):
    tasks: List[TaskBase]
    context: Optional[Dict[str, Any]] = {}
    user_preferences: Optional[UserPreferences] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SchedulingRequest(BaseModel):
    tasks: List[TaskBase]
    calendar_events: List[CalendarEvent] = []
    preferences: UserPreferences
    time_range: TimeRange
    optimization_goals: List[str] = ["minimize_stress", "maximize_productivity"]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class NLPRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    context: Optional[Dict[str, Any]] = {}
    user_timezone: str = "UTC"
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class InsightRequest(BaseModel):
    time_period: TimePeriod = TimePeriod.WEEK
    metrics: List[str] = ["productivity", "task_completion", "time_management"]
    include_recommendations: bool = True

class PredictionRequest(BaseModel):
    tasks: List[TaskBase]
    historical_data: Optional[Dict[str, Any]] = {}
    prediction_horizon: int = Field(7, ge=1, le=30)  # days
    
class BatchJobRequest(BaseModel):
    job_type: JobType
    parameters: Dict[str, Any] = {}
    details: Optional[str] = None

class TrainingDataRequest(BaseModel):
    task_id: str
    initial_priority: Optional[int] = None
    actual_completion_time: Optional[int] = None  # minutes
    was_deadline_met: Optional[bool] = None

# Response Models
class PrioritizedTask(TaskBase):
    ai_priority_score: float = Field(..., ge=0, le=1)
    priority_factors: Dict[str, float] = {}
    recommended_start_time: Optional[datetime] = None

class TaskPrioritizationResponse(BaseModel):
    prioritized_tasks: List[PrioritizedTask]
    confidence_scores: Dict[str, float]
    reasoning: List[str]
    model_version: str
    processing_time_ms: int

class ScheduleTimeBlock(BaseModel):
    task_id: str
    start_time: datetime
    end_time: datetime
    confidence_score: float = Field(..., ge=0, le=1)
    flexibility_score: float = Field(..., ge=0, le=1)
    energy_level_required: str = Field("medium", regex="^(low|medium|high)$")

class ProductivityInsight(BaseModel):
    metric: str
    value: float
    trend: str = Field(..., regex="^(increasing|decreasing|stable)$")
    description: str
    recommendation: Optional[str] = None

class SchedulingResponse(BaseModel):
    optimized_schedule: List[ScheduleTimeBlock]
    suggested_time_blocks: List[ScheduleTimeBlock]
    productivity_insights: List[ProductivityInsight]
    alternative_schedules: List[List[ScheduleTimeBlock]] = []
    optimization_score: float = Field(..., ge=0, le=1)
    processing_time_ms: int

class ExtractedEntity(BaseModel):
    entity_type: str  # "date", "time", "priority", "person", "location"
    value: str
    confidence: float = Field(..., ge=0, le=1)
    start_pos: int
    end_pos: int

class ParsedTask(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: Optional[TaskPriority] = None
    estimated_duration: Optional[int] = None
    category: Optional[str] = None
    tags: List[str] = []
    reminder_time: Optional[datetime] = None

class NLPResponse(BaseModel):
    parsed_task: ParsedTask
    extracted_entities: List[ExtractedEntity]
    confidence_score: float = Field(..., ge=0, le=1)
    suggestions: List[str] = []
    alternative_interpretations: List[ParsedTask] = []

class Insight(BaseModel):
    id: str
    type: str  # "productivity", "time_management", "goal_progress"
    title: str
    description: str
    severity: str = Field(..., regex="^(low|medium|high|critical)$")
    actionable: bool = True
    created_at: datetime

class Recommendation(BaseModel):
    id: str
    title: str
    description: str
    impact_score: float = Field(..., ge=0, le=1)
    effort_required: str = Field(..., regex="^(low|medium|high)$")
    category: str
    action_items: List[str] = []

class Trend(BaseModel):
    metric: str
    direction: str = Field(..., regex="^(up|down|stable)$")
    change_percentage: float
    time_period: str
    significance: str = Field(..., regex="^(low|medium|high)$")

class GoalProgress(BaseModel):
    goal_id: str
    goal_title: str
    current_progress: float = Field(..., ge=0, le=1)
    target_date: Optional[date] = None
    on_track: bool
    estimated_completion: Optional[date] = None

class InsightResponse(BaseModel):
    productivity_score: float = Field(..., ge=0, le=100)
    insights: List[Insight]
    recommendations: List[Recommendation]
    trends: List[Trend]
    goal_progress: List[GoalProgress]
    analysis_period: TimePeriod
    generated_at: datetime

class TaskCompletionPrediction(BaseModel):
    task_id: str
    estimated_completion_time: int  # minutes
    probability_on_time: float = Field(..., ge=0, le=1)
    predicted_completion_date: datetime
    confidence_interval: Dict[str, int]  # {"min": 60, "max": 120}

class RiskFactor(BaseModel):
    factor: str
    impact: str = Field(..., regex="^(low|medium|high|critical)$")
    description: str
    mitigation_suggestions: List[str] = []

class PredictionResponse(BaseModel):
    completion_predictions: List[TaskCompletionPrediction]
    deadline_probabilities: Dict[str, float]  # task_id -> probability
    risk_factors: List[RiskFactor]
    recommendations: List[str]
    model_accuracy: float = Field(..., ge=0, le=1)
    prediction_horizon_days: int

class BatchJobResponse(BaseModel):
    job_id: str
    status: str = Field(..., regex="^(PENDING|RUNNING|COMPLETED|FAILED)$")
    created_at: datetime
    estimated_completion: Optional[datetime] = None
    progress_percentage: Optional[float] = Field(None, ge=0, le=100)

# Database Models (SQLAlchemy)
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid

class TaskPrediction(Base):
    __tablename__ = "task_predictions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=True, index=True)
    model_version = Column(String, nullable=True)
    predicted_priority = Column(Integer, nullable=True)
    predicted_completion_time = Column(Integer, nullable=True)  # minutes
    recommended_due_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<TaskPrediction(id='{self.id}', task_id='{self.task_id}')>"

class TrainingData(Base):
    __tablename__ = "training_data"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True, index=True)
    task_id = Column(String, nullable=True, index=True)
    initial_priority = Column(Integer, nullable=True)
    actual_completion_time = Column(Integer, nullable=True)  # minutes
    was_deadline_met = Column(Boolean, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<TrainingData(id='{self.id}', user_id='{self.user_id}')>"

class BatchJob(Base):
    __tablename__ = "batch_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    details = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<BatchJob(id='{self.id}', type='{self.job_type}', status='{self.status}')>"

class UserAnalytics(Base):
    __tablename__ = "user_analytics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    productivity_score = Column(Float, nullable=True)
    tasks_completed = Column(Integer, nullable=False, default=0)
    tasks_created = Column(Integer, nullable=False, default=0)
    average_completion_time = Column(Float, nullable=True)  # minutes
    deadlines_met = Column(Integer, nullable=False, default=0)
    deadlines_missed = Column(Integer, nullable=False, default=0)
    focus_time_minutes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserAnalytics(user_id='{self.user_id}', date='{self.date}')>"

class AIInsight(Base):
    __tablename__ = "ai_insights"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    insight_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String, nullable=False, default="medium")
    actionable = Column(Boolean, nullable=False, default=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_dismissed = Column(Boolean, nullable=False, default=False)
    
    def __repr__(self):
        return f"<AIInsight(id='{self.id}', type='{self.insight_type}')>"

# Validation Schemas
class CreateTaskPredictionSchema(BaseModel):
    task_id: str
    user_id: Optional[str] = None
    predicted_priority: Optional[int] = Field(None, ge=1, le=5)
    predicted_completion_time: Optional[int] = Field(None, ge=1)
    recommended_due_date: Optional[datetime] = None
    model_version: Optional[str] = None

class CreateTrainingDataSchema(BaseModel):
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    initial_priority: Optional[int] = Field(None, ge=1, le=5)
    actual_completion_time: Optional[int] = Field(None, ge=1)
    was_deadline_met: Optional[bool] = None

class CreateBatchJobSchema(BaseModel):
    job_type: JobType
    details: Optional[Dict[str, Any]] = None

class UpdateBatchJobSchema(BaseModel):
    status: Optional[str] = Field(None, regex="^(PENDING|RUNNING|COMPLETED|FAILED)$")
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None

# Health Check Schemas
class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime
    service: str = "TaskMaster Pro AI Service"
    version: str = "1.0.0"
    database_status: str = "connected"
    cache_status: str = "connected"
    models_loaded: Dict[str, bool] = {}

# Error Schemas
class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime
    request_id: Optional[str] = None

class ValidationErrorResponse(BaseModel):
    error: str = "Validation Error"
    details: List[Dict[str, Any]]
    timestamp: datetime