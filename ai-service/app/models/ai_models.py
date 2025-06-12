# ai-service/app/models/ai_models.py
"""
SQLAlchemy models for AI service database tables
Production-ready with proper indexing and relationships
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, LargeBinary, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
from datetime import datetime

Base = declarative_base()

class UserBehaviorPattern(Base):
    """
    Stores user behavior patterns for personalization
    """
    __tablename__ = "user_behavior_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)  # 'time_preference', 'priority_style', etc.
    pattern_data = Column(JSONB, nullable=True)
    confidence_score = Column(Float, nullable=True, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<UserBehaviorPattern(user_id={self.user_id}, type={self.pattern_type})>"

class UserModelWeights(Base):
    """
    Stores serialized AI model weights for each user
    """
    __tablename__ = "user_model_weights"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    model_type = Column(String(50), nullable=False, index=True)  # 'prioritization', 'scheduling', 'nlp'
    weights = Column(LargeBinary, nullable=False)  # Serialized model weights
    accuracy_score = Column(Float, nullable=True)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<UserModelWeights(user_id={self.user_id}, type={self.model_type}, v={self.version})>"

class TaskEmbedding(Base):
    """
    Stores task embeddings for semantic similarity search
    Requires pgvector extension
    """
    __tablename__ = "task_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    task_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    embedding = Column("embedding", String)  # Will be vector(384) with pgvector
    category = Column(String(100), nullable=True, index=True)
    completion_time = Column(Integer, nullable=True)  # Minutes to complete
    success_score = Column(Float, nullable=True)  # 0-1 completion success
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<TaskEmbedding(task_id={self.task_id}, category={self.category})>"

class UserPreferenceEmbedding(Base):
    """
    Stores user preference embeddings for personalization
    """
    __tablename__ = "user_preference_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    preference_type = Column(String(50), nullable=False, index=True)
    embedding = Column("embedding", String)  # Will be vector(384) with pgvector
    weight = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<UserPreferenceEmbedding(user_id={self.user_id}, type={self.preference_type})>"

class AITrainingSession(Base):
    """
    Tracks AI model training sessions for monitoring and debugging
    """
    __tablename__ = "ai_training_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)  # None for global training
    session_type = Column(String(50), nullable=False, index=True)
    training_data = Column(JSONB, nullable=True)
    feedback_data = Column(JSONB, nullable=True)
    performance_metrics = Column(JSONB, nullable=True)
    model_version = Column(String(50), nullable=True)
    status = Column(String(20), nullable=False, default='started', index=True)  # started, completed, failed
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<AITrainingSession(id={self.id}, type={self.session_type}, status={self.status})>"

class UserVocabulary(Base):
    """
    Stores user-specific vocabulary and terminology for NLP personalization
    """
    __tablename__ = "user_vocabulary"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    term = Column(String(100), nullable=False, index=True)
    category = Column(String(50), nullable=True, index=True)
    frequency = Column(Integer, nullable=False, default=1)
    context_embedding = Column("context_embedding", String)  # Vector for term context
    importance_score = Column(Float, nullable=False, default=0.5)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<UserVocabulary(user_id={self.user_id}, term={self.term})>"

class AIModelRegistry(Base):
    """
    Registry of all AI models with versioning and metadata
    """
    __tablename__ = "ai_model_registry"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False, index=True)  # 'global', 'user_specific'
    description = Column(Text, nullable=True)
    architecture = Column(JSONB, nullable=True)  # Model architecture details
    hyperparameters = Column(JSONB, nullable=True)
    training_config = Column(JSONB, nullable=True)
    benchmark_scores = Column(JSONB, nullable=True)
    validation_metrics = Column(JSONB, nullable=True)
    is_active = Column(Boolean, nullable=False, default=False, index=True)
    deployment_date = Column(DateTime(timezone=True), nullable=True)
    deprecated_date = Column(DateTime(timezone=True), nullable=True)
    model_file_path = Column(String(500), nullable=True)
    checkpoint_path = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<AIModelRegistry(name={self.model_name}, version={self.model_version})>"

class PredictionLog(Base):
    """
    Logs all AI predictions for monitoring and feedback collection
    """
    __tablename__ = "prediction_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    prediction_type = Column(String(50), nullable=False, index=True)  # 'priority', 'time', 'completion'
    input_data = Column(JSONB, nullable=False)
    prediction_result = Column(JSONB, nullable=False)
    confidence_score = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    feedback_received = Column(Boolean, nullable=False, default=False, index=True)
    actual_outcome = Column(JSONB, nullable=True)
    feedback_timestamp = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<PredictionLog(user_id={self.user_id}, type={self.prediction_type})>"

# ai-service/app/schemas/ai_schemas.py
"""
Pydantic schemas for AI service API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum

class TaskPriority(int, Enum):
    """Task priority levels"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

class TaskStatus(str, Enum):
    """Task status options"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"

class TaskBase(BaseModel):
    """Base task model for AI processing"""
    id: str
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration: Optional[int] = Field(None, gt=0)  # Minutes
    due_date: Optional[datetime] = None
    category: Optional[str] = Field(None, max_length=50)
    tags: List[str] = Field(default_factory=list)
    is_recurring: bool = False
    dependencies: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        use_enum_values = True

class TaskPrioritizationRequest(BaseModel):
    """Request for task prioritization"""
    tasks: List[TaskBase] = Field(..., min_items=1, max_items=100)
    context: Optional[Dict[str, Any]] = None
    prioritization_method: str = Field(default="ai_personalized")
    
    @validator('tasks')
    def validate_tasks_not_empty(cls, v):
        if not v:
            raise ValueError('At least one task is required')
        return v

class PrioritizedTask(TaskBase):
    """Task with AI-generated priority information"""
    ai_priority_score: float = Field(..., ge=0, le=5)
    confidence: float = Field(..., ge=0, le=1)
    ai_explanation: str
    similar_tasks_count: Optional[int] = None

class TaskPrioritizationResponse(BaseModel):
    """Response for task prioritization"""
    prioritized_tasks: List[PrioritizedTask]
    explanations: List[str]
    model_confidence: float = Field(..., ge=0, le=1)
    processing_time_ms: int
    personalization_applied: bool
    recommendations: List[str] = Field(default_factory=list)

class TimeEstimationRequest(BaseModel):
    """Request for task time estimation"""
    task: TaskBase
    context: Optional[Dict[str, Any]] = None
    estimation_method: str = Field(default="ai_personalized")

class TimeEstimationResponse(BaseModel):
    """Response for task time estimation"""
    estimated_duration_minutes: float = Field(..., gt=0)
    confidence: float = Field(..., ge=0, le=1)
    explanation: str
    suggested_buffer_minutes: float = Field(..., ge=0)
    personalization_applied: bool
    similar_tasks_analyzed: int

class UserInsightsRequest(BaseModel):
    """Request for user insights"""
    analysis_period_days: int = Field(default=30, ge=7, le=365)
    insight_types: List[str] = Field(default_factory=list)
    include_recommendations: bool = True

class InsightItem(BaseModel):
    """Individual insight item"""
    type: str
    title: str
    description: str
    confidence: float = Field(..., ge=0, le=1)
    impact_score: float = Field(..., ge=0, le=1)
    supporting_data: Dict[str, Any]
    category: str
    actionable: bool = True

class UserInsightsResponse(BaseModel):
    """Response for user insights"""
    insights: List[InsightItem]
    total_insights: int
    analysis_period_days: int
    data_quality_score: float = Field(..., ge=0, le=1)
    last_updated: datetime

class TrainingDataSample(BaseModel):
    """Single training data sample"""
    task_data: TaskBase
    actual_priority: Optional[int] = Field(None, ge=1, le=5)
    actual_duration: Optional[int] = Field(None, gt=0)
    was_completed: Optional[bool] = None
    completion_date: Optional[datetime] = None
    feedback_score: Optional[float] = Field(None, ge=0, le=1)

class ModelTrainingRequest(BaseModel):
    """Request for model training"""
    training_data: List[TrainingDataSample] = Field(..., min_items=5, max_items=1000)
    training_type: str = Field(default="incremental")  # incremental, full_retrain
    validation_split: float = Field(default=0.2, ge=0.1, le=0.3)

class ModelTrainingResponse(BaseModel):
    """Response for model training"""
    training_id: str
    status: str  # started, in_progress, completed, failed
    estimated_completion_time: int  # seconds
    samples_count: int
    message: str

class BehaviorAnalysisResponse(BaseModel):
    """Response for behavior analysis"""
    analysis_id: str
    status: str
    estimated_completion_time: int
    message: str

class PersonalizationMetricsResponse(BaseModel):
    """Response for personalization metrics"""
    user_id: str
    personalization_level: float = Field(..., ge=0, le=1)
    model_accuracy: float = Field(..., ge=0, le=1)
    training_samples: int
    last_training: Optional[datetime]
    personality_confidence: float = Field(..., ge=0, le=1)
    available_features: List[str]
    recommendations: List[str]

class PredictionResult(BaseModel):
    """Generic prediction result"""
    prediction: float
    confidence: float = Field(..., ge=0, le=1)
    explanation: str
    model_version: datetime
    supporting_data: Dict[str, Any] = Field(default_factory=dict)

class BehaviorPattern(BaseModel):
    """User behavior pattern"""
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence_score: float = Field(..., ge=0, le=1)
    created_at: datetime

class PersonalizationMetrics(BaseModel):
    """Personalization metrics"""
    user_id: str
    accuracy_improvement: float
    prediction_confidence: float
    training_samples: int
    last_updated: datetime

class UserInsight(BaseModel):
    """User behavioral insight"""
    insight_type: str
    description: str
    confidence: float = Field(..., ge=0, le=1)
    impact_score: float = Field(..., ge=0, le=1)
    recommendations: List[str]
    supporting_evidence: Dict[str, Any]

class AdaptationRecommendation(BaseModel):
    """Model adaptation recommendation"""
    user_id: str
    adaptations: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0, le=1)
    expected_improvement: float = Field(..., ge=0, le=1)
    implementation_date: datetime

# Advanced schemas for complex AI operations

class TaskSimilarityRequest(BaseModel):
    """Request for finding similar tasks"""
    reference_task: TaskBase
    candidate_tasks: List[TaskBase]
    similarity_threshold: float = Field(default=0.7, ge=0, le=1)
    max_results: int = Field(default=10, ge=1, le=50)

class SimilarTask(BaseModel):
    """Similar task with similarity score"""
    task: TaskBase
    similarity_score: float = Field(..., ge=0, le=1)
    similarity_explanation: str
    historical_performance: Optional[Dict[str, Any]] = None

class TaskSimilarityResponse(BaseModel):
    """Response for task similarity"""
    similar_tasks: List[SimilarTask]
    total_candidates_analyzed: int
    average_similarity: float
    processing_time_ms: int

class SchedulingOptimizationRequest(BaseModel):
    """Request for AI-powered scheduling optimization"""
    tasks: List[TaskBase]
    available_time_slots: List[Dict[str, Any]]
    constraints: Dict[str, Any] = Field(default_factory=dict)
    optimization_goal: str = Field(default="maximize_productivity")  # maximize_productivity, minimize_stress, balance_workload

class ScheduledTask(BaseModel):
    """Task with AI-optimized scheduling"""
    task: TaskBase
    scheduled_start: datetime
    scheduled_end: datetime
    confidence: float = Field(..., ge=0, le=1)
    scheduling_reason: str
    alternative_slots: List[Dict[str, datetime]] = Field(default_factory=list)

class SchedulingOptimizationResponse(BaseModel):
    """Response for scheduling optimization"""
    scheduled_tasks: List[ScheduledTask]
    unscheduled_tasks: List[TaskBase]
    optimization_score: float = Field(..., ge=0, le=1)
    total_scheduled_time: int  # minutes
    recommendations: List[str]

class ProductivityPredictionRequest(BaseModel):
    """Request for productivity prediction"""
    planned_tasks: List[TaskBase]
    prediction_period: int = Field(..., ge=1, le=30)  # days
    context_factors: Dict[str, Any] = Field(default_factory=dict)

class ProductivityForecast(BaseModel):
    """Productivity forecast for a time period"""
    date: datetime
    predicted_completion_rate: float = Field(..., ge=0, le=1)
    predicted_focus_hours: float = Field(..., ge=0)
    stress_level_prediction: float = Field(..., ge=0, le=1)
    energy_level_prediction: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)

class ProductivityPredictionResponse(BaseModel):
    """Response for productivity prediction"""
    daily_forecasts: List[ProductivityForecast]
    overall_prediction: Dict[str, float]
    risk_factors: List[str]
    optimization_suggestions: List[str]

class AIHealthMetrics(BaseModel):
    """AI service health metrics"""
    model_status: Dict[str, str]
    performance_metrics: Dict[str, float]
    error_rates: Dict[str, float]
    last_training_times: Dict[str, datetime]
    active_users: int
    prediction_volume_24h: int

# Feedback and continuous learning schemas

class PredictionFeedback(BaseModel):
    """Feedback on AI predictions"""
    prediction_id: str
    prediction_type: str  # priority, time, completion
    actual_outcome: Union[int, float, bool]
    feedback_type: str  # explicit, implicit
    user_satisfaction: Optional[int] = Field(None, ge=1, le=5)
    comments: Optional[str] = None
    context_at_outcome: Dict[str, Any] = Field(default_factory=dict)

class LearningMetrics(BaseModel):
    """Learning and adaptation metrics"""
    user_id: str
    model_improvement_rate: float
    prediction_accuracy_trend: List[float]
    feedback_incorporation_rate: float
    personalization_convergence: float = Field(..., ge=0, le=1)
    last_significant_update: datetime

class ModelPerformanceReport(BaseModel):
    """Comprehensive model performance report"""
    model_name: str
    model_version: str
    evaluation_period: Dict[str, datetime]
    accuracy_metrics: Dict[str, float]
    user_satisfaction_scores: Dict[str, float]
    prediction_volume: Dict[str, int]
    error_analysis: Dict[str, Any]
    improvement_recommendations: List[str]
    comparison_with_baseline: Dict[str, float]

# Configuration and deployment schemas

class ModelConfiguration(BaseModel):
    """AI model configuration"""
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    deployment_config: Dict[str, Any]
    feature_flags: Dict[str, bool] = Field(default_factory=dict)

class DeploymentStatus(BaseModel):
    """Model deployment status"""
    model_name: str
    version: str
    deployment_stage: str  # development, staging, production
    health_status: str  # healthy, degraded, unhealthy
    performance_metrics: Dict[str, float]
    last_health_check: datetime
    issues: List[str] = Field(default_factory=list)

# Advanced analytics schemas

class UserSegmentAnalysis(BaseModel):
    """User segment analysis for personalization"""
    segment_id: str
    segment_name: str
    user_count: int
    common_patterns: List[str]
    performance_metrics: Dict[str, float]
    optimization_opportunities: List[str]

class FeatureImportanceAnalysis(BaseModel):
    """Feature importance analysis for model interpretability"""
    model_type: str
    feature_importance_scores: Dict[str, float]
    feature_correlations: Dict[str, Dict[str, float]]
    actionable_insights: List[str]
    model_explanation: str

class ABTestResult(BaseModel):
    """A/B test results for model improvements"""
    test_id: str
    test_name: str
    control_group_performance: Dict[str, float]
    treatment_group_performance: Dict[str, float]
    statistical_significance: float
    recommendation: str
    confidence_interval: Dict[str, Tuple[float, float]]

# Export commonly used schemas
__all__ = [
    # Core schemas
    'TaskBase', 'TaskPrioritizationRequest', 'TaskPrioritizationResponse',
    'TimeEstimationRequest', 'TimeEstimationResponse',
    'UserInsightsRequest', 'UserInsightsResponse',
    'ModelTrainingRequest', 'ModelTrainingResponse',
    'PersonalizationMetricsResponse', 'BehaviorAnalysisResponse',
    
    # Advanced schemas
    'TaskSimilarityRequest', 'TaskSimilarityResponse',
    'SchedulingOptimizationRequest', 'SchedulingOptimizationResponse',
    'ProductivityPredictionRequest', 'ProductivityPredictionResponse',
    
    # Feedback and learning
    'PredictionFeedback', 'LearningMetrics', 'ModelPerformanceReport',
    
    # Analytics
    'UserSegmentAnalysis', 'FeatureImportanceAnalysis', 'ABTestResult',
    
    # Configuration
    'ModelConfiguration', 'DeploymentStatus', 'AIHealthMetrics'
]