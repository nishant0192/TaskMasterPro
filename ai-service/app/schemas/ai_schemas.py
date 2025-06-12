# ai-service/app/schemas/ai_schemas.py
"""
AI-specific Pydantic schemas for requests and responses
Production-ready schemas with proper validation and error handling
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Tuple
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

class TimePeriod(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

class JobType(str, Enum):
    PREDICTION_UPDATE = "PREDICTION_UPDATE"
    MODEL_TRAINING = "MODEL_TRAINING"
    INSIGHT_GENERATION = "INSIGHT_GENERATION"
    BATCH_PRIORITIZATION = "BATCH_PRIORITIZATION"

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

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CalendarEvent(BaseModel):
    id: str
    title: str
    start_time: datetime
    end_time: datetime
    is_busy: bool = True
    location: Optional[str] = None

class UserPreferences(BaseModel):
    work_hours_start: int = Field(9, ge=0, le=23)
    work_hours_end: int = Field(17, ge=0, le=23)
    work_days: List[int] = Field([1, 2, 3, 4, 5])
    timezone: str = "UTC"
    break_duration: int = Field(15, ge=5, le=60)
    max_work_hours_per_day: int = Field(8, ge=1, le=16)
    productivity_hours: List[int] = Field([9, 10, 14, 15])

class TimeRange(BaseModel):
    start_date: date
    end_date: date

# Core AI Request/Response Schemas
class TaskPrioritizationRequest(BaseModel):
    tasks: List[TaskBase]
    context: Optional[Dict[str, Any]] = {}
    user_preferences: Optional[UserPreferences] = None

class PrioritizedTask(BaseModel):
    """Task with AI-generated priority scores"""
    id: str
    title: str
    description: Optional[str] = None
    ai_priority_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    ai_explanation: str
    suggested_order: int
    estimated_completion_time: Optional[int] = None

class TaskPrioritizationResponse(BaseModel):
    prioritized_tasks: List[PrioritizedTask]
    explanations: List[str]
    confidence_scores: Dict[str, float]
    processing_time_ms: int
    model_version: str

class TimeEstimationRequest(BaseModel):
    task: TaskBase
    user_context: Optional[Dict[str, Any]] = {}
    historical_data: Optional[List[Dict[str, Any]]] = []

class TimeEstimationResponse(BaseModel):
    estimated_duration: int
    confidence_interval: Dict[str, int]
    factors_considered: List[str]
    similar_tasks: List[str] = []

class UserInsightsRequest(BaseModel):
    user_id: str
    time_period: TimePeriod
    include_predictions: bool = True

class UserInsight(BaseModel):
    insight_type: str
    title: str
    description: str
    confidence: float = Field(..., ge=0, le=1)
    actionable: bool = True
    supporting_data: Dict[str, Any] = {}

class UserInsightsResponse(BaseModel):
    insights: List[UserInsight]
    productivity_score: float = Field(..., ge=0, le=100)
    recommendations: List[str]
    trend_analysis: Dict[str, Any]

class ModelTrainingRequest(BaseModel):
    user_id: str
    training_data: List[Dict[str, Any]]
    model_type: str
    force_retrain: bool = False

class ModelTrainingResponse(BaseModel):
    success: bool
    model_version: str
    performance_metrics: Dict[str, float]
    training_time_seconds: float
    samples_used: int

# Prediction and Analysis Schemas
class PredictionResult(BaseModel):
    """Generic prediction result"""
    prediction: float
    confidence: float
    explanation: str
    model_version: str
    metadata: Dict[str, Any] = {}

class PersonalizationMetrics(BaseModel):
    """User personalization metrics"""
    user_id: str
    personalization_level: float = Field(..., ge=0, le=1)
    model_accuracy: float = Field(..., ge=0, le=1)
    training_samples: int
    last_updated: datetime
    adaptation_rate: float

class PersonalizationMetricsResponse(BaseModel):
    metrics: PersonalizationMetrics
    improvement_suggestions: List[str]
    data_quality_score: float = Field(..., ge=0, le=1)

class BehaviorPattern(BaseModel):
    pattern_type: str
    description: str
    frequency: float = Field(..., ge=0, le=1)
    impact_score: float = Field(..., ge=0, le=1)
    detected_at: datetime

class BehaviorAnalysisResponse(BaseModel):
    patterns: List[BehaviorPattern]
    productivity_insights: List[str]
    optimization_opportunities: List[str]
    behavioral_score: float = Field(..., ge=0, le=100)

# Advanced Analytics Schemas
class TaskSimilarityRequest(BaseModel):
    target_task: TaskBase
    candidate_tasks: List[TaskBase]
    similarity_threshold: float = Field(0.7, ge=0, le=1)

class SimilarTask(BaseModel):
    task: TaskBase
    similarity_score: float = Field(..., ge=0, le=1)
    matching_features: List[str]

class TaskSimilarityResponse(BaseModel):
    similar_tasks: List[SimilarTask]
    similarity_matrix: Optional[List[List[float]]] = None
    processing_time_ms: int

class SchedulingOptimizationRequest(BaseModel):
    tasks: List[TaskBase]
    calendar_events: List[CalendarEvent]
    user_preferences: UserPreferences
    optimization_criteria: List[str] = ["productivity", "deadlines", "energy"]

class OptimizedSchedule(BaseModel):
    task_id: str
    suggested_start_time: datetime
    suggested_duration: int
    productivity_score: float = Field(..., ge=0, le=1)
    reasoning: str

class SchedulingOptimizationResponse(BaseModel):
    optimized_schedule: List[OptimizedSchedule]
    productivity_gain: float
    schedule_feasibility: float = Field(..., ge=0, le=1)
    alternative_schedules: List[List[OptimizedSchedule]] = []

class ProductivityPredictionRequest(BaseModel):
    user_id: str
    prediction_horizon_days: int = Field(7, ge=1, le=30)
    context_factors: Dict[str, Any] = {}

class ProductivityForecast(BaseModel):
    date: date
    predicted_productivity: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    key_factors: List[str]

class ProductivityPredictionResponse(BaseModel):
    forecasts: List[ProductivityForecast]
    trend_direction: str
    recommendations: List[str]
    model_accuracy: float = Field(..., ge=0, le=1)

# Feedback and Learning Schemas
class PredictionFeedback(BaseModel):
    prediction_id: str
    actual_outcome: Union[float, str, bool]
    user_satisfaction: int = Field(..., ge=1, le=5)
    feedback_notes: Optional[str] = None
    context_updates: Dict[str, Any] = {}

class LearningMetrics(BaseModel):
    model_type: str
    accuracy_trend: List[float]
    sample_count: int
    learning_rate: float
    adaptation_speed: str
    overfitting_risk: float = Field(..., ge=0, le=1)

class ModelPerformanceReport(BaseModel):
    model_name: str
    model_version: str
    performance_metrics: Dict[str, float]
    benchmark_comparison: Dict[str, float]
    recommendations: List[str]
    comparison_with_baseline: Dict[str, float]

# Configuration and deployment schemas
class ModelConfiguration(BaseModel):
    """AI model configuration"""
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    deployment_config: Dict[str, Any]
    feature_flags: Dict[str, bool] = {}

class DeploymentStatus(BaseModel):
    """Model deployment status"""
    model_name: str
    version: str
    deployment_stage: str
    health_status: str
    performance_metrics: Dict[str, float]
    last_health_check: datetime
    issues: List[str] = []

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

# Health and Monitoring Schemas
class AIHealthMetrics(BaseModel):
    service_status: str
    model_health: Dict[str, str]
    response_times: Dict[str, float]
    error_rates: Dict[str, float]
    resource_usage: Dict[str, float]
    last_updated: datetime

# Missing schemas that are imported by other modules
class AdaptationRecommendation(BaseModel):
    """Model adaptation recommendation"""
    user_id: str
    adaptations: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0, le=1)
    expected_improvement: float = Field(..., ge=0, le=1)
    implementation_date: datetime

# Export all schemas
__all__ = [
    # Core schemas
    'TaskBase', 'TaskPrioritizationRequest', 'TaskPrioritizationResponse', 'PrioritizedTask',
    'TimeEstimationRequest', 'TimeEstimationResponse',
    'UserInsightsRequest', 'UserInsightsResponse', 'UserInsight',
    'ModelTrainingRequest', 'ModelTrainingResponse',
    'PersonalizationMetrics', 'PersonalizationMetricsResponse',
    'BehaviorPattern', 'BehaviorAnalysisResponse',
    'PredictionResult',
    
    # Advanced schemas
    'TaskSimilarityRequest', 'TaskSimilarityResponse', 'SimilarTask',
    'SchedulingOptimizationRequest', 'SchedulingOptimizationResponse', 'OptimizedSchedule',
    'ProductivityPredictionRequest', 'ProductivityPredictionResponse', 'ProductivityForecast',
    
    # Feedback and learning
    'PredictionFeedback', 'LearningMetrics', 'ModelPerformanceReport',
    
    # Analytics
    'UserSegmentAnalysis', 'FeatureImportanceAnalysis', 'ABTestResult',
    
    # Configuration
    'ModelConfiguration', 'DeploymentStatus',
    
    # Health and monitoring
    'AIHealthMetrics', 'AdaptationRecommendation',
    
    # Base classes
    'UserPreferences', 'CalendarEvent', 'TimeRange', 'TaskStatus', 'TaskPriority', 'TimePeriod', 'JobType'
]