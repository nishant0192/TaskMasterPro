# ai-service/app/models/database.py
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, DateTime, JSON, Float, Integer, Boolean, Text, ARRAY, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey, Index
import uuid

Base = declarative_base()

class UserBehaviorPattern(Base):
    __tablename__ = "user_behavior_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    pattern_type = Column(String(100), nullable=False)  # 'productivity_hours', 'task_preferences', etc.
    pattern_data = Column(JSONB, nullable=False)
    confidence_score = Column(Float, default=0.5)
    frequency_count = Column(Integer, default=1)
    last_observed = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_behavior_patterns_user_type', 'user_id', 'pattern_type'),
        Index('idx_behavior_patterns_confidence', 'confidence_score'),
        Index('idx_behavior_patterns_frequency', 'frequency_count'),
    )

class UserModelWeights(Base):
    __tablename__ = "user_model_weights"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # 'prioritization', 'scheduling', 'nlp', 'prediction'
    model_subtype = Column(String(50))  # 'classification', 'regression', 'embedding'
    weights = Column(LargeBinary, nullable=False)  # Compressed serialized model weights
    metadata = Column(JSONB)  # Model parameters, architecture, etc.
    accuracy_score = Column(Float)
    training_samples = Column(Integer, default=0)
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_model_weights_user_active', 'user_id', 'model_type', 'is_active'),
        Index('idx_model_weights_accuracy', 'accuracy_score'),
    )

class AITrainingSession(Base):
    __tablename__ = "ai_training_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    session_type = Column(String(50), nullable=False)  # 'initial', 'incremental', 'feedback'
    model_type = Column(String(50), nullable=False)
    training_data = Column(JSONB, nullable=False)
    feedback_data = Column(JSONB)
    performance_metrics = Column(JSONB)
    before_accuracy = Column(Float)
    after_accuracy = Column(Float)
    training_duration_ms = Column(Integer)
    sample_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_training_sessions_user_model', 'user_id', 'model_type'),
        Index('idx_training_sessions_type', 'session_type'),
        Index('idx_training_sessions_performance', 'after_accuracy'),
    )

class TaskEmbedding(Base):
    __tablename__ = "task_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    task_id = Column(UUID(as_uuid=True), nullable=False)
    embedding = Column(String)  # Will store as string, convert to vector in pgvector
    title_embedding = Column(String)
    description_embedding = Column(String)
    
    # Task metadata for feature learning
    category = Column(String(100))
    priority = Column(Integer)
    estimated_duration = Column(Integer)  # minutes
    actual_duration = Column(Integer)  # minutes
    completion_status = Column(String(20))  # 'completed', 'abandoned', 'delayed'
    completion_quality = Column(Float)  # 0-1 score
    
    # Context when task was created/completed
    created_hour = Column(Integer)  # 0-23
    created_day_of_week = Column(Integer)  # 0-6
    completed_hour = Column(Integer)
    user_energy_level = Column(Float)  # 0-1 estimated energy
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_task_embeddings_user', 'user_id'),
        Index('idx_task_embeddings_category', 'user_id', 'category'),
        Index('idx_task_embeddings_completion', 'completion_status', 'completion_quality'),
    )

class UserPreferenceEmbedding(Base):
    __tablename__ = "user_preference_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    preference_type = Column(String(50), nullable=False)  # 'task_style', 'time_preference', etc.
    preference_context = Column(String(100))  # 'work', 'personal', 'health', etc.
    embedding = Column(String, nullable=False)  # Vector embedding as string
    weight = Column(Float, default=1.0)  # Importance weight
    confidence = Column(Float, default=0.5)
    usage_count = Column(Integer, default=1)
    last_used = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_preference_embeddings_user_type', 'user_id', 'preference_type'),
        Index('idx_preference_embeddings_weight', 'weight'),
    )

class UserVocabulary(Base):
    __tablename__ = "user_vocabulary"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    term = Column(String(200), nullable=False)
    normalized_form = Column(String(200), nullable=False)
    term_type = Column(String(50), nullable=False)  # 'abbreviation', 'synonym', 'category', etc.
    context = Column(String(100))  # Additional context
    frequency = Column(Integer, default=1)
    confidence = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_vocabulary_user_term', 'user_id', 'term'),
        Index('idx_vocabulary_frequency', 'frequency'),
        Index('idx_vocabulary_type', 'term_type'),
    )

class UserProductivityMetrics(Base):
    __tablename__ = "user_productivity_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    metric_date = Column(DateTime, nullable=False)
    
    # Core productivity metrics
    tasks_created = Column(Integer, default=0)
    tasks_completed = Column(Integer, default=0)
    tasks_abandoned = Column(Integer, default=0)
    avg_completion_time = Column(Float)  # minutes
    productivity_score = Column(Float)  # 0-1 composite score
    
    # Time-based metrics
    peak_productivity_hours = Column(ARRAY(Integer))  # Array of hours (0-23)
    total_focus_time = Column(Integer)  # minutes
    break_frequency = Column(Float)  # breaks per hour
    
    # Quality metrics
    deadline_success_rate = Column(Float)  # 0-1
    task_quality_score = Column(Float)  # 0-1
    priority_accuracy = Column(Float)  # How accurate user's initial priority was
    
    # Behavioral patterns
    procrastination_score = Column(Float)  # 0-1, higher = more procrastination
    consistency_score = Column(Float)  # 0-1, how consistent user is
    stress_indicators = Column(JSONB)  # Various stress-related metrics
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_productivity_metrics_user_date', 'user_id', 'metric_date'),
        Index('idx_productivity_metrics_score', 'productivity_score'),
    )

class AIPrediction(Base):
    __tablename__ = "ai_predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    prediction_type = Column(String(50), nullable=False)  # 'completion_time', 'priority', 'success_probability'
    target_task_id = Column(UUID(as_uuid=True))
    
    # Prediction details
    predicted_value = Column(Float, nullable=False)
    predicted_confidence = Column(Float, nullable=False)
    prediction_context = Column(JSONB)  # Context used to make prediction
    
    # Actual outcome (filled in later)
    actual_value = Column(Float)
    prediction_accuracy = Column(Float)  # How accurate was this prediction
    feedback_provided = Column(Boolean, default=False)
    user_satisfaction = Column(Integer)  # 1-5 rating of prediction usefulness
    
    created_at = Column(DateTime, default=datetime.utcnow)
    outcome_recorded_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_predictions_user_type', 'user_id', 'prediction_type'),
        Index('idx_predictions_accuracy', 'prediction_accuracy'),
        Index('idx_predictions_satisfaction', 'user_satisfaction'),
    )

class FeatureImportance(Base):
    __tablename__ = "feature_importance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)
    feature_name = Column(String(100), nullable=False)
    importance_score = Column(Float, nullable=False)
    model_version = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_feature_importance_user_model', 'user_id', 'model_type'),
        Index('idx_feature_importance_score', 'importance_score'),
    )

class UserAIPreferences(Base):
    __tablename__ = "user_ai_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, unique=True)
    
    # Learning preferences
    learning_rate = Column(Float, default=0.1)  # How quickly to adapt
    feedback_sensitivity = Column(Float, default=0.5)  # How much to weight user feedback
    privacy_level = Column(String(20), default='balanced')  # 'minimal', 'balanced', 'full'
    
    # Feature preferences
    enable_predictive_scheduling = Column(Boolean, default=True)
    enable_priority_suggestions = Column(Boolean, default=True)
    enable_time_estimates = Column(Boolean, default=True)
    enable_pattern_recognition = Column(Boolean, default=True)
    enable_proactive_insights = Column(Boolean, default=True)
    
    # Notification preferences for AI
    insight_frequency = Column(String(20), default='weekly')  # 'daily', 'weekly', 'monthly'
    prediction_confidence_threshold = Column(Float, default=0.7)
    suggestion_aggressiveness = Column(String(20), default='moderate')  # 'conservative', 'moderate', 'aggressive'
    
    # Model preferences
    prioritize_accuracy_over_speed = Column(Boolean, default=False)
    model_update_frequency = Column(String(20), default='weekly')  # 'daily', 'weekly', 'monthly'
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelPerformanceHistory(Base):
    __tablename__ = "model_performance_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)
    model_version = Column(Integer, nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    mean_absolute_error = Column(Float)
    user_satisfaction_avg = Column(Float)
    
    # Usage statistics
    predictions_made = Column(Integer, default=0)
    feedback_received = Column(Integer, default=0)
    positive_feedback_ratio = Column(Float)
    
    # Context
    evaluation_date = Column(DateTime, nullable=False)
    sample_size = Column(Integer)
    test_conditions = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_model_performance_user_model', 'user_id', 'model_type'),
        Index('idx_model_performance_date', 'evaluation_date'),
        Index('idx_model_performance_accuracy', 'accuracy'),
    )

class UserInteractionLog(Base):
    __tablename__ = "user_interaction_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    interaction_type = Column(String(50), nullable=False)  # 'task_created', 'priority_changed', etc.
    
    # Context of interaction
    task_id = Column(UUID(as_uuid=True))
    before_state = Column(JSONB)
    after_state = Column(JSONB)
    user_context = Column(JSONB)  # Time, location, calendar state, etc.
    
    # AI involvement
    ai_suggestion_provided = Column(Boolean, default=False)
    ai_prediction_made = Column(Boolean, default=False)
    suggestion_accepted = Column(Boolean)
    prediction_accuracy = Column(Float)
    
    # Timing
    interaction_timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(UUID(as_uuid=True))  # Group related interactions
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_interaction_logs_user_type', 'user_id', 'interaction_type'),
        Index('idx_interaction_logs_timestamp', 'interaction_timestamp'),
        Index('idx_interaction_logs_session', 'session_id'),
        Index('idx_interaction_logs_ai_involvement', 'ai_suggestion_provided', 'suggestion_accepted'),
    )

class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String(64), nullable=False, unique=True)  # SHA-256 hash
    content_type = Column(String(50), nullable=False)  # 'task_title', 'task_description', 'user_query'
    embedding = Column(String, nullable=False)  # Vector embedding as string
    model_version = Column(String(50), nullable=False)
    usage_count = Column(Integer, default=1)
    last_used = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_embedding_cache_hash', 'content_hash'),
        Index('idx_embedding_cache_usage', 'usage_count', 'last_used'),
    )

class AIModelRegistry(Base):
    __tablename__ = "ai_model_registry"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'global', 'user_specific'
    
    # Model metadata
    description = Column(Text)
    architecture = Column(JSONB)  # Model architecture details
    hyperparameters = Column(JSONB)
    training_config = Column(JSONB)
    
    # Performance benchmarks
    benchmark_scores = Column(JSONB)
    validation_metrics = Column(JSONB)
    
    # Deployment info
    is_active = Column(Boolean, default=False)
    deployment_date = Column(DateTime)
    deprecated_date = Column(DateTime)
    
    # File references
    model_file_path = Column(String(500))
    checkpoint_path = Column(String(500))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_model_registry_active', 'model_name', 'is_active'),
        Index('idx_model_registry_type', 'model_type'),
    )