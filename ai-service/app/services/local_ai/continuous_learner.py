# ai-service/app/services/local_ai/continuous_learner.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

from app.core.database import get_async_session
from app.models.database import (
    UserBehaviorPattern, UserInteractionLog, AIPrediction,
    AITrainingSession, ModelPerformanceHistory, UserAIPreferences,
    FeatureImportance, UserProductivityMetrics
)
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class LearningTask:
    """Represents a learning task for model update"""
    user_id: str
    model_type: str
    priority: float
    created_at: datetime
    training_data_size: int
    reason: str

@dataclass
class ModelUpdateResult:
    """Result of a model update"""
    user_id: str
    model_type: str
    success: bool
    previous_accuracy: float
    new_accuracy: float
    training_samples: int
    training_duration_ms: int
    improvements: Dict[str, float]

class ContinuousLearner:
    """Manages continuous learning and model updates across all users"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.is_initialized = False
        
        # Learning queue
        self.learning_queue: List[LearningTask] = []
        self.active_learning_tasks: Set[str] = set()
        
        # Learning configuration
        self.batch_size = 10
        self.min_feedback_threshold = 10
        self.accuracy_threshold = 0.7
        self.retraining_interval_hours = 24
        
        # Performance tracking
        self.model_performance_cache: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self):
        """Initialize the continuous learner"""
        try:
            logger.info("Initializing Continuous Learner...")
            
            # Start background learning task
            asyncio.create_task(self._learning_worker())
            
            # Start performance monitoring task
            asyncio.create_task(self._performance_monitor())
            
            # Start scheduled retraining task
            asyncio.create_task(self._scheduled_retraining())
            
            self.is_initialized = True
            logger.info("