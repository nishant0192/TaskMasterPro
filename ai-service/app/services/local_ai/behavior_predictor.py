# ai-service/app/services/local_ai/behavior_predictor.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

from app.core.database import get_async_session
from app.models.database import (
    UserBehaviorPattern, TaskEmbedding, UserProductivityMetrics,
    AIPrediction, UserInteractionLog, FeatureImportance
)
from app.models.schemas import TaskBase
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class TaskSuccessPrediction:
    """Prediction for task completion success"""
    task_id: str
    estimated_completion_time: int  # minutes
    on_time_probability: float
    predicted_completion_date: datetime
    confidence_interval: Dict[str, int]
    recommendations: List[str]

@dataclass
class BehavioralInsight:
    """Behavioral insight about user patterns"""
    insight_type: str
    title: str
    description: str
    confidence: float
    impact_score: float
    recommendations: List[str]
    supporting_data: Dict[str, Any]

@dataclass
class ProcrastinationRisk:
    """Risk assessment for task procrastination"""
    task_id: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    risk_score: float
    risk_factors: List[str]
    mitigation_strategies: List[str]

class BehaviorPredictor:
    """AI-powered behavioral analysis and prediction service"""
    
    def __init__(self, user_id: str, model_manager: ModelManager):
        self.user_id = user_id
        self.model_manager = model_manager
        self.is_initialized = False
        
        # Prediction models
        self.completion_model = None
        self.procrastination_model = None
        self.duration_model = None
        self.success_model = None
        
        # Behavior analysis
        self.behavior_patterns = {}
        self.productivity_trends = {}
        self.risk_profiles = {}
        
        # Feature importance tracking
        self.feature_importance = {}
        
    async def initialize(self):
        """Initialize the behavior predictor"""
        try:
            logger.info(f"Initializing behavior predictor for user {self.user_id}")
            
            # Load or train prediction models
            await self._load_or_train_models()
            
            # Load behavior patterns
            await self._load_behavior_patterns()
            
            # Load productivity trends
            await self._load_productivity_trends()
            
            # Load risk profiles
            await self._load_risk_profiles()
            
            self.is_initialized = True
            logger.info(f"Behavior predictor initialized for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize behavior predictor: {e}")
            raise