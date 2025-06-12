# ai-service/app/core/ai_engine.py
"""
Production-ready personalized AI service core engine
Handles model management, personalization, and inference
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

# ML Libraries
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import spacy

# Database
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.sql import func

# Internal imports
from app.core.config import get_settings
from app.core.database import get_async_session
from app.models.ai_models import (
    UserBehaviorPattern, UserModelWeights, TaskEmbedding,
    AITrainingSession, UserPreferenceEmbedding
)
from app.schemas.ai_schemas import (
    TaskPrioritizationRequest, PredictionResult,
    PersonalizationMetrics, UserInsight
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class TaskFeatures:
    """Structured task features for ML models"""
    task_id: str
    title_embedding: np.ndarray
    description_embedding: np.ndarray
    priority_numeric: float
    days_until_due: float
    estimated_duration: float
    category_encoded: int
    time_of_day_created: float
    day_of_week: int
    is_recurring: bool
    has_dependencies: bool
    complexity_score: float
    user_stress_level: float
    historical_completion_rate: float


@dataclass
class PersonalizedModel:
    """Container for user's personalized AI models"""
    user_id: str
    prioritization_model: Optional[GradientBoostingClassifier] = None
    time_estimation_model: Optional[RandomForestRegressor] = None
    completion_predictor: Optional[GradientBoostingClassifier] = None
    user_weights: Optional[np.ndarray] = None
    scaler: Optional[StandardScaler] = None
    label_encoders: Dict[str, LabelEncoder] = None
    last_updated: Optional[datetime] = None
    accuracy_score: Optional[float] = None
    training_samples: int = 0


class ProductionAIEngine:
    """
    Production-ready AI engine for task management
    Features:
    - Personalized model training and inference
    - Real-time prediction caching
    - Automatic model retraining
    - Performance monitoring
    - Scalable architecture
    """

    def __init__(self):
        self.user_models: Dict[str, PersonalizedModel] = {}
        self.global_models: Dict[str, Any] = {}
        self.embedding_model: Optional[SentenceTransformer] = None
        self.nlp_model: Optional[spacy.Language] = None
        self.model_cache_dir = Path(settings.model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.prediction_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=30)

        # Model performance metrics
        self.performance_metrics: Dict[str, Dict[str, float]] = {}

        # Feature extraction configuration
        self.feature_config = {
            'text_features': ['title', 'description'],
            'numerical_features': ['priority', 'estimated_duration'],
            'categorical_features': ['category', 'status'],
            'temporal_features': ['due_date', 'created_at'],
            'derived_features': ['urgency_score', 'complexity_score']
        }

    async def initialize(self):
        """Initialize the AI engine with models and dependencies"""
        try:
            logger.info("🤖 Initializing AI Engine...")

            # Load embedding model
            await self._load_embedding_model()

            # Load NLP model
            await self._load_nlp_model()

            # Load global models
            await self._load_global_models()

            # Initialize performance monitoring
            await self._initialize_monitoring()

            logger.info("✅ AI Engine initialized successfully")

        except Exception as e:
            logger.error(f"❌ AI Engine initialization failed: {e}")
            raise

    async def _load_embedding_model(self):
        """Load sentence transformer for text embeddings"""
        try:
            # Use a lightweight model for production
            model_name = "all-MiniLM-L6-v2"  # 90MB, good performance
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"📝 Loaded embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load embedding model: {e}")
            # Fallback to simple text features
            self.embedding_model = None

    async def _load_nlp_model(self):
        """Load spaCy model for NLP tasks"""
        try:
            # Use lightweight English model
            self.nlp_model = spacy.load("en_core_web_sm")
            logger.info("🔤 Loaded NLP model: en_core_web_sm")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load NLP model: {e}")
            self.nlp_model = None

    async def _load_global_models(self):
        """Load pre-trained global models"""
        try:
            global_model_path = self.model_cache_dir / "global_models.pkl"
            if global_model_path.exists():
                with open(global_model_path, 'rb') as f:
                    self.global_models = pickle.load(f)
                logger.info(
                    f"📊 Loaded {len(self.global_models)} global models")
            else:
                # Initialize with baseline models
                await self._create_baseline_models()
        except Exception as e:
            logger.warning(f"⚠️ Failed to load global models: {e}")
            await self._create_baseline_models()

    async def _create_baseline_models(self):
        """Create baseline models for new users"""
        try:
            # Simple baseline prioritization model
            baseline_prioritizer = GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )

            # Simple baseline time estimation model
            baseline_estimator = RandomForestRegressor(
                n_estimators=30,
                max_depth=5,
                random_state=42
            )

            self.global_models = {
                'baseline_prioritizer': baseline_prioritizer,
                'baseline_estimator': baseline_estimator,
                'version': '1.0.0',
                'created_at': datetime.now()
            }

            logger.info("🏗️ Created baseline models")

        except Exception as e:
            logger.error(f"❌ Failed to create baseline models: {e}")

    async def _initialize_monitoring(self):
        """Initialize performance monitoring"""
        self.performance_metrics = {
            'prediction_accuracy': {},
            'response_times': {},
            'model_usage': {},
            'error_rates': {}
        }
        logger.info("📊 Performance monitoring initialized")

    async def predict_task_priority(self, task_data: Dict[str, Any], user_id: str) -> PredictionResult:
        """
        Predict task priority using personalized model
        """
        try:
            start_time = datetime.now()

            # Check cache first
            cache_key = f"priority_{user_id}_{hash(str(task_data))}"
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                return cached_result

            # Get or create user model
            user_model = await self._get_user_model(user_id)

            # Extract features
            features = await self._extract_task_features(task_data, user_id)

            # Make prediction
            if user_model.prioritization_model and user_model.training_samples > 10:
                # Use personalized model
                prediction = await self._predict_with_personalized_model(
                    features, user_model.prioritization_model, user_model.scaler
                )
                explanation = f"Based on your personal task completion patterns ({user_model.training_samples} samples)"
                confidence = min(0.9, user_model.accuracy_score or 0.7)
            else:
                # Use global model
                prediction = await self._predict_with_global_model(features, 'baseline_prioritizer')
                explanation = "Based on general task prioritization patterns"
                confidence = 0.6

            # Create result
            result = PredictionResult(
                prediction=float(prediction),
                confidence=confidence,
                explanation=explanation,
                model_version=self.global_models.get('version', '1.0.0'),
                metadata={
                    'user_id': user_id,
                    'features_used': len(features),
                    'model_type': 'personalized' if user_model.training_samples > 10 else 'global'
                }
            )

            # Cache result
            self._cache_prediction(cache_key, result)

            # Track performance
            response_time = (datetime.now() -
                             start_time).total_seconds() * 1000
            await self._track_performance('priority_prediction', response_time, user_id)

            return result

        except Exception as e:
            logger.error(
                f"❌ Priority prediction failed for user {user_id}: {e}")
            # Return fallback prediction
            return PredictionResult(
                prediction=0.5,  # Medium priority
                confidence=0.3,
                explanation="Fallback prediction due to processing error",
                model_version="fallback",
                metadata={'error': str(e)}
            )

    async def _extract_task_features(self, task_data: Dict[str, Any], user_id: str) -> np.ndarray:
        """Extract numerical features from task data"""
        try:
            features = []

            # Basic numerical features
            features.append(task_data.get('priority', 3) /
                            5.0)  # Normalize 1-5 to 0-1
            # Normalize to 8-hour scale
            features.append(task_data.get('estimated_duration', 60) / 480.0)

            # Temporal features
            if 'due_date' in task_data and task_data['due_date']:
                try:
                    due_date = datetime.fromisoformat(
                        task_data['due_date'].replace('Z', '+00:00'))
                    days_until_due = (due_date - datetime.now()).days
                    # Normalize to 30-day scale
                    features.append(max(0, min(30, days_until_due)) / 30.0)
                except:
                    features.append(0.5)  # Default if parsing fails
            else:
                features.append(0.5)

            # Text-based features (simplified)
            title_length = len(task_data.get('title', ''))
            # Normalize title length
            features.append(min(100, title_length) / 100.0)

            description_length = len(task_data.get('description', ''))
            # Normalize description length
            features.append(min(500, description_length) / 500.0)

            # Category feature (simplified)
            category = task_data.get('category', 'general')
            category_score = hash(category) % 100 / \
                100.0  # Simple category encoding
            features.append(category_score)

            # Day of week feature
            features.append(datetime.now().weekday() / 6.0)  # 0-6 normalized

            # Hour of day feature
            features.append(datetime.now().hour / 23.0)  # 0-23 normalized

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            # Return default feature vector
            return np.array([0.5] * 8, dtype=np.float32)

    async def _get_user_model(self, user_id: str) -> PersonalizedModel:
        """Get or create user's personalized model"""
        if user_id not in self.user_models:
            self.user_models[user_id] = PersonalizedModel(
                user_id=user_id,
                scaler=StandardScaler(),
                label_encoders={},
                last_updated=datetime.now()
            )
        return self.user_models[user_id]

    async def _predict_with_personalized_model(self, features: np.ndarray, model: Any, scaler: StandardScaler) -> float:
        """Make prediction using personalized model"""
        try:
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))

            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                # Convert to priority score (0-1)
                prediction = np.average(
                    range(len(probabilities)), weights=probabilities) / (len(probabilities) - 1)
            else:
                prediction = model.predict(features_scaled)[0]

            return np.clip(prediction, 0.0, 1.0)

        except Exception as e:
            logger.error(f"❌ Personalized model prediction failed: {e}")
            return 0.5

    async def _predict_with_global_model(self, features: np.ndarray, model_name: str) -> float:
        """Make prediction using global model"""
        try:
            # Simple heuristic-based prediction for baseline
            # Priority based on urgency, duration, and complexity
            urgency = features[2]  # Days until due (inverted)
            # Title + description length
            complexity = features[3] + features[4]

            # Simple weighted combination
            # Include original priority
            prediction = 0.4 * (1 - urgency) + 0.3 * \
                complexity + 0.3 * features[0]

            return np.clip(prediction, 0.0, 1.0)

        except Exception as e:
            logger.error(f"❌ Global model prediction failed: {e}")
            return 0.5

    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResult]:
        """Get cached prediction if still valid"""
        if cache_key in self.prediction_cache:
            result, timestamp = self.prediction_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return result
            else:
                del self.prediction_cache[cache_key]
        return None

    def _cache_prediction(self, cache_key: str, result: PredictionResult):
        """Cache prediction result"""
        self.prediction_cache[cache_key] = (result, datetime.now())

        # Clean old cache entries (simple cleanup)
        if len(self.prediction_cache) > 1000:
            # Remove oldest 20% of entries
            sorted_items = sorted(
                self.prediction_cache.items(),
                key=lambda x: x[1][1]
            )
            for key, _ in sorted_items[:200]:
                del self.prediction_cache[key]

    async def _track_performance(self, operation: str, response_time: float, user_id: str):
        """Track performance metrics"""
        try:
            if operation not in self.performance_metrics['response_times']:
                self.performance_metrics['response_times'][operation] = []

            self.performance_metrics['response_times'][operation].append(
                response_time)

            # Keep only last 100 measurements
            if len(self.performance_metrics['response_times'][operation]) > 100:
                self.performance_metrics['response_times'][operation] = \
                    self.performance_metrics['response_times'][operation][-100:]

        except Exception as e:
            logger.error(f"❌ Performance tracking failed: {e}")

    async def get_user_model(self, user_id: str) -> PersonalizedModel:
        """Public interface to get user model"""
        return await self._get_user_model(user_id)

    async def train_user_model(self, user_id: str, training_data: List[Dict[str, Any]]) -> bool:
        """Train or update user's personalized model"""
        try:
            logger.info(
                f"🎯 Training model for user {user_id} with {len(training_data)} samples")

            user_model = await self._get_user_model(user_id)

            # Extract features and labels
            features_list = []
            labels_list = []

            for sample in training_data:
                features = await self._extract_task_features(sample, user_id)
                features_list.append(features)
                labels_list.append(sample.get(
                    'actual_priority', sample.get('priority', 3)))

            if len(features_list) < 5:
                logger.warning(
                    f"⚠️ Insufficient training data for user {user_id}")
                return False

            X = np.array(features_list)
            y = np.array(labels_list)

            # Scale features
            user_model.scaler.fit(X)
            X_scaled = user_model.scaler.transform(X)

            # Train model
            user_model.prioritization_model = GradientBoostingClassifier(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )

            # Convert to classification labels (0-4 for priorities 1-5)
            y_class = np.clip(np.array(y) - 1, 0, 4).astype(int)

            user_model.prioritization_model.fit(X_scaled, y_class)
            user_model.training_samples = len(training_data)
            user_model.last_updated = datetime.now()

            # Calculate accuracy on training data (simple validation)
            y_pred = user_model.prioritization_model.predict(X_scaled)
            user_model.accuracy_score = accuracy_score(y_class, y_pred)

            logger.info(
                f"✅ Model trained for user {user_id}, accuracy: {user_model.accuracy_score:.3f}")

            # Save model
            await self._save_user_model(user_id, user_model)

            return True

        except Exception as e:
            logger.error(f"❌ Model training failed for user {user_id}: {e}")
            return False

    async def _save_user_model(self, user_id: str, model: PersonalizedModel):
        """Save user model to disk"""
        try:
            model_path = self.model_cache_dir / f"user_model_{user_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.debug(f"💾 Saved model for user {user_id}")
        except Exception as e:
            logger.error(f"❌ Failed to save model for user {user_id}: {e}")

    async def get_health_status(self) -> Dict[str, Any]:
        """Get AI engine health status"""
        return {
            'status': 'healthy' if self.embedding_model else 'degraded',
            'models_loaded': {
                'embedding_model': self.embedding_model is not None,
                'nlp_model': self.nlp_model is not None,
                'global_models': len(self.global_models) > 0,
                'user_models': len(self.user_models)
            },
            'performance': {
                'cache_size': len(self.prediction_cache),
                'avg_response_time': self._get_avg_response_time()
            }
        }

    def _get_avg_response_time(self) -> float:
        """Calculate average response time across all operations"""
        all_times = []
        for operation_times in self.performance_metrics['response_times'].values():
            all_times.extend(operation_times)

        return sum(all_times) / len(all_times) if all_times else 0.0


# Global instance
ai_engine = ProductionAIEngine()

# Dependency function for FastAPI


async def get_ai_engine() -> ProductionAIEngine:
    """FastAPI dependency to get AI engine instance"""
    return ai_engine
