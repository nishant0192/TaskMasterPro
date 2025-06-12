# ai-service/app/core/ai_engine.py
"""
Production-ready personalized AI service core engine
Handles model management, personalization, and inference
"""

import asyncio
import logging
from turtle import clone
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
    Production-ready AI engine for personalized task management
    Handles model training, inference, and continuous learning
    """
    
    def __init__(self):
        self.embedding_model = None
        self.nlp_model = None
        self.global_models = {}
        self.user_models: Dict[str, PersonalizedModel] = {}
        self.feature_extractors = {}
        self.model_cache_dir = Path(settings.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = {}
        self.training_queue = asyncio.Queue()
        self.inference_cache = {}
        
    async def initialize(self):
        """Initialize AI engine with pre-trained models"""
        try:
            logger.info("🚀 Initializing Production AI Engine...")
            
            # Load base models
            await self._load_embedding_model()
            await self._load_nlp_model()
            await self._load_global_models()
            
            # Start background training worker
            asyncio.create_task(self._training_worker())
            
            logger.info("✅ AI Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI Engine: {e}")
            raise

    async def _load_embedding_model(self):
        """Load sentence transformer model for embeddings"""
        try:
            model_name = settings.embedding_model_name
            logger.info(f"Loading embedding model: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name)
            
            # Warm up the model
            _ = self.embedding_model.encode(["test sentence"])
            
            logger.info(f"✅ Embedding model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    async def _load_nlp_model(self):
        """Load spaCy model for NLP processing"""
        try:
            model_name = settings.spacy_model_name
            logger.info(f"Loading NLP model: {model_name}")
            
            self.nlp_model = spacy.load(model_name)
            
            logger.info(f"✅ NLP model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load NLP model: {e}")
            raise

    async def _load_global_models(self):
        """Load or train global baseline models"""
        try:
            models_to_load = [
                'prioritization_global.joblib',
                'time_estimation_global.joblib',
                'completion_prediction_global.joblib'
            ]
            
            for model_file in models_to_load:
                model_path = self.model_cache_dir / model_file
                
                if model_path.exists():
                    model = joblib.load(model_path)
                    model_name = model_file.replace('_global.joblib', '')
                    self.global_models[model_name] = model
                    logger.info(f"✅ Loaded global model: {model_name}")
                else:
                    logger.info(f"Training new global model: {model_file}")
                    await self._train_global_model(model_file)
                    
        except Exception as e:
            logger.error(f"Failed to load global models: {e}")
            raise

    async def get_user_model(self, user_id: str) -> PersonalizedModel:
        """Get or create personalized model for user"""
        if user_id not in self.user_models:
            await self._load_user_model(user_id)
            
        return self.user_models[user_id]

    async def _load_user_model(self, user_id: str):
        """Load user's personalized model from database"""
        try:
            async with get_async_session() as db:
                # Check if user has existing model weights
                query = select(UserModelWeights).where(
                    UserModelWeights.user_id == user_id
                )
                result = await db.execute(query)
                model_records = result.scalars().all()
                
                user_model = PersonalizedModel(user_id=user_id)
                
                if model_records:
                    # Load existing personalized models
                    for record in model_records:
                        model_data = pickle.loads(record.weights)
                        
                        if record.model_type == 'prioritization':
                            user_model.prioritization_model = model_data['model']
                            user_model.scaler = model_data.get('scaler')
                            user_model.label_encoders = model_data.get('encoders', {})
                        elif record.model_type == 'time_estimation':
                            user_model.time_estimation_model = model_data['model']
                        elif record.model_type == 'completion_prediction':
                            user_model.completion_predictor = model_data['model']
                            
                        user_model.accuracy_score = record.accuracy_score
                        user_model.last_updated = record.created_at
                        
                    logger.info(f"✅ Loaded personalized models for user {user_id}")
                else:
                    # Initialize with global models for new users
                    await self._initialize_user_with_global_models(user_model)
                    logger.info(f"🆕 Initialized new user model for {user_id}")
                
                self.user_models[user_id] = user_model
                
        except Exception as e:
            logger.error(f"Failed to load user model for {user_id}: {e}")
            # Fallback to global models
            self.user_models[user_id] = PersonalizedModel(user_id=user_id)

    async def _initialize_user_with_global_models(self, user_model: PersonalizedModel):
        """Initialize new user with copies of global models"""
        if 'prioritization' in self.global_models:
            # Clone global model for personalization
            user_model.prioritization_model = clone(self.global_models['prioritization'])
        
        if 'time_estimation' in self.global_models:
            user_model.time_estimation_model = clone(self.global_models['time_estimation'])
            
        if 'completion_prediction' in self.global_models:
            user_model.completion_predictor = clone(self.global_models['completion_prediction'])

    async def extract_task_features(self, task_data: Dict[str, Any], 
                                  user_id: str) -> TaskFeatures:
        """Extract comprehensive features from task data"""
        try:
            # Text embeddings
            title_text = task_data.get('title', '')
            description_text = task_data.get('description', '')
            
            title_embedding = self.embedding_model.encode([title_text])[0]
            description_embedding = self.embedding_model.encode([description_text])[0]
            
            # Temporal features
            created_at = datetime.fromisoformat(task_data.get('created_at', datetime.now().isoformat()))
            due_date = task_data.get('due_date')
            
            if due_date:
                due_date = datetime.fromisoformat(due_date)
                days_until_due = (due_date - datetime.now()).days
            else:
                days_until_due = 999  # No deadline
            
            # Category encoding
            category = task_data.get('category', 'general')
            if 'category' not in self.feature_extractors:
                self.feature_extractors['category'] = LabelEncoder()
            
            # User behavior context
            user_context = await self._get_user_context(user_id)
            
            return TaskFeatures(
                task_id=task_data.get('id', ''),
                title_embedding=title_embedding,
                description_embedding=description_embedding,
                priority_numeric=float(task_data.get('priority', 3)),
                days_until_due=days_until_due,
                estimated_duration=float(task_data.get('estimated_duration', 60)),
                category_encoded=self._encode_category(category),
                time_of_day_created=created_at.hour + created_at.minute/60,
                day_of_week=created_at.weekday(),
                is_recurring=bool(task_data.get('is_recurring', False)),
                has_dependencies=bool(task_data.get('dependencies', [])),
                complexity_score=await self._calculate_complexity_score(task_data),
                user_stress_level=user_context.get('stress_level', 0.5),
                historical_completion_rate=user_context.get('completion_rate', 0.8)
            )
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            raise

    async def predict_task_priority(self, task_data: Dict[str, Any], 
                                  user_id: str) -> PredictionResult:
        """Predict task priority using personalized model"""
        try:
            # Get user's model
            user_model = await self.get_user_model(user_id)
            
            # Extract features
            features = await self.extract_task_features(task_data, user_id)
            feature_vector = self._features_to_vector(features)
            
            # Make prediction
            if user_model.prioritization_model:
                # Use personalized model
                priority_scores = user_model.prioritization_model.predict_proba([feature_vector])[0]
                priority_prediction = np.argmax(priority_scores) + 1  # 1-5 scale
                confidence = np.max(priority_scores)
            else:
                # Fallback to global model
                if 'prioritization' in self.global_models:
                    priority_scores = self.global_models['prioritization'].predict_proba([feature_vector])[0]
                    priority_prediction = np.argmax(priority_scores) + 1
                    confidence = np.max(priority_scores)
                else:
                    # Ultimate fallback
                    priority_prediction = 3
                    confidence = 0.5
            
            # Generate explanation
            explanation = await self._generate_priority_explanation(
                features, priority_prediction, confidence
            )
            
            return PredictionResult(
                prediction=float(priority_prediction),
                confidence=float(confidence),
                explanation=explanation,
                model_version=user_model.last_updated or datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Priority prediction failed: {e}")
            return PredictionResult(
                prediction=3.0,
                confidence=0.5,
                explanation="Using default priority due to prediction error",
                model_version=datetime.now()
            )

    async def predict_completion_time(self, task_data: Dict[str, Any], 
                                    user_id: str) -> PredictionResult:
        """Predict task completion time using personalized model"""
        try:
            user_model = await self.get_user_model(user_id)
            features = await self.extract_task_features(task_data, user_id)
            feature_vector = self._features_to_vector(features)
            
            if user_model.time_estimation_model:
                time_prediction = user_model.time_estimation_model.predict([feature_vector])[0]
                # Calculate confidence based on model uncertainty
                confidence = self._calculate_time_confidence(user_model, feature_vector)
            else:
                # Use statistical baseline
                time_prediction = await self._statistical_time_estimate(task_data, user_id)
                confidence = 0.6
            
            explanation = await self._generate_time_explanation(
                features, time_prediction, confidence
            )
            
            return PredictionResult(
                prediction=float(time_prediction),
                confidence=float(confidence),
                explanation=explanation,
                model_version=user_model.last_updated or datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Time prediction failed: {e}")
            estimated_duration = task_data.get('estimated_duration', 60)
            return PredictionResult(
                prediction=float(estimated_duration),
                confidence=0.5,
                explanation="Using provided estimate due to prediction error",
                model_version=datetime.now()
            )

    async def train_personalized_model(self, user_id: str, 
                                     training_data: List[Dict[str, Any]]):
        """Train or update user's personalized model"""
        try:
            logger.info(f"Training personalized model for user {user_id}")
            
            if len(training_data) < 10:
                logger.warning(f"Insufficient training data for {user_id}: {len(training_data)} samples")
                return
            
            # Add to training queue for background processing
            await self.training_queue.put({
                'user_id': user_id,
                'training_data': training_data,
                'timestamp': datetime.now()
            })
            
            logger.info(f"Queued training for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to queue training for {user_id}: {e}")

    async def _training_worker(self):
        """Background worker for model training"""
        while True:
            try:
                # Get training job from queue
                training_job = await self.training_queue.get()
                
                user_id = training_job['user_id']
                training_data = training_job['training_data']
                
                logger.info(f"Processing training job for user {user_id}")
                
                # Perform actual training
                await self._perform_model_training(user_id, training_data)
                
                # Mark job as done
                self.training_queue.task_done()
                
            except Exception as e:
                logger.error(f"Training worker error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _perform_model_training(self, user_id: str, 
                                    training_data: List[Dict[str, Any]]):
        """Perform actual model training for user"""
        try:
            # Extract features and labels from training data
            features_list = []
            priority_labels = []
            time_labels = []
            completion_labels = []
            
            for data_point in training_data:
                features = await self.extract_task_features(data_point, user_id)
                feature_vector = self._features_to_vector(features)
                
                features_list.append(feature_vector)
                priority_labels.append(data_point.get('actual_priority', 3))
                time_labels.append(data_point.get('actual_duration', 60))
                completion_labels.append(data_point.get('was_completed', True))
            
            X = np.array(features_list)
            
            # Train prioritization model
            if len(set(priority_labels)) > 1:  # Need variety in labels
                await self._train_prioritization_model(user_id, X, priority_labels)
            
            # Train time estimation model
            if len(time_labels) > 5:
                await self._train_time_model(user_id, X, time_labels)
            
            # Train completion prediction model
            if len(set(completion_labels)) > 1:
                await self._train_completion_model(user_id, X, completion_labels)
            
            # Update model in database
            await self._save_user_model(user_id)
            
            logger.info(f"✅ Model training completed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Model training failed for {user_id}: {e}")

    async def _train_prioritization_model(self, user_id: str, X: np.ndarray, 
                                        y: List[int]):
        """Train personalized prioritization model"""
        try:
            user_model = self.user_models[user_id]
            
            # Scale features
            if user_model.scaler is None:
                user_model.scaler = StandardScaler()
                X_scaled = user_model.scaler.fit_transform(X)
            else:
                X_scaled = user_model.scaler.transform(X)
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_scaled, y)
            user_model.prioritization_model = model
            
            # Calculate accuracy
            y_pred = model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            user_model.accuracy_score = accuracy
            
            logger.info(f"Prioritization model trained for {user_id}, accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Prioritization training failed for {user_id}: {e}")

    # Helper methods
    def _features_to_vector(self, features: TaskFeatures) -> np.ndarray:
        """Convert TaskFeatures to numpy vector"""
        # Combine embeddings and numerical features
        numerical_features = np.array([
            features.priority_numeric,
            features.days_until_due,
            features.estimated_duration,
            features.category_encoded,
            features.time_of_day_created,
            features.day_of_week,
            float(features.is_recurring),
            float(features.has_dependencies),
            features.complexity_score,
            features.user_stress_level,
            features.historical_completion_rate
        ])
        
        # Combine with embeddings (use mean of title and description)
        text_features = (features.title_embedding + features.description_embedding) / 2
        
        return np.concatenate([numerical_features, text_features])

    async def _get_user_context(self, user_id: str) -> Dict[str, float]:
        """Get current user context for feature extraction"""
        try:
            async with get_async_session() as db:
                # Get recent behavior patterns
                query = select(UserBehaviorPattern).where(
                    UserBehaviorPattern.user_id == user_id
                ).order_by(UserBehaviorPattern.created_at.desc()).limit(10)
                
                result = await db.execute(query)
                patterns = result.scalars().all()
                
                context = {
                    'stress_level': 0.5,
                    'completion_rate': 0.8,
                    'productivity_score': 0.7
                }
                
                # Calculate context from patterns
                if patterns:
                    # Implementation would analyze patterns to determine current context
                    pass
                
                return context
                
        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
            return {'stress_level': 0.5, 'completion_rate': 0.8}

    def _encode_category(self, category: str) -> int:
        """Encode category to numerical value"""
        if 'category' not in self.feature_extractors:
            self.feature_extractors['category'] = LabelEncoder()
        
        try:
            return self.feature_extractors['category'].transform([category])[0]
        except ValueError:
            # Unknown category, fit and transform
            existing_classes = list(self.feature_extractors['category'].classes_)
            existing_classes.append(category)
            self.feature_extractors['category'].fit(existing_classes)
            return self.feature_extractors['category'].transform([category])[0]

    async def _calculate_complexity_score(self, task_data: Dict[str, Any]) -> float:
        """Calculate task complexity based on various factors"""
        try:
            # Analyze text complexity
            title = task_data.get('title', '')
            description = task_data.get('description', '')
            
            doc = self.nlp_model(f"{title} {description}")
            
            # Factors contributing to complexity
            word_count = len(doc)
            entity_count = len(doc.ents)
            dependency_depth = max([token.depth for token in doc], default=0)
            
            # Normalize to 0-1 scale
            complexity = min(1.0, (word_count / 100) + (entity_count / 10) + (dependency_depth / 10))
            
            return complexity
            
        except Exception as e:
            logger.error(f"Complexity calculation failed: {e}")
            return 0.5  # Default complexity

    async def cleanup(self):
        """Cleanup AI engine resources"""
        logger.info("🧹 Cleaning up AI Engine...")
        
        # Clear model cache
        self.user_models.clear()
        self.global_models.clear()
        self.inference_cache.clear()
        
        # Clear embedding model
        if self.embedding_model:
            del self.embedding_model
        
        # Clear NLP model
        if self.nlp_model:
            del self.nlp_model
        
        logger.info("✅ AI Engine cleanup complete")

# Global AI engine instance
ai_engine = ProductionAIEngine()

async def get_ai_engine() -> ProductionAIEngine:
    """Get the global AI engine instance"""
    return ai_engine