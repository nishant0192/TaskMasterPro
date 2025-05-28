# ai-service/app/services/model_manager.py
import asyncio
import logging
import numpy as np
import pickle
import gzip
import hashlib
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch

from app.core.config import get_settings
from app.core.database import get_async_session
from app.models.database import (
    UserModelWeights, AIModelRegistry, EmbeddingCache, 
    UserAIPreferences, ModelPerformanceHistory
)

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages AI model lifecycle, caching, and versioning"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_cache: Dict[str, Any] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.scaler_cache: Dict[str, StandardScaler] = {}
        self.last_cleanup = datetime.utcnow()
        self._lock = asyncio.Lock()
        
        # Global models (loaded once, shared across users)
        self.global_models = {
            'embedding_model': None,
            'base_prioritization': None,
            'base_scheduling': None,
            'base_nlp': None
        }
        
    async def initialize(self):
        """Initialize the model manager"""
        try:
            logger.info("Initializing Model Manager...")
            
            # Create directories
            await self._create_directories()
            
            # Load global models
            await self._load_global_models()
            
            # Setup cleanup task
            asyncio.create_task(self._periodic_cleanup())
            
            logger.info("Model Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Model Manager: {e}")
            raise

    async def _create_directories(self):
        """Create necessary directories for model storage"""
        directories = [
            self.settings.model_cache_dir,
            self.settings.pretrained_models_dir,
            self.settings.user_models_dir,
            self.settings.data_cache_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    async def _load_global_models(self):
        """Load global models that are shared across all users"""
        try:
            # Load embedding model
            logger.info("Loading sentence transformer model...")
            self.global_models['embedding_model'] = SentenceTransformer(
                self.settings.embedding_model_name,
                cache_folder=self.settings.pretrained_models_dir
            )
            
            # Load or create base models
            await self._load_or_create_base_models()
            
            logger.info("Global models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load global models: {e}")
            raise

    async def _load_or_create_base_models(self):
        """Load existing base models or create new ones"""
        base_model_path = os.path.join(self.settings.model_cache_dir, "base_models.joblib")
        
        try:
            if os.path.exists(base_model_path):
                logger.info("Loading existing base models...")
                base_models = joblib.load(base_model_path)
                self.global_models.update(base_models)
            else:
                logger.info("Creating new base models...")
                await self._create_base_models()
                
        except Exception as e:
            logger.warning(f"Error loading base models, creating new ones: {e}")
            await self._create_base_models()

    async def _create_base_models(self):
        """Create base models with default parameters"""
        # Base prioritization model (will be fine-tuned per user)
        self.global_models['base_prioritization'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Base scheduling model
        self.global_models['base_scheduling'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Base NLP components will be handled separately
        self.global_models['base_nlp'] = {
            'vocabulary_size': 10000,
            'embedding_dim': 384,
            'max_sequence_length': 512
        }
        
        # Save base models
        base_model_path = os.path.join(self.settings.model_cache_dir, "base_models.joblib")
        joblib.dump({
            'base_prioritization': self.global_models['base_prioritization'],
            'base_scheduling': self.global_models['base_scheduling'],
            'base_nlp': self.global_models['base_nlp']
        }, base_model_path)
        
        logger.info("Base models created and saved")

    async def get_user_model(self, user_id: str, model_type: str, 
                           model_subtype: str = None) -> Optional[Any]:
        """Get user-specific model, loading from cache or database"""
        cache_key = f"{user_id}_{model_type}_{model_subtype or 'default'}"
        
        # Check cache first
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        async with self._lock:
            # Double-check cache after acquiring lock
            if cache_key in self.model_cache:
                return self.model_cache[cache_key]
            
            # Load from database
            model = await self._load_user_model_from_db(user_id, model_type, model_subtype)
            
            if model:
                self.model_cache[cache_key] = model
                logger.debug(f"Loaded user model: {cache_key}")
                return model
            
            # Return base model if no user-specific model exists
            base_model = await self._get_base_model(model_type)
            if base_model:
                logger.debug(f"Using base model for: {cache_key}")
                return base_model
                
            return None

    async def _load_user_model_from_db(self, user_id: str, model_type: str, 
                                     model_subtype: str = None) -> Optional[Any]:
        """Load user model from database"""
        try:
            async with get_async_session() as session:
                query = select(UserModelWeights).where(
                    UserModelWeights.user_id == user_id,
                    UserModelWeights.model_type == model_type,
                    UserModelWeights.is_active == True
                )
                
                if model_subtype:
                    query = query.where(UserModelWeights.model_subtype == model_subtype)
                
                query = query.order_by(UserModelWeights.version.desc())
                result = await session.execute(query)
                model_record = result.scalar_one_or_none()
                
                if model_record:
                    # Decompress and deserialize model
                    model_data = gzip.decompress(model_record.weights)
                    model = pickle.loads(model_data)
                    return model
                    
        except Exception as e:
            logger.error(f"Error loading user model from DB: {e}")
            
        return None

    async def save_user_model(self, user_id: str, model_type: str, model: Any,
                            model_subtype: str = None, metadata: Dict[str, Any] = None,
                            accuracy_score: float = None, training_samples: int = 0):
        """Save user model to database and cache"""
        try:
            # Serialize and compress model
            model_data = pickle.dumps(model)
            compressed_data = gzip.compress(model_data)
            
            # Calculate next version number
            async with get_async_session() as session:
                query = select(UserModelWeights).where(
                    UserModelWeights.user_id == user_id,
                    UserModelWeights.model_type == model_type
                )
                if model_subtype:
                    query = query.where(UserModelWeights.model_subtype == model_subtype)
                
                query = query.order_by(UserModelWeights.version.desc())
                result = await session.execute(query)
                latest_model = result.scalar_one_or_none()
                
                next_version = (latest_model.version + 1) if latest_model else 1
                
                # Deactivate old models
                if latest_model:
                    await session.execute(
                        update(UserModelWeights)
                        .where(
                            UserModelWeights.user_id == user_id,
                            UserModelWeights.model_type == model_type,
                            UserModelWeights.model_subtype == model_subtype
                        )
                        .values(is_active=False)
                    )
                
                # Save new model
                new_model = UserModelWeights(
                    user_id=user_id,
                    model_type=model_type,
                    model_subtype=model_subtype,
                    weights=compressed_data,
                    metadata=metadata or {},
                    accuracy_score=accuracy_score,
                    training_samples=training_samples,
                    version=next_version,
                    is_active=True
                )
                
                session.add(new_model)
                await session.commit()
                
                # Update cache
                cache_key = f"{user_id}_{model_type}_{model_subtype or 'default'}"
                self.model_cache[cache_key] = model
                
                logger.info(f"Saved user model: {cache_key} (version {next_version})")
                
        except Exception as e:
            logger.error(f"Error saving user model: {e}")
            raise

    async def get_embeddings(self, texts: List[str], cache_embeddings: bool = True) -> np.ndarray:
        """Get embeddings for texts with caching"""
        if not texts:
            return np.array([])
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for existing embeddings
        if cache_embeddings:
            for i, text in enumerate(texts):
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                
                if text_hash in self.embedding_cache:
                    embeddings.append(self.embedding_cache[text_hash])
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            model = self.global_models['embedding_model']
            new_embeddings = model.encode(uncached_texts)
            
            # Update embeddings list and cache
            for i, embedding in zip(uncached_indices, new_embeddings):
                embeddings[i] = embedding
                
                if cache_embeddings:
                    text_hash = hashlib.sha256(texts[i].encode()).hexdigest()
                    self.embedding_cache[text_hash] = embedding
                    
                    # Also save to database cache if enabled
                    if len(self.embedding_cache) % 100 == 0:  # Batch save every 100 embeddings
                        await self._save_embeddings_to_db()
        
        return np.array(embeddings)

    async def _save_embeddings_to_db(self):
        """Save cached embeddings to database"""
        try:
            async with get_async_session() as session:
                for text_hash, embedding in list(self.embedding_cache.items()):
                    # Check if already exists
                    query = select(EmbeddingCache).where(EmbeddingCache.content_hash == text_hash)
                    result = await session.execute(query)
                    existing = result.scalar_one_or_none()
                    
                    if existing:
                        # Update usage count
                        await session.execute(
                            update(EmbeddingCache)
                            .where(EmbeddingCache.content_hash == text_hash)
                            .values(
                                usage_count=EmbeddingCache.usage_count + 1,
                                last_used=datetime.utcnow()
                            )
                        )
                    else:
                        # Create new cache entry
                        cache_entry = EmbeddingCache(
                            content_hash=text_hash,
                            content_type='mixed',  # Could be refined based on context
                            embedding=embedding.tobytes().hex(),
                            model_version=self.settings.embedding_model_name,
                            usage_count=1
                        )
                        session.add(cache_entry)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error saving embeddings to DB: {e}")

    async def _get_base_model(self, model_type: str) -> Optional[Any]:
        """Get base model by type"""
        base_key = f"base_{model_type}"
        return self.global_models.get(base_key)

    async def get_model_performance(self, user_id: str, model_type: str, 
                                  days_back: int = 30) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            async with get_async_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                
                query = select(ModelPerformanceHistory).where(
                    ModelPerformanceHistory.user_id == user_id,
                    ModelPerformanceHistory.model_type == model_type,
                    ModelPerformanceHistory.evaluation_date >= cutoff_date
                ).order_by(ModelPerformanceHistory.evaluation_date.desc())
                
                result = await session.execute(query)
                performance_records = result.scalars().all()
                
                if not performance_records:
                    return {'error': 'No performance data available'}
                
                # Calculate aggregated metrics
                latest_record = performance_records[0]
                avg_accuracy = np.mean([r.accuracy for r in performance_records if r.accuracy])
                avg_satisfaction = np.mean([r.user_satisfaction_avg for r in performance_records if r.user_satisfaction_avg])
                
                return {
                    'latest_accuracy': latest_record.accuracy,
                    'avg_accuracy': avg_accuracy,
                    'latest_satisfaction': latest_record.user_satisfaction_avg,
                    'avg_satisfaction': avg_satisfaction,
                    'predictions_made': latest_record.predictions_made,
                    'feedback_received': latest_record.feedback_received,
                    'positive_feedback_ratio': latest_record.positive_feedback_ratio,
                    'evaluation_count': len(performance_records)
                }
                
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {'error': str(e)}

    async def cleanup_cache(self, max_cache_size: int = None):
        """Clean up model and embedding caches"""
        max_size = max_cache_size or self.settings.embedding_cache_size
        
        # Clean embedding cache if too large
        if len(self.embedding_cache) > max_size:
            # Remove least recently used items (simplified LRU)
            items_to_remove = len(self.embedding_cache) - max_size
            keys_to_remove = list(self.embedding_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self.embedding_cache[key]
            
            logger.info(f"Cleaned {items_to_remove} items from embedding cache")
        
        # Clean model cache based on memory usage
        current_time = datetime.utcnow()
        cache_ttl = timedelta(seconds=self.settings.model_cache_ttl_seconds)
        
        keys_to_remove = []
        for key in self.model_cache.keys():
            # Simple time-based eviction (could be enhanced with usage tracking)
            if current_time - self.last_cleanup > cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.model_cache[key]
        
        if keys_to_remove:
            logger.info(f"Cleaned {len(keys_to_remove)} items from model cache")

    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_cache()
                self.last_cleanup = datetime.utcnow()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'model_cache_size': len(self.model_cache),
            'embedding_cache_size': len(self.embedding_cache),
            'scaler_cache_size': len(self.scaler_cache),
            'last_cleanup': self.last_cleanup.isoformat(),
            'global_models_loaded': len([k for k, v in self.global_models.items() if v is not None])
        }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Model Manager...")
        self.model_cache.clear()
        self.embedding_cache.clear()
        self.scaler_cache.clear()


# ai-service/app/services/local_ai/ai_coordinator.py
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.model_manager import ModelManager
from app.services.local_ai.task_prioritizer import PersonalizedTaskPrioritizer
from app.services.local_ai.smart_scheduler import PersonalizedScheduler
from app.services.local_ai.personal_nlp import PersonalizedNLP
from app.services.local_ai.behavior_predictor import BehaviorPredictor
from app.services.local_ai.continuous_learner import ContinuousLearner
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class AICoordinator:
    """Coordinates all AI services and manages user sessions"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.settings = get_settings()
        
        # AI service instances
        self.task_prioritizers: Dict[str, PersonalizedTaskPrioritizer] = {}
        self.schedulers: Dict[str, PersonalizedScheduler] = {}
        self.nlp_processors: Dict[str, PersonalizedNLP] = {}
        self.behavior_predictors: Dict[str, BehaviorPredictor] = {}
        self.continuous_learner = None
        
        # Session management
        self.active_sessions: Dict[str, datetime] = {}
        self.session_timeout = 3600  # 1 hour
        
    async def initialize(self):
        """Initialize the AI coordinator"""
        try:
            logger.info("Initializing AI Coordinator...")
            
            # Initialize continuous learner
            self.continuous_learner = ContinuousLearner(self.model_manager)
            await self.continuous_learner.initialize()
            
            # Start session cleanup task
            asyncio.create_task(self._cleanup_inactive_sessions())
            
            logger.info("AI Coordinator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Coordinator: {e}")
            raise

    async def get_user_prioritizer(self, user_id: str) -> PersonalizedTaskPrioritizer:
        """Get or create task prioritizer for user"""
        if user_id not in self.task_prioritizers:
            prioritizer = PersonalizedTaskPrioritizer(user_id, self.model_manager)
            await prioritizer.initialize()
            self.task_prioritizers[user_id] = prioritizer
        
        self._update_session(user_id)
        return self.task_prioritizers[user_id]

    async def get_user_scheduler(self, user_id: str) -> PersonalizedScheduler:
        """Get or create scheduler for user"""
        if user_id not in self.schedulers:
            scheduler = PersonalizedScheduler(user_id, self.model_manager)
            await scheduler.initialize()
            self.schedulers[user_id] = scheduler
        
        self._update_session(user_id)
        return self.schedulers[user_id]

    async def get_user_nlp(self, user_id: str) -> PersonalizedNLP:
        """Get or create NLP processor for user"""
        if user_id not in self.nlp_processors:
            nlp_processor = PersonalizedNLP(user_id, self.model_manager)
            await nlp_processor.initialize()
            self.nlp_processors[user_id] = nlp_processor
        
        self._update_session(user_id)
        return self.nlp_processors[user_id]

    async def get_behavior_predictor(self, user_id: str) -> BehaviorPredictor:
        """Get or create behavior predictor for user"""
        if user_id not in self.behavior_predictors:
            predictor = BehaviorPredictor(user_id, self.model_manager)
            await predictor.initialize()
            self.behavior_predictors[user_id] = predictor
        
        self._update_session(user_id)
        return self.behavior_predictors[user_id]

    async def process_user_feedback(self, user_id: str, feedback_data: Dict[str, Any]):
        """Process user feedback and trigger learning"""
        try:
            await self.continuous_learner.process_feedback(user_id, feedback_data)
            logger.debug(f"Processed feedback for user {user_id}")
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")

    async def trigger_model_update(self, user_id: str, model_type: str = None):
        """Trigger model update for user"""
        try:
            await self.continuous_learner.schedule_model_update(user_id, model_type)
            logger.info(f"Triggered model update for user {user_id}")
        except Exception as e:
            logger.error(f"Error triggering model update: {e}")

    def _update_session(self, user_id: str):
        """Update user session timestamp"""
        self.active_sessions[user_id] = datetime.utcnow()

    async def _cleanup_inactive_sessions(self):
        """Clean up inactive user sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.utcnow()
                inactive_users = []
                
                for user_id, last_activity in self.active_sessions.items():
                    if (current_time - last_activity).total_seconds() > self.session_timeout:
                        inactive_users.append(user_id)
                
                for user_id in inactive_users:
                    await self._cleanup_user_session(user_id)
                
                if inactive_users:
                    logger.info(f"Cleaned up {len(inactive_users)} inactive sessions")
                    
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")

    async def _cleanup_user_session(self, user_id: str):
        """Clean up specific user session"""
        try:
            # Remove from active sessions
            self.active_sessions.pop(user_id, None)
            
            # Clean up AI service instances
            if user_id in self.task_prioritizers:
                await self.task_prioritizers[user_id].cleanup()
                del self.task_prioritizers[user_id]
            
            if user_id in self.schedulers:
                await self.schedulers[user_id].cleanup()
                del self.schedulers[user_id]
            
            if user_id in self.nlp_processors:
                await self.nlp_processors[user_id].cleanup()
                del self.nlp_processors[user_id]
            
            if user_id in self.behavior_predictors:
                await self.behavior_predictors[user_id].cleanup()
                del self.behavior_predictors[user_id]
            
            logger.debug(f"Cleaned up session for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up user session {user_id}: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            'active_sessions': len(self.active_sessions),
            'task_prioritizers': len(self.task_prioritizers),
            'schedulers': len(self.schedulers),
            'nlp_processors': len(self.nlp_processors),
            'behavior_predictors': len(self.behavior_predictors),
            'model_manager_stats': await self.model_manager.get_cache_stats(),
            'continuous_learner_active': self.continuous_learner is not None
        }

    async def health_check(self) -> str:
        """Perform health check"""
        try:
            # Check model manager
            if not self.model_manager:
                return "unhealthy"
            
            # Check continuous learner
            if not self.continuous_learner:
                return "unhealthy"
            
            # Check if we can create embeddings
            test_embeddings = await self.model_manager.get_embeddings(["test"])
            if len(test_embeddings) == 0:
                return "unhealthy"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return "unhealthy"

    async def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for monitoring"""
        try:
            status = await self.get_system_status()
            
            # Add performance metrics
            total_predictions = 0
            total_feedback = 0
            
            # This would be enhanced with actual metrics collection
            return {
                **status,
                'ai_predictions_total': total_predictions,
                'ai_training_sessions_total': total_feedback,
                'model_cache_size': status['model_manager_stats']['model_cache_size'],
                'active_users': len(self.active_sessions)
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}

    async def cleanup(self):
        """Cleanup all resources"""
        logger.info("Cleaning up AI Coordinator...")
        
        # Clean up all user sessions
        for user_id in list(self.active_sessions.keys()):
            await self._cleanup_user_session(user_id)
        
        # Clean up continuous learner
        if self.continuous_learner:
            await self.continuous_learner.cleanup()
        
        logger.info("AI Coordinator cleanup complete")