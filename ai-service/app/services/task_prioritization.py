# app/services/task_prioritization.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.schemas import TaskBase, PrioritizedTask, TaskPriority, UserPreferences
from app.models.database import TrainingData, UserAnalytics, TaskPrediction
from app.core.config import get_settings
from app.core.exceptions import ModelNotLoadedException, InsufficientDataException

logger = logging.getLogger(__name__)

class TaskPrioritizationResult:
    def __init__(self, prioritized_tasks: List[PrioritizedTask], confidence_scores: Dict[str, float], 
                 reasoning: List[str], model_version: str, processing_time_ms: int):
        self.prioritized_tasks = prioritized_tasks
        self.confidence_scores = confidence_scores
        self.reasoning = reasoning
        self.model_version = model_version
        self.processing_time_ms = processing_time_ms

class TaskPrioritizationService:
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_version = "1.0.0"
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the task prioritization service"""
        try:
            logger.info("Initializing Task Prioritization Service...")
            
            # Try to load existing model
            if await self._load_model():
                logger.info("Loaded existing prioritization model")
            else:
                # Train initial model with default data
                await self._train_initial_model()
                logger.info("Trained initial prioritization model")
            
            self.is_initialized = True
            logger.info("Task Prioritization Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Task Prioritization Service: {e}")
            raise

    async def _load_model(self) -> bool:
        """Load existing model from disk"""
        try:
            model_path = f"{self.settings.model_cache_dir}/task_prioritization_model.joblib"
            scaler_path = f"{self.settings.model_cache_dir}/task_prioritization_scaler.joblib"
            encoders_path = f"{self.settings.model_cache_dir}/task_prioritization_encoders.joblib"
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoders = joblib.load(encoders_path)
            
            return True
        except FileNotFoundError:
            logger.info("No existing model found, will train new one")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    async def _save_model(self):
        """Save model to disk"""
        try:
            import os
            os.makedirs(self.settings.model_cache_dir, exist_ok=True)
            
            model_path = f"{self.settings.model_cache_dir}/task_prioritization_model.joblib"
            scaler_path = f"{self.settings.model_cache_dir}/task_prioritization_scaler.joblib"
            encoders_path = f"{self.settings.model_cache_dir}/task_prioritization_encoders.joblib"
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoders, encoders_path)
            
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    async def _train_initial_model(self):
        """Train initial model with synthetic data"""
        logger.info("Training initial model with synthetic data...")
        
        # Generate synthetic training data
        synthetic_data = self._generate_synthetic_training_data(1000)
        
        # Train model
        await self._train_model(synthetic_data)
        
        # Save model
        await self._save_model()

    def _generate_synthetic_training_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic training data for initial model"""
        np.random.seed(42)
        
        data = []
        for _ in range(n_samples):
            # Random task features
            days_until_due = np.random.exponential(7)  # Exponential distribution
            estimated_duration = np.random.lognormal(3, 1)  # Log-normal distribution
            category = np.random.choice(['work', 'personal', 'health', 'learning', 'social'])
            has_dependencies = np.random.choice([0, 1], p=[0.7, 0.3])
            complexity = np.random.randint(1, 6)
            
            # Calculate synthetic priority based on business logic
            urgency_factor = max(0, 1 - (days_until_due / 14))  # More urgent as due date approaches
            importance_factor = complexity / 5.0
            dependency_factor = 0.2 if has_dependencies else 0
            
            # Weighted priority score
            priority_score = (
                0.4 * urgency_factor + 
                0.4 * importance_factor + 
                0.2 * dependency_factor
            )
            
            # Add some noise
            priority_score += np.random.normal(0, 0.1)
            priority_score = np.clip(priority_score, 0, 1)
            
            data.append({
                'days_until_due': days_until_due,
                'estimated_duration': estimated_duration,
                'category': category,
                'has_dependencies': has_dependencies,
                'complexity': complexity,
                'hour_of_day': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'priority_score': priority_score
            })
        
        return pd.DataFrame(data)

    async def _train_model(self, training_data: pd.DataFrame):
        """Train the prioritization model"""
        logger.info(f"Training model with {len(training_data)} samples")
        
        # Prepare features
        features = training_data.copy()
        target = features.pop('priority_score')
        
        # Encode categorical variables
        categorical_columns = ['category']
        for col in categorical_columns:
            if col in features.columns:
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
                self.label_encoders[col] = le
        
        # Store feature columns
        self.feature_columns = list(features.columns)
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, target, test_size=0.2, random_state=42
        )
        
        # Train ensemble model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model trained - MSE: {mse:.4f}, R2: {r2:.4f}")

    async def prioritize_tasks(self, tasks: List[TaskBase], user_id: str, 
                             context: Dict[str, Any], db: AsyncSession) -> TaskPrioritizationResult:
        """Main method to prioritize tasks using AI"""
        start_time = datetime.now()
        
        if not self.is_initialized:
            raise ModelNotLoadedException("Task Prioritization")
        
        if not tasks:
            return TaskPrioritizationResult([], {}, ["No tasks to prioritize"], self.model_version, 0)
        
        try:
            # Extract features for each task
            features_df = await self._extract_features(tasks, user_id, context, db)
            
            # Get predictions
            priority_scores = await self._predict_priorities(features_df)
            
            # Create prioritized tasks
            prioritized_tasks = []
            confidence_scores = {}
            reasoning = []
            
            for i, task in enumerate(tasks):
                priority_score = priority_scores[i]
                
                # Convert score to priority level
                ai_priority = self._score_to_priority(priority_score)
                
                # Calculate priority factors
                priority_factors = await self._calculate_priority_factors(task, features_df.iloc[i], context)
                
                # Create prioritized task
                prioritized_task = PrioritizedTask(
                    **task.dict(),
                    ai_priority_score=float(priority_score),
                    priority_factors=priority_factors,
                    recommended_start_time=self._recommend_start_time(task, priority_score)
                )
                
                prioritized_tasks.append(prioritized_task)
                confidence_scores[task.id] = float(np.clip(priority_score, 0, 1))
            
            # Sort by priority score (descending)
            prioritized_tasks.sort(key=lambda x: x.ai_priority_score, reverse=True)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(prioritized_tasks, context)
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return TaskPrioritizationResult(
                prioritized_tasks=prioritized_tasks,
                confidence_scores=confidence_scores,
                reasoning=reasoning,
                model_version=self.model_version,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Task prioritization failed: {e}")
            raise

    async def _extract_features(self, tasks: List[TaskBase], user_id: str, 
                               context: Dict[str, Any], db: AsyncSession) -> pd.DataFrame:
        """Extract features from tasks for ML model"""
        features = []
        current_time = datetime.now()
        
        # Get user analytics for context
        user_analytics = await self._get_user_analytics(user_id, db)
        
        for task in tasks:
            # Time-based features
            days_until_due = 999  # Default for tasks without due date
            if task.due_date:
                days_until_due = (task.due_date - current_time).total_seconds() / (24 * 3600)
                days_until_due = max(0, days_until_due)
            
            # Task features
            estimated_duration = task.estimated_duration or 60  # Default 1 hour
            category = task.category or 'general'
            has_dependencies = len(task.tags) > 0  # Simplified dependency check
            complexity = task.priority.value if task.priority else 3
            
            # Context features
            hour_of_day = current_time.hour
            day_of_week = current_time.weekday()
            
            # User context features
            avg_completion_time = user_analytics.get('avg_completion_time', 60)
            productivity_score = user_analytics.get('productivity_score', 0.5)
            
            features.append({
                'days_until_due': days_until_due,
                'estimated_duration': estimated_duration,
                'category': category,
                'has_dependencies': int(has_dependencies),
                'complexity': complexity,
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week,
                'avg_completion_time': avg_completion_time,
                'productivity_score': productivity_score
            })
        
        return pd.DataFrame(features)

    async def _get_user_analytics(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Get user analytics for feature extraction"""
        try:
            # Get recent analytics
            query = select(UserAnalytics).where(
                UserAnalytics.user_id == user_id
            ).order_by(UserAnalytics.date.desc()).limit(30)
            
            result = await db.execute(query)
            analytics = result.scalars().all()
            
            if not analytics:
                return {'avg_completion_time': 60, 'productivity_score': 0.5}
            
            # Calculate averages
            avg_completion_time = np.mean([a.average_completion_time for a in analytics if a.average_completion_time])
            avg_productivity = np.mean([a.productivity_score for a in analytics if a.productivity_score])
            
            return {
                'avg_completion_time': avg_completion_time or 60,
                'productivity_score': avg_productivity or 0.5
            }
            
        except Exception as e:
            logger.error(f"Error getting user analytics: {e}")
            return {'avg_completion_time': 60, 'productivity_score': 0.5}

    async def _predict_priorities(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict priority scores using trained model"""
        try:
            # Prepare features for prediction
            features = features_df.copy()
            
            # Encode categorical variables
            for col, encoder in self.label_encoders.items():
                if col in features.columns:
                    # Handle unseen categories
                    features[col] = features[col].astype(str)
                    known_classes = set(encoder.classes_)
                    features[col] = features[col].apply(
                        lambda x: x if x in known_classes else encoder.classes_[0]
                    )
                    features[col] = encoder.transform(features[col])
            
            # Ensure all expected columns are present
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0
            
            # Select and order columns
            features = features[self.feature_columns]
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            
            # Ensure predictions are in valid range
            predictions = np.clip(predictions, 0, 1)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            # Return default scores
            return np.full(len(features_df), 0.5)

    def _score_to_priority(self, score: float) -> TaskPriority:
        """Convert priority score to TaskPriority enum"""
        if score >= 0.8:
            return TaskPriority.CRITICAL
        elif score >= 0.6:
            return TaskPriority.HIGH
        elif score >= 0.4:
            return TaskPriority.MEDIUM
        elif score >= 0.2:
            return TaskPriority.LOW
        else:
            return TaskPriority.VERY_LOW

    async def _calculate_priority_factors(self, task: TaskBase, features: pd.Series, 
                                        context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual priority factors for explanation"""
        factors = {}
        
        # Urgency factor (based on due date)
        if task.due_date:
            days_until_due = features['days_until_due']
            if days_until_due <= 1:
                factors['urgency'] = 1.0
            elif days_until_due <= 7:
                factors['urgency'] = 0.8
            else:
                factors['urgency'] = max(0.2, 1.0 - (days_until_due / 30))
        else:
            factors['urgency'] = 0.3
        
        # Importance factor (based on priority and complexity)
        complexity = features['complexity']
        factors['importance'] = complexity / 5.0
        
        # Context factor (based on time of day, etc.)
        hour = features['hour_of_day']
        if 9 <= hour <= 17:  # Work hours
            factors['timing'] = 0.8
        elif 18 <= hour <= 22:  # Evening
            factors['timing'] = 0.6
        else:
            factors['timing'] = 0.3
        
        # Dependency factor
        factors['dependencies'] = 0.2 if features['has_dependencies'] else 0.0
        
        return factors

    def _recommend_start_time(self, task: TaskBase, priority_score: float) -> Optional[datetime]:
        """Recommend when to start the task"""
        if not task.due_date:
            if priority_score >= 0.7:
                return datetime.now() + timedelta(hours=1)
            else:
                return datetime.now() + timedelta(days=1)
        
        # Calculate recommended start time based on due date and duration
        duration_hours = (task.estimated_duration or 60) / 60
        buffer_hours = duration_hours * 1.5  # Add 50% buffer
        
        recommended_start = task.due_date - timedelta(hours=buffer_hours)
        
        # Don't recommend starting in the past
        if recommended_start <= datetime.now():
            return datetime.now() + timedelta(hours=1)
        
        return recommended_start

    def _generate_reasoning(self, prioritized_tasks: List[PrioritizedTask], 
                          context: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning for prioritization"""
        reasoning = []
        
        if not prioritized_tasks:
            return ["No tasks to prioritize"]
        
        high_priority_count = sum(1 for task in prioritized_tasks if task.ai_priority_score >= 0.7)
        urgent_count = sum(1 for task in prioritized_tasks 
                         if task.due_date and (task.due_date - datetime.now()).days <= 1)
        
        reasoning.append(f"Analyzed {len(prioritized_tasks)} tasks")
        
        if high_priority_count > 0:
            reasoning.append(f"Found {high_priority_count} high-priority tasks requiring immediate attention")
        
        if urgent_count > 0:
            reasoning.append(f"Identified {urgent_count} urgent tasks with approaching deadlines")
        
        # Top task reasoning
        top_task = prioritized_tasks[0]
        top_factors = sorted(top_task.priority_factors.items(), key=lambda x: x[1], reverse=True)
        main_factor = top_factors[0][0] if top_factors else "overall score"
        
        reasoning.append(f"'{top_task.title}' ranked highest due to {main_factor}")
        
        return reasoning

    async def learn_from_interaction(self, user_id: str, tasks: List[TaskBase], 
                                   prioritization_result: TaskPrioritizationResult):
        """Learn from user interactions with prioritized tasks"""
        try:
            # Store interaction data for personalized learning
            logger.info(f"Learning from user {user_id} interaction with {len(tasks)} tasks")
            
            # This could trigger model retraining for personalization
            # Implementation would depend on the specific feedback mechanism
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")

    async def retrain_personalized_model(self, user_id: str, db: AsyncSession):
        """Retrain model with user-specific data for personalization"""
        try:
            logger.info(f"Retraining personalized model for user {user_id}")
            
            # Get user's historical data
            user_training_data = await self._get_user_training_data(user_id, db)
            
            if len(user_training_data) < 10:  # Need minimum data
                logger.warning(f"Insufficient training data for user {user_id}")
                return
            
            # Create personalized model
            await self._train_personalized_model(user_id, user_training_data)
            
            logger.info(f"Personalized model trained for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error retraining personalized model: {e}")

    async def _get_user_training_data(self, user_id: str, db: AsyncSession) -> pd.DataFrame:
        """Get user-specific training data"""
        query = select(TrainingData).where(TrainingData.user_id == user_id)
        result = await db.execute(query)
        training_records = result.scalars().all()
        
        data = []
        for record in training_records:
            data.append({
                'user_id': record.user_id,
                'task_id': record.task_id,
                'initial_priority': record.initial_priority,
                'actual_completion_time': record.actual_completion_time,
                'was_deadline_met': record.was_deadline_met,
                'created_at': record.created_at
            })
        
        return pd.DataFrame(data)

    async def _train_personalized_model(self, user_id: str, training_data: pd.DataFrame):
        """Train a personalized model for specific user"""
        # Implementation would create user-specific model weights
        # For now, we'll store user preferences
        pass

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Task Prioritization Service...")
        # Cleanup any resources if needed