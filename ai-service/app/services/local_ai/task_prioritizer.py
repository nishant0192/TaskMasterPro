# ai-service/app/services/local_ai/task_prioritizer.py
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.core.database import get_async_session
from app.models.database import (
    UserBehaviorPattern, TaskEmbedding, UserProductivityMetrics,
    AIPrediction, UserInteractionLog
)
from app.models.schemas import TaskBase, PrioritizedTask, TaskPriority
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)


class PersonalizedTaskPrioritizer:
    """AI-powered personalized task prioritization service"""

    def __init__(self, user_id: str, model_manager: ModelManager):
        self.user_id = user_id
        self.model_manager = model_manager
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.is_initialized = False

        # User-specific patterns
        self.user_patterns = {}
        self.priority_preferences = {}
        self.completion_history = {}

    async def initialize(self):
        """Initialize the task prioritizer for the specific user"""
        try:
            logger.info(
                f"Initializing task prioritizer for user {self.user_id}")

            # Load user-specific model or create one
            await self._load_or_create_user_model()

            # Load user behavior patterns
            await self._load_user_patterns()

            self.is_initialized = True
            logger.info(
                f"Task prioritizer initialized for user {self.user_id}")

        except Exception as e:
            logger.error(f"Failed to initialize task prioritizer: {e}")
            raise

    async def _load_or_create_user_model(self):
        """Load existing user model or create a new one"""
        # Try to load existing model
        self.model = await self.model_manager.get_user_model(
            self.user_id, 'prioritization', 'classification'
        )

        if self.model is None:
            # Create base model from global model
            base_model = await self.model_manager._get_base_model('prioritization')
            if base_model:
                self.model = base_model
            else:
                # Create new model
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )

            # Train initial model if we have data
            await self._train_initial_model()

        # Load or create scaler
        self.scaler = StandardScaler()

    async def _load_user_patterns(self):
        """Load user behavior patterns from database"""
        try:
            async with get_async_session() as session:
                # Get behavior patterns
                patterns_query = select(UserBehaviorPattern).where(
                    UserBehaviorPattern.user_id == self.user_id
                )
                result = await session.execute(patterns_query)
                patterns = result.scalars().all()

                for pattern in patterns:
                    self.user_patterns[pattern.pattern_type] = {
                        'data': pattern.pattern_data,
                        'confidence': pattern.confidence_score,
                        'frequency': pattern.frequency_count
                    }

                # Get recent productivity metrics
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                metrics_query = select(UserProductivityMetrics).where(
                    and_(
                        UserProductivityMetrics.user_id == self.user_id,
                        UserProductivityMetrics.metric_date >= cutoff_date
                    )
                ).order_by(UserProductivityMetrics.metric_date.desc())

                result = await session.execute(metrics_query)
                metrics = result.scalars().all()

                if metrics:
                    # Calculate average patterns
                    self.completion_history = {
                        'avg_completion_time': np.mean([m.avg_completion_time for m in metrics if m.avg_completion_time]),
                        'productivity_score': np.mean([m.productivity_score for m in metrics if m.productivity_score]),
                        'deadline_success_rate': np.mean([m.deadline_success_rate for m in metrics if m.deadline_success_rate]),
                        'peak_hours': self._aggregate_peak_hours([m.peak_productivity_hours for m in metrics if m.peak_productivity_hours])
                    }

        except Exception as e:
            logger.error(f"Error loading user patterns: {e}")

    def _aggregate_peak_hours(self, peak_hours_list: List[List[int]]) -> List[int]:
        """Aggregate peak hours from multiple records"""
        if not peak_hours_list:
            return [9, 10, 14, 15]  # Default peak hours

        hour_counts = {}
        for hours in peak_hours_list:
            if hours:
                for hour in hours:
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1

        # Return hours that appear in at least 30% of records
        threshold = len(peak_hours_list) * 0.3
        return [hour for hour, count in hour_counts.items() if count >= threshold]

    async def _train_initial_model(self):
        """Train initial model using available data"""
        try:
            training_data = await self._collect_training_data()

            if len(training_data) < 10:
                logger.warning(
                    f"Insufficient training data for user {self.user_id}, using synthetic data")
                training_data = self._generate_synthetic_training_data(50)

            if len(training_data) > 0:
                await self._train_model(training_data)

        except Exception as e:
            logger.error(f"Error training initial model: {e}")

    async def _collect_training_data(self) -> pd.DataFrame:
        """Collect training data from user's task history"""
        try:
            async with get_async_session() as session:
                # Get task embeddings with completion data
                query = select(TaskEmbedding).where(
                    and_(
                        TaskEmbedding.user_id == self.user_id,
                        TaskEmbedding.completion_status.in_(
                            ['completed', 'abandoned'])
                    )
                ).limit(1000)  # Limit to prevent memory issues

                result = await session.execute(query)
                task_records = result.scalars().all()

                training_data = []
                for record in task_records:
                    # Extract features
                    features = self._extract_features_from_task_record(record)

                    # Determine priority label based on actual completion behavior
                    priority_label = self._infer_priority_from_behavior(record)

                    features['priority_label'] = priority_label
                    training_data.append(features)

                return pd.DataFrame(training_data)

        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return pd.DataFrame()

    def _extract_features_from_task_record(self, record: TaskEmbedding) -> Dict[str, Any]:
        """Extract features from a task record"""
        now = datetime.utcnow()

        # Time-based features
        features = {
            'estimated_duration': record.estimated_duration or 60,
            'actual_duration': record.actual_duration or 60,
            'created_hour': record.created_hour or now.hour,
            'created_day_of_week': record.created_day_of_week or now.weekday(),
            'user_energy_level': record.user_energy_level or 0.5,
            'completion_quality': record.completion_quality or 0.5,
        }

        # Category encoding
        category_mapping = {
            'work': 1, 'personal': 2, 'health': 3, 'learning': 4,
            'social': 5, 'finance': 6, 'shopping': 7, 'travel': 8
        }
        features['category_encoded'] = category_mapping.get(record.category, 0)

        # Priority features
        features['original_priority'] = record.priority or 3

        # Context features
        features['is_peak_hour'] = 1 if record.created_hour in self.completion_history.get(
            'peak_hours', []) else 0
        features['is_work_day'] = 1 if record.created_day_of_week < 5 else 0

        return features

    def _infer_priority_from_behavior(self, record: TaskEmbedding) -> int:
        """Infer actual priority from user behavior"""
        # High priority indicators
        if record.completion_status == 'completed':
            if record.actual_duration and record.estimated_duration:
                time_ratio = record.actual_duration / record.estimated_duration
                if time_ratio < 1.2:  # Completed efficiently
                    return 4  # High priority

            if record.completion_quality and record.completion_quality > 0.8:
                return 4

        # Low priority indicators
        if record.completion_status == 'abandoned':
            return 1  # Very low priority

        # Medium priority default
        return record.priority or 3

    def _generate_synthetic_training_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic training data for cold start"""
        np.random.seed(42)

        data = []
        for _ in range(n_samples):
            # Generate synthetic task features
            estimated_duration = np.random.lognormal(
                3, 1)  # Log-normal distribution
            created_hour = np.random.randint(0, 24)
            created_day_of_week = np.random.randint(0, 7)
            user_energy_level = np.random.beta(2, 2)  # Beta distribution
            category_encoded = np.random.randint(0, 9)
            original_priority = np.random.randint(1, 6)

            # Calculate synthetic priority based on heuristics
            priority_score = 0

            # Time urgency
            if created_hour in [9, 10, 11, 14, 15]:  # Peak hours
                priority_score += 0.3

            # Duration impact
            if estimated_duration < 30:  # Quick tasks
                priority_score += 0.2
            elif estimated_duration > 120:  # Long tasks
                priority_score += 0.4

            # Energy level
            priority_score += user_energy_level * 0.3

            # Original priority influence
            priority_score += (original_priority / 5.0) * 0.4

            # Add noise
            priority_score += np.random.normal(0, 0.1)
            priority_score = np.clip(priority_score, 0, 1)

            # Convert to priority label
            if priority_score >= 0.8:
                priority_label = 5
            elif priority_score >= 0.6:
                priority_label = 4
            elif priority_score >= 0.4:
                priority_label = 3
            elif priority_score >= 0.2:
                priority_label = 2
            else:
                priority_label = 1

            data.append({
                'estimated_duration': estimated_duration,
                'actual_duration': estimated_duration * np.random.uniform(0.7, 1.5),
                'created_hour': created_hour,
                'created_day_of_week': created_day_of_week,
                'user_energy_level': user_energy_level,
                'completion_quality': np.random.beta(3, 2),
                'category_encoded': category_encoded,
                'original_priority': original_priority,
                'is_peak_hour': 1 if created_hour in [9, 10, 14, 15] else 0,
                'is_work_day': 1 if created_day_of_week < 5 else 0,
                'priority_label': priority_label
            })

        return pd.DataFrame(data)

    async def _train_model(self, training_data: pd.DataFrame):
        """Train the prioritization model"""
        try:
            if len(training_data) < 5:
                logger.warning("Insufficient training data")
                return

            # Prepare features and target
            feature_cols = [
                col for col in training_data.columns if col != 'priority_label']
            self.feature_columns = feature_cols

            X = training_data[feature_cols]
            y = training_data['priority_label']

            # Handle missing values
            X = X.fillna(X.mean())

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data if we have enough samples
            if len(X) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(
                f"Model trained for user {self.user_id} with accuracy: {accuracy:.3f}")

            # Save model
            await self.model_manager.save_user_model(
                self.user_id, 'prioritization', self.model,
                model_subtype='classification',
                metadata={'feature_columns': self.feature_columns,
                          'accuracy': accuracy},
                accuracy_score=accuracy,
                training_samples=len(training_data)
            )

        except Exception as e:
            logger.error(f"Error training model: {e}")

    async def prioritize_tasks(self, tasks: List[TaskBase],
                               context: Dict[str, Any] = None) -> List[PrioritizedTask]:
        """Prioritize tasks using personalized AI model"""
        if not self.is_initialized:
            await self.initialize()

        if not tasks:
            return []

        try:
            # Extract features for all tasks
            features_df = await self._extract_task_features(tasks, context or {})

            # Get predictions
            if len(features_df) > 0 and self.model:
                # Scale features
                X_scaled = self.scaler.transform(
                    features_df[self.feature_columns])

                # Get priority predictions
                priority_predictions = self.model.predict(X_scaled)
                priority_probabilities = self.model.predict_proba(X_scaled)

                # Create prioritized tasks
                prioritized_tasks = []
                for i, task in enumerate(tasks):
                    priority_score = priority_predictions[i]
                    confidence = np.max(priority_probabilities[i])

                    # Calculate priority factors for explanation
                    priority_factors = await self._calculate_priority_factors(
                        task, features_df.iloc[i], context or {}
                    )

                    # Determine recommended start time
                    recommended_start_time = self._recommend_start_time(
                        task, priority_score, features_df.iloc[i]
                    )

                    prioritized_task = PrioritizedTask(
                        **task.dict(),
                        ai_priority_score=float(
                            priority_score) / 5.0,  # Normalize to 0-1
                        priority_factors=priority_factors,
                        recommended_start_time=recommended_start_time
                    )
                    prioritized_tasks.append(prioritized_task)

                # Sort by priority score
                prioritized_tasks.sort(
                    key=lambda x: x.ai_priority_score, reverse=True)

                # Log prediction for learning
                await self._log_prediction(tasks, prioritized_tasks, context or {})

                return prioritized_tasks

        except Exception as e:
            logger.error(f"Error prioritizing tasks: {e}")

        # Fallback: return tasks with default prioritization
        return [
            PrioritizedTask(
                **task.dict(),
                ai_priority_score=0.5,
                priority_factors={'default': 1.0},
                recommended_start_time=None
            )
            for task in tasks
        ]

    async def _extract_task_features(self, tasks: List[TaskBase],
                                     context: Dict[str, Any]) -> pd.DataFrame:
        """Extract features from tasks for prediction"""
        features_list = []
        current_time = datetime.utcnow()

        for task in tasks:
            # Basic task features
            features = {
                'estimated_duration': task.estimated_duration or 60,
                'actual_duration': task.estimated_duration or 60,  # Use estimate as placeholder
                'created_hour': current_time.hour,
                'created_day_of_week': current_time.weekday(),
                'user_energy_level': context.get('energy_level', 0.7),
                'completion_quality': 0.7,  # Default expectation
                'original_priority': task.priority.value if task.priority else 3,
            }

            # Category encoding
            category_mapping = {
                'work': 1, 'personal': 2, 'health': 3, 'learning': 4,
                'social': 5, 'finance': 6, 'shopping': 7, 'travel': 8
            }
            features['category_encoded'] = category_mapping.get(
                task.category, 0)

            # Time-based features
            peak_hours = self.completion_history.get(
                'peak_hours', [9, 10, 14, 15])
            features['is_peak_hour'] = 1 if current_time.hour in peak_hours else 0
            features['is_work_day'] = 1 if current_time.weekday() < 5 else 0

            # Deadline urgency
            if task.due_date:
                hours_until_due = (
                    task.due_date - current_time).total_seconds() / 3600
                features['hours_until_due'] = max(0, hours_until_due)
                features['is_overdue'] = 1 if hours_until_due < 0 else 0
                features['urgency_score'] = max(
                    0, 1 - (hours_until_due / 168))  # 1 week = 168 hours
            else:
                features['hours_until_due'] = 999
                features['is_overdue'] = 0
                features['urgency_score'] = 0

            # Context features
            features['calendar_busy'] = 1 if context.get(
                'calendar_busy', False) else 0
            features['current_stress_level'] = context.get('stress_level', 0.3)

            features_list.append(features)

        df = pd.DataFrame(features_list)

        # Ensure all expected columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        return df

    async def _calculate_priority_factors(self, task: TaskBase, features: pd.Series,
                                          context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual priority factors for explanation"""
        factors = {}

        # Urgency factor
        if hasattr(features, 'urgency_score'):
            factors['urgency'] = float(features['urgency_score'])
        else:
            factors['urgency'] = 0.3

        # Importance factor (based on original priority)
        if hasattr(features, 'original_priority'):
            factors['importance'] = float(features['original_priority']) / 5.0
        else:
            factors['importance'] = 0.6

        # Energy alignment factor
        is_peak_hour = features.get('is_peak_hour', 0)
        energy_level = features.get('user_energy_level', 0.5)
        factors['energy_alignment'] = float(
            is_peak_hour * 0.5 + energy_level * 0.5)

        # Duration appropriateness
        duration = features.get('estimated_duration', 60)
        if duration < 30:
            factors['duration_fit'] = 0.8  # Quick tasks are good fillers
        elif duration > 120:
            factors['duration_fit'] = 0.6  # Long tasks need planning
        else:
            factors['duration_fit'] = 1.0  # Medium tasks are flexible

        # Context alignment
        calendar_busy = features.get('calendar_busy', 0)
        factors['context_fit'] = 1.0 - float(calendar_busy) * 0.3

        return factors

    def _recommend_start_time(self, task: TaskBase, priority_score: float,
                              features: pd.Series) -> Optional[datetime]:
        """Recommend when to start the task"""
        current_time = datetime.utcnow()

        # High priority tasks - start soon
        if priority_score >= 4:
            return current_time + timedelta(minutes=30)

        # Medium priority - within a few hours
        elif priority_score >= 3:
            return current_time + timedelta(hours=2)

        # Lower priority - can wait
        else:
            if task.due_date:
                # Start early enough to complete before deadline
                duration_hours = (task.estimated_duration or 60) / 60
                buffer_hours = duration_hours * 1.5  # 50% buffer
                return task.due_date - timedelta(hours=buffer_hours)
            else:
                return current_time + timedelta(days=1)

    async def _log_prediction(self, original_tasks: List[TaskBase],
                              prioritized_tasks: List[PrioritizedTask],
                              context: Dict[str, Any]):
        """Log prediction for continuous learning"""
        try:
            async with get_async_session() as session:
                for original, prioritized in zip(original_tasks, prioritized_tasks):
                    prediction = AIPrediction(
                        user_id=self.user_id,
                        prediction_type='task_priority',
                        target_task_id=original.id,
                        predicted_value=prioritized.ai_priority_score,
                        predicted_confidence=0.8,  # Placeholder
                        prediction_context=context
                    )
                    session.add(prediction)

                await session.commit()

        except Exception as e:
            logger.error(f"Error logging prediction: {e}")

    async def learn_from_feedback(self, task_id: str, actual_priority: int,
                                  completion_data: Dict[str, Any]):
        """Learn from user feedback on task priority"""
        try:
            # Store feedback for future training
            async with get_async_session() as session:
                # Update prediction record with actual outcome
                query = select(AIPrediction).where(
                    and_(
                        AIPrediction.user_id == self.user_id,
                        AIPrediction.target_task_id == task_id,
                        AIPrediction.prediction_type == 'task_priority'
                    )
                ).order_by(AIPrediction.created_at.desc())

                result = await session.execute(query)
                prediction = result.scalar_one_or_none()

                if prediction:
                    prediction.actual_value = actual_priority
                    prediction.feedback_provided = True
                    prediction.outcome_recorded_at = datetime.utcnow()

                    # Calculate accuracy
                    prediction.prediction_accuracy = 1.0 - abs(
                        prediction.predicted_value - actual_priority
                    ) / 5.0

                # Log interaction
                interaction = UserInteractionLog(
                    user_id=self.user_id,
                    interaction_type='priority_feedback',
                    task_id=task_id,
                    before_state={
                        'predicted_priority': prediction.predicted_value if prediction else None},
                    after_state={'actual_priority': actual_priority},
                    user_context=completion_data,
                    ai_prediction_made=True,
                    prediction_accuracy=prediction.prediction_accuracy if prediction else None
                )
                session.add(interaction)

                await session.commit()

            # Trigger model retraining if we have enough feedback
            await self._check_retrain_trigger()

        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")

    async def _check_retrain_trigger(self):
        """Check if model should be retrained based on feedback"""
        try:
            async with get_async_session() as session:
                # Count recent feedback
                cutoff_date = datetime.utcnow() - timedelta(days=7)

                query = select(AIPrediction).where(
                    and_(
                        AIPrediction.user_id == self.user_id,
                        AIPrediction.prediction_type == 'task_priority',
                        AIPrediction.feedback_provided == True,
                        AIPrediction.outcome_recorded_at >= cutoff_date
                    )
                )

                result = await session.execute(query)
                recent_predictions = result.scalars().all()

                if len(recent_predictions) >= 10:  # Minimum feedback threshold
                    # Check average accuracy
                    avg_accuracy = np.mean(
                        [p.prediction_accuracy for p in recent_predictions])

                    if avg_accuracy < 0.7:  # Accuracy threshold
                        logger.info(
                            f"Triggering model retrain for user {self.user_id} (accuracy: {avg_accuracy:.3f})")
                        await self._retrain_model()

        except Exception as e:
            logger.error(f"Error checking retrain trigger: {e}")

    async def _retrain_model(self):
        """Retrain the model with new data"""
        try:
            # Collect latest training data including feedback
            training_data = await self._collect_training_data()

            if len(training_data) >= 10:
                await self._train_model(training_data)
                logger.info(f"Model retrained for user {self.user_id}")

        except Exception as e:
            logger.error(f"Error retraining model: {e}")

    async def get_priority_insights(self) -> Dict[str, Any]:
        """Get insights about user's priority patterns"""
        try:
            insights = {
                'patterns_learned': len(self.user_patterns),
                'completion_history': self.completion_history,
                'model_accuracy': await self._get_recent_accuracy(),
                'priority_preferences': await self._analyze_priority_preferences()
            }

            return insights

        except Exception as e:
            logger.error(f"Error getting priority insights: {e}")
            return {}

    async def _get_recent_accuracy(self) -> float:
        """Get recent model accuracy"""
        try:
            async with get_async_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=30)

                query = select(AIPrediction).where(
                    and_(
                        AIPrediction.user_id == self.user_id,
                        AIPrediction.prediction_type == 'task_priority',
                        AIPrediction.feedback_provided == True,
                        AIPrediction.outcome_recorded_at >= cutoff_date
                    )
                )

                result = await session.execute(query)
                predictions = result.scalars().all()

                if predictions:
                    return np.mean([p.prediction_accuracy for p in predictions])

        except Exception as e:
            logger.error(f"Error getting recent accuracy: {e}")

        return 0.5  # Default

    async def _analyze_priority_preferences(self) -> Dict[str, Any]:
        """Analyze user's priority assignment preferences"""
        try:
            async with get_async_session() as session:
                # Get recent interactions
                cutoff_date = datetime.utcnow() - timedelta(days=30)

                query = select(UserInteractionLog).where(
                    and_(
                        UserInteractionLog.user_id == self.user_id,
                        UserInteractionLog.interaction_type.in_(
                            ['task_created', 'priority_changed']),
                        UserInteractionLog.interaction_timestamp >= cutoff_date
                    )
                )

                result = await session.execute(query)
                interactions = result.scalars().all()

                # Analyze patterns
                preferences = {
                    'prefers_high_priority': 0,
                    'changes_priority_often': 0,
                    'consistent_priority_usage': 0
                }

                high_priority_count = 0
                priority_changes = 0

                for interaction in interactions:
                    after_state = interaction.after_state or {}
                    before_state = interaction.before_state or {}

                    # Check for high priority preference
                    if after_state.get('priority', 0) >= 4:
                        high_priority_count += 1

                    # Check for priority changes
                    if (before_state.get('priority') and after_state.get('priority') and
                            before_state['priority'] != after_state['priority']):
                        priority_changes += 1

                if interactions:
                    preferences['prefers_high_priority'] = high_priority_count / \
                        len(interactions)
                    preferences['changes_priority_often'] = priority_changes / \
                        len(interactions)

                return preferences

        except Exception as e:
            logger.error(f"Error analyzing priority preferences: {e}")
            return {}

    async def cleanup(self):
        """Cleanup resources"""
        logger.debug(f"Cleaning up task prioritizer for user {self.user_id}")
        self.user_patterns.clear()
        self.priority_preferences.clear()
        self.completion_history.clear()
