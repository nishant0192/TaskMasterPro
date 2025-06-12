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
            logger.info(
                f"Initializing behavior predictor for user {self.user_id}")

            # Load or train prediction models
            await self._load_or_train_models()

            # Load behavior patterns
            await self._load_behavior_patterns()

            # Load productivity trends
            await self._load_productivity_trends()

            # Load risk profiles
            await self._load_risk_profiles()

            self.is_initialized = True
            logger.info(
                f"Behavior predictor initialized for user {self.user_id}")

        except Exception as e:
            logger.error(f"Failed to initialize behavior predictor: {e}")
            raise

    async def _load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            # Load completion probability model
            self.completion_model = await self.model_manager.get_user_model(
                self.user_id, 'prediction', 'completion_probability'
            )
            if not self.completion_model:
                await self._train_completion_model()

            # Load procrastination model
            self.procrastination_model = await self.model_manager.get_user_model(
                self.user_id, 'prediction', 'procrastination'
            )
            if not self.procrastination_model:
                await self._train_procrastination_model()

            # Load duration prediction model
            self.duration_model = await self.model_manager.get_user_model(
                self.user_id, 'prediction', 'duration'
            )
            if not self.duration_model:
                await self._train_duration_model()

            # Load success prediction model
            self.success_model = await self.model_manager.get_user_model(
                self.user_id, 'prediction', 'success'
            )
            if not self.success_model:
                await self._train_success_model()

        except Exception as e:
            logger.error(f"Error loading/training models: {e}")

    async def _train_completion_model(self):
        """Train task completion probability model"""
        try:
            # Collect training data
            training_data = await self._collect_completion_training_data()

            if len(training_data) < 10:
                # Use synthetic data for cold start
                training_data = self._generate_synthetic_completion_data(100)

            # Extract features and labels
            features = training_data.drop(
                ['completed', 'task_id'], axis=1, errors='ignore')
            labels = training_data['completed'].astype(int)

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Train model
            self.completion_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Split data if we have enough
            if len(features) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    features_scaled, labels, test_size=0.2, random_state=42
                )
                self.completion_model.fit(X_train, y_train)

                # Evaluate
                accuracy = accuracy_score(
                    y_test, self.completion_model.predict(X_test))
                logger.info(f"Completion model accuracy: {accuracy:.3f}")

                # Store feature importance
                self.feature_importance['completion'] = dict(
                    zip(features.columns, self.completion_model.feature_importances_)
                )
            else:
                self.completion_model.fit(features_scaled, labels)

            # Save model
            await self.model_manager.save_user_model(
                self.user_id, 'prediction', self.completion_model,
                model_subtype='completion_probability',
                metadata={'features': list(
                    features.columns), 'scaler': scaler},
                accuracy_score=accuracy if 'accuracy' in locals() else 0.5,
                training_samples=len(training_data)
            )

        except Exception as e:
            logger.error(f"Error training completion model: {e}")

    async def _train_procrastination_model(self):
        """Train procrastination risk model"""
        try:
            training_data = await self._collect_procrastination_training_data()

            if len(training_data) < 10:
                training_data = self._generate_synthetic_procrastination_data(
                    100)

            # Features engineering
            features = training_data.drop(
                ['procrastination_score', 'task_id'], axis=1, errors='ignore')
            labels = training_data['procrastination_score']

            # Train gradient boosting model
            self.procrastination_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            self.procrastination_model.fit(features_scaled, labels)

            # Save model
            await self.model_manager.save_user_model(
                self.user_id, 'prediction', self.procrastination_model,
                model_subtype='procrastination',
                metadata={'features': list(
                    features.columns), 'scaler': scaler},
                training_samples=len(training_data)
            )

        except Exception as e:
            logger.error(f"Error training procrastination model: {e}")

    async def _train_duration_model(self):
        """Train task duration prediction model"""
        try:
            training_data = await self._collect_duration_training_data()

            if len(training_data) < 10:
                training_data = self._generate_synthetic_duration_data(100)

            features = training_data.drop(
                ['actual_duration', 'task_id'], axis=1, errors='ignore')
            labels = training_data['actual_duration']

            # Random Forest for duration prediction
            self.duration_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            self.duration_model.fit(features_scaled, labels)

            # Save model
            await self.model_manager.save_user_model(
                self.user_id, 'prediction', self.duration_model,
                model_subtype='duration',
                metadata={'features': list(
                    features.columns), 'scaler': scaler},
                training_samples=len(training_data)
            )

        except Exception as e:
            logger.error(f"Error training duration model: {e}")

    async def _train_success_model(self):
        """Train overall task success prediction model"""
        try:
            training_data = await self._collect_success_training_data()

            if len(training_data) < 10:
                training_data = self._generate_synthetic_success_data(100)

            features = training_data.drop(
                ['success_score', 'task_id'], axis=1, errors='ignore')
            labels = training_data['success_score']

            # Gradient Boosting for success prediction
            self.success_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            self.success_model.fit(features_scaled, labels)

            # Save model
            await self.model_manager.save_user_model(
                self.user_id, 'prediction', self.success_model,
                model_subtype='success',
                metadata={'features': list(
                    features.columns), 'scaler': scaler},
                training_samples=len(training_data)
            )

        except Exception as e:
            logger.error(f"Error training success model: {e}")

    async def _collect_completion_training_data(self) -> pd.DataFrame:
        """Collect training data for completion prediction"""
        try:
            async with get_async_session() as session:
                query = select(TaskEmbedding).where(
                    and_(
                        TaskEmbedding.user_id == self.user_id,
                        TaskEmbedding.completion_status.in_(
                            ['completed', 'abandoned'])
                    )
                ).limit(1000)

                result = await session.execute(query)
                records = result.scalars().all()

                data = []
                for record in records:
                    features = {
                        'priority': record.priority or 3,
                        'estimated_duration': record.estimated_duration or 60,
                        'category_encoded': self._encode_category(record.category),
                        'created_hour': record.created_hour or 9,
                        'created_day_of_week': record.created_day_of_week or 1,
                        'user_energy_level': record.user_energy_level or 0.5,
                        'completed': 1 if record.completion_status == 'completed' else 0,
                        'task_id': record.task_id
                    }
                    data.append(features)

                return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error collecting completion training data: {e}")
            return pd.DataFrame()

    async def _collect_procrastination_training_data(self) -> pd.DataFrame:
        """Collect training data for procrastination prediction"""
        try:
            async with get_async_session() as session:
                # Get tasks with completion delays
                query = select(TaskEmbedding).where(
                    and_(
                        TaskEmbedding.user_id == self.user_id,
                        TaskEmbedding.actual_duration.isnot(None),
                        TaskEmbedding.estimated_duration.isnot(None)
                    )
                ).limit(1000)

                result = await session.execute(query)
                records = result.scalars().all()

                data = []
                for record in records:
                    # Calculate procrastination score based on delays
                    delay_ratio = (record.actual_duration /
                                   record.estimated_duration) - 1
                    procrastination_score = max(0, min(1, delay_ratio))

                    features = {
                        'priority': record.priority or 3,
                        'estimated_duration': record.estimated_duration,
                        'category_encoded': self._encode_category(record.category),
                        'created_hour': record.created_hour or 9,
                        'user_energy_level': record.user_energy_level or 0.5,
                        'completion_quality': record.completion_quality or 0.5,
                        'procrastination_score': procrastination_score,
                        'task_id': record.task_id
                    }
                    data.append(features)

                return pd.DataFrame(data)

        except Exception as e:
            logger.error(
                f"Error collecting procrastination training data: {e}")
            return pd.DataFrame()

    async def _collect_duration_training_data(self) -> pd.DataFrame:
        """Collect training data for duration prediction"""
        try:
            async with get_async_session() as session:
                query = select(TaskEmbedding).where(
                    and_(
                        TaskEmbedding.user_id == self.user_id,
                        TaskEmbedding.actual_duration.isnot(None)
                    )
                ).limit(1000)

                result = await session.execute(query)
                records = result.scalars().all()

                data = []
                for record in records:
                    features = {
                        'priority': record.priority or 3,
                        'estimated_duration': record.estimated_duration or 60,
                        'category_encoded': self._encode_category(record.category),
                        'created_hour': record.created_hour or 9,
                        'created_day_of_week': record.created_day_of_week or 1,
                        'user_energy_level': record.user_energy_level or 0.5,
                        'actual_duration': record.actual_duration,
                        'task_id': record.task_id
                    }
                    data.append(features)

                return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error collecting duration training data: {e}")
            return pd.DataFrame()

    async def _collect_success_training_data(self) -> pd.DataFrame:
        """Collect training data for success prediction"""
        try:
            async with get_async_session() as session:
                query = select(TaskEmbedding).where(
                    and_(
                        TaskEmbedding.user_id == self.user_id,
                        TaskEmbedding.completion_status.isnot(None),
                        TaskEmbedding.completion_quality.isnot(None)
                    )
                ).limit(1000)

                result = await session.execute(query)
                records = result.scalars().all()

                data = []
                for record in records:
                    # Calculate success score
                    if record.completion_status == 'completed':
                        success_score = record.completion_quality or 0.7
                    else:
                        success_score = 0.0

                    features = {
                        'priority': record.priority or 3,
                        'estimated_duration': record.estimated_duration or 60,
                        'category_encoded': self._encode_category(record.category),
                        'created_hour': record.created_hour or 9,
                        'created_day_of_week': record.created_day_of_week or 1,
                        'user_energy_level': record.user_energy_level or 0.5,
                        'success_score': success_score,
                        'task_id': record.task_id
                    }
                    data.append(features)

                return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error collecting success training data: {e}")
            return pd.DataFrame()

    def _encode_category(self, category: Optional[str]) -> int:
        """Encode task category"""
        category_mapping = {
            'work': 1, 'personal': 2, 'health': 3, 'learning': 4,
            'social': 5, 'finance': 6, 'shopping': 7, 'travel': 8
        }
        return category_mapping.get(category, 0) if category else 0

    def _generate_synthetic_completion_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic completion data"""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            priority = np.random.randint(1, 6)
            estimated_duration = np.random.lognormal(3.5, 1)
            category_encoded = np.random.randint(0, 9)
            created_hour = np.random.randint(0, 24)
            created_day_of_week = np.random.randint(0, 7)
            user_energy_level = np.random.beta(2, 2)

            # Completion probability based on features
            completion_prob = 0.5
            if priority >= 4:
                completion_prob += 0.2
            if estimated_duration < 60:
                completion_prob += 0.1
            if created_hour in [9, 10, 14, 15]:
                completion_prob += 0.1
            if user_energy_level > 0.7:
                completion_prob += 0.1

            completed = 1 if np.random.random() < completion_prob else 0

            data.append({
                'priority': priority,
                'estimated_duration': estimated_duration,
                'category_encoded': category_encoded,
                'created_hour': created_hour,
                'created_day_of_week': created_day_of_week,
                'user_energy_level': user_energy_level,
                'completed': completed
            })

        return pd.DataFrame(data)

    def _generate_synthetic_procrastination_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic procrastination data"""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            priority = np.random.randint(1, 6)
            estimated_duration = np.random.lognormal(3.5, 1)
            category_encoded = np.random.randint(0, 9)
            created_hour = np.random.randint(0, 24)
            user_energy_level = np.random.beta(2, 2)
            completion_quality = np.random.beta(3, 2)

            # Procrastination score based on features
            procrastination_score = 0.3
            if priority <= 2:  # Low priority
                procrastination_score += 0.3
            if estimated_duration > 120:  # Long tasks
                procrastination_score += 0.2
            if user_energy_level < 0.3:  # Low energy
                procrastination_score += 0.2

            procrastination_score = min(
                1.0, procrastination_score + np.random.normal(0, 0.1))

            data.append({
                'priority': priority,
                'estimated_duration': estimated_duration,
                'category_encoded': category_encoded,
                'created_hour': created_hour,
                'user_energy_level': user_energy_level,
                'completion_quality': completion_quality,
                'procrastination_score': max(0, procrastination_score)
            })

        return pd.DataFrame(data)

    def _generate_synthetic_duration_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic duration data"""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            priority = np.random.randint(1, 6)
            estimated_duration = np.random.lognormal(3.5, 1)
            category_encoded = np.random.randint(0, 9)
            created_hour = np.random.randint(0, 24)
            created_day_of_week = np.random.randint(0, 7)
            user_energy_level = np.random.beta(2, 2)

            # Actual duration based on features
            actual_duration = estimated_duration

            # Add variance based on task characteristics
            if priority <= 2:  # Low priority tasks often take longer
                actual_duration *= np.random.uniform(1.1, 1.5)
            if user_energy_level < 0.3:  # Low energy increases duration
                actual_duration *= np.random.uniform(1.1, 1.3)
            if created_hour not in [9, 10, 14, 15]:  # Non-peak hours
                actual_duration *= np.random.uniform(1.05, 1.2)

            # Add random noise
            actual_duration *= np.random.uniform(0.8, 1.2)

            data.append({
                'priority': priority,
                'estimated_duration': estimated_duration,
                'category_encoded': category_encoded,
                'created_hour': created_hour,
                'created_day_of_week': created_day_of_week,
                'user_energy_level': user_energy_level,
                'actual_duration': actual_duration
            })

        return pd.DataFrame(data)

    def _generate_synthetic_success_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic success data"""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            priority = np.random.randint(1, 6)
            estimated_duration = np.random.lognormal(3.5, 1)
            category_encoded = np.random.randint(0, 9)
            created_hour = np.random.randint(0, 24)
            created_day_of_week = np.random.randint(0, 7)
            user_energy_level = np.random.beta(2, 2)

            # Success score based on features
            success_score = 0.5

            if priority >= 4:  # High priority
                success_score += 0.2
            if estimated_duration < 90:  # Manageable duration
                success_score += 0.1
            if created_hour in [9, 10, 14, 15]:  # Peak hours
                success_score += 0.15
            if user_energy_level > 0.7:  # High energy
                success_score += 0.15

            # Add noise
            success_score = max(
                0, min(1, success_score + np.random.normal(0, 0.1)))

            data.append({
                'priority': priority,
                'estimated_duration': estimated_duration,
                'category_encoded': category_encoded,
                'created_hour': created_hour,
                'created_day_of_week': created_day_of_week,
                'user_energy_level': user_energy_level,
                'success_score': success_score
            })

        return pd.DataFrame(data)

    async def _load_behavior_patterns(self):
        """Load user behavior patterns"""
        try:
            async with get_async_session() as session:
                query = select(UserBehaviorPattern).where(
                    UserBehaviorPattern.user_id == self.user_id
                )

                result = await session.execute(query)
                patterns = result.scalars().all()

                for pattern in patterns:
                    self.behavior_patterns[pattern.pattern_type] = {
                        'data': pattern.pattern_data,
                        'confidence': pattern.confidence_score,
                        'frequency': pattern.frequency_count,
                        'last_observed': pattern.last_observed
                    }

        except Exception as e:
            logger.error(f"Error loading behavior patterns: {e}")

    async def _load_productivity_trends(self):
        """Load productivity trends"""
        try:
            async with get_async_session() as session:
                # Get last 90 days of productivity metrics
                cutoff_date = datetime.utcnow() - timedelta(days=90)

                query = select(UserProductivityMetrics).where(
                    and_(
                        UserProductivityMetrics.user_id == self.user_id,
                        UserProductivityMetrics.metric_date >= cutoff_date
                    )
                ).order_by(UserProductivityMetrics.metric_date.desc())

                result = await session.execute(query)
                metrics = result.scalars().all()

                if metrics:
                    # Analyze trends
                    self.productivity_trends = self._analyze_productivity_trends(
                        metrics)

        except Exception as e:
            logger.error(f"Error loading productivity trends: {e}")

    def _analyze_productivity_trends(self, metrics: List[UserProductivityMetrics]) -> Dict[str, Any]:
        """Analyze productivity trends from metrics"""
        if not metrics:
            return {}

        # Convert to DataFrame for easier analysis
        data = []
        for metric in metrics:
            data.append({
                'date': metric.metric_date,
                'productivity_score': metric.productivity_score or 0.5,
                'tasks_completed': metric.tasks_completed or 0,
                'tasks_abandoned': metric.tasks_abandoned or 0,
                'avg_completion_time': metric.avg_completion_time or 60,
                'deadline_success_rate': metric.deadline_success_rate or 0.5,
                'procrastination_score': metric.procrastination_score or 0.3
            })

        df = pd.DataFrame(data)
        df = df.sort_values('date')

        # Calculate trends
        trends = {
            'productivity_trend': self._calculate_trend(df['productivity_score']),
            'completion_trend': self._calculate_trend(df['tasks_completed']),
            'procrastination_trend': self._calculate_trend(df['procrastination_score']),
            'average_productivity': df['productivity_score'].mean(),
            'average_completion_rate': df['tasks_completed'].sum() / max(1, df['tasks_completed'].sum() + df['tasks_abandoned'].sum()),
            'peak_productivity_days': df.nlargest(5, 'productivity_score')['date'].tolist()
        }

        return trends

    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return 'stable'

        # Simple linear regression
        x = np.arange(len(series))
        y = series.values

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'

    async def _load_risk_profiles(self):
        """Load risk profiles for tasks"""
        try:
            # Analyze historical failures and delays
            async with get_async_session() as session:
                query = select(TaskEmbedding).where(
                    and_(
                        TaskEmbedding.user_id == self.user_id,
                        TaskEmbedding.completion_status.in_(
                            ['abandoned', 'delayed'])
                    )
                ).limit(500)

                result = await session.execute(query)
                failed_tasks = result.scalars().all()

                if failed_tasks:
                    # Identify common risk factors
                    risk_factors = self._identify_risk_factors(failed_tasks)
                    self.risk_profiles = risk_factors

        except Exception as e:
            logger.error(f"Error loading risk profiles: {e}")

    def _identify_risk_factors(self, failed_tasks: List[TaskEmbedding]) -> Dict[str, Any]:
        """Identify common risk factors from failed tasks"""
        risk_factors = {
            'high_risk_categories': [],
            'high_risk_hours': [],
            'high_risk_durations': [],
            'high_risk_priorities': []
        }

        # Analyze patterns
        categories = {}
        hours = {}
        priorities = {}

        for task in failed_tasks:
            if task.category:
                categories[task.category] = categories.get(
                    task.category, 0) + 1
            if task.created_hour is not None:
                hours[task.created_hour] = hours.get(task.created_hour, 0) + 1
            if task.priority:
                priorities[task.priority] = priorities.get(
                    task.priority, 0) + 1

        # Identify high-risk patterns
        total_failures = len(failed_tasks)

        for category, count in categories.items():
            if count / total_failures > 0.2:  # More than 20% of failures
                risk_factors['high_risk_categories'].append(category)

        for hour, count in hours.items():
            if count / total_failures > 0.1:  # More than 10% of failures
                risk_factors['high_risk_hours'].append(hour)

        for priority, count in priorities.items():
            if count / total_failures > 0.2:
                risk_factors['high_risk_priorities'].append(priority)

        return risk_factors

    async def predict_task_success(self, task: TaskBase,
                                   historical_data: Dict[str, Any] = None) -> TaskSuccessPrediction:
        """Predict task completion success"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Extract features
            features = await self._extract_task_features(task, historical_data)

            # Get predictions from different models
            completion_prob = await self._predict_completion_probability(features)
            estimated_duration = await self._predict_duration(features)
            procrastination_risk = await self._predict_procrastination_risk(features)
            success_score = await self._predict_success_score(features)

            # Calculate predicted completion date
            current_time = datetime.utcnow()
            if task.due_date:
                # Adjust based on procrastination risk
                if procrastination_risk > 0.7:
                    # High risk - likely to complete close to deadline
                    predicted_completion = task.due_date - timedelta(hours=2)
                elif procrastination_risk > 0.4:
                    # Medium risk - some buffer before deadline
                    predicted_completion = task.due_date - timedelta(days=1)
                else:
                    # Low risk - complete well before deadline
                    days_until_due = (task.due_date - current_time).days
                    predicted_completion = current_time + \
                        timedelta(days=days_until_due * 0.6)
            else:
                # No deadline - predict based on priority and duration
                if task.priority and task.priority.value >= 4:
                    predicted_completion = current_time + timedelta(days=2)
                else:
                    predicted_completion = current_time + timedelta(days=7)

            # Calculate confidence interval for duration
            confidence_interval = {
                'min': int(estimated_duration * 0.8),
                'max': int(estimated_duration * 1.5)
            }

            # Generate recommendations
            recommendations = self._generate_task_recommendations(
                task, completion_prob, procrastination_risk, success_score
            )

            return TaskSuccessPrediction(
                task_id=task.id,
                estimated_completion_time=int(estimated_duration),
                on_time_probability=completion_prob *
                (1 - procrastination_risk),
                predicted_completion_date=predicted_completion,
                confidence_interval=confidence_interval,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error predicting task success: {e}")
            # Return default prediction
            return TaskSuccessPrediction(
                task_id=task.id,
                estimated_completion_time=task.estimated_duration or 60,
                on_time_probability=0.5,
                predicted_completion_date=datetime.utcnow() + timedelta(days=3),
                confidence_interval={'min': 30, 'max': 120},
                recommendations=["Unable to generate personalized predictions"]
            )

    async def _extract_task_features(self, task: TaskBase,
                                     historical_data: Dict[str, Any] = None) -> np.ndarray:
        """Extract features from task for prediction"""
        current_time = datetime.utcnow()

        features = {
            'priority': task.priority.value if task.priority else 3,
            'estimated_duration': task.estimated_duration or 60,
            'category_encoded': self._encode_category(task.category),
            'created_hour': current_time.hour,
            'created_day_of_week': current_time.weekday(),
            'user_energy_level': historical_data.get('current_energy', 0.7) if historical_data else 0.7,
            'has_deadline': 1 if task.due_date else 0,
            'days_until_due': (task.due_date - current_time).days if task.due_date else 999,
            'description_length': len(task.description) if task.description else 0,
            'tag_count': len(task.tags)
        }

        # Add behavioral features
        if self.behavior_patterns:
            procrastination_pattern = self.behavior_patterns.get(
                'procrastination_patterns', {})
            if procrastination_pattern:
                features['user_procrastination_tendency'] = procrastination_pattern.get(
                    'data', {}).get('average_score', 0.3)
            else:
                features['user_procrastination_tendency'] = 0.3

        return np.array(list(features.values())).reshape(1, -1)

    async def _predict_completion_probability(self, features: np.ndarray) -> float:
        """Predict probability of task completion"""
        if self.completion_model:
            try:
                # Get model metadata for scaler
                metadata = await self._get_model_metadata('completion_probability')
                if metadata and 'scaler' in metadata:
                    features_scaled = metadata['scaler'].transform(features)
                    probability = self.completion_model.predict_proba(features_scaled)[
                        0][1]
                else:
                    probability = self.completion_model.predict_proba(features)[
                        0][1]

                return float(probability)
            except Exception as e:
                logger.error(f"Error predicting completion probability: {e}")

        return 0.7  # Default probability

    async def _predict_duration(self, features: np.ndarray) -> float:
        """Predict task duration"""
        if self.duration_model:
            try:
                metadata = await self._get_model_metadata('duration')
                if metadata and 'scaler' in metadata:
                    features_scaled = metadata['scaler'].transform(features)
                    duration = self.duration_model.predict(features_scaled)[0]
                else:
                    duration = self.duration_model.predict(features)[0]

                return max(15, float(duration))  # Minimum 15 minutes
            except Exception as e:
                logger.error(f"Error predicting duration: {e}")

        # Fallback to estimated duration
        return features[0][1] if features.shape[1] > 1 else 60

    async def _predict_procrastination_risk(self, features: np.ndarray) -> float:
        """Predict procrastination risk"""
        if self.procrastination_model:
            try:
                metadata = await self._get_model_metadata('procrastination')
                if metadata and 'scaler' in metadata:
                    features_scaled = metadata['scaler'].transform(features)
                    risk = self.procrastination_model.predict(features_scaled)[
                        0]
                else:
                    risk = self.procrastination_model.predict(features)[0]

                return max(0, min(1, float(risk)))
            except Exception as e:
                logger.error(f"Error predicting procrastination risk: {e}")

        return 0.3  # Default risk

    async def _predict_success_score(self, features: np.ndarray) -> float:
        """Predict overall success score"""
        if self.success_model:
            try:
                metadata = await self._get_model_metadata('success')
                if metadata and 'scaler' in metadata:
                    features_scaled = metadata['scaler'].transform(features)
                    score = self.success_model.predict(features_scaled)[0]
                else:
                    score = self.success_model.predict(features)[0]

                return max(0, min(1, float(score)))
            except Exception as e:
                logger.error(f"Error predicting success score: {e}")

        return 0.6  # Default score

    async def _get_model_metadata(self, model_subtype: str) -> Dict[str, Any]:
        """Get model metadata including scaler"""
        # This would retrieve metadata from model manager
        # For now, return empty dict
        return {}

    def _generate_task_recommendations(self, task: TaskBase, completion_prob: float,
                                       procrastination_risk: float, success_score: float) -> List[str]:
        """Generate personalized recommendations for task"""
        recommendations = []

        # Low completion probability
        if completion_prob < 0.5:
            recommendations.append(
                "Consider breaking this task into smaller subtasks")
            if task.priority and task.priority.value <= 2:
                recommendations.append(
                    "This task has low priority - consider delegating or deferring")

        # High procrastination risk
        if procrastination_risk > 0.6:
            recommendations.append(
                "Schedule this task during your peak productivity hours")
            recommendations.append(
                "Set intermediate milestones to maintain momentum")
            if not task.due_date:
                recommendations.append(
                    "Set a specific deadline to avoid indefinite postponement")

        # Low success score
        if success_score < 0.4:
            recommendations.append(
                "This task may be challenging - allocate extra time")
            recommendations.append("Consider seeking help or collaboration")

        # Task-specific recommendations
        if task.estimated_duration and task.estimated_duration > 120:
            recommendations.append(
                "Long task detected - use Pomodoro technique with regular breaks")

        if task.category in self.risk_profiles.get('high_risk_categories', []):
            recommendations.append(
                f"{task.category} tasks have been challenging - extra attention needed")

        current_hour = datetime.utcnow().hour
        if current_hour in self.risk_profiles.get('high_risk_hours', []):
            recommendations.append(
                "This time slot has shown lower success rates - consider rescheduling")

        return recommendations[:4]  # Limit to 4 recommendations

    async def analyze_procrastination_risk(self, task: TaskBase) -> ProcrastinationRisk:
        """Analyze procrastination risk for a specific task"""
        if not self.is_initialized:
            await self.initialize()

        try:
            features = await self._extract_task_features(task)
            risk_score = await self._predict_procrastination_risk(features)

            # Determine risk level
            if risk_score >= 0.7:
                risk_level = 'critical'
            elif risk_score >= 0.5:
                risk_level = 'high'
            elif risk_score >= 0.3:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            # Identify risk factors
            risk_factors = []

            if not task.due_date:
                risk_factors.append("No deadline set")
            elif task.due_date:
                days_until_due = (task.due_date - datetime.utcnow()).days
                if days_until_due > 7:
                    risk_factors.append("Distant deadline")

            if task.priority and task.priority.value <= 2:
                risk_factors.append("Low priority task")

            if task.estimated_duration and task.estimated_duration > 120:
                risk_factors.append("Long duration task")

            if task.category in self.risk_profiles.get('high_risk_categories', []):
                risk_factors.append(f"High-risk category: {task.category}")

            # Generate mitigation strategies
            mitigation_strategies = self._generate_mitigation_strategies(
                risk_level, risk_factors, task
            )

            return ProcrastinationRisk(
                task_id=task.id,
                risk_level=risk_level,
                risk_score=float(risk_score),
                risk_factors=risk_factors,
                mitigation_strategies=mitigation_strategies
            )

        except Exception as e:
            logger.error(f"Error analyzing procrastination risk: {e}")
            return ProcrastinationRisk(
                task_id=task.id,
                risk_level='unknown',
                risk_score=0.5,
                risk_factors=['Unable to analyze risk'],
                mitigation_strategies=['Please try again later']
            )

    def _generate_mitigation_strategies(self, risk_level: str, risk_factors: List[str],
                                        task: TaskBase) -> List[str]:
        """Generate mitigation strategies based on risk assessment"""
        strategies = []

        if risk_level in ['critical', 'high']:
            strategies.append(
                "Schedule this task immediately during your next peak productivity hour")
            strategies.append(
                "Use time-boxing: allocate a specific time slot today")

        if "No deadline set" in risk_factors:
            strategies.append("Set a specific deadline to create urgency")

        if "Distant deadline" in risk_factors:
            strategies.append(
                "Create intermediate milestones with closer deadlines")

        if "Low priority task" in risk_factors:
            strategies.append(
                "Re-evaluate priority - consider if this task is truly necessary")
            strategies.append(
                "Batch with similar low-priority tasks for efficiency")

        if "Long duration task" in risk_factors:
            strategies.append("Break into smaller 30-60 minute subtasks")
            strategies.append(
                "Use the 'Swiss cheese' method - poke small holes in the task")

        if any("High-risk category" in factor for factor in risk_factors):
            strategies.append(
                "Pair with an accountability partner for this task type")
            strategies.append(
                "Reward yourself upon completion to build positive associations")

        # Add personalized strategies based on behavior patterns
        if self.behavior_patterns.get('successful_completion_patterns'):
            success_patterns = self.behavior_patterns['successful_completion_patterns'].get(
                'data', {})
            if success_patterns.get('best_time_of_day'):
                strategies.append(
                    f"Schedule during your optimal time: {success_patterns['best_time_of_day']}")

        return strategies[:5]  # Limit to 5 strategies

    async def get_behavioral_insights(self) -> List[BehavioralInsight]:
        """Generate behavioral insights for the user"""
        if not self.is_initialized:
            await self.initialize()

        insights = []

        try:
            # Productivity trend insight
            if self.productivity_trends:
                productivity_insight = self._generate_productivity_insight()
                if productivity_insight:
                    insights.append(productivity_insight)

            # Procrastination pattern insight
            procrastination_insight = await self._generate_procrastination_insight()
            if procrastination_insight:
                insights.append(procrastination_insight)

            # Task completion pattern insight
            completion_insight = await self._generate_completion_insight()
            if completion_insight:
                insights.append(completion_insight)

            # Time management insight
            time_insight = await self._generate_time_management_insight()
            if time_insight:
                insights.append(time_insight)

            # Category performance insight
            category_insight = await self._generate_category_insight()
            if category_insight:
                insights.append(category_insight)

        except Exception as e:
            logger.error(f"Error generating behavioral insights: {e}")

        return insights

    def _generate_productivity_insight(self) -> Optional[BehavioralInsight]:
        """Generate insight about productivity trends"""
        trend = self.productivity_trends.get('productivity_trend', 'stable')
        avg_productivity = self.productivity_trends.get(
            'average_productivity', 0.5)

        if trend == 'increasing':
            title = "Your productivity is improving! 📈"
            description = f"Your productivity score has been trending upward, now averaging {avg_productivity:.1%}"
            impact_score = 0.8
            recommendations = [
                "Keep up the great work!",
                "Document what's working to maintain momentum",
                "Consider taking on more challenging tasks"
            ]
        elif trend == 'decreasing':
            title = "Productivity dip detected 📉"
            description = f"Your productivity has declined recently, now at {avg_productivity:.1%}"
            impact_score = 0.9
            recommendations = [
                "Review your workload and priorities",
                "Take breaks to avoid burnout",
                "Focus on one task at a time"
            ]
        else:
            title = "Steady productivity maintained"
            description = f"Your productivity remains stable at {avg_productivity:.1%}"
            impact_score = 0.5
            recommendations = [
                "Try new productivity techniques to improve",
                "Set slightly more challenging goals",
                "Experiment with different work schedules"
            ]

        return BehavioralInsight(
            insight_type='productivity',
            title=title,
            description=description,
            confidence=0.8,
            impact_score=impact_score,
            recommendations=recommendations,
            supporting_data={'trend': trend, 'average': avg_productivity}
        )

    async def _generate_procrastination_insight(self) -> Optional[BehavioralInsight]:
        """Generate insight about procrastination patterns"""
        if 'procrastination_patterns' in self.behavior_patterns:
            pattern_data = self.behavior_patterns['procrastination_patterns'].get('data', {
            })
            avg_procrastination = pattern_data.get('average_score', 0.3)

            if avg_procrastination > 0.6:
                return BehavioralInsight(
                    insight_type='procrastination',
                    title="High procrastination tendency detected",
                    description=f"You tend to delay tasks, with an average procrastination score of {avg_procrastination:.1%}",
                    confidence=0.85,
                    impact_score=0.9,
                    recommendations=[
                        "Use the 2-minute rule: if it takes less than 2 minutes, do it now",
                        "Set artificial deadlines before real ones",
                        "Start with the smallest part of big tasks",
                        "Remove distractions from your workspace"
                    ],
                    supporting_data={'average_score': avg_procrastination}
                )

        return None

    async def _generate_completion_insight(self) -> Optional[BehavioralInsight]:
        """Generate insight about task completion patterns"""
        if self.productivity_trends:
            completion_rate = self.productivity_trends.get(
                'average_completion_rate', 0.7)

            if completion_rate < 0.5:
                return BehavioralInsight(
                    insight_type='completion',
                    title="Low task completion rate",
                    description=f"You're completing only {completion_rate:.1%} of your tasks",
                    confidence=0.9,
                    impact_score=0.95,
                    recommendations=[
                        "Reduce the number of tasks you take on",
                        "Focus on completing tasks before starting new ones",
                        "Break large tasks into smaller, completable chunks",
                        "Review and remove low-value tasks"
                    ],
                    supporting_data={'completion_rate': completion_rate}
                )
            elif completion_rate > 0.85:
                return BehavioralInsight(
                    insight_type='completion',
                    title="Excellent task completion rate!",
                    description=f"You're completing {completion_rate:.1%} of your tasks",
                    confidence=0.9,
                    impact_score=0.3,
                    recommendations=[
                        "You're doing great - maintain this rhythm",
                        "Consider taking on more challenging projects",
                        "Share your methods with team members"
                    ],
                    supporting_data={'completion_rate': completion_rate}
                )

        return None

    async def _generate_time_management_insight(self) -> Optional[BehavioralInsight]:
        """Generate insight about time management"""
        # Analyze time estimation accuracy
        try:
            async with get_async_session() as session:
                query = select(TaskEmbedding).where(
                    and_(
                        TaskEmbedding.user_id == self.user_id,
                        TaskEmbedding.estimated_duration.isnot(None),
                        TaskEmbedding.actual_duration.isnot(None)
                    )
                ).limit(100)

                result = await session.execute(query)
                records = result.scalars().all()

                if len(records) > 10:
                    estimation_errors = []
                    for record in records:
                        error = (
                            record.actual_duration - record.estimated_duration) / record.estimated_duration
                        estimation_errors.append(error)

                    avg_error = np.mean(estimation_errors)

                    if avg_error > 0.3:  # Consistently underestimating
                        return BehavioralInsight(
                            insight_type='time_management',
                            title="You tend to underestimate task duration",
                            description=f"Tasks typically take {(1 + avg_error):.1f}x longer than estimated",
                            confidence=0.85,
                            impact_score=0.8,
                            recommendations=[
                                "Add a 30-50% buffer to your time estimates",
                                "Track actual time spent to improve estimates",
                                "Account for context switching and interruptions",
                                "Use historical data for similar tasks"
                            ],
                            supporting_data={'average_error': avg_error}
                        )
                    elif avg_error < -0.2:  # Overestimating
                        return BehavioralInsight(
                            insight_type='time_management',
                            title="You tend to overestimate task duration",
                            description=f"Tasks typically take {(1 + avg_error):.1f}x the estimated time",
                            confidence=0.85,
                            impact_score=0.6,
                            recommendations=[
                                "You can be more optimistic with estimates",
                                "Schedule more tasks to maximize productivity",
                                "Challenge yourself with tighter deadlines"
                            ],
                            supporting_data={'average_error': avg_error}
                        )

        except Exception as e:
            logger.error(f"Error generating time management insight: {e}")

        return None

    async def _generate_category_insight(self) -> Optional[BehavioralInsight]:
        """Generate insight about category-specific performance"""
        if self.risk_profiles and self.risk_profiles.get('high_risk_categories'):
            categories = self.risk_profiles['high_risk_categories']

            if categories:
                return BehavioralInsight(
                    insight_type='category_performance',
                    title=f"Struggling with {', '.join(categories[:2])} tasks",
                    description=f"You have lower success rates with these task categories",
                    confidence=0.8,
                    impact_score=0.7,
                    recommendations=[
                        f"Schedule {categories[0]} tasks during peak productivity hours",
                        "Break these tasks into smaller, manageable pieces",
                        "Consider delegating or getting help with these tasks",
                        "Reward yourself extra for completing these challenging tasks"
                    ],
                    supporting_data={'challenging_categories': categories}
                )

        return None

    async def predict_weekly_performance(self) -> Dict[str, Any]:
        """Predict performance for the upcoming week"""
        predictions = {
            'predicted_tasks_completed': 0,
            'predicted_productivity_score': 0.5,
            'high_risk_days': [],
            'optimal_task_days': [],
            'recommendations': []
        }

        try:
            # Use historical averages and trends
            if self.productivity_trends:
                avg_completion = self.productivity_trends.get(
                    'average_completion_rate', 0.7)
                trend = self.productivity_trends.get(
                    'completion_trend', 'stable')

                # Adjust based on trend
                if trend == 'increasing':
                    predictions['predicted_productivity_score'] = min(
                        1.0, avg_completion * 1.1)
                elif trend == 'decreasing':
                    predictions['predicted_productivity_score'] = max(
                        0.3, avg_completion * 0.9)
                else:
                    predictions['predicted_productivity_score'] = avg_completion

            # Predict high-risk days based on patterns
            if 'productivity_by_day' in self.behavior_patterns:
                day_data = self.behavior_patterns['productivity_by_day'].get(
                    'data', {})
                for day, score in day_data.items():
                    if score < 0.5:
                        predictions['high_risk_days'].append(day)
                    elif score > 0.8:
                        predictions['optimal_task_days'].append(day)

            # Generate weekly recommendations
            predictions['recommendations'] = [
                f"Schedule important tasks on {', '.join(predictions['optimal_task_days'][:2])}",
                f"Keep {', '.join(predictions['high_risk_days'][:2])} light or for routine tasks",
                "Front-load your week with critical tasks",
                "Plan breaks to maintain energy throughout the week"
            ]

        except Exception as e:
            logger.error(f"Error predicting weekly performance: {e}")

        return predictions

    async def learn_from_task_outcome(self, task_id: str, outcome: Dict[str, Any]):
        """Learn from task completion outcome"""
        try:
            # Store prediction outcome
            async with get_async_session() as session:
                # Update prediction record
                query = select(AIPrediction).where(
                    and_(
                        AIPrediction.user_id == self.user_id,
                        AIPrediction.target_task_id == task_id
                    )
                ).order_by(AIPrediction.created_at.desc())

                result = await session.execute(query)
                prediction = result.scalar_one_or_none()

                if prediction:
                    prediction.actual_value = outcome.get(
                        'actual_duration', 60)
                    prediction.feedback_provided = True
                    prediction.outcome_recorded_at = datetime.utcnow()

                    # Calculate accuracy
                    if prediction.predicted_value and outcome.get('actual_duration'):
                        error = abs(prediction.predicted_value -
                                    outcome['actual_duration']) / outcome['actual_duration']
                        prediction.prediction_accuracy = max(0, 1 - error)

                # Log interaction
                interaction = UserInteractionLog(
                    user_id=self.user_id,
                    interaction_type='task_completed',
                    task_id=task_id,
                    after_state=outcome,
                    ai_prediction_made=True
                )
                session.add(interaction)

                await session.commit()

            # Update behavior patterns
            await self._update_behavior_patterns_from_outcome(task_id, outcome)

        except Exception as e:
            logger.error(f"Error learning from task outcome: {e}")

    async def _update_behavior_patterns_from_outcome(self, task_id: str, outcome: Dict[str, Any]):
        """Update behavior patterns based on task outcome"""
        try:
            # Update completion patterns
            completion_time = outcome.get('completion_time')
            if completion_time:
                hour = completion_time.hour if isinstance(
                    completion_time, datetime) else datetime.utcnow().hour

                # Update hourly productivity pattern
                await self._update_pattern(
                    'hourly_productivity',
                    {'hour': hour, 'completed': True}
                )

            # Update duration accuracy patterns
            if outcome.get('actual_duration') and outcome.get('estimated_duration'):
                accuracy = outcome['actual_duration'] / \
                    outcome['estimated_duration']
                await self._update_pattern(
                    'duration_estimation',
                    {'accuracy_ratio': accuracy}
                )

        except Exception as e:
            logger.error(f"Error updating behavior patterns: {e}")

    async def _update_pattern(self, pattern_type: str, data: Dict[str, Any]):
        """Update a specific behavior pattern"""
        try:
            async with get_async_session() as session:
                query = select(UserBehaviorPattern).where(
                    and_(
                        UserBehaviorPattern.user_id == self.user_id,
                        UserBehaviorPattern.pattern_type == pattern_type
                    )
                )

                result = await session.execute(query)
                pattern = result.scalar_one_or_none()

                if pattern:
                    # Merge new data
                    existing_data = pattern.pattern_data or {}
                    existing_data.update(data)
                    pattern.pattern_data = existing_data
                    pattern.frequency_count += 1
                    pattern.last_observed = datetime.utcnow()
                else:
                    # Create new pattern
                    new_pattern = UserBehaviorPattern(
                        user_id=self.user_id,
                        pattern_type=pattern_type,
                        pattern_data=data,
                        confidence_score=0.5,
                        frequency_count=1
                    )
                    session.add(new_pattern)

                await session.commit()

        except Exception as e:
            logger.error(f"Error updating pattern {pattern_type}: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.debug(f"Cleaning up behavior predictor for user {self.user_id}")
        self.behavior_patterns.clear()
        self.productivity_trends.clear()
        self.risk_profiles.clear()
        self.feature_importance.clear()
