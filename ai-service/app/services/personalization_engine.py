# ai-service/app/services/personalization_engine.py - Part 1
"""
Advanced Personalization Engine for TaskMaster Pro
Production-ready personalization with machine learning, behavior analysis, and adaptive recommendations
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
import json
import pickle
import hashlib
from pathlib import Path
from enum import Enum
import statistics

# ML Libraries
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import silhouette_score, accuracy_score
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning(
        "⚠️ ML libraries not available, using simplified algorithms")

# Database
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.sql import func

# Internal imports
from app.core.config import get_settings
from app.core.database import get_async_session
from app.schemas.ai_schemas import (
    PersonalizationMetrics, UserInsight, BehaviorPattern,
    AdaptationRecommendation
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Enums and Constants


class PersonalityType(Enum):
    """User personality types for task management"""
    ACHIEVER = "achiever"          # Goal-oriented, competitive
    EXPLORER = "explorer"          # Curious, variety-seeking
    SOCIALIZER = "socializer"      # Collaboration-focused
    PERFECTIONIST = "perfectionist"  # Detail-oriented, quality-focused
    PRAGMATIST = "pragmatist"      # Efficiency-focused, practical


class WorkStyle(Enum):
    """Work style preferences"""
    DEEP_FOCUS = "deep_focus"      # Long, uninterrupted work sessions
    BURST_WORKER = "burst_worker"  # Short, intense work bursts
    STEADY_PACER = "steady_pacer"  # Consistent, moderate pace
    FLEXIBLE = "flexible"          # Adapts to situation


class MotivationType(Enum):
    """What motivates the user"""
    AUTONOMY = "autonomy"          # Freedom and control
    MASTERY = "mastery"           # Learning and improvement
    PURPOSE = "purpose"           # Meaningful impact
    PROGRESS = "progress"         # Visible advancement
    RECOGNITION = "recognition"   # Acknowledgment and praise


@dataclass
class UserPersonality:
    """Comprehensive user personality profile"""
    user_id: str
    personality_type: PersonalityType = PersonalityType.PRAGMATIST
    work_style: WorkStyle = WorkStyle.FLEXIBLE
    motivation_type: MotivationType = MotivationType.PROGRESS

    # Behavioral tendencies (0-1 scale)
    procrastination_tendency: float = 0.5
    perfectionism_score: float = 0.5
    time_optimism: float = 0.5          # Tendency to underestimate time
    stress_tolerance: float = 0.5
    multitasking_preference: float = 0.5
    planning_preference: float = 0.5     # Likes detailed planning vs spontaneous
    collaboration_preference: float = 0.5

    # Time preferences
    peak_energy_hours: List[int] = field(
        default_factory=lambda: [9, 10, 14, 15])
    preferred_work_duration: int = 90    # Minutes
    break_frequency: int = 60           # Minutes between breaks

    # Task preferences
    preferred_task_types: List[str] = field(default_factory=list)
    avoided_task_types: List[str] = field(default_factory=list)
    complexity_preference: float = 0.5   # Prefers simple vs complex tasks

    # Metadata
    confidence_score: float = 0.5       # How confident we are in this profile
    last_updated: datetime = field(default_factory=datetime.now)
    sample_size: int = 0                # Number of data points used


@dataclass
class BehaviorSnapshot:
    """Snapshot of user behavior metrics"""
    timestamp: datetime
    completion_rate: float
    punctuality_score: float
    productivity_score: float
    stress_level: float
    focus_duration_avg: float
    task_switching_frequency: float

# ai-service/app/services/personalization_engine.py - Part 2


class PersonalizationEngine:
    """
    Advanced personalization engine with machine learning capabilities

    Features:
    - Real-time behavior analysis and pattern recognition
    - Adaptive personality profiling with confidence tracking
    - Predictive modeling for task success and timing
    - Dynamic recommendation generation
    - Continuous learning from user feedback
    - A/B testing for recommendation optimization
    """

    def __init__(self):
        # Core data structures
        self.user_personalities: Dict[str, UserPersonality] = {}
        self.behavior_history: Dict[str,
                                    List[BehaviorSnapshot]] = defaultdict(list)
        self.interaction_patterns: Dict[str,
                                        Dict[str, Any]] = defaultdict(dict)

        # Machine learning models
        self.personality_classifier = None
        self.behavior_predictor = None
        self.recommendation_scorer = None

        # Pattern detection
        self.pattern_detectors = {}
        self.behavioral_clusters = {}

        # Caching and performance
        self.insights_cache: Dict[str, Tuple[List[UserInsight], datetime]] = {}
        self.predictions_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(hours=2)

        # Configuration
        self.model_cache_dir = Path(
            settings.model_cache_dir) / "personalization"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Analytics and optimization
        self.recommendation_feedback: Dict[str,
                                           List[Dict[str, Any]]] = defaultdict(list)
        self.a_b_tests: Dict[str, Dict[str, Any]] = {}

        # Adaptation parameters
        self.adaptation_sensitivity = 0.1    # How quickly to adapt to new data
        self.min_samples_for_adaptation = 10  # Minimum samples before major changes
        self.stability_threshold = 0.8       # Confidence threshold for stable predictions

        logger.info("🧠 Advanced Personalization Engine initializing...")

    async def initialize(self):
        """Initialize the personalization engine with all components"""
        try:
            logger.info("🚀 Initializing Personalization Engine components...")

            # Load existing data
            await self._load_user_data()

            # Initialize ML models
            if ML_AVAILABLE:
                await self._initialize_ml_models()
            else:
                await self._initialize_rule_based_models()

            # Initialize pattern detectors
            await self._initialize_pattern_detectors()

            # Load behavioral clusters
            await self._load_behavioral_clusters()

            # Start background tasks
            asyncio.create_task(self._background_model_training())
            asyncio.create_task(self._periodic_cache_cleanup())

            logger.info("✅ Personalization Engine fully initialized")

        except Exception as e:
            logger.error(
                f"❌ Personalization Engine initialization failed: {e}")
            # Initialize with minimal functionality
            await self._initialize_fallback_mode()

    async def _load_user_data(self):
        """Load existing user personalities and behavior data"""
        try:
            # Load personalities
            personalities_file = self.model_cache_dir / "user_personalities.pkl"
            if personalities_file.exists():
                with open(personalities_file, 'rb') as f:
                    data = pickle.load(f)
                    self.user_personalities = data.get('personalities', {})
                    self.behavior_history = data.get(
                        'behavior_history', defaultdict(list))
                logger.info(
                    f"📊 Loaded data for {len(self.user_personalities)} users")

            # Load interaction patterns
            patterns_file = self.model_cache_dir / "interaction_patterns.pkl"
            if patterns_file.exists():
                with open(patterns_file, 'rb') as f:
                    self.interaction_patterns = pickle.load(f)
                logger.info(
                    f"🔍 Loaded interaction patterns for {len(self.interaction_patterns)} users")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load user data: {e}")
            self.user_personalities = {}
            self.behavior_history = defaultdict(list)
            self.interaction_patterns = defaultdict(dict)

    async def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Personality classification model
            self.personality_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )

            # Behavior prediction model
            self.behavior_predictor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

            # Recommendation scoring model
            self.recommendation_scorer = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )

            # Load pre-trained models if available
            await self._load_trained_models()

            logger.info("🤖 ML models initialized")

        except Exception as e:
            logger.error(f"❌ ML model initialization failed: {e}")
            await self._initialize_rule_based_models()

    async def _initialize_rule_based_models(self):
        """Initialize rule-based models as fallback"""
        try:
            self.personality_classifier = None
            self.behavior_predictor = None
            self.recommendation_scorer = None

            # Use simple heuristic-based approaches
            logger.info("📝 Rule-based models initialized as fallback")

        except Exception as e:
            logger.error(f"❌ Rule-based model initialization failed: {e}")

    async def _initialize_pattern_detectors(self):
        """Initialize behavior pattern detection algorithms"""
        self.pattern_detectors = {
            'productivity_rhythm': self._detect_productivity_rhythm,
            'procrastination_pattern': self._detect_procrastination_pattern,
            'focus_pattern': self._detect_focus_patterns,
            'stress_response': self._detect_stress_response,
            'task_switching': self._detect_task_switching_patterns,
            'collaboration_style': self._detect_collaboration_style,
            'learning_curve': self._detect_learning_patterns,
            'energy_management': self._detect_energy_patterns
        }
        logger.info(
            f"🔍 Initialized {len(self.pattern_detectors)} pattern detectors")

    async def _load_behavioral_clusters(self):
        """Load or create behavioral clusters for user segmentation"""
        try:
            clusters_file = self.model_cache_dir / "behavioral_clusters.pkl"
            if clusters_file.exists():
                with open(clusters_file, 'rb') as f:
                    self.behavioral_clusters = pickle.load(f)
                logger.info(
                    f"📊 Loaded {len(self.behavioral_clusters)} behavioral clusters")
            else:
                # Create initial clusters when we have enough data
                await self._create_behavioral_clusters()

        except Exception as e:
            logger.warning(f"⚠️ Failed to load behavioral clusters: {e}")
            self.behavioral_clusters = {}

    async def _initialize_fallback_mode(self):
        """Initialize with minimal functionality when full initialization fails"""
        self.user_personalities = {}
        self.behavior_history = defaultdict(list)
        self.pattern_detectors = {
            'basic_productivity': self._basic_productivity_detection
        }
        logger.info("⚡ Fallback mode initialized")


# ai-service/app/services/personalization_engine.py - Part 3


    async def analyze_user_behavior(
        self,
        user_id: str,
        task_history: List[Dict[str, Any]],
        interaction_data: Optional[Dict[str, Any]] = None
    ) -> List[UserInsight]:
        """
        Comprehensive user behavior analysis with advanced pattern recognition

        Args:
            user_id: User identifier
            task_history: List of user's task data
            interaction_data: Additional interaction data (clicks, time spent, etc.)

        Returns:
            List of personalized insights
        """
        try:
            start_time = datetime.now()

            # Check cache first
            cache_key = self._generate_cache_key(
                "insights", user_id, task_history)
            cached_insights = self._get_cached_result(
                cache_key, self.insights_cache)
            if cached_insights:
                logger.debug(f"🎯 Using cached insights for user {user_id}")
                return cached_insights

            logger.info(
                f"🔍 Analyzing behavior for user {user_id} with {len(task_history)} tasks")

            # Update behavior metrics
            behavior_snapshot = await self._calculate_behavior_snapshot(user_id, task_history)
            self.behavior_history[user_id].append(behavior_snapshot)

            # Limit history size
            if len(self.behavior_history[user_id]) > 100:
                self.behavior_history[user_id] = self.behavior_history[user_id][-100:]

            # Update interaction patterns
            if interaction_data:
                await self._update_interaction_patterns(user_id, interaction_data)

            # Detect behavior patterns
            detected_patterns = await self._detect_all_patterns(user_id, task_history)

            # Update personality profile
            await self._update_personality_profile(user_id, detected_patterns, task_history)

            # Generate insights from patterns and personality
            insights = await self._generate_comprehensive_insights(
                user_id, detected_patterns, behavior_snapshot
            )

            # Add predictive insights
            predictive_insights = await self._generate_predictive_insights(user_id, task_history)
            insights.extend(predictive_insights)

            # Cache results
            self._cache_result(cache_key, insights, self.insights_cache)

            # Record analytics
            analysis_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"✅ Behavior analysis completed for user {user_id} in {analysis_time:.2f}s")

            return insights

        except Exception as e:
            logger.error(f"❌ Behavior analysis failed for user {user_id}: {e}")
            return await self._generate_fallback_insights(user_id)

    async def _calculate_behavior_snapshot(
        self,
        user_id: str,
        task_history: List[Dict[str, Any]]
    ) -> BehaviorSnapshot:
        """Calculate current behavior metrics snapshot"""
        try:
            if not task_history:
                return BehaviorSnapshot(
                    timestamp=datetime.now(),
                    completion_rate=0.0,
                    punctuality_score=0.0,
                    productivity_score=0.0,
                    stress_level=0.5,
                    focus_duration_avg=60.0,
                    task_switching_frequency=0.0
                )

            # Calculate completion rate
            completed_tasks = [
                t for t in task_history if t.get('status') == 'DONE']
            completion_rate = len(completed_tasks) / len(task_history)

            # Calculate punctuality score
            on_time_tasks = [
                t for t in completed_tasks
                if self._was_completed_on_time(t)
            ]
            punctuality_score = len(
                on_time_tasks) / len(completed_tasks) if completed_tasks else 0.0

            # Calculate productivity score (efficiency metric)
            productivity_score = self._calculate_productivity_score(
                task_history)

            # Calculate stress level indicators
            stress_level = self._calculate_stress_level(task_history)

            # Calculate focus patterns
            focus_duration_avg = self._calculate_average_focus_duration(
                task_history)
            task_switching_frequency = self._calculate_task_switching_frequency(
                task_history)

            return BehaviorSnapshot(
                timestamp=datetime.now(),
                completion_rate=completion_rate,
                punctuality_score=punctuality_score,
                productivity_score=productivity_score,
                stress_level=stress_level,
                focus_duration_avg=focus_duration_avg,
                task_switching_frequency=task_switching_frequency
            )

        except Exception as e:
            logger.error(f"❌ Failed to calculate behavior snapshot: {e}")
            return BehaviorSnapshot(
                timestamp=datetime.now(),
                completion_rate=0.5,
                punctuality_score=0.5,
                productivity_score=0.5,
                stress_level=0.5,
                focus_duration_avg=60.0,
                task_switching_frequency=1.0
            )

    def _was_completed_on_time(self, task: Dict[str, Any]) -> bool:
        """Check if task was completed on time"""
        try:
            due_date = task.get('due_date')
            completed_at = task.get('completed_at')

            if not due_date or not completed_at:
                return True  # Assume on time if no deadline

            due = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            completed = datetime.fromisoformat(
                completed_at.replace('Z', '+00:00'))

            return completed <= due

        except Exception:
            return True

    def _calculate_productivity_score(self, task_history: List[Dict[str, Any]]) -> float:
        """Calculate productivity score based on estimated vs actual time"""
        try:
            efficiency_scores = []

            for task in task_history:
                if task.get('status') != 'DONE':
                    continue

                estimated = task.get('estimated_duration', 60)
                actual = task.get('actual_duration')

                if actual and estimated > 0:
                    # Efficiency = estimated / actual (capped at 2.0)
                    efficiency = min(2.0, estimated / actual)
                    efficiency_scores.append(efficiency)

            if efficiency_scores:
                return statistics.mean(efficiency_scores)

            return 0.5  # Default neutral score

        except Exception as e:
            logger.error(f"❌ Failed to calculate productivity score: {e}")
            return 0.5

    def _calculate_stress_level(self, task_history: List[Dict[str, Any]]) -> float:
        """Calculate stress level from task patterns"""
        try:
            stress_indicators = []

            # Check for deadline pressure
            deadline_pressure = self._calculate_deadline_pressure(task_history)
            stress_indicators.append(deadline_pressure)

            # Check for workload spikes
            workload_variance = self._calculate_workload_variance(task_history)
            stress_indicators.append(workload_variance)

            # Check for incomplete task accumulation
            incomplete_ratio = len([t for t in task_history if t.get(
                'status') != 'DONE']) / len(task_history)
            stress_indicators.append(incomplete_ratio)

            return statistics.mean(stress_indicators)

        except Exception as e:
            logger.error(f"❌ Failed to calculate stress level: {e}")
            return 0.5

    def _calculate_deadline_pressure(self, task_history: List[Dict[str, Any]]) -> float:
        """Calculate deadline pressure indicator"""
        try:
            pressure_scores = []

            for task in task_history:
                if not task.get('due_date') or task.get('status') != 'DONE':
                    continue

                try:
                    due_date = datetime.fromisoformat(
                        task['due_date'].replace('Z', '+00:00'))
                    completed_date = datetime.fromisoformat(
                        task.get('completed_at', task['due_date']).replace('Z', '+00:00'))
                    created_date = datetime.fromisoformat(
                        task.get('created_at', task['due_date']).replace('Z', '+00:00'))

                    total_time = (due_date - created_date).total_seconds()
                    used_time = (completed_date - created_date).total_seconds()

                    if total_time > 0:
                        time_pressure = used_time / total_time
                        pressure_scores.append(time_pressure)

                except Exception:
                    continue

            if pressure_scores:
                # High scores indicate last-minute completions (stress)
                return statistics.mean(pressure_scores)

            return 0.5

        except Exception as e:
            logger.error(f"❌ Failed to calculate deadline pressure: {e}")
            return 0.5

    def _calculate_workload_variance(self, task_history: List[Dict[str, Any]]) -> float:
        """Calculate workload variance as stress indicator"""
        try:
            # Group tasks by day
            daily_counts = defaultdict(int)

            for task in task_history:
                created_at = task.get('created_at')
                if created_at:
                    try:
                        date = datetime.fromisoformat(
                            created_at.replace('Z', '+00:00')).date()
                        daily_counts[date] += 1
                    except Exception:
                        continue

            if len(daily_counts) > 1:
                counts = list(daily_counts.values())
                variance = statistics.variance(counts)
                mean_count = statistics.mean(counts)

                # Coefficient of variation as stress indicator
                if mean_count > 0:
                    cv = (variance ** 0.5) / mean_count
                    return min(1.0, cv / 2.0)  # Normalize

            return 0.0

        except Exception as e:
            logger.error(f"❌ Failed to calculate workload variance: {e}")
            return 0.0

    def _calculate_average_focus_duration(self, task_history: List[Dict[str, Any]]) -> float:
        """Calculate average focus duration from task data"""
        try:
            durations = []

            for task in task_history:
                if task.get('status') == 'DONE':
                    duration = task.get('actual_duration') or task.get(
                        'estimated_duration', 60)
                    durations.append(duration)

            if durations:
                return statistics.mean(durations)

            return 60.0  # Default 1 hour

        except Exception as e:
            logger.error(f"❌ Failed to calculate focus duration: {e}")
            return 60.0

    def _calculate_task_switching_frequency(self, task_history: List[Dict[str, Any]]) -> float:
        """Calculate how often user switches between tasks"""
        try:
            # Simple approximation: number of tasks / number of unique days
            unique_days = set()

            for task in task_history:
                created_at = task.get('created_at')
                if created_at:
                    try:
                        date = datetime.fromisoformat(
                            created_at.replace('Z', '+00:00')).date()
                        unique_days.add(date)
                    except Exception:
                        continue

            if unique_days:
                return len(task_history) / len(unique_days)

            return 1.0

        except Exception as e:
            logger.error(
                f"❌ Failed to calculate task switching frequency: {e}")
            return 1.0

    async def _update_interaction_patterns(self, user_id: str, interaction_data: Dict[str, Any]):
        """Update user interaction patterns"""
        try:
            patterns = self.interaction_patterns[user_id]

            # Update click patterns
            if 'clicks' in interaction_data:
                patterns.setdefault('click_heatmap', {})
                for element, count in interaction_data['clicks'].items():
                    patterns['click_heatmap'][element] = patterns['click_heatmap'].get(
                        element, 0) + count

            # Update time spent patterns
            if 'time_spent' in interaction_data:
                patterns.setdefault('session_durations', [])
                patterns['session_durations'].append(
                    interaction_data['time_spent'])

                # Keep only last 100 sessions
                if len(patterns['session_durations']) > 100:
                    patterns['session_durations'] = patterns['session_durations'][-100:]

            # Update feature usage
            if 'features_used' in interaction_data:
                patterns.setdefault('feature_usage', Counter())
                for feature in interaction_data['features_used']:
                    patterns['feature_usage'][feature] += 1

        except Exception as e:
            logger.error(f"❌ Failed to update interaction patterns: {e}")

    async def _detect_all_patterns(self, user_id: str, task_history: List[Dict[str, Any]]) -> List[BehaviorPattern]:
        """Detect all behavior patterns for a user"""
        patterns = []

        for pattern_name, detector_func in self.pattern_detectors.items():
            try:
                pattern = await detector_func(user_id, task_history)
                if pattern:
                    patterns.append(pattern)

            except Exception as e:
                logger.error(
                    f"❌ Pattern detection failed for {pattern_name}: {e}")

        return patterns

# ai-service/app/services/personalization_engine.py - Part 4

    # Pattern Detection Methods
    async def _detect_productivity_rhythm(self, user_id: str, task_history: List[Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Detect user's productivity rhythm patterns"""
        try:
            completed_tasks = [
                t for t in task_history if t.get('status') == 'DONE']
            if len(completed_tasks) < 5:
                return None

            # Analyze completion times by hour
            hour_productivity = defaultdict(list)

            for task in completed_tasks:
                completed_at = task.get('completed_at')
                if completed_at:
                    try:
                        dt = datetime.fromisoformat(
                            completed_at.replace('Z', '+00:00'))
                        hour = dt.hour

                        # Calculate productivity metric
                        estimated = task.get('estimated_duration', 60)
                        actual = task.get('actual_duration', estimated)
                        productivity = estimated / max(actual, 1)

                        hour_productivity[hour].append(productivity)
                    except Exception:
                        continue

            if not hour_productivity:
                return None

            # Find peak hours
            hour_averages = {h: statistics.mean(
                scores) for h, scores in hour_productivity.items()}
            peak_hours = sorted(hour_averages.items(),
                                key=lambda x: x[1], reverse=True)[:4]

            # Determine rhythm type
            peak_hour_list = [h for h, _ in peak_hours]
            morning_count = sum(1 for h in peak_hour_list if 6 <= h <= 11)
            afternoon_count = sum(1 for h in peak_hour_list if 12 <= h <= 17)
            evening_count = sum(1 for h in peak_hour_list if 18 <= h <= 23)

            if morning_count >= 2:
                rhythm_type = "morning_person"
                description = f"Peak productivity in morning hours: {', '.join(map(str, sorted(peak_hour_list)))}"
            elif evening_count >= 2:
                rhythm_type = "night_owl"
                description = f"Peak productivity in evening hours: {', '.join(map(str, sorted(peak_hour_list)))}"
            else:
                rhythm_type = "flexible"
                description = f"Flexible productivity pattern across: {', '.join(map(str, sorted(peak_hour_list)))}"

            # Calculate confidence based on data consistency
            productivity_values = list(hour_averages.values())
            confidence = min(
                0.9, 1.0 - (statistics.stdev(productivity_values) / statistics.mean(productivity_values)))

            return BehaviorPattern(
                pattern_type="productivity_rhythm",
                description=description,
                frequency=1.0,
                impact_score=0.8,
                detected_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"❌ Productivity rhythm detection failed: {e}")
            return None

# ai-service/app/services/personalization_engine.py - Part 4 (Continued)

    async def _detect_procrastination_pattern(self, user_id: str, task_history: List[Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Detect user's procrastination patterns and triggers"""
        try:
            completed_tasks = [t for t in task_history if t.get(
                'status') == 'DONE' and t.get('due_date')]
            if len(completed_tasks) < 5:
                return None

            delays = []
            rush_completions = 0

            for task in completed_tasks:
                try:
                    due_date = datetime.fromisoformat(
                        task['due_date'].replace('Z', '+00:00'))
                    completed_at = datetime.fromisoformat(
                        task['completed_at'].replace('Z', '+00:00'))
                    created_at = datetime.fromisoformat(
                        task.get('created_at', task['due_date']).replace('Z', '+00:00'))

                    # Calculate delay (negative = early, positive = late)
                    delay_hours = (
                        completed_at - due_date).total_seconds() / 3600
                    delays.append(delay_hours)

                    # Check for rush completions (completed in last 20% of available time)
                    total_time = (due_date - created_at).total_seconds()
                    time_used = (completed_at - created_at).total_seconds()

                    if total_time > 0 and time_used / total_time > 0.8:
                        rush_completions += 1

                except Exception:
                    continue

            if not delays:
                return None

            avg_delay = statistics.mean(delays)
            late_rate = sum(1 for d in delays if d > 0) / len(delays)
            rush_rate = rush_completions / len(completed_tasks)

            # Classify procrastination pattern
            if late_rate > 0.6 or avg_delay > 12:  # 12+ hours average delay
                severity = "high" if late_rate > 0.8 else "moderate"
                pattern_description = f"Frequent late task completion ({late_rate:.1%} late, avg delay: {avg_delay:.1f}h)"

                if rush_rate > 0.7:
                    pattern_description += ". Tends to complete tasks in final moments."

                return BehaviorPattern(
                    pattern_type="procrastination_chronic",
                    description=pattern_description,
                    frequency=late_rate,
                    impact_score=0.8 if severity == "high" else 0.6,
                    detected_at=datetime.now()
                )

            elif rush_rate > 0.6:  # Deadline pressure pattern
                return BehaviorPattern(
                    pattern_type="deadline_pressure",
                    description=f"Works well under pressure ({rush_rate:.1%} of tasks completed in final 20% of time)",
                    frequency=rush_rate,
                    impact_score=0.5,
                    detected_at=datetime.now()
                )

            return None

        except Exception as e:
            logger.error(f"❌ Procrastination pattern detection failed: {e}")
            return None

    async def _detect_focus_patterns(self, user_id: str, task_history: List[Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Detect user's focus and attention patterns"""
        try:
            work_sessions = []

            for task in task_history:
                if task.get('status') == 'DONE':
                    duration = task.get('actual_duration') or task.get(
                        'estimated_duration')
                    if duration:
                        work_sessions.append(duration)

            if len(work_sessions) < 10:
                return None

            avg_duration = statistics.mean(work_sessions)
            session_variance = statistics.variance(
                work_sessions) if len(work_sessions) > 1 else 0

            # Categorize focus patterns
            if avg_duration > 120:  # 2+ hours
                if session_variance < avg_duration * 0.3:  # Low variance
                    pattern_type = "deep_focus_sustained"
                    description = f"Sustained deep focus sessions (avg: {avg_duration:.0f} min)"
                else:
                    pattern_type = "deep_focus_variable"
                    description = f"Variable deep focus with some very long sessions (avg: {avg_duration:.0f} min)"

            elif avg_duration < 30:  # Less than 30 minutes
                pattern_type = "short_burst"
                description = f"Prefers short work bursts (avg: {avg_duration:.0f} min)"

            else:  # 30-120 minutes
                pattern_type = "moderate_focus"
                description = f"Moderate focus sessions (avg: {avg_duration:.0f} min)"

            return BehaviorPattern(
                pattern_type=pattern_type,
                description=description,
                frequency=1.0,
                impact_score=0.7,
                detected_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"❌ Focus pattern detection failed: {e}")
            return None

    async def _detect_stress_response(self, user_id: str, task_history: List[Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Detect how user responds to stress and high workload"""
        try:
            # Group tasks by week to identify busy periods
            weekly_data = defaultdict(list)

            for task in task_history:
                created_at = task.get('created_at')
                if created_at:
                    try:
                        date = datetime.fromisoformat(
                            created_at.replace('Z', '+00:00'))
                        week = date.isocalendar()[:2]  # (year, week)
                        weekly_data[week].append(task)
                    except Exception:
                        continue

            if len(weekly_data) < 4:  # Need at least 4 weeks of data
                return None

            # Calculate weekly metrics
            weekly_metrics = []
            for week, tasks in weekly_data.items():
                task_count = len(tasks)
                completed_count = len(
                    [t for t in tasks if t.get('status') == 'DONE'])
                completion_rate = completed_count / task_count if task_count > 0 else 0

                # Calculate average time to completion
                completion_times = []
                for task in tasks:
                    if task.get('status') == 'DONE' and task.get('created_at') and task.get('completed_at'):
                        try:
                            created = datetime.fromisoformat(
                                task['created_at'].replace('Z', '+00:00'))
                            completed = datetime.fromisoformat(
                                task['completed_at'].replace('Z', '+00:00'))
                            hours_to_complete = (
                                completed - created).total_seconds() / 3600
                            completion_times.append(hours_to_complete)
                        except Exception:
                            continue

                avg_completion_time = statistics.mean(
                    completion_times) if completion_times else 0

                weekly_metrics.append({
                    'week': week,
                    'task_count': task_count,
                    'completion_rate': completion_rate,
                    'avg_completion_time': avg_completion_time
                })

            # Identify stress response patterns
            high_load_weeks = [w for w in weekly_metrics if w['task_count'] > statistics.mean(
                [w['task_count'] for w in weekly_metrics]) * 1.5]

            if len(high_load_weeks) < 2:
                return None

            high_load_completion_rate = statistics.mean(
                [w['completion_rate'] for w in high_load_weeks])
            normal_load_completion_rate = statistics.mean(
                [w['completion_rate'] for w in weekly_metrics if w not in high_load_weeks])

            if high_load_completion_rate > normal_load_completion_rate * 1.1:
                stress_response = "thrives_under_pressure"
                description = f"Performance improves under high workload ({high_load_completion_rate:.1%} vs {normal_load_completion_rate:.1%} completion rate)"
            elif high_load_completion_rate < normal_load_completion_rate * 0.8:
                stress_response = "overwhelmed_by_pressure"
                description = f"Performance decreases under high workload ({high_load_completion_rate:.1%} vs {normal_load_completion_rate:.1%} completion rate)"
            else:
                stress_response = "stable_under_pressure"
                description = f"Maintains consistent performance under pressure ({high_load_completion_rate:.1%} vs {normal_load_completion_rate:.1%} completion rate)"

            return BehaviorPattern(
                pattern_type=stress_response,
                description=description,
                frequency=len(high_load_weeks) / len(weekly_metrics),
                impact_score=0.8,
                detected_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"❌ Stress response detection failed: {e}")
            return None

    async def _detect_task_switching_patterns(self, user_id: str, task_history: List[Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Detect user's task switching and multitasking patterns"""
        try:
            # Sort tasks by creation time
            sorted_tasks = sorted(
                [t for t in task_history if t.get('created_at')],
                key=lambda x: x['created_at']
            )

            if len(sorted_tasks) < 10:
                return None

            # Analyze task switching patterns
            category_switches = 0
            priority_switches = 0
            quick_switches = 0  # Tasks created within 5 minutes of each other

            for i in range(1, len(sorted_tasks)):
                prev_task = sorted_tasks[i-1]
                curr_task = sorted_tasks[i]

                # Time between task creations
                try:
                    prev_time = datetime.fromisoformat(
                        prev_task['created_at'].replace('Z', '+00:00'))
                    curr_time = datetime.fromisoformat(
                        curr_task['created_at'].replace('Z', '+00:00'))
                    time_diff = (
                        curr_time - prev_time).total_seconds() / 60  # minutes

                    if time_diff < 5:
                        quick_switches += 1
                except Exception:
                    continue

                # Category switches
                if prev_task.get('category') != curr_task.get('category'):
                    category_switches += 1

                # Priority switches
                if prev_task.get('priority') != curr_task.get('priority'):
                    priority_switches += 1

            total_transitions = len(sorted_tasks) - 1
            category_switch_rate = category_switches / \
                total_transitions if total_transitions > 0 else 0
            quick_switch_rate = quick_switches / \
                total_transitions if total_transitions > 0 else 0

            # Determine switching pattern
            if quick_switch_rate > 0.3:
                pattern_type = "rapid_task_creation"
                description = f"Frequently creates multiple tasks in quick succession ({quick_switch_rate:.1%} within 5 minutes)"
                impact_score = 0.6
            elif category_switch_rate > 0.7:
                pattern_type = "context_switching"
                description = f"Frequently switches between different task categories ({category_switch_rate:.1%} switches)"
                impact_score = 0.7
            elif category_switch_rate < 0.3:
                pattern_type = "focused_batching"
                description = f"Tends to batch similar tasks together ({category_switch_rate:.1%} category switches)"
                impact_score = 0.5
            else:
                return None  # Normal switching pattern

            return BehaviorPattern(
                pattern_type=pattern_type,
                description=description,
                frequency=max(category_switch_rate, quick_switch_rate),
                impact_score=impact_score,
                detected_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"❌ Task switching pattern detection failed: {e}")
            return None

    async def _detect_collaboration_style(self, user_id: str, task_history: List[Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Detect user's collaboration and delegation patterns"""
        try:
            assigned_tasks = [t for t in task_history if t.get(
                'assigned_to') and t['assigned_to'] != user_id]
            shared_tasks = [t for t in task_history if t.get('shared_with')]
            solo_tasks = [t for t in task_history if not t.get(
                'assigned_to') and not t.get('shared_with')]

            total_tasks = len(task_history)
            if total_tasks < 10:
                return None

            delegation_rate = len(assigned_tasks) / total_tasks
            collaboration_rate = len(shared_tasks) / total_tasks
            solo_rate = len(solo_tasks) / total_tasks

            # Determine collaboration style
            if delegation_rate > 0.3:
                pattern_type = "delegator"
                description = f"Frequently delegates tasks to others ({delegation_rate:.1%} of tasks)"
            elif collaboration_rate > 0.4:
                pattern_type = "collaborator"
                description = f"Prefers collaborative task approach ({collaboration_rate:.1%} shared tasks)"
            elif solo_rate > 0.8:
                pattern_type = "independent_worker"
                description = f"Prefers working independently ({solo_rate:.1%} solo tasks)"
            else:
                pattern_type = "mixed_collaboration"
                description = f"Uses mixed collaboration approach (delegate: {delegation_rate:.1%}, collaborate: {collaboration_rate:.1%}, solo: {solo_rate:.1%})"

            return BehaviorPattern(
                pattern_type=pattern_type,
                description=description,
                frequency=max(delegation_rate, collaboration_rate, solo_rate),
                impact_score=0.6,
                detected_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"❌ Collaboration style detection failed: {e}")
            return None

    async def _detect_learning_patterns(self, user_id: str, task_history: List[Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Detect how user learns and improves over time"""
        try:
            # Group tasks by category to see improvement patterns
            category_performance = defaultdict(list)

            for task in task_history:
                if task.get('status') == 'DONE' and task.get('category'):
                    estimated = task.get('estimated_duration', 60)
                    actual = task.get('actual_duration', estimated)
                    efficiency = estimated / max(actual, 1)

                    try:
                        completed_at = datetime.fromisoformat(
                            task['completed_at'].replace('Z', '+00:00'))
                        category_performance[task['category']].append({
                            'efficiency': efficiency,
                            'date': completed_at
                        })
                    except Exception:
                        continue

            learning_categories = []

            for category, performances in category_performance.items():
                if len(performances) >= 5:  # Need enough data points
                    # Sort by date
                    performances.sort(key=lambda x: x['date'])

                    # Calculate trend in efficiency
                    early_efficiency = statistics.mean(
                        [p['efficiency'] for p in performances[:len(performances)//2]])
                    late_efficiency = statistics.mean(
                        [p['efficiency'] for p in performances[len(performances)//2:]])

                    improvement = (late_efficiency - early_efficiency) / \
                        early_efficiency if early_efficiency > 0 else 0

                    if improvement > 0.2:  # 20% improvement
                        learning_categories.append({
                            'category': category,
                            'improvement': improvement,
                            'tasks': len(performances)
                        })

            if learning_categories:
                # Sort by improvement
                learning_categories.sort(
                    key=lambda x: x['improvement'], reverse=True)
                best_learning = learning_categories[0]

                return BehaviorPattern(
                    pattern_type="learning_improvement",
                    description=f"Shows strong learning curve in {best_learning['category']} tasks ({best_learning['improvement']:.1%} efficiency improvement)",
                    frequency=len(
                        learning_categories) / len(category_performance) if category_performance else 0,
                    impact_score=0.8,
                    detected_at=datetime.now()
                )

            return None

        except Exception as e:
            logger.error(f"❌ Learning pattern detection failed: {e}")
            return None

    async def _detect_energy_patterns(self, user_id: str, task_history: List[Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Detect user's energy management and peak performance patterns"""
        try:
            hourly_performance = defaultdict(list)

            for task in task_history:
                if task.get('status') == 'DONE' and task.get('completed_at'):
                    try:
                        completed_at = datetime.fromisoformat(
                            task['completed_at'].replace('Z', '+00:00'))
                        hour = completed_at.hour

                        # Calculate performance metric
                        estimated = task.get('estimated_duration', 60)
                        actual = task.get('actual_duration', estimated)
                        efficiency = estimated / max(actual, 1)

                        hourly_performance[hour].append(efficiency)
                    except Exception:
                        continue

            if len(hourly_performance) < 6:  # Need data across multiple hours
                return None

            # Calculate average performance by hour
            hour_averages = {hour: statistics.mean(efficiencies)
                             for hour, efficiencies in hourly_performance.items()
                             if len(efficiencies) >= 2}

            if not hour_averages:
                return None

            # Find peak performance hours
            sorted_hours = sorted(hour_averages.items(),
                                  key=lambda x: x[1], reverse=True)
            peak_hours = [hour for hour, _ in sorted_hours[:3]]

            # Classify energy pattern
            morning_hours = [h for h in peak_hours if 6 <= h <= 11]
            afternoon_hours = [h for h in peak_hours if 12 <= h <= 17]
            evening_hours = [h for h in peak_hours if 18 <= h <= 23]

            if len(morning_hours) >= 2:
                energy_type = "morning_energy"
                description = f"Peak performance in morning hours: {', '.join(map(str, sorted(morning_hours)))}"
            elif len(evening_hours) >= 2:
                energy_type = "evening_energy"
                description = f"Peak performance in evening hours: {', '.join(map(str, sorted(evening_hours)))}"
            elif len(afternoon_hours) >= 2:
                energy_type = "afternoon_energy"
                description = f"Peak performance in afternoon hours: {', '.join(map(str, sorted(afternoon_hours)))}"
            else:
                energy_type = "distributed_energy"
                description = f"Consistent performance across day: {', '.join(map(str, sorted(peak_hours)))}"

            # Calculate confidence based on variance
            performances = list(hour_averages.values())
            performance_variance = statistics.variance(
                performances) if len(performances) > 1 else 0
            confidence = min(0.9, 1.0 - (performance_variance / statistics.mean(
                performances)) if statistics.mean(performances) > 0 else 0.5)

            return BehaviorPattern(
                pattern_type=energy_type,
                description=description,
                frequency=confidence,
                impact_score=0.7,
                detected_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"❌ Energy pattern detection failed: {e}")
            return None

    async def _basic_productivity_detection(self, user_id: str, task_history: List[Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Basic productivity pattern detection for fallback mode"""
        try:
            if len(task_history) < 5:
                return None

            completed_tasks = [
                t for t in task_history if t.get('status') == 'DONE']
            completion_rate = len(completed_tasks) / len(task_history)

            if completion_rate > 0.8:
                return BehaviorPattern(
                    pattern_type="high_completion",
                    description=f"High task completion rate ({completion_rate:.1%})",
                    frequency=1.0,
                    impact_score=0.8,
                    detected_at=datetime.now()
                )
            elif completion_rate < 0.4:
                return BehaviorPattern(
                    pattern_type="low_completion",
                    description=f"Low task completion rate ({completion_rate:.1%}) - may need support",
                    frequency=1.0,
                    impact_score=0.8,
                    detected_at=datetime.now()
                )

            return None

        except Exception as e:
            logger.error(f"❌ Basic productivity detection failed: {e}")
            return None


# ai-service/app/services/personalization_engine.py - Part 5


    async def _update_personality_profile(
        self,
        user_id: str,
        patterns: List[BehaviorPattern],
        task_history: List[Dict[str, Any]]
    ):
        """Update user personality profile based on detected patterns"""
        try:
            current_profile = self.user_personalities.get(user_id)

            if not current_profile:
                current_profile = UserPersonality(user_id=user_id)
                self.user_personalities[user_id] = current_profile

            # Update based on patterns
            for pattern in patterns:
                await self._apply_pattern_to_profile(current_profile, pattern)

            # Update based on recent behavior snapshot
            if len(self.behavior_history[user_id]) > 0:
                recent_snapshot = self.behavior_history[user_id][-1]
                await self._apply_snapshot_to_profile(current_profile, recent_snapshot)

            # Increase confidence and sample size
            current_profile.sample_size = len(task_history)
            current_profile.confidence_score = min(
                0.9, 0.3 + (current_profile.sample_size / 100) * 0.6)
            current_profile.last_updated = datetime.now()

            logger.debug(f"🔄 Updated personality profile for user {user_id}")

        except Exception as e:
            logger.error(f"❌ Failed to update personality profile: {e}")

    async def _apply_pattern_to_profile(self, profile: UserPersonality, pattern: BehaviorPattern):
        """Apply a detected pattern to update personality profile"""
        try:
            pattern_type = pattern.pattern_type
            confidence = pattern.frequency * pattern.impact_score
            adaptation_rate = self.adaptation_sensitivity * confidence

            if pattern_type == "procrastination_chronic":
                profile.procrastination_tendency = min(1.0,
                                                       profile.procrastination_tendency + adaptation_rate * 0.5)

            elif pattern_type == "deadline_pressure":
                profile.stress_tolerance = min(1.0,
                                               profile.stress_tolerance + adaptation_rate * 0.3)

            elif pattern_type in ["deep_focus_sustained", "deep_focus_variable"]:
                profile.preferred_work_duration = max(
                    90, profile.preferred_work_duration + 30)
                profile.multitasking_preference = max(0.0,
                                                      profile.multitasking_preference - adaptation_rate * 0.3)

            elif pattern_type == "short_burst":
                profile.preferred_work_duration = min(
                    45, profile.preferred_work_duration - 15)
                profile.multitasking_preference = min(1.0,
                                                      profile.multitasking_preference + adaptation_rate * 0.3)

            elif pattern_type == "thrives_under_pressure":
                profile.stress_tolerance = min(1.0,
                                               profile.stress_tolerance + adaptation_rate * 0.4)

            elif pattern_type == "overwhelmed_by_pressure":
                profile.stress_tolerance = max(0.0,
                                               profile.stress_tolerance - adaptation_rate * 0.4)

            elif pattern_type in ["morning_energy", "morning_person"]:
                profile.peak_energy_hours = [7, 8, 9, 10]

            elif pattern_type in ["evening_energy", "night_owl"]:
                profile.peak_energy_hours = [19, 20, 21, 22]

            elif pattern_type == "learning_improvement":
                profile.complexity_preference = min(1.0,
                                                    profile.complexity_preference + adaptation_rate * 0.2)

            elif pattern_type == "delegator":
                profile.collaboration_preference = min(1.0,
                                                       profile.collaboration_preference + adaptation_rate * 0.4)

            elif pattern_type == "independent_worker":
                profile.collaboration_preference = max(0.0,
                                                       profile.collaboration_preference - adaptation_rate * 0.4)

        except Exception as e:
            logger.error(f"❌ Failed to apply pattern to profile: {e}")

    async def _apply_snapshot_to_profile(self, profile: UserPersonality, snapshot: BehaviorSnapshot):
        """Apply behavior snapshot data to personality profile"""
        try:
            adaptation_rate = self.adaptation_sensitivity

            # Update time optimism based on actual vs estimated durations
            if snapshot.productivity_score < 0.8:  # Taking longer than estimated
                profile.time_optimism = min(1.0,
                                            profile.time_optimism + adaptation_rate * 0.1)
            elif snapshot.productivity_score > 1.2:  # Finishing early
                profile.time_optimism = max(0.0,
                                            profile.time_optimism - adaptation_rate * 0.1)

            # Update perfectionism based on time spent
            if snapshot.focus_duration_avg > profile.preferred_work_duration * 1.5:
                profile.perfectionism_score = min(1.0,
                                                  profile.perfectionism_score + adaptation_rate * 0.1)

            # Update planning preference based on punctuality
            if snapshot.punctuality_score > 0.8:
                profile.planning_preference = min(1.0,
                                                  profile.planning_preference + adaptation_rate * 0.1)
            elif snapshot.punctuality_score < 0.4:
                profile.planning_preference = max(0.0,
                                                  profile.planning_preference - adaptation_rate * 0.1)

        except Exception as e:
            logger.error(f"❌ Failed to apply snapshot to profile: {e}")

    async def _generate_comprehensive_insights(
        self,
        user_id: str,
        patterns: List[BehaviorPattern],
        snapshot: BehaviorSnapshot
    ) -> List[UserInsight]:
        """Generate comprehensive insights from patterns and current behavior"""
        insights = []

        try:
            profile = self.user_personalities.get(user_id)

            # Pattern-based insights
            for pattern in patterns:
                insight = await self._pattern_to_insight(pattern, profile)
                if insight:
                    insights.append(insight)

            # Performance insights
            performance_insight = await self._generate_performance_insight(snapshot, profile)
            if performance_insight:
                insights.append(performance_insight)

            # Recommendation insights
            if profile and profile.confidence_score > 0.6:
                recommendation_insight = await self._generate_recommendation_insight(profile, snapshot)
                if recommendation_insight:
                    insights.append(recommendation_insight)

            # Trend insights
            if len(self.behavior_history[user_id]) >= 5:
                trend_insight = await self._generate_trend_insight(user_id)
                if trend_insight:
                    insights.append(trend_insight)

            # Sort by impact score
            insights.sort(key=lambda x: x.impact_score, reverse=True)

            return insights[:10]  # Return top 10 insights

        except Exception as e:
            logger.error(f"❌ Failed to generate comprehensive insights: {e}")
            return insights

    async def _pattern_to_insight(self, pattern: BehaviorPattern, profile: Optional[UserPersonality]) -> Optional[UserInsight]:
        """Convert a behavior pattern to a user insight"""
        try:
            pattern_insights = {
                "procrastination_chronic": {
                    "type": "warning",
                    "title": "Procrastination Pattern Detected",
                    "recommendation": "Try breaking large tasks into smaller, manageable chunks with personal deadlines before the real ones."
                },
                "deadline_pressure": {
                    "type": "neutral",
                    "title": "Thrives Under Pressure",
                    "recommendation": "Consider intentionally creating urgency for important tasks to leverage your pressure-responsive productivity."
                },
                "deep_focus_sustained": {
                    "type": "positive",
                    "title": "Excellent Deep Focus Ability",
                    "recommendation": "Schedule your most complex tasks during your natural deep focus periods."
                },
                "morning_energy": {
                    "type": "insight",
                    "title": "Morning Peak Performance",
                    "recommendation": "Schedule your most important and challenging tasks in the morning when your energy is highest."
                },
                "learning_improvement": {
                    "type": "positive",
                    "title": "Strong Learning Curve",
                    "recommendation": "You show excellent improvement over time. Consider taking on more challenging tasks in areas where you've shown growth."
                }
            }

            insight_template = pattern_insights.get(pattern.pattern_type)
            if not insight_template:
                return None

            return UserInsight(
                insight_type=insight_template["type"],
                title=insight_template["title"],
                description=pattern.description,
                recommendation=insight_template["recommendation"],
                confidence_score=pattern.frequency * pattern.impact_score,
                impact_score=pattern.impact_score,
                evidence_data={
                    "pattern_type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "detected_at": pattern.detected_at.isoformat()
                }
            )

# ai-service/app/services/personalization_engine.py - Part 6

        except Exception as e:
            logger.error(f"❌ Failed to convert pattern to insight: {e}")
            return None

    async def _generate_performance_insight(
        self,
        snapshot: BehaviorSnapshot,
        profile: Optional[UserPersonality]
    ) -> Optional[UserInsight]:
        """Generate insight about current performance"""
        try:
            if snapshot.completion_rate > 0.9:
                return UserInsight(
                    insight_type="positive",
                    title="Excellent Task Completion",
                    description=f"Outstanding completion rate of {snapshot.completion_rate:.1%}",
                    recommendation="You're doing great! Consider taking on more challenging or meaningful tasks.",
                    confidence_score=0.9,
                    impact_score=0.7,
                    evidence_data={
                        "completion_rate": snapshot.completion_rate,
                        "punctuality_score": snapshot.punctuality_score
                    }
                )

            elif snapshot.completion_rate < 0.4:
                return UserInsight(
                    insight_type="warning",
                    title="Low Task Completion Rate",
                    description=f"Completion rate of {snapshot.completion_rate:.1%} suggests potential overwhelm",
                    recommendation="Consider reducing task load or breaking large tasks into smaller, manageable pieces.",
                    confidence_score=0.8,
                    impact_score=0.9,
                    evidence_data={
                        "completion_rate": snapshot.completion_rate,
                        "stress_level": snapshot.stress_level
                    }
                )

            elif snapshot.stress_level > 0.7:
                return UserInsight(
                    insight_type="warning",
                    title="High Stress Level Detected",
                    description=f"Stress indicators suggest you may be under pressure",
                    recommendation="Consider scheduling breaks, reducing concurrent tasks, or extending deadlines where possible.",
                    confidence_score=0.7,
                    impact_score=0.8,
                    evidence_data={
                        "stress_level": snapshot.stress_level,
                        "task_switching_frequency": snapshot.task_switching_frequency
                    }
                )

            return None

        except Exception as e:
            logger.error(f"❌ Failed to generate performance insight: {e}")
            return None

    async def _generate_recommendation_insight(
        self,
        profile: UserPersonality,
        snapshot: BehaviorSnapshot
    ) -> Optional[UserInsight]:
        """Generate personalized recommendations based on profile"""
        try:
            recommendations = []

            # Procrastination recommendations
            if profile.procrastination_tendency > 0.7:
                recommendations.append(
                    "Set artificial deadlines 2-3 days before real deadlines")

            # Time management recommendations
            if profile.time_optimism > 0.7:
                recommendations.append(
                    "Add 50% buffer time to your task estimates")

            # Energy management recommendations
            if profile.peak_energy_hours:
                peak_hour_str = ", ".join(
                    map(str, profile.peak_energy_hours[:2]))
                recommendations.append(
                    f"Schedule your most important tasks around {peak_hour_str}:00")

            # Focus recommendations
            if profile.preferred_work_duration > 90:
                recommendations.append(
                    "Block out 2+ hour time slots for deep work sessions")
            elif profile.preferred_work_duration < 60:
                recommendations.append(
                    "Use short 25-45 minute focused work sessions with breaks")

            # Collaboration recommendations
            if profile.collaboration_preference > 0.7:
                recommendations.append(
                    "Look for opportunities to involve others in your tasks")
            elif profile.collaboration_preference < 0.3:
                recommendations.append(
                    "Protect your independent work time from interruptions")

            if not recommendations:
                return None

            return UserInsight(
                insight_type="recommendation",
                title="Personalized Productivity Recommendations",
                description="Based on your behavior patterns, here are personalized suggestions:",
                # Top 3 recommendations
                recommendation="; ".join(recommendations[:3]),
                confidence_score=profile.confidence_score,
                impact_score=0.8,
                evidence_data={
                    "personality_type": profile.personality_type.value if hasattr(profile, 'personality_type') else "unknown",
                    "work_style": profile.work_style.value if hasattr(profile, 'work_style') else "unknown",
                    "sample_size": profile.sample_size
                }
            )

        except Exception as e:
            logger.error(f"❌ Failed to generate recommendation insight: {e}")
            return None

    async def _generate_trend_insight(self, user_id: str) -> Optional[UserInsight]:
        """Generate insight about behavior trends over time"""
        try:
            history = self.behavior_history[user_id]
            if len(history) < 5:
                return None

            # Analyze recent vs older behavior
            recent_snapshots = history[-5:]
            older_snapshots = history[-10:-
                                      5] if len(history) >= 10 else history[:-5]

            if not older_snapshots:
                return None

            recent_completion = statistics.mean(
                [s.completion_rate for s in recent_snapshots])
            older_completion = statistics.mean(
                [s.completion_rate for s in older_snapshots])

            recent_productivity = statistics.mean(
                [s.productivity_score for s in recent_snapshots])
            older_productivity = statistics.mean(
                [s.productivity_score for s in older_snapshots])

            completion_trend = (recent_completion - older_completion) / \
                older_completion if older_completion > 0 else 0
            productivity_trend = (recent_productivity - older_productivity) / \
                older_productivity if older_productivity > 0 else 0

            # Determine trend type
            if completion_trend > 0.15 or productivity_trend > 0.15:
                trend_type = "improving"
                description = f"Your performance is trending upward (completion: {completion_trend:+.1%}, productivity: {productivity_trend:+.1%})"
                recommendation = "Great momentum! Consider setting more ambitious goals or taking on new challenges."
            elif completion_trend < -0.15 or productivity_trend < -0.15:
                trend_type = "declining"
                description = f"Your performance shows a declining trend (completion: {completion_trend:+.1%}, productivity: {productivity_trend:+.1%})"
                recommendation = "Consider reviewing your workload, taking breaks, or adjusting your approach."
            else:
                trend_type = "stable"
                description = "Your performance has been consistent over time"
                recommendation = "Steady performance! Look for opportunities to optimize or try new productivity techniques."

            return UserInsight(
                insight_type="trend" if trend_type == "stable" else (
                    "positive" if trend_type == "improving" else "warning"),
                title=f"Performance Trend: {trend_type.title()}",
                description=description,
                recommendation=recommendation,
                confidence_score=0.8,
                impact_score=0.7,
                evidence_data={
                    "trend_type": trend_type,
                    "completion_trend": completion_trend,
                    "productivity_trend": productivity_trend,
                    "sample_size": len(history)
                }
            )

        except Exception as e:
            logger.error(f"❌ Failed to generate trend insight: {e}")
            return None

    async def _generate_predictive_insights(
        self,
        user_id: str,
        task_history: List[Dict[str, Any]]
    ) -> List[UserInsight]:
        """Generate predictive insights using ML models or heuristics"""
        insights = []

        try:
            profile = self.user_personalities.get(user_id)
            if not profile:
                return insights

            # Predict potential issues
            upcoming_busy_period = await self._predict_busy_period(task_history)
            if upcoming_busy_period:
                insights.append(upcoming_busy_period)

            # Predict optimal scheduling
            optimal_schedule = await self._predict_optimal_schedule(profile, task_history)
            if optimal_schedule:
                insights.append(optimal_schedule)

            # Predict burnout risk
            burnout_risk = await self._predict_burnout_risk(user_id, profile)
            if burnout_risk:
                insights.append(burnout_risk)

            return insights

        except Exception as e:
            logger.error(f"❌ Failed to generate predictive insights: {e}")
            return insights

    async def _predict_busy_period(self, task_history: List[Dict[str, Any]]) -> Optional[UserInsight]:
        """Predict if user is entering a busy period"""
        try:
            # Look at task creation patterns
            recent_tasks = [
                t for t in task_history
                if t.get('created_at') and
                datetime.fromisoformat(t['created_at'].replace(
                    'Z', '+00:00')) > datetime.now() - timedelta(days=7)
            ]

            if len(recent_tasks) < 5:
                return None

            # Check for increasing task creation rate
            daily_counts = defaultdict(int)
            for task in recent_tasks:
                date = datetime.fromisoformat(
                    task['created_at'].replace('Z', '+00:00')).date()
                daily_counts[date] += 1

            if len(daily_counts) < 3:
                return None

            recent_avg = statistics.mean(list(daily_counts.values())[-3:])
            overall_avg = statistics.mean(list(daily_counts.values()))

            if recent_avg > overall_avg * 1.5:
                return UserInsight(
                    insight_type="warning",
                    title="Busy Period Detected",
                    description=f"Task creation rate has increased {(recent_avg/overall_avg-1):.0%} in recent days",
                    recommendation="Consider prioritizing tasks and blocking time for important work to avoid overwhelm.",
                    confidence_score=0.7,
                    impact_score=0.8,
                    evidence_data={
                        "recent_task_rate": recent_avg,
                        "normal_task_rate": overall_avg,
                        "increase_factor": recent_avg / overall_avg
                    }
                )

            return None

        except Exception as e:
            logger.error(f"❌ Failed to predict busy period: {e}")
            return None

    async def _predict_optimal_schedule(
        self,
        profile: UserPersonality,
        task_history: List[Dict[str, Any]]
    ) -> Optional[UserInsight]:
        """Predict optimal scheduling based on user patterns"""
        try:
            # Find user's most productive time patterns
            if not profile.peak_energy_hours:
                return None

            peak_hour_str = f"{profile.peak_energy_hours[0]}:00-{profile.peak_energy_hours[-1]}:00"

            # Generate schedule recommendation based on work style
            if profile.preferred_work_duration > 90:
                schedule_rec = f"Block {peak_hour_str} for deep work sessions (90+ minutes)"
            else:
                schedule_rec = f"Schedule focused work during {peak_hour_str} in shorter bursts"

            # Add break recommendations
            if hasattr(profile, 'break_frequency') and profile.break_frequency:
                schedule_rec += f" with breaks every {profile.break_frequency} minutes"

            return UserInsight(
                insight_type="recommendation",
                title="Optimal Schedule Prediction",
                description=f"Based on your patterns, optimal productivity window is {peak_hour_str}",
                recommendation=schedule_rec,
                confidence_score=profile.confidence_score,
                impact_score=0.7,
                evidence_data={
                    "peak_hours": profile.peak_energy_hours,
                    "work_duration": profile.preferred_work_duration,
                    "confidence": profile.confidence_score
                }
            )

        except Exception as e:
            logger.error(f"❌ Failed to predict optimal schedule: {e}")
            return None

    async def _predict_burnout_risk(self, user_id: str, profile: UserPersonality) -> Optional[UserInsight]:
        """Predict burnout risk based on behavior patterns"""
        try:
            if len(self.behavior_history[user_id]) < 7:
                return None

            # Last 7 snapshots
            recent_history = self.behavior_history[user_id][-7:]

            # Calculate risk factors
            avg_stress = statistics.mean(
                [s.stress_level for s in recent_history])
            completion_decline = self._calculate_completion_decline(
                recent_history)
            task_switching_increase = self._calculate_switching_increase(
                recent_history)

            risk_score = (avg_stress * 0.4 + completion_decline *
                          0.4 + task_switching_increase * 0.2)

            if risk_score > 0.7:
                return UserInsight(
                    insight_type="warning",
                    title="Burnout Risk Detected",
                    description=f"Multiple stress indicators suggest potential burnout risk (score: {risk_score:.1f})",
                    recommendation="Consider reducing workload, taking breaks, or discussing priorities with your team.",
                    confidence_score=0.8,
                    impact_score=0.9,
                    evidence_data={
                        "risk_score": risk_score,
                        "avg_stress": avg_stress,
                        "completion_decline": completion_decline,
                        "switching_increase": task_switching_increase
                    }
                )

            return None

        except Exception as e:
            logger.error(f"❌ Failed to predict burnout risk: {e}")
            return None

    def _calculate_completion_decline(self, history: List[BehaviorSnapshot]) -> float:
        """Calculate if completion rate is declining"""
        try:
            if len(history) < 4:
                return 0.0

            early_completion = statistics.mean(
                [s.completion_rate for s in history[:len(history)//2]])
            recent_completion = statistics.mean(
                [s.completion_rate for s in history[len(history)//2:]])

            if early_completion > 0:
                decline = max(
                    0.0, (early_completion - recent_completion) / early_completion)
                return min(1.0, decline)

            return 0.0

        except Exception:
            return 0.0

    def _calculate_switching_increase(self, history: List[BehaviorSnapshot]) -> float:
        """Calculate if task switching is increasing"""
        try:
            if len(history) < 4:
                return 0.0

            early_switching = statistics.mean(
                [s.task_switching_frequency for s in history[:len(history)//2]])
            recent_switching = statistics.mean(
                [s.task_switching_frequency for s in history[len(history)//2:]])

            if early_switching > 0:
                increase = max(
                    0.0, (recent_switching - early_switching) / early_switching)
                return min(1.0, increase / 2.0)  # Normalize to 0-1

            return 0.0

        except Exception:
            return 0.0

    async def _generate_fallback_insights(self, user_id: str) -> List[UserInsight]:
        """Generate basic insights when full analysis fails"""
        return [
            UserInsight(
                insight_type="info",
                title="Getting to Know You",
                description="I'm learning about your work patterns. Complete a few more tasks for personalized insights!",
                recommendation="Continue using the app normally, and I'll provide personalized recommendations soon.",
                confidence_score=0.5,
                impact_score=0.3,
                evidence_data={"user_id": user_id}
            )
        ]

# ai-service/app/services/personalization_engine.py - Part 7

    # Cache management methods
    def _generate_cache_key(self, prefix: str, user_id: str, data: Any) -> str:
        """Generate a cache key for results"""
        try:
            data_hash = hashlib.md5(str(data).encode()).hexdigest()[:8]
            return f"{prefix}_{user_id}_{data_hash}"
        except Exception:
            return f"{prefix}_{user_id}_{datetime.now().strftime('%Y%m%d%H')}"

    def _get_cached_result(self, cache_key: str, cache_dict: Dict) -> Optional[Any]:
        """Get cached result if still valid"""
        try:
            if cache_key in cache_dict:
                result, timestamp = cache_dict[cache_key]
                if datetime.now() - timestamp < self.cache_ttl:
                    return result
                else:
                    del cache_dict[cache_key]
            return None
        except Exception:
            return None

    def _cache_result(self, cache_key: str, result: Any, cache_dict: Dict):
        """Cache result with timestamp"""
        try:
            cache_dict[cache_key] = (result, datetime.now())
            # Limit cache size
            if len(cache_dict) > 1000:
                # Remove oldest entries
                sorted_items = sorted(cache_dict.items(),
                                      key=lambda x: x[1][1])
                for key, _ in sorted_items[:200]:
                    del cache_dict[key]
        except Exception:
            pass

    # Background tasks
    async def _background_model_training(self):
        """Background task for continuous model training"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                if ML_AVAILABLE and len(self.user_personalities) > 10:
                    await self._retrain_models()

                # Save user data
                await self._save_user_data()

            except Exception as e:
                logger.error(f"❌ Background training failed: {e}")

    async def _periodic_cache_cleanup(self):
        """Periodic cache cleanup"""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes

                # Clean expired cache entries
                current_time = datetime.now()

                for cache_dict in [self.insights_cache, self.predictions_cache]:
                    expired_keys = [
                        key for key, (_, timestamp) in cache_dict.items()
                        if current_time - timestamp > self.cache_ttl
                    ]

                    for key in expired_keys:
                        del cache_dict[key]

                logger.debug(f"🧹 Cache cleanup completed")

            except Exception as e:
                logger.error(f"❌ Cache cleanup failed: {e}")

    async def _retrain_models(self):
        """Retrain ML models with new data"""
        try:
            if not ML_AVAILABLE:
                return

            logger.info("🔄 Starting model retraining...")

            # Prepare training data from all users
            training_data = await self._prepare_training_data()

            if len(training_data) < 50:  # Need minimum samples
                logger.info("⏳ Not enough data for retraining yet")
                return

            # Retrain personality classifier
            if self.personality_classifier:
                X, y = self._prepare_personality_training_data(training_data)
                if len(X) > 10:
                    self.personality_classifier.fit(X, y)

            # Retrain behavior predictor
            if self.behavior_predictor:
                X, y = self._prepare_behavior_training_data(training_data)
                if len(X) > 10:
                    self.behavior_predictor.fit(X, y)

            # Save updated models
            await self._save_trained_models()

            logger.info("✅ Model retraining completed")

        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")

    async def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data from all user behavior"""
        training_samples = []

        for user_id, personality in self.user_personalities.items():
            if personality.sample_size < 5:
                continue

            history = self.behavior_history.get(user_id, [])
            if len(history) < 3:
                continue

            # Create training sample
            sample = {
                'user_id': user_id,
                'personality': asdict(personality),
                # Last 10 snapshots
                'behavior_history': [asdict(snapshot) for snapshot in history[-10:]],
                'patterns': []  # Would include detected patterns
            }

            training_samples.append(sample)

        return training_samples

    def _prepare_personality_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[List, List]:
        """Prepare data for personality classification training"""
        X, y = [], []

        for sample in training_data:
            # Extract features from behavior history
            behavior_features = []
            for snapshot in sample['behavior_history']:
                behavior_features.extend([
                    snapshot['completion_rate'],
                    snapshot['punctuality_score'],
                    snapshot['productivity_score'],
                    snapshot['stress_level'],
                    snapshot['focus_duration_avg'] / 120.0,  # Normalize
                    snapshot['task_switching_frequency'] / 10.0  # Normalize
                ])

            # Pad or truncate to fixed size
            while len(behavior_features) < 60:  # 10 snapshots * 6 features
                behavior_features.append(0.0)
            behavior_features = behavior_features[:60]

            X.append(behavior_features)

            # Target is personality type (simplified)
            personality = sample['personality']
            if personality['procrastination_tendency'] > 0.7:
                y.append('procrastinator')
            elif personality['perfectionism_score'] > 0.7:
                y.append('perfectionist')
            else:
                y.append('balanced')

        return X, y

    def _prepare_behavior_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[List, List]:
        """Prepare data for behavior prediction training"""
        X, y = [], []

        for sample in training_data:
            history = sample['behavior_history']
            if len(history) < 2:
                continue

            # Use all but last snapshot as features
            features = []
            for snapshot in history[:-1]:
                features.extend([
                    snapshot['completion_rate'],
                    snapshot['punctuality_score'],
                    snapshot['productivity_score'],
                    snapshot['stress_level']
                ])

            # Pad to fixed size
            while len(features) < 36:  # 9 snapshots * 4 features
                features.append(0.0)
            features = features[:36]

            X.append(features)

            # Target is next snapshot's completion rate
            y.append(history[-1]['completion_rate'])

        return X, y

    async def _save_trained_models(self):
        """Save trained ML models to disk"""
        try:
            models_file = self.model_cache_dir / "trained_models.pkl"

            models_data = {
                'personality_classifier': self.personality_classifier,
                'behavior_predictor': self.behavior_predictor,
                'recommendation_scorer': self.recommendation_scorer,
                'updated_at': datetime.now()
            }

            with open(models_file, 'wb') as f:
                pickle.dump(models_data, f)

            logger.info("💾 Trained models saved")

        except Exception as e:
            logger.error(f"❌ Failed to save trained models: {e}")

    async def _load_trained_models(self):
        """Load pre-trained ML models from disk"""
        try:
            models_file = self.model_cache_dir / "trained_models.pkl"

            if models_file.exists():
                with open(models_file, 'rb') as f:
                    models_data = pickle.load(f)

                self.personality_classifier = models_data.get(
                    'personality_classifier')
                self.behavior_predictor = models_data.get('behavior_predictor')
                self.recommendation_scorer = models_data.get(
                    'recommendation_scorer')

                logger.info("📚 Pre-trained models loaded")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load trained models: {e}")

    async def _save_user_data(self):
        """Save user personalities and behavior data"""
        try:
            # Save personalities
            personalities_file = self.model_cache_dir / "user_personalities.pkl"
            data = {
                'personalities': self.user_personalities,
                'behavior_history': dict(self.behavior_history),
                'updated_at': datetime.now()
            }

            with open(personalities_file, 'wb') as f:
                pickle.dump(data, f)

            # Save interaction patterns
            patterns_file = self.model_cache_dir / "interaction_patterns.pkl"
            with open(patterns_file, 'wb') as f:
                pickle.dump(dict(self.interaction_patterns), f)

            logger.debug("💾 User data saved")

        except Exception as e:
            logger.error(f"❌ Failed to save user data: {e}")

    async def _create_behavioral_clusters(self):
        """Create behavioral clusters for user segmentation"""
        try:
            if len(self.user_personalities) < 10:
                return

            # Extract features for clustering
            features = []
            user_ids = []

            for user_id, personality in self.user_personalities.items():
                if personality.sample_size < 5:
                    continue

                feature_vector = [
                    personality.procrastination_tendency,
                    personality.perfectionism_score,
                    personality.time_optimism,
                    personality.multitasking_preference,
                    personality.stress_tolerance,
                    personality.planning_preference,
                    personality.collaboration_preference,
                    personality.preferred_work_duration / 180.0,  # Normalize
                ]

                features.append(feature_vector)
                user_ids.append(user_id)

            if len(features) < 5:
                return

            # Perform clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Try different numbers of clusters
            best_score = -1
            best_n_clusters = 3

            for n_clusters in range(2, min(8, len(features)//2)):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(features_scaled)

                if len(set(cluster_labels)) > 1:
                    score = silhouette_score(features_scaled, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters

            # Final clustering
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)

            # Store cluster information
            self.behavioral_clusters = {
                'model': kmeans,
                'scaler': scaler,
                'user_clusters': dict(zip(user_ids, cluster_labels)),
                'n_clusters': best_n_clusters,
                'silhouette_score': best_score
            }

            logger.info(
                f"📊 Created {best_n_clusters} behavioral clusters (silhouette: {best_score:.3f})")

        except Exception as e:
            logger.error(f"❌ Failed to create behavioral clusters: {e}")

# ai-service/app/services/personalization_engine.py - Part 8

    # Public API Methods
    async def get_user_insights(
        self,
        user_id: str,
        task_history: List[Dict[str, Any]],
        interaction_data: Optional[Dict[str, Any]] = None
    ) -> List[UserInsight]:
        """
        Main public method to get user insights

        Args:
            user_id: User identifier
            task_history: User's task history
            interaction_data: Optional interaction data

        Returns:
            List of personalized insights
        """
        return await self.analyze_user_behavior(user_id, task_history, interaction_data)

    async def get_user_personality(self, user_id: str) -> Optional[UserPersonality]:
        """Get user's personality profile"""
        return self.user_personalities.get(user_id)

    async def update_user_feedback(
        self,
        user_id: str,
        insight_id: str,
        feedback: Dict[str, Any]
    ):
        """Update system based on user feedback"""
        try:
            self.recommendation_feedback[user_id].append({
                'insight_id': insight_id,
                'feedback': feedback,
                'timestamp': datetime.now()
            })

            # Adapt based on feedback
            if feedback.get('helpful') is False:
                await self._adapt_to_negative_feedback(user_id, insight_id, feedback)

            logger.debug(f"📝 Feedback recorded for user {user_id}")

        except Exception as e:
            logger.error(f"❌ Failed to update user feedback: {e}")

    async def _adapt_to_negative_feedback(
        self,
        user_id: str,
        insight_id: str,
        feedback: Dict[str, Any]
    ):
        """Adapt personalization based on negative feedback"""
        try:
            profile = self.user_personalities.get(user_id)
            if not profile:
                return

            # Reduce confidence in personality traits that led to poor recommendations
            reason = feedback.get('reason', '')

            if 'timing' in reason.lower():
                # Adjust peak energy hours
                if hasattr(profile, 'peak_energy_hours'):
                    profile.confidence_score *= 0.9

            elif 'workload' in reason.lower():
                # Adjust stress tolerance assessment
                profile.stress_tolerance *= 0.9
                profile.confidence_score *= 0.9

            elif 'focus' in reason.lower():
                # Adjust work duration preferences
                profile.preferred_work_duration = int(
                    profile.preferred_work_duration * 0.9)
                profile.confidence_score *= 0.9

            logger.debug(
                f"🔧 Adapted profile for user {user_id} based on feedback")

        except Exception as e:
            logger.error(f"❌ Failed to adapt to negative feedback: {e}")

    async def get_personalization_metrics(self, user_id: str) -> Optional[PersonalizationMetrics]:
        """Get personalization metrics for a user"""
        try:
            profile = self.user_personalities.get(user_id)
            if not profile:
                return None

            feedback_data = self.recommendation_feedback.get(user_id, [])
            helpful_feedback = len(
                [f for f in feedback_data if f['feedback'].get('helpful') is True])
            total_feedback = len(feedback_data)

            return PersonalizationMetrics(
                user_id=user_id,
                confidence_score=profile.confidence_score,
                sample_size=profile.sample_size,
                accuracy_score=helpful_feedback / total_feedback if total_feedback > 0 else 0.0,
                last_updated=profile.last_updated,
                insights_generated=len(
                    self.insights_cache.get(f"insights_{user_id}", [])),
                patterns_detected=len(
                    [p for p in self.behavior_history.get(user_id, [])]),
                adaptation_count=len(feedback_data)
            )

        except Exception as e:
            logger.error(f"❌ Failed to get personalization metrics: {e}")
            return None

    async def generate_adaptation_recommendations(
        self,
        user_id: str,
        current_context: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Generate real-time adaptation recommendations"""
        try:
            profile = self.user_personalities.get(user_id)
            if not profile or profile.confidence_score < 0.5:
                return []

            recommendations = []
            current_time = datetime.now()
            current_hour = current_time.hour

            # Time-based recommendations
            if current_hour in profile.peak_energy_hours:
                recommendations.append(AdaptationRecommendation(
                    recommendation_type="schedule_optimization",
                    title="Peak Energy Time",
                    description="This is one of your peak energy hours",
                    action="Schedule your most important or challenging tasks now",
                    confidence=profile.confidence_score,
                    priority="high",
                    context={"current_hour": current_hour,
                             "peak_hours": profile.peak_energy_hours}
                ))

            # Workload-based recommendations
            current_tasks = current_context.get('active_tasks', 0)
            if current_tasks > 5 and profile.stress_tolerance < 0.6:
                recommendations.append(AdaptationRecommendation(
                    recommendation_type="workload_management",
                    title="High Task Load Detected",
                    description=f"You have {current_tasks} active tasks and tend to be sensitive to high workloads",
                    action="Consider prioritizing 2-3 most important tasks and deferring others",
                    confidence=0.8,
                    priority="medium",
                    context={"active_tasks": current_tasks,
                             "stress_tolerance": profile.stress_tolerance}
                ))

            # Focus session recommendations
            if profile.preferred_work_duration > 90:
                recommendations.append(AdaptationRecommendation(
                    recommendation_type="focus_optimization",
                    title="Deep Focus Session",
                    description="You work best in longer focused sessions",
                    action=f"Block the next {profile.preferred_work_duration} minutes for uninterrupted work",
                    confidence=profile.confidence_score,
                    priority="medium",
                    context={
                        "preferred_duration": profile.preferred_work_duration}
                ))

            # Procrastination intervention
            if profile.procrastination_tendency > 0.7:
                overdue_tasks = current_context.get('overdue_tasks', 0)
                if overdue_tasks > 0:
                    recommendations.append(AdaptationRecommendation(
                        recommendation_type="procrastination_intervention",
                        title="Procrastination Alert",
                        description=f"You have {overdue_tasks} overdue tasks and tend to procrastinate",
                        action="Start with the smallest or easiest overdue task to build momentum",
                        confidence=0.9,
                        priority="high",
                        context={"overdue_tasks": overdue_tasks,
                                 "procrastination_score": profile.procrastination_tendency}
                    ))

            # Break recommendations
            last_break = current_context.get('last_break_time')
            if last_break:
                try:
                    last_break_dt = datetime.fromisoformat(
                        last_break.replace('Z', '+00:00'))
                    minutes_since_break = (
                        current_time - last_break_dt).total_seconds() / 60

                    if minutes_since_break > profile.break_frequency:
                        recommendations.append(AdaptationRecommendation(
                            recommendation_type="break_reminder",
                            title="Break Time",
                            description=f"It's been {minutes_since_break:.0f} minutes since your last break",
                            action="Take a 5-10 minute break to maintain productivity",
                            confidence=0.7,
                            priority="low",
                            context={"minutes_since_break": minutes_since_break,
                                     "break_frequency": profile.break_frequency}
                        ))
                except Exception:
                    pass

            # Sort by priority and confidence
            priority_order = {"high": 3, "medium": 2, "low": 1}
            recommendations.sort(
                key=lambda x: (priority_order.get(
                    x.priority, 0), x.confidence),
                reverse=True
            )

            return recommendations[:5]  # Return top 5 recommendations

        except Exception as e:
            logger.error(
                f"❌ Failed to generate adaptation recommendations: {e}")
            return []

    async def predict_task_success(
        self,
        user_id: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict likelihood of task success"""
        try:
            profile = self.user_personalities.get(user_id)
            if not profile:
                return {"completion_probability": 0.5, "on_time_probability": 0.5}

            # Extract task features
            estimated_duration = task_data.get('estimated_duration', 60)
            priority = task_data.get('priority', 3)
            due_date = task_data.get('due_date')
            category = task_data.get('category', 'general')

            # Base probabilities
            completion_prob = 0.7  # Default base probability
            on_time_prob = 0.6

            # Adjust based on personality
            if profile.procrastination_tendency > 0.7:
                completion_prob *= 0.8
                on_time_prob *= 0.6

            if profile.perfectionism_score > 0.7:
                completion_prob *= 1.1  # Perfectionists complete more
                on_time_prob *= 0.8  # But often late

            # Adjust based on task characteristics
            if estimated_duration > profile.preferred_work_duration * 1.5:
                completion_prob *= 0.9  # Longer tasks are harder

            if priority >= 4:
                completion_prob *= 1.2  # High priority tasks get more attention
                on_time_prob *= 1.1

            # Time pressure adjustment
            if due_date:
                try:
                    due_dt = datetime.fromisoformat(
                        due_date.replace('Z', '+00:00'))
                    hours_until_due = (due_dt - datetime.now()
                                       ).total_seconds() / 3600

                    if hours_until_due < 24:  # Less than 24 hours
                        if profile.stress_tolerance > 0.7:
                            completion_prob *= 1.2  # Thrives under pressure
                        else:
                            completion_prob *= 0.8  # Overwhelmed by pressure
                except Exception:
                    pass

            # Clamp probabilities
            completion_prob = max(0.1, min(0.95, completion_prob))
            on_time_prob = max(0.1, min(0.95, on_time_prob))

            return {
                "completion_probability": completion_prob,
                "on_time_probability": on_time_prob,
                "confidence": profile.confidence_score
            }

        except Exception as e:
            logger.error(f"❌ Failed to predict task success: {e}")
            return {"completion_probability": 0.5, "on_time_probability": 0.5, "confidence": 0.0}

# ai-service/app/services/personalization_engine.py - Part 9

    async def suggest_optimal_work_schedule(
        self,
        user_id: str,
        available_time_slots: List[Dict[str, Any]],
        tasks_to_schedule: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest optimal work schedule based on user patterns"""
        try:
            profile = self.user_personalities.get(user_id)
            if not profile:
                return []

            scheduled_tasks = []

            # Sort tasks by priority and deadline
            sorted_tasks = sorted(
                tasks_to_schedule,
                key=lambda t: (
                    -t.get('priority', 0),
                    datetime.fromisoformat(t['due_date'].replace(
                        'Z', '+00:00')) if t.get('due_date') else datetime.max
                )
            )

            # Sort time slots by user's energy patterns
            def slot_score(slot):
                slot_hour = datetime.fromisoformat(
                    slot['start_time'].replace('Z', '+00:00')).hour
                if slot_hour in profile.peak_energy_hours:
                    return 3  # Peak energy
                elif 6 <= slot_hour <= 22:
                    return 2  # Normal hours
                else:
                    return 1  # Low energy hours

            sorted_slots = sorted(available_time_slots,
                                  key=slot_score, reverse=True)

            # Schedule tasks
            for task in sorted_tasks:
                task_duration = task.get('estimated_duration', 60)

                # Adjust duration based on user patterns
                if profile.time_optimism > 0.7:
                    # Add buffer for optimistic estimators
                    task_duration = int(task_duration * 1.3)

                # Find suitable slot
                for slot in sorted_slots:
                    slot_start = datetime.fromisoformat(
                        slot['start_time'].replace('Z', '+00:00'))
                    slot_end = datetime.fromisoformat(
                        slot['end_time'].replace('Z', '+00:00'))
                    slot_duration = (
                        slot_end - slot_start).total_seconds() / 60

                    if slot_duration >= task_duration:
                        # Schedule the task
                        scheduled_end = slot_start + \
                            timedelta(minutes=task_duration)

                        scheduled_tasks.append({
                            "task_id": task.get('id'),
                            "task_title": task.get('title'),
                            "scheduled_start": slot_start.isoformat(),
                            "scheduled_end": scheduled_end.isoformat(),
                            "confidence": profile.confidence_score,
                            "reasoning": f"Scheduled during {'peak' if slot_score(slot) == 3 else 'good'} energy time"
                        })

                        # Update slot availability
                        if slot_duration > task_duration + 15:  # Leave 15-minute buffer
                            # Split the slot
                            remaining_start = scheduled_end + \
                                timedelta(minutes=15)
                            slot['start_time'] = remaining_start.isoformat()
                        else:
                            # Remove the slot
                            sorted_slots.remove(slot)

                        break

            return scheduled_tasks

        except Exception as e:
            logger.error(f"❌ Failed to suggest optimal schedule: {e}")
            return []

    async def get_productivity_coaching(self, user_id: str) -> Dict[str, Any]:
        """Generate personalized productivity coaching advice"""
        try:
            profile = self.user_personalities.get(user_id)
            if not profile or profile.sample_size < 10:
                return {
                    "message": "Keep using the app to build your productivity profile!",
                    "tips": ["Complete tasks regularly", "Set realistic deadlines", "Track your progress"],
                    "confidence": 0.3
                }

            coaching_advice = []
            improvement_areas = []
            strengths = []

            # Identify strengths
            if profile.procrastination_tendency < 0.3:
                strengths.append("excellent at starting tasks promptly")

            if profile.perfectionism_score > 0.7:
                strengths.append("high attention to detail and quality")

            if profile.stress_tolerance > 0.7:
                strengths.append("performs well under pressure")

            if profile.planning_preference > 0.7:
                strengths.append("strong planning and organization skills")

            # Identify improvement areas
            if profile.procrastination_tendency > 0.7:
                improvement_areas.append("reducing procrastination")
                coaching_advice.append(
                    "Break large tasks into smaller, 15-minute chunks to overcome initial resistance")

            if profile.time_optimism > 0.7:
                improvement_areas.append("realistic time estimation")
                coaching_advice.append(
                    "Add 25-50% buffer time to your initial estimates based on your completion history")

            if profile.stress_tolerance < 0.4:
                improvement_areas.append("stress management")
                coaching_advice.append(
                    "When feeling overwhelmed, focus on just one task at a time and take regular breaks")

            if profile.multitasking_preference > 0.8 and profile.preferred_work_duration < 45:
                improvement_areas.append("deep focus development")
                coaching_advice.append(
                    "Practice gradually extending your focus periods by 5-10 minutes each week")

            # Generate personalized message
            if strengths and improvement_areas:
                message = f"Your strengths include being {', '.join(strengths[:2])}. Focus on {improvement_areas[0]} for the biggest impact."
            elif strengths:
                message = f"You excel at {', '.join(strengths[:2])}. Consider taking on more challenging projects that leverage these strengths."
            elif improvement_areas:
                message = f"Your main growth opportunity is {improvement_areas[0]}. Small, consistent improvements here will make a big difference."
            else:
                message = "You have a balanced productivity profile. Continue building consistent habits."

            # Add energy-based advice
            if profile.peak_energy_hours:
                peak_hours = ', '.join(map(str, profile.peak_energy_hours[:2]))
                coaching_advice.append(
                    f"Schedule your most important work during your peak hours: {peak_hours}:00")

            return {
                "message": message,
                "strengths": strengths,
                "improvement_areas": improvement_areas,
                "coaching_tips": coaching_advice[:3],  # Top 3 tips
                "confidence": profile.confidence_score,
                "profile_maturity": "mature" if profile.sample_size > 50 else "developing"
            }

        except Exception as e:
            logger.error(f"❌ Failed to generate productivity coaching: {e}")
            return {
                "message": "Unable to generate personalized coaching at this time.",
                "tips": ["Focus on consistency", "Set clear priorities", "Take regular breaks"],
                "confidence": 0.0
            }

    async def analyze_productivity_trends(
        self,
        user_id: str,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze productivity trends over time"""
        try:
            history = self.behavior_history.get(user_id, [])
            if len(history) < 7:
                return {"error": "Insufficient data for trend analysis"}

            # Filter to time period
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            recent_history = [
                snapshot for snapshot in history
                if snapshot.timestamp > cutoff_date
            ]

            if len(recent_history) < 3:
                return {"error": f"Insufficient data for {time_period_days}-day analysis"}

            # Calculate trends
            completion_rates = [s.completion_rate for s in recent_history]
            productivity_scores = [
                s.productivity_score for s in recent_history]
            stress_levels = [s.stress_level for s in recent_history]

            # Linear trend calculation (simple slope)
            def calculate_trend(values):
                if len(values) < 2:
                    return 0.0
                x = list(range(len(values)))
                n = len(values)
                slope = (n * sum(i * v for i, v in zip(x, values)) - sum(x)
                         * sum(values)) / (n * sum(i * i for i in x) - sum(x) ** 2)
                return slope

            completion_trend = calculate_trend(completion_rates)
            productivity_trend = calculate_trend(productivity_scores)
            stress_trend = calculate_trend(stress_levels)

            # Interpret trends
            def interpret_trend(trend, metric_name):
                if trend > 0.01:
                    return f"{metric_name} is improving"
                elif trend < -0.01:
                    return f"{metric_name} is declining"
                else:
                    return f"{metric_name} is stable"

            # Weekly patterns
            weekly_performance = defaultdict(list)
            for snapshot in recent_history:
                day_of_week = snapshot.timestamp.strftime('%A')
                weekly_performance[day_of_week].append(
                    snapshot.productivity_score)

            best_day = max(weekly_performance.items(), key=lambda x: statistics.mean(x[1]))[
                0] if weekly_performance else "Monday"
            worst_day = min(weekly_performance.items(), key=lambda x: statistics.mean(x[1]))[
                0] if weekly_performance else "Friday"

            return {
                "period_days": time_period_days,
                "data_points": len(recent_history),
                "trends": {
                    "completion_rate": {
                        "direction": interpret_trend(completion_trend, "Task completion"),
                        "slope": completion_trend,
                        "current_avg": statistics.mean(completion_rates[-3:]) if len(completion_rates) >= 3 else statistics.mean(completion_rates)
                    },
                    "productivity": {
                        "direction": interpret_trend(productivity_trend, "Productivity"),
                        "slope": productivity_trend,
                        "current_avg": statistics.mean(productivity_scores[-3:]) if len(productivity_scores) >= 3 else statistics.mean(productivity_scores)
                    },
                    "stress": {
                        "direction": interpret_trend(stress_trend, "Stress level"),
                        "slope": stress_trend,
                        "current_avg": statistics.mean(stress_levels[-3:]) if len(stress_levels) >= 3 else statistics.mean(stress_levels)
                    }
                },
                "weekly_patterns": {
                    "best_day": best_day,
                    "worst_day": worst_day,
                    "day_averages": {day: statistics.mean(scores) for day, scores in weekly_performance.items()}
                },
                "insights": self._generate_trend_insights(completion_trend, productivity_trend, stress_trend),
                # Higher confidence with more data
                "confidence": min(0.9, len(recent_history) / 20)
            }

        except Exception as e:
            logger.error(f"❌ Failed to analyze productivity trends: {e}")
            return {"error": f"Trend analysis failed: {str(e)}"}

    def _generate_trend_insights(self, completion_trend, productivity_trend, stress_trend):
        """Generate insights from trend analysis"""
        insights = []

        if completion_trend > 0.02 and productivity_trend > 0.02:
            insights.append(
                "🚀 You're on a great trajectory! Both completion rate and productivity are improving.")

        elif completion_trend < -0.02 and stress_trend > 0.02:
            insights.append(
                "⚠️ Increasing stress may be impacting your completion rate. Consider reducing workload.")

        elif productivity_trend > 0.02:
            insights.append(
                "📈 Your efficiency is improving! You're getting better at estimating and completing tasks.")

        elif stress_trend > 0.03:
            insights.append(
                "😰 Stress levels are rising. This might be a good time to focus on stress management techniques.")

        elif completion_trend < -0.02:
            insights.append(
                "📉 Task completion is declining. Consider reviewing your task load and priorities.")

        else:
            insights.append(
                "📊 Your productivity metrics are stable. Consider experimenting with new techniques for improvement.")

        return insights

    async def get_behavioral_cluster_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user's behavioral cluster"""
        try:
            if not self.behavioral_clusters or user_id not in self.behavioral_clusters.get('user_clusters', {}):
                return {"error": "User not in any behavioral cluster yet"}

            user_cluster = self.behavioral_clusters['user_clusters'][user_id]
            cluster_model = self.behavioral_clusters['model']

            # Get cluster center (average personality traits)
            cluster_center = cluster_model.cluster_centers_[user_cluster]

            # Map features back to personality traits
            trait_names = [
                'procrastination_tendency',
                'perfectionism_score',
                'time_optimism',
                'multitasking_preference',
                'stress_tolerance',
                'planning_preference',
                'collaboration_preference',
                'preferred_work_duration'
            ]

            cluster_traits = dict(zip(trait_names, cluster_center))

            # Find similar users in the same cluster
            similar_users = [
                uid for uid, cluster in self.behavioral_clusters['user_clusters'].items()
                if cluster == user_cluster and uid != user_id
            ]

            # Generate cluster description
            dominant_traits = sorted(cluster_traits.items(
            ), key=lambda x: abs(x[1]), reverse=True)[:3]

            return {
                "cluster_id": user_cluster,
                "cluster_size": len([c for c in self.behavioral_clusters['user_clusters'].values() if c == user_cluster]),
                "similar_users_count": len(similar_users),
                "dominant_traits": dominant_traits,
                "cluster_description": self._generate_cluster_description(dominant_traits),
                "confidence": self.behavioral_clusters.get('silhouette_score', 0.0)
            }

        except Exception as e:
            logger.error(f"❌ Failed to get cluster insights: {e}")
            return {"error": f"Cluster analysis failed: {str(e)}"}

    def _generate_cluster_description(self, dominant_traits: List[Tuple[str, float]]) -> str:
        """Generate human-readable cluster description"""
        descriptions = {
            'procrastination_tendency': ("low procrastination", "high procrastination"),
            'perfectionism_score': ("pragmatic approach", "perfectionist approach"),
            'time_optimism': ("realistic time estimates", "optimistic time estimates"),
            'multitasking_preference': ("focused single-tasking", "multitasking preference"),
            'stress_tolerance': ("stress-sensitive", "stress-resilient"),
            'planning_preference': ("spontaneous approach", "detailed planning"),
            'collaboration_preference': ("independent work style", "collaborative work style"),
        }

        trait_descriptions = []
        for trait, value in dominant_traits:
            if trait in descriptions:
                desc = descriptions[trait][1] if value > 0 else descriptions[trait][0]
                trait_descriptions.append(desc)

        if len(trait_descriptions) >= 2:
            return f"Users with {trait_descriptions[0]} and {trait_descriptions[1]}"
        elif len(trait_descriptions) == 1:
            return f"Users with {trait_descriptions[0]}"
        else:
            return "Balanced productivity style"

    # Cleanup and shutdown methods
    async def shutdown(self):
        """Gracefully shutdown the personalization engine"""
        try:
            logger.info("🔄 Shutting down Personalization Engine...")

            # Save current state
            await self._save_user_data()
            await self._save_trained_models()

            # Save behavioral clusters
            if self.behavioral_clusters:
                clusters_file = self.model_cache_dir / "behavioral_clusters.pkl"
                with open(clusters_file, 'wb') as f:
                    pickle.dump(self.behavioral_clusters, f)

            logger.info("✅ Personalization Engine shutdown complete")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


# Factory function for creating the personalization engine
async def create_personalization_engine() -> PersonalizationEngine:
    """Create and initialize a personalization engine instance"""
    engine = PersonalizationEngine()
    await engine.initialize()
    return engine


# Global instance for the application
_personalization_engine: Optional[PersonalizationEngine] = None


async def get_personalization_engine() -> PersonalizationEngine:
    """Get the global personalization engine instance"""
    global _personalization_engine

    if _personalization_engine is None:
        _personalization_engine = await create_personalization_engine()

    return _personalization_engine
