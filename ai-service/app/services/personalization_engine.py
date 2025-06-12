# ai-service/app/services/personalization_engine.py
"""
Advanced personalization engine for continuous learning and adaptation
Handles user-specific model fine-tuning and behavioral pattern recognition
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import pickle
from pathlib import Path

# ML Libraries
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

# Database
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.sql import func

# Internal imports
from app.core.config import get_settings
from app.core.database import get_async_session
from app.models.ai_models import (
    UserBehaviorPattern, UserModelWeights, TaskEmbedding,
    AITrainingSession, UserPreferenceEmbedding, UserVocabulary
)
from app.schemas.ai_schemas import (
    PersonalizationMetrics, UserInsight, BehaviorPattern,
    AdaptationRecommendation
)

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class UserPersonality:
    """User personality profile for personalization"""
    user_id: str
    procrastination_tendency: float  # 0-1, higher = more likely to procrastinate
    perfectionism_score: float      # 0-1, higher = more perfectionist
    time_optimism: float            # 0-1, higher = underestimates time
    priority_style: str             # 'deadline_driven', 'importance_first', 'effort_based'
    work_rhythm: str                # 'morning_person', 'night_owl', 'flexible'
    multitasking_preference: float  # 0-1, higher = prefers multitasking
    stress_response: str            # 'productive', 'paralyzed', 'chaotic'
    motivation_type: str            # 'achievement', 'progress', 'completion'
    updated_at: datetime

@dataclass
class BehaviorInsight:
    """Insights derived from user behavior analysis"""
    insight_type: str
    confidence: float
    description: str
    recommendation: str
    impact_score: float
    supporting_data: Dict[str, Any]

class PersonalizationEngine:
    """
    Advanced personalization engine that learns user patterns and adapts models
    """
    
    def __init__(self):
        self.user_personalities: Dict[str, UserPersonality] = {}
        self.behavior_analyzers: Dict[str, Any] = {}
        self.adaptation_queue = asyncio.Queue()
        self.insight_cache: Dict[str, List[BehaviorInsight]] = defaultdict(list)
        
        # Learning parameters
        self.min_interactions_for_personalization = 20
        self.personality_update_threshold = 50
        self.adaptation_frequency = timedelta(days=7)
        
    async def initialize(self):
        """Initialize personalization engine"""
        try:
            logger.info("🧠 Initializing Personalization Engine...")
            
            # Load existing user personalities
            await self._load_user_personalities()
            
            # Initialize behavior analyzers
            await self._initialize_analyzers()
            
            # Start adaptation worker
            asyncio.create_task(self._adaptation_worker())
            
            logger.info("✅ Personalization Engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Personalization Engine: {e}")
            raise

    async def analyze_user_behavior(self, user_id: str) -> List[BehaviorInsight]:
        """Comprehensive analysis of user behavior patterns"""
        try:
            async with get_async_session() as db:
                # Get user's historical data
                behavior_data = await self._get_user_behavior_data(db, user_id)
                
                if len(behavior_data) < 10:
                    return [BehaviorInsight(
                        insight_type="insufficient_data",
                        confidence=1.0,
                        description="Need more interaction data for meaningful insights",
                        recommendation="Continue using the app to unlock personalized insights",
                        impact_score=0.0,
                        supporting_data={"data_points": len(behavior_data)}
                    )]
                
                insights = []
                
                # Analyze different behavioral aspects
                insights.extend(await self._analyze_productivity_patterns(behavior_data))
                insights.extend(await self._analyze_time_management(behavior_data))
                insights.extend(await self._analyze_priority_behavior(behavior_data))
                insights.extend(await self._analyze_completion_patterns(behavior_data))
                insights.extend(await self._analyze_stress_indicators(behavior_data))
                
                # Cache insights
                self.insight_cache[user_id] = insights
                
                # Update user personality profile
                await self._update_user_personality(user_id, behavior_data)
                
                return insights
                
        except Exception as e:
            logger.error(f"Behavior analysis failed for {user_id}: {e}")
            return []

    async def _analyze_productivity_patterns(self, behavior_data: pd.DataFrame) -> List[BehaviorInsight]:
        """Analyze user's productivity patterns"""
        insights = []
        
        try:
            # Time-of-day productivity analysis
            if 'completed_at' in behavior_data.columns:
                behavior_data['hour'] = pd.to_datetime(behavior_data['completed_at']).dt.hour
                hourly_productivity = behavior_data.groupby('hour').size()
                
                peak_hour = hourly_productivity.idxmax()
                peak_productivity = hourly_productivity.max()
                avg_productivity = hourly_productivity.mean()
                
                if peak_productivity > avg_productivity * 1.5:
                    if peak_hour < 10:
                        rhythm = "morning person"
                    elif peak_hour > 18:
                        rhythm = "night owl"
                    else:
                        rhythm = "midday performer"
                    
                    insights.append(BehaviorInsight(
                        insight_type="productivity_rhythm",
                        confidence=min(0.9, (peak_productivity / avg_productivity - 1)),
                        description=f"You're most productive as a {rhythm}, with peak performance around {peak_hour}:00",
                        recommendation=f"Schedule your most important tasks around {peak_hour}:00 for optimal results",
                        impact_score=0.8,
                        supporting_data={
                            "peak_hour": peak_hour,
                            "productivity_boost": f"{((peak_productivity / avg_productivity - 1) * 100):.1f}%"
                        }
                    ))
            
            # Weekly patterns
            if 'completed_at' in behavior_data.columns:
                behavior_data['day_of_week'] = pd.to_datetime(behavior_data['completed_at']).dt.day_name()
                weekly_productivity = behavior_data.groupby('day_of_week').size()
                
                best_day = weekly_productivity.idxmax()
                worst_day = weekly_productivity.idxmin()
                
                insights.append(BehaviorInsight(
                    insight_type="weekly_pattern",
                    confidence=0.7,
                    description=f"Your most productive day is {best_day}, least productive is {worst_day}",
                    recommendation=f"Plan challenging tasks for {best_day} and lighter work for {worst_day}",
                    impact_score=0.6,
                    supporting_data={
                        "best_day": best_day,
                        "worst_day": worst_day,
                        "productivity_variation": f"{((weekly_productivity.max() / weekly_productivity.min() - 1) * 100):.1f}%"
                    }
                ))
                
        except Exception as e:
            logger.error(f"Productivity pattern analysis failed: {e}")
            
        return insights

    async def _analyze_time_management(self, behavior_data: pd.DataFrame) -> List[BehaviorInsight]:
        """Analyze user's time management patterns"""
        insights = []
        
        try:
            if 'estimated_duration' in behavior_data.columns and 'actual_duration' in behavior_data.columns:
                # Time estimation accuracy
                time_data = behavior_data.dropna(subset=['estimated_duration', 'actual_duration'])
                
                if len(time_data) > 5:
                    estimation_ratio = time_data['actual_duration'] / time_data['estimated_duration']
                    avg_ratio = estimation_ratio.mean()
                    consistency = 1 - estimation_ratio.std()
                    
                    if avg_ratio > 1.3:
                        tendency = "underestimate"
                        advice = "Add 30-50% buffer time to your estimates"
                    elif avg_ratio < 0.7:
                        tendency = "overestimate" 
                        advice = "You can be more ambitious with your time estimates"
                    else:
                        tendency = "accurate"
                        advice = "Your time estimation skills are well-calibrated"
                    
                    insights.append(BehaviorInsight(
                        insight_type="time_estimation",
                        confidence=min(0.9, consistency),
                        description=f"You tend to {tendency} task duration by {abs(avg_ratio - 1) * 100:.1f}%",
                        recommendation=advice,
                        impact_score=0.7,
                        supporting_data={
                            "average_ratio": avg_ratio,
                            "consistency_score": consistency,
                            "sample_size": len(time_data)
                        }
                    ))
                
        except Exception as e:
            logger.error(f"Time management analysis failed: {e}")
            
        return insights

    async def _analyze_priority_behavior(self, behavior_data: pd.DataFrame) -> List[BehaviorInsight]:
        """Analyze how user handles task priorities"""
        insights = []
        
        try:
            if 'priority' in behavior_data.columns and 'completed_at' in behavior_data.columns:
                # Priority completion patterns
                priority_completion = behavior_data.groupby('priority').size()
                
                if len(priority_completion) > 2:
                    high_priority_completed = priority_completion.get(5, 0) + priority_completion.get(4, 0)
                    low_priority_completed = priority_completion.get(1, 0) + priority_completion.get(2, 0)
                    
                    if low_priority_completed > high_priority_completed * 1.5:
                        insights.append(BehaviorInsight(
                            insight_type="priority_avoidance",
                            confidence=0.8,
                            description="You tend to complete low-priority tasks more often than high-priority ones",
                            recommendation="Try the 'Eat the Frog' technique: tackle your most important task first each day",
                            impact_score=0.9,
                            supporting_data={
                                "high_priority_ratio": high_priority_completed / len(behavior_data),
                                "low_priority_ratio": low_priority_completed / len(behavior_data)
                            }
                        ))
                    elif high_priority_completed > low_priority_completed * 2:
                        insights.append(BehaviorInsight(
                            insight_type="priority_focused",
                            confidence=0.8,
                            description="You're excellent at focusing on high-priority tasks",
                            recommendation="Consider scheduling some low-priority tasks to prevent them from becoming urgent",
                            impact_score=0.7,
                            supporting_data={
                                "high_priority_ratio": high_priority_completed / len(behavior_data),
                                "focus_strength": "high_priority_focused"
                            }
                        ))
                        
        except Exception as e:
            logger.error(f"Priority behavior analysis failed: {e}")
            
        return insights

    async def _analyze_completion_patterns(self, behavior_data: pd.DataFrame) -> List[BehaviorInsight]:
        """Analyze task completion patterns and procrastination tendencies"""
        insights = []
        
        try:
            if 'due_date' in behavior_data.columns and 'completed_at' in behavior_data.columns:
                # Deadline behavior analysis
                completed_data = behavior_data.dropna(subset=['due_date', 'completed_at'])
                
                if len(completed_data) > 5:
                    completed_data['due_date'] = pd.to_datetime(completed_data['due_date'])
                    completed_data['completed_at'] = pd.to_datetime(completed_data['completed_at'])
                    completed_data['days_early'] = (completed_data['due_date'] - completed_data['completed_at']).dt.days
                    
                    avg_days_early = completed_data['days_early'].mean()
                    on_time_rate = (completed_data['days_early'] >= 0).mean()
                    
                    if avg_days_early < -2:
                        procrastination_score = min(1.0, abs(avg_days_early) / 7)
                        insights.append(BehaviorInsight(
                            insight_type="procrastination_tendency",
                            confidence=0.9,
                            description=f"You tend to complete tasks {abs(avg_days_early):.1f} days after the deadline",
                            recommendation="Try breaking large tasks into smaller chunks and setting personal deadlines before the real ones",
                            impact_score=0.8,
                            supporting_data={
                                "average_delay": abs(avg_days_early),
                                "on_time_rate": on_time_rate,
                                "procrastination_score": procrastination_score
                            }
                        ))
                    elif avg_days_early > 2:
                        insights.append(BehaviorInsight(
                            insight_type="early_completion",
                            confidence=0.8,
                            description=f"You typically complete tasks {avg_days_early:.1f} days early",
                            recommendation="You might be over-planning. Consider taking on more challenging projects",
                            impact_score=0.6,
                            supporting_data={
                                "average_early": avg_days_early,
                                "early_completion_rate": (completed_data['days_early'] > 0).mean()
                            }
                        ))
                        
        except Exception as e:
            logger.error(f"Completion pattern analysis failed: {e}")
            
        return insights

    async def _analyze_stress_indicators(self, behavior_data: pd.DataFrame) -> List[BehaviorInsight]:
        """Analyze indicators of stress and overwhelm"""
        insights = []
        
        try:
            # Task abandonment patterns
            if 'status' in behavior_data.columns:
                abandonment_rate = (behavior_data['status'] == 'abandoned').mean()
                
                if abandonment_rate > 0.2:
                    insights.append(BehaviorInsight(
                        insight_type="task_overwhelm",
                        confidence=0.8,
                        description=f"You abandon {abandonment_rate * 100:.1f}% of your tasks",
                        recommendation="Consider setting smaller, more achievable goals to reduce overwhelm",
                        impact_score=0.9,
                        supporting_data={
                            "abandonment_rate": abandonment_rate,
                            "abandoned_count": (behavior_data['status'] == 'abandoned').sum()
                        }
                    ))
            
            # Task creation vs completion ratio
            if 'created_at' in behavior_data.columns and 'completed_at' in behavior_data.columns:
                total_created = len(behavior_data)
                total_completed = len(behavior_data.dropna(subset=['completed_at']))
                completion_rate = total_completed / total_created if total_created > 0 else 0
                
                if completion_rate < 0.6:
                    insights.append(BehaviorInsight(
                        insight_type="low_completion_rate",
                        confidence=0.9,
                        description=f"You complete only {completion_rate * 100:.1f}% of created tasks",
                        recommendation="Focus on creating fewer, more important tasks rather than many tasks",
                        impact_score=0.8,
                        supporting_data={
                            "completion_rate": completion_rate,
                            "total_created": total_created,
                            "total_completed": total_completed
                        }
                    ))
                elif completion_rate > 0.9:
                    insights.append(BehaviorInsight(
                        insight_type="high_completion_rate",
                        confidence=0.8,
                        description=f"Excellent! You complete {completion_rate * 100:.1f}% of your tasks",
                        recommendation="You might be ready for more challenging or ambitious goals",
                        impact_score=0.7,
                        supporting_data={
                            "completion_rate": completion_rate,
                            "efficiency_score": "high"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Stress indicator analysis failed: {e}")
            
        return insights

    async def adapt_user_model(self, user_id: str) -> AdaptationRecommendation:
        """Adapt user's model based on recent behavior and insights"""
        try:
            # Get recent insights
            insights = self.insight_cache.get(user_id, [])
            
            if not insights:
                insights = await self.analyze_user_behavior(user_id)
            
            # Get user personality
            personality = self.user_personalities.get(user_id)
            
            # Generate adaptation recommendations
            adaptations = []
            
            for insight in insights:
                if insight.impact_score > 0.7:  # High impact insights
                    adaptation = await self._generate_model_adaptation(insight, personality)
                    if adaptation:
                        adaptations.append(adaptation)
            
            # Prioritize adaptations
            adaptations.sort(key=lambda x: x.get('priority', 0), reverse=True)
            
            return AdaptationRecommendation(
                user_id=user_id,
                adaptations=adaptations[:5],  # Top 5 adaptations
                confidence=np.mean([a.get('confidence', 0.5) for a in adaptations]),
                expected_improvement=sum([a.get('impact', 0) for a in adaptations]) / len(adaptations) if adaptations else 0,
                implementation_date=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Model adaptation failed for {user_id}: {e}")
            return AdaptationRecommendation(
                user_id=user_id,
                adaptations=[],
                confidence=0.0,
                expected_improvement=0.0,
                implementation_date=datetime.now()
            )

    async def _generate_model_adaptation(self, insight: BehaviorInsight, 
                                       personality: Optional[UserPersonality]) -> Optional[Dict[str, Any]]:
        """Generate specific model adaptations based on insights"""
        try:
            adaptation = {
                'type': 'model_parameter_adjustment',
                'confidence': insight.confidence,
                'impact': insight.impact_score,
                'priority': insight.impact_score * insight.confidence
            }
            
            if insight.insight_type == "procrastination_tendency":
                adaptation.update({
                    'adjustment_type': 'deadline_buffer',
                    'parameter': 'time_estimation_multiplier',
                    'value': 1.3,  # Add 30% buffer for procrastinators
                    'reasoning': 'Increase time estimates to account for procrastination tendency'
                })
                
            elif insight.insight_type == "priority_avoidance":
                adaptation.update({
                    'adjustment_type': 'priority_boost',
                    'parameter': 'high_priority_weight',
                    'value': 1.5,  # Boost high priority tasks
                    'reasoning': 'Increase importance of high-priority tasks in recommendations'
                })
                
            elif insight.insight_type == "productivity_rhythm":
                peak_hour = insight.supporting_data.get('peak_hour', 9)
                adaptation.update({
                    'adjustment_type': 'scheduling_preference',
                    'parameter': 'optimal_work_hours',
                    'value': [peak_hour - 2, peak_hour + 2],  # 4-hour window around peak
                    'reasoning': f'Schedule important tasks during peak productivity hours around {peak_hour}:00'
                })
                
            elif insight.insight_type == "time_estimation":
                ratio = insight.supporting_data.get('average_ratio', 1.0)
                adaptation.update({
                    'adjustment_type': 'time_calibration',
                    'parameter': 'time_estimation_correction',
                    'value': 1.0 / ratio,  # Correct for systematic bias
                    'reasoning': f'Adjust time estimates to correct for {abs(ratio - 1) * 100:.1f}% estimation bias'
                })
                
            else:
                return None  # No specific adaptation for this insight type
                
            return adaptation
            
        except Exception as e:
            logger.error(f"Failed to generate adaptation for insight {insight.insight_type}: {e}")
            return None

    async def _update_user_personality(self, user_id: str, behavior_data: pd.DataFrame):
        """Update user personality profile based on behavior analysis"""
        try:
            # Calculate personality traits from behavior data
            personality_scores = await self._calculate_personality_scores(behavior_data)
            
            # Get existing personality or create new one
            if user_id in self.user_personalities:
                personality = self.user_personalities[user_id]
                # Use exponential moving average to update traits
                alpha = 0.3  # Learning rate
                for trait, new_score in personality_scores.items():
                    old_score = getattr(personality, trait, 0.5)
                    updated_score = alpha * new_score + (1 - alpha) * old_score
                    setattr(personality, trait, updated_score)
                personality.updated_at = datetime.now()
            else:
                personality = UserPersonality(
                    user_id=user_id,
                    updated_at=datetime.now(),
                    **personality_scores
                )
                self.user_personalities[user_id] = personality
            
            # Save to database
            await self._save_user_personality(personality)
            
        except Exception as e:
            logger.error(f"Failed to update personality for {user_id}: {e}")

    async def _calculate_personality_scores(self, behavior_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate personality trait scores from behavior data"""
        scores = {}
        
        try:
            # Procrastination tendency
            if 'due_date' in behavior_data.columns and 'completed_at' in behavior_data.columns:
                completed_data = behavior_data.dropna(subset=['due_date', 'completed_at'])
                if len(completed_data) > 0:
                    late_rate = (pd.to_datetime(completed_data['completed_at']) > 
                               pd.to_datetime(completed_data['due_date'])).mean()
                    scores['procrastination_tendency'] = min(1.0, late_rate * 2)
                else:
                    scores['procrastination_tendency'] = 0.5
            else:
                scores['procrastination_tendency'] = 0.5
            
            # Time optimism (underestimating time)
            if 'estimated_duration' in behavior_data.columns and 'actual_duration' in behavior_data.columns:
                time_data = behavior_data.dropna(subset=['estimated_duration', 'actual_duration'])
                if len(time_data) > 0:
                    time_ratio = (time_data['actual_duration'] / time_data['estimated_duration']).mean()
                    # Higher ratio = more underestimation = higher time optimism
                    scores['time_optimism'] = min(1.0, max(0.0, (time_ratio - 0.5) / 1.5))
                else:
                    scores['time_optimism'] = 0.5
            else:
                scores['time_optimism'] = 0.5
            
            # Perfectionism (time spent vs estimated, revision patterns)
            if 'actual_duration' in behavior_data.columns and 'estimated_duration' in behavior_data.columns:
                time_data = behavior_data.dropna(subset=['estimated_duration', 'actual_duration'])
                if len(time_data) > 0:
                    over_time_rate = (time_data['actual_duration'] > 
                                    time_data['estimated_duration'] * 1.2).mean()
                    scores['perfectionism_score'] = min(1.0, over_time_rate * 1.5)
                else:
                    scores['perfectionism_score'] = 0.5
            else:
                scores['perfectionism_score'] = 0.5
            
            # Priority style analysis
            if 'priority' in behavior_data.columns:
                priority_counts = behavior_data['priority'].value_counts()
                if len(priority_counts) > 0:
                    high_priority_focus = (priority_counts.get(4, 0) + priority_counts.get(5, 0)) / len(behavior_data)
                    if high_priority_focus > 0.6:
                        scores['priority_style'] = 'importance_first'
                    elif scores.get('procrastination_tendency', 0.5) > 0.7:
                        scores['priority_style'] = 'deadline_driven'
                    else:
                        scores['priority_style'] = 'effort_based'
                else:
                    scores['priority_style'] = 'effort_based'
            else:
                scores['priority_style'] = 'effort_based'
            
            # Work rhythm (from completion times)
            if 'completed_at' in behavior_data.columns:
                completed_times = pd.to_datetime(behavior_data['completed_at']).dt.hour
                if len(completed_times) > 0:
                    avg_hour = completed_times.mean()
                    if avg_hour < 10:
                        scores['work_rhythm'] = 'morning_person'
                    elif avg_hour > 18:
                        scores['work_rhythm'] = 'night_owl'
                    else:
                        scores['work_rhythm'] = 'flexible'
                else:
                    scores['work_rhythm'] = 'flexible'
            else:
                scores['work_rhythm'] = 'flexible'
            
            # Multitasking preference (concurrent tasks, switching patterns)
            # This would require more detailed interaction data
            scores['multitasking_preference'] = 0.5  # Default neutral
            
            # Stress response (based on completion rates during busy periods)
            scores['stress_response'] = 'productive'  # Default
            
            # Motivation type (achievement, progress, completion)
            scores['motivation_type'] = 'progress'  # Default
            
        except Exception as e:
            logger.error(f"Error calculating personality scores: {e}")
            # Return default scores
            scores = {
                'procrastination_tendency': 0.5,
                'perfectionism_score': 0.5,
                'time_optimism': 0.5,
                'priority_style': 'effort_based',
                'work_rhythm': 'flexible',
                'multitasking_preference': 0.5,
                'stress_response': 'productive',
                'motivation_type': 'progress'
            }
        
        return scores

    async def _get_user_behavior_data(self, db: AsyncSession, user_id: str) -> pd.DataFrame:
        """Get comprehensive user behavior data from database"""
        try:
            # This would fetch from your tasks and behavior tables
            # For now, returning a placeholder structure
            query = select(UserBehaviorPattern).where(
                UserBehaviorPattern.user_id == user_id
            ).order_by(UserBehaviorPattern.created_at.desc()).limit(1000)
            
            result = await db.execute(query)
            patterns = result.scalars().all()
            
            # Convert to DataFrame
            data = []
            for pattern in patterns:
                pattern_data = pattern.pattern_data if pattern.pattern_data else {}
                data.append({
                    'user_id': pattern.user_id,
                    'pattern_type': pattern.pattern_type,
                    'created_at': pattern.created_at,
                    **pattern_data  # Spread the JSON data
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get behavior data for {user_id}: {e}")
            return pd.DataFrame()

    async def _save_user_personality(self, personality: UserPersonality):
        """Save user personality to database"""
        try:
            async with get_async_session() as db:
                # Store as behavior pattern
                personality_data = asdict(personality)
                personality_data.pop('user_id')
                personality_data['updated_at'] = personality_data['updated_at'].isoformat()
                
                pattern = UserBehaviorPattern(
                    user_id=personality.user_id,
                    pattern_type='personality_profile',
                    pattern_data=personality_data,
                    confidence_score=1.0,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # Upsert pattern
                existing_query = select(UserBehaviorPattern).where(
                    and_(
                        UserBehaviorPattern.user_id == personality.user_id,
                        UserBehaviorPattern.pattern_type == 'personality_profile'
                    )
                )
                result = await db.execute(existing_query)
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.pattern_data = personality_data
                    existing.updated_at = datetime.now()
                else:
                    db.add(pattern)
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to save personality for {personality.user_id}: {e}")

    async def _load_user_personalities(self):
        """Load existing user personalities from database"""
        try:
            async with get_async_session() as db:
                query = select(UserBehaviorPattern).where(
                    UserBehaviorPattern.pattern_type == 'personality_profile'
                )
                result = await db.execute(query)
                patterns = result.scalars().all()
                
                for pattern in patterns:
                    try:
                        personality_data = pattern.pattern_data
                        personality_data['updated_at'] = datetime.fromisoformat(
                            personality_data['updated_at']
                        )
                        
                        personality = UserPersonality(
                            user_id=pattern.user_id,
                            **personality_data
                        )
                        self.user_personalities[pattern.user_id] = personality
                        
                    except Exception as e:
                        logger.error(f"Failed to load personality for {pattern.user_id}: {e}")
                        
                logger.info(f"Loaded {len(self.user_personalities)} user personalities")
                
        except Exception as e:
            logger.error(f"Failed to load user personalities: {e}")

    async def _initialize_analyzers(self):
        """Initialize behavioral analysis components"""
        try:
            # Initialize clustering for user segmentation
            self.behavior_analyzers['user_clusters'] = KMeans(n_clusters=5, random_state=42)
            
            # Initialize pattern detection models
            self.behavior_analyzers['anomaly_detector'] = None  # Would use IsolationForest
            
            logger.info("✅ Behavior analyzers initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize analyzers: {e}")

    async def _adaptation_worker(self):
        """Background worker for processing model adaptations"""
        while True:
            try:
                # Process adaptation requests
                adaptation_job = await self.adaptation_queue.get()
                
                user_id = adaptation_job['user_id']
                logger.info(f"Processing adaptation for user {user_id}")
                
                # Perform adaptation
                await self.adapt_user_model(user_id)
                
                self.adaptation_queue.task_done()
                
            except Exception as e:
                logger.error(f"Adaptation worker error: {e}")
                await asyncio.sleep(5)

    async def schedule_adaptation(self, user_id: str):
        """Schedule a model adaptation for a user"""
        try:
            await self.adaptation_queue.put({
                'user_id': user_id,
                'timestamp': datetime.now()
            })
            logger.info(f"Scheduled adaptation for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to schedule adaptation for {user_id}: {e}")

    async def get_user_insights(self, user_id: str, limit: int = 10) -> List[BehaviorInsight]:
        """Get cached or fresh user insights"""
        try:
            # Check cache first
            if user_id in self.insight_cache and self.insight_cache[user_id]:
                return self.insight_cache[user_id][:limit]
            
            # Generate fresh insights
            insights = await self.analyze_user_behavior(user_id)
            return insights[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get insights for {user_id}: {e}")
            return []

    async def cleanup(self):
        """Cleanup personalization engine resources"""
        logger.info("🧹 Cleaning up Personalization Engine...")
        
        self.user_personalities.clear()
        self.behavior_analyzers.clear()
        self.insight_cache.clear()
        
        logger.info("✅ Personalization Engine cleanup complete")

# Global personalization engine instance
personalization_engine = PersonalizationEngine()

async def get_personalization_engine() -> PersonalizationEngine:
    """Get the global personalization engine instance"""
    return personalization_engine