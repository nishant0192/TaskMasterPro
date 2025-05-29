# ai-service/app/services/local_ai/smart_scheduler.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import random
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.core.database import get_async_session
from app.models.database import (
    UserBehaviorPattern, TaskEmbedding, UserProductivityMetrics,
    AIPrediction, UserInteractionLog
)
from app.models.schemas import (
    TaskBase, CalendarEvent, UserPreferences, TimeRange, 
    ScheduleTimeBlock, ProductivityInsight
)
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class UserProductivityPattern:
    """Personalized productivity patterns for each user"""
    peak_hours: List[int]
    low_energy_hours: List[int]
    preferred_break_duration: int
    max_focus_duration: int
    task_switching_penalty: float
    deadline_stress_factor: float
    morning_person_score: float
    collaboration_preference: float
    optimal_task_batch_size: int
    context_switch_recovery_time: int

@dataclass
class SchedulingResult:
    optimized_schedule: List[ScheduleTimeBlock]
    suggested_time_blocks: List[ScheduleTimeBlock]
    productivity_insights: List[ProductivityInsight]
    alternative_schedules: List[List[ScheduleTimeBlock]]
    optimization_score: float
    processing_time_ms: int

class GeneticScheduler:
    """Genetic algorithm for optimal task scheduling"""
    
    def __init__(self, productivity_profile: UserProductivityPattern, 
                 calendar_events: List[CalendarEvent], user_preferences: UserPreferences):
        self.productivity_profile = productivity_profile
        self.calendar_events = calendar_events
        self.user_preferences = user_preferences
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

    def optimize(self, tasks: List[TaskBase], time_estimates: Dict[str, int]) -> List[ScheduleTimeBlock]:
        """Run genetic algorithm to find optimal schedule"""
        # Initialize population
        population = self._initialize_population(tasks, time_estimates)
        
        best_fitness = float('-inf')
        best_schedule = None
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._calculate_fitness(schedule) for schedule in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_schedule = population[max_fitness_idx].copy()
            
            # Selection
            parents = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._crossover(parents[i], parents[i + 1])
                    new_population.extend([
                        self._mutate(child1), 
                        self._mutate(child2)
                    ])
                else:
                    new_population.append(self._mutate(parents[i]))
            
            population = new_population[:self.population_size]
        
        return best_schedule or []

    def _initialize_population(self, tasks: List[TaskBase], 
                             time_estimates: Dict[str, int]) -> List[List[ScheduleTimeBlock]]:
        """Initialize random population of schedules"""
        population = []
        available_slots = self._get_available_time_slots()
        
        for _ in range(self.population_size):
            schedule = self._create_random_schedule(tasks, time_estimates, available_slots)
            population.append(schedule)
        
        return population

    def _create_random_schedule(self, tasks: List[TaskBase], time_estimates: Dict[str, int],
                              available_slots: List[Tuple[datetime, datetime]]) -> List[ScheduleTimeBlock]:
        """Create a random valid schedule"""
        schedule = []
        remaining_slots = available_slots.copy()
        task_list = tasks.copy()
        random.shuffle(task_list)
        
        for task in task_list:
            duration = time_estimates.get(task.id, task.estimated_duration or 60)
            
            # Find a slot that can fit this task
            suitable_slots = []
            for i, (start, end) in enumerate(remaining_slots):
                if (end - start).total_seconds() / 60 >= duration:
                    suitable_slots.append(i)
            
            if suitable_slots:
                slot_idx = random.choice(suitable_slots)
                slot_start, slot_end = remaining_slots[slot_idx]
                
                # Schedule task at beginning of slot
                task_end = slot_start + timedelta(minutes=duration)
                
                schedule.append(ScheduleTimeBlock(
                    task_id=task.id,
                    start_time=slot_start,
                    end_time=task_end,
                    confidence_score=0.5,
                    flexibility_score=0.5,
                    energy_level_required=self._get_energy_requirement(task)
                ))
                
                # Update remaining slots
                remaining_slots[slot_idx] = (task_end, slot_end)
                if (slot_end - task_end).total_seconds() / 60 < 30:  # Less than 30min left
                    remaining_slots.pop(slot_idx)
        
        return schedule

    def _get_available_time_slots(self) -> List[Tuple[datetime, datetime]]:
        """Get available time slots considering calendar events"""
        # This would integrate with calendar events
        # For now, create basic work hours
        slots = []
        current_date = datetime.now().date()
        
        for day_offset in range(7):  # Next 7 days
            day = current_date + timedelta(days=day_offset)
            if day.weekday() in self.user_preferences.work_days:
                work_start = datetime.combine(day, time(self.user_preferences.work_hours_start))
                work_end = datetime.combine(day, time(self.user_preferences.work_hours_end))
                
                # Split into blocks avoiding calendar events
                current_time = work_start
                for event in self.calendar_events:
                    if event.start_time.date() == day and event.is_busy:
                        if current_time < event.start_time:
                            slots.append((current_time, event.start_time))
                        current_time = max(current_time, event.end_time)
                
                if current_time < work_end:
                    slots.append((current_time, work_end))
        
        return slots

    def _calculate_fitness(self, schedule: List[ScheduleTimeBlock]) -> float:
        """Calculate fitness score for a schedule"""
        if not schedule:
            return 0
        
        fitness = 0
        
        # Energy alignment bonus
        for block in schedule:
            hour = block.start_time.hour
            if hour in self.productivity_profile.peak_hours:
                if block.energy_level_required == "high":
                    fitness += 10
                else:
                    fitness += 5
            elif hour in self.productivity_profile.low_energy_hours:
                if block.energy_level_required == "low":
                    fitness += 5
                else:
                    fitness -= 5
            else:
                fitness += 3  # Neutral hours
        
        # Task switching penalty
        for i in range(1, len(schedule)):
            time_gap = (schedule[i].start_time - schedule[i-1].end_time).total_seconds() / 60
            if time_gap < self.productivity_profile.context_switch_recovery_time:
                fitness -= self.productivity_profile.task_switching_penalty * 10
        
        # Focus duration bonus
        consecutive_work_time = 0
        for i, block in enumerate(schedule):
            duration = (block.end_time - block.start_time).total_seconds() / 60
            consecutive_work_time += duration
            
            if i < len(schedule) - 1:
                gap = (schedule[i+1].start_time - block.end_time).total_seconds() / 60
                if gap > self.productivity_profile.preferred_break_duration:
                    consecutive_work_time = 0
            
            # Bonus for optimal focus sessions
            if consecutive_work_time <= self.productivity_profile.max_focus_duration:
                fitness += 5
            else:
                fitness -= 3  # Penalty for too long focus sessions
        
        return fitness

    def _selection(self, population: List[List[ScheduleTimeBlock]], 
                  fitness_scores: List[float]) -> List[List[ScheduleTimeBlock]]:
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected

    def _crossover(self, parent1: List[ScheduleTimeBlock], 
                  parent2: List[ScheduleTimeBlock]) -> Tuple[List[ScheduleTimeBlock], List[ScheduleTimeBlock]]:
        """Order crossover for schedules"""
        if random.random() > self.crossover_rate or len(parent1) == 0 or len(parent2) == 0:
            return parent1.copy(), parent2.copy()
        
        # Simple crossover - swap random portions
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2

    def _mutate(self, schedule: List[ScheduleTimeBlock]) -> List[ScheduleTimeBlock]:
        """Mutate schedule by randomly adjusting task times"""
        if random.random() > self.mutation_rate or len(schedule) == 0:
            return schedule
        
        # Random time shift mutation
        mutated = schedule.copy()
        task_idx = random.randint(0, len(mutated) - 1)
        
        # Shift task by random amount (within constraints)
        time_shift = random.randint(-60, 60)  # ±1 hour
        mutated[task_idx].start_time += timedelta(minutes=time_shift)
        mutated[task_idx].end_time += timedelta(minutes=time_shift)
        
        return mutated

    def _get_energy_requirement(self, task: TaskBase) -> str:
        """Determine energy requirement for task"""
        if task.priority and task.priority.value >= 4:
            return "high"
        elif task.estimated_duration and task.estimated_duration > 120:
            return "high"
        elif task.category in ["learning", "creative", "planning"]:
            return "high"
        elif task.category in ["admin", "email", "organizing"]:
            return "low"
        else:
            return "medium"

class PersonalizedScheduler:
    """AI-powered personalized scheduling service"""
    
    def __init__(self, user_id: str, model_manager: ModelManager):
        self.user_id = user_id
        self.model_manager = model_manager
        self.is_initialized = False
        self.productivity_profile: Optional[UserProductivityPattern] = None
        self.task_duration_model = None
        self.scheduling_preferences = {}

    async def initialize(self):
        """Initialize the personalized scheduler"""
        try:
            logger.info(f"Initializing personalized scheduler for user {self.user_id}")
            
            # Load or create user productivity profile
            await self._load_or_create_productivity_profile()
            
            # Load task duration prediction model
            await self._load_duration_model()
            
            # Load scheduling preferences
            await self._load_scheduling_preferences()
            
            self.is_initialized = True
            logger.info(f"Personalized scheduler initialized for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize personalized scheduler: {e}")
            raise

    async def _load_or_create_productivity_profile(self):
        """Load user's productivity profile or create default"""
        try:
            async with get_async_session() as session:
                # Load productivity patterns
                query = select(UserBehaviorPattern).where(
                    and_(
                        UserBehaviorPattern.user_id == self.user_id,
                        UserBehaviorPattern.pattern_type.in_([
                            'productivity_hours', 'energy_patterns', 'focus_patterns'
                        ])
                    )
                )
                
                result = await session.execute(query)
                patterns = result.scalars().all()
                
                # Load productivity metrics
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                metrics_query = select(UserProductivityMetrics).where(
                    and_(
                        UserProductivityMetrics.user_id == self.user_id,
                        UserProductivityMetrics.metric_date >= cutoff_date
                    )
                ).order_by(UserProductivityMetrics.metric_date.desc()).limit(30)
                
                result = await session.execute(metrics_query)
                metrics = result.scalars().all()
                
                if metrics and patterns:
                    self.productivity_profile = await self._build_profile_from_data(patterns, metrics)
                else:
                    self.productivity_profile = self._create_default_profile()
                    
        except Exception as e:
            logger.error(f"Error loading productivity profile: {e}")
            self.productivity_profile = self._create_default_profile()

    async def _build_profile_from_data(self, patterns: List, metrics: List) -> UserProductivityPattern:
        """Build productivity profile from user data"""
        # Aggregate peak hours from metrics
        all_peak_hours = []
        total_focus_time = []
        productivity_scores = []
        
        for metric in metrics:
            if metric.peak_productivity_hours:
                all_peak_hours.extend(metric.peak_productivity_hours)
            if metric.total_focus_time:
                total_focus_time.append(metric.total_focus_time)
            if metric.productivity_score:
                productivity_scores.append(metric.productivity_score)
        
        # Calculate most common peak hours
        if all_peak_hours:
            hour_counts = defaultdict(int)
            for hour in all_peak_hours:
                hour_counts[hour] += 1
            peak_hours = [hour for hour, count in hour_counts.items() 
                         if count >= len(metrics) * 0.3]
        else:
            peak_hours = [9, 10, 14, 15]
        
        # Calculate average focus duration
        avg_focus_duration = int(np.mean(total_focus_time)) if total_focus_time else 90
        
        # Determine morning person score
        morning_hours = [h for h in peak_hours if h < 12]
        evening_hours = [h for h in peak_hours if h >= 17]
        morning_person_score = len(morning_hours) / (len(morning_hours) + len(evening_hours) + 1)
        
        # Extract patterns data
        pattern_data = {}
        for pattern in patterns:
            pattern_data[pattern.pattern_type] = pattern.pattern_data or {}
        
        return UserProductivityPattern(
            peak_hours=peak_hours,
            low_energy_hours=pattern_data.get('energy_patterns', {}).get('low_hours', [13, 16, 17]),
            preferred_break_duration=pattern_data.get('focus_patterns', {}).get('break_duration', 15),
            max_focus_duration=min(avg_focus_duration, 120),
            task_switching_penalty=pattern_data.get('focus_patterns', {}).get('switching_penalty', 0.2),
            deadline_stress_factor=0.3,
            morning_person_score=morning_person_score,
            collaboration_preference=0.5,
            optimal_task_batch_size=pattern_data.get('productivity_hours', {}).get('batch_size', 3),
            context_switch_recovery_time=pattern_data.get('focus_patterns', {}).get('recovery_time', 10)
        )

    def _create_default_profile(self) -> UserProductivityPattern:
        """Create default productivity profile for new users"""
        return UserProductivityPattern(
            peak_hours=[9, 10, 14, 15],
            low_energy_hours=[13, 16, 17],
            preferred_break_duration=15,
            max_focus_duration=90,
            task_switching_penalty=0.2,
            deadline_stress_factor=0.3,
            morning_person_score=0.6,
            collaboration_preference=0.5,
            optimal_task_batch_size=3,
            context_switch_recovery_time=10
        )

    async def _load_duration_model(self):
        """Load task duration prediction model"""
        try:
            self.task_duration_model = await self.model_manager.get_user_model(
                self.user_id, 'scheduling', 'duration_prediction'
            )
            
            if not self.task_duration_model:
                # Create simple duration model
                await self._create_duration_model()
                
        except Exception as e:
            logger.error(f"Error loading duration model: {e}")

    async def _create_duration_model(self):
        """Create task duration prediction model"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Collect historical task duration data
            duration_data = await self._collect_duration_training_data()
            
            if len(duration_data) >= 10:
                # Train duration prediction model
                features = ['category_encoded', 'priority', 'estimated_duration', 
                           'created_hour', 'created_day_of_week']
                
                X = duration_data[features]
                y = duration_data['actual_duration']
                
                # Handle missing values
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())
                
                # Train model
                self.task_duration_model = RandomForestRegressor(
                    n_estimators=50, random_state=42
                )
                self.task_duration_model.fit(X, y)
                
                # Save model
                await self.model_manager.save_user_model(
                    self.user_id, 'scheduling', self.task_duration_model,
                    model_subtype='duration_prediction',
                    metadata={'features': features},
                    training_samples=len(duration_data)
                )
            
        except Exception as e:
            logger.error(f"Error creating duration model: {e}")

    async def _collect_duration_training_data(self) -> pd.DataFrame:
        """Collect training data for duration prediction"""
        try:
            async with get_async_session() as session:
                query = select(TaskEmbedding).where(
                    and_(
                        TaskEmbedding.user_id == self.user_id,
                        TaskEmbedding.actual_duration.isnot(None),
                        TaskEmbedding.estimated_duration.isnot(None)
                    )
                ).limit(500)
                
                result = await session.execute(query)
                records = result.scalars().all()
                
                data = []
                for record in records:
                    # Category encoding
                    category_mapping = {
                        'work': 1, 'personal': 2, 'health': 3, 'learning': 4,
                        'social': 5, 'finance': 6, 'shopping': 7, 'travel': 8
                    }
                    
                    data.append({
                        'category_encoded': category_mapping.get(record.category, 0),
                        'priority': record.priority or 3,
                        'estimated_duration': record.estimated_duration,
                        'actual_duration': record.actual_duration,
                        'created_hour': record.created_hour or 9,
                        'created_day_of_week': record.created_day_of_week or 1
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"Error collecting duration training data: {e}")
            return pd.DataFrame()

    async def _load_scheduling_preferences(self):
        """Load user scheduling preferences"""
        try:
            async with get_async_session() as session:
                query = select(UserBehaviorPattern).where(
                    and_(
                        UserBehaviorPattern.user_id == self.user_id,
                        UserBehaviorPattern.pattern_type == 'scheduling_preferences'
                    )
                )
                
                result = await session.execute(query)
                pattern = result.scalar_one_or_none()
                
                if pattern:
                    self.scheduling_preferences = pattern.pattern_data or {}
                else:
                    self.scheduling_preferences = {
                        'prefer_morning_important_tasks': True,
                        'batch_similar_tasks': True,
                        'avoid_context_switching': True,
                        'buffer_time_percentage': 0.2
                    }
                    
        except Exception as e:
            logger.error(f"Error loading scheduling preferences: {e}")

    async def optimize_schedule(self, tasks: List[TaskBase], user_id: str,
                              calendar_events: List[CalendarEvent], 
                              preferences: UserPreferences, time_range: TimeRange,
                              db: AsyncSession) -> SchedulingResult:
        """Optimize user's schedule using AI"""
        start_time = datetime.utcnow()
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Predict task durations
            time_estimates = await self._predict_task_durations(tasks)
            
            # Generate schedule using genetic algorithm
            genetic_scheduler = GeneticScheduler(
                self.productivity_profile, calendar_events, preferences
            )
            
            optimized_schedule = genetic_scheduler.optimize(tasks, time_estimates)
            
            # Generate alternative schedules using different strategies
            alternatives = await self._generate_alternative_schedules(
                tasks, time_estimates, calendar_events, preferences
            )
            
            # Generate productivity insights
            insights = await self._generate_productivity_insights(
                optimized_schedule, tasks, time_estimates
            )
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                optimized_schedule, tasks, time_estimates
            )
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Learn from this scheduling session
            await self._learn_from_scheduling(optimized_schedule, tasks, preferences)
            
            return SchedulingResult(
                optimized_schedule=optimized_schedule,
                suggested_time_blocks=optimized_schedule,
                productivity_insights=insights,
                alternative_schedules=alternatives,
                optimization_score=optimization_score,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error optimizing schedule: {e}")
            raise

    async def _predict_task_durations(self, tasks: List[TaskBase]) -> Dict[str, int]:
        """Predict actual durations for tasks"""
        time_estimates = {}
        
        for task in tasks:
            if self.task_duration_model:
                try:
                    # Extract features
                    category_mapping = {
                        'work': 1, 'personal': 2, 'health': 3, 'learning': 4,
                        'social': 5, 'finance': 6, 'shopping': 7, 'travel': 8
                    }
                    
                    current_time = datetime.utcnow()
                    features = np.array([[
                        category_mapping.get(task.category, 0),
                        task.priority.value if task.priority else 3,
                        task.estimated_duration or 60,
                        current_time.hour,
                        current_time.weekday()
                    ]])
                    
                    predicted_duration = self.task_duration_model.predict(features)[0]
                    time_estimates[task.id] = max(15, int(predicted_duration))  # Minimum 15 minutes
                    
                except Exception as e:
                    logger.warning(f"Error predicting duration for task {task.id}: {e}")
                    time_estimates[task.id] = task.estimated_duration or 60
            else:
                # Use estimated duration or category-based defaults
                if task.estimated_duration:
                    time_estimates[task.id] = task.estimated_duration
                else:
                    category_defaults = {
                        'email': 30, 'meeting': 60, 'coding': 120, 'admin': 45,
                        'learning': 90, 'creative': 120, 'planning': 60
                    }
                    time_estimates[task.id] = category_defaults.get(task.category, 60)
        
        return time_estimates

    async def _generate_alternative_schedules(self, tasks: List[TaskBase], 
                                            time_estimates: Dict[str, int],
                                            calendar_events: List[CalendarEvent],
                                            preferences: UserPreferences) -> List[List[ScheduleTimeBlock]]:
        """Generate alternative scheduling strategies"""
        alternatives = []
        
        try:
            # Alternative 1: Priority-first scheduling
            priority_schedule = await self._schedule_by_priority(
                tasks, time_estimates, calendar_events, preferences
            )
            if priority_schedule:
                alternatives.append(priority_schedule)
            
            # Alternative 2: Deadline-driven scheduling
            deadline_schedule = await self._schedule_by_deadlines(
                tasks, time_estimates, calendar_events, preferences
            )
            if deadline_schedule:
                alternatives.append(deadline_schedule)
            
            # Alternative 3: Energy-optimized scheduling
            energy_schedule = await self._schedule_by_energy(
                tasks, time_estimates, calendar_events, preferences
            )
            if energy_schedule:
                alternatives.append(energy_schedule)
                
        except Exception as e:
            logger.error(f"Error generating alternative schedules: {e}")
        
        return alternatives[:3]  # Return top 3 alternatives

    async def _schedule_by_priority(self, tasks: List[TaskBase], 
                                  time_estimates: Dict[str, int],
                                  calendar_events: List[CalendarEvent],
                                  preferences: UserPreferences) -> List[ScheduleTimeBlock]:
        """Schedule tasks by priority order"""
        schedule = []
        available_slots = self._get_available_slots(calendar_events, preferences)
        
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, 
                            key=lambda x: x.priority.value if x.priority else 3, 
                            reverse=True)
        
        for task in sorted_tasks:
            duration = time_estimates.get(task.id, 60)
            slot = self._find_best_slot(available_slots, duration, task)
            
            if slot:
                start_time, end_time = slot
                schedule.append(ScheduleTimeBlock(
                    task_id=task.id,
                    start_time=start_time,
                    end_time=end_time,
                    confidence_score=0.8,
                    flexibility_score=0.6,
                    energy_level_required=self._get_energy_requirement(task)
                ))
                
                # Update available slots
                available_slots = self._update_available_slots(available_slots, start_time, end_time)
        
        return schedule

    async def _schedule_by_deadlines(self, tasks: List[TaskBase], 
                                   time_estimates: Dict[str, int],
                                   calendar_events: List[CalendarEvent],
                                   preferences: UserPreferences) -> List[ScheduleTimeBlock]:
        """Schedule tasks by deadline urgency"""
        schedule = []
        available_slots = self._get_available_slots(calendar_events, preferences)
        
        # Sort tasks by deadline urgency
        tasks_with_deadlines = [t for t in tasks if t.due_date]
        tasks_without_deadlines = [t for t in tasks if not t.due_date]
        
        sorted_tasks = sorted(tasks_with_deadlines, key=lambda x: x.due_date)
        sorted_tasks.extend(tasks_without_deadlines)  # Add non-deadline tasks at end
        
        for task in sorted_tasks:
            duration = time_estimates.get(task.id, 60)
            
            if task.due_date:
                # For deadline tasks, try to schedule closer to deadline but with buffer
                buffer_hours = duration / 60 * 1.5  # 50% buffer
                latest_start = task.due_date - timedelta(hours=buffer_hours)
                slot = self._find_slot_before_deadline(available_slots, duration, latest_start)
            else:
                slot = self._find_best_slot(available_slots, duration, task)
            
            if slot:
                start_time, end_time = slot
                schedule.append(ScheduleTimeBlock(
                    task_id=task.id,
                    start_time=start_time,
                    end_time=end_time,
                    confidence_score=0.9 if task.due_date else 0.6,
                    flexibility_score=0.3 if task.due_date else 0.8,
                    energy_level_required=self._get_energy_requirement(task)
                ))
                
                available_slots = self._update_available_slots(available_slots, start_time, end_time)
        
        return schedule

    async def _schedule_by_energy(self, tasks: List[TaskBase], 
                                time_estimates: Dict[str, int],
                                calendar_events: List[CalendarEvent],
                                preferences: UserPreferences) -> List[ScheduleTimeBlock]:
        """Schedule tasks optimized for energy levels"""
        schedule = []
        available_slots = self._get_available_slots(calendar_events, preferences)
        
        # Categorize tasks by energy requirement
        high_energy_tasks = [t for t in tasks if self._get_energy_requirement(t) == "high"]
        medium_energy_tasks = [t for t in tasks if self._get_energy_requirement(t) == "medium"]
        low_energy_tasks = [t for t in tasks if self._get_energy_requirement(t) == "low"]
        
        # Schedule high-energy tasks during peak hours
        for task in high_energy_tasks:
            duration = time_estimates.get(task.id, 60)
            slot = self._find_peak_hour_slot(available_slots, duration)
            
            if slot:
                start_time, end_time = slot
                schedule.append(ScheduleTimeBlock(
                    task_id=task.id,
                    start_time=start_time,
                    end_time=end_time,
                    confidence_score=0.9,
                    flexibility_score=0.4,
                    energy_level_required="high"
                ))
                available_slots = self._update_available_slots(available_slots, start_time, end_time)
        
        # Schedule medium-energy tasks in good hours
        for task in medium_energy_tasks:
            duration = time_estimates.get(task.id, 60)
            slot = self._find_good_hour_slot(available_slots, duration)
            
            if slot:
                start_time, end_time = slot
                schedule.append(ScheduleTimeBlock(
                    task_id=task.id,
                    start_time=start_time,
                    end_time=end_time,
                    confidence_score=0.7,
                    flexibility_score=0.6,
                    energy_level_required="medium"
                ))
                available_slots = self._update_available_slots(available_slots, start_time, end_time)
        
        # Schedule low-energy tasks anytime
        for task in low_energy_tasks:
            duration = time_estimates.get(task.id, 60)
            slot = self._find_best_slot(available_slots, duration, task)
            
            if slot:
                start_time, end_time = slot
                schedule.append(ScheduleTimeBlock(
                    task_id=task.id,
                    start_time=start_time,
                    end_time=end_time,
                    confidence_score=0.6,
                    flexibility_score=0.8,
                    energy_level_required="low"
                ))
                available_slots = self._update_available_slots(available_slots, start_time, end_time)
        
        return schedule

    def _get_available_slots(self, calendar_events: List[CalendarEvent], 
                           preferences: UserPreferences) -> List[Tuple[datetime, datetime]]:
        """Get available time slots considering calendar events"""
        slots = []
        current_date = datetime.now().date()
        
        for day_offset in range(7):  # Next 7 days
            day = current_date + timedelta(days=day_offset)
            if day.weekday() in preferences.work_days:
                work_start = datetime.combine(day, time(preferences.work_hours_start))
                work_end = datetime.combine(day, time(preferences.work_hours_end))
                
                # Get busy periods for this day
                busy_periods = []
                for event in calendar_events:
                    if event.start_time.date() == day and event.is_busy:
                        busy_periods.append((event.start_time, event.end_time))
                
                # Sort busy periods by start time
                busy_periods.sort()
                
                # Find free slots between busy periods
                current_time = work_start
                for busy_start, busy_end in busy_periods:
                    if current_time < busy_start:
                        slots.append((current_time, busy_start))
                    current_time = max(current_time, busy_end)
                
                # Add remaining time after last busy period
                if current_time < work_end:
                    slots.append((current_time, work_end))
        
        return slots

    def _find_best_slot(self, available_slots: List[Tuple[datetime, datetime]], 
                       duration: int, task: TaskBase) -> Optional[Tuple[datetime, datetime]]:
        """Find the best available slot for a task"""
        best_slot = None
        best_score = -1
        
        for slot_start, slot_end in available_slots:
            slot_duration = (slot_end - slot_start).total_seconds() / 60
            
            if slot_duration >= duration:
                task_end = slot_start + timedelta(minutes=duration)
                
                # Score this slot
                score = self._score_time_slot(slot_start, task)
                
                if score > best_score:
                    best_score = score
                    best_slot = (slot_start, task_end)
        
        return best_slot

    def _score_time_slot(self, start_time: datetime, task: TaskBase) -> float:
        """Score a time slot for task scheduling"""
        score = 0.5  # Base score
        hour = start_time.hour
        
        # Energy level alignment
        energy_required = self._get_energy_requirement(task)
        if hour in self.productivity_profile.peak_hours:
            if energy_required == "high":
                score += 0.3
            else:
                score += 0.1
        elif hour in self.productivity_profile.low_energy_hours:
            if energy_required == "low":
                score += 0.2
            else:
                score -= 0.2
        
        # Priority factor
        if task.priority:
            priority_bonus = task.priority.value / 5.0 * 0.2
            score += priority_bonus
        
        # Morning person preference
        if hour < 12 and self.productivity_profile.morning_person_score > 0.7:
            score += 0.1
        elif hour >= 12 and self.productivity_profile.morning_person_score < 0.3:
            score += 0.1
        
        # Deadline urgency
        if task.due_date:
            hours_until_due = (task.due_date - start_time).total_seconds() / 3600
            if hours_until_due < 24:
                score += 0.2
            elif hours_until_due < 72:
                score += 0.1
        
        return min(1.0, score)

    def _find_peak_hour_slot(self, available_slots: List[Tuple[datetime, datetime]], 
                           duration: int) -> Optional[Tuple[datetime, datetime]]:
        """Find slot during peak productivity hours"""
        for slot_start, slot_end in available_slots:
            if (slot_start.hour in self.productivity_profile.peak_hours and
                (slot_end - slot_start).total_seconds() / 60 >= duration):
                task_end = slot_start + timedelta(minutes=duration)
                return (slot_start, task_end)
        return None

    def _find_good_hour_slot(self, available_slots: List[Tuple[datetime, datetime]], 
                           duration: int) -> Optional[Tuple[datetime, datetime]]:
        """Find slot during good (not peak, not low) hours"""
        good_hours = [h for h in range(24) 
                     if h not in self.productivity_profile.peak_hours 
                     and h not in self.productivity_profile.low_energy_hours]
        
        for slot_start, slot_end in available_slots:
            if (slot_start.hour in good_hours and
                (slot_end - slot_start).total_seconds() / 60 >= duration):
                task_end = slot_start + timedelta(minutes=duration)
                return (slot_start, task_end)
        return None

    def _find_slot_before_deadline(self, available_slots: List[Tuple[datetime, datetime]], 
                                 duration: int, latest_start: datetime) -> Optional[Tuple[datetime, datetime]]:
        """Find slot that completes before deadline"""
        suitable_slots = []
        
        for slot_start, slot_end in available_slots:
            if (slot_start <= latest_start and
                (slot_end - slot_start).total_seconds() / 60 >= duration):
                task_end = slot_start + timedelta(minutes=duration)
                suitable_slots.append((slot_start, task_end))
        
        # Return the latest possible slot (closest to deadline)
        if suitable_slots:
            return max(suitable_slots, key=lambda x: x[0])
        
        return None

    def _update_available_slots(self, available_slots: List[Tuple[datetime, datetime]], 
                              scheduled_start: datetime, scheduled_end: datetime) -> List[Tuple[datetime, datetime]]:
        """Update available slots after scheduling a task"""
        updated_slots = []
        
        for slot_start, slot_end in available_slots:
            # If slot doesn't overlap with scheduled time, keep it
            if slot_end <= scheduled_start or slot_start >= scheduled_end:
                updated_slots.append((slot_start, slot_end))
            else:
                # Split slot if needed
                if slot_start < scheduled_start:
                    # Keep the part before scheduled task
                    if (scheduled_start - slot_start).total_seconds() / 60 >= 30:  # Min 30 minutes
                        updated_slots.append((slot_start, scheduled_start))
                
                if slot_end > scheduled_end:
                    # Keep the part after scheduled task
                    if (slot_end - scheduled_end).total_seconds() / 60 >= 30:  # Min 30 minutes
                        updated_slots.append((scheduled_end, slot_end))
        
        return updated_slots

    def _get_energy_requirement(self, task: TaskBase) -> str:
        """Determine energy requirement for task"""
        if task.priority and task.priority.value >= 4:
            return "high"
        elif task.estimated_duration and task.estimated_duration > 120:
            return "high"
        elif task.category in ["learning", "creative", "planning", "coding", "writing"]:
            return "high"
        elif task.category in ["admin", "email", "organizing", "routine"]:
            return "low"
        else:
            return "medium"

    async def _generate_productivity_insights(self, schedule: List[ScheduleTimeBlock],
                                            tasks: List[TaskBase], 
                                            time_estimates: Dict[str, int]) -> List[ProductivityInsight]:
        """Generate insights about the optimized schedule"""
        insights = []
        
        if not schedule:
            return insights
        
        # Peak hour utilization insight
        peak_hour_blocks = sum(1 for block in schedule 
                             if block.start_time.hour in self.productivity_profile.peak_hours)
        peak_utilization = peak_hour_blocks / len(self.productivity_profile.peak_hours) \
                          if self.productivity_profile.peak_hours else 0
        
        insights.append(ProductivityInsight(
            metric="peak_hour_utilization",
            value=peak_utilization,
            trend="stable",
            description=f"Peak hours utilization: {peak_utilization:.1%}",
            recommendation="Schedule most important tasks during peak hours for better efficiency"
        ))
        
        # Energy-task alignment insight
        high_energy_tasks = sum(1 for block in schedule if block.energy_level_required == "high")
        high_energy_in_peak = sum(1 for block in schedule 
                                if block.energy_level_required == "high" 
                                and block.start_time.hour in self.productivity_profile.peak_hours)
        
        energy_alignment = high_energy_in_peak / high_energy_tasks if high_energy_tasks > 0 else 1
        
        insights.append(ProductivityInsight(
            metric="energy_task_alignment",
            value=energy_alignment,
            trend="increasing" if energy_alignment > 0.7 else "stable",
            description=f"Energy-task alignment: {energy_alignment:.1%}",
            recommendation="High-energy tasks are well-aligned with your peak hours"
        ))
        
        # Focus session optimization
        total_work_time = sum((block.end_time - block.start_time).total_seconds() / 60 
                            for block in schedule)
        avg_session_length = total_work_time / len(schedule) if schedule else 0
        
        insights.append(ProductivityInsight(
            metric="focus_session_length",
            value=avg_session_length,
            trend="stable",
            description=f"Average focus session: {avg_session_length:.0f} minutes",
            recommendation=f"Optimal focus sessions for you are around {self.productivity_profile.max_focus_duration} minutes"
        ))
        
        # Deadline pressure insight
        deadline_tasks = [task for task in tasks if task.due_date]
        if deadline_tasks:
            urgent_tasks = sum(1 for task in deadline_tasks 
                             if task.due_date and 
                             (task.due_date - datetime.utcnow()).total_seconds() < 24 * 3600)
            
            deadline_pressure = urgent_tasks / len(deadline_tasks)
            
            insights.append(ProductivityInsight(
                metric="deadline_pressure",
                value=deadline_pressure,
                trend="increasing" if deadline_pressure > 0.3 else "stable",
                description=f"Deadline pressure: {deadline_pressure:.1%} of tasks due within 24h",
                recommendation="Consider scheduling buffer time for urgent tasks"
            ))
        
        return insights

    def _calculate_optimization_score(self, schedule: List[ScheduleTimeBlock], 
                                    tasks: List[TaskBase], 
                                    time_estimates: Dict[str, int]) -> float:
        """Calculate overall optimization score for the schedule"""
        if not schedule:
            return 0.0
        
        score = 0.0
        total_factors = 0
        
        # Energy alignment score
        energy_aligned = 0
        for block in schedule:
            hour = block.start_time.hour
            energy_req = block.energy_level_required
            
            if hour in self.productivity_profile.peak_hours and energy_req == "high":
                energy_aligned += 1
            elif hour not in self.productivity_profile.low_energy_hours and energy_req == "medium":
                energy_aligned += 0.8
            elif energy_req == "low":
                energy_aligned += 0.6
        
        energy_score = energy_aligned / len(schedule)
        score += energy_score * 0.3
        total_factors += 0.3
        
        # Time utilization score
        total_scheduled_time = sum((block.end_time - block.start_time).total_seconds() / 60 
                                 for block in schedule)
        total_estimated_time = sum(time_estimates.values())
        
        if total_estimated_time > 0:
            utilization_score = min(1.0, total_scheduled_time / total_estimated_time)
            score += utilization_score * 0.2
            total_factors += 0.2
        
        # Priority alignment score
        priority_scores = []
        for block in schedule:
            task = next((t for t in tasks if t.id == block.task_id), None)
            if task and task.priority:
                # Higher priority tasks should be scheduled earlier in optimal times
                hour_score = 1.0 if block.start_time.hour in self.productivity_profile.peak_hours else 0.5
                priority_factor = task.priority.value / 5.0
                priority_scores.append(hour_score * priority_factor)
        
        if priority_scores:
            priority_alignment = np.mean(priority_scores)
            score += priority_alignment * 0.3
            total_factors += 0.3
        
        # Deadline adherence score
        deadline_scores = []
        for block in schedule:
            task = next((t for t in tasks if t.id == block.task_id), None)
            if task and task.due_date:
                time_until_due = (task.due_date - block.end_time).total_seconds() / 3600
                if time_until_due >= 0:
                    deadline_scores.append(1.0)
                else:
                    deadline_scores.append(0.0)  # Task scheduled after deadline
        
        if deadline_scores:
            deadline_adherence = np.mean(deadline_scores)
            score += deadline_adherence * 0.2
            total_factors += 0.2
        
        return score / total_factors if total_factors > 0 else 0.5

    async def _learn_from_scheduling(self, schedule: List[ScheduleTimeBlock], 
                                   tasks: List[TaskBase], preferences: UserPreferences):
        """Learn from scheduling decisions to improve future recommendations"""
        try:
            # Store scheduling patterns
            scheduling_data = {
                'schedule_length': len(schedule),
                'peak_hour_usage': sum(1 for block in schedule 
                                     if block.start_time.hour in self.productivity_profile.peak_hours),
                'energy_distribution': {
                    'high': sum(1 for block in schedule if block.energy_level_required == "high"),
                    'medium': sum(1 for block in schedule if block.energy_level_required == "medium"),
                    'low': sum(1 for block in schedule if block.energy_level_required == "low")
                },
                'preferences_used': {
                    'work_hours_start': preferences.work_hours_start,
                    'work_hours_end': preferences.work_hours_end,
                    'break_duration': preferences.break_duration
                }
            }
            
            # Update user behavior patterns
            await self._update_scheduling_patterns(scheduling_data)
            
        except Exception as e:
            logger.error(f"Error learning from scheduling: {e}")

    async def _update_scheduling_patterns(self, scheduling_data: Dict[str, Any]):
        """Update user's scheduling behavior patterns"""
        try:
            async with get_async_session() as session:
                query = select(UserBehaviorPattern).where(
                    and_(
                        UserBehaviorPattern.user_id == self.user_id,
                        UserBehaviorPattern.pattern_type == 'scheduling_patterns'
                    )
                )
                
                result = await session.execute(query)
                pattern = result.scalar_one_or_none()
                
                if pattern:
                    # Merge with existing data
                    existing_data = pattern.pattern_data or {}
                    
                    # Update frequency counters
                    for key, value in scheduling_data.items():
                        if key in existing_data:
                            if isinstance(value, dict):
                                existing_data[key].update(value)
                            else:
                                existing_data[key] = (existing_data[key] + value) / 2  # Moving average
                        else:
                            existing_data[key] = value
                    
                    pattern.pattern_data = existing_data
                    pattern.frequency_count += 1
                    pattern.confidence_score = min(1.0, pattern.confidence_score + 0.05)
                    pattern.last_observed = datetime.utcnow()
                    pattern.updated_at = datetime.utcnow()
                else:
                    # Create new pattern
                    new_pattern = UserBehaviorPattern(
                        user_id=self.user_id,
                        pattern_type='scheduling_patterns',
                        pattern_data=scheduling_data,
                        confidence_score=0.6,
                        frequency_count=1
                    )
                    session.add(new_pattern)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error updating scheduling patterns: {e}")

    async def get_scheduling_insights(self) -> Dict[str, Any]:
        """Get insights about user's scheduling patterns"""
        try:
            insights = {
                'productivity_profile': {
                    'peak_hours': self.productivity_profile.peak_hours if self.productivity_profile else [],
                    'low_energy_hours': self.productivity_profile.low_energy_hours if self.productivity_profile else [],
                    'max_focus_duration': self.productivity_profile.max_focus_duration if self.productivity_profile else 90,
                    'morning_person_score': self.productivity_profile.morning_person_score if self.productivity_profile else 0.5
                },
                'scheduling_preferences': self.scheduling_preferences,
                'model_performance': {
                    'duration_model_loaded': self.task_duration_model is not None,
                    'patterns_learned': len(self.scheduling_preferences)
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting scheduling insights: {e}")
            return {}

    async def cleanup(self):
        """Cleanup resources"""
        logger.debug(f"Cleaning up personalized scheduler for user {self.user_id}")
        self.scheduling_preferences.clear()