# app/services/scheduling_optimization.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import random
from collections import defaultdict

from app.models.schemas import (
    TaskBase, CalendarEvent, UserPreferences, TimeRange, 
    ScheduleTimeBlock, ProductivityInsight
)
from app.models.database import UserAnalytics, TaskPrediction
from app.core.exceptions import InsufficientDataException

logger = logging.getLogger(__name__)

@dataclass
class SchedulingResult:
    optimized_schedule: List[ScheduleTimeBlock]
    suggested_time_blocks: List[ScheduleTimeBlock]
    productivity_insights: List[ProductivityInsight]
    alternative_schedules: List[List[ScheduleTimeBlock]]
    optimization_score: float
    processing_time_ms: int

@dataclass
class UserProductivityPattern:
    """Personalized productivity patterns for each user"""
    peak_hours: List[int]  # Hours when user is most productive
    low_energy_hours: List[int]  # Hours when user has low energy
    preferred_break_duration: int  # Minutes
    max_focus_duration: int  # Maximum minutes of continuous work
    task_switching_penalty: float  # Penalty for switching between different task types
    deadline_stress_factor: float  # How much deadlines affect user performance
    morning_person_score: float  # 0-1, higher means more productive in morning
    collaboration_preference: float  # 0-1, preference for collaborative vs solo work

class SchedulingOptimizationService:
    def __init__(self):
        self.is_initialized = False
        self.user_patterns: Dict[str, UserProductivityPattern] = {}
        
    async def initialize(self):
        """Initialize the scheduling optimization service"""
        try:
            logger.info("Initializing Scheduling Optimization Service...")
            
            # Load any cached user patterns
            await self._load_user_patterns()
            
            self.is_initialized = True
            logger.info("Scheduling Optimization Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Scheduling Optimization Service: {e}")
            raise

    async def optimize_schedule(self, tasks: List[TaskBase], user_id: str,
                              calendar_events: List[CalendarEvent], 
                              preferences: UserPreferences, time_range: TimeRange,
                              db: AsyncSession) -> SchedulingResult:
        """Main method to optimize user's schedule"""
        start_time = datetime.now()
        
        if not self.is_initialized:
            raise Exception("Service not initialized")
        
        try:
            # Get or create user productivity pattern
            user_pattern = await self._get_user_pattern(user_id, db)
            
            # Analyze available time slots
            available_slots = self._analyze_available_time(
                calendar_events, preferences, time_range
            )
            
            # Generate multiple scheduling options using different algorithms
            schedules = await self._generate_schedule_options(
                tasks, available_slots, user_pattern, preferences
            )
            
            # Select best schedule
            best_schedule = self._select_best_schedule(schedules, user_pattern)
            
            # Generate alternative schedules
            alternatives = schedules[:3] if len(schedules) > 1 else []
            
            # Create productivity insights
            insights = await self._generate_productivity_insights(
                best_schedule, user_pattern, tasks
            )
            
            # Calculate optimization score
            opt_score = self._calculate_optimization_score(best_schedule, user_pattern)
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return SchedulingResult(
                optimized_schedule=best_schedule,
                suggested_time_blocks=best_schedule,  # For now, same as optimized
                productivity_insights=insights,
                alternative_schedules=alternatives,
                optimization_score=opt_score,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Schedule optimization failed: {e}")
            raise

    async def _get_user_pattern(self, user_id: str, db: AsyncSession) -> UserProductivityPattern:
        """Get or create user productivity pattern"""
        if user_id in self.user_patterns:
            return self.user_patterns[user_id]
        
        # Try to load from database analytics
        pattern = await self._analyze_user_productivity_pattern(user_id, db)
        
        # Cache the pattern
        self.user_patterns[user_id] = pattern
        
        return pattern

    async def _analyze_user_productivity_pattern(self, user_id: str, db: AsyncSession) -> UserProductivityPattern:
        """Analyze user's historical data to determine productivity patterns"""
        try:
            # Get user analytics
            query = select(UserAnalytics).where(
                UserAnalytics.user_id == user_id
            ).order_by(UserAnalytics.date.desc()).limit(90)  # Last 90 days
            
            result = await db.execute(query)
            analytics = result.scalars().all()
            
            if len(analytics) < 7:  # Need at least a week of data
                return self._create_default_pattern()
            
            # Analyze patterns from historical data
            hourly_productivity = defaultdict(list)
            completion_times = []
            deadline_performance = []
            
            for record in analytics:
                # Extract hour-based productivity (would need more detailed data)
                # For now, simulate based on general patterns
                if record.productivity_score:
                    # Simulate hourly breakdown
                    for hour in range(9, 18):  # Work hours
                        productivity = record.productivity_score + random.gauss(0, 0.1)
                        hourly_productivity[hour].append(max(0, min(1, productivity)))
                
                if record.average_completion_time:
                    completion_times.append(record.average_completion_time)
                
                deadline_ratio = record.deadlines_met / max(1, record.deadlines_met + record.deadlines_missed)
                deadline_performance.append(deadline_ratio)
            
            # Calculate peak hours
            peak_hours = []
            for hour, scores in hourly_productivity.items():
                if scores and np.mean(scores) > 0.7:
                    peak_hours.append(hour)
            
            # Default peak hours if none found
            if not peak_hours:
                peak_hours = [9, 10, 14, 15]  # Typical productive hours
            
            # Calculate other patterns
            avg_completion_time = np.mean(completion_times) if completion_times else 60
            avg_deadline_performance = np.mean(deadline_performance) if deadline_performance else 0.8
            
            return UserProductivityPattern(
                peak_hours=peak_hours,
                low_energy_hours=[13, 16, 17],  # Post-lunch and late afternoon
                preferred_break_duration=15,
                max_focus_duration=min(120, int(avg_completion_time * 2)),
                task_switching_penalty=0.1,
                deadline_stress_factor=1.0 - avg_deadline_performance,
                morning_person_score=0.7 if min(peak_hours) < 12 else 0.3,
                collaboration_preference=0.5  # Default neutral
            )
            
        except Exception as e:
            logger.error(f"Error analyzing user pattern: {e}")
            return self._create_default_pattern()

    def _create_default_pattern(self) -> UserProductivityPattern:
        """Create default productivity pattern for new users"""
        return UserProductivityPattern(
            peak_hours=[9, 10, 14, 15],
            low_energy_hours=[13, 16, 17],
            preferred_break_duration=15,
            max_focus_duration=90,
            task_switching_penalty=0.1,
            deadline_stress_factor=0.3,
            morning_person_score=0.6,
            collaboration_preference=0.5
        )

    def _analyze_available_time(self, calendar_events: List[CalendarEvent], 
                               preferences: UserPreferences, time_range: TimeRange) -> List[Tuple[datetime, datetime]]:
        """Analyze available time slots between calendar events"""
        available_slots = []
        
        current_date = time_range.start_date
        end_date = time_range.end_date
        
        while current_date <= end_date:
            # Skip weekends if not in work_days
            if current_date.weekday() not in preferences.work_days:
                current_date += timedelta(days=1)
                continue
            
            # Create work day boundaries
            work_start = datetime.combine(current_date, time(preferences.work_hours_start))
            work_end = datetime.combine(current_date, time(preferences.work_hours_end))
            
            # Get events for this day
            day_events = [
                event for event in calendar_events
                if event.start_time.date() == current_date and event.is_busy
            ]
            
            # Sort events by start time
            day_events.sort(key=lambda x: x.start_time)
            
            # Find gaps between events
            current_time = work_start
            
            for event in day_events:
                # Gap before event
                if current_time < event.start_time:
                    gap_duration = (event.start_time - current_time).total_seconds() / 60
                    if gap_duration >= 30:  # Minimum 30-minute slots
                        available_slots.append((current_time, event.start_time))
                
                current_time = max(current_time, event.end_time)
            
            # Gap after last event until end of work day
            if current_time < work_end:
                gap_duration = (work_end - current_time).total_seconds() / 60
                if gap_duration >= 30:
                    available_slots.append((current_time, work_end))
            
            current_date += timedelta(days=1)
        
        return available_slots

    async def _generate_schedule_options(self, tasks: List[TaskBase], 
                                       available_slots: List[Tuple[datetime, datetime]],
                                       user_pattern: UserProductivityPattern,
                                       preferences: UserPreferences) -> List[List[ScheduleTimeBlock]]:
        """Generate multiple scheduling options using different strategies"""
        schedules = []
        
        # Strategy 1: Priority-based scheduling
        priority_schedule = self._schedule_by_priority(tasks, available_slots, user_pattern)
        if priority_schedule:
            schedules.append(priority_schedule)
        
        # Strategy 2: Energy-optimized scheduling
        energy_schedule = self._schedule_by_energy_levels(tasks, available_slots, user_pattern)
        if energy_schedule:
            schedules.append(energy_schedule)
        
        # Strategy 3: Deadline-driven scheduling
        deadline_schedule = self._schedule_by_deadlines(tasks, available_slots, user_pattern)
        if deadline_schedule:
            schedules.append(deadline_schedule)
        
        # Strategy 4: Balanced approach
        balanced_schedule = self._schedule_balanced(tasks, available_slots, user_pattern)
        if balanced_schedule:
            schedules.append(balanced_schedule)
        
        return schedules

    def _schedule_by_priority(self, tasks: List[TaskBase], 
                            available_slots: List[Tuple[datetime, datetime]],
                            user_pattern: UserProductivityPattern) -> List[ScheduleTimeBlock]:
        """Schedule tasks based on priority levels"""
        schedule = []
        remaining_tasks = tasks.copy()
        remaining_slots = available_slots.copy()
        
        # Sort tasks by priority (highest first)
        remaining_tasks.sort(key=lambda x: x.priority.value if x.priority else 3, reverse=True)
        
        for task in remaining_tasks:
            duration = task.estimated_duration or 60
            
            # Find best slot for this task
            best_slot = self._find_best_slot(task, remaining_slots, user_pattern, duration)
            
            if best_slot:
                slot_start, slot_end = best_slot
                task_end = slot_start + timedelta(minutes=duration)
                
                if task_end <= slot_end:
                    # Create time block
                    time_block = ScheduleTimeBlock(
                        task_id=task.id,
                        start_time=slot_start,
                        end_time=task_end,
                        confidence_score=0.8,
                        flexibility_score=0.6,
                        energy_level_required=self._determine_energy_level(task, user_pattern)
                    )
                    schedule.append(time_block)
                    
                    # Update remaining slots
                    remaining_slots = self._update_slots_after_scheduling(
                        remaining_slots, slot_start, task_end
                    )
        
        return schedule

    def _schedule_by_energy_levels(self, tasks: List[TaskBase], 
                                 available_slots: List[Tuple[datetime, datetime]],
                                 user_pattern: UserProductivityPattern) -> List[ScheduleTimeBlock]:
        """Schedule tasks based on user's energy levels throughout the day"""
        schedule = []
        remaining_tasks = tasks.copy()
        remaining_slots = available_slots.copy()
        
        # Categorize tasks by energy requirement
        high_energy_tasks = []
        medium_energy_tasks = []
        low_energy_tasks = []
        
        for task in remaining_tasks:
            energy_level = self._determine_energy_level(task, user_pattern)
            if energy_level == "high":
                high_energy_tasks.append(task)
            elif energy_level == "medium":
                medium_energy_tasks.append(task)
            else:
                low_energy_tasks.append(task)
        
        # Schedule high-energy tasks during peak hours
        for task in high_energy_tasks:
            best_slot = self._find_peak_hour_slot(task, remaining_slots, user_pattern)
            if best_slot:
                schedule.append(self._create_time_block(task, best_slot, 0.9, 0.4))
                remaining_slots = self._update_slots_after_scheduling(
                    remaining_slots, best_slot[0], best_slot[1]
                )
        
        # Schedule medium-energy tasks in good hours
        for task in medium_energy_tasks:
            best_slot = self._find_good_hour_slot(task, remaining_slots, user_pattern)
            if best_slot:
                schedule.append(self._create_time_block(task, best_slot, 0.7, 0.6))
                remaining_slots = self._update_slots_after_scheduling(
                    remaining_slots, best_slot[0], best_slot[1]
                )
        
        # Schedule low-energy tasks anytime
        for task in low_energy_tasks:
            best_slot = self._find_any_slot(task, remaining_slots)
            if best_slot:
                schedule.append(self._create_time_block(task, best_slot, 0.6, 0.8))
                remaining_slots = self._update_slots_after_scheduling(
                    remaining_slots, best_slot[0], best_slot[1]
                )
        
        return schedule

    def _schedule_by_deadlines(self, tasks: List[TaskBase], 
                             available_slots: List[Tuple[datetime, datetime]],
                             user_pattern: UserProductivityPattern) -> List[ScheduleTimeBlock]:
        """Schedule tasks based on approaching deadlines"""
        schedule = []
        remaining_tasks = [task for task in tasks if task.due_date]  # Only tasks with deadlines
        remaining_slots = available_slots.copy()
        
        # Sort by deadline (earliest first)
        remaining_tasks.sort(key=lambda x: x.due_date)
        
        for task in remaining_tasks:
            # Calculate urgency
            time_until_deadline = (task.due_date - datetime.now()).total_seconds() / 3600  # hours
            urgency_score = max(0, 1 - (time_until_deadline / 168))  # 168 hours = 1 week
            
            # Find slot closest to current time for urgent tasks
            best_slot = self._find_urgent_slot(task, remaining_slots, urgency_score)
            
            if best_slot:
                confidence = min(0.9, 0.5 + urgency_score * 0.4)
                flexibility = max(0.2, 0.8 - urgency_score * 0.6)
                
                schedule.append(self._create_time_block(task, best_slot, confidence, flexibility))
                remaining_slots = self._update_slots_after_scheduling(
                    remaining_slots, best_slot[0], best_slot[1]
                )
        
        return schedule

    def _schedule_balanced(self, tasks: List[TaskBase], 
                         available_slots: List[Tuple[datetime, datetime]],
                         user_pattern: UserProductivityPattern) -> List[ScheduleTimeBlock]:
        """Balanced scheduling considering multiple factors"""
        schedule = []
        remaining_tasks = tasks.copy()
        remaining_slots = available_slots.copy()
        
        # Score each task-slot combination
        task_slot_scores = []
        
        for task in remaining_tasks:
            duration = task.estimated_duration or 60
            
            for slot_start, slot_end in remaining_slots:
                if slot_start + timedelta(minutes=duration) <= slot_end:
                    score = self._calculate_task_slot_score(task, slot_start, user_pattern)
                    task_slot_scores.append((score, task, slot_start, slot_start + timedelta(minutes=duration)))
        
        # Sort by score (highest first)
        task_slot_scores.sort(key=lambda x: x[0], reverse=True)
        
        scheduled_tasks = set()
        used_time_ranges = []
        
        for score, task, start_time, end_time in task_slot_scores:
            if task.id in scheduled_tasks:
                continue
            
            # Check if slot is still available
            if self._is_time_available(start_time, end_time, used_time_ranges):
                schedule.append(ScheduleTimeBlock(
                    task_id=task.id,
                    start_time=start_time,
                    end_time=end_time,
                    confidence_score=min(0.9, score),
                    flexibility_score=0.7,
                    energy_level_required=self._determine_energy_level(task, user_pattern)
                ))
                
                scheduled_tasks.add(task.id)
                used_time_ranges.append((start_time, end_time))
        
        return schedule

    def _calculate_task_slot_score(self, task: TaskBase, slot_start: datetime, 
                                 user_pattern: UserProductivityPattern) -> float:
        """Calculate how well a task fits in a particular time slot"""
        score = 0.5  # Base score
        
        hour = slot_start.hour
        
        # Energy level match
        energy_required = self._determine_energy_level(task, user_pattern)
        if hour in user_pattern.peak_hours and energy_required == "high":
            score += 0.3
        elif hour not in user_pattern.low_energy_hours and energy_required == "medium":
            score += 0.2
        elif energy_required == "low":
            score += 0.1
        
        # Priority factor
        if task.priority:
            priority_factor = task.priority.value / 5.0
            score += priority_factor * 0.2
        
        # Deadline urgency
        if task.due_date:
            hours_until_due = (task.due_date - slot_start).total_seconds() / 3600
            if hours_until_due < 24:  # Due within 24 hours
                score += 0.3
            elif hours_until_due < 72:  # Due within 3 days
                score += 0.1
        
        # Morning person preference
        if hour < 12 and user_pattern.morning_person_score > 0.7:
            score += 0.1
        elif hour >= 12 and user_pattern.morning_person_score < 0.3:
            score += 0.1
        
        return min(1.0, score)

    def _determine_energy_level(self, task: TaskBase, user_pattern: UserProductivityPattern) -> str:
        """Determine energy level required for a task"""
        # Simple heuristic based on task properties
        if task.priority and task.priority.value >= 4:
            return "high"
        elif task.estimated_duration and task.estimated_duration > 120:  # More than 2 hours
            return "high"
        elif task.category in ["learning", "creative", "planning"]:
            return "high"
        elif task.category in ["admin", "email", "organizing"]:
            return "low"
        else:
            return "medium"

    def _find_best_slot(self, task: TaskBase, available_slots: List[Tuple[datetime, datetime]],
                       user_pattern: UserProductivityPattern, duration: int) -> Optional[Tuple[datetime, datetime]]:
        """Find the best available slot for a task"""
        best_slot = None
        best_score = -1
        
        for slot_start, slot_end in available_slots:
            if slot_start + timedelta(minutes=duration) <= slot_end:
                score = self._calculate_task_slot_score(task, slot_start, user_pattern)
                if score > best_score:
                    best_score = score
                    best_slot = (slot_start, slot_start + timedelta(minutes=duration))
        
        return best_slot

    def _find_peak_hour_slot(self, task: TaskBase, available_slots: List[Tuple[datetime, datetime]],
                           user_pattern: UserProductivityPattern) -> Optional[Tuple[datetime, datetime]]:
        """Find slot during peak productivity hours"""
        duration = task.estimated_duration or 60
        
        for slot_start, slot_end in available_slots:
            if (slot_start.hour in user_pattern.peak_hours and 
                slot_start + timedelta(minutes=duration) <= slot_end):
                return (slot_start, slot_start + timedelta(minutes=duration))
        
        return None

    def _create_time_block(self, task: TaskBase, slot: Tuple[datetime, datetime],
                          confidence: float, flexibility: float) -> ScheduleTimeBlock:
        """Create a time block for a task"""
        return ScheduleTimeBlock(
            task_id=task.id,
            start_time=slot[0],
            end_time=slot[1],
            confidence_score=confidence,
            flexibility_score=flexibility,
            energy_level_required=self._determine_energy_level(task, self._create_default_pattern())
        )

    def _update_slots_after_scheduling(self, slots: List[Tuple[datetime, datetime]], 
                                     scheduled_start: datetime, scheduled_end: datetime) -> List[Tuple[datetime, datetime]]:
        """Update available slots after scheduling a task"""
        updated_slots = []
        
        for slot_start, slot_end in slots:
            # If slot doesn't overlap with scheduled time, keep it
            if slot_end <= scheduled_start or slot_start >= scheduled_end:
                updated_slots.append((slot_start, slot_end))
            else:
                # Split slot if needed
                if slot_start < scheduled_start:
                    updated_slots.append((slot_start, scheduled_start))
                if slot_end > scheduled_end:
                    updated_slots.append((scheduled_end, slot_end))
        
        return updated_slots

    def _is_time_available(self, start_time: datetime, end_time: datetime, 
                          used_ranges: List[Tuple[datetime, datetime]]) -> bool:
        """Check if a time range is available"""
        for used_start, used_end in used_ranges:
            if not (end_time <= used_start or start_time >= used_end):
                return False
        return True

    def _select_best_schedule(self, schedules: List[List[ScheduleTimeBlock]], 
                            user_pattern: UserProductivityPattern) -> List[ScheduleTimeBlock]:
        """Select the best schedule from multiple options"""
        if not schedules:
            return []
        
        best_schedule = schedules[0]
        best_score = self._calculate_optimization_score(best_schedule, user_pattern)
        
        for schedule in schedules[1:]:
            score = self._calculate_optimization_score(schedule, user_pattern)
            if score > best_score:
                best_score = score
                best_schedule = schedule
        
        return best_schedule

    def _calculate_optimization_score(self, schedule: List[ScheduleTimeBlock], 
                                    user_pattern: UserProductivityPattern) -> float:
        """Calculate overall optimization score for a schedule"""
        if not schedule:
            return 0.0
        
        total_score = 0.0
        
        for block in schedule:
            # Energy level alignment
            hour = block.start_time.hour
            if hour in user_pattern.peak_hours:
                total_score += 0.3
            elif hour not in user_pattern.low_energy_hours:
                total_score += 0.2
            else:
                total_score += 0.1
            
            # Confidence and flexibility
            total_score += block.confidence_score * 0.2
            total_score += block.flexibility_score * 0.1
        
        return total_score / len(schedule)

    async def _generate_productivity_insights(self, schedule: List[ScheduleTimeBlock],
                                            user_pattern: UserProductivityPattern,
                                            tasks: List[TaskBase]) -> List[ProductivityInsight]:
        """Generate insights about the optimized schedule"""
        insights = []
        
        if not schedule:
            return insights
        
        # Peak hour utilization
        peak_hour_blocks = sum(1 for block in schedule if block.start_time.hour in user_pattern.peak_hours)
        peak_utilization = peak_hour_blocks / len(user_pattern.peak_hours) if user_pattern.peak_hours else 0
        
        insights.append(ProductivityInsight(
            metric="peak_hour_utilization",
            value=peak_utilization,
            trend="stable",
            description=f"Utilizing {peak_utilization:.1%} of your peak productivity hours",
            recommendation="Consider moving high-priority tasks to peak hours for better efficiency"
        ))
        
        # Schedule density
        total_scheduled_minutes = sum((block.end_time - block.start_time).total_seconds() / 60 for block in schedule)
        avg_confidence = np.mean([block.confidence_score for block in schedule])
        
        insights.append(ProductivityInsight(
            metric="schedule_confidence",
            value=avg_confidence,
            trend="increasing" if avg_confidence > 0.7 else "stable",
            description=f"Average scheduling confidence: {avg_confidence:.1%}",
            recommendation="High confidence indicates well-optimized schedule"
        ))
        
        return insights

    async def update_user_analytics(self, user_id: str, scheduling_result: SchedulingResult):
        """Update user analytics based on scheduling results"""
        try:
            # This would typically update user patterns based on scheduling success
            logger.info(f"Updating analytics for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating user analytics: {e}")

    async def _load_user_patterns(self):
        """Load cached user patterns"""
        # Implementation would load from cache/database
        pass

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Scheduling Optimization Service...")
        self.user_patterns.clear()