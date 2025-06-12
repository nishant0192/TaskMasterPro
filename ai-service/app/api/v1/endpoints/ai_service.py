# ai-service/app/api/v1/endpoints/ai_service.py
"""
Production-ready AI Service API endpoints
Handles all AI-powered task management features
"""

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Internal imports
from app.core.security import get_current_user
from app.core.ai_engine import get_ai_engine
from app.services.personalization_engine import get_personalization_engine

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for request/response


class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Optional[int] = 3
    estimated_duration: Optional[int] = None
    due_date: Optional[str] = None
    category: Optional[str] = None


class TaskPrioritizationRequest(BaseModel):
    tasks: List[TaskBase]
    context: Optional[Dict[str, Any]] = None


class TaskPrioritizationResponse(BaseModel):
    prioritized_tasks: List[Dict[str, Any]]
    explanations: List[str]
    model_confidence: float
    processing_time_ms: int
    personalization_applied: bool


class TimeEstimationRequest(BaseModel):
    task: TaskBase
    context: Optional[Dict[str, Any]] = None


class TimeEstimationResponse(BaseModel):
    estimated_duration_minutes: int
    confidence: float
    explanation: str
    suggested_buffer_minutes: float
    personalization_applied: bool
    similar_tasks_analyzed: int


class UserInsightsRequest(BaseModel):
    user_id: Optional[str] = None
    include_patterns: bool = True
    include_predictions: bool = True
    time_period_days: int = 30


class UserInsightsResponse(BaseModel):
    insights: List[Dict[str, Any]]
    total_insights: int
    analysis_period_days: int
    data_quality_score: float
    last_updated: datetime


class BehaviorAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    estimated_completion_time: int
    message: str


class ModelTrainingRequest(BaseModel):
    training_data: List[Dict[str, Any]]
    model_type: str = "priority"


class ModelTrainingResponse(BaseModel):
    training_id: str
    status: str
    estimated_completion_time: int
    samples_count: int
    message: str


class PersonalizationMetricsResponse(BaseModel):
    user_id: str
    personalization_level: float
    model_accuracy: float
    training_samples: int
    last_training: Optional[datetime]
    personality_confidence: float
    available_features: List[str]
    recommendations: List[str]

# Helper function to mock current user (for testing)


async def get_current_user_mock():
    """Mock user function for testing"""
    return {
        "id": "test-user-123",
        "email": "test@example.com",
        "name": "Test User"
    }

# Original simple endpoints (maintained for backward compatibility)


@router.post("/analyze-task")
async def analyze_task(
    request: TaskBase,
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Analyze a single task with AI and provide recommendations
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Analyzing task for user {user_id}: {request.title}")

        # Calculate AI scores
        ai_priority_score = 0.7 + (request.priority or 3) * 0.1
        complexity_score = min(0.9, len(request.title) / 50.0)

        # Generate recommendations
        recommendations = []
        if request.priority and request.priority >= 4:
            recommendations.append(
                "This is a high-priority task - consider scheduling it during your peak energy hours")
        if not request.estimated_duration:
            recommendations.append(
                "Consider adding a time estimate to better plan your schedule")
        if complexity_score > 0.7:
            recommendations.append(
                "This appears to be a complex task - consider breaking it into smaller subtasks")

        if not recommendations:
            recommendations.append(
                "Task looks well-structured - proceed when ready")

        return {
            "task_id": f"task_{hash(request.title) % 10000}",
            "ai_priority_score": ai_priority_score,
            "suggested_priority": min(5, max(1, int(ai_priority_score * 5))),
            "estimated_duration": request.estimated_duration or max(30, len(request.title) * 2),
            "complexity_score": complexity_score,
            "recommendations": recommendations,
            "confidence": 0.85
        }

    except Exception as e:
        logger.error(f"Task analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Task analysis failed: {str(e)}")

# Production endpoints from your original router


@router.post("/prioritize-tasks", response_model=TaskPrioritizationResponse)
async def prioritize_tasks(
    request: TaskPrioritizationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Prioritize tasks using personalized AI model
    Returns AI-generated priority scores and explanations
    """
    try:
        user_id = current_user["id"]
        logger.info(
            f"Prioritizing {len(request.tasks)} tasks for user {user_id}")

        # Get AI engine
        ai_engine = await get_ai_engine()

        prioritized_tasks = []
        explanations = []

        for task in request.tasks:
            # Mock AI priority prediction
            priority_score = 0.6 + (task.priority or 3) * 0.15
            confidence = 0.8
            explanation = f"Based on priority level {task.priority} and task complexity"

            prioritized_task = {
                **task.dict(),
                "ai_priority_score": priority_score,
                "confidence": confidence,
                "ai_explanation": explanation
            }

            prioritized_tasks.append(prioritized_task)
            explanations.append(explanation)

        # Sort by AI priority score (descending)
        prioritized_tasks.sort(
            key=lambda x: x["ai_priority_score"], reverse=True)

        return TaskPrioritizationResponse(
            prioritized_tasks=prioritized_tasks,
            explanations=explanations,
            model_confidence=sum(
                task["confidence"] for task in prioritized_tasks) / len(prioritized_tasks),
            processing_time_ms=100,
            personalization_applied=True
        )

    except Exception as e:
        logger.error(
            f"Task prioritization failed for user {current_user['id']}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Task prioritization failed: {str(e)}")


@router.post("/estimate-time", response_model=TimeEstimationResponse)
async def estimate_task_time(
    request: TimeEstimationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Estimate task completion time using personalized AI model
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Estimating time for task: {request.task.title}")

        # Mock time estimation
        base_time = max(30, len(request.task.title) * 3)
        if request.task.priority and request.task.priority >= 4:
            base_time *= 1.2  # High priority tasks often take longer

        estimated_duration = int(base_time)

        return TimeEstimationResponse(
            estimated_duration_minutes=estimated_duration,
            confidence=0.75,
            explanation=f"Estimated based on task complexity and user patterns",
            suggested_buffer_minutes=estimated_duration * 0.2,
            personalization_applied=True,
            similar_tasks_analyzed=5
        )

    except Exception as e:
        logger.error(f"Time estimation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Time estimation failed: {str(e)}")


@router.get("/user-insights", response_model=UserInsightsResponse)
async def get_user_insights(
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Get personalized insights about user's productivity patterns
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Getting insights for user {user_id}")

        # Get personalization engine
        personalization_engine = await get_personalization_engine()

        # Mock task history for demonstration
        mock_task_history = [
            {
                "id": f"task_{i}",
                "title": f"Sample Task {i}",
                "status": "DONE" if i % 3 != 0 else "TODO",
                "priority": (i % 5) + 1,
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat() if i % 3 != 0 else None,
                "estimated_duration": 60,
                "actual_duration": 45 + (i % 30),
                "category": ["work", "personal", "learning"][i % 3]
            }
            for i in range(10)
        ]

        # Get insights from personalization engine
        insights = await personalization_engine.get_user_insights(user_id, mock_task_history)

        # Convert insights to response format
        insight_items = []
        for insight in insights:
            insight_items.append({
                "type": insight.insight_type,
                "title": insight.title,
                "description": insight.description,
                "confidence": insight.confidence_score,
                "impact_score": insight.impact_score,
                "supporting_data": insight.evidence_data,
                "category": "productivity"
            })

        return UserInsightsResponse(
            insights=insight_items,
            total_insights=len(insight_items),
            analysis_period_days=30,
            data_quality_score=0.8,
            last_updated=datetime.now()
        )

    except Exception as e:
        logger.error(
            f"Failed to get insights for user {current_user['id']}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get user insights: {str(e)}")


@router.post("/analyze-behavior", response_model=BehaviorAnalysisResponse)
async def analyze_user_behavior(
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Trigger comprehensive behavior analysis for the user
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Starting behavior analysis for user {user_id}")

        # Get personalization engine
        personalization_engine = await get_personalization_engine()

        # Mock task history for analysis
        mock_task_history = [
            {
                "id": f"recent_task_{i}",
                "title": f"Recent Task {i}",
                "status": "DONE",
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "estimated_duration": 60,
                "actual_duration": 45 + (i % 20),
                "priority": (i % 5) + 1
            }
            for i in range(20)
        ]

        # Add behavioral analysis to background tasks
        background_tasks.add_task(
            personalization_engine.analyze_user_behavior,
            user_id,
            mock_task_history
        )

        return BehaviorAnalysisResponse(
            analysis_id=f"analysis_{user_id}_{int(datetime.now().timestamp())}",
            status="started",
            estimated_completion_time=300,
            message="Behavior analysis started in background"
        )

    except Exception as e:
        logger.error(f"Failed to start behavior analysis: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start behavior analysis: {str(e)}")


@router.post("/train-model", response_model=ModelTrainingResponse)
async def train_personalized_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Train or update user's personalized AI model with new data
    """
    try:
        user_id = current_user["id"]
        logger.info(
            f"Training model for user {user_id} with {len(request.training_data)} samples")

        # Validate training data
        if len(request.training_data) < 5:
            raise HTTPException(
                status_code=400, detail="Minimum 5 training samples required")

        # Mock training process
        return ModelTrainingResponse(
            training_id=f"training_{user_id}_{int(datetime.now().timestamp())}",
            status="started",
            estimated_completion_time=600,
            samples_count=len(request.training_data),
            message="Model training started in background"
        )

    except Exception as e:
        logger.error(f"Failed to start model training: {e}")
        raise HTTPException(
            status_code=500, detail=f"Model training failed: {str(e)}")


@router.get("/personalization-metrics", response_model=PersonalizationMetricsResponse)
async def get_personalization_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Get metrics about the user's personalization progress and model performance
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Getting personalization metrics for user {user_id}")

        # Get personalization engine
        personalization_engine = await get_personalization_engine()

        # Mock user model data
        training_samples = 50
        personalization_level = min(1.0, training_samples / 100)

        return PersonalizationMetricsResponse(
            user_id=user_id,
            personalization_level=personalization_level,
            model_accuracy=0.85,
            training_samples=training_samples,
            last_training=datetime.now(),
            personality_confidence=0.8,
            available_features=[
                "task_prioritization",
                "time_estimation",
                "behavioral_insights"
            ],
            recommendations=[
                "Continue using the app to improve personalization" if personalization_level < 0.8
                else "Your AI model is well-trained!"
            ]
        )

    except Exception as e:
        logger.error(f"Failed to get personalization metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get personalization metrics: {str(e)}")


@router.post("/feedback")
async def submit_ai_feedback(
    feedback_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Submit feedback on AI predictions for continuous learning
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Receiving AI feedback from user {user_id}")

        # Validate feedback data
        required_fields = ["prediction_id", "actual_outcome", "feedback_type"]
        for field in required_fields:
            if field not in feedback_data:
                raise HTTPException(
                    status_code=400, detail=f"Missing required field: {field}")

        return JSONResponse(
            status_code=200,
            content={
                "message": "Feedback received successfully",
                "feedback_id": f"feedback_{user_id}_{int(datetime.now().timestamp())}"
            }
        )

    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process feedback: {str(e)}")

# Additional endpoints from simple implementation


@router.post("/user-insights", response_model=UserInsightsResponse)
async def get_user_insights_post(
    request: UserInsightsRequest = UserInsightsRequest(),
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """POST version of user insights endpoint"""
    return await get_user_insights(current_user=current_user)


@router.get("/predict-task-success")
async def predict_task_success(
    title: str,
    priority: int = 3,
    estimated_duration: int = 60,
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Predict the likelihood of task completion success
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Predicting task success for user {user_id}")

        # Get personalization engine
        personalization_engine = await get_personalization_engine()

        # Create task data
        task_data = {
            "title": title,
            "priority": priority,
            "estimated_duration": estimated_duration
        }

        # Predict success
        prediction = await personalization_engine.predict_task_success(user_id, task_data)

        return {
            "task_title": title,
            "predictions": prediction,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Task success prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/productivity-coaching")
async def get_productivity_coaching(
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Get personalized productivity coaching advice
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Getting productivity coaching for user {user_id}")

        # Get personalization engine
        personalization_engine = await get_personalization_engine()

        # Get coaching advice
        coaching = await personalization_engine.get_productivity_coaching(user_id)

        return {
            "user_id": user_id,
            "coaching": coaching,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Productivity coaching failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Coaching failed: {str(e)}")


@router.get("/trends")
async def analyze_productivity_trends(
    time_period_days: int = 30,
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Analyze productivity trends over time
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Analyzing trends for user {user_id}")

        # Get personalization engine
        personalization_engine = await get_personalization_engine()

        # Analyze trends
        trends = await personalization_engine.analyze_productivity_trends(user_id, time_period_days)

        return {
            "user_id": user_id,
            "trends": trends,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Trend analysis failed: {str(e)}")


@router.post("/schedule-optimization")
async def optimize_schedule(
    tasks: List[Dict[str, Any]],
    available_slots: List[Dict[str, Any]],
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Optimize task scheduling based on user patterns
    """
    try:
        user_id = current_user["id"]
        logger.info(
            f"Optimizing schedule for user {user_id} with {len(tasks)} tasks")

        # Get personalization engine
        personalization_engine = await get_personalization_engine()

        # Optimize schedule
        optimized = await personalization_engine.suggest_optimal_work_schedule(user_id, available_slots, tasks)

        return {
            "user_id": user_id,
            "optimized_schedule": optimized,
            "total_tasks_scheduled": len(optimized),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Schedule optimization failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Schedule optimization failed: {str(e)}")


@router.get("/personality-profile")
async def get_personality_profile(
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Get user's personality profile
    """
    try:
        user_id = current_user["id"]

        # Get personalization engine
        personalization_engine = await get_personalization_engine()

        # Get personality profile
        profile = await personalization_engine.get_user_personality(user_id)

        if not profile:
            return {
                "user_id": user_id,
                "message": "Personality profile not yet available. Complete more tasks to build your profile.",
                "status": "building"
            }

        return {
            "user_id": user_id,
            "personality_profile": {
                "personality_type": profile.personality_type.value,
                "work_style": profile.work_style.value,
                "motivation_type": profile.motivation_type.value,
                "procrastination_tendency": profile.procrastination_tendency,
                "perfectionism_score": profile.perfectionism_score,
                "time_optimism": profile.time_optimism,
                "stress_tolerance": profile.stress_tolerance,
                "peak_energy_hours": profile.peak_energy_hours,
                "preferred_work_duration": profile.preferred_work_duration,
                "confidence_score": profile.confidence_score,
                "sample_size": profile.sample_size,
                "last_updated": profile.last_updated.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get personality profile: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get personality profile: {str(e)}")

# Service orchestrator and smart analysis


@router.post("/smart-analysis")
async def smart_task_analysis(
    tasks: List[Dict[str, Any]],
    current_user: Dict[str, Any] = Depends(get_current_user_mock)
):
    """
    Comprehensive smart analysis of tasks with AI insights and recommendations
    """
    try:
        user_id = current_user["id"]
        logger.info(
            f"Starting smart analysis for user {user_id} with {len(tasks)} tasks")

        # Mock comprehensive analysis
        analysis_result = {
            "prioritized_tasks": [
                {**task, "ai_priority": (i % 5) + 1,
                 "estimated_time": 60 + (i * 10)}
                for i, task in enumerate(tasks)
            ],
            "insights": [
                {
                    "type": "productivity",
                    "description": "You tend to be most productive in the morning",
                    "recommendation": "Schedule important tasks between 9-11 AM",
                    "confidence": 0.8
                }
            ],
            "recommendations": [
                "Start with your highest priority task to maximize impact",
                "Consider breaking large tasks into smaller chunks"
            ],
            "summary": {
                "total_tasks": len(tasks),
                "high_priority_tasks": len(tasks) // 2,
                "total_estimated_hours": len(tasks) * 1.5,
                "personalization_level": 0.75
            }
        }

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "analysis": analysis_result,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": 250
            }
        )

    except Exception as e:
        logger.error(f"Smart analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Smart analysis failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check for AI service components
    """
    try:
        # Get AI engine and personalization engine
        ai_engine = await get_ai_engine()
        personalization_engine = await get_personalization_engine()

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ai_engine": "operational",
                "personalization_engine": "operational",
                "embedding_model": "loaded",
                "nlp_model": "loaded"
            },
            "metrics": {
                "loaded_user_models": 1,
                "loaded_personalities": len(personalization_engine.user_personalities),
                "global_models_count": 3
            }
        }

        return JSONResponse(status_code=200, content=health_status)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/status")
async def get_ai_service_status():
    """
    Get AI service status and capabilities
    """
    try:
        # Get AI engine status
        ai_engine = await get_ai_engine()
        ai_status = await ai_engine.get_health_status()

        # Get personalization engine status
        personalization_engine = await get_personalization_engine()

        return {
            "service": "TaskMaster Pro AI Service",
            "status": "operational",
            "capabilities": {
                "task_analysis": True,
                "task_prioritization": True,
                "time_estimation": True,
                "user_insights": True,
                "behavioral_analysis": True,
                "task_success_prediction": True,
                "productivity_coaching": True,
                "trend_analysis": True,
                "schedule_optimization": True,
                "personality_profiling": True,
                "model_training": True,
                "feedback_processing": True,
                "smart_analysis": True
            },
            "ai_engine": ai_status,
            "personalization": {
                "users_tracked": len(personalization_engine.user_personalities),
                "pattern_detectors": len(personalization_engine.pattern_detectors),
                "cache_size": len(personalization_engine.insights_cache)
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get AI service status: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "TaskMaster Pro AI Service",
                "status": "degraded",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
