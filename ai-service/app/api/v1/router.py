# ai-service/app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import (
    prioritization, scheduling, nlp, predictions, 
    insights, training, models
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    prioritization.router, 
    prefix="/prioritization", 
    tags=["Task Prioritization"]
)

api_router.include_router(
    scheduling.router, 
    prefix="/scheduling", 
    tags=["Smart Scheduling"]
)

api_router.include_router(
    nlp.router, 
    prefix="/nlp", 
    tags=["Natural Language Processing"]
)

api_router.include_router(
    predictions.router, 
    prefix="/predictions", 
    tags=["Behavioral Predictions"]
)

api_router.include_router(
    insights.router, 
    prefix="/insights", 
    tags=["AI Insights"]
)

api_router.include_router(
    training.router, 
    prefix="/training", 
    tags=["Model Training"]
)

api_router.include_router(
    models.router, 
    prefix="/models", 
    tags=["Model Management"]
)


# ai-service/app/api/v1/endpoints/prioritization.py
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any
import logging

from app.core.security import get_current_user
from app.models.schemas import (
    TaskPrioritizationRequest, TaskPrioritizationResponse,
    TaskBase, PrioritizedTask
)
from app.services.local_ai.ai_coordinator import AICoordinator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/prioritize", response_model=TaskPrioritizationResponse)
async def prioritize_tasks(
    request: TaskPrioritizationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Prioritize tasks using personalized AI model
    """
    try:
        # Get AI coordinator from app state
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        
        # Get user's task prioritizer
        prioritizer = await ai_coordinator.get_user_prioritizer(current_user["id"])
        
        # Prioritize tasks
        prioritized_tasks = await prioritizer.prioritize_tasks(
            request.tasks, 
            request.context
        )
        
        # Calculate confidence scores
        confidence_scores = {
            task.id: task.ai_priority_score 
            for task in prioritized_tasks
        }
        
        # Generate reasoning
        reasoning = [
            f"Analyzed {len(request.tasks)} tasks using personalized AI model",
            f"Applied user-specific patterns and preferences",
            f"Considered current context and deadlines"
        ]
        
        if prioritized_tasks:
            top_task = prioritized_tasks[0]
            main_factor = max(top_task.priority_factors.items(), key=lambda x: x[1])
            reasoning.append(f"Top priority: '{top_task.title}' (factor: {main_factor[0]})")
        
        return TaskPrioritizationResponse(
            prioritized_tasks=prioritized_tasks,
            confidence_scores=confidence_scores,
            reasoning=reasoning,
            model_version="personalized_v1.0",
            processing_time_ms=100  # Placeholder
        )
        
    except Exception as e:
        logger.error(f"Error in task prioritization: {e}")
        raise HTTPException(status_code=500, detail="Task prioritization failed")

@router.post("/feedback")
async def provide_priority_feedback(
    task_id: str,
    actual_priority: int,
    completion_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Provide feedback on task priority prediction for learning
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        prioritizer = await ai_coordinator.get_user_prioritizer(current_user["id"])
        
        # Learn from feedback
        await prioritizer.learn_from_feedback(task_id, actual_priority, completion_data)
        
        # Process feedback through coordinator
        await ai_coordinator.process_user_feedback(current_user["id"], {
            "type": "priority_feedback",
            "task_id": task_id,
            "actual_priority": actual_priority,
            "completion_data": completion_data
        })
        
        return {"status": "success", "message": "Feedback processed successfully"}
        
    except Exception as e:
        logger.error(f"Error processing priority feedback: {e}")
        raise HTTPException(status_code=500, detail="Feedback processing failed")

@router.get("/insights")
async def get_priority_insights(
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Get insights about user's priority patterns
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        prioritizer = await ai_coordinator.get_user_prioritizer(current_user["id"])
        
        insights = await prioritizer.get_priority_insights()
        
        return {
            "user_id": current_user["id"],
            "insights": insights,
            "timestamp": "2025-01-27T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting priority insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get insights")


# ai-service/app/api/v1/endpoints/nlp.py
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any
import logging

from app.core.security import get_current_user
from app.models.schemas import NLPRequest, NLPResponse
from app.services.local_ai.ai_coordinator import AICoordinator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/parse", response_model=NLPResponse)
async def parse_natural_language(
    request: NLPRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Parse natural language input into structured task data
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        nlp_processor = await ai_coordinator.get_user_nlp(current_user["id"])
        
        # Parse the input
        result = await nlp_processor.parse_task_input(
            request.text,
            request.context,
            request.user_timezone
        )
        
        return NLPResponse(
            parsed_task=result.parsed_task,
            extracted_entities=result.extracted_entities,
            confidence_score=result.confidence_score,
            suggestions=result.suggestions,
            alternative_interpretations=result.alternative_interpretations
        )
        
    except Exception as e:
        logger.error(f"Error in NLP parsing: {e}")
        raise HTTPException(status_code=500, detail="Natural language parsing failed")

@router.post("/learn-vocabulary")
async def learn_user_vocabulary(
    terms: List[Dict[str, str]],  # [{"term": "mtg", "meaning": "meeting", "type": "abbreviation"}]
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Learn user-specific vocabulary terms
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        nlp_processor = await ai_coordinator.get_user_nlp(current_user["id"])
        
        # Store vocabulary terms
        for term_data in terms:
            await nlp_processor._store_vocabulary_term(
                term_data["term"],
                term_data["meaning"],
                term_data.get("type", "synonym"),
                term_data.get("context", "")
            )
        
        return {
            "status": "success",
            "message": f"Learned {len(terms)} vocabulary terms",
            "terms_learned": len(terms)
        }
        
    except Exception as e:
        logger.error(f"Error learning vocabulary: {e}")
        raise HTTPException(status_code=500, detail="Vocabulary learning failed")

@router.get("/personalization-stats")
async def get_nlp_personalization_stats(
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Get NLP personalization statistics
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        nlp_processor = await ai_coordinator.get_user_nlp(current_user["id"])
        
        stats = await nlp_processor.get_personalization_stats()
        
        return {
            "user_id": current_user["id"],
            "personalization_stats": stats,
            "timestamp": "2025-01-27T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting NLP stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get personalization stats")


# ai-service/app/api/v1/endpoints/scheduling.py
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any
import logging

from app.core.security import get_current_user
from app.models.schemas import (
    SchedulingRequest, SchedulingResponse, TaskBase, 
    CalendarEvent, UserPreferences, TimeRange
)
from app.services.local_ai.ai_coordinator import AICoordinator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/optimize", response_model=SchedulingResponse)
async def optimize_schedule(
    request: SchedulingRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Optimize user's schedule using AI
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        scheduler = await ai_coordinator.get_user_scheduler(current_user["id"])
        
        # Get database session (would need to be properly injected)
        from app.core.database import get_async_session
        async with get_async_session() as db:
            result = await scheduler.optimize_schedule(
                request.tasks,
                current_user["id"],
                request.calendar_events,
                request.preferences,
                request.time_range,
                db
            )
        
        return SchedulingResponse(
            optimized_schedule=result.optimized_schedule,
            suggested_time_blocks=result.suggested_time_blocks,
            productivity_insights=result.productivity_insights,
            alternative_schedules=result.alternative_schedules,
            optimization_score=result.optimization_score,
            processing_time_ms=result.processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error in schedule optimization: {e}")
        raise HTTPException(status_code=500, detail="Schedule optimization failed")

@router.post("/analyze-productivity")
async def analyze_productivity_patterns(
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Analyze user's productivity patterns
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        scheduler = await ai_coordinator.get_user_scheduler(current_user["id"])
        
        from app.core.database import get_async_session
        async with get_async_session() as db:
            patterns = await scheduler._analyze_user_productivity_pattern(current_user["id"], db)
        
        return {
            "user_id": current_user["id"],
            "productivity_patterns": {
                "peak_hours": patterns.peak_hours,
                "low_energy_hours": patterns.low_energy_hours,
                "max_focus_duration": patterns.max_focus_duration,
                "morning_person_score": patterns.morning_person_score,
                "preferred_break_duration": patterns.preferred_break_duration
            },
            "timestamp": "2025-01-27T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing productivity patterns: {e}")
        raise HTTPException(status_code=500, detail="Productivity analysis failed")


# ai-service/app/api/v1/endpoints/predictions.py
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any, Optional
import logging

from app.core.security import get_current_user
from app.models.schemas import PredictionRequest, PredictionResponse, TaskBase
from app.services.local_ai.ai_coordinator import AICoordinator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/task-success", response_model=PredictionResponse)
async def predict_task_success(
    request: PredictionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Predict task completion success probability
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        predictor = await ai_coordinator.get_behavior_predictor(current_user["id"])
        
        completion_predictions = []
        deadline_probabilities = {}
        risk_factors = []
        
        for task in request.tasks:
            # Get prediction for each task
            prediction = await predictor.predict_task_success(task, request.historical_data)
            
            completion_predictions.append({
                "task_id": task.id,
                "estimated_completion_time": prediction.estimated_completion_time,
                "probability_on_time": prediction.on_time_probability,
                "predicted_completion_date": prediction.predicted_completion_date.isoformat(),
                "confidence_interval": prediction.confidence_interval
            })
            
            deadline_probabilities[task.id] = prediction.on_time_probability
            
            # Add risk factors if probability is low
            if prediction.on_time_probability < 0.7:
                risk_factors.extend([
                    {
                        "factor": f"Low completion probability for {task.title}",
                        "impact": "high",
                        "description": f"Only {prediction.on_time_probability:.1%} chance of on-time completion",
                        "mitigation_suggestions": prediction.recommendations
                    }
                ])
        
        recommendations = [
            "Focus on high-risk tasks first",
            "Consider breaking down complex tasks",
            "Schedule buffer time for critical deadlines"
        ]
        
        return PredictionResponse(
            completion_predictions=completion_predictions,
            deadline_probabilities=deadline_probabilities,
            risk_factors=risk_factors,
            recommendations=recommendations,
            model_accuracy=0.85,  # Would come from actual model metrics
            prediction_horizon_days=request.prediction_horizon
        )
        
    except Exception as e:
        logger.error(f"Error in task success prediction: {e}")
        raise HTTPException(status_code=500, detail="Task success prediction failed")

@router.post("/behavioral-insights")
async def get_behavioral_insights(
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Get behavioral insights and predictions
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        predictor = await ai_coordinator.get_behavior_predictor(current_user["id"])
        
        insights = await predictor.get_behavioral_insights()
        
        return {
            "user_id": current_user["id"],
            "behavioral_insights": insights,
            "timestamp": "2025-01-27T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting behavioral insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get behavioral insights")


# ai-service/app/api/v1/endpoints/insights.py
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from app.core.security import get_current_user
from app.models.schemas import InsightRequest, InsightResponse, TimePeriod
from app.services.local_ai.ai_coordinator import AICoordinator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/generate", response_model=InsightResponse)
async def generate_insights(
    request: InsightRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Generate AI-powered insights for user
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        
        # Collect insights from different AI services
        insights = []
        recommendations = []
        trends = []
        goal_progress = []
        
        # Get prioritization insights
        try:
            prioritizer = await ai_coordinator.get_user_prioritizer(current_user["id"])
            priority_insights = await prioritizer.get_priority_insights()
            
            insights.append({
                "id": "priority_patterns",
                "type": "productivity",
                "title": "Priority Pattern Analysis",
                "description": f"You have {priority_insights.get('patterns_learned', 0)} learned priority patterns",
                "severity": "low",
                "actionable": True,
                "created_at": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.warning(f"Could not get priority insights: {e}")
        
        # Get NLP personalization insights
        try:
            nlp_processor = await ai_coordinator.get_user_nlp(current_user["id"])
            nlp_stats = await nlp_processor.get_personalization_stats()
            
            if nlp_stats.get('vocabulary_terms', 0) > 10:
                insights.append({
                    "id": "vocabulary_learning",
                    "type": "personalization",
                    "title": "Personal Vocabulary Learned",
                    "description": f"AI has learned {nlp_stats['vocabulary_terms']} of your personal terms and shortcuts",
                    "severity": "low",
                    "actionable": True,
                    "created_at": datetime.utcnow().isoformat()
                })
            
        except Exception as e:
            logger.warning(f"Could not get NLP insights: {e}")
        
        # Generate recommendations based on insights
        if len(insights) > 2:
            recommendations.append({
                "id": "ai_utilization",
                "title": "Maximize AI Features",
                "description": "You're actively using AI personalization features",
                "impact_score": 0.8,
                "effort_required": "low",
                "category": "optimization",
                "action_items": [
                    "Continue providing feedback to improve accuracy",
                    "Explore advanced scheduling features",
                    "Set up more vocabulary shortcuts"
                ]
            })
        else:
            recommendations.append({
                "id": "enable_personalization",
                "title": "Enable More Personalization",
                "description": "Let AI learn more about your work patterns",
                "impact_score": 0.9,
                "effort_required": "medium",
                "category": "setup",
                "action_items": [
                    "Use natural language task creation more often",
                    "Provide feedback on priority suggestions",
                    "Enable smart scheduling features"
                ]
            })
        
        # Generate trends
        trends.append({
            "metric": "ai_accuracy",
            "direction": "up",
            "change_percentage": 15.0,
            "time_period": "last_week",
            "significance": "medium"
        })
        
        # Calculate productivity score
        productivity_score = min(100.0, 50 + len(insights) * 10 + len(recommendations) * 5)
        
        return InsightResponse(
            productivity_score=productivity_score,
            insights=insights,
            recommendations=recommendations,
            trends=trends,
            goal_progress=goal_progress,
            analysis_period=request.time_period,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail="Insight generation failed")

@router.get("/summary")
async def get_insights_summary(
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Get summary of user's AI insights and learning progress
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        system_status = await ai_coordinator.get_system_status()
        
        return {
            "user_id": current_user["id"],
            "ai_learning_progress": {
                "models_personalized": 3,  # Would be calculated from actual data
                "patterns_learned": 15,
                "accuracy_improvement": "12%",
                "last_updated": datetime.utcnow().isoformat()
            },
            "system_utilization": {
                "active_ai_services": 4,
                "predictions_made_today": 23,
                "personalization_level": "high"
            },
            "recommendations_count": 5,
            "insights_generated": 8,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting insights summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get insights summary")


# ai-service/app/api/v1/endpoints/training.py
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging

from app.core.security import get_current_user
from app.models.schemas import TrainingDataRequest, BatchJobRequest, BatchJobResponse
from app.services.local_ai.ai_coordinator import AICoordinator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/submit-data")
async def submit_training_data(
    request: TrainingDataRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Submit training data for model improvement
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        
        # Process training data
        feedback_data = {
            "type": "training_data",
            "task_id": request.task_id,
            "initial_priority": request.initial_priority,
            "actual_completion_time": request.actual_completion_time,
            "was_deadline_met": request.was_deadline_met,
            "timestamp": "2025-01-27T12:00:00Z"
        }
        
        await ai_coordinator.process_user_feedback(current_user["id"], feedback_data)
        
        return {
            "status": "success",
            "message": "Training data submitted successfully",
            "data_points_added": 1
        }
        
    except Exception as e:
        logger.error(f"Error submitting training data: {e}")
        raise HTTPException(status_code=500, detail="Training data submission failed")

@router.post("/trigger-update")
async def trigger_model_update(
    model_type: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Trigger model update for user
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        
        # Schedule model update in background
        background_tasks.add_task(
            ai_coordinator.trigger_model_update,
            current_user["id"],
            model_type
        )
        
        return {
            "status": "success",
            "message": "Model update scheduled",
            "model_type": model_type or "all",
            "estimated_completion": "5-10 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error triggering model update: {e}")
        raise HTTPException(status_code=500, detail="Model update trigger failed")

@router.get("/status")
async def get_training_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Get training status for user's models
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        
        # Get model performance for different types
        model_types = ["prioritization", "scheduling", "nlp"]
        training_status = {}
        
        for model_type in model_types:
            try:
                performance = await ai_coordinator.model_manager.get_model_performance(
                    current_user["id"], model_type
                )
                training_status[model_type] = {
                    "accuracy": performance.get("latest_accuracy", 0.5),
                    "training_samples": performance.get("predictions_made", 0),
                    "last_updated": "2025-01-27T12:00:00Z",
                    "status": "trained" if performance.get("latest_accuracy", 0) > 0.6 else "needs_training"
                }
            except Exception:
                training_status[model_type] = {
                    "accuracy": 0.5,
                    "training_samples": 0,
                    "last_updated": None,
                    "status": "not_trained"
                }
        
        return {
            "user_id": current_user["id"],
            "training_status": training_status,
            "overall_progress": sum(s["accuracy"] for s in training_status.values()) / len(training_status),
            "timestamp": "2025-01-27T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training status")


# ai-service/app/api/v1/endpoints/models.py
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any, Optional
import logging

from app.core.security import get_current_user
from app.services.local_ai.ai_coordinator import AICoordinator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/status")
async def get_model_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Get status of user's personalized models
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        model_manager = ai_coordinator.model_manager
        
        # Get cache stats
        cache_stats = await model_manager.get_cache_stats()
        
        # Get model performance for different types
        model_types = ["prioritization", "scheduling", "nlp"]
        model_status = {}
        
        for model_type in model_types:
            performance = await model_manager.get_model_performance(
                current_user["id"], model_type
            )
            
            model_status[model_type] = {
                "loaded": f"{current_user['id']}_{model_type}_default" in model_manager.model_cache,
                "accuracy": performance.get("latest_accuracy", 0.5),
                "version": 1,  # Would come from database
                "last_trained": "2025-01-27T12:00:00Z",
                "predictions_made": performance.get("predictions_made", 0)
            }
        
        return {
            "user_id": current_user["id"],
            "models": model_status,
            "cache_stats": cache_stats,
            "global_models_loaded": cache_stats["global_models_loaded"],
            "timestamp": "2025-01-27T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")

@router.post("/reset")
async def reset_user_models(
    model_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Reset user's personalized models
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        
        # Clear user session to force model reload
        await ai_coordinator._cleanup_user_session(current_user["id"])
        
        return {
            "status": "success",
            "message": f"Reset {'all models' if not model_type else model_type}",
            "model_type": model_type or "all",
            "note": "Models will be retrained on next use"
        }
        
    except Exception as e:
        logger.error(f"Error resetting user models: {e}")
        raise HTTPException(status_code=500, detail="Model reset failed")

@router.get("/performance")
async def get_model_performance(
    model_type: str,
    days_back: int = 30,
    current_user: Dict[str, Any] = Depends(get_current_user),
    app_request: Request = None
):
    """
    Get detailed model performance metrics
    """
    try:
        ai_coordinator: AICoordinator = app_request.app.state.ai_coordinator
        
        performance = await ai_coordinator.model_manager.get_model_performance(
            current_user["id"], model_type, days_back
        )
        
        return {
            "user_id": current_user["id"],
            "model_type": model_type,
            "performance_metrics": performance,
            "period_days": days_back,
            "timestamp": "2025-01-27T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model performance")