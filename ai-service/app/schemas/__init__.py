# ai-service/app/schemas/__init__.py
"""
Schemas package for AI service
Production-ready schema imports and exports
"""

from .ai_schemas import *

# Re-export all commonly used schemas
__all__ = [
    # Core AI schemas
    'TaskBase', 'TaskPrioritizationRequest', 'TaskPrioritizationResponse', 'PrioritizedTask',
    'TimeEstimationRequest', 'TimeEstimationResponse',
    'UserInsightsRequest', 'UserInsightsResponse', 'UserInsight',
    'ModelTrainingRequest', 'ModelTrainingResponse',
    'PersonalizationMetrics', 'PersonalizationMetricsResponse',
    'BehaviorPattern', 'BehaviorAnalysisResponse',
    'PredictionResult',
    
    # Advanced schemas
    'TaskSimilarityRequest', 'TaskSimilarityResponse', 'SimilarTask',
    'SchedulingOptimizationRequest', 'SchedulingOptimizationResponse', 'OptimizedSchedule',
    'ProductivityPredictionRequest', 'ProductivityPredictionResponse', 'ProductivityForecast',
    
    # Feedback and learning
    'PredictionFeedback', 'LearningMetrics', 'ModelPerformanceReport',
    
    # Analytics
    'UserSegmentAnalysis', 'FeatureImportanceAnalysis', 'ABTestResult',
    
    # Configuration
    'ModelConfiguration', 'DeploymentStatus',
    
    # Health and monitoring
    'AIHealthMetrics', 'AdaptationRecommendation',
    
    # Base classes and utilities
    'UserPreferences', 'CalendarEvent', 'TimeRange', 'TaskStatus', 'TaskPriority', 'TimePeriod', 'JobType'
]