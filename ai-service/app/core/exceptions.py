# ai-service/app/core/exceptions.py
from typing import Any, Dict, Optional


class AIServiceException(Exception):
    """Base exception for AI service"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelNotLoadedException(AIServiceException):
    """Raised when AI model is not loaded"""

    def __init__(self, model_type: str):
        super().__init__(
            f"Model '{model_type}' is not loaded",
            {"model_type": model_type}
        )


class InsufficientDataException(AIServiceException):
    """Raised when there's insufficient data for training/prediction"""

    def __init__(self, required: int, available: int, operation: str):
        super().__init__(
            f"Insufficient data for {operation}: required {required}, available {available}",
            {
                "required": required,
                "available": available,
                "operation": operation
            }
        )


class ModelTrainingException(AIServiceException):
    """Raised when model training fails"""

    def __init__(self, model_type: str, reason: str):
        super().__init__(
            f"Failed to train {model_type} model: {reason}",
            {
                "model_type": model_type,
                "reason": reason
            }
        )


class PredictionException(AIServiceException):
    """Raised when prediction fails"""

    def __init__(self, model_type: str, reason: str):
        super().__init__(
            f"Prediction failed for {model_type}: {reason}",
            {
                "model_type": model_type,
                "reason": reason
            }
        )


class UserNotFoundException(AIServiceException):
    """Raised when user is not found"""

    def __init__(self, user_id: str):
        super().__init__(
            f"User '{user_id}' not found",
            {"user_id": user_id}
        )


class ConfigurationException(AIServiceException):
    """Raised when there's a configuration error"""

    def __init__(self, config_key: str, reason: str):
        super().__init__(
            f"Configuration error for '{config_key}': {reason}",
            {
                "config_key": config_key,
                "reason": reason
            }
        )
