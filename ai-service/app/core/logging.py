# ai-service/app/core/logging.py
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import structlog

from app.core.config import get_settings

class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        if hasattr(record, "model_type"):
            log_data["model_type"] = record.model_type
        
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def setup_logging():
    """Setup logging configuration"""
    settings = get_settings()
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if settings.enable_structured_logging:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(settings.log_format)
        )
    
    root_logger.addHandler(console_handler)
    
    # File handler (rotating)
    if settings.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / settings.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        
        if settings.enable_structured_logging:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(settings.log_format)
            )
        
        root_logger.addHandler(file_handler)
    
    # Configure structlog if enabled
    if settings.enable_structured_logging:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging initialized",
        extra={
            "log_level": settings.log_level,
            "structured_logging": settings.enable_structured_logging,
            "log_file": settings.log_file
        }
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    if get_settings().enable_structured_logging:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)

class LogContext:
    """Context manager for adding context to logs"""
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.logger = structlog.get_logger()
    
    def __enter__(self):
        self.logger = self.logger.bind(**self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger = self.logger.unbind(*self.context.keys())

def log_performance(operation: str, duration_ms: float, 
                   success: bool = True, **extra):
    """Log performance metrics"""
    logger = get_logger("performance")
    
    log_data = {
        "operation": operation,
        "duration_ms": duration_ms,
        "success": success,
        **extra
    }
    
    if success:
        logger.info(f"Operation {operation} completed", extra=log_data)
    else:
        logger.warning(f"Operation {operation} failed", extra=log_data)

def log_ai_prediction(user_id: str, model_type: str, 
                     prediction_type: str, confidence: float, **extra):
    """Log AI predictions for monitoring"""
    logger = get_logger("ai.predictions")
    
    logger.info(
        "AI prediction made",
        extra={
            "user_id": user_id,
            "model_type": model_type,
            "prediction_type": prediction_type,
            "confidence": confidence,
            **extra
        }
    )

def log_model_training(user_id: str, model_type: str, 
                      duration_ms: float, accuracy: float, 
                      samples: int, **extra):
    """Log model training events"""
    logger = get_logger("ai.training")
    
    logger.info(
        "Model training completed",
        extra={
            "user_id": user_id,
            "model_type": model_type,
            "duration_ms": duration_ms,
            "accuracy": accuracy,
            "samples": samples,
            **extra
        }
    )