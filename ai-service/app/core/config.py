# ai-service/app/core/config.py - Fixed version
from functools import lru_cache
from typing import List, Optional, Any, Dict
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Application
    app_name: str = "TaskMaster Pro AI Service"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")
    max_workers: int = Field(default=1, env="MAX_WORKERS")
    
    # Security
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS"
    )
    jwt_secret: str = Field(default="your-super-secret-jwt-key", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    
    # Backend CORS origins (from your error)
    backend_cors_origins: str = Field(
        default="http://localhost:3000,https://your-frontend.app",
        env="backend_cors_origins"
    )
    
    # Database
    ai_database_url: str = Field(
        default="postgresql+asyncpg://ai_user:ai_password_dev@localhost:5433/taskmaster_ai",
        env="AI_DATABASE_URL"
    )
    
    # Alternative database URL field (from your error)
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/ai_service",
        env="database_url"
    )
    
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6380/0", env="REDIS_URL")
    celery_broker_url: str = Field(default="redis://localhost:6380/1", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6380/2", env="CELERY_RESULT_BACKEND")
    
    # AI Models
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    pretrained_models_dir: str = Field(default="./models/pretrained", env="PRETRAINED_MODELS_DIR")
    user_models_dir: str = Field(default="./models/user_models", env="USER_MODELS_DIR")
    
    # Model paths (from your error)
    model_path: str = Field(default="/models/latest", env="model_path")
    vector_store_path: str = Field(default="/models/vectors", env="vector_store_path")
    
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    spacy_model_name: str = Field(default="en_core_web_sm", env="SPACY_MODEL_NAME")
    
    # Gemini API (from your error)
    gemini_api_key: str = Field(default="", env="gemini_api_key")
    
    # AI Configuration
    enable_advanced_ai: bool = Field(default=True, env="ENABLE_ADVANCED_AI")
    enable_model_training: bool = Field(default=True, env="ENABLE_MODEL_TRAINING")
    enable_batch_processing: bool = Field(default=True, env="ENABLE_BATCH_PROCESSING")
    enable_personalization: bool = Field(default=True, env="ENABLE_PERSONALIZATION")
    
    # Performance
    max_model_memory_mb: int = Field(default=2048, env="MAX_MODEL_MEMORY_MB")
    model_cache_ttl_seconds: int = Field(default=3600, env="MODEL_CACHE_TTL_SECONDS")
    embedding_cache_size: int = Field(default=10000, env="EMBEDDING_CACHE_SIZE")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_concurrent_predictions: int = Field(default=10, env="MAX_CONCURRENT_PREDICTIONS")
    
    # Training
    min_training_samples: int = Field(default=10, env="MIN_TRAINING_SAMPLES")
    max_training_samples: int = Field(default=10000, env="MAX_TRAINING_SAMPLES")
    training_validation_split: float = Field(default=0.2, env="TRAINING_VALIDATION_SPLIT")
    model_retrain_threshold: float = Field(default=0.1, env="MODEL_RETRAIN_THRESHOLD")
    
    # Personalization
    personalization_learning_rate: float = Field(default=0.01, env="PERSONALIZATION_LEARNING_RATE")
    user_model_update_frequency_hours: int = Field(default=24, env="USER_MODEL_UPDATE_FREQUENCY_HOURS")
    cold_start_threshold_days: int = Field(default=7, env="COLD_START_THRESHOLD_DAYS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # External APIs
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    enable_external_apis: bool = Field(default=False, env="ENABLE_EXTERNAL_APIS")
    
    # Data Storage
    data_cache_dir: str = Field(default="./data/cache", env="DATA_CACHE_DIR")
    embeddings_cache_dir: str = Field(default="./data/embeddings", env="EMBEDDINGS_CACHE_DIR")
    training_data_dir: str = Field(default="./data/training", env="TRAINING_DATA_DIR")
    
    # Feature Flags
    enable_nlp_personalization: bool = Field(default=True, env="ENABLE_NLP_PERSONALIZATION")
    enable_smart_scheduling: bool = Field(default=True, env="ENABLE_SMART_SCHEDULING")
    enable_predictive_analytics: bool = Field(default=True, env="ENABLE_PREDICTIVE_ANALYTICS")
    enable_behavioral_learning: bool = Field(default=True, env="ENABLE_BEHAVIORAL_LEARNING")
    enable_cross_user_learning: bool = Field(default=False, env="ENABLE_CROSS_USER_LEARNING")
    
    @validator('allowed_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator('backend_cors_origins', pre=True)
    def parse_backend_cors_origins(cls, v):
        if isinstance(v, str):
            return v  # Keep as string for now
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    @validator('training_validation_split')
    def validate_training_split(cls, v):
        if not 0 < v < 1:
            raise ValueError('Training validation split must be between 0 and 1')
        return v
    
    @validator('personalization_learning_rate')
    def validate_learning_rate(cls, v):
        if not 0 < v <= 1:
            raise ValueError('Learning rate must be between 0 and 1')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        # Allow extra fields to prevent validation errors
        extra = "allow"
        
    @property
    def database_url_sync(self) -> str:
        """Synchronous database URL for migrations"""
        return self.ai_database_url.replace("+asyncpg", "")
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration"""
        return {
            "embedding_model": self.embedding_model_name,
            "spacy_model": self.spacy_model_name,
            "cache_dir": self.model_cache_dir,
            "max_memory_mb": self.max_model_memory_mb,
            "batch_size": self.batch_size,
            "enable_training": self.enable_model_training,
            "enable_personalization": self.enable_personalization
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": self.ai_database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "echo": self.debug
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "url": self.redis_url,
            "broker_url": self.celery_broker_url,
            "result_backend": self.celery_result_backend
        }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Export commonly used settings
settings = get_settings()