# ai-service/app/core/config.py
import os
from typing import List

from pydantic import BaseSettings, AnyHttpUrl, validator

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "TaskMasterPro AI Service"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    LOG_LEVEL: str = "INFO"

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    # Database
    AI_DATABASE_URL: str

    # Model paths
    MODEL_DIR: str = "./models"
    FINE_TUNED_MODEL_NAME: str = "fine_tuned_model"
    OPTIMIZED_MODEL_NAME: str = "optimized_model.pth"

    # ChromaDB / RAG
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    GENERATOR_MODEL_NAME: str = "t5-small"
    CHROMA_DB_HOST: str = "localhost"
    CHROMA_DB_PORT: int = 8000

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and v:
            return [i.strip() for i in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
