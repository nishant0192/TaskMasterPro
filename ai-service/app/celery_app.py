from celery import Celery
from app.core.config import settings

celery = Celery(
    "ai_service",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# optional: import tasks so celery auto-discovers them
celery.autodiscover_tasks(["app.services.trainer", "app.services.inference"])
