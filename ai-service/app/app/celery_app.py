# ai-service/app/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "ai_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)
celery_app.conf.task_routes = {
    "app.tasks.*": {"queue": "ai_queue"}
}
