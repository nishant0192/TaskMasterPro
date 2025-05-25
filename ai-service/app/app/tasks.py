# ai-service/app/tasks.py
from app.celery_app import celery_app
from app.services.training_service import TrainingService
from app.services.rag_service import RAGService

training_service = TrainingService()
rag_service = RAGService()

@celery_app.task(name="app.tasks.train")
def train_task():
    return training_service.fine_tune()

@celery_app.task(name="app.tasks.ingest_rag")
def ingest_rag_task():
    rag_service.ingest_all()
    return {"status": "ingested"}

@celery_app.task(name="app.tasks.batch_predict")
def batch_predict_task():
    # e.g., load all TaskPrediction and update
    return {"status": "done"}
