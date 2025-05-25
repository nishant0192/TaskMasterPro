# ai-service/app/api/endpoints/batch.py
from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.tasks import ingest_rag_task, train_task, batch_predict_task

router = APIRouter()

@router.post("/ingest")
def ingest(background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_rag_task.delay)
    return {"status": "ingest scheduled"}

@router.post("/train")
def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_task.delay)
    return {"status": "training scheduled"}

@router.post("/batch_predict")
def batch_predict(background_tasks: BackgroundTasks):
    background_tasks.add_task(batch_predict_task.delay)
    return {"status": "batch predict scheduled"}
