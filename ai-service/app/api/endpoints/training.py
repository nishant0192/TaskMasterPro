# ai-service/app/api/endpoints/train.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.services.training_service import TrainingService

router = APIRouter()
training_service = TrainingService()

class TrainResponse(BaseModel):
    model_dir: str

@router.post("/", response_model=TrainResponse)
def train(background_tasks: BackgroundTasks):
    """
    Starts fine-tuning in background.
    """
    try:
        background_tasks.add_task(training_service.fine_tune)
        return TrainResponse(model_dir=training_service.output_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
