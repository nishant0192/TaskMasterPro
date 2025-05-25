# ai-service/app/api/endpoints/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    predicted_priority: int

@router.post("/", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return prediction_service.predict(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
