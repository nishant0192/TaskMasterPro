from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.inference import run_inference
from app.db.session import get_db

router = APIRouter()


@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest, db: AsyncSession = Depends(get_db)):
    try:
        preds = await run_inference(request.texts, top_k=request.top_k)
        return PredictResponse(results=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
