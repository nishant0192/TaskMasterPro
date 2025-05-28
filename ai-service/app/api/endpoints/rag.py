# ai-service/app/api/endpoints/rag.py
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.rag_service import RAGService

router = APIRouter()
rag_service = RAGService()

class RAGRequest(BaseModel):
    text: str
    top_k: Optional[int] = Query(5, ge=1, le=20)

class RAGResponse(BaseModel):
    retrieved: list
    suggestion: str

@router.post("/", response_model=RAGResponse)
def rag(req: RAGRequest):
    try:
        return rag_service.rag(req.text, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
