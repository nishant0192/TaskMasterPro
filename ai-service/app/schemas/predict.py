from pydantic import BaseModel
from typing import Any, List


class PredictRequest(BaseModel):
    texts: List[str]
    top_k: int = 5


class PredictResponse(BaseModel):
    results: List[Any]
