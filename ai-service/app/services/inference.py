import asyncio
from app.core.config import settings
from app.core.logging import logger

# e.g. load your model once
_model = None


async def _load_model():
    global _model
    if _model is None:
        # placeholder: load from disk or remote
        _model = ...  # load your ML/LLM model
        logger.info("Model loaded", path=settings.MODEL_PATH)
    return _model


async def run_inference(texts: list[str], top_k: int = 5) -> list[dict]:
    model = await _load_model()
    # run async inference; this is just illustrative
    results = []
    for text in texts:
        # e.g. resp = await model.predict(text, top_k=top_k)
        resp = {"input": text, "output": text[::-1]}  # dummy
        results.append(resp)
    return results
