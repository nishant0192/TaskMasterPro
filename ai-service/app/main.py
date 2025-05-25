# ai-service/app/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.base import engine, SQLModel
from app.api.endpoints import predict, train, rag, batch

# Create tables (use Alembic for real migrations)
SQLModel.metadata.create_all(engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(predict.router, prefix=f"{settings.API_V1_STR}/predict", tags=["predict"])
app.include_router(train.router,   prefix=f"{settings.API_V1_STR}/train",   tags=["train"])
app.include_router(rag.router,     prefix=f"{settings.API_V1_STR}/rag",     tags=["rag"])
app.include_router(batch.router,   prefix=f"{settings.API_V1_STR}/batch",   tags=["batch"])

@app.get(f"{settings.API_V1_STR}/health", tags=["health"])
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.RELOAD,
    )
