# ai-service/app/main.py
from fastapi import FastAPI
from app.api.endpoints import predict, training, batch

app = FastAPI(
    title="AI Service for TaskMasterPro",
    description="Provides prediction, training, and batch processing for TaskMasterPro",
    version="1.0.0"
)

# Include API routers from different endpoints
app.include_router(predict.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(batch.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
