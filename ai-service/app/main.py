# ai-service/app/main.py
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import get_settings
from app.core.database import init_database, close_database
from app.core.logging import setup_logging
from app.core.security import get_current_user
from app.api.v1.router import api_router
from app.services.model_manager import ModelManager
from app.services.local_ai.ai_coordinator import AICoordinator
from app.utils.startup import download_required_models, setup_directories

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global instances
model_manager: ModelManager = None
ai_coordinator: AICoordinator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global model_manager, ai_coordinator
    
    logger.info("🚀 Starting TaskMaster AI Service...")
    settings = get_settings()
    
    try:
        # Initialize database
        await init_database()
        logger.info("✅ Database initialized")
        
        # Setup directories
        await setup_directories()
        logger.info("✅ Directories setup complete")
        
        # Download required models
        await download_required_models()
        logger.info("✅ AI models ready")
        
        # Initialize model manager
        model_manager = ModelManager()
        await model_manager.initialize()
        logger.info("✅ Model Manager initialized")
        
        # Initialize AI coordinator
        ai_coordinator = AICoordinator(model_manager)
        await ai_coordinator.initialize()
        logger.info("✅ AI Coordinator initialized")
        
        # Store in app state
        app.state.model_manager = model_manager
        app.state.ai_coordinator = ai_coordinator
        
        logger.info("🎉 TaskMaster AI Service startup complete")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("🔄 Shutting down TaskMaster AI Service...")
        
        if ai_coordinator:
            await ai_coordinator.cleanup()
            logger.info("✅ AI Coordinator cleanup complete")
        
        if model_manager:
            await model_manager.cleanup()
            logger.info("✅ Model Manager cleanup complete")
        
        await close_database()
        logger.info("✅ Database connections closed")
        
        logger.info("👋 TaskMaster AI Service shutdown complete")

# Create FastAPI application
def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title="TaskMaster Pro AI Service",
        description="Production-ready AI service for intelligent task management and productivity optimization",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Security middleware
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
        )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        response = await call_next(request)
        
        process_time = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        return response
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    return app

# Create app instance
app = create_app()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    settings = get_settings()
    
    # Check database connection
    try:
        from app.core.database import get_async_session
        async with get_async_session() as session:
            await session.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    # Check AI services
    ai_status = "healthy"
    if hasattr(app.state, 'ai_coordinator'):
        ai_status = await app.state.ai_coordinator.health_check()
    
    # Overall status
    is_healthy = db_status == "healthy" and ai_status == "healthy"
    status_code = 200 if is_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": asyncio.get_event_loop().time(),
            "service": "TaskMaster Pro AI Service",
            "version": "1.0.0",
            "environment": settings.environment,
            "components": {
                "database": db_status,
                "ai_services": ai_status,
                "models_loaded": len(getattr(app.state, 'model_manager', {}).model_cache) if hasattr(app.state, 'model_manager') else 0
            }
        }
    )

# Metrics endpoint for monitoring
@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    metrics_data = {
        "ai_predictions_total": 0,
        "ai_training_sessions_total": 0,
        "model_cache_size": 0,
        "active_users": 0
    }
    
    if hasattr(app.state, 'ai_coordinator'):
        metrics_data = await app.state.ai_coordinator.get_metrics()
    
    # Format as Prometheus metrics
    prometheus_metrics = []
    for key, value in metrics_data.items():
        prometheus_metrics.append(f"taskmaster_ai_{key} {value}")
    
    return "\n".join(prometheus_metrics)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "request_id": id(request)
        }
    )

# API rate limiting exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": id(request)
        }
    )

if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=1 if settings.debug else settings.max_workers,
        access_log=settings.debug,
        use_colors=True,
        loop="asyncio"
    )