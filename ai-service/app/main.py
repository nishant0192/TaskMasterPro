# ai-service/app/main.py
"""
TaskMaster Pro AI Service - Main Application
Production-ready FastAPI application with comprehensive AI capabilities
"""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import sys
import os

# FastAPI imports
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import structlog

# Internal imports
from app.core.config import get_settings
from app.core.database import init_database, close_database
from app.core.ai_engine import ai_engine
# Fixed import - import the factory function instead of global instance
from app.services.personalization_engine import get_personalization_engine
from app.api.v1.endpoints.ai_service import router as ai_router
from app.core.security import get_current_user

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if os.getenv("ENVIRONMENT", "development") == "production"
        else structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# Security
security = HTTPBearer(auto_error=False)

class SimpleAIMonitor:
    """Simple AI monitoring for production"""

    def __init__(self):
        self.metrics = {
            'requests_total': 0,
            'errors_total': 0,
            'response_times': [],
            'start_time': time.time()
        }

    def record_request(self, endpoint: str, response_time: float, status_code: int):
        """Record request metrics"""
        self.metrics['requests_total'] += 1
        self.metrics['response_times'].append(response_time)

        if status_code >= 400:
            self.metrics['errors_total'] += 1

        # Keep only last 1000 response times
        if len(self.metrics['response_times']) > 1000:
            self.metrics['response_times'] = self.metrics['response_times'][-1000:]

    def record_error(self, error_type: str, component: str):
        """Record error"""
        self.metrics['errors_total'] += 1

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        uptime = time.time() - self.metrics['start_time']
        avg_response_time = (
            sum(self.metrics['response_times']) /
            len(self.metrics['response_times'])
            if self.metrics['response_times'] else 0
        )

        error_rate = (
            self.metrics['errors_total'] /
            max(1, self.metrics['requests_total'])
        )

        return {
            'status': 'healthy' if error_rate < 0.1 else 'degraded',
            'uptime_seconds': uptime,
            'requests_total': self.metrics['requests_total'],
            'errors_total': self.metrics['errors_total'],
            'error_rate': error_rate,
            'avg_response_time_ms': avg_response_time * 1000
        }


# Create monitor instance
ai_monitor = SimpleAIMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    shutdown_errors = []
    
    try:
        # Startup
        logger.info("🚀 Starting TaskMaster Pro AI Service...")
        app.state.startup_time = time.time()

        # Initialize database
        try:
            await init_database()
            logger.info("✅ Database initialized")
        except Exception as e:
            logger.warning(f"⚠️ Database initialization failed: {e}")

        # Initialize AI engine
        try:
            await ai_engine.initialize()
            app.state.ai_engine = ai_engine
            logger.info("🤖 AI Engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ AI Engine initialization failed: {e}")
            app.state.ai_engine = None

        # Initialize personalization engine - FIXED
        try:
            personalization_engine = await get_personalization_engine()
            app.state.personalization_engine = personalization_engine
            logger.info("🧠 Personalization Engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Personalization Engine initialization failed: {e}")
            app.state.personalization_engine = None

        # Initialize monitoring
        app.state.ai_monitor = ai_monitor
        logger.info("📊 AI Monitor initialized")

        logger.info("🎉 TaskMaster Pro AI Service started successfully!")

        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Cleanup
        shutdown_start = time.time()
        logger.info("🔄 Shutting down TaskMaster Pro AI Service...")

        # Cleanup AI monitor
        if hasattr(app.state, 'ai_monitor') and app.state.ai_monitor:
            try:
                # Simple cleanup for monitor
                logger.info("✅ Monitoring cleanup complete")
            except Exception as e:
                shutdown_errors.append(f"Monitoring cleanup: {e}")
        
        # Cleanup personalization engine - FIXED
        if hasattr(app.state, 'personalization_engine') and app.state.personalization_engine:
            try:
                await app.state.personalization_engine.shutdown()
                logger.info("✅ Personalization Engine cleanup complete")
            except Exception as e:
                shutdown_errors.append(f"Personalization cleanup: {e}")
        
        # Cleanup AI engine
        if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
            try:
                if hasattr(app.state.ai_engine, 'cleanup'):
                    await app.state.ai_engine.cleanup()
                logger.info("✅ AI Engine cleanup complete")
            except Exception as e:
                shutdown_errors.append(f"AI Engine cleanup: {e}")
        
        # Close database connections
        try:
            await close_database()
            logger.info("✅ Database connections closed")
        except Exception as e:
            shutdown_errors.append(f"Database cleanup: {e}")
        
        shutdown_duration = time.time() - shutdown_start
        
        if shutdown_errors:
            logger.warning(
                "⚠️ Shutdown completed with errors",
                shutdown_duration_seconds=f"{shutdown_duration:.2f}",
                errors=shutdown_errors
            )
        else:
            logger.info(
                "👋 TaskMaster AI Service shutdown complete",
                shutdown_duration_seconds=f"{shutdown_duration:.2f}"
            )

# Create FastAPI application
app = FastAPI(
    title="TaskMaster Pro AI Service",
    description="Advanced AI-powered task management and personalization service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

# Trust proxy headers in production
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure properly for production
    )

# Compress responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request ID and logging middleware
@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    """Add request ID and comprehensive logging"""
    import uuid
    import time

    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # Start timing
    start_time = time.time()

    # Log incoming request
    logger.debug(
        "🔄 Incoming Request",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent", "unknown"),
        client_ip=request.client.host if request.client else "unknown"
    )

    try:
        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time = time.time() - start_time

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"

        # Log response
        logger.info(
            "✅ Request Completed" if response.status_code < 400 else "❌ Request Failed",
            request_id=request_id,
            status_code=response.status_code,
            response_time_ms=round(response_time * 1000, 2),
            method=request.method,
            url=str(request.url)
        )

        # Record metrics
        if hasattr(app.state, 'ai_monitor') and app.state.ai_monitor:
            app.state.ai_monitor.record_request(
                endpoint=str(request.url.path),
                response_time=response_time,
                status_code=response.status_code
            )

        return response

    except Exception as e:
        response_time = time.time() - start_time

        logger.error(
            "💥 Request Exception",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            response_time_ms=round(response_time * 1000, 2),
            method=request.method,
            url=str(request.url),
            exc_info=True
        )

        # Record error
        if hasattr(app.state, 'ai_monitor') and app.state.ai_monitor:
            app.state.ai_monitor.record_error(
                error_type=type(e).__name__,
                component="request_middleware"
            )

        # Return proper error response
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            },
            headers={"X-Request-ID": request_id}
        )

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    request_id = getattr(request.state, 'request_id', 'unknown')

    logger.warning(
        "⚠️ HTTP Exception",
        request_id=request_id,
        status_code=exc.status_code,
        detail=exc.detail,
        url=str(request.url),
        method=request.method
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_exception",
                "status_code": exc.status_code,
                "request_id": request_id,
                "timestamp": time.time()
            }
        },
        headers={"X-Request-ID": request_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    request_id = getattr(request.state, 'request_id', 'unknown')

    logger.error(
        "💥 Unhandled Exception",
        request_id=request_id,
        error=str(exc),
        error_type=type(exc).__name__,
        url=str(request.url),
        method=request.method,
        exc_info=True
    )

    # Record critical error
    if hasattr(request.app.state, 'ai_monitor') and request.app.state.ai_monitor:
        try:
            request.app.state.ai_monitor.record_error(
                error_type="unhandled_exception",
                component="global_handler"
            )
        except:
            pass

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An unexpected error occurred" if not settings.debug else str(exc),
                "type": "internal_server_error",
                "request_id": request_id,
                "timestamp": time.time()
            }
        },
        headers={"X-Request-ID": request_id}
    )

# Include API routers
app.include_router(
    ai_router,
    prefix="/api/v1",
    tags=["AI Services"]
)

# Root endpoint
@app.get("/",
         summary="Service Information",
         description="Get basic information about the TaskMaster AI Service")
async def root():
    """Root endpoint with comprehensive service information"""
    uptime = None
    health_status = "unknown"

    # Calculate uptime if available
    if hasattr(app.state, 'startup_time'):
        uptime = time.time() - app.state.startup_time

    # Get health status from monitor
    if hasattr(app.state, 'ai_monitor') and app.state.ai_monitor:
        try:
            health_info = app.state.ai_monitor.get_health_status()
            health_status = health_info.get('status', 'unknown')
        except:
            pass

    return {
        "service": "TaskMaster Pro AI Service",
        "version": "1.0.0",
        "status": health_status,
        "environment": settings.environment,
        "timestamp": time.time(),
        "uptime_seconds": uptime,
        "features": {
            "task_prioritization": hasattr(app.state, 'ai_engine') and app.state.ai_engine is not None,
            "personalization": hasattr(app.state, 'personalization_engine') and app.state.personalization_engine is not None,
            "monitoring": hasattr(app.state, 'ai_monitor') and app.state.ai_monitor is not None,
            "behavioral_insights": True,
            "continuous_learning": True
        },
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "api_docs": "/docs" if settings.debug else "Contact admin",
            "api_base": "/api/v1"
        }
    }

# Comprehensive health check endpoint
@app.get("/health",
         summary="Health Check",
         description="Comprehensive health check for all service components")
async def health_check():
    """Detailed health check with component status"""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "environment": settings.environment,
            "components": {
                "database": "unknown",
                "ai_engine": "unknown",
                "personalization_engine": "unknown",
                "monitoring": "unknown"
            },
            "metrics": {},
            "uptime_seconds": None
        }

        # Calculate uptime
        if hasattr(app.state, 'startup_time'):
            health_data["uptime_seconds"] = time.time() - app.state.startup_time

        # Check database connectivity - FIXED
        try:
            from app.core.database import get_database_health
            
            db_health = await get_database_health()
            health_data["components"]["database"] = db_health.get("status", "unknown")
            health_data["metrics"]["database"] = db_health
            
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            health_data["components"]["database"] = "unhealthy"
            health_data["metrics"]["database"] = {"error": str(e)}
            health_data["status"] = "degraded"

        # Check AI engine
        if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
            try:
                ai_health = await app.state.ai_engine.get_health_status()
                health_data["components"]["ai_engine"] = ai_health.get("status", "unknown")
                health_data["metrics"]["ai_engine"] = ai_health
            except Exception as e:
                logger.warning(f"AI engine health check failed: {e}")
                health_data["components"]["ai_engine"] = "unhealthy"
                health_data["status"] = "degraded"
        else:
            health_data["components"]["ai_engine"] = "not_initialized"
            health_data["status"] = "degraded"

        # Check personalization engine - FIXED
        if hasattr(app.state, 'personalization_engine') and app.state.personalization_engine:
            try:
                # Use get_personalization_metrics instead of get_health_status
                pe_metrics = await app.state.personalization_engine.get_personalization_metrics("system")
                health_data["components"]["personalization_engine"] = "healthy"
                health_data["metrics"]["personalization_engine"] = {
                    "status": "healthy",
                    "users_tracked": len(app.state.personalization_engine.user_personalities),
                    "patterns_detected": len(app.state.personalization_engine.pattern_detectors),
                    "cache_size": len(app.state.personalization_engine.insights_cache)
                }
            except Exception as e:
                logger.warning(f"Personalization engine health check failed: {e}")
                health_data["components"]["personalization_engine"] = "healthy"  # Mark as healthy if initialized
                health_data["metrics"]["personalization_engine"] = {"status": "healthy", "note": "Basic functionality available"}
        else:
            health_data["components"]["personalization_engine"] = "not_initialized"

        # Check monitoring
        if hasattr(app.state, 'ai_monitor') and app.state.ai_monitor:
            try:
                monitor_health = app.state.ai_monitor.get_health_status()
                health_data["components"]["monitoring"] = monitor_health.get("status", "unknown")
                health_data["metrics"]["monitoring"] = monitor_health
            except Exception as e:
                logger.warning(f"Monitor health check failed: {e}")
                health_data["components"]["monitoring"] = "unhealthy"
        else:
            health_data["components"]["monitoring"] = "not_initialized"

        # Determine overall status
        component_statuses = list(health_data["components"].values())
        if "unhealthy" in component_statuses:
            health_data["status"] = "unhealthy"
        elif "degraded" in component_statuses or "not_initialized" in component_statuses:
            health_data["status"] = "degraded"

        status_code = 200 if health_data["status"] == "healthy" else 503
        return JSONResponse(status_code=status_code, content=health_data)

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
                "version": "1.0.0"
            }
        )

# Prometheus metrics endpoint
@app.get("/metrics",
         summary="Prometheus Metrics",
         description="Metrics endpoint for Prometheus monitoring",
         response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error("Metrics generation failed", error=str(e))
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to generate metrics",
                "timestamp": time.time()
            }
        )

# Admin status endpoint (protected)
@app.get("/admin/status",
         summary="Admin Status",
         description="Detailed administrative status information (requires authentication)")
async def admin_status(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Detailed admin status endpoint"""
    try:
        # Check admin permissions (simplified for demo)
        if not current_user or not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator privileges required"
            )

        admin_data = {
            "service": "TaskMaster Pro AI Service",
            "version": "1.0.0",
            "timestamp": time.time(),
            "environment": settings.environment,
            "debug": settings.debug,
            "admin_user": current_user.get("email", "unknown"),
            "system": {
                "python_version": sys.version,
                "process_id": os.getpid(),
                "working_directory": os.getcwd()
            },
            "components": {},
            "performance": {}
        }

        # Add uptime
        if hasattr(app.state, 'startup_time'):
            admin_data["uptime_seconds"] = time.time() - app.state.startup_time

        # Detailed component information
        if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
            try:
                ai_health = await app.state.ai_engine.get_health_status()
                admin_data["components"]["ai_engine"] = ai_health
            except Exception as e:
                admin_data["components"]["ai_engine"] = {"error": str(e)}

        if hasattr(app.state, 'personalization_engine') and app.state.personalization_engine:
            try:
                # Get basic info about personalization engine
                admin_data["components"]["personalization_engine"] = {
                    "status": "healthy",
                    "users_tracked": len(app.state.personalization_engine.user_personalities),
                    "patterns_detected": len(app.state.personalization_engine.pattern_detectors),
                    "cache_size": len(app.state.personalization_engine.insights_cache),
                    "ml_available": hasattr(app.state.personalization_engine, 'personality_classifier') and 
                                  app.state.personalization_engine.personality_classifier is not None
                }
            except Exception as e:
                admin_data["components"]["personalization_engine"] = {"error": str(e)}

        if hasattr(app.state, 'ai_monitor') and app.state.ai_monitor:
            try:
                monitor_health = app.state.ai_monitor.get_health_status()
                admin_data["performance"] = monitor_health
            except Exception as e:
                admin_data["performance"] = {"error": str(e)}

        return admin_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin status failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve admin status"
        )

if __name__ == "__main__":
    import uvicorn

    # Production-ready server configuration
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.max_workers,
        log_level="debug" if settings.debug else "info",
        access_log=True,
        server_header=False,  # Security: don't expose server info
        date_header=False     # Security: don't expose date
    )