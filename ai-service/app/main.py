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
from app.services.personalization_engine import personalization_engine
from app.core.monitoring import ai_monitor
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management with robust error handling
    """
    startup_start_time = time.time()
    
    try:
        logger.info(
            "🚀 Starting TaskMaster AI Service",
            version="1.0.0",
            environment=settings.environment,
            debug=settings.debug,
            python_version=sys.version,
            pid=os.getpid()
        )
        
        # Initialize database with retry logic
        logger.info("📦 Initializing database connection...")
        for attempt in range(3):
            try:
                await init_database()
                logger.info("✅ Database connection established")
                break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    logger.error("❌ Database initialization failed after 3 attempts", error=str(e))
                    raise
                logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...", error=str(e))
                await asyncio.sleep(5)
        
        # Initialize AI Engine
        logger.info("🧠 Initializing AI Engine...")
        try:
            await ai_engine.initialize()
            app.state.ai_engine = ai_engine
            logger.info("✅ AI Engine initialized successfully")
        except Exception as e:
            logger.error("❌ AI Engine initialization failed", error=str(e), exc_info=True)
            # Continue without AI engine for graceful degradation
            app.state.ai_engine = None
            logger.warning("⚠️ Running without AI Engine - some features will be unavailable")
        
        # Initialize Personalization Engine
        logger.info("🎯 Initializing Personalization Engine...")
        try:
            await personalization_engine.initialize()
            app.state.personalization_engine = personalization_engine
            logger.info("✅ Personalization Engine initialized successfully")
        except Exception as e:
            logger.error("❌ Personalization Engine initialization failed", error=str(e), exc_info=True)
            app.state.personalization_engine = None
            logger.warning("⚠️ Running without Personalization Engine - basic features only")
        
        # Initialize Monitoring
        logger.info("🔍 Starting monitoring system...")
        try:
            await ai_monitor.start_monitoring()
            app.state.ai_monitor = ai_monitor
            logger.info("✅ Monitoring system started")
        except Exception as e:
            logger.error("❌ Monitoring initialization failed", error=str(e), exc_info=True)
            app.state.ai_monitor = None
            logger.warning("⚠️ Running without monitoring")
        
        # Calculate and log startup time
        startup_duration = time.time() - startup_start_time
        app.state.startup_time = startup_start_time
        
        # Determine service health
        components_healthy = sum([
            app.state.ai_engine is not None,
            app.state.personalization_engine is not None,
            app.state.ai_monitor is not None
        ])
        
        if components_healthy == 3:
            health_status = "fully_operational"
        elif components_healthy >= 1:
            health_status = "degraded_mode"
        else:
            health_status = "minimal_mode"
        
        logger.info(
            "🎉 TaskMaster AI Service startup complete",
            startup_duration_seconds=f"{startup_duration:.2f}",
            health_status=health_status,
            components_initialized=components_healthy,
            total_components=3
        )
        
        yield
        
    except Exception as e:
        logger.error(
            "❌ Critical startup failure",
            error=str(e),
            traceback=traceback.format_exc()
        )
        # Re-raise to prevent the app from starting in a broken state
        raise
    
    finally:
        # Graceful shutdown
        logger.info("🔄 Initiating graceful shutdown...")
        shutdown_start = time.time()
        
        shutdown_errors = []
        
        # Cleanup monitoring
        if hasattr(app.state, 'ai_monitor') and app.state.ai_monitor:
            try:
                await app.state.ai_monitor.cleanup()
                logger.info("✅ Monitoring cleanup complete")
            except Exception as e:
                shutdown_errors.append(f"Monitoring cleanup: {e}")
        
        # Cleanup personalization engine
        if hasattr(app.state, 'personalization_engine') and app.state.personalization_engine:
            try:
                await app.state.personalization_engine.cleanup()
                logger.info("✅ Personalization Engine cleanup complete")
            except Exception as e:
                shutdown_errors.append(f"Personalization cleanup: {e}")
        
        # Cleanup AI engine
        if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
            try:
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

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application with comprehensive setup
    """
    
    # Application metadata
    app_metadata = {
        "title": "TaskMaster Pro AI Service",
        "description": """
## 🤖 Intelligent Task Management AI Service

A production-ready AI service that provides personalized task management capabilities.

### 🌟 Core Features

- **🎯 Personalized Task Prioritization**: ML-powered priority scoring that learns from your behavior
- **⏰ Smart Time Estimation**: Accurate completion time predictions based on your patterns  
- **📊 Behavioral Insights**: Deep analysis of productivity patterns and personalized recommendations
- **🧠 Continuous Learning**: Models that adapt and improve with every interaction
- **🔒 Privacy-First**: Local AI models that keep your data secure

### 🚀 AI Capabilities

- **Local-First Architecture**: Fast, private AI inference without external API dependencies
- **Vector Similarity Search**: Semantic task matching and recommendation engine
- **Natural Language Processing**: Advanced text analysis for task understanding
- **Predictive Analytics**: Forecast productivity patterns and optimize workflows
- **Real-Time Personalization**: Individual user models that evolve with usage

### 📈 Production Features

- **High Availability**: Graceful degradation and fault tolerance
- **Comprehensive Monitoring**: Detailed metrics, health checks, and alerting
- **Horizontal Scaling**: Kubernetes-ready for enterprise deployment
- **Security**: Authentication, rate limiting, and data encryption
- **Performance**: Optimized for low latency and high throughput

### 🔗 API Endpoints

- `POST /api/v1/ai/prioritize-tasks` - Get AI-powered task prioritization
- `POST /api/v1/ai/estimate-time` - Predict task completion times
- `GET /api/v1/ai/user-insights` - Retrieve behavioral insights and recommendations
- `POST /api/v1/ai/train-model` - Train personalized models with user data
- `POST /api/v1/ai/smart-analysis` - Comprehensive task analysis with AI insights

### 📚 Documentation

Visit `/docs` for interactive API documentation and testing interface.
        """,
        "version": "1.0.0",
        "contact": {
            "name": "TaskMaster Pro Team",
            "email": "support@taskmaster.com",
            "url": "https://taskmaster.com/support"
        },
        "license_info": {
            "name": "Proprietary",
            "url": "https://taskmaster.com/license"
        },
        "terms_of_service": "https://taskmaster.com/terms"
    }
    
    # Create FastAPI app with conditional docs
    app = FastAPI(
        **app_metadata,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
        # Custom OpenAPI schema
        swagger_ui_parameters={
            "syntaxHighlight.theme": "obsidian",
            "tryItOutEnabled": True,
            "filter": True
        }
    )
    
    # Add security middleware for production
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=[
                "localhost",
                "127.0.0.1",
                "*.taskmaster.com",
                "*.taskmaster.ai"
            ]
        )
    
    # CORS middleware with production-ready settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins if settings.allowed_origins else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID", 
            "X-Processing-Time", 
            "X-Rate-Limit-Remaining",
            "X-Health-Status"
        ],
        max_age=3600  # Cache preflight requests for 1 hour
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request tracking and logging middleware
    @app.middleware("http")
    async def request_tracking_middleware(request: Request, call_next):
        """
        Comprehensive request tracking with metrics and security
        """
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000000)}_{os.getpid()}"
        
        # Extract client information
        client_ip = request.headers.get("x-forwarded-for", 
                                      request.headers.get("x-real-ip", 
                                                        request.client.host if request.client else "unknown"))
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Create request context
        request_context = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": user_agent[:200],  # Truncate long user agents
            "content_length": request.headers.get("content-length", 0)
        }
        
        # Log incoming request
        logger.info("📥 HTTP Request", **request_context)
        
        # Add request ID to request state for downstream use
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing metrics
            process_time = time.time() - start_time
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{process_time:.3f}s"
            
            # Add health status if available
            if hasattr(request.app.state, 'ai_monitor') and request.app.state.ai_monitor:
                try:
                    health = request.app.state.ai_monitor.get_health_status()
                    response.headers["X-Health-Status"] = health.get('status', 'unknown')
                except:
                    pass
            
            # Log successful response
            logger.info(
                "📤 HTTP Response",
                request_id=request_id,
                status_code=response.status_code,
                processing_time_seconds=f"{process_time:.3f}",
                response_size=response.headers.get("content-length", "unknown")
            )
            
            # Record metrics for AI endpoints
            if hasattr(request.app.state, 'ai_monitor') and request.app.state.ai_monitor:
                if request.url.path.startswith("/api/v1/ai/"):
                    try:
                        # Extract user info from auth header if available
                        user_id = "anonymous"
                        auth_header = request.headers.get("authorization")
                        if auth_header:
                            # This would extract user ID from JWT token
                            # For now, using a placeholder
                            user_id = "authenticated_user"
                        
                        # Record API call metrics
                        request.app.state.ai_monitor.record_prediction(
                            user_id=user_id,
                            prediction_type=request.url.path.split("/")[-1],  # Last path segment
                            duration=process_time
                        )
                    except Exception as e:
                        logger.warning("Failed to record metrics", error=str(e))
            
            return response
            
        except Exception as e:
            # Calculate error processing time
            process_time = time.time() - start_time
            
            # Log error with context
            logger.error(
                "❌ HTTP Request Failed",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
                processing_time_seconds=f"{process_time:.3f}",
                traceback=traceback.format_exc() if settings.debug else None
            )
            
            # Record error metrics
            if hasattr(request.app.state, 'ai_monitor') and request.app.state.ai_monitor:
                try:
                    request.app.state.ai_monitor.record_error(
                        error_type=type(e).__name__,
                        component="api_middleware"
                    )
                except:
                    pass
            
            # Return user-friendly error response
            error_detail = str(e) if settings.debug else "Internal server error"
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": error_detail,
                        "type": "internal_server_error",
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                },
                headers={"X-Request-ID": request_id}
            )
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with structured logging"""
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
        prefix="/api/v1/ai",
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
        
        # Get health status
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
                "api_base": "/api/v1/ai"
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
            
            # Check database connectivity
            try:
                from app.core.database import get_async_session
                async with get_async_session() as db:
                    await db.execute("SELECT 1")
                health_data["components"]["database"] = "healthy"
            except Exception as e:
                health_data["components"]["database"] = "unhealthy"
                health_data["status"] = "degraded"
                logger.warning("Database health check failed", error=str(e))
            
            # Check AI engine
            if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
                health_data["components"]["ai_engine"] = "healthy"
                health_data["metrics"]["loaded_user_models"] = len(app.state.ai_engine.user_models)
                health_data["metrics"]["global_models"] = len(app.state.ai_engine.global_models)
            else:
                health_data["components"]["ai_engine"] = "unavailable"
                health_data["status"] = "degraded"
            
            # Check personalization engine
            if hasattr(app.state, 'personalization_engine') and app.state.personalization_engine:
                health_data["components"]["personalization_engine"] = "healthy"
                health_data["metrics"]["loaded_personalities"] = len(app.state.personalization_engine.user_personalities)
            else:
                health_data["components"]["personalization_engine"] = "unavailable"
                health_data["status"] = "degraded"
            
            # Check monitoring
            if hasattr(app.state, 'ai_monitor') and app.state.ai_monitor:
                health_data["components"]["monitoring"] = "healthy"
                try:
                    monitor_health = app.state.ai_monitor.get_health_status()
                    health_data["monitoring"] = monitor_health
                except Exception:
                    pass
            else:
                health_data["components"]["monitoring"] = "unavailable"
            
            # Determine overall status
            unhealthy_components = [k for k, v in health_data["components"].items() if v == "unhealthy"]
            unavailable_components = [k for k, v in health_data["components"].items() if v == "unavailable"]
            
            if unhealthy_components:
                health_data["status"] = "unhealthy"
                return JSONResponse(status_code=503, content=health_data)
            elif unavailable_components:
                health_data["status"] = "degraded"
                return JSONResponse(status_code=200, content=health_data)
            
            return health_data
            
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
            # Check admin permissions
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
            
            # Component details
            if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
                admin_data["components"]["ai_engine"] = {
                    "status": "operational",
                    "user_models_loaded": len(app.state.ai_engine.user_models),
                    "global_models_loaded": len(app.state.ai_engine.global_models),
                    "embedding_model_loaded": app.state.ai_engine.embedding_model is not None,
                    "nlp_model_loaded": app.state.ai_engine.nlp_model is not None
                }
            
            if hasattr(app.state, 'personalization_engine') and app.state.personalization_engine:
                admin_data["components"]["personalization_engine"] = {
                    "status": "operational",
                    "personalities_loaded": len(app.state.personalization_engine.user_personalities),
                    "insights_cached": len(app.state.personalization_engine.insight_cache)
                }
            
            # Performance data
            if hasattr(app.state, 'ai_monitor') and app.state.ai_monitor:
                try:
                    performance_summary = app.state.ai_monitor.get_performance_summary()
                    admin_data["performance"] = performance_summary
                except Exception as e:
                    admin_data["performance"] = {"error": str(e)}
            
            return admin_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Admin status failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve admin status"
            )
    
    # Model reload endpoint (admin only)
    @app.post("/admin/reload-models",
              summary="Reload AI Models",
              description="Reload all AI models (administrator only)")
    async def admin_reload_models(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
        """Reload AI models endpoint"""
        try:
            # Check admin permissions
            if not current_user or not current_user.get("is_admin", False):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Administrator privileges required"
                )
            
            reload_results = {
                "timestamp": time.time(),
                "admin_user": current_user.get("email", "unknown"),
                "results": {}
            }
            
            # Reload AI engine
            if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
                try:
                    await app.state.ai_engine.cleanup()
                    await app.state.ai_engine.initialize()
                    reload_results["results"]["ai_engine"] = "success"
                    logger.info("AI models reloaded by admin", admin_user=current_user.get("email"))
                except Exception as e:
                    reload_results["results"]["ai_engine"] = f"failed: {str(e)}"
                    logger.error("AI engine reload failed", error=str(e))
            else:
                reload_results["results"]["ai_engine"] = "not_available"
            
            # Reload personalization engine
            if hasattr(app.state, 'personalization_engine') and app.state.personalization_engine:
                try:
                    await app.state.personalization_engine.cleanup()
                    await app.state.personalization_engine.initialize()
                    reload_results["results"]["personalization_engine"] = "success"
                except Exception as e:
                    reload_results["results"]["personalization_engine"] = f"failed: {str(e)}"
                    logger.error("Personalization engine reload failed", error=str(e))
            else:
                reload_results["results"]["personalization_engine"] = "not_available"
            
            return {
                "status": "completed",
                "message": "Model reload operation completed",
                **reload_results
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Model reload operation failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail="Model reload operation failed"
            )
    
    return app

# Create the application instance
app = create_app()

# For development - run directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🔧 Starting AI Service in development mode")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )