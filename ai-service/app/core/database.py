# ai-service/app/core/database.py - Updated with fixed health check
import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# Global variables
async_engine = None
async_session_maker = None

async def init_database():
    """Initialize the async database connection"""
    global async_engine, async_session_maker
    
    try:
        from app.core.config import get_settings
        settings = get_settings()
        
        # Use the database URL from settings
        database_url = getattr(settings, 'ai_database_url', None) or getattr(settings, 'database_url', None)
        
        if not database_url:
            # Fallback database URL
            database_url = "postgresql+asyncpg://postgres:nishant%3F%40980@localhost:5432/taskmasterpro"
            logger.warning(f"Using fallback database URL")
        
        logger.info(f"Connecting to database...")
        
        # Create async engine
        async_engine = create_async_engine(
            database_url,
            echo=getattr(settings, 'debug', False),
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # Create session maker
        async_session_maker = async_sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test the connection
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
            
        logger.info("✅ Database connection established successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        # Don't raise the error - allow graceful degradation
        pass

async def close_database():
    """Close the database connection"""
    global async_engine
    
    if async_engine:
        await async_engine.dispose()
        logger.info("✅ Database connection closed")

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session"""
    if not async_session_maker:
        raise RuntimeError("Database not initialized")
    
    async with async_session_maker() as session:
        yield session

@asynccontextmanager
async def get_db_session():
    """Context manager for database sessions"""
    async for session in get_async_session():
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()

# FIXED: Updated health check functions
async def test_database_connection() -> bool:
    """Test if database connection is working"""
    try:
        if not async_session_maker:
            return False
            
        # Use the session properly with async context manager
        async with async_session_maker() as session:
            result = await session.execute(text("SELECT 1 as test"))
            test_value = result.scalar()
            return test_value == 1
            
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

async def get_database_health() -> dict:
    """Get database health status - FIXED"""
    try:
        # Check if components are initialized
        if not async_engine or not async_session_maker:
            return {
                "status": "unhealthy",
                "error": "Database not initialized",
                "connected": False,
                "engine_initialized": async_engine is not None,
                "session_maker_initialized": async_session_maker is not None
            }
        
        # Test connection with proper error handling
        is_connected = False
        try:
            is_connected = await test_database_connection()
        except Exception as conn_error:
            logger.error(f"Connection test failed: {conn_error}")
        
        health_info = {
            "status": "healthy" if is_connected else "unhealthy",
            "connected": is_connected,
            "engine_initialized": async_engine is not None,
            "session_maker_initialized": async_session_maker is not None
        }
        
        # Add pool information if available
        if async_engine and hasattr(async_engine, 'pool'):
            try:
                pool = async_engine.pool
                health_info.update({
                    "pool_size": pool.size(),
                    "checked_out_connections": pool.checkedout(),
                    "overflow_connections": pool.overflow(),
                    "checked_in_connections": pool.checkedin()
                })
            except Exception as pool_error:
                health_info["pool_error"] = str(pool_error)
            
        return health_info
        
    except Exception as e:
        logger.error(f"Database health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "connected": False,
            "engine_initialized": async_engine is not None,
            "session_maker_initialized": async_session_maker is not None
        }

# Simple synchronous health check for compatibility
def get_database_status() -> dict:
    """Get basic database status synchronously"""
    return {
        "engine_initialized": async_engine is not None,
        "session_maker_initialized": async_session_maker is not None,
        "status": "initialized" if (async_engine and async_session_maker) else "not_initialized"
    }