# ai-service/app/core/database.py
import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
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
            database_url = "postgresql+asyncpg://postgres:password@localhost:5432/taskmasterpro"
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
            await session.execute("SELECT 1")
            
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