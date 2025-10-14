"""
YMERA Enterprise - Database Wrapper Module
Production-Ready Database Connection Management - v4.0
Centralized database access for API Gateway routes
"""

# ===============================================================================
# STANDARD IMPORTS
# ===============================================================================

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
import structlog

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.database")

# ===============================================================================
# DATABASE BASE
# ===============================================================================

Base = declarative_base()

# ===============================================================================
# DATABASE ENGINE CONFIGURATION
# ===============================================================================

class DatabaseConfig:
    """Database configuration with environment-based settings"""
    
    def __init__(self):
        # Try to import settings, fallback to environment variables
        try:
            from app.CORE_CONFIGURATION.config_settings import get_settings
            settings = get_settings()
            self.database_url = settings.DATABASE_URL
            self.echo = settings.DATABASE_ECHO
            self.pool_size = settings.DATABASE_POOL_SIZE
            self.max_overflow = settings.DATABASE_MAX_OVERFLOW
        except ImportError:
            import os
            self.database_url = os.getenv(
                "DATABASE_URL",
                "postgresql+asyncpg://postgres:postgres@localhost:5432/ymera"
            )
            self.echo = os.getenv("DATABASE_ECHO", "False").lower() == "true"
            self.pool_size = int(os.getenv("DATABASE_POOL_SIZE", "5"))
            self.max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))

# ===============================================================================
# DATABASE ENGINE AND SESSION MANAGEMENT
# ===============================================================================

class DatabaseManager:
    """Centralized database manager for all API routes"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
        self.logger = logger.bind(component="database_manager")
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory"""
        if self._initialized:
            self.logger.warning("Database already initialized")
            return
        
        try:
            # Create async engine with proper pooling
            self._engine = create_async_engine(
                self.config.database_url,
                echo=self.config.echo,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
            
            self._initialized = True
            self.logger.info(
                "Database initialized successfully",
                url=self.config.database_url.split('@')[1] if '@' in self.config.database_url else "local",
                pool_size=self.config.pool_size
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize database", error=str(e))
            raise RuntimeError(f"Database initialization failed: {e}") from e
    
    async def dispose(self) -> None:
        """Dispose of database engine and cleanup resources"""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            self.logger.info("Database disposed successfully")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup.
        
        Yields:
            AsyncSession: Database session
            
        Example:
            async with db_manager.get_session() as session:
                result = await session.execute(query)
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._session_factory:
            raise RuntimeError("Database session factory not initialized")
        
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self.logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()
    
    async def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            bool: True if database is healthy
        """
        try:
            if not self._initialized:
                return False
            
            async with self.get_session() as session:
                await session.execute("SELECT 1")
                return True
                
        except Exception as e:
            self.logger.error("Database health check failed", error=str(e))
            return False

# ===============================================================================
# GLOBAL DATABASE MANAGER INSTANCE
# ===============================================================================

# Create singleton instance
_db_manager = DatabaseManager()

# ===============================================================================
# DEPENDENCY INJECTION FUNCTIONS
# ===============================================================================

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting database session.
    
    Yields:
        AsyncSession: Database session
        
    Example:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with _db_manager.get_session() as session:
        yield session

async def init_database() -> None:
    """Initialize database on application startup"""
    await _db_manager.initialize()

async def close_database() -> None:
    """Close database on application shutdown"""
    await _db_manager.dispose()

# ===============================================================================
# DATABASE UTILITIES
# ===============================================================================

async def create_tables() -> None:
    """
    Create all database tables.
    
    Warning: This should only be used in development.
    Use Alembic migrations in production.
    """
    if not _db_manager._initialized:
        await _db_manager.initialize()
    
    if not _db_manager._engine:
        raise RuntimeError("Database engine not initialized")
    
    async with _db_manager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created successfully")

async def drop_tables() -> None:
    """
    Drop all database tables.
    
    Warning: This is destructive and should only be used in development.
    """
    if not _db_manager._initialized:
        await _db_manager.initialize()
    
    if not _db_manager._engine:
        raise RuntimeError("Database engine not initialized")
    
    async with _db_manager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.warning("Database tables dropped")

# ===============================================================================
# TRANSACTION MANAGEMENT
# ===============================================================================

@asynccontextmanager
async def transaction(session: AsyncSession):
    """
    Context manager for explicit transaction handling.
    
    Args:
        session: Database session
        
    Example:
        async with transaction(session):
            # Perform multiple operations
            session.add(item1)
            session.add(item2)
            # Automatic commit on success, rollback on error
    """
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error("Transaction failed", error=str(e))
        raise

# ===============================================================================
# HEALTH CHECK ENDPOINT HELPER
# ===============================================================================

async def get_database_stats() -> dict:
    """
    Get database connection pool statistics.
    
    Returns:
        dict: Database statistics
    """
    if not _db_manager._engine:
        return {"status": "not_initialized"}
    
    pool = _db_manager._engine.pool
    
    return {
        "status": "initialized",
        "pool_size": pool.size(),
        "checked_in_connections": pool.checkedin(),
        "checked_out_connections": pool.checkedout(),
        "overflow_connections": pool.overflow(),
        "total_connections": pool.size() + pool.overflow(),
    }

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    # Core components
    "Base",
    "DatabaseManager",
    "DatabaseConfig",
    
    # Dependency injection
    "get_db_session",
    "init_database",
    "close_database",
    
    # Utilities
    "create_tables",
    "drop_tables",
    "transaction",
    "get_database_stats",
    
    # Global instance (use with caution)
    "_db_manager",
]
