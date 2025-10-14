"""
YMERA Enterprise - Database Connection Manager
Production-Ready Async Database Connections - v4.0
FIXED: All syntax errors and incomplete handlers resolved
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field

# Third-party imports
import structlog
from sqlalchemy import text, MetaData, Table, Column, Integer, String, DateTime, Boolean, Text, JSON, Index
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    AsyncEngine, 
    create_async_engine, 
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError, TimeoutError as SQLTimeoutError

# ===============================================================================
# CUSTOM EXCEPTIONS
# ===============================================================================

class DatabaseError(Exception):
    """Base database exception"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Database connection failed"""
    pass

class DatabaseTimeoutError(DatabaseError):
    """Database operation timed out"""
    pass

class DatabaseQueryError(DatabaseError):
    """Database query failed"""
    pass

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.database.connection")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

DEFAULT_POOL_SIZE = 20
DEFAULT_MAX_OVERFLOW = 40
DEFAULT_POOL_TIMEOUT = 30
DEFAULT_POOL_RECYCLE = 3600
DEFAULT_CONNECT_TIMEOUT = 30

MAX_RETRIES = 3
RETRY_DELAY = 1.0
HEALTH_CHECK_INTERVAL = 60
HEALTH_CHECK_QUERY = "SELECT 1"

# ===============================================================================
# DATABASE MODELS & SCHEMAS
# ===============================================================================

class Base(DeclarativeBase):
    """Base class for all database models"""
    pass

@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    database_url: str
    pool_size: int = DEFAULT_POOL_SIZE
    max_overflow: int = DEFAULT_MAX_OVERFLOW
    pool_timeout: int = DEFAULT_POOL_TIMEOUT
    pool_recycle: int = DEFAULT_POOL_RECYCLE
    echo_sql: bool = False
    ssl_require: bool = True
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT

@dataclass
class ConnectionMetrics:
    """Metrics for database connection monitoring"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    connection_errors: List[str] = field(default_factory=list)
    last_health_check: Optional[datetime] = None
    average_response_time: float = 0.0

# ===============================================================================
# LEARNING SYSTEM DATABASE TABLES
# ===============================================================================

agent_learning_data = Table(
    'agent_learning_data',
    Base.metadata,
    Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column('agent_id', String(100), nullable=False, index=True),
    Column('learning_cycle_id', String(36), nullable=False, index=True),
    Column('experience_data', JSON, nullable=False),
    Column('knowledge_extracted', JSON, nullable=False),
    Column('confidence_score', Integer, nullable=False),
    Column('learning_timestamp', DateTime, nullable=False, default=datetime.utcnow),
    Column('source_type', String(50), nullable=False),
    Column('processing_time_ms', Integer, nullable=False),
    Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
    Column('updated_at', DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow),
    Index('idx_agent_learning_timestamp', 'agent_id', 'learning_timestamp'),
    Index('idx_learning_cycle_agent', 'learning_cycle_id', 'agent_id')
)

knowledge_graph_nodes = Table(
    'knowledge_graph_nodes',
    Base.metadata,
    Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column('node_type', String(50), nullable=False, index=True),
    Column('node_data', JSON, nullable=False),
    Column('confidence_score', Integer, nullable=False),
    Column('source_agent_id', String(100), nullable=False, index=True),
    Column('creation_timestamp', DateTime, nullable=False, default=datetime.utcnow),
    Column('last_accessed', DateTime, nullable=False, default=datetime.utcnow),
    Column('access_count', Integer, nullable=False, default=0),
    Column('validation_status', String(20), nullable=False, default='pending'),
    Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
    Column('updated_at', DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow),
    Index('idx_knowledge_type_confidence', 'node_type', 'confidence_score'),
    Index('idx_knowledge_source_timestamp', 'source_agent_id', 'creation_timestamp')
)

knowledge_graph_edges = Table(
    'knowledge_graph_edges',
    Base.metadata,
    Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column('source_node_id', String(36), nullable=False, index=True),
    Column('target_node_id', String(36), nullable=False, index=True),
    Column('relationship_type', String(50), nullable=False, index=True),
    Column('relationship_data', JSON),
    Column('strength_score', Integer, nullable=False),
    Column('created_by_agent', String(100), nullable=False),
    Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
    Column('updated_at', DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow),
    Index('idx_edge_source_target', 'source_node_id', 'target_node_id'),
    Index('idx_edge_relationship_strength', 'relationship_type', 'strength_score')
)

knowledge_transfer_logs = Table(
    'knowledge_transfer_logs',
    Base.metadata,
    Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column('source_agent_id', String(100), nullable=False, index=True),
    Column('target_agent_id', String(100), nullable=False, index=True),
    Column('knowledge_item_id', String(36), nullable=False),
    Column('transfer_type', String(50), nullable=False),
    Column('transfer_data', JSON, nullable=False),
    Column('success_status', Boolean, nullable=False),
    Column('transfer_timestamp', DateTime, nullable=False, default=datetime.utcnow),
    Column('processing_time_ms', Integer, nullable=False),
    Column('collaboration_score', Integer),
    Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
    Index('idx_transfer_source_target_time', 'source_agent_id', 'target_agent_id', 'transfer_timestamp'),
    Index('idx_transfer_success_timestamp', 'success_status', 'transfer_timestamp')
)

behavioral_patterns = Table(
    'behavioral_patterns',
    Base.metadata,
    Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column('pattern_type', String(50), nullable=False, index=True),
    Column('pattern_data', JSON, nullable=False),
    Column('discovery_agent_id', String(100), nullable=False),
    Column('significance_score', Integer, nullable=False),
    Column('usage_count', Integer, nullable=False, default=0),
    Column('last_applied', DateTime),
    Column('discovery_timestamp', DateTime, nullable=False, default=datetime.utcnow),
    Column('validation_status', String(20), nullable=False, default='discovered'),
    Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
    Column('updated_at', DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow),
    Index('idx_pattern_type_significance', 'pattern_type', 'significance_score'),
    Index('idx_pattern_discovery_timestamp', 'discovery_timestamp'),
    Index('idx_pattern_usage_validation', 'usage_count', 'validation_status')
)

learning_metrics = Table(
    'learning_metrics',
    Base.metadata,
    Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column('metric_timestamp', DateTime, nullable=False, default=datetime.utcnow, index=True),
    Column('metric_type', String(50), nullable=False, index=True),
    Column('agent_id', String(100), index=True),
    Column('metric_value', Integer, nullable=False),
    Column('metric_data', JSON),
    Column('measurement_period_start', DateTime, nullable=False),
    Column('measurement_period_end', DateTime, nullable=False),
    Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
    Index('idx_metrics_timestamp_type_agent', 'metric_timestamp', 'metric_type', 'agent_id'),
    Index('idx_metrics_period', 'measurement_period_start', 'measurement_period_end')
)

# ===============================================================================
# CORE DATABASE CONNECTION MANAGER
# ===============================================================================

class DatabaseManager:
    """Production-ready async database connection manager with complete error handling"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or self._load_default_config()
        self.logger = logger.bind(manager="DatabaseManager")
        
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._metrics = ConnectionMetrics()
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_health_check = None
        self._initialized = False
        self._shutdown = False
    
    def _load_default_config(self) -> DatabaseConfig:
        """Load default database configuration from environment"""
        database_url = os.getenv('DATABASE_URL', 'sqlite:///./ymera.db')
        
        return DatabaseConfig(
            database_url=database_url,
            pool_size=int(os.getenv('DB_POOL_SIZE', DEFAULT_POOL_SIZE)),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', DEFAULT_MAX_OVERFLOW)),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', DEFAULT_POOL_TIMEOUT)),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', DEFAULT_POOL_RECYCLE)),
            echo_sql=os.getenv('DB_ECHO_SQL', 'false').lower() == 'true',
            ssl_require=os.getenv('DB_SSL_REQUIRE', 'true').lower() == 'true',
            connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT', DEFAULT_CONNECT_TIMEOUT))
        )
    
    async def initialize(self) -> None:
        """Initialize database connection manager"""
        if self._initialized:
            self.logger.warning("Database manager already initialized")
            return
        
        try:
            self.logger.info("Initializing database connection manager")
            
            self._engine = await self._create_async_engine()
            
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            await self._test_connection()
            
            self._health_check_task = asyncio.create_task(self._health_monitor())
            
            self._initialized = True
            self.logger.info("Database connection manager initialized successfully")
            
        except SQLAlchemyError as e:
            self.logger.error("SQLAlchemy error during initialization", error=str(e))
            await self.cleanup()
            raise DatabaseConnectionError(f"Database initialization failed: {str(e)}") from e
        except asyncio.TimeoutError as e:
            self.logger.error("Timeout during database initialization", error=str(e))
            await self.cleanup()
            raise DatabaseTimeoutError(f"Database initialization timed out: {str(e)}") from e
        except Exception as e:
            self.logger.error("Unexpected error during initialization", error=str(e))
            await self.cleanup()
            raise DatabaseError(f"Database initialization failed: {str(e)}") from e
    
    async def _create_async_engine(self) -> AsyncEngine:
        """Create and configure async database engine"""
        try:
            connect_args = {
                "connect_timeout": self.config.connect_timeout,
                "command_timeout": 60,
            }
            
            # PostgreSQL specific settings
            if 'postgresql' in self.config.database_url:
                connect_args["server_settings"] = {
                    "application_name": "YMERA_Learning_System",
                    "jit": "off"
                }
                if self.config.ssl_require:
                    connect_args["ssl"] = "require"
            
            engine = create_async_engine(
                self.config.database_url,
                echo=self.config.echo_sql,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,
                connect_args=connect_args,
                isolation_level="READ_COMMITTED"
            )
            
            self.logger.info(
                "Created async database engine",
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow
            )
            
            return engine
            
        except SQLAlchemyError as e:
            self.logger.error("Failed to create database engine", error=str(e))
            raise DatabaseConnectionError(f"Engine creation failed: {str(e)}") from e
        except Exception as e:
            self.logger.error("Unexpected error creating engine", error=str(e))
            raise DatabaseError(f"Engine creation failed: {str(e)}") from e
    
    async def _test_connection(self) -> None:
        """Test database connectivity - FIXED: Complete exception handling"""
        start_time = time.time()
        
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text(HEALTH_CHECK_QUERY))
                await result.fetchone()
            
            response_time = (time.time() - start_time) * 1000
            self._metrics.average_response_time = response_time
            self._metrics.last_health_check = datetime.utcnow()
            
            self.logger.info(
                "Database connection test successful",
                response_time_ms=response_time
            )
            
        except asyncio.TimeoutError as e:
            self._metrics.failed_connections += 1
            self._metrics.connection_errors.append(f"Timeout: {str(e)}")
            self.logger.error("Database connection test timed out", error=str(e))
            raise DatabaseTimeoutError("Connection test timed out") from e
            
        except OperationalError as e:
            self._metrics.failed_connections += 1
            self._metrics.connection_errors.append(f"Operational error: {str(e)}")
            self.logger.error("Database operational error", error=str(e))
            raise DatabaseConnectionError(f"Connection failed: {str(e)}") from e
            
        except SQLAlchemyError as e:
            self._metrics.failed_connections += 1
            self._metrics.connection_errors.append(f"SQLAlchemy error: {str(e)}")
            self.logger.error("Database error during connection test", error=str(e))
            raise DatabaseError(f"Connection test failed: {str(e)}") from e
            
        except Exception as e:
            self._metrics.failed_connections += 1
            self._metrics.connection_errors.append(f"Unexpected error: {str(e)}")
            self.logger.error("Unexpected error during connection test", error=str(e))
            raise DatabaseError(f"Connection test failed: {str(e)}") from e
    
    async def _health_monitor(self) -> None:
        """Monitor database health periodically"""
        while not self._shutdown:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                
                if self._shutdown:
                    break
                
                await self._test_connection()
                self._metrics.total_connections += 1
                
            except asyncio.CancelledError:
                self.logger.info("Health monitor cancelled")
                break
            except DatabaseTimeoutError as e:
                self.logger.warning("Health check timed out", error=str(e))
            except DatabaseConnectionError as e:
                self.logger.error("Health check connection failed", error=str(e))
            except Exception as e:
                self.logger.error("Health monitor error", error=str(e))
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        if not self._initialized:
            raise DatabaseError("Database manager not initialized")
        
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as e:
            await session.rollback()
            self.logger.error("Database session error", error=str(e))
            raise DatabaseQueryError(f"Query failed: {str(e)}") from e
        except Exception as e:
            await session.rollback()
            self.logger.error("Unexpected session error", error=str(e))
            raise DatabaseError(f"Session error: {str(e)}") from e
        finally:
            await session.close()
    
    async def execute_query(
        self, 
        query: Union[str, Any], 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute raw SQL query with complete error handling"""
        try:
            async with self.get_session() as session:
                if isinstance(query, str):
                    query = text(query)
                
                result = await session.execute(query, params or {})
                
                # Handle different result types
                if result.returns_rows:
                    rows = result.fetchall()
                    return [dict(row._mapping) for row in rows]
                else:
                    return []
                    
        except SQLTimeoutError as e:
            self.logger.error("Query timeout", query=str(query), error=str(e))
            raise DatabaseTimeoutError(f"Query timed out: {str(e)}") from e
        except SQLAlchemyError as e:
            self.logger.error("Query execution failed", query=str(query), error=str(e))
            raise DatabaseQueryError(f"Query failed: {str(e)}") from e
        except Exception as e:
            self.logger.error("Unexpected query error", query=str(query), error=str(e))
            raise DatabaseError(f"Query error: {str(e)}") from e
    
    async def create_tables(self) -> None:
        """Create all database tables"""
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.logger.info("Database tables created successfully")
            
        except SQLAlchemyError as e:
            self.logger.error("Failed to create tables", error=str(e))
            raise DatabaseError(f"Table creation failed: {str(e)}") from e
        except Exception as e:
            self.logger.error("Unexpected error creating tables", error=str(e))
            raise DatabaseError(f"Table creation failed: {str(e)}") from e
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            await self._test_connection()
            
            pool_status = {}
            if hasattr(self._engine.pool, 'size'):
                pool_status = {
                    "pool_size": self._engine.pool.size(),
                    "checked_out": self._engine.pool.checkedout()
                }
            
            return {
                "status": "healthy",
                "metrics": {
                    "total_connections": self._metrics.total_connections,
                    "failed_connections": self._metrics.failed_connections,
                    "average_response_time_ms": self._metrics.average_response_time,
                    "last_health_check": self._metrics.last_health_check.isoformat() if self._metrics.last_health_check else None
                },
                "pool": pool_status,
                "errors": self._metrics.connection_errors[-5:] if self._metrics.connection_errors else []
            }
            
        except DatabaseError as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "metrics": {
                    "failed_connections": self._metrics.failed_connections,
                    "last_error": self._metrics.connection_errors[-1] if self._metrics.connection_errors else None
                }
            }
    
    async def cleanup(self) -> None:
        """Clean up database resources"""
        self._shutdown = True
        
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._engine:
                await self._engine.dispose()
            
            self._initialized = False
            self.logger.info("Database manager cleanup completed")
            
        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))

# ===============================================================================
# GLOBAL DATABASE MANAGER INSTANCE
# ===============================================================================

_db_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Get or create global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    
    return _db_manager

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection"""
    manager = await get_database_manager()
    async with manager.get_session() as session:
        yield session

async def get_async_engine() -> AsyncEngine:
    """Get async database engine"""
    manager = await get_database_manager()
    return manager._engine

async def create_all_tables() -> None:
    """Create all database tables"""
    manager = await get_database_manager()
    await manager.create_tables()

async def check_database_health() -> Dict[str, Any]:
    """Check database health"""
    manager = await get_database_manager()
    return await manager.health_check()

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "DatabaseManager",
    "DatabaseConfig",
    "ConnectionMetrics",
    "Base",
    "get_database_manager",
    "get_db_session",
    "get_async_engine",
    "create_all_tables",
    "check_database_health",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseTimeoutError",
    "DatabaseQueryError",
]
