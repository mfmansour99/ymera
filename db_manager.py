"""
YMERA Platform - Database Manager
Production-ready database operations with async support
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, DateTime, JSON, Integer, Text, Boolean, Float, select, and_, or_
from sqlalchemy.pool import NullPool
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

# Database Models
class AgentModel(Base):
    __tablename__ = 'agents'
    
    id = Column(String, primary_key=True)
    name = Column(String)
    type = Column(String)
    capabilities = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    status = Column(String)
    metrics = Column(JSON)

class TaskModel(Base):
    __tablename__ = 'tasks'
    
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(Text)
    status = Column(String)
    priority = Column(String)
    agent_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)

class AuditLogModel(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String)
    details = Column(JSON)
    user_id = Column(String, nullable=True)
    system_state = Column(JSON, nullable=True)

class AlertModel(Base):
    __tablename__ = 'alerts'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    severity = Column(String)
    source = Column(String)
    message = Column(Text)
    data = Column(JSON)
    acknowledged = Column(Boolean, default=False)
    resolved = Column(Boolean, default=False)

class MetricModel(Base):
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_type = Column(String)
    metric_name = Column(String)
    value = Column(Float)
    tags = Column(JSON)


class DatabaseManager:
    """Async database manager for YMERA platform"""
    
    def __init__(self, database_url: str = None, pool_size: int = 10, max_overflow: int = 20):
        """Initialize database manager"""
        self.database_url = database_url or "sqlite+aiosqlite:///ymera.db"
        
        # Create async engine
        if 'sqlite' in self.database_url:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                poolclass=NullPool
            )
        else:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True
            )
        
        # Create session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize database (create tables)"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.initialized = True
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def ping(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.async_session() as session:
                await session.execute(select(1))
            return True
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            return False
    
    # Agent Operations
    async def store_agent_registration(self, agent_id: str, agent_data: Dict[str, Any], metadata: Dict[str, Any]):
        """Store agent registration"""
        try:
            async with self.async_session() as session:
                agent = AgentModel(
                    id=agent_id,
                    name=agent_data.get('name', agent_id),
                    type=agent_data.get('type', 'unknown'),
                    capabilities=agent_data.get('capabilities', []),
                    status='active',
                    metrics=metadata
                )
                session.add(agent)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to store agent registration: {e}")
    
    async def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get all active agents"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(AgentModel).where(AgentModel.status == 'active')
                )
                agents = result.scalars().all()
                return [
                    {
                        'agent_id': agent.id,
                        'name': agent.name,
                        'type': agent.type,
                        'capabilities': agent.capabilities,
                        'last_active': agent.last_active,
                        'metrics': agent.metrics
                    }
                    for agent in agents
                ]
        except Exception as e:
            logger.error(f"Failed to get active agents: {e}")
            return []
    
    async def update_agent_heartbeat(self, agent_id: str, data: Dict[str, Any]):
        """Update agent heartbeat"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(AgentModel).where(AgentModel.id == agent_id)
                )
                agent = result.scalar_one_or_none()
                if agent:
                    agent.last_active = datetime.utcnow()
                    agent.metrics = data
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to update agent heartbeat: {e}")
    
    # Task Operations
    async def store_task(self, task_id: str, task_data: Dict[str, Any]):
        """Store task"""
        try:
            async with self.async_session() as session:
                task = TaskModel(
                    id=task_id,
                    name=task_data.get('name', ''),
                    description=task_data.get('description', ''),
                    status=task_data.get('status', 'pending'),
                    priority=task_data.get('priority', 'medium'),
                    agent_id=task_data.get('agent_id')
                )
                session.add(task)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to store task: {e}")
    
    async def update_task_status(self, task_id: str, status: str):
        """Update task status"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(TaskModel).where(TaskModel.id == task_id)
                )
                task = result.scalar_one_or_none()
                if task:
                    task.status = status
                    task.updated_at = datetime.utcnow()
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
    
    async def update_task_assignment(self, task_id: str, agent_id: str):
        """Update task assignment"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(TaskModel).where(TaskModel.id == task_id)
                )
                task = result.scalar_one_or_none()
                if task:
                    task.agent_id = agent_id
                    task.status = 'assigned'
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to update task assignment: {e}")
    
    async def update_task_completion(self, task_id: str, result: Any, execution_time: float, quality_score: float):
        """Update task completion"""
        try:
            async with self.async_session() as session:
                db_result = await session.execute(
                    select(TaskModel).where(TaskModel.id == task_id)
                )
                task = db_result.scalar_one_or_none()
                if task:
                    task.status = 'completed'
                    task.completed_at = datetime.utcnow()
                    task.result = result
                    task.execution_time = execution_time
                    task.quality_score = quality_score
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to update task completion: {e}")
    
    async def update_task_failure(self, task_id: str, error: str, error_type: str, retry_count: int):
        """Update task failure"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(TaskModel).where(TaskModel.id == task_id)
                )
                task = result.scalar_one_or_none()
                if task:
                    task.status = 'failed'
                    task.error = f"[{error_type}] {error}"
                    task.completed_at = datetime.utcnow()
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to update task failure: {e}")
    
    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get task status"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(TaskModel.status).where(TaskModel.id == task_id)
                )
                status = result.scalar_one_or_none()
                return status
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return None
    
    async def get_task_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get task history"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(TaskModel).order_by(TaskModel.created_at.desc()).limit(limit)
                )
                tasks = result.scalars().all()
                return [
                    {
                        'task_id': task.id,
                        'name': task.name,
                        'status': task.status,
                        'agent_id': task.agent_id,
                        'created_at': task.created_at,
                        'execution_time': task.execution_time,
                        'quality_score': task.quality_score
                    }
                    for task in tasks
                ]
        except Exception as e:
            logger.error(f"Failed to get task history: {e}")
            return []
    
    async def get_recent_task_outcomes(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent task outcomes"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(TaskModel).where(TaskModel.created_at >= cutoff)
                )
                tasks = result.scalars().all()
                return [
                    {
                        'task_id': task.id,
                        'status': task.status,
                        'execution_time': task.execution_time,
                        'quality_score': task.quality_score,
                        'error': task.error
                    }
                    for task in tasks
                ]
        except Exception as e:
            logger.error(f"Failed to get recent task outcomes: {e}")
            return []
    
    # Audit Operations
    async def store_audit_entry(self, entry: Dict[str, Any]):
        """Store audit entry"""
        try:
            async with self.async_session() as session:
                audit = AuditLogModel(
                    id=entry.get('id'),
                    action=entry.get('action'),
                    details=entry.get('details'),
                    system_state=entry.get('system_state')
                )
                session.add(audit)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to store audit entry: {e}")
    
    async def count_audit_entries_today(self) -> int:
        """Count audit entries created today"""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(AuditLogModel).where(AuditLogModel.timestamp >= today)
                )
                return len(result.scalars().all())
        except Exception as e:
            logger.error(f"Failed to count audit entries: {e}")
            return 0
    
    # Alert Operations
    async def store_alert(self, alert: Dict[str, Any]):
        """Store alert"""
        try:
            async with self.async_session() as session:
                alert_model = AlertModel(
                    id=alert.get('id'),
                    severity=alert.get('severity'),
                    source=alert.get('source'),
                    message=alert.get('message'),
                    data=alert.get('data')
                )
                session.add(alert_model)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    async def get_recent_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent errors"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(TaskModel).where(
                        and_(
                            TaskModel.status == 'failed',
                            TaskModel.completed_at >= cutoff
                        )
                    )
                )
                tasks = result.scalars().all()
                return [
                    {
                        'task_id': task.id,
                        'error': task.error,
                        'agent_id': task.agent_id,
                        'timestamp': task.completed_at
                    }
                    for task in tasks
                ]
        except Exception as e:
            logger.error(f"Failed to get recent errors: {e}")
            return []
    
    # Cleanup Operations
    async def count_tasks_older_than(self, days: int) -> int:
        """Count tasks older than specified days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(TaskModel).where(TaskModel.created_at < cutoff)
                )
                return len(result.scalars().all())
        except Exception as e:
            logger.error(f"Failed to count old tasks: {e}")
            return 0
    
    async def cleanup_old_data(self, days: int = 90):
        """Clean up old data"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        try:
            async with self.async_session() as session:
                # Delete old completed tasks
                await session.execute(
                    TaskModel.__table__.delete().where(
                        and_(
                            TaskModel.created_at < cutoff,
                            TaskModel.status.in_(['completed', 'failed', 'cancelled'])
                        )
                    )
                )
                await session.commit()
                logger.info(f"Cleaned up data older than {days} days")
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    async def close(self):
        """Close database connections"""
        await self.engine.dispose()
        logger.info("Database connections closed")


# Export
__all__ = ['DatabaseManager', 'Base', 'AgentModel', 'TaskModel', 'AuditLogModel', 'AlertModel']
