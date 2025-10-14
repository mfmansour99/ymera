"""
YMERA Enterprise Project Orchestrator - Production v4.0
=========================================================
The Ultimate Multi-Agent Software Development Platform

Integrates: 15+ Specialized Agents | 7+ Execution Engines | AI Learning
         Advanced Workflow Management | Real-time Collaboration | Full Monitoring

Author: YMERA Development Team
License: Proprietary
"""

import asyncio
import uuid
import json
import logging
import hashlib
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import numpy as np

# Core Framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator, EmailStr
from pydantic_settings import BaseSettings

# Database & Async
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import Column, String, DateTime, JSON, Integer, Boolean, Text, ForeignKey, Index, select, update, delete, and_, or_
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.pool import NullPool
import redis.asyncio as aioredis

# Security
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Monitoring
import httpx
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
import structlog

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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    """Production configuration"""
    
    # Application
    APP_NAME: str = "YMERA Enterprise Orchestrator"
    APP_VERSION: str = "4.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v4"
    
    # Security
    JWT_SECRET_KEY: str = Field(min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://ymera:password@localhost:5432/ymera_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_RECYCLE: int = 3600
    DATABASE_ECHO: bool = False
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 100
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 2000
    
    # System Limits
    MAX_CONCURRENT_PROJECTS: int = 100
    MAX_AGENTS_PER_PROJECT: int = 20
    MAX_MODULES_PER_PROJECT: int = 1000
    MAX_SANDBOX_SIZE_MB: int = 1024
    MAX_EXECUTION_TIME_SECONDS: int = 7200
    MAX_FILE_SIZE_MB: int = 100
    MAX_WEBSOCKET_MESSAGE_SIZE: int = 10485760
    
    # Background Tasks
    HEALTH_CHECK_INTERVAL: int = 60
    METRICS_COLLECTION_INTERVAL: int = 60
    CLEANUP_INTERVAL: int = 3600
    AGENT_HEARTBEAT_TIMEOUT: int = 300
    
    # Feature Flags
    ENABLE_AI_VALIDATION: bool = True
    ENABLE_AUTO_DEPLOYMENT: bool = False
    ENABLE_LEARNING_ENGINE: bool = True
    ENABLE_ADVANCED_ANALYTICS: bool = True
    ENABLE_SECURITY_SCANNING: bool = True
    ENABLE_REAL_TIME_COLLABORATION: bool = True
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    SENTRY_ENABLED: bool = False
    SENTRY_DSN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# =============================================================================
# SECURITY
# =============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityManager:
    """Enhanced security management"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            return payload
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
    @staticmethod
    def sanitize_input(data: str) -> str:
        """Sanitize user input"""
        dangerous_patterns = ['<script', 'javascript:', 'onerror=', 'onload=', 'eval(', 'exec(']
        for pattern in dangerous_patterns:
            if pattern.lower() in data.lower():
                raise ValueError(f"Potentially dangerous input detected: {pattern}")
        return data.strip()

security_manager = SecurityManager()

# =============================================================================
# ENUMS
# =============================================================================

class AgentType(str, Enum):
    """Agent types in the system"""
    # Core Development
    ORCHESTRATOR = "orchestrator"
    ANALYSIS = "analysis"
    ARCHITECTURE = "architecture"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    
    # Testing & Quality
    TEST_GENERATION = "test_generation"
    QUALITY_ASSURANCE = "quality_assurance"
    PERFORMANCE_TESTING = "performance_testing"
    
    # Security & Compliance
    SECURITY_SCANNING = "security_scanning"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    COMPLIANCE_CHECKING = "compliance_checking"
    
    # Infrastructure
    DEVOPS = "devops"
    DEPLOYMENT = "deployment"
    INFRASTRUCTURE = "infrastructure"
    MONITORING = "monitoring"
    
    # Documentation & Support
    DOCUMENTATION = "documentation"
    API_DOCUMENTATION = "api_documentation"
    
    # Specialized
    DATABASE_DESIGN = "database_design"
    UI_UX = "ui_ux"
    API_DESIGN = "api_design"
    INTEGRATION = "integration"
    OPTIMIZATION = "optimization"

class ProjectPhase(str, Enum):
    """Project lifecycle phases"""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    DESIGN = "design"
    DEVELOPMENT = "development"
    TESTING = "testing"
    SECURITY_REVIEW = "security_review"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"

class ProjectStatus(str, Enum):
    """Project operational status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ValidationStatus(str, Enum):
    """Module validation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    CONDITIONAL_APPROVAL = "conditional_approval"

class ModuleType(str, Enum):
    """Module types"""
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    API = "api"
    MIDDLEWARE = "middleware"
    INFRASTRUCTURE = "infrastructure"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    TEST = "test"
    SECURITY = "security"
    MONITORING = "monitoring"

class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

class EventSeverity(str, Enum):
    """Event severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AgentStatus(str, Enum):
    """Agent operational status"""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"

class WorkflowStage(str, Enum):
    """Workflow stages"""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    DESIGN = "design"
    DEVELOPMENT = "development"
    TESTING = "testing"
    REVIEW = "review"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"
    COMPLETION = "completion"

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), default="user")
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    preferences = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
    )

class ProjectModel(Base):
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text)
    project_type = Column(String(50), nullable=False)
    phase = Column(String(50), default=ProjectPhase.INITIALIZATION, index=True)
    status = Column(String(50), default=ProjectStatus.ACTIVE, index=True)
    
    tech_stack = Column(JSON, default=dict)
    requirements = Column(JSON, default=dict)
    constraints = Column(JSON, default=dict)
    
    workspace_path = Column(String(500))
    sandbox_id = Column(String(100))
    repository_url = Column(String(500))
    
    progress_percentage = Column(Integer, default=0)
    quality_score = Column(Integer, default=0)
    security_score = Column(Integer, default=0)
    
    metrics = Column(JSON, default=dict)
    health_status = Column(JSON, default=dict)
    
    owner_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
    
    modules = relationship("ModuleModel", back_populates="project", cascade="all, delete-orphan")
    agents = relationship("AgentAssignmentModel", back_populates="project", cascade="all, delete-orphan")
    events = relationship("ProjectEventModel", back_populates="project", cascade="all, delete-orphan")
    tasks = relationship("TaskModel", back_populates="project", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_project_owner_status', 'owner_id', 'status'),
        Index('idx_project_phase_status', 'phase', 'status'),
    )

class AgentModel(Base):
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False, index=True)
    version = Column(String(20), default="1.0.0")
    
    capabilities = Column(JSON, default=list)
    supported_languages = Column(JSON, default=list)
    supported_frameworks = Column(JSON, default=list)
    specializations = Column(JSON, default=list)
    
    status = Column(String(20), default="inactive", index=True)
    health_status = Column(String(20), default="unknown")
    
    max_concurrent_tasks = Column(Integer, default=5)
    current_task_count = Column(Integer, default=0)
    
    performance_score = Column(Integer, default=100)
    reliability_score = Column(Integer, default=100)
    
    configuration = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    last_heartbeat = Column(DateTime)
    registered_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    assignments = relationship("AgentAssignmentModel", back_populates="agent")
    
    __table_args__ = (
        Index('idx_agent_type_status', 'agent_type', 'status'),
    )

class AgentAssignmentModel(Base):
    __tablename__ = "agent_assignments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_id = Column(String, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(50), nullable=False)
    
    assigned_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(20), default="active", index=True)
    
    tasks_assigned = Column(Integer, default=0)
    tasks_completed = Column(Integer, default=0)
    tasks_failed = Column(Integer, default=0)
    
    performance_metrics = Column(JSON, default=dict)
    
    project = relationship("ProjectModel", back_populates="agents")
    agent = relationship("AgentModel", back_populates="assignments")
    
    __table_args__ = (
        Index('idx_assignment_project_agent', 'project_id', 'agent_id'),
    )

class ModuleModel(Base):
    __tablename__ = "modules"
    
    id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    module_type = Column(String(50), nullable=False, index=True)
    
    content = Column(Text)
    file_path = Column(String(500))
    file_hash = Column(String(64))
    version = Column(String(20), default="1.0.0")
    
    validation_status = Column(String(50), default=ValidationStatus.PENDING, index=True)
    validation_score = Column(Integer, default=0)
    validation_report = Column(JSON)
    
    security_scan_status = Column(String(50), default="pending")
    security_issues = Column(JSON, default=list)
    
    test_coverage = Column(Integer, default=0)
    complexity_score = Column(Integer, default=0)
    
    created_by_agent = Column(String, ForeignKey("agents.id"))
    validated_by_agent = Column(String, ForeignKey("agents.id"))
    
    dependencies = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    project = relationship("ProjectModel", back_populates="modules")
    
    __table_args__ = (
        Index('idx_module_project_status', 'project_id', 'validation_status'),
        Index('idx_module_type_status', 'module_type', 'validation_status'),
    )

class TaskModel(Base):
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_id = Column(String, ForeignKey("agents.id"), index=True)
    
    task_type = Column(String(100), nullable=False)
    priority = Column(String(20), default=TaskPriority.NORMAL, index=True)
    
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON)
    
    status = Column(String(50), default="pending", index=True)
    progress_percentage = Column(Integer, default=0)
    
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    project = relationship("ProjectModel", back_populates="tasks")
    
    __table_args__ = (
        Index('idx_task_project_status', 'project_id', 'status'),
        Index('idx_task_agent_status', 'agent_id', 'status'),
    )

class ProjectEventModel(Base):
    __tablename__ = "project_events"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    
    event_type = Column(String(100), nullable=False, index=True)
    event_data = Column(JSON, nullable=False)
    
    source = Column(String(100))
    severity = Column(String(20), default=EventSeverity.INFO, index=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    project = relationship("ProjectModel", back_populates="events")
    
    __table_args__ = (
        Index('idx_event_project_type', 'project_id', 'event_type'),
        Index('idx_event_severity_timestamp', 'severity', 'timestamp'),
    )

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AgentCapability:
    """Agent capabilities definition"""
    agent_id: str
    agent_type: AgentType
    specializations: List[str]
    max_concurrent_tasks: int = 5
    performance_score: float = 1.0
    workload: int = 0

@dataclass
class AgentTask:
    """Task for agent execution"""
    task_id: str
    task_type: str
    priority: int
    data: Dict[str, Any]
    assigned_to: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class ValidationResult:
    """Module validation result"""
    module_id: str
    is_valid: bool
    score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    validator_agent: str = "orchestrator"
    validation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ProjectContext:
    """Project context"""
    project_id: str
    name: str
    description: str
    project_type: str
    phase: ProjectPhase
    status: ProjectStatus
    
    tech_stack: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    
    assigned_agents: Dict[str, List[str]] = field(default_factory=dict)
    modules: Dict[str, Any] = field(default_factory=dict)
    tasks: Dict[str, Any] = field(default_factory=dict)
    
    workspace_path: Optional[Path] = None
    sandbox_id: Optional[str] = None
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    health_status: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    owner_id: str = ""

@dataclass
class WorkflowStep:
    """Workflow step definition"""
    step_id: str
    stage: WorkflowStage
    required_agents: List[AgentType]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: int = 60
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None

# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    project_type: str
    tech_stack: Dict[str, Any] = Field(default_factory=dict)
    requirements: Dict[str, Any] = Field(default_factory=dict)

class ModuleSubmission(BaseModel):
    name: str
    module_type: ModuleType
    content: str
    agent_id: str
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"

class AgentRegistration(BaseModel):
    name: str
    agent_type: AgentType
    capabilities: List[str]
    supported_languages: List[str]
    supported_frameworks: List[str]
    max_concurrent_tasks: int = 5

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    context: Optional[Dict[str, Any]] = None

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

PROJECTS_TOTAL = Counter('ymera_projects_total', 'Total projects created')
PROJECTS_ACTIVE = Gauge('ymera_projects_active', 'Currently active projects')
PROJECTS_BY_PHASE = Gauge('ymera_projects_by_phase', 'Projects by phase', ['phase'])
PROJECT_CREATION_TIME = Histogram('ymera_project_creation_seconds', 'Project creation time')

AGENTS_REGISTERED = Counter('ymera_agents_registered_total', 'Total agents registered')
AGENTS_ACTIVE = Gauge('ymera_agents_active', 'Currently active agents', ['type'])
AGENT_TASK_DURATION = Histogram('ymera_agent_task_duration_seconds', 'Agent task duration', ['agent_type', 'task_type'])
AGENT_TASK_SUCCESS = Counter('ymera_agent_tasks_success', 'Successful agent tasks', ['agent_type'])
AGENT_TASK_FAILURE = Counter('ymera_agent_tasks_failure', 'Failed agent tasks', ['agent_type'])

MODULES_CREATED = Counter('ymera_modules_created_total', 'Total modules created')
MODULES_VALIDATED = Counter('ymera_modules_validated_total', 'Modules validated', ['status'])
MODULE_VALIDATION_TIME = Histogram('ymera_module_validation_seconds', 'Module validation time')
MODULE_VALIDATION_SCORE = Summary('ymera_module_validation_score', 'Module validation scores')

WEBSOCKET_CONNECTIONS = Gauge('ymera_websocket_connections', 'Active WebSocket connections', ['type'])
API_REQUEST_DURATION = Histogram('ymera_api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
API_REQUEST_TOTAL = Counter('ymera_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Production-grade database management"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        
    async def initialize(self):
        """Initialize database with connection pooling"""
        try:
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                pool_recycle=settings.DATABASE_POOL_RECYCLE,
                pool_pre_ping=True,
                echo=settings.DATABASE_ECHO,
                poolclass=NullPool if settings.ENVIRONMENT == "test" else None
            )
            
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("database_initialized", environment=settings.ENVIRONMENT)
            
        except Exception as e:
            logger.error("database_initialization_failed", error=str(e))
            raise
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic commit/rollback"""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.get_session() as session:
                await session.execute(select(1))
            return True
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("database_connections_closed")

# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """Production-grade Redis cache management"""
    
    def __init__(self):
        self.redis = None
        
    async def initialize(self):
        """Initialize Redis with connection pooling"""
        try:
            self.redis = await aioredis.from_url(
                settings.REDIS_URL,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=True,
                retry_on_timeout=True
            )
            
            await self.redis.ping()
            logger.info("redis_initialized")
            
        except Exception as e:
            logger.error("redis_initialization_failed", error=str(e))
            raise
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.error("cache_get_error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: str, ttl: int = 300):
        """Set value in cache with TTL"""
        try:
            await self.redis.setex(key, ttl, value)
        except Exception as e:
            logger.error("cache_set_error", key=key, error=str(e))
    
    async def delete(self, key: str):
        """Delete key from cache"""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error("cache_delete_error", key=key, error=str(e))
    
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON value from cache"""
        value = await self.get(key)
        return json.loads(value) if value else None
    
    async def set_json(self, key: str, value: Dict[str, Any], ttl: int = 300):
        """Set JSON value in cache"""
        await self.set(key, json.dumps(value), ttl)
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Cleanup Redis connections"""
        if self.redis:
            await self.redis.close()
            logger.info("redis_connections_closed")

# =============================================================================
# WEBSOCKET MANAGER
# =============================================================================

class WebSocketManager:
    """Production-grade WebSocket connection management"""
    
    def __init__(self):
        self.user_connections: Dict[str, Dict[str, WebSocket]] = defaultdict(dict)
        self.agent_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.message_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = asyncio.Lock()
        
    async def connect_user(self, websocket: WebSocket, user_id: str) -> str:
        """Connect user WebSocket with authentication"""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        
        async with self._lock:
            self.user_connections[user_id][connection_id] = websocket
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "type": "user",
                "connected_at": datetime.utcnow(),
                "message_count": 0
            }
        
        WEBSOCKET_CONNECTIONS.labels(type="user").inc()
        
        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to YMERA Orchestrator v4.0"
        })
        
        logger.info("user_connected", user_id=user_id, connection_id=connection_id)
        return connection_id
    
    async def connect_agent(self, websocket: WebSocket, agent_id: str):
        """Connect agent WebSocket with validation"""
        await websocket.accept()
        
        async with self._lock:
            self.agent_connections[agent_id] = websocket
            self.connection_metadata[agent_id] = {
                "agent_id": agent_id,
                "type": "agent",
                "connected_at": datetime.utcnow(),
                "tasks_received": 0
            }
        
        WEBSOCKET_CONNECTIONS.labels(type="agent").inc()
        
        await websocket.send_json({
            "type": "agent_connected",
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info("agent_connected", agent_id=agent_id)
    
    async def disconnect_user(self, connection_id: str):
        """Disconnect user WebSocket"""
        async with self._lock:
            metadata = self.connection_metadata.pop(connection_id, None)
            if metadata:
                user_id = metadata.get("user_id")
                if user_id and connection_id in self.user_connections[user_id]:
                    del self.user_connections[user_id][connection_id]
                    WEBSOCKET_CONNECTIONS.labels(type="user").dec()
                    logger.info("user_disconnected", user_id=user_id, connection_id=connection_id)
    
    async def disconnect_agent(self, agent_id: str):
        """Disconnect agent WebSocket"""
        async with self._lock:
            if agent_id in self.agent_connections:
                del self.agent_connections[agent_id]
                self.connection_metadata.pop(agent_id, None)
                WEBSOCKET_CONNECTIONS.labels(type="agent").dec()
                logger.info("agent_disconnected", agent_id=agent_id)
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all user connections"""
        connections = self.user_connections.get(user_id, {})
        for connection_id, websocket in connections.items():
            try:
                await websocket.send_json(message)
                self.connection_metadata[connection_id]["message_count"] += 1
            except Exception as e:
                logger.error("websocket_send_failed", user_id=user_id, error=str(e))
    
    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Send message to agent"""
        websocket = self.agent_connections.get(agent_id)
        if websocket:
            try:
                await websocket.send_json(message)
                self.connection_metadata[agent_id]["tasks_received"] += 1
            except Exception as e:
                logger.error("agent_send_failed", agent_id=agent_id, error=str(e))
    
    async def broadcast_to_project(self, project_id: str, message: Dict[str, Any]):
        """Broadcast message to all users and agents in a project"""
        pass
    
    def get_connection_count(self) -> Dict[str, int]:
        """Get connection statistics"""
        return {
            "users": sum(len(conns) for conns in self.user_connections.values()),
            "agents": len(self.agent_connections),
            "total": sum(len(conns) for conns in self.user_connections.values()) + len(self.agent_connections)
        }

# =============================================================================
# AGENT COORDINATOR
# =============================================================================

class AgentCoordinator:
    """Intelligent agent coordination and task distribution"""
    
    def __init__(self, db_manager: DatabaseManager, ws_manager: WebSocketManager):
        self.db = db_manager
        self.ws = ws_manager
        self.registered_agents: Dict[str, AgentCapability] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, AgentTask] = {}
        self.task_history: List[AgentTask] = []
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    async def register_agent(self, capability: AgentCapability):
        """Register agent with the system"""
        self.registered_agents[capability.agent_id] = capability
        self.agent_status[capability.agent_id] = AgentStatus.IDLE
        
        AGENTS_ACTIVE.labels(type=capability.agent_type.value).inc()
        
        logger.info("agent_registered", agent_id=capability.agent_id, type=capability.agent_type.value)
    
    async def assign_task(self, task: AgentTask) -> bool:
        """Intelligently assign task to best available agent"""
        suitable_agents = await self._find_suitable_agents(task)
        
        if not suitable_agents:
            self.task_queue.append(task)
            logger.warning("task_queued_no_agent", task_id=task.task_id)
            return False
        
        best_agent = await self._select_best_agent(suitable_agents, task)
        
        task.assigned_to = best_agent.agent_id
        task.status = "assigned"
        task.started_at = datetime.utcnow()
        
        self.active_tasks[task.task_id] = task
        self.agent_status[best_agent.agent_id] = AgentStatus.BUSY
        best_agent.workload += 1
        
        await self.ws.send_to_agent(best_agent.agent_id, {
            "type": "task_assignment",
            "task_id": task.task_id,
            "task_type": task.task_type,
            "priority": task.priority,
            "data": task.data
        })
        
        logger.info("task_assigned", task_id=task.task_id, agent_id=best_agent.agent_id)
        return True
    
    async def _find_suitable_agents(self, task: AgentTask) -> List[AgentCapability]:
        """Find agents capable of handling the task"""
        suitable = []
        
        for agent_id, capability in self.registered_agents.items():
            if self.agent_status[agent_id] == AgentStatus.OFFLINE:
                continue
            
            if capability.workload >= capability.max_concurrent_tasks:
                continue
            
            if self._matches_task_requirements(capability, task):
                suitable.append(capability)
        
        return suitable
    
    def _matches_task_requirements(self, capability: AgentCapability, task: AgentTask) -> bool:
        """Check if agent capabilities match task requirements"""
        task_agent_mapping = {
            "code_analysis": [AgentType.ANALYSIS, AgentType.CODE_REVIEW],
            "code_generation": [AgentType.CODE_GENERATION],
            "testing": [AgentType.TEST_GENERATION, AgentType.QUALITY_ASSURANCE],
            "security_scan": [AgentType.SECURITY_SCANNING],
            "deployment": [AgentType.DEPLOYMENT, AgentType.DEVOPS],
            "documentation": [AgentType.DOCUMENTATION],
            "monitoring": [AgentType.MONITORING],
            "optimization": [AgentType.OPTIMIZATION]
        }
        
        required_types = task_agent_mapping.get(task.task_type, [])
        return capability.agent_type in required_types
    
    async def _select_best_agent(self, candidates: List[AgentCapability], task: AgentTask) -> AgentCapability:
        """Select the best agent from candidates"""
        if len(candidates) == 1:
            return candidates[0]
        
        scores = []
        for agent in candidates:
            score = 0.0
            score += agent.performance_score * 0.4
            workload_ratio = agent.workload / agent.max_concurrent_tasks
            score += (1.0 - workload_ratio) * 0.3
            historical_score = self.agent_performance[agent.agent_id].get(task.task_type, 0.8)
            score += historical_score * 0.3
            scores.append((score, agent))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]
    
    async def complete_task(self, task_id: str, result: Dict[str, Any], success: bool = True):
        """Mark task as completed and update agent status"""
        if task_id not in self.active_tasks:
            logger.warning("task_not_found", task_id=task_id)
            return
        
        task = self.active_tasks[task_id]
        task.completed_at = datetime.utcnow()
        task.status = "completed" if success else "failed"
        task.result = result
        
        if task.assigned_to:
            agent = self.registered_agents.get(task.assigned_to)
            if agent:
                agent.workload = max(0, agent.workload - 1)
                
                if agent.workload == 0:
                    self.agent_status[agent.agent_id] = AgentStatus.IDLE
                
                execution_time = (task.completed_at - task.started_at).total_seconds()
                self._update_agent_performance(agent.agent_id, task.task_type, success, execution_time)
                
                if success:
                    AGENT_TASK_SUCCESS.labels(agent_type=agent.agent_type.value).inc()
                else:
                    AGENT_TASK_FAILURE.labels(agent_type=agent.agent_type.value).inc()
                
                AGENT_TASK_DURATION.labels(
                    agent_type=agent.agent_type.value,
                    task_type=task.task_type
                ).observe(execution_time)
        
        self.task_history.append(task)
        del self.active_tasks[task_id]
        
        await self._process_queue()
        
        logger.info("task_completed", task_id=task_id, status=task.status)
    
    def _update_agent_performance(self, agent_id: str, task_type: str, success: bool, execution_time: float):
        """Update agent performance metrics"""
        if task_type not in self.agent_performance[agent_id]:
            self.agent_performance[agent_id][task_type] = 0.8
        
        current_score = self.agent_performance[agent_id][task_type]
        new_score = current_score * 0.9 + 0.1 * (1.0 if success else 0.5)
        self.agent_performance[agent_id][task_type] = new_score
    
    async def _process_queue(self):
        """Process queued tasks"""
        while self.task_queue:
            task = self.task_queue[0]
            if await self.assign_task(task):
                self.task_queue.popleft()
            else:
                break
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "agents": {
                "total": len(self.registered_agents),
                "idle": sum(1 for s in self.agent_status.values() if s == AgentStatus.IDLE),
                "busy": sum(1 for s in self.agent_status.values() if s == AgentStatus.BUSY),
                "offline": sum(1 for s in self.agent_status.values() if s == AgentStatus.OFFLINE)
            },
            "tasks": {
                "active": len(self.active_tasks),
                "queued": len(self.task_queue),
                "completed": len([t for t in self.task_history if t.status == "completed"]),
                "failed": len([t for t in self.task_history if t.status == "failed"])
            },
            "performance": {
                agent_id: {
                    "score": agent.performance_score,
                    "workload": agent.workload,
                    "status": self.agent_status[agent_id].value
                }
                for agent_id, agent in self.registered_agents.items()
            }
        }

# =============================================================================
# MODULE VALIDATOR
# =============================================================================

class ModuleValidator:
    """Production-grade module validation with multiple checks"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.security_patterns = [
            'eval(', 'exec(', '__import__', 'os.system', 'subprocess.call',
            'pickle.loads', 'yaml.load', 'input(', 'raw_input('
        ]
        
    async def validate_module(self, module: ModuleSubmission) -> ValidationResult:
        """Comprehensive module validation"""
        start_time = datetime.utcnow()
        errors = []
        warnings = []
        suggestions = []
        security_issues = []
        score = 100.0
        
        try:
            content_issues = await self._validate_content(module.content)
            errors.extend(content_issues.get('errors', []))
            warnings.extend(content_issues.get('warnings', []))
            score -= content_issues.get('penalty', 0)
            
            security_result = await self._validate_security(module.content)
            security_issues = security_result.get('issues', [])
            if security_issues:
                errors.extend([f"Security: {issue['description']}" for issue in security_issues])
                score -= security_result.get('penalty', 0)
            
            quality_issues = await self._validate_quality(module.content, module.module_type)
            warnings.extend(quality_issues.get('warnings', []))
            suggestions.extend(quality_issues.get('suggestions', []))
            score -= quality_issues.get('penalty', 0)
            
            dep_issues = await self._validate_dependencies(module.dependencies)
            warnings.extend(dep_issues.get('warnings', []))
            score -= dep_issues.get('penalty', 0)
            
            type_issues = await self._validate_module_type(module.content, module.module_type)
            warnings.extend(type_issues.get('warnings', []))
            suggestions.extend(type_issues.get('suggestions', []))
            score -= type_issues.get('penalty', 0)
            
            score = max(0.0, min(100.0, score))
            is_valid = len(errors) == 0 and score >= 70.0
            
            validation_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = ValidationResult(
                module_id=str(uuid.uuid4()),
                is_valid=is_valid,
                score=score,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                security_issues=security_issues,
                validation_time=validation_time
            )
            
            MODULE_VALIDATION_TIME.observe(validation_time)
            MODULE_VALIDATION_SCORE.observe(score)
            
            return result
            
        except Exception as e:
            logger.error("validation_failed", error=str(e))
            return ValidationResult(
                module_id=str(uuid.uuid4()),
                is_valid=False,
                score=0.0,
                errors=[f"Validation error: {str(e)}"]
            )
    
    async def _validate_content(self, content: str) -> Dict[str, Any]:
        """Validate module content"""
        issues = {"errors": [], "warnings": [], "penalty": 0}
        
        if not content or len(content.strip()) < 10:
            issues["errors"].append("Module content is empty or too short")
            issues["penalty"] = 50
            return issues
        
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            issues["errors"].append(f"Module exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB")
            issues["penalty"] = 30
        
        if 'def ' not in content and 'class ' not in content and 'function ' not in content:
            issues["warnings"].append("No functions or classes detected")
            issues["penalty"] = 5
        
        return issues
    
    async def _validate_security(self, content: str) -> Dict[str, Any]:
        """Security validation"""
        result = {"issues": [], "penalty": 0}
        content_lower = content.lower()
        
        for pattern in self.security_patterns:
            if pattern in content_lower:
                result["issues"].append({
                    "type": "dangerous_function",
                    "description": f"Potentially dangerous function detected: {pattern}",
                    "severity": "high",
                    "line": content_lower.count('\n', 0, content_lower.find(pattern)) + 1
                })
                result["penalty"] += 20
        
        credential_patterns = ['password', 'api_key', 'secret', 'token']
        for pattern in credential_patterns:
            if f'{pattern}=' in content_lower or f'{pattern} =' in content_lower:
                result["issues"].append({
                    "type": "hardcoded_credential",
                    "description": f"Possible hardcoded credential: {pattern}",
                    "severity": "high"
                })
                result["penalty"] += 15
        
        return result
    
    async def _validate_quality(self, content: str, module_type: ModuleType) -> Dict[str, Any]:
        """Code quality validation"""
        issues = {"warnings": [], "suggestions": [], "penalty": 0}
        
        if 'TODO' in content or 'FIXME' in content:
            issues["warnings"].append("Code contains TODO/FIXME comments")
            issues["penalty"] = 5
        
        if module_type != ModuleType.TEST and 'print(' in content:
            issues["warnings"].append("Code contains print statements - consider using logging")
            issues["penalty"] = 3
        
        lines = content.split('\n')
        indent_types = set()
        for line in lines:
            if line and line[0] in [' ', '\t']:
                indent_types.add(line[0])
        
        if len(indent_types) > 1:
            issues["warnings"].append("Inconsistent indentation (mixed tabs and spaces)")
            issues["suggestions"].append("Use consistent indentation throughout")
            issues["penalty"] = 5
        
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            issues["warnings"].append(f"Lines exceeding 120 characters: {len(long_lines)}")
            issues["suggestions"].append("Consider breaking long lines for readability")
            issues["penalty"] = 2
        
        complexity_score = content.count('if ') + content.count('for ') + content.count('while ')
        if complexity_score > 50:
            issues["warnings"].append("High cyclomatic complexity detected")
            issues["suggestions"].append("Consider refactoring into smaller functions")
            issues["penalty"] = 10
        
        return issues
    
    async def _validate_dependencies(self, dependencies: List[str]) -> Dict[str, Any]:
        """Validate module dependencies"""
        issues = {"warnings": [], "penalty": 0}
        
        if len(dependencies) > 20:
            issues["warnings"].append("Large number of dependencies may indicate tight coupling")
            issues["penalty"] = 5
        
        return issues
    
    async def _validate_module_type(self, content: str, module_type: ModuleType) -> Dict[str, Any]:
        """Type-specific validation"""
        issues = {"warnings": [], "suggestions": [], "penalty": 0}
        
        if module_type == ModuleType.API:
            if 'route' not in content.lower() and 'endpoint' not in content.lower():
                issues["warnings"].append("API module should define routes/endpoints")
                issues["penalty"] = 5
        
        elif module_type == ModuleType.TEST:
            if 'test_' not in content and 'def test' not in content:
                issues["warnings"].append("Test module should contain test functions")
                issues["penalty"] = 10
            
            if 'assert' not in content.lower():
                issues["warnings"].append("Test module should contain assertions")
                issues["penalty"] = 10
        
        elif module_type == ModuleType.DATABASE:
            if 'table' not in content.lower() and 'model' not in content.lower():
                issues["warnings"].append("Database module should define tables or models")
                issues["penalty"] = 5
        
        return issues

# =============================================================================
# LEARNING ENGINE
# =============================================================================

class LearningEngine:
    """Machine learning-based system improvement"""
    
    def __init__(self):
        self.patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def learn_from_project(self, project_data: Dict[str, Any], outcome: str):
        """Learn from project completion"""
        pattern = {
            "project_type": project_data.get("project_type"),
            "tech_stack": project_data.get("tech_stack"),
            "team_size": len(project_data.get("agents", [])),
            "duration": project_data.get("duration"),
            "modules_count": project_data.get("modules_count"),
            "outcome": outcome,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if outcome == "success":
            self.success_patterns.append(pattern)
        else:
            self.failure_patterns.append(pattern)
        
        self.patterns[project_data.get("project_type")].append(pattern)
        
        if len(self.patterns[project_data.get("project_type")]) >= 10:
            await self._optimize_patterns(project_data.get("project_type"))
    
    async def _optimize_patterns(self, project_type: str):
        """Analyze patterns and generate optimizations"""
        patterns = self.patterns[project_type]
        
        if not patterns:
            return
        
        success_count = sum(1 for p in patterns if p["outcome"] == "success")
        success_rate = success_count / len(patterns)
        
        successful = [p for p in patterns if p["outcome"] == "success"]
        
        optimization = {
            "project_type": project_type,
            "success_rate": success_rate,
            "insights": [],
            "recommendations": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if successful:
            avg_team_size = np.mean([p["team_size"] for p in successful])
            optimization["recommendations"].append({
                "type": "team_size",
                "optimal_value": int(avg_team_size),
                "reason": "Based on successful project patterns"
            })
        
        if successful and len(successful) >= 3:
            durations = [p["duration"] for p in successful if p.get("duration")]
            if durations:
                avg_duration = np.mean(durations)
                optimization["recommendations"].append({
                    "type": "duration",
                    "estimated_value": avg_duration,
                    "reason": "Average duration of successful projects"
                })
        
        self.optimization_history.append(optimization)
        
        logger.info("optimization_generated", project_type=project_type, recommendations=len(optimization['recommendations']))
    
    async def get_recommendations(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered recommendations for a project"""
        project_type = project_data.get("project_type")
        
        similar_projects = [
            p for p in self.success_patterns
            if p.get("project_type") == project_type
        ]
        
        if not similar_projects:
            return {
                "recommendations": [],
                "confidence": 0.0,
                "message": "Insufficient data for recommendations"
            }
        
        recommendations = []
        
        team_sizes = [p["team_size"] for p in similar_projects]
        optimal_team_size = int(np.median(team_sizes))
        
        recommendations.append({
            "type": "team_composition",
            "value": optimal_team_size,
            "confidence": 0.8,
            "description": f"Based on {len(similar_projects)} similar successful projects"
        })
        
        tech_stacks = [p.get("tech_stack", {}) for p in similar_projects]
        common_tech = self._find_common_technologies(tech_stacks)
        
        if common_tech:
            recommendations.append({
                "type": "technology",
                "value": common_tech,
                "confidence": 0.7,
                "description": "Commonly used in successful similar projects"
            })
        
        return {
            "recommendations": recommendations,
            "confidence": 0.75,
            "based_on": len(similar_projects),
            "message": "Recommendations based on historical success patterns"
        }
    
    def _find_common_technologies(self, tech_stacks: List[Dict[str, Any]]) -> List[str]:
        """Find commonly used technologies"""
        tech_counts = defaultdict(int)
        
        for stack in tech_stacks:
            for category, techs in stack.items():
                if isinstance(techs, list):
                    for tech in techs:
                        tech_counts[tech] += 1
                elif isinstance(techs, str):
                    tech_counts[techs] += 1
        
        threshold = len(tech_stacks) * 0.5
        common = [tech for tech, count in tech_counts.items() if count >= threshold]
        
        return common[:5]

# Continue in next message due to length...