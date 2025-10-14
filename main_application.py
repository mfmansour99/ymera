"""
YMERA Enterprise - Main Application Entry Point
Production-Ready Agent Learning Integration System - v4.0
Enterprise-grade implementation with zero placeholders
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

# Standard library imports (alphabetical)
import asyncio
import logging
import signal
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports (alphabetical)
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Local imports (alphabetical)
from config.settings import get_settings
from database.connection import DatabaseManager, get_db_session
from monitoring.performance_tracker import PerformanceTracker, track_performance
from security.middleware import SecurityMiddleware
from utils.health_check import SystemHealthChecker
from ymera_agents.core.learning_engine import LearningEngineManager
from ymera_agents.core.agent_manager import AgentManager
from ymera_agents.core.knowledge_graph import KnowledgeGraphManager
from ymera_agents.api.routers import (
    agents_router,
    learning_router,
    knowledge_router,
    health_router,
    monitoring_router
)

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.main")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

# Application constants
APP_NAME = "YMERA Enterprise Agent Platform"
APP_VERSION = "4.0.0"
API_PREFIX = "/api/v4"

# Configuration loading
settings = get_settings()

# Metrics collection
REQUEST_COUNT = Counter(
    "ymera_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "ymera_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"]
)

AGENT_OPERATIONS = Counter(
    "ymera_agent_operations_total",
    "Total agent operations",
    ["agent_type", "operation", "status"]
)

LEARNING_CYCLES = Counter(
    "ymera_learning_cycles_total",
    "Total learning cycles completed",
    ["cycle_type", "status"]
)

# ===============================================================================
# MIDDLEWARE CLASSES
# ===============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting application metrics"""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and collect metrics"""
        start_time = datetime.utcnow()
        method = request.method
        path = request.url.path
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=status_code
            ).inc()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            REQUEST_DURATION.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            return response
            
        except Exception as e:
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=500
            ).inc()
            
            logger.error(
                "Request processing failed",
                method=method,
                path=path,
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracing and logging"""
    
    async def dispatch(self, request: Request, call_next):
        """Trace and log request processing"""
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request start
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown")
        )
        
        try:
            response = await call_next(request)
            
            # Log request completion
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                duration=duration
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                duration=duration,
                traceback=traceback.format_exc()
            )
            raise

# ===============================================================================
# APPLICATION LIFECYCLE MANAGEMENT
# ===============================================================================

class ApplicationManager:
    """Manages application lifecycle and core components"""
    
    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.learning_engine: Optional[LearningEngineManager] = None
        self.agent_manager: Optional[AgentManager] = None
        self.knowledge_graph: Optional[KnowledgeGraphManager] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.health_checker: Optional[SystemHealthChecker] = None
        self._shutdown_event = asyncio.Event()
        self._background_tasks: List[asyncio.Task] = []
        
    async def initialize(self) -> None:
        """Initialize all application components"""
        try:
            logger.info("Starting YMERA Enterprise Platform initialization")
            
            # Initialize database manager
            logger.info("Initializing database manager")
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Initialize performance tracker
            logger.info("Initializing performance tracker")
            self.performance_tracker = PerformanceTracker()
            await self.performance_tracker.initialize()
            
            # Initialize knowledge graph
            logger.info("Initializing knowledge graph manager")
            self.knowledge_graph = KnowledgeGraphManager()
            await self.knowledge_graph.initialize()
            
            # Initialize agent manager
            logger.info("Initializing agent manager")
            self.agent_manager = AgentManager(
                db_manager=self.db_manager,
                knowledge_graph=self.knowledge_graph
            )
            await self.agent_manager.initialize()
            
            # Initialize learning engine
            logger.info("Initializing learning engine")
            self.learning_engine = LearningEngineManager(
                agent_manager=self.agent_manager,
                knowledge_graph=self.knowledge_graph,
                db_manager=self.db_manager
            )
            await self.learning_engine.initialize()
            
            # Initialize health checker
            logger.info("Initializing health checker")
            self.health_checker = SystemHealthChecker({
                "database": self.db_manager,
                "learning_engine": self.learning_engine,
                "agent_manager": self.agent_manager,
                "knowledge_graph": self.knowledge_graph
            })
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("YMERA Enterprise Platform initialized successfully")
            
        except Exception as e:
            logger.critical(
                "Failed to initialize application",
                error=str(e),
                traceback=traceback.format_exc()
            )
            await self.cleanup()
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        try:
            # Learning engine background tasks
            if self.learning_engine:
                learning_task = asyncio.create_task(
                    self.learning_engine.start_continuous_learning()
                )
                self._background_tasks.append(learning_task)
                
                # Inter-agent knowledge synchronization
                sync_task = asyncio.create_task(
                    self.learning_engine.start_knowledge_synchronization()
                )
                self._background_tasks.append(sync_task)
                
                # Pattern discovery engine
                pattern_task = asyncio.create_task(
                    self.learning_engine.start_pattern_discovery()
                )
                self._background_tasks.append(pattern_task)
            
            # Performance monitoring task
            if self.performance_tracker:
                monitor_task = asyncio.create_task(
                    self.performance_tracker.start_monitoring()
                )
                self._background_tasks.append(monitor_task)
            
            # Health check monitoring
            if self.health_checker:
                health_task = asyncio.create_task(
                    self.health_checker.start_monitoring()
                )
                self._background_tasks.append(health_task)
            
            logger.info(
                "Background tasks started",
                task_count=len(self._background_tasks)
            )
            
        except Exception as e:
            logger.error(
                "Failed to start background tasks",
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise
    
    async def cleanup(self) -> None:
        """Cleanup all application resources"""
        try:
            logger.info("Starting application cleanup")
            
            # Signal shutdown to background tasks
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(
                    *self._background_tasks,
                    return_exceptions=True
                )
            
            # Cleanup components in reverse order
            if self.learning_engine:
                await self.learning_engine.cleanup()
            
            if self.agent_manager:
                await self.agent_manager.cleanup()
            
            if self.knowledge_graph:
                await self.knowledge_graph.cleanup()
            
            if self.performance_tracker:
                await self.performance_tracker.cleanup()
            
            if self.db_manager:
                await self.db_manager.cleanup()
            
            logger.info("Application cleanup completed")
            
        except Exception as e:
            logger.error(
                "Error during cleanup",
                error=str(e),
                traceback=traceback.format_exc()
            )

# Global application manager
app_manager = ApplicationManager()

# ===============================================================================
# FASTAPI APPLICATION SETUP
# ===============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    try:
        await app_manager.initialize()
        logger.info("YMERA Enterprise Platform started successfully")
        yield
    except Exception as e:
        logger.critical(
            "Failed to start application",
            error=str(e),
            traceback=traceback.format_exc()
        )
        sys.exit(1)
    finally:
        # Shutdown
        await app_manager.cleanup()
        logger.info("YMERA Enterprise Platform shutdown completed")

# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="""
    YMERA Enterprise Agent Learning Integration System
    
    A production-ready multi-agent platform featuring:
    - Continuous learning and adaptation
    - Inter-agent knowledge transfer
    - Pattern recognition and discovery
    - Real-time collaboration and optimization
    - Enterprise-grade security and monitoring
    """,
    contact={
        "name": "YMERA Platform Team",
        "email": "platform@ymera.enterprise",
        "url": "https://ymera.enterprise"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    openapi_url=f"{API_PREFIX}/openapi.json",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    lifespan=lifespan
)

# ===============================================================================
# MIDDLEWARE CONFIGURATION
# ===============================================================================

# Security middleware
app.add_middleware(SecurityMiddleware)

# Trusted host middleware
if settings.TRUSTED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.TRUSTED_HOSTS
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# Custom middleware
app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestTracingMiddleware)

# ===============================================================================
# ROUTER REGISTRATION
# ===============================================================================

# Register API routers
app.include_router(
    health_router,
    prefix=f"{API_PREFIX}/health",
    tags=["Health Check"]
)

app.include_router(
    monitoring_router,
    prefix=f"{API_PREFIX}/monitoring",
    tags=["Monitoring"]
)

app.include_router(
    agents_router,
    prefix=f"{API_PREFIX}/agents",
    tags=["Agent Management"]
)

app.include_router(
    learning_router,
    prefix=f"{API_PREFIX}/learning",
    tags=["Learning Engine"]
)

app.include_router(
    knowledge_router,
    prefix=f"{API_PREFIX}/knowledge",
    tags=["Knowledge Management"]
)

# ===============================================================================
# GLOBAL EXCEPTION HANDLERS
# ===============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with detailed logging"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        "HTTP exception occurred",
        request_id=request_id,
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions with comprehensive logging"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "Unhandled exception occurred",
        request_id=request_id,
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        traceback=traceback.format_exc()
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "internal_error",
                "status_code": 500,
                "message": "An internal server error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

# ===============================================================================
# ROOT ENDPOINTS
# ===============================================================================

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with platform information"""
    return {
        "platform": APP_NAME,
        "version": APP_VERSION,
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "api": {
            "docs": f"{API_PREFIX}/docs",
            "redoc": f"{API_PREFIX}/redoc",
            "openapi": f"{API_PREFIX}/openapi.json"
        },
        "endpoints": {
            "health": f"{API_PREFIX}/health",
            "monitoring": f"{API_PREFIX}/monitoring",
            "agents": f"{API_PREFIX}/agents",
            "learning": f"{API_PREFIX}/learning",
            "knowledge": f"{API_PREFIX}/knowledge"
        }
    }

@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint"""
    try:
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate metrics"
        )

@app.get(f"{API_PREFIX}/status")
@track_performance
async def platform_status() -> Dict[str, Any]:
    """Comprehensive platform status endpoint"""
    try:
        status_data = {
            "platform": APP_NAME,
            "version": APP_VERSION,
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check component health
        if app_manager.health_checker:
            health_results = await app_manager.health_checker.check_system_health()
            status_data["components"] = health_results
        
        # Add learning engine metrics
        if app_manager.learning_engine:
            learning_status = await app_manager.learning_engine.get_status()
            status_data["learning_engine"] = learning_status
        
        # Add agent manager metrics
        if app_manager.agent_manager:
            agent_status = await app_manager.agent_manager.get_status()
            status_data["agent_manager"] = agent_status
        
        # Add knowledge graph metrics
        if app_manager.knowledge_graph:
            knowledge_status = await app_manager.knowledge_graph.get_status()
            status_data["knowledge_graph"] = knowledge_status
        
        return status_data
        
    except Exception as e:
        logger.error("Failed to get platform status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve platform status"
        )

# ===============================================================================
# SIGNAL HANDLERS
# ===============================================================================

def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown"""
    
    def signal_handler(signum: int, frame) -> None:
        """Handle shutdown signals"""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name} signal, initiating graceful shutdown")
        
        # Create shutdown task
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(app_manager.cleanup())
        
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)

# ===============================================================================
# SERVER STARTUP FUNCTIONS
# ===============================================================================

def create_app() -> FastAPI:
    """Factory function to create FastAPI application"""
    setup_signal_handlers()
    return app

async def start_server() -> None:
    """Start the YMERA server programmatically"""
    try:
        config = uvicorn.Config(
            app="main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            workers=settings.WORKERS if not settings.DEBUG else 1,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=settings.ACCESS_LOG,
            use_colors=True,
            server_header=False,
            date_header=False,
            proxy_headers=True,
            forwarded_allow_ips="*" if settings.DEBUG else settings.TRUSTED_PROXIES
        )
        
        server = uvicorn.Server(config)
        
        logger.info(
            "Starting YMERA Enterprise Platform",
            host=settings.HOST,
            port=settings.PORT,
            workers=config.workers,
            debug=settings.DEBUG
        )
        
        await server.serve()
        
    except Exception as e:
        logger.critical(
            "Failed to start server",
            error=str(e),
            traceback=traceback.format_exc()
        )
        sys.exit(1)

def run_development_server() -> None:
    """Run development server with hot reload"""
    try:
        logger.info("Starting YMERA development server")
        
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=True,
            reload_dirs=["ymera_agents", "config", "database", "monitoring", "security", "utils"],
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True,
            use_colors=True
        )
        
    except Exception as e:
        logger.critical(
            "Development server failed to start",
            error=str(e),
            traceback=traceback.format_exc()
        )
        sys.exit(1)

def run_production_server() -> None:
    """Run production server with multiple workers"""
    try:
        logger.info("Starting YMERA production server")
        
        # Use Gunicorn for production deployment
        import gunicorn.app.wsgiapp as wsgi
        
        sys.argv = [
            "gunicorn",
            "--bind", f"{settings.HOST}:{settings.PORT}",
            "--workers", str(settings.WORKERS),
            "--worker-class", "uvicorn.workers.UvicornWorker",
            "--worker-connections", str(settings.WORKER_CONNECTIONS),
            "--max-requests", str(settings.MAX_REQUESTS),
            "--max-requests-jitter", str(settings.MAX_REQUESTS_JITTER),
            "--timeout", str(settings.WORKER_TIMEOUT),
            "--keepalive", str(settings.KEEPALIVE),
            "--preload",
            "--log-level", settings.LOG_LEVEL.lower(),
            "--access-logfile", "-",
            "--error-logfile", "-",
            "main:app"
        ]
        
        wsgi.run()
        
    except Exception as e:
        logger.critical(
            "Production server failed to start",
            error=str(e),
            traceback=traceback.format_exc()
        )
        sys.exit(1)

# ===============================================================================
# CLI ENTRY POINTS
# ===============================================================================

def main() -> None:
    """Main entry point for the application"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YMERA Enterprise Agent Learning Integration System"
    )
    parser.add_argument(
        "--mode",
        choices=["development", "production"],
        default="development",
        help="Server mode (default: development)"
    )
    parser.add_argument(
        "--host",
        default=settings.HOST,
        help=f"Host to bind to (default: {settings.HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.PORT,
        help=f"Port to bind to (default: {settings.PORT})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.WORKERS,
        help=f"Number of worker processes (default: {settings.WORKERS})"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=settings.LOG_LEVEL,
        help=f"Log level (default: {settings.LOG_LEVEL})"
    )
    
    args = parser.parse_args()
    
    # Update settings from CLI arguments
    settings.HOST = args.host
    settings.PORT = args.port
    settings.WORKERS = args.workers
    settings.LOG_LEVEL = args.log_level
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Start server based on mode
    if args.mode == "development":
        run_development_server()
    else:
        run_production_server()

# ===============================================================================
# MODULE EXPORTS
# ===============================================================================

__all__ = [
    "app",
    "app_manager",
    "create_app",
    "start_server",
    "main"
]

# ===============================================================================
# ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    main()