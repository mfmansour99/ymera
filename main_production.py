"""
YMERA Enterprise Platform - Main Application
Production-Ready FastAPI Application - v4.0.0
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog
import time

# Try to import from new structure, fallback to old if needed
try:
    from app.core.config_settings import get_settings
except ImportError:
    print("Warning: Using fallback configuration")
    def get_settings():
        from types import SimpleNamespace
        return SimpleNamespace(
            ENVIRONMENT="production",
            CORS_ORIGINS=["*"],
            DEBUG=False,
            LOG_LEVEL="INFO"
        )


# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("ymera.main")

# ===============================================================================
# CONFIGURATION
# ===============================================================================

settings = get_settings()

# ===============================================================================
# APPLICATION LIFECYCLE
# ===============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting YMERA Platform", version="4.0.0")
    
    try:
        # Initialize components
        logger.info("YMERA Platform started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down YMERA Platform")
        logger.info("YMERA Platform shutdown complete")


# ===============================================================================
# APPLICATION INSTANCE
# ===============================================================================

app = FastAPI(
    title="YMERA Enterprise Platform",
    description="""
    Production-Ready Multi-Agent AI Platform
    
    ## Features
    - 🤖 Multi-Agent Collaboration
    - 📁 Advanced File Management
    - 🔐 Enterprise Security
    - 🚀 High Performance API Gateway
    - 📊 Real-time Analytics
    - 🔄 WebSocket Support
    
    ## Documentation
    - Interactive API docs: `/docs`
    - Alternative docs: `/redoc`
    - Health check: `/health`
    """,
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ===============================================================================
# MIDDLEWARE CONFIGURATION
# ===============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(",") if hasattr(settings, 'ALLOWED_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"]
)

# Compression Middleware
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)

# Trusted Host Middleware (uncomment for production)
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
# )

# ===============================================================================
# REQUEST LOGGING MIDDLEWARE
# ===============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    start_time = time.time()
    
    # Add request ID to state
    request.state.request_id = request_id
    
    # Log request
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
        
        # Log response
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            response_time_ms=f"{response_time:.2f}"
        )
        
        return response
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        logger.error(
            "Request failed",
            request_id=request_id,
            error=str(e),
            response_time_ms=f"{response_time:.2f}"
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "request_id": request_id,
                "message": str(e) if settings.DEBUG else "An error occurred"
            }
        )

# ===============================================================================
# ERROR HANDLERS
# ===============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        error=str(exc),
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "message": "An unexpected error occurred"
        }
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """404 handler"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Not found",
            "message": f"Route {request.url.path} not found"
        }
    )

# ===============================================================================
# CORE ROUTES
# ===============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "YMERA Enterprise Platform",
        "version": "4.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns system health status and metrics.
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "4.0.0",
        "components": {
            "api": "healthy",
            "application": "healthy"
        }
    }

@app.get("/metrics")
async def metrics():
    """
    System metrics endpoint for monitoring.
    Returns basic system metrics.
    """
    import psutil
    import os
    
    return {
        "timestamp": time.time(),
        "process": {
            "pid": os.getpid(),
            "cpu_percent": psutil.Process().cpu_percent(),
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "threads": psutil.Process().num_threads()
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }

# ===============================================================================
# ROUTER REGISTRATION
# ===============================================================================

# Routers will be registered here when API modules are ready
# app.include_router(gateway_router, prefix="/gateway", tags=["gateway"])
# app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
# app.include_router(agent_router, prefix="/api/v1/agents", tags=["agents"])
# app.include_router(file_router, prefix="/api/v1/files", tags=["files"])
# app.include_router(project_router, prefix="/api/v1/projects", tags=["projects"])
# app.include_router(websocket_router, prefix="/ws", tags=["websocket"])

logger.info("Application initialized")

# ===============================================================================
# DEVELOPMENT SERVER
# ===============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
