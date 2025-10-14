"""
YMERA Enterprise - Main Application Entry Point
Production-Ready Multi-Agent Platform - v4.0
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn

# Import all agent components
from agent_integration import (
    ProductionAgentLearningIntegrator,
    initialize_agent_integration,
    health_check as integration_health_check
)
from agent_registry import (
    initialize_agent_registry,
    get_agent_registry,
    shutdown_agent_registry,
    health_check as registry_health_check,
    AgentRegistrationRequest,
    AgentUpdateRequest,
    AgentQueryRequest,
    validate_agent_registration,
    create_agent_info_from_request,
    format_agent_info_for_response
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("ymera.main")

# ===============================================================================
# APPLICATION LIFECYCLE MANAGEMENT
# ===============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("Starting YMERA Agent Platform")
    
    # Startup
    try:
        # Initialize agent registry
        app.state.agent_registry = await initialize_agent_registry()
        logger.info("Agent registry initialized")
        
        # Initialize learning integration
        app.state.learning_integrator = await initialize_agent_integration()
        logger.info("Learning integration initialized")
        
        # Start background tasks
        app.state.background_tasks = []
        
        logger.info("YMERA Agent Platform started successfully")
        
    except Exception as e:
        logger.critical("Failed to start application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    try:
        logger.info("Shutting down YMERA Agent Platform")
        
        # Cancel background tasks
        for task in app.state.background_tasks:
            task.cancel()
        
        # Cleanup learning integrator
        if hasattr(app.state, 'learning_integrator'):
            await app.state.learning_integrator.cleanup()
        
        # Cleanup agent registry
        await shutdown_agent_registry()
        
        logger.info("YMERA Agent Platform shut down successfully")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))

# ===============================================================================
# APPLICATION INITIALIZATION
# ===============================================================================

app = FastAPI(
    title="YMERA Enterprise Agent Platform",
    description="Production-ready multi-agent learning and collaboration platform",
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# ===============================================================================
# MIDDLEWARE CONFIGURATION
# ===============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware (configure for production)
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
# )

# ===============================================================================
# EXCEPTION HANDLERS
# ===============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(
        "Unexpected exception occurred",
        error=str(exc),
        path=request.url.path,
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ===============================================================================
# HEALTH CHECK ENDPOINTS
# ===============================================================================

@app.get("/health")
async def overall_health_check():
    """Overall system health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "4.0.0",
            "components": {}
        }
        
        # Check agent registry
        try:
            registry_health = await registry_health_check()
            health_status["components"]["agent_registry"] = registry_health
        except Exception as e:
            health_status["components"]["agent_registry"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check learning integration
        try:
            integration_health = await integration_health_check()
            health_status["components"]["learning_integration"] = integration_health
        except Exception as e:
            health_status["components"]["learning_integration"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Check critical components
        registry = await get_agent_registry()
        
        return {
            "ready": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }

# ===============================================================================
# AGENT REGISTRY ENDPOINTS
# ===============================================================================

@app.post("/api/agents/register", status_code=status.HTTP_201_CREATED)
async def register_agent(request: AgentRegistrationRequest):
    """Register a new agent"""
    try:
        registry = await get_agent_registry()
        
        # Create agent info from request
        agent_info = await create_agent_info_from_request(request)
        
        # Register agent
        agent_id = await registry.register_agent(agent_info)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "message": "Agent registered successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Agent registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@app.delete("/api/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent"""
    try:
        registry = await get_agent_registry()
        success = await registry.unregister_agent(agent_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Agent {agent_id} unregistered successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent unregistration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unregistration failed: {str(e)}"
        )

@app.put("/api/agents/{agent_id}/status")
async def update_agent_status(agent_id: str, request: AgentUpdateRequest):
    """Update agent status"""
    try:
        registry = await get_agent_registry()
        
        from agent_registry import AgentStatus
        status_enum = AgentStatus(request.status) if request.status else None
        
        success = await registry.update_agent_status(
            agent_id,
            status_enum,
            request.metrics
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Agent {agent_id} status updated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Status update failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status update failed: {str(e)}"
        )

@app.post("/api/agents/query")
async def query_agents(request: AgentQueryRequest):
    """Query agents with filters"""
    try:
        registry = await get_agent_registry()
        
        filters = {}
        if request.agent_types:
            filters["agent_types"] = request.agent_types
        if request.statuses:
            filters["statuses"] = request.statuses
        if request.capabilities:
            filters["capabilities"] = request.capabilities
        if request.tags:
            filters["tags"] = request.tags
        filters["include_offline"] = request.include_offline
        
        agents = await registry.get_active_agents(filters)
        
        # Format response
        agents_data = [
            format_agent_info_for_response(agent, request.include_metrics)
            for agent in agents
        ]
        
        return {
            "success": True,
            "total_count": len(agents_data),
            "agents": agents_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Agent query failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details"""
    try:
        registry = await get_agent_registry()
        agent = await registry.get_agent_by_id(agent_id)
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        return {
            "success": True,
            "agent": format_agent_info_for_response(agent, True),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get agent failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get agent failed: {str(e)}"
        )

@app.get("/api/agents/statistics")
async def get_registry_statistics():
    """Get agent registry statistics"""
    try:
        registry = await get_agent_registry()
        stats = await registry.get_registry_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Get statistics failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get statistics failed: {str(e)}"
        )

# ===============================================================================
# LEARNING INTEGRATION ENDPOINTS
# ===============================================================================

@app.post("/api/learning/experiences/{agent_id}")
async def process_experiences(agent_id: str, experiences: Dict[str, Any]):
    """Process agent learning experiences"""
    try:
        from agent_integration import LearningExperience
        
        # Convert experiences to proper format
        experience_list = []
        for exp_data in experiences.get("experiences", []):
            experience = LearningExperience(
                agent_id=agent_id,
                experience_id=exp_data.get("experience_id", str(asyncio.get_event_loop().time())),
                timestamp=datetime.fromisoformat(exp_data["timestamp"]) if "timestamp" in exp_data else datetime.utcnow(),
                experience_type=exp_data["experience_type"],
                context=exp_data.get("context", {}),
                outcome=exp_data.get("outcome", {}),
                success=exp_data.get("success", False),
                confidence=exp_data.get("confidence", 0.5),
                metadata=exp_data.get("metadata", {})
            )
            experience_list.append(experience)
        
        # Process experiences
        integrator = app.state.learning_integrator
        result = await integrator.process_agent_experiences(agent_id, experience_list)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Experience processing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experience processing failed: {str(e)}"
        )

@app.post("/api/learning/transfer")
async def transfer_knowledge(request: Dict[str, Any]):
    """Transfer knowledge between agents"""
    try:
        from agent_integration import LearningTransferRequest
        
        transfer_request = LearningTransferRequest(**request)
        
        integrator = app.state.learning_integrator
        result = await integrator.transfer_knowledge(
            transfer_request.source_agent_id,
            transfer_request.target_agent_ids,
            transfer_request.knowledge_types,
            transfer_request.similarity_threshold
        )
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Knowledge transfer failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge transfer failed: {str(e)}"
        )

@app.post("/api/learning/sync/{agent_id}")
async def sync_agent_knowledge(agent_id: str):
    """Synchronize agent knowledge"""
    try:
        integrator = app.state.learning_integrator
        result = await integrator.sync_agent_knowledge(agent_id)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Knowledge sync failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge sync failed: {str(e)}"
        )

# ===============================================================================
# SYSTEM INFORMATION ENDPOINTS
# ===============================================================================

@app.get("/api/system/info")
async def get_system_info():
    """Get system information"""
    return {
        "name": "YMERA Enterprise Agent Platform",
        "version": "4.0.0",
        "description": "Production-ready multi-agent learning and collaboration platform",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "agent_registry": "operational",
            "learning_integration": "operational",
            "knowledge_graph": "available",
            "monitoring": "enabled"
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "YMERA Enterprise Agent Platform",
        "version": "4.0.0",
        "status": "operational",
        "documentation": "/api/docs",
        "health_check": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }

# ===============================================================================
# APPLICATION STARTUP
# ===============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False in production
        workers=4,
        log_level="info",
        access_log=True,
        use_colors=True
    )