"""
YMERA Enterprise - Core Engine Module
Production-Ready Core Engine Package - v4.0
Enterprise-grade implementation with complete imports
"""

# ===============================================================================
# STANDARD IMPORTS
# ===============================================================================
import logging
from typing import Dict, List, Any, Optional

# ===============================================================================
# STRUCTURED LOGGING
# ===============================================================================
try:
    import structlog
    logger = structlog.get_logger("ymera.core_engine")
except ImportError:
    logger = logging.getLogger("ymera.core_engine")
    logger.warning("structlog not available, using standard logging")

# ===============================================================================
# CORE ENGINE IMPORTS
# ===============================================================================
try:
    # Core Engine main class
    from .core_engine import CoreEngine, LearningEngineConfig
    
    # Utils module - ALL utility functions
    from . import utils
    
    # Export utils submodules if they exist
    if hasattr(utils, 'validators'):
        from .utils import validators
    if hasattr(utils, 'helpers'):
        from .utils import helpers
    if hasattr(utils, 'encryption'):
        from .utils import encryption
    if hasattr(utils, 'cache_manager'):
        from .utils import cache_manager
    
    logger.info("Core Engine module loaded successfully", version="4.0.0")
    
except ImportError as e:
    error_msg = f"Failed to import Core Engine components: {e}"
    logger.error(error_msg)
    raise ImportError(error_msg) from e

# ===============================================================================
# MODULE METADATA
# ===============================================================================
__version__ = "4.0.0"
__author__ = "YMERA Enterprise Team"
__description__ = "Production-ready Core Engine with Learning capabilities"

# ===============================================================================
# PUBLIC API
# ===============================================================================
__all__ = [
    # Core classes
    'CoreEngine',
    'LearningEngineConfig',
    
    # Utils module
    'utils',
    
    # Version info
    '__version__',
]

# ===============================================================================
# MODULE INITIALIZATION
# ===============================================================================
def get_version() -> str:
    """Get Core Engine version"""
    return __version__

def check_dependencies() -> Dict[str, Any]:
    """Check if all required dependencies are available"""
    dependencies_status = {
        "core_engine": True,
        "utils": True,
        "structlog": False,
        "aioredis": False,
        "asyncio": False,
    }
    
    # Check optional dependencies
    try:
        import structlog
        dependencies_status["structlog"] = True
    except ImportError:
        pass
    
    try:
        import aioredis
        dependencies_status["aioredis"] = True
    except ImportError:
        pass
    
    try:
        import asyncio
        dependencies_status["asyncio"] = True
    except ImportError:
        pass
    
    return {
        "status": "ready" if all(dependencies_status.values()) else "partial",
        "dependencies": dependencies_status,
        "version": __version__
    }

# ===============================================================================
# AUTOMATIC HEALTH CHECK ON IMPORT
# ===============================================================================
_health_status = check_dependencies()
if _health_status["status"] != "ready":
    logger.warning(
        "Core Engine loaded with missing optional dependencies",
        missing=[k for k, v in _health_status["dependencies"].items() if not v]
    )
