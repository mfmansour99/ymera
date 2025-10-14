"""YMERA Enterprise - API Gateway Core Routes Module v4.0"""

import logging
logger = logging.getLogger("ymera.api_gateway")

try:
    from .gateway_routing import create_api_router
    from .ymera_auth_routes import auth_router
    from .ymera_file_routes import file_router
    
    __all__ = [
        "create_api_router",
        "auth_router",
        "file_router",
    ]
    _IMPORT_SUCCESS = True
    logger.info("API Gateway Core Routes module loaded successfully")
except ImportError as e:
    _IMPORT_SUCCESS = False
    logger.error(f"Failed to import API Gateway components: {e}")
    logger.error(f"Import error details: {type(e).__name__}: {e}")
    __all__ = []

__version__ = "4.0.0"
