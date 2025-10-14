"""
YMERA Enterprise - Core Configuration Module
Production-Ready Configuration System - v4.0
Enterprise-grade implementation with complete error handling

File: backend/app/CORE_CONFIGURATION/__init__.py
"""

import sys
import logging
from typing import Optional

# Setup basic logging for initialization errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "4.0.0"
__author__ = "YMERA Enterprise Team"
__description__ = "Production-ready core configuration system"

# ===============================================================================
# CORE IMPORTS - Order matters to avoid circular dependencies
# ===============================================================================
# Import hierarchy: Settings -> Security -> Database -> Manager -> Registry

try:
    # Step 1: Import Settings first (no dependencies)
    from .config_settings import (
        Settings,
        DevelopmentSettings,
        ProductionSettings,
        TestingSettings,
        get_settings,
        get_settings_class,
        validate_settings
    )
    logger.info("✓ Settings module loaded successfully")
    
except ImportError as e:
    logger.error(f"✗ Failed to import Settings: {e}")
    raise ImportError(
        f"Critical: Cannot import Settings from config_settings.py. "
        f"Ensure the file exists and has no syntax errors. Error: {e}"
    ) from e

try:
    # Step 2: Import Security Config (depends on Settings)
    from .config_security import (
        SecurityConfig,
        TokenType,
        PermissionLevel,
        TokenPayload,
        AuthenticationResponse,
        User,
        PasswordManager,
        JWTManager,
        EncryptionManager,
        RateLimiter,
        SecurityDependencies,
        SecurityMiddleware,
        get_security_config,
        initialize_security,
        generate_session_id,
        constant_time_compare,
        hash_api_key,
        verify_api_key
    )
    logger.info("✓ Security module loaded successfully")
    
except ImportError as e:
    logger.error(f"✗ Failed to import Security: {e}")
    raise ImportError(
        f"Critical: Cannot import Security configuration. "
        f"Check config_security.py for errors. Error: {e}"
    ) from e

try:
    # Step 3: Import Database Config (depends on Settings)
    from .config_database import (
        Base,
        metadata,
        DatabaseConfig,
        DatabaseManager,
        MigrationManager,
        DatabaseError,
        DatabaseConnectionError,
        DatabaseTimeoutError,
        MigrationError,
        get_database_config,
        get_database_session,
        get_db_session,
        initialize_database,
        close_database,
        create_tables,
        drop_tables
    )
    logger.info("✓ Database module loaded successfully")
    
except ImportError as e:
    logger.error(f"✗ Failed to import Database: {e}")
    raise ImportError(
        f"Critical: Cannot import Database configuration. "
        f"Check config_database.py for errors. Error: {e}"
    ) from e

try:
    # Step 4: Import Configuration Manager (depends on all above)
    from .config_manager import (
        ConfigManager,
        YMERAConfig,
        DatabaseConfig as ManagerDatabaseConfig,
        RedisConfig,
        SecurityConfig as ManagerSecurityConfig,
        AIConfig,
        VectorDBConfig,
        GitHubConfig,
        MonitoringConfig,
        ServerConfig,
        AgentConfig
    )
    logger.info("✓ Configuration Manager loaded successfully")
    
except ImportError as e:
    logger.error(f"✗ Failed to import ConfigManager: {e}")
    raise ImportError(
        f"Critical: Cannot import Configuration Manager. "
        f"Check config_manager.py for errors. Error: {e}"
    ) from e

try:
    # Step 5: Import Configuration Registry (optional, depends on all configs)
    from .config_init import (
        ConfigurationRegistry,
        config,
        validate_environment,
        setup_environment,
        get_config_for_environment
    )
    logger.info("✓ Configuration Registry loaded successfully")
    
except ImportError as e:
    logger.warning(f"⚠ Configuration Registry not available: {e}")
    # These are optional, set to None if not available
    config = None
    ConfigurationRegistry = None
    validate_environment = None
    setup_environment = None
    get_config_for_environment = None

# ===============================================================================
# GLOBAL CONFIGURATION INSTANCES (Singletons)
# ===============================================================================

_settings_instance: Optional[Settings] = None
_database_config_instance: Optional[DatabaseConfig] = None
_security_config_instance: Optional[SecurityConfig] = None

def get_global_settings() -> Settings:
    """
    Get or create global settings instance (singleton pattern).
    
    Returns:
        Settings: Cached settings instance
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = get_settings()
    return _settings_instance

def get_global_database_config() -> DatabaseConfig:
    """
    Get or create global database config instance (singleton pattern).
    
    Returns:
        DatabaseConfig: Cached database configuration
    """
    global _database_config_instance
    if _database_config_instance is None:
        _database_config_instance = get_database_config()
    return _database_config_instance

def get_global_security_config() -> SecurityConfig:
    """
    Get or create global security config instance (singleton pattern).
    
    Returns:
        SecurityConfig: Cached security configuration
    """
    global _security_config_instance
    if _security_config_instance is None:
        _security_config_instance = get_security_config()
    return _security_config_instance

# ===============================================================================
# VALIDATION & DIAGNOSTICS
# ===============================================================================

def validate_configuration() -> dict:
    """
    Validate entire configuration system.
    
    Performs comprehensive validation of all configuration modules
    including settings, database, and security configurations.
    
    Returns:
        dict: Validation results with status, modules info, errors, and warnings
            {
                "valid": bool,
                "modules": dict,
                "errors": list,
                "warnings": list
            }
    """
    results = {
        "valid": True,
        "modules": {},
        "errors": [],
        "warnings": []
    }
    
    # Validate Settings
    try:
        settings = get_global_settings()
        settings_validation = validate_settings(settings)
        results["modules"]["settings"] = {
            "loaded": True,
            "environment": settings.ENVIRONMENT,
            "validation": settings_validation
        }
        if not settings_validation["valid"]:
            results["valid"] = False
            results["errors"].extend(settings_validation["errors"])
        results["warnings"].extend(settings_validation["warnings"])
        
    except Exception as e:
        results["valid"] = False
        results["modules"]["settings"] = {"loaded": False, "error": str(e)}
        results["errors"].append(f"Settings validation failed: {e}")
    
    # Validate Database Config
    try:
        db_config = get_global_database_config()
        results["modules"]["database"] = {
            "loaded": True,
            "database_type": db_config.database_type,
            "host": db_config.host,
            "pool_size": db_config.pool_size
        }
        
    except Exception as e:
        results["valid"] = False
        results["modules"]["database"] = {"loaded": False, "error": str(e)}
        results["errors"].append(f"Database config validation failed: {e}")
    
    # Validate Security Config
    try:
        security_config = get_global_security_config()
        results["modules"]["security"] = {
            "loaded": True,
            "jwt_algorithm": security_config.jwt_algorithm,
            "password_min_length": security_config.password_min_length
        }
        
    except Exception as e:
        results["valid"] = False
        results["modules"]["security"] = {"loaded": False, "error": str(e)}
        results["errors"].append(f"Security config validation failed: {e}")
    
    return results

def run_diagnostic() -> None:
    """
    Run comprehensive diagnostic and print results to console.
    
    This function performs a full system diagnostic including:
    - Module loading status
    - Configuration validation
    - Error detection
    - Warning identification
    
    Exits with code 1 if validation fails, 0 if successful.
    """
    print("\n" + "="*80)
    print("YMERA CORE CONFIGURATION - DIAGNOSTIC REPORT")
    print("="*80 + "\n")
    
    results = validate_configuration()
    
    print(f"Overall Status: {'✓ PASSED' if results['valid'] else '✗ FAILED'}\n")
    
    print("Module Status:")
    print("-" * 80)
    for module_name, module_info in results["modules"].items():
        status = "✓" if module_info.get("loaded") else "✗"
        print(f"{status} {module_name.upper()}: {module_info}")
    
    if results["errors"]:
        print("\nErrors:")
        print("-" * 80)
        for error in results["errors"]:
            print(f"  ✗ {error}")
    
    if results["warnings"]:
        print("\nWarnings:")
        print("-" * 80)
        for warning in results["warnings"]:
            print(f"  ⚠ {warning}")
    
    print("\n" + "="*80 + "\n")
    
    if not results["valid"]:
        print("❌ Configuration validation FAILED. Please fix errors above.")
        sys.exit(1)
    else:
        print("✅ Configuration validation PASSED. System ready for production.")

# ===============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# ===============================================================================

# Alias for common imports
AppSettings = Settings
get_app_settings = get_settings

# ===============================================================================
# COMPREHENSIVE EXPORTS
# ===============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    
    # Settings exports
    "Settings",
    "DevelopmentSettings",
    "ProductionSettings",
    "TestingSettings",
    "get_settings",
    "get_settings_class",
    "validate_settings",
    
    # Database exports
    "Base",
    "metadata",
    "DatabaseConfig",
    "DatabaseManager",
    "MigrationManager",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseTimeoutError",
    "MigrationError",
    "get_database_config",
    "get_database_session",
    "get_db_session",
    "initialize_database",
    "close_database",
    "create_tables",
    "drop_tables",
    
    # Security exports
    "SecurityConfig",
    "TokenType",
    "PermissionLevel",
    "TokenPayload",
    "AuthenticationResponse",
    "User",
    "PasswordManager",
    "JWTManager",
    "EncryptionManager",
    "RateLimiter",
    "SecurityDependencies",
    "SecurityMiddleware",
    "get_security_config",
    "initialize_security",
    "generate_session_id",
    "constant_time_compare",
    "hash_api_key",
    "verify_api_key",
    
    # Configuration Manager exports
    "ConfigManager",
    "YMERAConfig",
    "RedisConfig",
    "AIConfig",
    "VectorDBConfig",
    "GitHubConfig",
    "MonitoringConfig",
    "ServerConfig",
    "AgentConfig",
    
    # Registry exports (if available)
    "ConfigurationRegistry",
    "config",
    "validate_environment",
    "setup_environment",
    "get_config_for_environment",
    
    # Global instance getters
    "get_global_settings",
    "get_global_database_config",
    "get_global_security_config",
    
    # Validation & diagnostics
    "validate_configuration",
    "run_diagnostic",
    
    # Backwards compatibility
    "AppSettings",
    "get_app_settings",
]

# ===============================================================================
# INITIALIZATION MESSAGE
# ===============================================================================

logger.info(
    f"YMERA Core Configuration v{__version__} loaded successfully. "
    f"All {len([x for x in __all__ if not x.startswith('_')])} exports available."
)
