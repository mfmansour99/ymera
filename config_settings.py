"""
Configuration Settings Compatibility Module
Provides backward compatibility for various config patterns
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# ENVIRONMENT VARIABLE HELPERS
# ============================================================================

def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """Get environment variable with optional default and required check"""
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} not set")
    return value

def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_int_env(key: str, default: int = 0) -> int:
    """Get integer environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

# ============================================================================
# SETTINGS CLASS
# ============================================================================

@dataclass
class Settings:
    """Unified settings class compatible with multiple patterns"""
    
    # Application Settings
    APP_NAME: str = "YMERA Enterprise"
    APP_VERSION: str = "5.0"
    ENVIRONMENT: str = field(default_factory=lambda: get_env("ENVIRONMENT", "development"))
    DEBUG: bool = field(default_factory=lambda: get_bool_env("DEBUG", False))
    
    # Database Settings
    DATABASE_URL: str = field(
        default_factory=lambda: get_env(
            "DATABASE_URL", 
            "sqlite+aiosqlite:///./ymera.db"
        )
    )
    DATABASE_ECHO: bool = field(default_factory=lambda: get_bool_env("DATABASE_ECHO", False))
    DATABASE_POOL_SIZE: int = field(default_factory=lambda: get_int_env("DATABASE_POOL_SIZE", 5))
    DATABASE_MAX_OVERFLOW: int = field(default_factory=lambda: get_int_env("DATABASE_MAX_OVERFLOW", 10))
    
    # Redis Settings
    REDIS_URL: str = field(
        default_factory=lambda: get_env(
            "REDIS_URL", 
            "redis://localhost:6379/0"
        )
    )
    REDIS_MAX_CONNECTIONS: int = field(default_factory=lambda: get_int_env("REDIS_MAX_CONNECTIONS", 20))
    REDIS_ENABLED: bool = field(default_factory=lambda: get_bool_env("REDIS_ENABLED", True))
    
    # Cache Settings
    CACHE_TTL: int = field(default_factory=lambda: get_int_env("CACHE_TTL", 300))
    CACHE_ENABLED: bool = field(default_factory=lambda: get_bool_env("CACHE_ENABLED", True))
    
    # Knowledge Graph Settings
    KG_MAX_NODES: int = field(default_factory=lambda: get_int_env("KG_MAX_NODES", 1000000))
    KG_MAX_CONNECTIONS_PER_NODE: int = field(
        default_factory=lambda: get_int_env("KG_MAX_CONNECTIONS_PER_NODE", 1000)
    )
    KG_SIMILARITY_THRESHOLD: float = 0.8
    KG_RETENTION_DAYS: int = field(default_factory=lambda: get_int_env("KG_RETENTION_DAYS", 365))
    KG_EMBEDDING_DIMENSION: int = field(default_factory=lambda: get_int_env("KG_EMBEDDING_DIMENSION", 128))
    
    # Vector Database Settings (Pinecone)
    PINECONE_API_KEY: str = field(default_factory=lambda: get_env("PINECONE_API_KEY", ""))
    PINECONE_ENVIRONMENT: str = field(default_factory=lambda: get_env("PINECONE_ENVIRONMENT", "us-west1-gcp"))
    PINECONE_INDEX_NAME: str = field(default_factory=lambda: get_env("PINECONE_INDEX_NAME", "ymera-enterprise"))
    PINECONE_DIMENSION: int = field(default_factory=lambda: get_int_env("PINECONE_DIMENSION", 1536))
    
    # Security Settings
    SECRET_KEY: str = field(
        default_factory=lambda: get_env(
            "SECRET_KEY", 
            "dev-secret-key-change-in-production"
        )
    )
    ENCRYPTION_KEY: str = field(default_factory=lambda: get_env("ENCRYPTION_KEY", ""))
    
    # API Settings
    API_HOST: str = field(default_factory=lambda: get_env("API_HOST", "0.0.0.0"))
    API_PORT: int = field(default_factory=lambda: get_int_env("API_PORT", 8000))
    API_WORKERS: int = field(default_factory=lambda: get_int_env("API_WORKERS", 4))
    API_TIMEOUT: int = field(default_factory=lambda: get_int_env("API_TIMEOUT", 60))
    
    # Logging Settings
    LOG_LEVEL: str = field(default_factory=lambda: get_env("LOG_LEVEL", "INFO"))
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[str] = field(default_factory=lambda: get_env("LOG_FILE"))
    
    # Monitoring Settings
    METRICS_ENABLED: bool = field(default_factory=lambda: get_bool_env("METRICS_ENABLED", True))
    METRICS_PORT: int = field(default_factory=lambda: get_int_env("METRICS_PORT", 9090))
    
    # Agent Settings
    MAX_AGENTS: int = field(default_factory=lambda: get_int_env("MAX_AGENTS", 100))
    AGENT_TIMEOUT: int = field(default_factory=lambda: get_int_env("AGENT_TIMEOUT", 300))
    
    # File Storage Settings
    UPLOAD_DIR: str = field(default_factory=lambda: get_env("UPLOAD_DIR", "./uploads"))
    MAX_UPLOAD_SIZE: int = field(default_factory=lambda: get_int_env("MAX_UPLOAD_SIZE", 104857600))  # 100MB
    
    # Performance Settings
    ENABLE_PROFILING: bool = field(default_factory=lambda: get_bool_env("ENABLE_PROFILING", False))
    ENABLE_TRACING: bool = field(default_factory=lambda: get_bool_env("ENABLE_TRACING", False))
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure upload directory exists
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith('_') and key.isupper()
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value with optional default"""
        return getattr(self, key, default)

# ============================================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================================

_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    """Get or create global settings instance (singleton pattern)"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

def reset_settings():
    """Reset global settings instance (useful for testing)"""
    global _settings_instance
    _settings_instance = None

# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================

# For code expecting specific function names
def get_config_settings() -> Settings:
    """Legacy function name compatibility"""
    return get_settings()

def load_settings() -> Settings:
    """Alternative function name"""
    return get_settings()

# ============================================================================
# CONFIGURATION PROFILES
# ============================================================================

class DevelopmentConfig(Settings):
    """Development environment configuration"""
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    DATABASE_ECHO: bool = True
    LOG_LEVEL: str = "DEBUG"

class ProductionConfig(Settings):
    """Production environment configuration"""
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    DATABASE_ECHO: bool = False
    LOG_LEVEL: str = "INFO"
    METRICS_ENABLED: bool = True

class TestingConfig(Settings):
    """Testing environment configuration"""
    ENVIRONMENT: str = "testing"
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite+aiosqlite:///:memory:"
    REDIS_ENABLED: bool = False
    CACHE_ENABLED: bool = False

def get_config_by_environment(env: str = None) -> Settings:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    config_class = configs.get(env.lower(), Settings)
    return config_class()

# ============================================================================
# VALIDATORS
# ============================================================================

def validate_database_url(url: str) -> bool:
    """Validate database URL format"""
    valid_prefixes = ['sqlite', 'postgresql', 'mysql', 'oracle']
    return any(url.startswith(prefix) for prefix in valid_prefixes)

def validate_redis_url(url: str) -> bool:
    """Validate Redis URL format"""
    return url.startswith('redis://') or url.startswith('rediss://')

def validate_settings(settings: Settings) -> tuple[bool, list[str]]:
    """Validate settings configuration"""
    errors = []
    
    # Database validation
    if not validate_database_url(settings.DATABASE_URL):
        errors.append(f"Invalid database URL: {settings.DATABASE_URL}")
    
    # Redis validation
    if settings.REDIS_ENABLED and not validate_redis_url(settings.REDIS_URL):
        errors.append(f"Invalid Redis URL: {settings.REDIS_URL}")
    
    # Security validation
    if settings.ENVIRONMENT == "production":
        if settings.SECRET_KEY == "dev-secret-key-change-in-production":
            errors.append("Production environment requires custom SECRET_KEY")
        
        if settings.DEBUG:
            errors.append("DEBUG should be False in production")
    
    # Pinecone validation
    if settings.PINECONE_API_KEY and len(settings.PINECONE_API_KEY) < 10:
        errors.append("Invalid Pinecone API key")
    
    return len(errors) == 0, errors

# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

class TemporarySettings:
    """Context manager for temporary settings changes (useful for testing)"""
    
    def __init__(self, **overrides):
        self.overrides = overrides
        self.original_values = {}
        self.settings = get_settings()
    
    def __enter__(self):
        for key, value in self.overrides.items():
            self.original_values[key] = getattr(self.settings, key, None)
            setattr(self.settings, key, value)
        return self.settings
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.original_values.items():
            setattr(self.settings, key, value)

# ============================================================================
# CONFIGURATION EXPORT
# ============================================================================

def export_config_to_env_file(filepath: str = ".env.example") -> None:
    """Export current configuration as .env file template"""
    settings = get_settings()
    
    env_content = [
        "# YMERA Enterprise Configuration",
        "# Generated configuration template",
        "",
        "# Application Settings",
        f"ENVIRONMENT={settings.ENVIRONMENT}",
        f"DEBUG={settings.DEBUG}",
        "",
        "# Database Settings",
        f"DATABASE_URL={settings.DATABASE_URL}",
        f"DATABASE_POOL_SIZE={settings.DATABASE_POOL_SIZE}",
        "",
        "# Redis Settings",
        f"REDIS_URL={settings.REDIS_URL}",
        f"REDIS_ENABLED={settings.REDIS_ENABLED}",
        "",
        "# Knowledge Graph Settings",
        f"KG_MAX_NODES={settings.KG_MAX_NODES}",
        f"KG_RETENTION_DAYS={settings.KG_RETENTION_DAYS}",
        "",
        "# Security Settings",
        "SECRET_KEY=your-secret-key-here",
        "ENCRYPTION_KEY=your-encryption-key-here",
        "",
        "# API Settings",
        f"API_HOST={settings.API_HOST}",
        f"API_PORT={settings.API_PORT}",
        "",
        "# Pinecone Settings",
        "PINECONE_API_KEY=your-pinecone-api-key",
        f"PINECONE_ENVIRONMENT={settings.PINECONE_ENVIRONMENT}",
        f"PINECONE_INDEX_NAME={settings.PINECONE_INDEX_NAME}",
        ""
    ]
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(env_content))
    
    print(f"Configuration template exported to {filepath}")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def print_configuration_examples():
    """Print usage examples"""
    examples = """
    ═══════════════════════════════════════════════════════════════
    Configuration Settings Usage Examples
    ═══════════════════════════════════════════════════════════════
    
    1. Basic Usage:
       from config_settings import get_settings
       
       settings = get_settings()
       print(settings.DATABASE_URL)
    
    2. Environment-Specific Config:
       from config_settings import get_config_by_environment
       
       config = get_config_by_environment("production")
    
    3. Temporary Override (Testing):
       from config_settings import TemporarySettings
       
       with TemporarySettings(DEBUG=True, REDIS_ENABLED=False):
           # Code with overridden settings
           pass
    
    4. Validation:
       from config_settings import validate_settings
       
       is_valid, errors = validate_settings(settings)
       if not is_valid:
           print("Configuration errors:", errors)
    
    5. Export Template:
       from config_settings import export_config_to_env_file
       
       export_config_to_env_file(".env.example")
    
    ═══════════════════════════════════════════════════════════════
    """
    print(examples)

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Initialize settings on import
settings = get_settings()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Settings",
    "get_settings",
    "get_config_settings",
    "load_settings",
    "reset_settings",
    "get_config_by_environment",
    "DevelopmentConfig",
    "ProductionConfig",
    "TestingConfig",
    "validate_settings",
    "TemporarySettings",
    "export_config_to_env_file",
    "settings"
]

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Configuration Settings Module Test\n")
    
    # Test 1: Get settings
    print("1. Testing get_settings()...")
    s = get_settings()
    print(f"   ✓ Environment: {s.ENVIRONMENT}")
    print(f"   ✓ Database URL: {s.DATABASE_URL}")
    print(f"   ✓ Redis URL: {s.REDIS_URL}")
    
    # Test 2: Validation
    print("\n2. Testing validation...")
    is_valid, errors = validate_settings(s)
    print(f"   ✓ Valid: {is_valid}")
    if errors:
        for error in errors:
            print(f"   ⚠ {error}")
    
    # Test 3: Temporary settings
    print("\n3. Testing temporary settings...")
    print(f"   Original DEBUG: {s.DEBUG}")
    with TemporarySettings(DEBUG=True):
        print(f"   Temporary DEBUG: {s.DEBUG}")
    print(f"   Restored DEBUG: {s.DEBUG}")
    
    # Test 4: Export template
    print("\n4. Testing export template...")
    try:
        export_config_to_env_file(".env.test")
        print("   ✓ Template exported successfully")
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
    
    # Test 5: Configuration dict
    print("\n5. Testing configuration dict...")
    config_dict = s.to_dict()
    print(f"   ✓ Configuration has {len(config_dict)} settings")
    
    print("\n✓ All tests completed!")
    print("\nFor usage examples, run:")
    print("  from config_settings import print_configuration_examples")
    print("  print_configuration_examples()")