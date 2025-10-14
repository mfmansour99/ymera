"""YMERA Enterprise - Settings Compatibility Module v4.0"""

# Re-export from CORE_CONFIGURATION
from app.CORE_CONFIGURATION.config_settings import get_settings, Settings

__all__ = ["get_settings", "Settings"]
