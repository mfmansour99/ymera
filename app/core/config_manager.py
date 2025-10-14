"""
YMERA Platform - Configuration Manager
Production-ready configuration management with environment variable support
"""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConfigManager:
    """Central configuration manager for YMERA platform"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_file = config_file or os.getenv('CONFIG_FILE', 'config.json')
        self._load_config()
        
        # API Keys - Multiple keys per service for load balancing
        self.openai_api_keys = self._load_keys('OPENAI_API_KEY')
        self.openai_service_keys = self._load_keys('OPENAI_SERVICE_KEY')
        self.claude_api_keys = self._load_keys('CLAUDE_API_KEY')
        self.gemini_api_keys = self._load_keys('GEMINI_API_KEY')
        self.deepseek_api_keys = self._load_keys('DEEPSEEK_API_KEY')
        self.groq_api_keys = self._load_keys('GROQ_API_KEY')
        self.github_api_keys = self._load_keys('GITHUB_TOKEN')
        self.github_admin_key = os.getenv('GITHUB_ADMIN_TOKEN', '')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY', '')
        
        # Database Configuration
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///ymera.db')
        self.database_pool_size = int(os.getenv('DB_POOL_SIZE', '10'))
        self.database_max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '20'))
        
        # Redis Configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', '50'))
        
        # System Limits
        self.max_agent_load = int(os.getenv('MAX_AGENT_LOAD', '5'))
        self.max_concurrent_tasks = int(os.getenv('MAX_CONCURRENT_TASKS', '10'))
        self.max_task_queue_time_minutes = int(os.getenv('MAX_QUEUE_TIME', '30'))
        self.default_max_retries = int(os.getenv('DEFAULT_MAX_RETRIES', '3'))
        self.max_browser_sessions = int(os.getenv('MAX_BROWSER_SESSIONS', '20'))
        self.max_browser_sessions_per_agent = int(os.getenv('MAX_BROWSER_PER_AGENT', '3'))
        self.max_compute_allocations = int(os.getenv('MAX_COMPUTE_ALLOC', '15'))
        self.max_queue_size = int(os.getenv('MAX_QUEUE_SIZE', '50'))
        
        # Performance Thresholds
        self.performance_thresholds = {
            'max_avg_execution_time': float(os.getenv('MAX_EXEC_TIME', '300')),
            'min_success_rate': float(os.getenv('MIN_SUCCESS_RATE', '0.8')),
            'max_load': float(os.getenv('MAX_LOAD', '0.9')),
            'max_response_time': float(os.getenv('MAX_RESPONSE_TIME', '30'))
        }
        
        # Security Configuration
        self.encryption_key = os.getenv('ENCRYPTION_KEY', 'dev-key-32-chars-long-change-me!!')
        self.jwt_secret_key = os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret-change-in-production')
        self.jwt_algorithm = os.getenv('JWT_ALGORITHM', 'HS256')
        self.jwt_expiration_hours = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
        
        # Logging Configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_format = os.getenv('LOG_FORMAT', 'json')
        self.log_file = os.getenv('LOG_FILE', 'logs/ymera.log')
        
        # Monitoring Configuration
        self.enable_metrics = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        self.metrics_port = int(os.getenv('METRICS_PORT', '9090'))
        self.enable_tracing = os.getenv('ENABLE_TRACING', 'false').lower() == 'true'
        
        # Feature Flags
        self.enable_learning = os.getenv('ENABLE_LEARNING', 'true').lower() == 'true'
        self.enable_auto_scaling = os.getenv('ENABLE_AUTO_SCALING', 'true').lower() == 'true'
        self.enable_caching = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
        
        # Agent Configuration
        self.agent_heartbeat_interval = int(os.getenv('AGENT_HEARTBEAT_INTERVAL', '30'))
        self.agent_timeout_seconds = int(os.getenv('AGENT_TIMEOUT', '300'))
        
    def _load_keys(self, prefix: str) -> List[str]:
        """Load multiple API keys with numeric suffixes"""
        keys = []
        i = 1
        while True:
            key = os.getenv(f'{prefix}_{i}')
            if not key:
                # Try without number for first key
                key = os.getenv(prefix) if i == 1 else None
            if key:
                keys.append(key)
                i += 1
            else:
                break
        
        # If no keys found, return empty list
        return keys if keys else []
    
    def _load_config(self):
        """Load configuration from JSON file if exists"""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.file_config = json.load(f)
            else:
                self.file_config = {}
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            self.file_config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        # Priority: Environment > File > Default
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value
        
        # Check file config
        keys = key.split('.')
        value = self.file_config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """Set configuration value in file config"""
        keys = key.split('.')
        config = self.file_config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self):
        """Save configuration to file"""
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.file_config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check critical API keys
        if not self.openai_api_keys and not self.claude_api_keys:
            issues.append("No AI API keys configured (OpenAI or Claude required)")
        
        # Check database URL
        if not self.database_url:
            issues.append("DATABASE_URL not configured")
        
        # Check Redis URL
        if not self.redis_url:
            issues.append("REDIS_URL not configured")
        
        # Check encryption key length
        if len(self.encryption_key) < 32:
            issues.append("ENCRYPTION_KEY must be at least 32 characters")
        
        # Check JWT secret
        if self.jwt_secret_key == 'dev-jwt-secret-change-in-production':
            issues.append("JWT_SECRET_KEY must be changed from default in production")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary (excluding sensitive data)"""
        return {
            'database_url': self._mask_url(self.database_url),
            'redis_url': self._mask_url(self.redis_url),
            'max_agent_load': self.max_agent_load,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'performance_thresholds': self.performance_thresholds,
            'log_level': self.log_level,
            'enable_metrics': self.enable_metrics,
            'enable_learning': self.enable_learning,
            'api_keys_configured': {
                'openai': len(self.openai_api_keys),
                'claude': len(self.claude_api_keys),
                'gemini': len(self.gemini_api_keys),
                'deepseek': len(self.deepseek_api_keys),
                'groq': len(self.groq_api_keys),
                'github': len(self.github_api_keys)
            }
        }
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive parts of URLs"""
        if '@' in url:
            parts = url.split('@')
            if len(parts) == 2:
                protocol = parts[0].split('//')[0] if '//' in parts[0] else ''
                return f"{protocol}//***:***@{parts[1]}"
        return url
    
    def __repr__(self) -> str:
        return f"ConfigManager(keys={sum(len(v) if isinstance(v, list) else 1 for k, v in self.__dict__.items() if 'key' in k.lower())})"


# Global config instance
config = ConfigManager()


# Export for easy import
__all__ = ['ConfigManager', 'config']
