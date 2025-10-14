"""
YMERA Enterprise Multi-Agent System
Production-Ready FastAPI Application with AI Learning Engine
Enhanced for Replit Deployment with Full Enterprise Features
"""

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
import logging
import os
import asyncio
import json
import time
from datetime import datetime, timedelta
import signal
import sys
import uuid
import hashlib
import hmac
from pathlib import Path
import aiofiles
import redis.asyncio as aioredis
import asyncpg
import httpx
from pydantic import BaseModel, Field
import jwt
from cryptography.fernet import Fernet
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import yaml
from enum import Enum
import traceback
import psutil

# Configure asyncio for optimal performance
if sys.platform != 'win32':
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

# ============================================================================
# CORE CONFIGURATION SYSTEM
# ============================================================================

class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    REPLIT = "replit"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/ymera")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "20"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "30"))
    connection_timeout: int = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))

@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
    connection_timeout: int = int(os.getenv("REDIS_CONNECTION_TIMEOUT", "10"))
    socket_keepalive: bool = True
    socket_keepalive_options: dict = None
    health_check_interval: int = 30

@dataclass
class AIConfig:
    """AI Services Configuration"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    default_provider: str = os.getenv("DEFAULT_AI_PROVIDER", "openai")
    fallback_providers: List[str] = None
    max_retries: int = int(os.getenv("AI_MAX_RETRIES", "3"))
    timeout: int = int(os.getenv("AI_TIMEOUT", "60"))
    rate_limit_per_minute: int = int(os.getenv("AI_RATE_LIMIT", "100"))

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    refresh_token_expire_days: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
    encryption_key: str = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    bcrypt_rounds: int = 12
    session_timeout: int = int(os.getenv("SESSION_TIMEOUT", "3600"))
    max_login_attempts: int = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    lockout_duration: int = int(os.getenv("LOCKOUT_DURATION", "900"))

@dataclass
class LearningConfig:
    """Learning Engine Configuration"""
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.001"))
    batch_size: int = int(os.getenv("LEARNING_BATCH_SIZE", "32"))
    memory_size: int = int(os.getenv("LEARNING_MEMORY_SIZE", "10000"))
    update_frequency: int = int(os.getenv("LEARNING_UPDATE_FREQUENCY", "300"))
    retention_days: int = int(os.getenv("LEARNING_RETENTION_DAYS", "90"))
    min_confidence: float = float(os.getenv("LEARNING_MIN_CONFIDENCE", "0.7"))
    auto_adapt: bool = os.getenv("LEARNING_AUTO_ADAPT", "true").lower() == "true"

class YMERAConfig:
    """Centralized configuration management"""
    
    def __init__(self):
        self.environment: EnvironmentType = EnvironmentType(os.getenv("ENVIRONMENT", "production"))
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Service configs
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.ai = AIConfig()
        self.security = SecurityConfig()
        self.learning = LearningConfig()
        
        # API Configuration
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("PORT", "8000"))
        self.api_workers: int = int(os.getenv("WORKERS", "4"))
        self.max_request_size: int = int(os.getenv("MAX_REQUEST_SIZE", "16777216"))
        
        # External Services
        self.github_token: str = os.getenv("GITHUB_TOKEN", "")
        self.pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
        self.pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "ymera-knowledge")
        
        # Browser Access Configuration
        self.browser_enabled: bool = os.getenv("BROWSER_ENABLED", "true").lower() == "true"
        self.browser_timeout: int = int(os.getenv("BROWSER_TIMEOUT", "30"))
        self.browser_max_pages: int = int(os.getenv("BROWSER_MAX_PAGES", "5"))
        
        # Replit specific configurations
        if self.environment == EnvironmentType.REPLIT:
            self._configure_for_replit()
    
    def _configure_for_replit(self):
        """Configure settings specific to Replit environment"""
        self.api_host = "0.0.0.0"
        self.database.url = os.getenv("REPL_DB_URL", self.database.url)
        self.redis.url = os.getenv("REPL_REDIS_URL", self.redis.url)
        # Adjust for Replit's resource constraints
        self.database.pool_size = min(self.database.pool_size, 10)
        self.redis.max_connections = min(self.redis.max_connections, 20)

# Global configuration instance
config = YMERAConfig()

# ============================================================================
# ENHANCED LOGGING SYSTEM
# ============================================================================

class StructuredLogger:
    """Production-ready structured logging system"""
    
    def __init__(self, name: str = "YMERA"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging with proper formatting"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s - '
                '%(filename)s:%(lineno)d - %(funcName)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, config.log_level.upper()))
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)

logger = StructuredLogger("YMERA-Main")

# ============================================================================
# DATABASE MANAGEMENT SYSTEM
# ============================================================================

class DatabaseManager:
    """Production-ready PostgreSQL database manager with connection pooling"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.is_connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 5
    
    async def initialize(self):
        """Initialize database connection pool with retry logic"""
        while self.connection_attempts < self.max_connection_attempts:
            try:
                self.pool = await asyncpg.create_pool(
                    config.database.url,
                    min_size=5,
                    max_size=config.database.pool_size,
                    command_timeout=config.database.connection_timeout,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300.0,
                )
                
                # Test connection
                async with self.pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                
                self.is_connected = True
                logger.info("Database connection pool initialized successfully")
                await self._create_tables()
                return
                
            except Exception as e:
                self.connection_attempts += 1
                logger.error(f"Database connection attempt {self.connection_attempts} failed: {e}")
                if self.connection_attempts < self.max_connection_attempts:
                    await asyncio.sleep(2 ** self.connection_attempts)
                else:
                    raise Exception("Failed to establish database connection after maximum attempts")
    
    async def _create_tables(self):
        """Create necessary database tables"""
        schema = """
        -- Users and Authentication
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email VARCHAR(255) UNIQUE NOT NULL,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            is_active BOOLEAN DEFAULT true,
            is_verified BOOLEAN DEFAULT false,
            role VARCHAR(50) DEFAULT 'user',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_login TIMESTAMP WITH TIME ZONE,
            login_attempts INTEGER DEFAULT 0,
            locked_until TIMESTAMP WITH TIME ZONE
        );
        
        -- Projects
        CREATE TABLE IF NOT EXISTS projects (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            repository_url VARCHAR(500),
            owner_id UUID REFERENCES users(id),
            status VARCHAR(50) DEFAULT 'active',
            settings JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Agents
        CREATE TABLE IF NOT EXISTS agents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(100) NOT NULL,
            type VARCHAR(50) NOT NULL,
            status VARCHAR(50) DEFAULT 'inactive',
            capabilities JSONB DEFAULT '[]',
            configuration JSONB DEFAULT '{}',
            performance_metrics JSONB DEFAULT '{}',
            learning_data JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_active TIMESTAMP WITH TIME ZONE
        );
        
        -- Agent Sessions
        CREATE TABLE IF NOT EXISTS agent_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID REFERENCES agents(id),
            project_id UUID REFERENCES projects(id),
            user_id UUID REFERENCES users(id),
            status VARCHAR(50) DEFAULT 'active',
            start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            end_time TIMESTAMP WITH TIME ZONE,
            context JSONB DEFAULT '{}',
            results JSONB DEFAULT '{}'
        );
        
        -- Learning Data
        CREATE TABLE IF NOT EXISTS learning_records (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID REFERENCES agents(id),
            session_id UUID REFERENCES agent_sessions(id),
            interaction_type VARCHAR(100) NOT NULL,
            input_data JSONB NOT NULL,
            output_data JSONB NOT NULL,
            feedback_score FLOAT,
            success_metrics JSONB DEFAULT '{}',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            learning_context JSONB DEFAULT '{}'
        );
        
        -- Knowledge Base
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            content_type VARCHAR(50) DEFAULT 'text',
            tags TEXT[] DEFAULT '{}',
            metadata JSONB DEFAULT '{}',
            embedding_id VARCHAR(255),
            relevance_score FLOAT DEFAULT 0.0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_by UUID REFERENCES users(id)
        );
        
        -- Browser Access Logs
        CREATE TABLE IF NOT EXISTS browser_access_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID REFERENCES agents(id),
            session_id UUID REFERENCES agent_sessions(id),
            url TEXT NOT NULL,
            action VARCHAR(100) NOT NULL,
            success BOOLEAN NOT NULL,
            response_time INTEGER,
            data_extracted JSONB DEFAULT '{}',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            risk_score FLOAT DEFAULT 0.0
        );
        
        -- System Metrics
        CREATE TABLE IF NOT EXISTS system_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            metric_type VARCHAR(50) NOT NULL,
            metadata JSONB DEFAULT '{}',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id);
        CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(type);
        CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
        CREATE INDEX IF NOT EXISTS idx_learning_records_agent ON learning_records(agent_id);
        CREATE INDEX IF NOT EXISTS idx_learning_records_timestamp ON learning_records(timestamp);
        CREATE INDEX IF NOT EXISTS idx_knowledge_entries_tags ON knowledge_entries USING GIN(tags);
        CREATE INDEX IF NOT EXISTS idx_browser_access_agent ON browser_access_logs(agent_id);
        CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, timestamp);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(schema)
        
        logger.info("Database schema created/updated successfully")
    
    async def execute_query(self, query: str, *args) -> List[Dict]:
        """Execute a query and return results"""
        if not self.is_connected:
            raise Exception("Database not connected")
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute a command (INSERT, UPDATE, DELETE)"""
        if not self.is_connected:
            raise Exception("Database not connected")
        
        async with self.pool.acquire() as conn:
            return await conn.execute(command, *args)
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.is_connected = False
            logger.info("Database connection pool closed")

# ============================================================================
# REDIS CACHE AND MESSAGE QUEUE MANAGER
# ============================================================================

class RedisManager:
    """Production-ready Redis manager for caching and messaging"""
    
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.pubsub_client: Optional[aioredis.Redis] = None
        self.is_connected = False
    
    async def initialize(self):
        """Initialize Redis connections"""
        try:
            # Main Redis client for caching
            self.redis_client = await aioredis.from_url(
                config.redis.url,
                max_connections=config.redis.max_connections,
                socket_connect_timeout=config.redis.connection_timeout,
                socket_keepalive=config.redis.socket_keepalive,
                health_check_interval=config.redis.health_check_interval,
                decode_responses=True
            )
            
            # Separate client for pub/sub
            self.pubsub_client = await aioredis.from_url(
                config.redis.url,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            logger.info("Redis connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            return await self.redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(await self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False
    
    async def publish(self, channel: str, message: str) -> int:
        """Publish message to channel"""
        try:
            return await self.pubsub_client.publish(channel, message)
        except Exception as e:
            logger.error(f"Redis PUBLISH error: {e}")
            return 0
    
    async def subscribe(self, *channels) -> aioredis.client.PubSub:
        """Subscribe to channels"""
        pubsub = self.pubsub_client.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub
    
    async def close(self):
        """Close Redis connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.pubsub_client:
            await self.pubsub_client.close()
        self.is_connected = False
        logger.info("Redis connections closed")

# ============================================================================
# MULTI-LLM AI MANAGER
# ============================================================================

class AIProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    DEEPSEEK = "deepseek"

class MultiLLMManager:
    """Production-ready multi-LLM manager with fallback and load balancing"""
    
    def __init__(self):
        self.clients = {}
        self.provider_status = {}
        self.request_counts = {}
        self.last_health_check = {}
        self.client_session = None
    
    async def initialize(self):
        """Initialize AI clients"""
        self.client_session = httpx.AsyncClient(timeout=config.ai.timeout)
        
        # Initialize providers based on available API keys
        if config.ai.openai_api_key:
            self.clients[AIProvider.OPENAI] = self._setup_openai_client()
        
        if config.ai.anthropic_api_key:
            self.clients[AIProvider.ANTHROPIC] = self._setup_anthropic_client()
        
        if config.ai.gemini_api_key:
            self.clients[AIProvider.GEMINI] = self._setup_gemini_client()
        
        if config.ai.groq_api_key:
            self.clients[AIProvider.GROQ] = self._setup_groq_client()
        
        if config.ai.deepseek_api_key:
            self.clients[AIProvider.DEEPSEEK] = self._setup_deepseek_client()
        
        # Initialize status tracking
        for provider in self.clients:
            self.provider_status[provider] = True
            self.request_counts[provider] = 0
            self.last_health_check[provider] = datetime.utcnow()
        
        logger.info(f"Multi-LLM manager initialized with {len(self.clients)} providers")
    
    def _setup_openai_client(self):
        """Setup OpenAI client configuration"""
        return {
            "api_key": config.ai.openai_api_key,
            "base_url": "https://api.openai.com/v1",
            "models": {
                "chat": "gpt-4-turbo-preview",
                "embedding": "text-embedding-3-small",
                "analysis": "gpt-4-turbo-preview"
            }
        }
    
    def _setup_anthropic_client(self):
        """Setup Anthropic client configuration"""
        return {
            "api_key": config.ai.anthropic_api_key,
            "base_url": "https://api.anthropic.com/v1",
            "models": {
                "chat": "claude-3-sonnet-20240229",
                "analysis": "claude-3-sonnet-20240229"
            }
        }
    
    def _setup_gemini_client(self):
        """Setup Gemini client configuration"""
        return {
            "api_key": config.ai.gemini_api_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "models": {
                "chat": "gemini-1.5-pro",
                "analysis": "gemini-1.5-pro"
            }
        }
    
    def _setup_groq_client(self):
        """Setup Groq client configuration"""
        return {
            "api_key": config.ai.groq_api_key,
            "base_url": "https://api.groq.com/openai/v1",
            "models": {
                "chat": "mixtral-8x7b-32768",
                "analysis": "mixtral-8x7b-32768"
            }
        }
    
    def _setup_deepseek_client(self):
        """Setup DeepSeek client configuration"""
        return {
            "api_key": config.ai.deepseek_api_key,
            "base_url": "https://api.deepseek.com/v1",
            "models": {
                "chat": "deepseek-chat",
                "analysis": "deepseek-coder"
            }
        }
    
    async def get_completion(self, 
                           messages: List[Dict],
                           provider: Optional[str] = None,
                           model_type: str = "chat",
                           max_tokens: int = 4000,
                           temperature: float = 0.7) -> Dict:
        """Get completion from AI provider with fallback logic"""
        
        # Determine provider to use
        target_provider = provider or config.ai.default_provider
        
        # Try primary provider first
        if target_provider in self.clients and self.provider_status.get(target_provider, False):
            try:
                result = await self._make_request(target_provider, messages, model_type, max_tokens, temperature)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Primary provider {target_provider} failed: {e}")
                self.provider_status[target_provider] = False
        
        # Try fallback providers
        fallback_providers = [p for p in self.clients.keys() if p != target_provider and self.provider_status.get(p, False)]
        
        for fallback_provider in fallback_providers:
            try:
                result = await self._make_request(fallback_provider, messages, model_type, max_tokens, temperature)
                if result:
                    logger.info(f"Fallback to {fallback_provider} successful")
                    return result
            except Exception as e:
                logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
                self.provider_status[fallback_provider] = False
        
        raise Exception("All AI providers are unavailable")
    
    async def _make_request(self, 
                          provider: str, 
                          messages: List[Dict],
                          model_type: str,
                          max_tokens: int,
                          temperature: float) -> Dict:
        """Make request to specific AI provider"""
        
        client_config = self.clients[provider]
        model = client_config["models"].get(model_type, client_config["models"]["chat"])
        
        # Rate limiting check
        if self.request_counts[provider] >= config.ai.rate_limit_per_minute:
            raise Exception(f"Rate limit exceeded for {provider}")
        
        # Prepare request based on provider
        if provider == AIProvider.OPENAI or provider == AIProvider.GROQ:
            response = await self._openai_compatible_request(
                client_config, model, messages, max_tokens, temperature
            )
        elif provider == AIProvider.ANTHROPIC:
            response = await self._anthropic_request(
                client_config, model, messages, max_tokens, temperature
            )
        elif provider == AIProvider.GEMINI:
            response = await self._gemini_request(
                client_config, model, messages, max_tokens, temperature
            )
        elif provider == AIProvider.DEEPSEEK:
            response = await self._openai_compatible_request(
                client_config, model, messages, max_tokens, temperature
            )
        else:
            raise Exception(f"Unsupported provider: {provider}")
        
        self.request_counts[provider] += 1
        return response
    
    async def _openai_compatible_request(self, config, model, messages, max_tokens, temperature):
        """Make OpenAI-compatible API request"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = await self.client_session.post(
            f"{config['base_url']}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _anthropic_request(self, config, model, messages, max_tokens, temperature):
        """Make Anthropic API request"""
        headers = {
            "x-api-key": config['api_key'],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convert messages to Anthropic format
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        
        payload = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system_message:
            payload["system"] = system_message
        
        response = await self.client_session.post(
            f"{config['base_url']}/messages",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _gemini_request(self, config, model, messages, max_tokens, temperature):
        """Make Gemini API request"""
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        response = await self.client_session.post(
            f"{config['base_url']}/models/{model}:generateContent?key={config['api_key']}",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_embedding(self, text: str, provider: str = AIProvider.OPENAI) -> List[float]:
        """Get text embedding from specified provider"""
        if provider not in self.clients:
            raise Exception(f"Provider {provider} not available")
        
        client_config = self.clients[provider]
        
        if provider == AIProvider.OPENAI:
            headers = {
                "Authorization": f"Bearer {client_config['api_key']}", "Content-Type": "application/json"
            }
            
            payload = {
                "input": text,
                "model": client_config["models"]["embedding"]
            }
            
            response = await self.client_session.post(
                f"{client_config['base_url']}/embeddings",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
        
        raise Exception(f"Embedding not supported for provider: {provider}")
    
    async def close(self):
        """Close HTTP client session"""
        if self.client_session:
            await self.client_session.aclose()
        logger.info("Multi-LLM manager closed")

# ============================================================================
# ADVANCED LEARNING ENGINE
# ============================================================================

class LearningEngine:
    """Advanced AI learning engine with memory management and adaptation"""
    
    def __init__(self, db_manager: DatabaseManager, redis_manager: RedisManager):
        self.db = db_manager
        self.redis = redis_manager
        self.memory_buffer = []
        self.learning_models = {}
        self.adaptation_metrics = {}
        self.is_learning_active = False
    
    async def initialize(self):
        """Initialize learning engine"""
        self.is_learning_active = True
        await self._load_existing_models()
        asyncio.create_task(self._learning_loop())
        logger.info("Learning engine initialized successfully")
    
    async def _load_existing_models(self):
        """Load existing learning models from database"""
        try:
            models_data = await self.db.execute_query(
                "SELECT agent_id, learning_data FROM agents WHERE learning_data IS NOT NULL"
            )
            
            for model_data in models_data:
                agent_id = model_data["agent_id"]
                learning_data = model_data["learning_data"]
                self.learning_models[agent_id] = learning_data
            
            logger.info(f"Loaded {len(self.learning_models)} learning models")
        except Exception as e:
            logger.error(f"Failed to load learning models: {e}")
    
    async def record_interaction(self, 
                                agent_id: str,
                                session_id: str,
                                interaction_type: str,
                                input_data: Dict,
                                output_data: Dict,
                                feedback_score: Optional[float] = None,
                                success_metrics: Optional[Dict] = None):
        """Record learning interaction"""
        try:
            learning_record = {
                "agent_id": agent_id,
                "session_id": session_id,
                "interaction_type": interaction_type,
                "input_data": json.dumps(input_data),
                "output_data": json.dumps(output_data),
                "feedback_score": feedback_score,
                "success_metrics": json.dumps(success_metrics or {}),
                "timestamp": datetime.utcnow().isoformat(),
                "learning_context": json.dumps({
                    "confidence": self._calculate_confidence(output_data),
                    "complexity": self._calculate_complexity(input_data),
                    "performance": self._calculate_performance(success_metrics or {})
                })
            }
            
            # Add to memory buffer
            self.memory_buffer.append(learning_record)
            
            # Store in database
            await self.db.execute_command("""
                INSERT INTO learning_records 
                (agent_id, session_id, interaction_type, input_data, output_data, 
                 feedback_score, success_metrics, learning_context)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7::jsonb, $8::jsonb)
            """, agent_id, session_id, interaction_type, 
                json.dumps(input_data), json.dumps(output_data),
                feedback_score, json.dumps(success_metrics or {}),
                json.dumps(learning_record["learning_context"])
            )
            
            # Trigger learning update if buffer is full
            if len(self.memory_buffer) >= config.learning.batch_size:
                await self._update_learning_models()
            
        except Exception as e:
            logger.error(f"Failed to record learning interaction: {e}")
    
    def _calculate_confidence(self, output_data: Dict) -> float:
        """Calculate confidence score for output"""
        # Simple heuristic - can be enhanced with more sophisticated metrics
        if "confidence" in output_data:
            return float(output_data["confidence"])
        
        # Estimate based on output length and structure
        if "content" in output_data:
            content = str(output_data["content"])
            length_score = min(len(content) / 1000.0, 1.0)
            structure_score = 0.8 if any(marker in content for marker in [".", "!", "?"]) else 0.5
            return (length_score + structure_score) / 2
        
        return 0.5  # Default neutral confidence
    
    def _calculate_complexity(self, input_data: Dict) -> float:
        """Calculate complexity score for input"""
        complexity_indicators = [
            len(str(input_data).split()),  # Word count
            len(input_data.keys()) if isinstance(input_data, dict) else 1,  # Structure complexity
            sum(1 for char in str(input_data) if char in "?!."),  # Punctuation complexity
        ]
        
        # Normalize to 0-1 range
        return min(sum(complexity_indicators) / 100.0, 1.0)
    
    def _calculate_performance(self, success_metrics: Dict) -> float:
        """Calculate performance score from success metrics"""
        if not success_metrics:
            return 0.5
        
        scores = []
        for key, value in success_metrics.items():
            if isinstance(value, (int, float)):
                scores.append(min(max(float(value), 0.0), 1.0))
        
        return sum(scores) / len(scores) if scores else 0.5
    
    async def _update_learning_models(self):
        """Update learning models based on recent interactions"""
        try:
            if not self.memory_buffer:
                return
            
            # Group by agent_id
            agent_interactions = {}
            for record in self.memory_buffer:
                agent_id = record["agent_id"]
                if agent_id not in agent_interactions:
                    agent_interactions[agent_id] = []
                agent_interactions[agent_id].append(record)
            
            # Update models for each agent
            for agent_id, interactions in agent_interactions.items():
                await self._update_agent_model(agent_id, interactions)
            
            # Clear memory buffer
            self.memory_buffer = []
            logger.info(f"Updated learning models for {len(agent_interactions)} agents")
            
        except Exception as e:
            logger.error(f"Failed to update learning models: {e}")
    
    async def _update_agent_model(self, agent_id: str, interactions: List[Dict]):
        """Update learning model for specific agent"""
        try:
            # Get current model
            current_model = self.learning_models.get(agent_id, {
                "version": 1,
                "patterns": {},
                "performance_history": [],
                "adaptation_weights": {},
                "last_updated": datetime.utcnow().isoformat()
            })
            
            # Extract patterns from interactions
            patterns = self._extract_patterns(interactions)
            
            # Update model patterns
            for pattern_type, pattern_data in patterns.items():
                if pattern_type not in current_model["patterns"]:
                    current_model["patterns"][pattern_type] = {}
                
                for pattern_key, pattern_value in pattern_data.items():
                    if pattern_key in current_model["patterns"][pattern_type]:
                        # Weighted average with existing pattern
                        existing_value = current_model["patterns"][pattern_type][pattern_key]
                        weight = config.learning.learning_rate
                        current_model["patterns"][pattern_type][pattern_key] = (
                            existing_value * (1 - weight) + pattern_value * weight
                        )
                    else:
                        current_model["patterns"][pattern_type][pattern_key] = pattern_value
            
            # Update performance history
            avg_performance = sum(float(i.get("learning_context", "{}").get("performance", 0.5)) 
                                for i in interactions) / len(interactions)
            current_model["performance_history"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "performance": avg_performance,
                "interaction_count": len(interactions)
            })
            
            # Keep only recent history
            if len(current_model["performance_history"]) > 100:
                current_model["performance_history"] = current_model["performance_history"][-100:]
            
            # Update model
            current_model["version"] += 1
            current_model["last_updated"] = datetime.utcnow().isoformat()
            self.learning_models[agent_id] = current_model
            
            # Save to database
            await self.db.execute_command("""
                UPDATE agents 
                SET learning_data = $1::jsonb, updated_at = NOW()
                WHERE id = $2
            """, json.dumps(current_model), agent_id)
            
        except Exception as e:
            logger.error(f"Failed to update agent model {agent_id}: {e}")
    
    def _extract_patterns(self, interactions: List[Dict]) -> Dict:
        """Extract learning patterns from interactions"""
        patterns = {
            "input_patterns": {},
            "output_patterns": {},
            "success_patterns": {},
            "timing_patterns": {}
        }
        
        try:
            for interaction in interactions:
                # Input patterns
                input_data = json.loads(interaction["input_data"]) if isinstance(interaction["input_data"], str) else interaction["input_data"]
                input_type = interaction["interaction_type"]
                
                if input_type not in patterns["input_patterns"]:
                    patterns["input_patterns"][input_type] = 0
                patterns["input_patterns"][input_type] += 1
                
                # Output patterns
                output_data = json.loads(interaction["output_data"]) if isinstance(interaction["output_data"], str) else interaction["output_data"]
                output_length = len(str(output_data))
                
                length_category = "short" if output_length < 100 else "medium" if output_length < 500 else "long"
                if length_category not in patterns["output_patterns"]:
                    patterns["output_patterns"][length_category] = 0
                patterns["output_patterns"][length_category] += 1
                
                # Success patterns
                feedback_score = interaction.get("feedback_score", 0.5)
                success_category = "high" if feedback_score > 0.8 else "medium" if feedback_score > 0.5 else "low"
                
                if success_category not in patterns["success_patterns"]:
                    patterns["success_patterns"][success_category] = 0
                patterns["success_patterns"][success_category] += 1
            
            # Normalize patterns
            total_interactions = len(interactions)
            for pattern_type in patterns:
                for pattern_key in patterns[pattern_type]:
                    patterns[pattern_type][pattern_key] /= total_interactions
            
        except Exception as e:
            logger.error(f"Failed to extract patterns: {e}")
        
        return patterns
    
    async def _learning_loop(self):
        """Background learning loop"""
        while self.is_learning_active:
            try:
                await asyncio.sleep(config.learning.update_frequency)
                
                if self.memory_buffer:
                    await self._update_learning_models()
                
                # Cleanup old learning records
                await self._cleanup_old_records()
                
                # Update adaptation metrics
                await self._update_adaptation_metrics()
                
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_old_records(self):
        """Clean up old learning records"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=config.learning.retention_days)
            
            await self.db.execute_command("""
                DELETE FROM learning_records 
                WHERE timestamp < $1
            """, cutoff_date)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
    
    async def _update_adaptation_metrics(self):
        """Update system adaptation metrics"""
        try:
            # Calculate overall system performance
            recent_records = await self.db.execute_query("""
                SELECT agent_id, AVG(feedback_score) as avg_feedback, COUNT(*) as interaction_count
                FROM learning_records 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY agent_id
            """)
            
            total_performance = 0
            total_interactions = 0
            
            for record in recent_records:
                if record["avg_feedback"]:
                    total_performance += float(record["avg_feedback"]) * int(record["interaction_count"])
                    total_interactions += int(record["interaction_count"])
            
            if total_interactions > 0:
                system_performance = total_performance / total_interactions
                
                # Store system metric
                await self.db.execute_command("""
                    INSERT INTO system_metrics (metric_name, metric_value, metric_type, metadata)
                    VALUES ($1, $2, $3, $4::jsonb)
                """, "system_performance", system_performance, "learning", 
                    json.dumps({"total_interactions": total_interactions, "agents_count": len(recent_records)}))
            
        except Exception as e:
            logger.error(f"Failed to update adaptation metrics: {e}")
    
    async def get_recommendations(self, agent_id: str, context: Dict) -> Dict:
        """Get learning-based recommendations for agent"""
        try:
            model = self.learning_models.get(agent_id)
            if not model:
                return {"recommendations": [], "confidence": 0.0}
            
            recommendations = []
            
            # Pattern-based recommendations
            input_patterns = model["patterns"].get("input_patterns", {})
            success_patterns = model["patterns"].get("success_patterns", {})
            
            # Analyze context against learned patterns
            context_type = context.get("type", "general")
            
            if context_type in input_patterns:
                pattern_strength = input_patterns[context_type]
                if pattern_strength > 0.1:  # Significant pattern
                    recommendations.append({
                        "type": "pattern_optimization",
                        "suggestion": f"Optimize for {context_type} interactions",
                        "confidence": pattern_strength,
                        "reasoning": f"This interaction type represents {pattern_strength:.1%} of successful patterns"
                    })
            
            # Performance-based recommendations
            performance_history = model["performance_history"]
            if len(performance_history) >= 2:
                recent_performance = sum(p["performance"] for p in performance_history[-5:]) / min(5, len(performance_history))
                older_performance = sum(p["performance"] for p in performance_history[-10:-5]) / min(5, len(performance_history) - 5)
                
                if recent_performance < older_performance - 0.1:
                    recommendations.append({
                        "type": "performance_alert",
                        "suggestion": "Recent performance decline detected - review recent changes",
                        "confidence": 0.8,
                        "reasoning": f"Performance dropped from {older_performance:.2f} to {recent_performance:.2f}"
                    })
            
            overall_confidence = sum(r["confidence"] for r in recommendations) / len(recommendations) if recommendations else 0.0
            
            return {
                "recommendations": recommendations,
                "confidence": overall_confidence,
                "model_version": model["version"],
                "last_updated": model["last_updated"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get recommendations for {agent_id}: {e}")
            return {"recommendations": [], "confidence": 0.0}
    
    async def stop(self):
        """Stop learning engine"""
        self.is_learning_active = False
        if self.memory_buffer:
            await self._update_learning_models()
        logger.info("Learning engine stopped")

# ============================================================================
# SECURE BROWSER ACCESS MANAGER
# ============================================================================

class SecureBrowserManager:
    """Secure browser access manager with safety controls"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.active_sessions = {}
        self.risk_assessor = RiskAssessmentEngine()
        self.rate_limiter = {}
    
    async def initialize(self):
        """Initialize browser manager"""
        logger.info("Secure browser manager initialized")
    
    async def access_url(self, 
                        agent_id: str, 
                        session_id: str, 
                        url: str, 
                        action: str = "GET",
                        safety_level: str = "standard") -> Dict:
        """Securely access URL with safety checks"""
        
        # Rate limiting check
        if not await self._check_rate_limit(agent_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Risk assessment
        risk_score = await self.risk_assessor.assess_url_risk(url, action)
        
        if risk_score > 0.8:
            await self._log_browser_access(agent_id, session_id, url, action, False, risk_score=risk_score)
            raise HTTPException(status_code=403, detail="URL access blocked due to high risk score")
        
        start_time = time.time()
        
        try:
            # Perform secure web access
            response_data = await self._perform_web_access(url, action, safety_level)
            response_time = int((time.time() - start_time) * 1000)
            
            # Log successful access
            await self._log_browser_access(
                agent_id, session_id, url, action, True, 
                response_time, response_data, risk_score
            )
            
            return {
                "success": True,
                "data": response_data,
                "response_time": response_time,
                "risk_score": risk_score,
                "safety_level": safety_level
            }
            
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            
            # Log failed access
            await self._log_browser_access(
                agent_id, session_id, url, action, False, 
                response_time, {"error": str(e)}, risk_score
            )
            
            return {
                "success": False,
                "error": str(e),
                "response_time": response_time,
                "risk_score": risk_score
            }
    
    async def _check_rate_limit(self, agent_id: str) -> bool:
        """Check if agent is within rate limits"""
        current_time = time.time()
        
        if agent_id not in self.rate_limiter:
            self.rate_limiter[agent_id] = []
        
        # Clean old entries
        self.rate_limiter[agent_id] = [
            timestamp for timestamp in self.rate_limiter[agent_id]
            if current_time - timestamp < 3600  # 1 hour window
        ]
        
        # Check limit
        if len(self.rate_limiter[agent_id]) >= config.browser_max_pages:
            return False
        
        self.rate_limiter[agent_id].append(current_time)
        return True
    
    async def _perform_web_access(self, url: str, action: str, safety_level: str) -> Dict:
        """Perform actual web access with safety controls"""
        
        async with httpx.AsyncClient(
            timeout=config.browser_timeout,
            follow_redirects=True,
            verify=safety_level in ["high", "maximum"]
        ) as client:
            
            headers = {
                "User-Agent": "YMERA-AI-Agent/1.0 (Secure Browser Access)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
            
            if action.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif action.upper() == "POST":
                response = await client.post(url, headers=headers)
            else:
                raise ValueError(f"Unsupported action: {action}")
            
            response.raise_for_status()
            
            # Extract relevant data based on content type
            content_type = response.headers.get("content-type", "").lower()
            
            if "text/html" in content_type:
                # Basic HTML parsing for safety
                content = response.text[:50000]  # Limit content size
                
                # Remove potentially dangerous content
                import re
                safe_content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                safe_content = re.sub(r'<iframe[^>]*>.*?</iframe>', '', safe_content, flags=re.DOTALL | re.IGNORECASE)
                
                return {
                    "content_type": "text/html",
                    "title": self._extract_title(safe_content),
                    "content": safe_content,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": str(response.url)
                }
            
            elif "application/json" in content_type:
                return {
                    "content_type": "application/json",
                    "data": response.json(),
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": str(response.url)
                }
            
            else:
                return {
                    "content_type": content_type,
                    "content": response.text[:10000],  # Limit content size
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": str(response.url)
                }
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content"""
        import re
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()[:200]  # Limit title length
        return "No title found"
    
    async def _log_browser_access(self, 
                                 agent_id: str, 
                                 session_id: str, 
                                 url: str, 
                                 action: str, 
                                 success: bool, 
                                 response_time: int = None, 
                                 data_extracted: Dict = None, 
                                 risk_score: float = 0.0):
        """Log browser access for audit and learning"""
        try:
            await self.db.execute_command("""
                INSERT INTO browser_access_logs 
                (agent_id, session_id, url, action, success, response_time, data_extracted, risk_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
            """, agent_id, session_id, url, action, success, response_time, 
                json.dumps(data_extracted or {}), risk_score)
                
        except Exception as e:
            logger.error(f"Failed to log browser access: {e}")

class RiskAssessmentEngine:
    """Risk assessment engine for URL and content safety"""
    
    def __init__(self):
        self.dangerous_domains = {
            "malware.com", "phishing.net", "scam.org"  # Example dangerous domains
        }
        self.suspicious_patterns = [
            r"javascript:",
            r"data:",
            r"vbscript:",
            r"<script",
            r"eval\(",
            r"document\.cookie"
        ]
    
    async def assess_url_risk(self, url: str, action: str) -> float:
        """Assess risk level of URL access (0.0 = safe, 1.0 = very dangerous)"""
        risk_score = 0.0
        
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            
            # Domain-based risk
            domain = parsed_url.netloc.lower()
            if any(dangerous in domain for dangerous in self.dangerous_domains):
                risk_score += 0.8
            
            # Protocol risk
            if parsed_url.scheme not in ["http", "https"]:
                risk_score += 0.6
            
            # Suspicious patterns in URL
            import re
            for pattern in self.suspicious_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    risk_score += 0.3
                    break
            
            # Action-based risk
            if action.upper() in ["POST", "PUT", "DELETE"]:
                risk_score += 0.2
            
            # IP address instead of domain
            if re.match(r'^\d+\.\d+\.\d+\.\d+', domain):
                risk_score += 0.3
            
            # Unusual ports
            if parsed_url.port and parsed_url.port not in [80, 443, 8080, 8443]:
                risk_score += 0.2
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Risk assessment error: {e}")
            return 0.5  # Default moderate risk

# ============================================================================
# INTELLIGENT AGENT SYSTEM
# ============================================================================

class AgentCapability(str, Enum):
    CODE_ANALYSIS = "code_analysis"
    WEB_SCRAPING = "web_scraping"
    DATA_PROCESSING = "data_processing"
    NATURAL_LANGUAGE = "natural_language"
    API_INTEGRATION = "api_integration"
    LEARNING = "learning"
    SECURITY_AUDIT = "security_audit"

class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class IntelligentAgent:
    """Individual intelligent agent with specific capabilities"""
    
    def __init__(self, 
                 agent_id: str, 
                 name: str, 
                 agent_type: str, 
                 capabilities: List[AgentCapability],
                 db_manager: DatabaseManager,
                 redis_manager: RedisManager,
                 ai_manager: MultiLLMManager,
                 learning_engine: LearningEngine,
                 browser_manager: SecureBrowserManager):
        
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.status = AgentStatus.INACTIVE
        
        # Managers
        self.db = db_manager
        self.redis = redis_manager
        self.ai = ai_manager
        self.learning = learning_engine
        self.browser = browser_manager
        
        # Agent state
        self.context_memory = {}
        self.active_sessions = {}
        self.performance_metrics = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "error_count": 0,
            "last_activity": None
        }
    
    async def initialize(self):
        """Initialize agent"""
        try:
            # Load agent configuration from database
            agent_data = await self.db.execute_query("""
                SELECT configuration, performance_metrics, learning_data 
                FROM agents WHERE id = $1
            """, self.agent_id)
            
            if agent_data:
                data = agent_data[0]
                self.configuration = data.get("configuration", {})
                self.performance_metrics.update(data.get("performance_metrics", {}))
                
            self.status = AgentStatus.ACTIVE
            await self._update_agent_status()
            
            logger.info(f"Agent {self.name} ({self.agent_id}) initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            raise
    
    async def process_task(self, 
                          task: Dict, 
                          session_id: str, 
                          user_id: str,
                          project_id: Optional[str] = None) -> Dict:
        """Process a task using agent capabilities"""
        
        task_start_time = time.time()
        self.status = AgentStatus.BUSY
        await self._update_agent_status()
        
        try:
            # Validate task requirements
            required_capability = task.get("required_capability")
            if required_capability and required_capability not in self.capabilities:
                raise ValueError(f"Agent lacks required capability: {required_capability}")
            
            # Create task context
            task_context = {
                "task_id": str(uuid.uuid4()),
                "session_id": session_id,
                "user_id": user_id,
                "project_id": project_id,
                "task_type": task.get("type", "general"),
                "timestamp": datetime.utcnow().isoformat(),
                "agent_context": self.context_memory.get(session_id, {})
            }
            
            # Get learning recommendations
            recommendations = await self.learning.get_recommendations(
                self.agent_id, task_context
            )
            
            # Process task based on type and capabilities
            result = await self._execute_task(task, task_context, recommendations)
            
            # Calculate performance metrics
            processing_time = time.time() - task_start_time
            success = result.get("success", True)
            
            # Update performance metrics
            await self._update_performance_metrics(processing_time, success)
            
            # Record learning interaction
            await self.learning.record_interaction(
                agent_id=self.agent_id,
                session_id=session_id,
                interaction_type=task.get("type", "general"),
                input_data=task,
                output_data=result,
                feedback_score=result.get("quality_score", 0.8),
                success_metrics={
                    "processing_time": processing_time,
                    "success": success,
                    "complexity": len(str(task)) / 1000.0
                }
            )
            
            # Update context memory
            if session_id not in self.context_memory:
                self.context_memory[session_id] = {}
            
            self.context_memory[session_id].update({
                self.context_memory[session_id].update({
                "last_task": task,
                "last_result": result,
                "task_history": self.context_memory[session_id].get("task_history", [])[-10:] + [task_context],
                "performance_trend": self._calculate_performance_trend(),
                "recommendations_applied": recommendations
            })
            
            # Store task result
            await self._store_task_result(task_context["task_id"], task, result, processing_time)
            
            self.status = AgentStatus.ACTIVE
            await self._update_agent_status()
            
            return {
                **result,
                "task_id": task_context["task_id"],
                "processing_time": processing_time,
                "agent_id": self.agent_id,
                "recommendations": recommendations,
                "context_updated": True
            }
            
        except Exception as e:
            logger.error(f"Task processing error in agent {self.agent_id}: {e}")
            
            # Update error metrics
            self.performance_metrics["error_count"] += 1
            await self._update_performance_metrics(time.time() - task_start_time, False)
            
            self.status = AgentStatus.ERROR
            await self._update_agent_status()
            
            return {
                "success": False,
                "error": str(e),
                "task_id": task_context.get("task_id"),
                "processing_time": time.time() - task_start_time,
                "agent_id": self.agent_id
            }
    
    async def _execute_task(self, task: Dict, context: Dict, recommendations: Dict) -> Dict:
        """Execute task based on agent capabilities"""
        task_type = task.get("type", "general")
        
        try:
            if task_type == "code_analysis" and AgentCapability.CODE_ANALYSIS in self.capabilities:
                return await self._handle_code_analysis(task, context)
            
            elif task_type == "web_scraping" and AgentCapability.WEB_SCRAPING in self.capabilities:
                return await self._handle_web_scraping(task, context)
            
            elif task_type == "data_processing" and AgentCapability.DATA_PROCESSING in self.capabilities:
                return await self._handle_data_processing(task, context)
            
            elif task_type == "natural_language" and AgentCapability.NATURAL_LANGUAGE in self.capabilities:
                return await self._handle_natural_language(task, context, recommendations)
            
            elif task_type == "api_integration" and AgentCapability.API_INTEGRATION in self.capabilities:
                return await self._handle_api_integration(task, context)
            
            elif task_type == "security_audit" and AgentCapability.SECURITY_AUDIT in self.capabilities:
                return await self._handle_security_audit(task, context)
            
            else:
                # Default general processing
                return await self._handle_general_task(task, context, recommendations)
                
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {
                "success": False,
                "error": f"Task execution failed: {str(e)}",
                "task_type": task_type
            }
    
    async def _handle_code_analysis(self, task: Dict, context: Dict) -> Dict:
        """Handle code analysis tasks"""
        code_content = task.get("code", "")
        analysis_type = task.get("analysis_type", "general")
        
        if not code_content:
            return {"success": False, "error": "No code provided for analysis"}
        
        # Use AI for code analysis
        prompt = f"""
        Analyze the following code for {analysis_type} issues:
        
        ```
        {code_content}
        ```
        
        Provide analysis covering:
        1. Code quality and style
        2. Potential bugs and issues
        3. Performance considerations
        4. Security vulnerabilities
        5. Improvement suggestions
        
        Format the response as structured JSON.
        """
        
        ai_response = await self.ai.generate_response(
            prompt=prompt,
            provider="openai",
            model_params={"temperature": 0.1, "max_tokens": 2000}
        )
        
        try:
            import json
            analysis_result = json.loads(ai_response.get("content", "{}"))
        except:
            analysis_result = {"raw_analysis": ai_response.get("content", "")}
        
        return {
            "success": True,
            "analysis": analysis_result,
            "analysis_type": analysis_type,
            "code_length": len(code_content),
            "quality_score": 0.85
        }
    
    async def _handle_web_scraping(self, task: Dict, context: Dict) -> Dict:
        """Handle web scraping tasks"""
        url = task.get("url", "")
        scraping_rules = task.get("rules", {})
        
        if not url:
            return {"success": False, "error": "No URL provided for scraping"}
        
        # Use secure browser manager
        browser_result = await self.browser.access_url(
            agent_id=self.agent_id,
            session_id=context["session_id"],
            url=url,
            safety_level="high"
        )
        
        if not browser_result["success"]:
            return {
                "success": False,
                "error": f"Failed to access URL: {browser_result.get('error', 'Unknown error')}"
            }
        
        # Extract data based on rules
        extracted_data = await self._extract_data_from_content(
            browser_result["data"], 
            scraping_rules
        )
        
        return {
            "success": True,
            "url": url,
            "data": extracted_data,
            "response_time": browser_result.get("response_time"),
            "risk_score": browser_result.get("risk_score"),
            "quality_score": 0.9
        }
    
    async def _handle_data_processing(self, task: Dict, context: Dict) -> Dict:
        """Handle data processing tasks"""
        data = task.get("data", [])
        operation = task.get("operation", "analyze")
        parameters = task.get("parameters", {})
        
        if not data:
            return {"success": False, "error": "No data provided for processing"}
        
        try:
            if operation == "analyze":
                result = await self._analyze_data(data, parameters)
            elif operation == "transform":
                result = await self._transform_data(data, parameters)
            elif operation == "aggregate":
                result = await self._aggregate_data(data, parameters)
            elif operation == "filter":
                result = await self._filter_data(data, parameters)
            else:
                return {"success": False, "error": f"Unsupported operation: {operation}"}
            
            return {
                "success": True,
                "operation": operation,
                "result": result,
                "input_size": len(data),
                "output_size": len(result) if isinstance(result, (list, dict)) else 1,
                "quality_score": 0.88
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Data processing failed: {str(e)}"
            }
    
    async def _handle_natural_language(self, task: Dict, context: Dict, recommendations: Dict) -> Dict:
        """Handle natural language processing tasks"""
        text_input = task.get("input", "")
        task_subtype = task.get("subtype", "general")
        
        if not text_input:
            return {"success": False, "error": "No text input provided"}
        
        # Apply learning recommendations if available
        model_params = {"temperature": 0.7, "max_tokens": 1500}
        if recommendations.get("recommendations"):
            for rec in recommendations["recommendations"]:
                if rec["type"] == "pattern_optimization":
                    model_params["temperature"] = 0.5  # More focused for pattern optimization
        
        # Generate appropriate prompt based on subtype
        if task_subtype == "summarization":
            prompt = f"Provide a concise summary of the following text:\n\n{text_input}"
        elif task_subtype == "analysis":
            prompt = f"Analyze the following text for key themes, sentiment, and insights:\n\n{text_input}"
        elif task_subtype == "question_answering":
            question = task.get("question", "")
            prompt = f"Answer the following question based on the given text:\n\nQuestion: {question}\n\nText: {text_input}"
        else:
            prompt = f"Process the following text according to the request: {task.get('instruction', 'general processing')}\n\n{text_input}"
        
        # Use AI for processing
        ai_response = await self.ai.generate_response(
            prompt=prompt,
            provider="anthropic",  # Use Anthropic for natural language tasks
            model_params=model_params
        )
        
        return {
            "success": True,
            "result": ai_response.get("content", ""),
            "subtype": task_subtype,
            "input_length": len(text_input),
            "recommendations_applied": len(recommendations.get("recommendations", [])),
            "quality_score": 0.87
        }
    
    async def _handle_api_integration(self, task: Dict, context: Dict) -> Dict:
        """Handle API integration tasks"""
        api_endpoint = task.get("endpoint", "")
        method = task.get("method", "GET")
        headers = task.get("headers", {})
        data = task.get("data", {})
        
        if not api_endpoint:
            return {"success": False, "error": "No API endpoint provided"}
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method.upper() == "GET":
                    response = await client.get(api_endpoint, headers=headers, params=data)
                elif method.upper() == "POST":
                    response = await client.post(api_endpoint, headers=headers, json=data)
                elif method.upper() == "PUT":
                    response = await client.put(api_endpoint, headers=headers, json=data)
                elif method.upper() == "DELETE":
                    response = await client.delete(api_endpoint, headers=headers)
                else:
                    return {"success": False, "error": f"Unsupported HTTP method: {method}"}
                
                response.raise_for_status()
                
                # Try to parse JSON response
                try:
                    response_data = response.json()
                except:
                    response_data = response.text
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "data": response_data,
                    "headers": dict(response.headers),
                    "method": method,
                    "endpoint": api_endpoint,
                    "quality_score": 0.92
                }
                
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP error {e.response.status_code}: {e.response.text}",
                "status_code": e.response.status_code
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"API integration failed: {str(e)}"
            }
    
    async def _handle_security_audit(self, task: Dict, context: Dict) -> Dict:
        """Handle security audit tasks"""
        audit_target = task.get("target", "")
        audit_type = task.get("audit_type", "general")
        
        if not audit_target:
            return {"success": False, "error": "No audit target provided"}
        
        audit_results = {
            "target": audit_target,
            "audit_type": audit_type,
            "findings": [],
            "risk_level": "low",
            "recommendations": []
        }
        
        try:
            if audit_type == "code_security":
                # Analyze code for security vulnerabilities
                security_issues = await self._analyze_code_security(audit_target)
                audit_results["findings"].extend(security_issues)
            
            elif audit_type == "url_security":
                # Assess URL security risks
                risk_score = await self.browser.risk_assessor.assess_url_risk(audit_target, "GET")
                audit_results["risk_level"] = "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
                audit_results["findings"].append({
                    "type": "url_risk_assessment",
                    "risk_score": risk_score,
                    "details": f"URL risk assessment completed with score: {risk_score}"
                })
            
            elif audit_type == "configuration_audit":
                # Audit configuration settings
                config_issues = await self._audit_configuration(audit_target)
                audit_results["findings"].extend(config_issues)
            
            # Generate recommendations based on findings
            if audit_results["findings"]:
                high_risk_findings = [f for f in audit_results["findings"] if f.get("severity", "low") == "high"]
                if high_risk_findings:
                    audit_results["risk_level"] = "high"
                    audit_results["recommendations"].append("Immediate action required for high-risk findings")
            
            return {
                "success": True,
                "audit_results": audit_results,
                "findings_count": len(audit_results["findings"]),
                "quality_score": 0.91
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Security audit failed: {str(e)}"
            }
    
    async def _handle_general_task(self, task: Dict, context: Dict, recommendations: Dict) -> Dict:
        """Handle general tasks that don't fit specific categories"""
        task_description = task.get("description", "")
        
        if not task_description:
            return {"success": False, "error": "No task description provided"}
        
        # Use AI for general task processing
        prompt = f"""
        Process the following task request:
        
        {task_description}
        
        Context: {json.dumps(context, indent=2)}
        
        Provide a helpful and comprehensive response.
        """
        
        ai_response = await self.ai.generate_response(
            prompt=prompt,
            provider="openai",
            model_params={"temperature": 0.7, "max_tokens": 1000}
        )
        
        return {
            "success": True,
            "response": ai_response.get("content", ""),
            "context_used": True,
            "recommendations_available": len(recommendations.get("recommendations", [])) > 0,
            "quality_score": 0.8
        }
    
    async def _extract_data_from_content(self, content_data: Dict, rules: Dict) -> Dict:
        """Extract data from web content based on rules"""
        extracted = {}
        
        try:
            if content_data.get("content_type") == "text/html":
                html_content = content_data.get("content", "")
                
                # Simple extraction rules
                if "title" in rules:
                    extracted["title"] = content_data.get("title", "")
                
                if "text_content" in rules:
                    # Remove HTML tags for text extraction
                    import re
                    text_content = re.sub(r'<[^>]+>', ' ', html_content)
                    text_content = ' '.join(text_content.split())  # Normalize whitespace
                    extracted["text_content"] = text_content[:rules.get("max_text_length", 5000)]
                
                if "links" in rules:
                    # Extract links
                    import re
                    links = re.findall(r'href=[\'"]([^\'"]*)[\'"]', html_content)
                    extracted["links"] = links[:rules.get("max_links", 50)]
            
            elif content_data.get("content_type") == "application/json":
                json_data = content_data.get("data", {})
                
                # Extract based on JSON rules
                for key in rules.get("json_fields", []):
                    if key in json_data:
                        extracted[key] = json_data[key]
            
            return extracted
            
        except Exception as e:
            logger.error(f"Data extraction error: {e}")
            return {"error": f"Data extraction failed: {str(e)}"}
    
    async def _analyze_data(self, data: List, parameters: Dict) -> Dict:
        """Analyze data and return insights"""
        try:
            # Basic statistical analysis
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                # Numerical data analysis
                import statistics
                
                analysis = {
                    "count": len(data),
                    "mean": statistics.mean(data),
                    "median": statistics.median(data),
                    "min": min(data),
                    "max": max(data),
                    "std_dev": statistics.stdev(data) if len(data) > 1 else 0
                }
                
                # Additional insights
                analysis["range"] = analysis["max"] - analysis["min"]
                analysis["variance"] = analysis["std_dev"] ** 2
                
                return analysis
            
            else:
                # General data analysis
                analysis = {
                    "count": len(data),
                    "type": type(data).__name__,
                    "sample": data[:5] if len(data) > 5 else data
                }
                
                # Analyze structure if it's a list of dicts
                if data and isinstance(data[0], dict):
                    analysis["keys"] = list(data[0].keys()) if data else []
                    analysis["structure"] = "list_of_dicts"
                
                return analysis
        
        except Exception as e:
            return {"error": f"Data analysis failed: {str(e)}"}
    
    async def _transform_data(self, data: List, parameters: Dict) -> List:
        """Transform data based on parameters"""
        try:
            transformation = parameters.get("transformation", "identity")
            
            if transformation == "normalize" and all(isinstance(x, (int, float)) for x in data):
                # Normalize numerical data
                min_val, max_val = min(data), max(data)
                if max_val != min_val:
                    return [(x - min_val) / (max_val - min_val) for x in data]
                else:
                    return [0.5] * len(data)
            
            elif transformation == "uppercase" and all(isinstance(x, str) for x in data):
                return [x.upper() for x in data]
            
            elif transformation == "filter_positive" and all(isinstance(x, (int, float)) for x in data):
                return [x for x in data if x > 0]
            
            else:
                return data  # Return original if no transformation applies
        
        except Exception as e:
            logger.error(f"Data transformation error: {e}")
            return data
    
    async def _aggregate_data(self, data: List, parameters: Dict) -> Dict:
        """Aggregate data based on parameters"""
        try:
            aggregation = parameters.get("aggregation", "count")
            
            if aggregation == "count":
                return {"count": len(data)}
            
            elif aggregation == "sum" and all(isinstance(x, (int, float)) for x in data):
                return {"sum": sum(data)}
            
            elif aggregation == "group_by" and isinstance(parameters.get("field"), str):
                field = parameters["field"]
                groups = {}
                
                for item in data:
                    if isinstance(item, dict) and field in item:
                        key = item[field]
                        if key not in groups:
                            groups[key] = []
                        groups[key].append(item)
                
                return {"groups": {k: len(v) for k, v in groups.items()}}
            
            else:
                return {"count": len(data)}
        
        except Exception as e:
            logger.error(f"Data aggregation error: {e}")
            return {"error": f"Aggregation failed: {str(e)}"}
    
    async def _filter_data(self, data: List, parameters: Dict) -> List:
        """Filter data based on parameters"""
        try:
            filter_type = parameters.get("filter_type", "none")
            
            if filter_type == "range" and parameters.get("min") is not None and parameters.get("max") is not None:
                min_val, max_val = parameters["min"], parameters["max"]
                return [x for x in data if isinstance(x, (int, float)) and min_val <= x <= max_val]
            
            elif filter_type == "contains" and parameters.get("substring"):
                substring = parameters["substring"]
                return [x for x in data if isinstance(x, str) and substring in x]
            
            elif filter_type == "field_equals" and parameters.get("field") and parameters.get("value") is not None:
                field, value = parameters["field"], parameters["value"]
                return [x for x in data if isinstance(x, dict) and x.get(field) == value]
            
            else:
                return data
        
        except Exception as e:
            logger.error(f"Data filtering error: {e}")
            return data
    
    async def _analyze_code_security(self, code_content: str) -> List[Dict]:
        """Analyze code for security vulnerabilities"""
        findings = []
        
        try:
            # Basic security patterns to look for
            security_patterns = [
                (r"eval\s*\(", "high", "Use of eval() function poses security risks"),
                (r"exec\s*\(", "high", "Use of exec() function poses security risks"),
                (r"input\s*\(", "medium", "Direct user input without validation"),
                (r"os\.system\s*\(", "high", "Direct system command execution"),
                (r"subprocess\.call\s*\(", "medium", "Subprocess execution without validation"),
                (r"pickle\.loads?\s*\(", "high", "Pickle deserialization can be dangerous"),
                (r"sql.*=.*\+", "high", "Potential SQL injection vulnerability"),
                (r"password\s*=\s*[\"'][^\"']*[\"']", "medium", "Hardcoded password detected"),
                (r"api[_-]?key\s*=\s*[\"'][^\"']*[\"']", "medium", "Hardcoded API key detected")
            ]
            
            import re
            for pattern, severity, description in security_patterns:
                matches = re.findall(pattern, code_content, re.IGNORECASE)
                if matches:
                    findings.append({
                        "type": "security_vulnerability",
                        "severity": severity,
                        "description": description,
                        "pattern": pattern,
                        "occurrences": len(matches)
                    })
            
            return findings
        
        except Exception as e:
            logger.error(f"Code security analysis error: {e}")
            return [{"type": "error", "description": f"Security analysis failed: {str(e)}"}]
    
    async def _audit_configuration(self, config_data: str) -> List[Dict]:
        """Audit configuration for security issues"""
        findings = []
        
        try:
            # Parse configuration (assuming JSON format)
            import json
            try:
                config = json.loads(config_data)
            except:
                # If not JSON, treat as plain text
                config = {"raw_config": config_data}
            
            # Check for common configuration issues
            if isinstance(config, dict):
                # Check for insecure settings
                if config.get("debug") is True:
                    findings.append({
                        "type": "configuration_issue",
                        "severity": "medium",
                        "description": "Debug mode is enabled in production",
                        "field": "debug"
                    })
                
                if config.get("ssl_verify") is False:
                    findings.append({
                        "type": "configuration_issue",
                        "severity": "high",
                        "description": "SSL verification is disabled",
                        "field": "ssl_verify"
                    })
                
                # Check for weak encryption settings
                if isinstance(config.get("encryption"), dict):
                    encryption_config = config["encryption"]
                    if encryption_config.get("algorithm") in ["md5", "sha1"]:
                        findings.append({
                            "type": "configuration_issue",
                            "severity": "high",
                            "description": "Weak encryption algorithm detected",
                            "field": "encryption.algorithm"
                        })
            
            return findings
        
        except Exception as e:
            logger.error(f"Configuration audit error: {e}")
            return [{"type": "error", "description": f"Configuration audit failed: {str(e)}"}]
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend based on recent metrics"""
        history = self.performance_metrics.get("performance_history", [])
        
        if len(history) < 2:
            return "insufficient_data"
        
        recent_performance = sum(h.get("performance", 0.5) for h in history[-5:]) / min(5, len(history))
        older_performance = sum(h.get("performance", 0.5) for h in history[-10:-5]) / min(5, len(history) - 5)
        
        if recent_performance > older_performance + 0.1:
            return "improving"
        elif recent_performance < older_performance - 0.1:
            return "declining"
        else:
            return "stable"
    
    async def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update agent performance metrics"""
        try:
            self.performance_metrics["tasks_completed"] += 1
            self.performance_metrics["last_activity"] = datetime.utcnow().isoformat()
            
            if not success:
                self.performance_metrics["error_count"] += 1
            
            # Update success rate
            total_tasks = self.performance_metrics["tasks_completed"]
            successful_tasks = total_tasks - self.performance_metrics["error_count"]
            self.performance_metrics["success_rate"] = successful_tasks / total_tasks if total_tasks > 0 else 0.0
            
            # Update average response time
            current_avg = self.performance_metrics.get("avg_response_time", 0.0)
            self.performance_metrics["avg_response_time"] = (
                (current_avg * (total_tasks - 1) + processing_time) / total_tasks
            )
            
            # Store metrics in database
            await self.db.execute_command("""
                UPDATE agents 
                SET performance_metrics = $1::jsonb, updated_at = NOW()
                WHERE id = $2
            """, json.dumps(self.performance_metrics), self.agent_id)
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    async def _store_task_result(self, task_id: str, task: Dict, result: Dict, processing_time: float):
        """Store task result in database"""
        try:
            await self.db.execute_command("""
                INSERT INTO task_results 
                (task_id, agent_id, task_data, result_data, processing_time, success)
                VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6)
            """, task_id, self.agent_id, json.dumps(task), json.dumps(result), 
                processing_time, result.get("success", True))
                
        except Exception as e:
            logger.error(f"Failed to store task result: {e}")
    
    async def _update_agent_status(self):
        """Update agent status in database and cache"""
        try:
            # Update in database
            await self.db.execute_command("""
                UPDATE agents 
                SET status = $1, updated_at = NOW()
                WHERE id = $2
            """, self.status.value, self.agent_id)
            
            # Update in Redis cache
            await self.redis.set(
                f"agent_status:{self.agent_id}", 
                self.status.value, 
                ex=300  # 5 minute expiry
            )
            
        except Exception as e:
            logger.error(f"Failed to update agent status: {e}")
    
    async def get_status(self) -> Dict:
        """Get current agent status and metrics"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "performance_metrics": self.performance_metrics,
            "active_sessions": len(self.active_sessions),
            "context_memory_size": len(self.context_memory),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown agent gracefully"""
        logger.info(f"Shutting down agent {self.name} ({self.agent_id})")
        
        self.status = AgentStatus.INACTIVE
        await self._update_agent_status()
        
        # Clear context memory
        self.context_memory.clear()
        self.active_sessions.clear()
        
        logger.info(f"Agent {self.name} shutdown complete")


# ============================================================================
# AGENT MANAGER SYSTEM
# ============================================================================

class AgentManager:
    """Manages multiple intelligent agents"""
    
    def __init__(self, 
                 db_manager: DatabaseManager, 
                 redis_manager: RedisManager,
                 ai_manager: MultiLLMManager,
                 learning_engine: LearningEngine,
                 browser_manager: SecureBrowserManager):
        
        self.db = db_manager
        self.redis = redis_manager
        self.ai = ai_manager
        self.learning = learning_engine
        self.browser = browser_manager
        
        self.agents: Dict[str, IntelligentAgent] = {}
        self.agent_pools: Dict[str, List[str]] = {}  # Capability -> [agent_ids]
        self.load_balancer = AgentLoadBalancer()
        
    async def initialize(self):
        """Initialize agent manager and load existing agents"""
        try:
            # Load agents from database
            agent_records = await self.db.execute_query("""
                SELECT id, name, agent_type, capabilities, status, configuration
                FROM agents WHERE status != 'deleted'
            """)
            
            for record in agent_records:
                await self._load_agent(record)
            
            # Initialize load balancer
            self.load_balancer.initialize(list(self.agents.keys()))
            
            logger.info(f"Agent manager initialized with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent manager: {e}")
            raise
    
    async def _load_agent(self, agent_record: Dict):
        """Load individual agent from database record"""
        try:
            agent_id = agent_record["id"] # Parse capabilities
            capabilities = []
            if agent_record.get("capabilities"):
                cap_list = json.loads(agent_record["capabilities"])
                for cap in cap_list:
                    try:
                        capabilities.append(AgentCapability(cap))
                    except ValueError:
                        logger.warning(f"Unknown capability: {cap}")
            
            # Create agent instance
            agent = IntelligentAgent(
                name=agent_record["name"],
                agent_type=agent_record["agent_type"],
                capabilities=capabilities,
                db_manager=self.db,
                redis_manager=self.redis,
                ai_manager=self.ai,
                learning_engine=self.learning,
                browser_manager=self.browser,
                agent_id=agent_id,
                configuration=json.loads(agent_record.get("configuration", "{}"))
            )
            
            # Initialize and store agent
            await agent.initialize()
            self.agents[agent_id] = agent
            
            # Update capability pools
            for capability in capabilities:
                if capability.value not in self.agent_pools:
                    self.agent_pools[capability.value] = []
                self.agent_pools[capability.value].append(agent_id)
            
            logger.info(f"Loaded agent: {agent.name} ({agent_id})")
            
        except Exception as e:
            logger.error(f"Failed to load agent {agent_record.get('id', 'unknown')}: {e}")
    
    async def create_agent(self, 
                          name: str, 
                          agent_type: str, 
                          capabilities: List[AgentCapability],
                          configuration: Dict = None) -> str:
        """Create a new intelligent agent"""
        try:
            if configuration is None:
                configuration = {}
            
            # Generate unique agent ID
            agent_id = f"agent_{uuid.uuid4().hex[:12]}"
            
            # Store in database
            await self.db.execute_command("""
                INSERT INTO agents (id, name, agent_type, capabilities, status, configuration)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb)
            """, agent_id, name, agent_type, 
                json.dumps([cap.value for cap in capabilities]),
                AgentStatus.INACTIVE.value,
                json.dumps(configuration))
            
            # Create agent instance
            agent = IntelligentAgent(
                name=name,
                agent_type=agent_type,
                capabilities=capabilities,
                db_manager=self.db,
                redis_manager=self.redis,
                ai_manager=self.ai,
                learning_engine=self.learning,
                browser_manager=self.browser,
                agent_id=agent_id,
                configuration=configuration
            )
            
            # Initialize and register agent
            await agent.initialize()
            self.agents[agent_id] = agent
            
            # Update capability pools
            for capability in capabilities:
                if capability.value not in self.agent_pools:
                    self.agent_pools[capability.value] = []
                self.agent_pools[capability.value].append(agent_id)
            
            # Update load balancer
            self.load_balancer.add_agent(agent_id)
            
            logger.info(f"Created new agent: {name} ({agent_id})")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to create agent {name}: {e}")
            raise
    
    async def assign_task(self, task: Dict, session_id: str = None) -> Dict:
        """Assign task to the most suitable agent"""
        try:
            task_id = task.get("task_id", f"task_{uuid.uuid4().hex[:8]}")
            task_type = task.get("type", "general")
            required_capabilities = task.get("required_capabilities", [])
            
            # Find suitable agents
            suitable_agents = await self._find_suitable_agents(task_type, required_capabilities)
            
            if not suitable_agents:
                return {
                    "success": False,
                    "error": "No suitable agents available for this task",
                    "task_id": task_id
                }
            
            # Select best agent using load balancer
            selected_agent_id = self.load_balancer.select_agent(suitable_agents)
            selected_agent = self.agents[selected_agent_id]
            
            # Execute task
            result = await selected_agent.process_task(task, session_id)
            
            # Update load balancer metrics
            processing_time = result.get("processing_time", 0)
            success = result.get("success", True)
            self.load_balancer.update_metrics(selected_agent_id, processing_time, success)
            
            return {
                **result,
                "assigned_agent": selected_agent_id,
                "agent_name": selected_agent.name
            }
            
        except Exception as e:
            logger.error(f"Task assignment failed: {e}")
            return {
                "success": False,
                "error": f"Task assignment failed: {str(e)}",
                "task_id": task.get("task_id")
            }
    
    async def _find_suitable_agents(self, task_type: str, required_capabilities: List[str]) -> List[str]:
        """Find agents suitable for the given task"""
        suitable_agents = set()
        
        # If specific capabilities are required
        if required_capabilities:
            for capability in required_capabilities:
                if capability in self.agent_pools:
                    suitable_agents.update(self.agent_pools[capability])
        else:
            # For general tasks, consider agents based on task type
            if task_type == "code_analysis":
                if AgentCapability.CODE_ANALYSIS.value in self.agent_pools:
                    suitable_agents.update(self.agent_pools[AgentCapability.CODE_ANALYSIS.value])
            elif task_type == "web_scraping":
                if AgentCapability.WEB_SCRAPING.value in self.agent_pools:
                    suitable_agents.update(self.agent_pools[AgentCapability.WEB_SCRAPING.value])
            elif task_type == "data_processing":
                if AgentCapability.DATA_PROCESSING.value in self.agent_pools:
                    suitable_agents.update(self.agent_pools[AgentCapability.DATA_PROCESSING.value])
            elif task_type == "natural_language":
                if AgentCapability.NATURAL_LANGUAGE.value in self.agent_pools:
                    suitable_agents.update(self.agent_pools[AgentCapability.NATURAL_LANGUAGE.value])
            else:
                # For general tasks, include all active agents
                suitable_agents = set(self.agents.keys())
        
        # Filter by agent status (only include active or idle agents)
        active_agents = []
        for agent_id in suitable_agents:
            agent = self.agents.get(agent_id)
            if agent and agent.status in [AgentStatus.ACTIVE, AgentStatus.IDLE]:
                active_agents.append(agent_id)
        
        return active_agents
    
    async def get_agent_status(self, agent_id: str) -> Dict:
        """Get status of specific agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": "Agent not found"}
        
        return await agent.get_status()
    
    async def get_all_agents_status(self) -> Dict:
        """Get status of all agents"""
        agents_status = {}
        
        for agent_id, agent in self.agents.items():
            agents_status[agent_id] = await agent.get_status()
        
        return {
            "total_agents": len(self.agents),
            "agents": agents_status,
            "capability_distribution": {
                cap: len(agents) for cap, agents in self.agent_pools.items()
            },
            "load_balancer_metrics": self.load_balancer.get_metrics()
        }
    
    async def shutdown_agent(self, agent_id: str) -> bool:
        """Shutdown specific agent"""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                return False
            
            await agent.shutdown()
            
            # Remove from pools
            for capability_agents in self.agent_pools.values():
                if agent_id in capability_agents:
                    capability_agents.remove(agent_id)
            
            # Remove from load balancer
            self.load_balancer.remove_agent(agent_id)
            
            # Remove from active agents
            del self.agents[agent_id]
            
            logger.info(f"Agent {agent_id} shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown agent {agent_id}: {e}")
            return False
    
    async def shutdown_all_agents(self):
        """Shutdown all agents"""
        logger.info("Shutting down all agents...")
        
        shutdown_tasks = []
        for agent_id in list(self.agents.keys()):
            shutdown_tasks.append(self.shutdown_agent(agent_id))
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.agents.clear()
        self.agent_pools.clear()
        
        logger.info("All agents shut down")
    
    async def scale_agents(self, capability: AgentCapability, target_count: int) -> List[str]:
        """Scale agents for specific capability"""
        current_agents = self.agent_pools.get(capability.value, [])
        current_count = len(current_agents)
        
        created_agents = []
        
        if target_count > current_count:
            # Create additional agents
            agents_to_create = target_count - current_count
            
            for i in range(agents_to_create):
                agent_name = f"{capability.value.replace('_', ' ').title()} Agent {i+1}"
                agent_id = await self.create_agent(
                    name=agent_name,
                    agent_type="specialized",
                    capabilities=[capability],
                    configuration={"auto_scaled": True}
                )
                created_agents.append(agent_id)
        
        elif target_count < current_count:
            # Remove excess agents (prioritize auto-scaled ones)
            agents_to_remove = current_count - target_count
            
            # Sort agents by creation time, auto-scaled first
            removal_candidates = []
            for agent_id in current_agents:
                agent = self.agents[agent_id]
                if hasattr(agent, 'configuration') and agent.configuration.get("auto_scaled"):
                    removal_candidates.insert(0, agent_id)  # Auto-scaled agents first
                else:
                    removal_candidates.append(agent_id)
            
            for i in range(min(agents_to_remove, len(removal_candidates))):
                await self.shutdown_agent(removal_candidates[i])
        
        return created_agents
    
    async def get_performance_report(self) -> Dict:
        """Generate performance report for all agents"""
        try:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_agents": len(self.agents),
                    "active_agents": 0,
                    "total_tasks_completed": 0,
                    "average_success_rate": 0.0,
                    "average_response_time": 0.0
                },
                "agents": {},
                "capability_performance": {},
                "recommendations": []
            }
            
            total_success_rates = []
            total_response_times = []
            
            # Collect agent metrics
            for agent_id, agent in self.agents.items():
                status = await agent.get_status()
                report["agents"][agent_id] = status
                
                if status["status"] == "active":
                    report["summary"]["active_agents"] += 1
                
                metrics = status["performance_metrics"]
                report["summary"]["total_tasks_completed"] += metrics.get("tasks_completed", 0)
                
                if metrics.get("success_rate") is not None:
                    total_success_rates.append(metrics["success_rate"])
                
                if metrics.get("avg_response_time") is not None:
                    total_response_times.append(metrics["avg_response_time"])
            
            # Calculate averages
            if total_success_rates:
                report["summary"]["average_success_rate"] = sum(total_success_rates) / len(total_success_rates)
            
            if total_response_times:
                report["summary"]["average_response_time"] = sum(total_response_times) / len(total_response_times)
            
            # Capability performance analysis
            for capability, agent_ids in self.agent_pools.items():
                cap_metrics = {
                    "agent_count": len(agent_ids),
                    "total_tasks": 0,
                    "average_success_rate": 0.0,
                    "average_response_time": 0.0
                }
                
                cap_success_rates = []
                cap_response_times = []
                
                for agent_id in agent_ids:
                    if agent_id in self.agents:
                        status = await self.agents[agent_id].get_status()
                        metrics = status["performance_metrics"]
                        
                        cap_metrics["total_tasks"] += metrics.get("tasks_completed", 0)
                        
                        if metrics.get("success_rate") is not None:
                            cap_success_rates.append(metrics["success_rate"])
                        
                        if metrics.get("avg_response_time") is not None:
                            cap_response_times.append(metrics["avg_response_time"])
                
                if cap_success_rates:
                    cap_metrics["average_success_rate"] = sum(cap_success_rates) / len(cap_success_rates)
                
                if cap_response_times:
                    cap_metrics["average_response_time"] = sum(cap_response_times) / len(cap_response_times)
                
                report["capability_performance"][capability] = cap_metrics
            
            # Generate recommendations
            recommendations = []
            
            # Check for underperforming capabilities
            for capability, metrics in report["capability_performance"].items():
                if metrics["average_success_rate"] < 0.8:
                    recommendations.append({
                        "type": "performance_improvement",
                        "priority": "high",
                        "message": f"Capability {capability} has low success rate ({metrics['average_success_rate']:.2%})"
                    })
                
                if metrics["average_response_time"] > 30.0:
                    recommendations.append({
                        "type": "performance_optimization",
                        "priority": "medium",
                        "message": f"Capability {capability} has high response time ({metrics['average_response_time']:.2f}s)"
                    })
            
            # Check for scaling needs
            load_metrics = self.load_balancer.get_metrics()
            for agent_id, agent_metrics in load_metrics.items():
                if agent_metrics.get("load", 0) > 0.8:
                    recommendations.append({
                        "type": "scaling_recommendation",
                        "priority": "medium",
                        "message": f"Agent {agent_id} is under high load, consider scaling"
                    })
            
            report["recommendations"] = recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"error": f"Failed to generate report: {str(e)}"}


# ============================================================================
# AGENT LOAD BALANCER
# ============================================================================

class AgentLoadBalancer:
    """Manages load balancing across agents"""
    
    def __init__(self):
        self.agent_metrics: Dict[str, Dict] = {}
        self.last_assigned: Dict[str, float] = {}
        
    def initialize(self, agent_ids: List[str]):
        """Initialize load balancer with agent IDs"""
        for agent_id in agent_ids:
            self.agent_metrics[agent_id] = {
                "load": 0.0,
                "response_time": 0.0,
                "success_rate": 1.0,
                "task_count": 0,
                "last_updated": time.time()
            }
            self.last_assigned[agent_id] = 0.0
    
    def add_agent(self, agent_id: str):
        """Add new agent to load balancer"""
        self.agent_metrics[agent_id] = {
            "load": 0.0,
            "response_time": 0.0,
            "success_rate": 1.0,
            "task_count": 0,
            "last_updated": time.time()
        }
        self.last_assigned[agent_id] = 0.0
    
    def remove_agent(self, agent_id: str):
        """Remove agent from load balancer"""
        self.agent_metrics.pop(agent_id, None)
        self.last_assigned.pop(agent_id, None)
    
    def select_agent(self, available_agents: List[str]) -> str:
        """Select best agent for task assignment"""
        if not available_agents:
            raise ValueError("No agents available for selection")
        
        if len(available_agents) == 1:
            return available_agents[0]
        
        # Calculate scores for each available agent
        agent_scores = {}
        current_time = time.time()
        
        for agent_id in available_agents:
            if agent_id not in self.agent_metrics:
                # New agent, give it priority
                agent_scores[agent_id] = 1.0
                continue
            
            metrics = self.agent_metrics[agent_id]
            
            # Load score (lower is better)
            load_score = 1.0 - min(metrics["load"], 1.0)
            
            # Response time score (faster is better)
            max_response_time = 60.0  # 60 seconds max
            response_time_score = 1.0 - min(metrics["response_time"] / max_response_time, 1.0)
            
            # Success rate score
            success_score = metrics["success_rate"]
            
            # Time since last assignment (round-robin factor)
            time_since_last = current_time - self.last_assigned.get(agent_id, 0)
            time_score = min(time_since_last / 300.0, 1.0)  # 5 minutes max
            
            # Weighted combination
            agent_scores[agent_id] = (
                load_score * 0.4 +
                response_time_score * 0.3 +
                success_score * 0.2 +
                time_score * 0.1
            )
        
        # Select agent with highest score
        selected_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
        self.last_assigned[selected_agent] = current_time
        
        return selected_agent
    
    def update_metrics(self, agent_id: str, response_time: float, success: bool):
        """Update agent metrics after task completion"""
        if agent_id not in self.agent_metrics:
            self.add_agent(agent_id)
        
        metrics = self.agent_metrics[agent_id]
        
        # Update response time (exponential moving average)
        alpha = 0.3
        metrics["response_time"] = (
            alpha * response_time + (1 - alpha) * metrics["response_time"]
        )
        
        # Update success rate
        metrics["task_count"] += 1
        current_success_count = metrics["success_rate"] * (metrics["task_count"] - 1)
        if success:
            current_success_count += 1
        metrics["success_rate"] = current_success_count / metrics["task_count"]
        
        # Update load (simple task count based for now)
        metrics["load"] = min(metrics["task_count"] / 10.0, 1.0)  # Scale to 0-1
        
        metrics["last_updated"] = time.time()
    
    def get_metrics(self) -> Dict:
        """Get current load balancer metrics"""
        return {
            "total_agents": len(self.agent_metrics),
            "agent_metrics": self.agent_metrics.copy(),
            "last_assigned": self.last_assigned.copy()
        }


# ============================================================================
# EXAMPLE USAGE AND INITIALIZATION
# ============================================================================

async def create_sample_agents(agent_manager: AgentManager) -> List[str]:
    """Create sample agents for demonstration"""
    sample_agents = []
    
    try:
        # Code analysis specialist
        code_agent_id = await agent_manager.create_agent(
            name="Code Analyzer Pro",
            agent_type="specialist",
            capabilities=[AgentCapability.CODE_ANALYSIS],
            configuration={
                "specialization": "code_quality",
                "max_code_size": 100000,
                "analysis_depth": "comprehensive"
            }
        )
        sample_agents.append(code_agent_id)
        
        # Web scraping specialist
        web_agent_id = await agent_manager.create_agent(
            name="Web Scraper Elite",
            agent_type="specialist",
            capabilities=[AgentCapability.WEB_SCRAPING],
            configuration={
                "rate_limit": 1.0,
                "respect_robots": True,
                "max_pages": 100
            }
        )
        sample_agents.append(web_agent_id)
        
        # Data processing specialist
        data_agent_id = await agent_manager.create_agent(
            name="Data Processor Max",
            agent_type="specialist",
            capabilities=[AgentCapability.DATA_PROCESSING],
            configuration={
                "max_data_size": 1000000,
                "processing_timeout": 300,
                "memory_limit": "2GB"
            }
        )
        sample_agents.append(data_agent_id)
        
        # Multi-capable generalist
        general_agent_id = await agent_manager.create_agent(
            name="General Assistant Plus",
            agent_type="generalist",
            capabilities=[
                AgentCapability.NATURAL_LANGUAGE,
                AgentCapability.API_INTEGRATION,
                AgentCapability.DATA_PROCESSING
            ],
            configuration={
                "versatility": "high",
                "context_memory": 50,
                "learning_enabled": True
            }
        )
        sample_agents.append(general_agent_id)
        
        # Security specialist
        security_agent_id = await agent_manager.create_agent(
            name="Security Auditor Pro",
            agent_type="specialist",
            capabilities=[
                AgentCapability.SECURITY_AUDIT,
                AgentCapability.CODE_ANALYSIS
            ],
            configuration={
                "security_level": "paranoid",
                "audit_depth": "comprehensive",
                "compliance_standards": ["OWASP", "NIST"]
            }
        )
        sample_agents.append(security_agent_id)
        
        logger.info(f"Created {len(sample_agents)} sample agents")
        return sample_agents
        
    except Exception as e:
        logger.error(f"Failed to create sample agents: {e}")
        return sample_agents


# Example usage function
async def demonstrate_agent_system():
    """Demonstrate the intelligent agent system"""
    try:
        # Initialize required components (these would be initialized elsewhere)
        # db_manager = DatabaseManager(...)
        # redis_manager = RedisManager(...)
        # ai_manager = MultiLLMManager(...)
        # learning_engine = LearningEngine(...)
        # browser_manager = SecureBrowserManager(...)
        
        # Initialize agent manager
        # agent_manager = AgentManager(db_manager, redis_manager, ai_manager, learning_engine, browser_manager)
        # await agent_manager.initialize()
        
        # Create sample agents
        # sample_agents = await create_sample_agents(agent_manager)
        
        # Example task assignments
        tasks = [
            {
                "task_id": "code_review_001",
                "type": "code_analysis",
                "code": "def unsafe_function(user_input): return eval(user_input)",
                "analysis_type": "security"
            },
            {
                "task_id": "web_scrape_001", 
                "type": "web_scraping",
                "url": "https://example.com",
                "rules": {"title": True, "text_content": True}
            },
            {
                "task_id": "data_process_001",
                "type": "data_processing",
                "data": [1, 2, 3, 4, 5, 10, 15, 20],
                "operation": "analyze"
            },
            {
                "task_id": "nl_process_001",
                "type": "natural_language",
                "subtype": "summarization",
                "input": "This is a long document that needs to be summarized for easier reading and understanding."
            }
        ]
        
        # Process tasks
        # for task in tasks:
        #     result = await agent_manager.assign_task(task, session_id="demo_session")
        #     print(f"Task {task['task_id']} result: {result}")
        
        # Get performance report
        # report = await agent_manager.get_performance_report()
        # print(f"Performance Report: {json.dumps(report, indent=2)}")
        
        print("Intelligent Agent System demonstration complete")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")


if __name__ == "__main__":
    asyncio.run(demonstrate_agent_system())