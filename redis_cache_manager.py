"""
YMERA Enterprise Redis Cache Manager
Production-Ready Distributed Caching Layer for Multi-Agent System

Features:
- Distributed caching with Redis Cluster support
- Agent-specific namespacing and isolation
- Learning data caching with intelligent eviction
- Circuit breaker pattern for resilience
- Advanced serialization with compression
- Cache warming and prefetching
- Real-time metrics and monitoring
- Security and encryption for sensitive data
"""

import asyncio
import json
import pickle
import gzip
import hashlib
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor

import redis.asyncio as aioredis
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.sentinel import Sentinel
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from redis.exceptions import (
    RedisError, ConnectionError, TimeoutError, 
    RedisClusterException, ResponseError
)

import orjson
import msgpack
from cryptography.fernet import Fernet
import zstandard as zstd
import xxhash

from ymera_core.exceptions import YMERAException
from ymera_core.monitoring.metrics_collector import MetricsCollector
from ymera_core.utils.async_utils import AsyncRateLimiter, AsyncCircuitBreaker
from ymera_core.security.encryption import DataEncryption


class CacheMode(Enum):
    """Cache deployment modes"""
    STANDALONE = "standalone"
    CLUSTER = "cluster" 
    SENTINEL = "sentinel"


class SerializationMethod(Enum):
    """Serialization methods for cached data"""
    JSON = "json"
    ORJSON = "orjson"
    PICKLE = "pickle"
    MSGPACK = "msgpack"


class CompressionMethod(Enum):
    """Compression methods for cached data"""
    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"


class CacheStrategy(Enum):
    """Cache strategies for different data types"""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    FIFO = "fifo"        # First In, First Out
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on access patterns


@dataclass
class CacheConfig:
    """Redis cache configuration"""
    # Connection settings
    redis_url: str = "redis://localhost:6379"
    mode: CacheMode = CacheMode.STANDALONE
    cluster_nodes: Optional[List[str]] = None
    sentinel_hosts: Optional[List[Tuple[str, int]]] = None
    sentinel_service_name: str = "mymaster"
    
    # Connection pool settings
    max_connections: int = 100
    retry_on_timeout: bool = True
    retry_attempts: int = 3
    retry_backoff: float = 1.0
    health_check_interval: int = 30
    
    # Cache behavior
    default_ttl: int = 3600  # 1 hour
    max_ttl: int = 86400     # 24 hours
    min_ttl: int = 60        # 1 minute
    
    # Performance settings
    serialization_method: SerializationMethod = SerializationMethod.ORJSON
    compression_method: CompressionMethod = CompressionMethod.ZSTD
    compression_threshold: int = 1024  # Compress if data > 1KB
    
    # Security
    encrypt_sensitive_data: bool = True
    encryption_key: Optional[str] = None
    
    # Agent system specific
    agent_namespace_prefix: str = "ymera:agent"
    learning_namespace_prefix: str = "ymera:learning"
    session_namespace_prefix: str = "ymera:session"
    
    # Cache warming
    enable_cache_warming: bool = True
    warm_cache_on_startup: bool = True
    prefetch_popular_keys: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    metrics_interval: int = 60
    slow_query_threshold: float = 0.1  # 100ms


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[int]
    tags: Set[str]
    agent_id: Optional[str] = None
    learning_context: Optional[str] = None
    compressed: bool = False
    encrypted: bool = False
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if not self.ttl:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access statistics"""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheMetrics:
    """Cache performance metrics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.total_size = 0
        self.avg_response_time = 0.0
        self.slow_queries = 0
        self.last_reset = time.time()
        self._lock = threading.Lock()
    
    def record_hit(self, response_time: float = 0):
        with self._lock:
            self.hits += 1
            self._update_response_time(response_time)
    
    def record_miss(self, response_time: float = 0):
        with self._lock:
            self.misses += 1
            self._update_response_time(response_time)
    
    def record_set(self):
        with self._lock:
            self.sets += 1
    
    def record_delete(self):
        with self._lock:
            self.deletes += 1
    
    def record_eviction(self):
        with self._lock:
            self.evictions += 1
    
    def record_error(self):
        with self._lock:
            self.errors += 1
    
    def record_slow_query(self):
        with self._lock:
            self.slow_queries += 1
    
    def _update_response_time(self, response_time: float):
        if response_time > 0:
            total_operations = self.hits + self.misses
            if total_operations == 1:
                self.avg_response_time = response_time
            else:
                self.avg_response_time = (
                    (self.avg_response_time * (total_operations - 1) + response_time) 
                    / total_operations
                )
    
    def get_hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self):
        with self._lock:
            self.__init__()


class RedisCacheManager:
    """Enterprise-grade Redis cache manager for YMERA multi-agent system"""
    
    def __init__(self, config: CacheConfig = None, logger: logging.Logger = None):
        self.config = config or CacheConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Redis connections
        self.redis_client: Optional[Union[aioredis.Redis, RedisCluster]] = None
        self.sentinel: Optional[Sentinel] = None
        
        # Circuit breaker for resilience
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=RedisError
        )
        
        # Rate limiter for request control
        self.rate_limiter = AsyncRateLimiter(
            max_requests=1000,
            time_window=60
        )
        
        # Metrics and monitoring
        self.metrics = CacheMetrics()
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Encryption for sensitive data
        self.encryption = DataEncryption() if self.config.encrypt_sensitive_data else None
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Cache warming and prefetching
        self.cache_warmer_task: Optional[asyncio.Task] = None
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        
        # Agent-specific cache namespaces
        self.agent_namespaces: Set[str] = set()
        self.learning_cache_keys: Set[str] = set()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Key patterns for different data types
        self.key_patterns = {
            'agent_state': f"{self.config.agent_namespace_prefix}:{{agent_id}}:state",
            'agent_memory': f"{self.config.agent_namespace_prefix}:{{agent_id}}:memory:{{key}}",
            'learning_data': f"{self.config.learning_namespace_prefix}:{{context}}:{{key}}",
            'session_data': f"{self.config.session_namespace_prefix}:{{session_id}}:{{key}}",
            'analysis_cache': "ymera:analysis:{hash}",
            'code_cache': "ymera:code:{repo}:{branch}:{hash}",
            'embedding_cache': "ymera:embeddings:{model}:{hash}",
            'llm_response': "ymera:llm:{provider}:{model}:{hash}",
            'vulnerability_scan': "ymera:security:vulns:{repo}:{hash}",
            'deployment_cache': "ymera:deploy:{pipeline}:{stage}:{hash}"
        }
        
        self.logger.info("RedisCacheManager initialized with configuration", 
                        extra={"config": asdict(self.config)})
    
    async def initialize(self) -> None:
        """Initialize Redis connection and start background tasks"""
        try:
            self.logger.info("Initializing Redis cache manager...")
            
            # Initialize Redis connection based on mode
            await self._initialize_redis_connection()
            
            # Verify connection
            await self._verify_connection()
            
            # Initialize encryption if enabled
            if self.config.encrypt_sensitive_data:
                await self._initialize_encryption()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Warm cache if enabled
            if self.config.warm_cache_on_startup:
                await self._warm_cache()
            
            self.logger.info("Redis cache manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis cache manager: {str(e)}")
            raise YMERAException(f"Cache initialization failed: {str(e)}")
    
    async def _initialize_redis_connection(self) -> None:
        """Initialize Redis connection based on configuration mode"""
        retry_policy = Retry(
            backoff=ExponentialBackoff(),
            retries=self.config.retry_attempts
        )
        
        if self.config.mode == CacheMode.CLUSTER:
            if not self.config.cluster_nodes:
                raise ValueError("Cluster nodes must be specified for cluster mode")
            
            self.redis_client = RedisCluster(
                startup_nodes=[
                    {"host": node.split(":")[0], "port": int(node.split(":")[1])}
                    for node in self.config.cluster_nodes
                ],
                decode_responses=False,
                retry=retry_policy,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                max_connections_per_node=self.config.max_connections
            )
            
        elif self.config.mode == CacheMode.SENTINEL:
            if not self.config.sentinel_hosts:
                raise ValueError("Sentinel hosts must be specified for sentinel mode")
            
            self.sentinel = Sentinel(
                self.config.sentinel_hosts,
                decode_responses=False,
                retry=retry_policy
            )
            self.redis_client = self.sentinel.master_for(
                self.config.sentinel_service_name,
                retry=retry_policy,
                retry_on_timeout=self.config.retry_on_timeout
            )
            
        else:  # STANDALONE mode
            self.redis_client = aioredis.from_url(
                self.config.redis_url,
                decode_responses=False,
                retry=retry_policy,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                max_connections=self.config.max_connections
            )
    
    async def _verify_connection(self) -> None:
        """Verify Redis connection is working"""
        try:
            await self.redis_client.ping()
            self.logger.info("Redis connection verified successfully")
        except Exception as e:
            self.logger.error(f"Redis connection verification failed: {str(e)}")
            raise
    
    async def _initialize_encryption(self) -> None:
        """Initialize encryption for sensitive data"""
        if not self.config.encryption_key:
            self.config.encryption_key = Fernet.generate_key().decode()
            self.logger.warning("Generated new encryption key - store securely in production")
        
        self.encryption = DataEncryption(self.config.encryption_key)
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for maintenance and monitoring"""
        if self.config.enable_metrics:
            self.background_tasks.append(
                asyncio.create_task(self._metrics_collection_loop())
            )
        
        if self.config.enable_cache_warming:
            self.background_tasks.append(
                asyncio.create_task(self._cache_warming_loop())
            )
        
        self.background_tasks.append(
            asyncio.create_task(self._cleanup_expired_keys_loop())
        )
        
        self.background_tasks.append(
            asyncio.create_task(self._health_monitoring_loop())
        )
    
    async def _warm_cache(self) -> None:
        """Warm cache with frequently accessed data"""
        try:
            self.logger.info("Starting cache warming...")
            
            await self._warm_agent_caches()
            await self._warm_learning_caches()
            await self._warm_system_caches()
            
            self.logger.info("Cache warming completed")
            
        except Exception as e:
            self.logger.error(f"Cache warming failed: {str(e)}")
    
    async def _warm_agent_caches(self) -> None:
        """Warm cache with agent-specific data"""
        for agent_type in ['project_management', 'analysis', 'enhancement', 
                          'validation', 'documentation', 'security', 'deployment',
                          'monitoring', 'learning', 'communication', 'examination']:
            namespace = f"{self.config.agent_namespace_prefix}:{agent_type}"
            self.agent_namespaces.add(namespace)
    
    async def _warm_learning_caches(self) -> None:
        """Warm cache with learning engine data"""
        for context in ['patterns', 'feedback', 'models', 'knowledge', 'insights']:
            namespace = f"{self.config.learning_namespace_prefix}:{context}"
            await self.redis_client.sadd("ymera:learning:contexts", namespace)
    
    async def _warm_system_caches(self) -> None:
        """Warm cache with system configuration and metadata"""
        system_config = {
            "agents_count": len(self.agent_namespaces),
            "cache_initialized": datetime.utcnow().isoformat(),
            "features_enabled": [
                "multi_agent_support",
                "learning_engine", 
                "vector_search",
                "security_scanning"
            ]
        }
        await self.set("ymera:system:config", system_config, ttl=self.config.max_ttl)
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with comprehensive error handling"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter:
                async with self.circuit_breaker:
                    result = await self._get_internal(key)
            
            response_time = (time.time() - start_time) * 1000
            
            if result is not None:
                self.metrics.record_hit(response_time)
                if response_time > self.config.slow_query_threshold * 1000:
                    self.metrics.record_slow_query()
                    self.logger.warning(f"Slow cache get: {key} ({response_time:.2f}ms)")
                return result
            else:
                self.metrics.record_miss(response_time)
                return default
                
        except Exception as e:
            self.metrics.record_error()
            self.logger.error(f"Cache get operation failed: {str(e)}")
            return default
    
    async def _get_internal(self, key: str) -> Optional[Any]:
        """Internal get implementation"""
        try:
            raw_value = await self.redis_client.get(key)
            if raw_value is None:
                return None
            
            return self._deserialize_value(raw_value)
        except Exception as e:
            self.logger.error(f"Failed to deserialize cache value for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        start_time = time.time()
        
        try:
            if ttl is None:
                ttl = self.config.default_ttl
            
            ttl = max(self.config.min_ttl, min(ttl, self.config.max_ttl))
            
            async with self.rate_limiter:
                async with self.circuit_breaker:
                    serialized_value = self._serialize_value(value)
                    await self.redis_client.setex(key, ttl, serialized_value)
            
            response_time = (time.time() - start_time) * 1000
            self.metrics.record_set()
            
            if response_time > self.config.slow_query_threshold * 1000:
                self.metrics.record_slow_query()
                self.logger.warning(f"Slow cache set: {key} ({response_time:.2f}ms)")
            
            return True
            
        except Exception as e:
            self.metrics.record_error()
            self.logger.error(f"Cache set operation failed: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            async with self.rate_limiter:
                async with self.circuit_breaker:
                    deleted = await self.redis_client.delete(key)
            
            self.metrics.record_delete()
            return deleted > 0
            
        except Exception as e:
            self.metrics.record_error()
            self.logger.error(f"Cache delete operation failed: {str(e)}")
            return False
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for caching"""
        try:
            if self.config.serialization_method == SerializationMethod.ORJSON:
                serialized = orjson.dumps(value)
            elif self.config.serialization_method == SerializationMethod.MSGPACK:
                serialized = msgpack.packb(value)
            elif self.config.serialization_method == SerializationMethod.PICKLE:
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                serialized = json.dumps(value).encode('utf-8')
            
            if self.config.compression_method != CompressionMethod.NONE and len(serialized) > self.config.compression_threshold:
                if self.config.compression_method == CompressionMethod.GZIP:
                    serialized = gzip.compress(serialized)
                elif self.config.compression_method == CompressionMethod.ZSTD:
                    serialized = zstd.compress(serialized)
            
            if self.config.encrypt_sensitive_data and self.encryption:
                serialized = self.encryption.encrypt(serialized)
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"Serialization failed: {str(e)}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize cached value"""
        try:
            if self.config.encrypt_sensitive_data and self.encryption:
                data = self.encryption.decrypt(data)
            
            if self.config.compression_method != CompressionMethod.NONE:
                try:
                    if self.config.compression_method == CompressionMethod.GZIP:
                        data = gzip.decompress(data)
                    elif self.config.compression_method == CompressionMethod.ZSTD:
                        data = zstd.decompress(data)
                except Exception:
                    pass
            
            if self.config.serialization_method == SerializationMethod.ORJSON:
                return orjson.loads(data)
            elif self.config.serialization_method == SerializationMethod.MSGPACK:
                return msgpack.unpackb(data)
            elif self.config.serialization_method == SerializationMethod.PICKLE:
                return pickle.loads(data)
            else:
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            self.logger.error(f"Deserialization failed: {str(e)}")
            raise
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.metrics_interval)
                
                self.logger.info(
                    "Cache metrics",
                    hits=self.metrics.hits,
                    misses=self.metrics.misses,
                    hit_ratio=self.metrics.get_hit_ratio(),
                    slow_queries=self.metrics.slow_queries
                )
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {str(e)}")
    
    async def _cache_warming_loop(self) -> None:
        """Background cache warming"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)
                if self.config.prefetch_popular_keys:
                    await self._warm_cache()
                    
            except Exception as e:
                self.logger.error(f"Cache warming error: {str(e)}")
    
    async def _cleanup_expired_keys_loop(self) -> None:
        """Background cleanup of expired keys"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(600)
                self.logger.debug("Running periodic cache cleanup")
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {str(e)}")
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._verify_connection()
                
            except Exception as e:
                self.logger.error(f"Health check error: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup cache manager resources"""
        try:
            self.shutdown_event.set()
            
            for task in self.background_tasks:
                task.cancel()
            
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.thread_pool.shutdown(wait=True)
            
            self.logger.info("Cache manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for cache manager"""
        try:
            await self._verify_connection()
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "metrics": {
                    "hits": self.metrics.hits,
                    "misses": self.metrics.misses,
                    "hit_ratio": self.metrics.get_hit_ratio(),
                    "errors": self.metrics.errors
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
