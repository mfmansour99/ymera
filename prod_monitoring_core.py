"""
YMERA Enterprise Monitoring System - Production Ready
Unified health monitoring, metrics collection, and alerting
Version: 5.0.0 - Production
"""

import asyncio
import json
import time
import logging
import psutil
import redis.asyncio as aioredis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import statistics
import uuid
import socket
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class HealthStatus(str, Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """System component types"""
    DATABASE = "database"
    CACHE = "cache"
    AGENT = "agent"
    LEARNING_ENGINE = "learning_engine"
    AI_SERVICE = "ai_service"
    SYSTEM = "system"
    API = "api"


class MetricType(str, Enum):
    """Metric types for classification"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'status': self.status.value,
            'threshold_warning': self.threshold_warning,
            'threshold_critical': self.threshold_critical,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message
        }


@dataclass
class ComponentHealth:
    """Component health status"""
    component_id: str
    component_type: ComponentType
    status: HealthStatus
    metrics: List[HealthMetric]
    last_check: datetime
    uptime_percent: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'status': self.status.value,
            'metrics': [m.to_dict() for m in self.metrics],
            'last_check': self.last_check.isoformat(),
            'uptime_percent': self.uptime_percent,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class SystemHealth:
    """Overall system health"""
    overall_status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    components: Dict[str, ComponentHealth]
    system_metrics: List[HealthMetric]
    active_alerts: List[Dict[str, Any]]
    
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_status': self.overall_status.value,
            'timestamp': self.timestamp.isoformat(),
            'uptime_seconds': self.uptime_seconds,
            'components': {k: v.to_dict() for k, v in self.components.items()},
            'system_metrics': [m.to_dict() for m in self.system_metrics],
            'active_alerts': self.active_alerts,
            'summary': {
                'total_components': len(self.components),
                'healthy': self.healthy_count,
                'degraded': self.degraded_count,
                'unhealthy': self.unhealthy_count
            }
        }


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    component_id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'component_id': self.component_id,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'metadata': self.metadata
        }


@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    component_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'metric_type': self.metric_type.value,
            'component_id': self.component_id,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


# ============================================================================
# HEALTH CHECKERS
# ============================================================================

class BaseHealthChecker:
    """Base class for health checkers"""
    
    def __init__(self, component_id: str, component_type: ComponentType, config: Dict[str, Any]):
        self.component_id = component_id
        self.component_type = component_type
        self.config = config
        self.check_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_check: Optional[datetime] = None
    
    async def check_health(self) -> ComponentHealth:
        """Perform health check - must be implemented by subclasses"""
        raise NotImplementedError
    
    def _create_metric(
        self,
        name: str,
        value: float,
        unit: str,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        message: Optional[str] = None
    ) -> HealthMetric:
        """Create health metric with automatic status evaluation"""
        status = HealthStatus.HEALTHY
        
        if critical_threshold is not None and value >= critical_threshold:
            status = HealthStatus.UNHEALTHY
        elif warning_threshold is not None and value >= warning_threshold:
            status = HealthStatus.DEGRADED
        
        return HealthMetric(
            name=name,
            value=value,
            unit=unit,
            status=status,
            threshold_warning=warning_threshold,
            threshold_critical=critical_threshold,
            message=message
        )
    
    def _calculate_uptime(self) -> float:
        """Calculate uptime percentage"""
        if self.check_count == 0:
            return 100.0
        return (self.success_count / self.check_count) * 100


class SystemHealthChecker(BaseHealthChecker):
    """System resource health checker"""
    
    def __init__(self, component_id: str = "system", config: Dict[str, Any] = None):
        super().__init__(component_id, ComponentType.SYSTEM, config or {})
        self.cpu_threshold = self.config.get('cpu_threshold', 80.0)
        self.memory_threshold = self.config.get('memory_threshold', 85.0)
        self.disk_threshold = self.config.get('disk_threshold', 90.0)
    
    async def check_health(self) -> ComponentHealth:
        """Check system resources"""
        start_time = time.time()
        metrics = []
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(self._create_metric(
                "cpu_usage", cpu_percent, "%",
                self.cpu_threshold, self.cpu_threshold + 10
            ))
            
            # Memory
            memory = psutil.virtual_memory()
            metrics.append(self._create_metric(
                "memory_usage", memory.percent, "%",
                self.memory_threshold, self.memory_threshold + 10
            ))
            metrics.append(self._create_metric(
                "memory_available", memory.available / (1024**3), "GB"
            ))
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(self._create_metric(
                "disk_usage", disk_percent, "%",
                self.disk_threshold, self.disk_threshold + 5
            ))
            
            # Network
            net = psutil.net_io_counters()
            metrics.append(self._create_metric(
                "network_sent", net.bytes_sent / (1024**2), "MB"
            ))
            metrics.append(self._create_metric(
                "network_recv", net.bytes_recv / (1024**2), "MB"
            ))
            
            # Processes
            metrics.append(self._create_metric(
                "process_count", len(psutil.pids()), "count"
            ))
            
            self.success_count += 1
            overall_status = max(
                (m.status for m in metrics),
                key=lambda s: list(HealthStatus).index(s)
            )
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"System health check failed: {e}")
            overall_status = HealthStatus.UNHEALTHY
            metrics.append(self._create_metric(
                "check_error", 1, "boolean",
                message=str(e)
            ))
        
        self.check_count += 1
        self.last_check = datetime.utcnow()
        
        return ComponentHealth(
            component_id=self.component_id,
            component_type=self.component_type,
            status=overall_status,
            metrics=metrics,
            last_check=self.last_check,
            uptime_percent=self._calculate_uptime(),
            metadata={
                'check_duration_ms': (time.time() - start_time) * 1000,
                'check_count': self.check_count
            }
        )


class RedisHealthChecker(BaseHealthChecker):
    """Redis health checker"""
    
    def __init__(self, component_id: str, redis_client: aioredis.Redis, config: Dict[str, Any] = None):
        super().__init__(component_id, ComponentType.CACHE, config or {})
        self.redis_client = redis_client
    
    async def check_health(self) -> ComponentHealth:
        """Check Redis connectivity and performance"""
        start_time = time.time()
        metrics = []
        
        try:
            # Ping test
            await self.redis_client.ping()
            ping_time = (time.time() - start_time) * 1000
            
            metrics.append(self._create_metric(
                "ping_time", ping_time, "ms", 50, 200
            ))
            
            # Get Redis info
            info = await self.redis_client.info()
            
            # Memory usage
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            if max_memory > 0:
                memory_percent = (used_memory / max_memory) * 100
                metrics.append(self._create_metric(
                    "memory_usage", memory_percent, "%", 80, 95
                ))
            
            metrics.append(self._create_metric(
                "used_memory", used_memory / (1024**2), "MB"
            ))
            
            # Connections
            metrics.append(self._create_metric(
                "connected_clients", info.get('connected_clients', 0), "count", 100, 200
            ))
            
            # Operations
            metrics.append(self._create_metric(
                "ops_per_sec", info.get('instantaneous_ops_per_sec', 0), "ops/s"
            ))
            
            self.success_count += 1
            overall_status = max(
                (m.status for m in metrics),
                key=lambda s: list(HealthStatus).index(s)
            )
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Redis health check failed: {e}")
            overall_status = HealthStatus.UNHEALTHY
            metrics.append(self._create_metric(
                "connection_error", 1, "boolean",
                message=str(e)
            ))
        
        self.check_count += 1
        self.last_check = datetime.utcnow()
        
        return ComponentHealth(
            component_id=self.component_id,
            component_type=self.component_type,
            status=overall_status,
            metrics=metrics,
            last_check=self.last_check,
            uptime_percent=self._calculate_uptime(),
            metadata={
                'check_duration_ms': (time.time() - start_time) * 1000
            }
        )


class DatabaseHealthChecker(BaseHealthChecker):
    """Database health checker"""
    
    def __init__(self, component_id: str, db_connection_pool, config: Dict[str, Any] = None):
        super().__init__(component_id, ComponentType.DATABASE, config or {})
        self.db_pool = db_connection_pool
    
    async def check_health(self) -> ComponentHealth:
        """Check database connectivity and performance"""
        start_time = time.time()
        metrics = []
        
        try:
            # Simple query test
            async with self.db_pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            
            connection_time = (time.time() - start_time) * 1000
            metrics.append(self._create_metric(
                "connection_time", connection_time, "ms", 100, 1000
            ))
            
            # Pool statistics
            pool_size = self.db_pool.get_size()
            free_size = self.db_pool.get_idle_size()
            
            metrics.append(self._create_metric(
                "pool_size", pool_size, "connections"
            ))
            metrics.append(self._create_metric(
                "active_connections", pool_size - free_size, "connections",
                pool_size * 0.8, pool_size * 0.95
            ))
            
            self.success_count += 1
            overall_status = max(
                (m.status for m in metrics),
                key=lambda s: list(HealthStatus).index(s)
            )
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Database health check failed: {e}")
            overall_status = HealthStatus.UNHEALTHY
            metrics.append(self._create_metric(
                "connection_error", 1, "boolean",
                message=str(e)
            ))
        
        self.check_count += 1
        self.last_check = datetime.utcnow()
        
        return ComponentHealth(
            component_id=self.component_id,
            component_type=self.component_type,
            status=overall_status,
            metrics=metrics,
            last_check=self.last_check,
            uptime_percent=self._calculate_uptime(),
            metadata={
                'check_duration_ms': (time.time() - start_time) * 1000
            }
        )


# ============================================================================
# METRICS AGGREGATOR
# ============================================================================

class MetricsAggregator:
    """Aggregate and analyze metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_metric(self, metric: MetricPoint):
        """Add metric to aggregation window"""
        key = f"{metric.component_id}.{metric.name}"
        self.windows[key].append(metric)
    
    def get_statistics(self, component_id: str, metric_name: str) -> Dict[str, float]:
        """Get statistical analysis of a metric"""
        key = f"{component_id}.{metric_name}"
        window = self.windows.get(key, deque())
        
        if not window:
            return {}
        
        values = [m.value for m in window if isinstance(m.value, (int, float))]
        if not values:
            return {}
        
        try:
            return {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'latest': values[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}


# ============================================================================
# MONITORING SYSTEM (Main Class)
# ============================================================================

class MonitoringSystem:
    """
    Production-ready unified monitoring system
    Combines health checking, metrics collection, and alerting
    """
    
    def __init__(
        self,
        redis_url: str,
        check_interval: int = 30,
        retention_hours: int = 24,
        alert_threshold: int = 3
    ):
        self.redis_url = redis_url
        self.check_interval = check_interval
        self.retention_hours = retention_hours
        self.alert_threshold = alert_threshold
        
        # Components
        self.redis_client: Optional[aioredis.Redis] = None
        self.health_checkers: Dict[str, BaseHealthChecker] = {}
        self.aggregator = MetricsAggregator()
        
        # State
        self.is_running = False
        self.is_initialized = False
        self.start_time: Optional[datetime] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Storage
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alerts: deque = deque(maxlen=500)
        self.metrics_buffer: List[MetricPoint] = []
        
        # Instance info
        self.instance_id = str(uuid.uuid4())[:8]
        self.hostname = socket.gethostname()
        
        logger.info(f"Monitoring system initialized - Instance: {self.instance_id}")
    
    async def initialize(self) -> bool:
        """Initialize the monitoring system"""
        try:
            logger.info("Initializing monitoring system...")
            
            # Connect to Redis
            self.redis_client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Initialize core health checkers
            await self._initialize_core_checkers()
            
            # Set up Redis structures
            await self._setup_redis_structures()
            
            self.start_time = datetime.utcnow()
            self.is_initialized = True
            
            logger.info("Monitoring system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {e}")
            return False
    
    async def _initialize_core_checkers(self):
        """Initialize core system health checkers"""
        # System resources
        self.health_checkers['system'] = SystemHealthChecker()
        
        # Redis (self-check)
        self.health_checkers['redis'] = RedisHealthChecker('redis', self.redis_client)
        
        logger.info(f"Initialized {len(self.health_checkers)} core health checkers")
    
    async def _setup_redis_structures(self):
        """Set up Redis data structures"""
        config_key = f"ymera:monitoring:{self.instance_id}:config"
        await self.redis_client.hset(config_key, mapping={
            'check_interval': self.check_interval,
            'retention_hours': self.retention_hours,
            'start_time': self.start_time.isoformat(),
            'hostname': self.hostname
        })
        await self.redis_client.expire(config_key, self.retention_hours * 3600)
    
    async def start(self):
        """Start monitoring"""
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring started")
    
    async def stop(self):
        """Stop monitoring gracefully"""
        logger.info("Stopping monitoring system...")
        self.is_running = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining metrics
        await self._flush_metrics()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Run all health checks
                await self._run_health_checks()
                
                # Process alerts
                await self._process_alerts()
                
                # Flush metrics
                await self._flush_metrics()
                
                # Clean up old data
                if int(time.time()) % 3600 == 0:  # Hourly
                    await self._cleanup_old_data()
                
                # Sleep
                duration = time.time() - loop_start
                sleep_time = max(0, self.check_interval - duration)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _run_health_checks(self):
        """Run all health checks"""
        tasks = [
            self._safe_health_check(checker_id, checker)
            for checker_id, checker in self.health_checkers.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for checker_id, result in zip(self.health_checkers.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Health check failed for {checker_id}: {result}")
            else:
                self.health_history[checker_id].append(result)
                
                # Convert health metrics to metric points
                for metric in result.metrics:
                    self.metrics_buffer.append(MetricPoint(
                        name=metric.name,
                        value=metric.value,
                        metric_type=MetricType.GAUGE,
                        component_id=checker_id,
                        tags={'unit': metric.unit}
                    ))
    
    async def _safe_health_check(self, checker_id: str, checker: BaseHealthChecker) -> ComponentHealth:
        """Safely execute health check with timeout"""
        try:
            return await asyncio.wait_for(checker.check_health(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout: {checker_id}")
            return ComponentHealth(
                component_id=checker_id,
                component_type=checker.component_type,
                status=HealthStatus.UNHEALTHY,
                metrics=[],
                last_check=datetime.utcnow(),
                uptime_percent=0,
                error_message="Health check timeout"
            )
        except Exception as e:
            logger.error(f"Health check error for {checker_id}: {e}")
            raise
    
    async def _process_alerts(self):
        """Process and generate alerts"""
        for checker_id, history in self.health_history.items():
            if len(history) < self.alert_threshold:
                continue
            
            recent = list(history)[-self.alert_threshold:]
            consecutive_failures = sum(
                1 for h in recent 
                if h.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
            )
            
            if consecutive_failures >= self.alert_threshold:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    component_id=checker_id,
                    severity=AlertSeverity.CRITICAL if consecutive_failures >= self.alert_threshold + 2 else AlertSeverity.WARNING,
                    message=f"Component {checker_id} has {consecutive_failures} consecutive failures",
                    timestamp=datetime.utcnow(),
                    metadata={
                        'consecutive_failures': consecutive_failures,
                        'recent_statuses': [h.status.value for h in recent]
                    }
                )
                
                self.alerts.append(alert)
                await self._store_alert(alert)
    
    async def _store_alert(self, alert: Alert):
        """Store alert in Redis"""
        try:
            key = f"ymera:monitoring:{self.instance_id}:alerts"
            await self.redis_client.lpush(key, json.dumps(alert.to_dict()))
            await self.redis_client.ltrim(key, 0, 499)
            await self.redis_client.expire(key, self.retention_hours * 3600)
            
            logger.warning(f"Alert generated: {alert.severity.value} - {alert.message}")
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    async def _flush_metrics(self):
        """Flush metrics buffer to Redis"""
        if not self.metrics_buffer:
            return
        
        try:
            pipeline = self.redis_client.pipeline()
            
            for metric in self.metrics_buffer:
                key = f"ymera:metrics:{self.instance_id}:{metric.component_id}:{metric.name}"
                pipeline.zadd(key, {json.dumps(metric.to_dict()): int(metric.timestamp.timestamp())})
                pipeline.expire(key, self.retention_hours * 3600)
                
                # Add to aggregator
                self.aggregator.add_metric(metric)
            
            await pipeline.execute()
            logger.debug(f"Flushed {len(self.metrics_buffer)} metrics")
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data from Redis"""
        try:
            cutoff = int((datetime.utcnow() - timedelta(hours=self.retention_hours)).timestamp())
            
            pattern = f"ymera:metrics:{self.instance_id}:*"
            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.zremrangebyscore(key, 0, cutoff)
            
            logger.info("Cleaned up old metrics data")
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    # Public API
    
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        if not self.health_history:
            await self._run_health_checks()
        
        components = {}
        healthy = degraded = unhealthy = 0
        
        for checker_id, history in self.health_history.items():
            if history:
                latest = history[-1]
                components[checker_id] = latest
                
                if latest.status == HealthStatus.HEALTHY:
                    healthy += 1
                elif latest.status == HealthStatus.DEGRADED:
                    degraded += 1
                elif latest.status == HealthStatus.UNHEALTHY:
                    unhealthy += 1
        
        # Overall status
        if unhealthy > 0:
            overall = HealthStatus.UNHEALTHY
        elif degraded > 0:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        # System metrics
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        system_metrics = [
            HealthMetric("uptime", uptime, "seconds", HealthStatus.HEALTHY),
            HealthMetric("component_count", len(components), "count", HealthStatus.HEALTHY),
            HealthMetric("health_percentage", (healthy / max(len(components), 1)) * 100, "%", HealthStatus.HEALTHY)
        ]
        
        # Recent alerts
        recent_alerts = [
            a.to_dict() for a in list(self.alerts)[-10:]
            if not a.acknowledged
        ]
        
        return SystemHealth(
            overall_status=overall,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime,
            components=components,
            system_metrics=system_metrics,
            active_alerts=recent_alerts,
            healthy_count=healthy,
            degraded_count=degraded,
            unhealthy_count=unhealthy
        )
    
    async def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health for specific component"""
        history = self.health_history.get(component_id)
        return history[-1] if history else None
    
    async def get_metrics(self, component_id: str, metric_name: str) -> Dict[str, Any]:
        """Get metrics statistics"""
        stats = self.aggregator.get_statistics(component_id, metric_name)
        
        # Get historical data from Redis
        key = f"ymera:metrics:{self.instance_id}:{component_id}:{metric_name}"
        try:
            data = await self.redis_client.zrange(key, -50, -1, withscores=True)
            historical = [
                {'value': json.loads(d[0])['value'], 'timestamp': d[1]}
                for d in data
            ]
        except Exception as e:
            logger.error(f"Failed to get historical metrics: {e}")
            historical = []
        
        return {
            'component_id': component_id,
            'metric_name': metric_name,
            'statistics': stats,
            'historical': historical
        }
    
    async def get_alerts(self, acknowledged: Optional[bool] = None) -> List[Alert]:
        """Get alerts with optional filtering"""
        alerts = list(self.alerts)
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        return alerts
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                
                # Update in Redis
                try:
                    key = f"ymera:monitoring:{self.instance_id}:alerts"
                    alerts_data = await self.redis_client.lrange(key, 0, -1)
                    
                    for i, data in enumerate(alerts_data):
                        alert_dict = json.loads(data)
                        if alert_dict['alert_id'] == alert_id:
                            alert_dict['acknowledged'] = True
                            alert_dict['acknowledged_by'] = acknowledged_by
                            alert_dict['acknowledged_at'] = alert.acknowledged_at.isoformat()
                            await self.redis_client.lset(key, i, json.dumps(alert_dict))
                            break
                except Exception as e:
                    logger.error(f"Failed to update alert in Redis: {e}")
                
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        
        return False
    
    def add_health_checker(self, checker_id: str, checker: BaseHealthChecker):
        """Add custom health checker"""
        self.health_checkers[checker_id] = checker
        logger.info(f"Added health checker: {checker_id}")
    
    def remove_health_checker(self, checker_id: str):
        """Remove health checker"""
        if checker_id in self.health_checkers:
            del self.health_checkers[checker_id]
            logger.info(f"Removed health checker: {checker_id}")
    
    async def record_metric(
        self,
        component_id: str,
        metric_name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a custom metric"""
        metric = MetricPoint(
            name=metric_name,
            value=value,
            metric_type=metric_type,
            component_id=component_id,
            tags=tags or {}
        )
        self.metrics_buffer.append(metric)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        system_health = await self.get_system_health()
        
        return {
            'system_health': system_health.to_dict(),
            'instance_info': {
                'instance_id': self.instance_id,
                'hostname': self.hostname,
                'platform': platform.system(),
                'python_version': platform.python_version()
            },
            'monitoring_stats': {
                'is_running': self.is_running,
                'check_interval': self.check_interval,
                'retention_hours': self.retention_hours,
                'total_checkers': len(self.health_checkers),
                'total_alerts': len(self.alerts),
                'unacknowledged_alerts': sum(1 for a in self.alerts if not a.acknowledged)
            }
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_monitoring_system(
    redis_url: str,
    check_interval: int = 30,
    retention_hours: int = 24
) -> MonitoringSystem:
    """Create and initialize monitoring system"""
    system = MonitoringSystem(
        redis_url=redis_url,
        check_interval=check_interval,
        retention_hours=retention_hours
    )
    
    if await system.initialize():
        await system.start()
        return system
    else:
        raise RuntimeError("Failed to initialize monitoring system")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    """Example usage"""
    
    # Create monitoring system
    monitoring = await create_monitoring_system(
        redis_url="redis://localhost:6379/0",
        check_interval=30,
        retention_hours=24
    )
    
    try:
        # Let it run for a bit
        await asyncio.sleep(60)
        
        # Get system health
        health = await monitoring.get_system_health()
        print(f"Overall Status: {health.overall_status.value}")
        print(f"Components: {len(health.components)}")
        print(f"Healthy: {health.healthy_count}, Degraded: {health.degraded_count}, Unhealthy: {health.unhealthy_count}")
        
        # Get specific component
        system_health = await monitoring.get_component_health('system')
        if system_health:
            print(f"\nSystem Status: {system_health.status.value}")
            print(f"Uptime: {system_health.uptime_percent:.2f}%")
        
        # Get alerts
        alerts = await monitoring.get_alerts(acknowledged=False)
        print(f"\nActive Alerts: {len(alerts)}")
        
        # Get dashboard data
        dashboard = await monitoring.get_dashboard_data()
        print(f"\nDashboard: {json.dumps(dashboard, indent=2, default=str)}")
        
    finally:
        # Stop monitoring
        await monitoring.stop()


if __name__ == "__main__":
    asyncio.run(main())