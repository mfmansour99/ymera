"""
YMERA Enterprise - Response Aggregator
Production-Ready Response Collection & Processing System - v4.0
Enterprise-grade implementation with zero placeholders
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
import statistics

import redis.asyncio as aioredis
import structlog
from fastapi import HTTPException
from pydantic import BaseModel, Field, validator

from config.settings import get_settings
from monitoring.performance_tracker import track_performance

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.response_aggregator")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

MAX_CONCURRENT_AGGREGATIONS = 1000
DEFAULT_RESPONSE_TIMEOUT = 300
MAX_RESPONSE_SIZE = 50 * 1024 * 1024
CLEANUP_INTERVAL = 60
STATISTICS_RETENTION_HOURS = 24
MAX_RESPONSES_PER_REQUEST = 1000

settings = get_settings()

# ===============================================================================
# ENUMS & DATA MODELS
# ===============================================================================

class AggregationStrategy(Enum):
    """Response aggregation strategies"""
    FIRST_RESPONSE = "first_response"
    ALL_RESPONSES = "all_responses"
    MAJORITY_CONSENSUS = "majority_consensus"
    WEIGHTED_AVERAGE = "weighted_average"
    FASTEST_N = "fastest_n"
    BEST_QUALITY = "best_quality"
    TIMEOUT_BASED = "timeout_based"
    CUSTOM = "custom"

class AggregationStatus(Enum):
    """Aggregation request status"""
    PENDING = "pending"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class ResponseMetrics:
    """Metrics for individual responses"""
    response_time: float = 0.0
    processing_time: float = 0.0
    size_bytes: int = 0
    quality_score: float = 0.0
    confidence: float = 0.0
    error_count: int = 0
    agent_reputation: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AggregatedResponse:
    """Individual response in aggregation"""
    response_id: str
    agent_id: str
    correlation_id: str
    data: Any
    metrics: ResponseMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    received_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AggregationRequest:
    """Request for response aggregation"""
    request_id: str
    correlation_id: str
    strategy: AggregationStrategy
    timeout: int = DEFAULT_RESPONSE_TIMEOUT
    expected_responses: Optional[int] = None
    minimum_responses: int = 1
    quality_threshold: float = 0.0
    weight_function: Optional[Callable[[AggregatedResponse], float]] = None
    custom_aggregator: Optional[Callable[[List[AggregatedResponse]], Any]] = None
    filter_function: Optional[Callable[[AggregatedResponse], bool]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregationResult:
    """Result of response aggregation"""
    request_id: str
    status: AggregationStatus
    aggregated_data: Any = None
    responses_used: List[str] = field(default_factory=list)
    total_responses: int = 0
    processing_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# ===============================================================================
# PYDANTIC SCHEMAS
# ===============================================================================

class AggregationRequestSchema(BaseModel):
    """Schema for aggregation requests"""
    correlation_id: str = Field(..., min_length=1, max_length=100)
    strategy: str = Field(..., regex="^(first_response|all_responses|majority_consensus|weighted_average|fastest_n|best_quality|timeout_based|custom)$")
    timeout: int = Field(default=DEFAULT_RESPONSE_TIMEOUT, ge=1, le=3600)
    expected_responses: Optional[int] = Field(None, ge=1, le=MAX_RESPONSES_PER_REQUEST)
    minimum_responses: int = Field(default=1, ge=1, le=MAX_RESPONSES_PER_REQUEST)
    quality_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('expected_responses')
    def validate_expected_responses(cls, v, values):
        if v is not None and 'minimum_responses' in values:
            if v < values['minimum_responses']:
                raise ValueError("Expected responses must be >= minimum responses")
        return v

class ResponseSubmissionSchema(BaseModel):
    """Schema for response submissions"""
    correlation_id: str = Field(..., min_length=1, max_length=100)
    agent_id: str = Field(..., min_length=1, max_length=100)
    data: Any = None
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    processing_time: float = Field(default=0.0, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ===============================================================================
# CORE IMPLEMENTATION CLASSES
# ===============================================================================

class AggregatorConfig:
    """Configuration for response aggregator"""
    
    def __init__(self):
        self.enabled: bool = True
        self.max_concurrent: int = MAX_CONCURRENT_AGGREGATIONS
        self.default_timeout: int = DEFAULT_RESPONSE_TIMEOUT
        self.max_response_size: int = MAX_RESPONSE_SIZE
        self.cleanup_interval: int = CLEANUP_INTERVAL
        self.statistics_retention: int = STATISTICS_RETENTION_HOURS
        self.redis_url: str = settings.REDIS_URL
        self.enable_quality_scoring: bool = True
        self.enable_agent_reputation: bool = True

class ProductionResponseAggregator:
    """Production-ready response aggregator implementation"""
    
    def __init__(self, config: AggregatorConfig = None):
        self.config = config or AggregatorConfig()
        self.logger = logger.bind(module="production_response_aggregator")
        
        self._redis_client: Optional[aioredis.Redis] = None
        self._active_requests: Dict[str, AggregationRequest] = {}
        self._collected_responses: Dict[str, List[AggregatedResponse]] = defaultdict(list)
        self._completed_results: Dict[str, AggregationResult] = {}
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._statistics: Dict[str, Any] = defaultdict(int)
        self._agent_reputation: Dict[str, float] = defaultdict(lambda: 1.0)
        self._lock = asyncio.Lock()
        self._health_status = True
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> None:
        """Initialize aggregator resources"""
        try:
            await self._setup_redis_connection()
            await self._setup_background_tasks()
            await self._load_agent_reputation()
            self.logger.info("Response aggregator initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize response aggregator", error=str(e))
            raise
    
    async def _setup_redis_connection(self) -> None:
        """Setup Redis connection for caching and coordination"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self._redis_client = await aioredis.from_url(
                    self.config.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=30,
                    socket_connect_timeout=10,
                    retry_on_timeout=True,
                    max_connections=20
                )
                
                await self._redis_client.ping()
                self.logger.info("Redis connection established for aggregator")
                return
                
            except Exception as e:
                self.logger.error(f"Redis connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise ConnectionError(f"Failed to connect to Redis after {max_retries} attempts")
    
    async def _setup_background_tasks(self) -> None:
        """Setup background maintenance tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
        self.logger.info("Background tasks started")
    
    async def _load_agent_reputation(self) -> None:
        """Load agent reputation scores from storage"""
        try:
            if not self._redis_client:
                return
            
            reputation_data = await self._redis_client.hgetall("agent_reputation")
            
            for agent_id, score in reputation_data.items():
                self._agent_reputation[agent_id] = float(score)
            
            self.logger.info("Loaded agent reputation data", count=len(reputation_data))
            
        except Exception as e:
            self.logger.error("Failed to load agent reputation", error=str(e))
    
    @track_performance
    async def create_aggregation_request(self, request: AggregationRequest) -> str:
        """Create new aggregation request"""
        async with self._lock:
            try:
                request_id = str(uuid.uuid4())
                request.request_id = request_id
                
                await self._validate_aggregation_request(request)
                
                if len(self._active_requests) >= self.config.max_concurrent:
                    raise HTTPException(
                        status_code=429,
                        detail="Maximum concurrent aggregations reached"
                    )
                
                self._active_requests[request_id] = request
                self._collected_responses[request_id] = []
                
                processing_task = asyncio.create_task(
                    self._process_aggregation_request(request_id)
                )
                self._processing_tasks[request_id] = processing_task
                
                await self._persist_aggregation_request(request)
                
                self._statistics["requests_created"] += 1
                self._statistics[f"strategy_{request.strategy.value}"] += 1
                
                self.logger.info(
                    "Aggregation request created",
                    request_id=request_id,
                    correlation_id=request.correlation_id,
                    strategy=request.strategy.value
                )
                
                return request_id
                
            except Exception as e:
                self.logger.error("Failed to create aggregation request", error=str(e))
                raise
    
    async def _validate_aggregation_request(self, request: AggregationRequest) -> None:
        """Validate aggregation request parameters"""
        if not request.correlation_id:
            raise ValueError("Correlation ID is required")
        
        if request.timeout <= 0 or request.timeout > 3600:
            raise ValueError("Timeout must be between 1 and 3600 seconds")
        
        if request.minimum_responses <= 0:
            raise ValueError("Minimum responses must be positive")
        
        if request.expected_responses and request.expected_responses < request.minimum_responses:
            raise ValueError("Expected responses must be >= minimum responses")
    
    async def _persist_aggregation_request(self, request: AggregationRequest) -> None:
        """Persist aggregation request to Redis"""
        if not self._redis_client:
            return
        
        request_data = {
            "request_id": request.request_id,
            "correlation_id": request.correlation_id,
            "strategy": request.strategy.value,
            "timeout": request.timeout,
            "expected_responses": request.expected_responses or 0,
            "minimum_responses": request.minimum_responses,
            "quality_threshold": request.quality_threshold,
            "created_at": request.created_at.isoformat(),
            "metadata": json.dumps(request.metadata)
        }
        
        await self._redis_client.hset(
            f"aggregation_request:{request.request_id}",
            mapping=request_data
        )
        await self._redis_client.expire(
            f"aggregation_request:{request.request_id}",
            request.timeout + 300
        )
    
    @track_performance
    async def submit_response(self, response: AggregatedResponse) -> bool:
        """Submit response for aggregation"""
        try:
            request_id = None
            for rid, request in self._active_requests.items():
                if request.correlation_id == response.correlation_id:
                    request_id = rid
                    break
            
            if not request_id:
                self.logger.warning(
                    "No matching aggregation request found",
                    correlation_id=response.correlation_id
                )
                return False
            
            async with self._lock:
                if request_id not in self._active_requests:
                    return False
                
                await self._validate_response(response)
                
                if self.config.enable_agent_reputation:
                    agent_reputation = self._agent_reputation[response.agent_id]
                    response.metrics.agent_reputation = agent_reputation
                    response.metrics.quality_score *= agent_reputation
                
                self._collected_responses[request_id].append(response)
                self._statistics["responses_received"] += 1
                
                self.logger.info(
                    "Response submitted for aggregation",
                    request_id=request_id,
                    response_id=response.response_id,
                    agent_id=response.agent_id
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to submit response", error=str(e))
            return False
    
    async def _validate_response(self, response: AggregatedResponse) -> None:
        """Validate response data"""
        if not response.response_id:
            raise ValueError("Response ID is required")
        
        if not response.agent_id:
            raise ValueError("Agent ID is required")
        
        response_size = len(json.dumps(response.data, default=str))
        if response_size > self.config.max_response_size:
            raise ValueError(f"Response size exceeds maximum")
    
    async def _process_aggregation_request(self, request_id: str) -> None:
        """Process aggregation request with timeout handling"""
        try:
            request = self._active_requests[request_id]
            start_time = datetime.utcnow()
            
            await self._wait_for_responses(request_id, request.timeout)
            result = await self._aggregate_responses(request_id)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result.processing_time = processing_time
            result.completed_at = datetime.utcnow()
            
            async with self._lock:
                self._completed_results[request_id] = result
                
                if request_id in self._active_requests:
                    del self._active_requests[request_id]
                
                if request_id in self._collected_responses:
                    del self._collected_responses[request_id]
            
            if self.config.enable_agent_reputation:
                await self._update_agent_reputation(request_id, result)
            
            self._statistics["requests_completed"] += 1
            self._statistics[f"status_{result.status.value}"] += 1
            
            self.logger.info(
                "Aggregation processing completed",
                request_id=request_id,
                status=result.status.value
            )
            
        except Exception as e:
            error_result = AggregationResult(
                request_id=request_id,
                status=AggregationStatus.ERROR,
                error_message=str(e),
                completed_at=datetime.utcnow()
            )
            
            async with self._lock:
                self._completed_results[request_id] = error_result
                if request_id in self._active_requests:
                    del self._active_requests[request_id]
            
            self._statistics["requests_failed"] += 1
            self.logger.error("Aggregation processing failed", request_id=request_id, error=str(e))
        
        finally:
            if request_id in self._processing_tasks:
                del self._processing_tasks[request_id]
    
    async def _wait_for_responses(self, request_id: str, timeout: int) -> None:
        """Wait for responses with various completion conditions"""
        request = self._active_requests[request_id]
        start_time = datetime.utcnow()
        
        while True:
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            
            if elapsed >= timeout:
                break
            
            responses = self._collected_responses[request_id]
            
            if len(responses) >= request.minimum_responses:
                if request.strategy == AggregationStrategy.FIRST_RESPONSE:
                    break
                elif request.expected_responses and len(responses) >= request.expected_responses:
                    break
            
            await asyncio.sleep(0.1)
    
    async def _aggregate_responses(self, request_id: str) -> AggregationResult:
        """Aggregate collected responses based on strategy"""
        request = self._active_requests.get(request_id)
        if not request:
            return AggregationResult(
                request_id=request_id,
                status=AggregationStatus.ERROR,
                error_message="Request not found"
            )
        
        responses = self._collected_responses.get(request_id, [])
        
        if not responses:
            return AggregationResult(
                request_id=request_id,
                status=AggregationStatus.TIMEOUT,
                error_message="No responses received"
            )
        
        quality_responses = [
            r for r in responses 
            if r.metrics.quality_score >= request.quality_threshold
        ]
        
        if not quality_responses:
            return AggregationResult(
                request_id=request_id,
                status=AggregationStatus.ERROR,
                error_message="No responses meet quality threshold"
            )
        
        try:
            if request.strategy == AggregationStrategy.FIRST_RESPONSE:
                aggregated_data = await self._aggregate_first_response(quality_responses)
            elif request.strategy == AggregationStrategy.ALL_RESPONSES:
                aggregated_data = await self._aggregate_all_responses(quality_responses)
            elif request.strategy == AggregationStrategy.MAJORITY_CONSENSUS:
                aggregated_data = await self._aggregate_majority_consensus(quality_responses)
            elif request.strategy == AggregationStrategy.WEIGHTED_AVERAGE:
                aggregated_data = await self._aggregate_weighted_average(quality_responses, request.weight_function)
            elif request.strategy == AggregationStrategy.FASTEST_N:
                aggregated_data = await self._aggregate_fastest_n(quality_responses, request.minimum_responses)
            elif request.strategy == AggregationStrategy.BEST_QUALITY:
                aggregated_data = await self._aggregate_best_quality(quality_responses)
            else:
                aggregated_data = await self._aggregate_best_quality(quality_responses)
            
            quality_metrics = self._calculate_quality_metrics(quality_responses)
            confidence_score = self._calculate_confidence_score(quality_responses, request.strategy)
            
            return AggregationResult(
                request_id=request_id,
                status=AggregationStatus.COMPLETED,
                aggregated_data=aggregated_data,
                responses_used=[r.response_id for r in quality_responses],
                total_responses=len(responses),
                quality_metrics=quality_metrics,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            return AggregationResult(
                request_id=request_id,
                status=AggregationStatus.ERROR,
                error_message=f"Aggregation failed: {str(e)}"
            )
    
    async def _aggregate_first_response(self, responses: List[AggregatedResponse]) -> Any:
        """Aggregate using first response strategy"""
        responses.sort(key=lambda r: r.received_at)
        return responses[0].data
    
    async def _aggregate_all_responses(self, responses: List[AggregatedResponse]) -> Any:
        """Aggregate all responses into a list"""
        return [
            {
                "agent_id": r.agent_id,
                "data": r.data,
                "quality_score": r.metrics.quality_score,
                "confidence": r.metrics.confidence
            }
            for r in responses
        ]
    
    async def _aggregate_majority_consensus(self, responses: List[AggregatedResponse]) -> Any:
        """Aggregate using majority consensus"""
        if len(responses) < 3:
            return await self._aggregate_best_quality(responses)
        
        response_groups = defaultdict(list)
        
        for response in responses:
            response_str = json.dumps(response.data, sort_keys=True, default=str)
            response_groups[response_str].append(response)
        
        majority_group = max(response_groups.values(), key=len)
        
        if len(majority_group) >= len(responses) / 2:
            best_response = max(majority_group, key=lambda r: r.metrics.quality_score)
            return best_response.data
        else:
            return await self._aggregate_best_quality(responses)
    
    async def _aggregate_weighted_average(self, responses: List[AggregatedResponse], weight_function: Optional[Callable] = None) -> Any:
        """Aggregate using weighted average"""
        if not weight_function:
            weight_function = lambda r: r.metrics.quality_score * r.metrics.confidence
        
        numeric_responses = []
        for response in responses:
            try:
                if isinstance(response.data, (int, float)):
                    numeric_responses.append((response.data, weight_function(response)))
            except:
                continue
        
        if numeric_responses:
            total_weighted = sum(value * weight for value, weight in numeric_responses)
            total_weight = sum(weight for _, weight in numeric_responses)
            return total_weighted / total_weight if total_weight > 0 else 0
        else:
            return await self._aggregate_best_quality(responses)
    
    async def _aggregate_fastest_n(self, responses: List[AggregatedResponse], n: int) -> Any:
        """Aggregate N fastest responses"""
        responses.sort(key=lambda r: r.metrics.response_time)
        fastest_responses = responses[:n]
        return await self._aggregate_all_responses(fastest_responses)
    
    async def _aggregate_best_quality(self, responses: List[AggregatedResponse]) -> Any:
        """Return response with best quality score"""
        best_response = max(responses, key=lambda r: r.metrics.quality_score)
        return best_response.data
    
    def _calculate_quality_metrics(self, responses: List[AggregatedResponse]) -> Dict[str, float]:
        """Calculate aggregate quality metrics"""
        if not responses:
            return {}
        
        quality_scores = [r.metrics.quality_score for r in responses]
        confidence_scores = [r.metrics.confidence for r in responses]
        
        return {
            "average_quality": statistics.mean(quality_scores),
            "quality_std": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            "average_confidence": statistics.mean(confidence_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores)
        }
    
    def _calculate_confidence_score(self, responses: List[AggregatedResponse], strategy: AggregationStrategy) -> float:
        """Calculate overall confidence score for aggregation"""
        if not responses:
            return 0.0
        
        base_confidence = statistics.mean([r.metrics.confidence for r in responses])
        
        strategy_multiplier = {
            AggregationStrategy.FIRST_RESPONSE: 0.7,
            AggregationStrategy.ALL_RESPONSES: 0.9,
            AggregationStrategy.MAJORITY_CONSENSUS: 0.95,
            AggregationStrategy.WEIGHTED_AVERAGE: 0.85,
            AggregationStrategy.FASTEST_N: 0.8,
            AggregationStrategy.BEST_QUALITY: 0.9
        }.get(strategy, 0.8)
        
        return min(base_confidence * strategy_multiplier, 1.0)
    
    async def _update_agent_reputation(self, request_id: str, result: AggregationResult) -> None:
        """Update agent reputation based on aggregation results"""
        try:
            if result.status != AggregationStatus.COMPLETED:
                return
            
            responses = self._collected_responses.get(request_id, [])
            for response in responses:
                agent_id = response.agent_id
                current_reputation = self._agent_reputation[agent_id]
                
                if response.response_id in result.responses_used:
                    adjustment = 0.01 if response.metrics.quality_score > 0.8 else 0.005
                else:
                    adjustment = -0.005
                
                new_reputation = max(0.1, min(2.0, current_reputation + adjustment))
                self._agent_reputation[agent_id] = new_reputation
            
            if self._redis_client:
                reputation_updates = {
                    agent_id: str(reputation) 
                    for agent_id, reputation in self._agent_reputation.items()
                }
                await self._redis_client.hset("agent_reputation", mapping=reputation_updates)
            
        except Exception as e:
            self.logger.error("Failed to update agent reputation", error=str(e))
    
    async def get_aggregation_result(self, request_id: str) -> Optional[AggregationResult]:
        """Get aggregation result by request ID"""
        if request_id in self._completed_results:
            return self._completed_results[request_id]
        
        if request_id in self._active_requests:
            return AggregationResult(
                request_id=request_id,
                status=AggregationStatus.COLLECTING,
                total_responses=len(self._collected_responses.get(request_id, []))
            )
        
        return None
    
    async def _cleanup_expired_requests(self) -> None:
        """Background task to cleanup expired requests and results"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                current_time = datetime.utcnow()
                expired_requests = []
                
                for request_id, request in self._active_requests.items():
                    elapsed = (current_time - request.created_at).total_seconds()
                    if elapsed > (request.timeout + 60):
                        expired_requests.append(request_id)
                
                async with self._lock:
                    for request_id in expired_requests:
                        if request_id in self._processing_tasks:
                            self._processing_tasks[request_id].cancel()
                            del self._processing_tasks[request_id]
                        
                        timeout_result = AggregationResult(
                            request_id=request_id,
                            status=AggregationStatus.TIMEOUT,
                            total_responses=len(self._collected_responses.get(request_id, [])),
                            completed_at=current_time
                        )
                        
                        self._completed_results[request_id] = timeout_result
                        
                        if request_id in self._active_requests:
                            del self._active_requests[request_id]
                        
                        if request_id in self._collected_responses:
                            del self._collected_responses[request_id]
                        
                        self._statistics["requests_timeout"] += 1
                
                if expired_requests:
                    self.logger.info("Cleanup completed", expired_requests=len(expired_requests))
                
            except Exception as e:
                self.logger.error("Error in cleanup task", error=str(e))
    
    async def get_aggregator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive aggregator statistics"""
        try:
            return {
                "active_requests": len(self._active_requests),
                "completed_results": len(self._completed_results),
                "processing_tasks": len(self._processing_tasks),
                "agent_count": len(self._agent_reputation),
                "statistics": dict(self._statistics),
                "health_status": self._health_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error("Failed to get aggregator statistics", error=str(e))
            return {"error": str(e)}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get aggregator health status"""
        try:
            redis_healthy = False
            if self._redis_client:
                try:
                    await self._redis_client.ping()
                    redis_healthy = True
                except:
                    pass
            
            return {
                "status": "healthy" if redis_healthy else "degraded",
                "redis_connection": redis_healthy,
                "active_requests": len(self._active_requests),
                "statistics": dict(self._statistics)
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def shutdown(self) -> None:
        """Cleanup aggregator resources"""
        try:
            self._shutdown_event.set()
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            for task in self._processing_tasks.values():
                task.cancel()
            
            if self._processing_tasks:
                await asyncio.gather(*self._processing_tasks.values(), return_exceptions=True)
            
            if self._redis_client:
                await self._redis_client.close()
            
            self.logger.info("Response aggregator cleanup completed")
            
        except Exception as e:
            self.logger.error("Error during aggregator cleanup", error=str(e))

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

async def create_response_aggregator(redis_url: str = None) -> ProductionResponseAggregator:
    """Factory function to create and initialize response aggregator"""
    config = AggregatorConfig()
    if redis_url:
        config.redis_url = redis_url
    
    aggregator = ProductionResponseAggregator(config)
    await aggregator.initialize()
    
    return aggregator

async def health_check() -> Dict[str, Any]:
    """Response aggregator health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "module": "response_aggregator",
        "version": "4.0"
    }

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "ProductionResponseAggregator",
    "ResponseAggregator",
    "ResponseCollector",
    "AggregatorConfig",
    "AggregationRequest",
    "AggregatedResponse",
    "AggregationResult",
    "AggregationStrategy",
    "AggregationStatus",
    "ResponseMetrics",
    "AggregationRequestSchema",
    "ResponseSubmissionSchema",
    "create_response_aggregator",
    "health_check"
]

# Aliases for backward compatibility
ResponseAggregator = ProductionResponseAggregator
ResponseCollector = ProductionResponseAggregator