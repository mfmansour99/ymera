"""
YMERA Enterprise - Monitoring Compatibility Module
Production-Ready Performance Tracking - v4.1
Fixes missing monitoring module imports
"""

import time
import functools
import logging
from typing import Any, Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def track_performance(func: Callable) -> Callable:
    """
    Decorator to track function performance metrics.
    Compatible replacement for missing monitoring.performance_tracker.
    
    Usage:
        @track_performance
        async def my_function():
            pass
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(
                f"Performance: {function_name} completed",
                extra={
                    "function": function_name,
                    "execution_time": execution_time,
                    "status": "success"
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                f"Performance: {function_name} failed",
                extra={
                    "function": function_name,
                    "execution_time": execution_time,
                    "status": "error",
                    "error": str(e)
                }
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(
                f"Performance: {function_name} completed",
                extra={
                    "function": function_name,
                    "execution_time": execution_time,
                    "status": "success"
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                f"Performance: {function_name} failed",
                extra={
                    "function": function_name,
                    "execution_time": execution_time,
                    "status": "error",
                    "error": str(e)
                }
            )
            raise
    
    # Return appropriate wrapper based on function type
    if hasattr(func, '__await__'):
        return async_wrapper
    else:
        return sync_wrapper


class PerformanceTracker:
    """
    Comprehensive performance tracking system.
    Compatible replacement for missing monitoring module.
    """
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_tracking(self, operation_id: str) -> None:
        """Start tracking an operation"""
        self.start_times[operation_id] = time.time()
    
    def end_tracking(self, operation_id: str, metadata: Optional[dict] = None) -> float:
        """End tracking and return execution time"""
        if operation_id not in self.start_times:
            logger.warning(f"No start time found for operation: {operation_id}")
            return 0.0
        
        execution_time = time.time() - self.start_times[operation_id]
        
        self.metrics[operation_id] = {
            "execution_time": execution_time,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        del self.start_times[operation_id]
        
        return execution_time
    
    def get_metrics(self, operation_id: Optional[str] = None) -> dict:
        """Get performance metrics"""
        if operation_id:
            return self.metrics.get(operation_id, {})
        return self.metrics
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics"""
        self.metrics.clear()
        self.start_times.clear()


# Global tracker instance
_global_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance"""
    return _global_tracker


# Compatibility alias
MonitoringClient = PerformanceTracker

__all__ = [
    "track_performance",
    "PerformanceTracker",
    "get_performance_tracker",
    "MonitoringClient"  # Compatibility alias
]
