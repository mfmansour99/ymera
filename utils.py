"""
YMERA Enterprise - Core Engine Utilities
Production-Ready Utility Functions - v4.0
Essential utilities for Core Engine operations
"""

# ===============================================================================
# STANDARD IMPORTS
# ===============================================================================
import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
import logging

# ===============================================================================
# STRUCTURED LOGGING
# ===============================================================================
try:
    import structlog
    logger = structlog.get_logger("ymera.core_engine.utils")
except ImportError:
    logger = logging.getLogger("ymera.core_engine.utils")

# ===============================================================================
# CONSTANTS
# ===============================================================================
SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 3600
SECONDS_IN_DAY = 86400

# ===============================================================================
# IDENTIFIER GENERATION
# ===============================================================================

def generate_unique_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a unique identifier for Core Engine operations.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random portion (default: 8)
        
    Returns:
        Unique identifier string
        
    Example:
        >>> generate_unique_id("task", 8)
        'task_172334_a3f4c2d1'
    """
    unique_part = str(uuid.uuid4()).replace('-', '')[:length]
    timestamp = str(int(time.time() * 1000))[-6:]
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_part}"
    return f"{timestamp}_{unique_part}"


def generate_cycle_id() -> str:
    """Generate unique ID for learning cycle"""
    return generate_unique_id("cycle", 12)


def generate_task_id() -> str:
    """Generate unique ID for async task"""
    return generate_unique_id("task", 10)

# ===============================================================================
# TIMESTAMP UTILITIES
# ===============================================================================

def get_utc_timestamp() -> datetime:
    """Get current UTC timestamp"""
    return datetime.now(timezone.utc)


def format_timestamp(
    dt: Optional[datetime] = None,
    format_type: str = "iso"
) -> str:
    """
    Format timestamp in various formats.
    
    Args:
        dt: DateTime object (defaults to current time)
        format_type: Format type ('iso', 'human', 'compact')
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = get_utc_timestamp()
    
    formats = {
        'iso': lambda d: d.isoformat(),
        'human': lambda d: d.strftime('%Y-%m-%d %H:%M:%S UTC'),
        'compact': lambda d: d.strftime('%Y%m%d_%H%M%S'),
    }
    
    formatter = formats.get(format_type, formats['iso'])
    return formatter(dt)


def timestamp_to_seconds(dt: datetime) -> float:
    """Convert datetime to seconds since epoch"""
    return dt.timestamp()

# ===============================================================================
# HASHING UTILITIES
# ===============================================================================

def calculate_hash(
    data: Union[str, bytes, Dict[str, Any]],
    algorithm: str = "sha256"
) -> str:
    """
    Calculate hash of data for integrity checking.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm (default: sha256)
        
    Returns:
        Hexadecimal hash string
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hasher = hashlib.new(algorithm)
    
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        hasher.update(data_str.encode('utf-8'))
    elif isinstance(data, str):
        hasher.update(data.encode('utf-8'))
    elif isinstance(data, bytes):
        hasher.update(data)
    else:
        hasher.update(str(data).encode('utf-8'))
    
    return hasher.hexdigest()


def verify_integrity(data: Any, expected_hash: str) -> bool:
    """Verify data integrity against hash"""
    actual_hash = calculate_hash(data)
    return actual_hash == expected_hash

# ===============================================================================
# JSON UTILITIES
# ===============================================================================

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with error handling.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed data or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("JSON parsing failed", error=str(e))
        return default


def safe_json_dumps(
    data: Any,
    ensure_ascii: bool = False,
    indent: Optional[int] = None
) -> str:
    """
    Safely serialize data to JSON with error handling.
    
    Args:
        data: Data to serialize
        ensure_ascii: Whether to escape non-ASCII characters
        indent: Indentation for pretty printing
        
    Returns:
        JSON string or empty string on error
    """
    def json_serializer(obj):
        """Default serializer for non-standard types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    try:
        return json.dumps(
            data,
            default=json_serializer,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=(',', ':') if indent is None else None
        )
    except (TypeError, ValueError) as e:
        logger.warning("JSON serialization failed", error=str(e))
        return ""

# ===============================================================================
# DICTIONARY UTILITIES
# ===============================================================================

def deep_merge_dict(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge into dict1
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if (key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(
    data: Dict[str, Any],
    separator: str = ".",
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        data: Dictionary to flatten
        separator: Separator for nested keys
        prefix: Prefix for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.extend(
                        flatten_dict(item, separator, f"{new_key}[{i}]").items()
                    )
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, value))
    
    return dict(items)

# ===============================================================================
# ASYNC UTILITIES
# ===============================================================================

async def retry_async_operation(
    operation: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs
) -> Any:
    """
    Retry an async operation with exponential backoff.
    
    Args:
        operation: Async function to retry
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of the operation
        
    Raises:
        Last exception if all retries fail
    """
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            last_error = e
            
            if attempt < max_attempts - 1:
                delay = base_delay * (backoff_factor ** attempt)
                
                logger.debug(
                    "Operation failed, retrying",
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e)
                )
                
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Operation failed after all retries",
                    attempts=max_attempts,
                    error=str(e)
                )
    
    raise last_error


async def run_with_timeout(
    coro: Callable,
    timeout: float,
    *args,
    **kwargs
) -> Any:
    """
    Run async operation with timeout.
    
    Args:
        coro: Coroutine function to run
        timeout: Timeout in seconds
        *args: Arguments for the coroutine
        **kwargs: Keyword arguments for the coroutine
        
    Returns:
        Result of the operation
        
    Raises:
        asyncio.TimeoutError if operation times out
    """
    return await asyncio.wait_for(coro(*args, **kwargs), timeout=timeout)

# ===============================================================================
# PERFORMANCE UTILITIES
# ===============================================================================

def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"Function {func.__name__} executed",
                execution_time=f"{execution_time:.3f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed",
                execution_time=f"{execution_time:.3f}s",
                error=str(e)
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"Function {func.__name__} executed",
                execution_time=f"{execution_time:.3f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed",
                execution_time=f"{execution_time:.3f}s",
                error=str(e)
            )
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

# ===============================================================================
# FORMAT UTILITIES
# ===============================================================================

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.23 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2.5h", "45m")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

# ===============================================================================
# VALIDATION UTILITIES
# ===============================================================================

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate configuration has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key names
        
    Returns:
        True if valid, False otherwise
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        logger.error(
            "Configuration validation failed",
            missing_keys=missing_keys
        )
        return False
    
    return True


def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string by removing dangerous characters.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length to truncate to
        
    Returns:
        Sanitized string
    """
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    # ID generation
    'generate_unique_id',
    'generate_cycle_id',
    'generate_task_id',
    
    # Timestamp utilities
    'get_utc_timestamp',
    'format_timestamp',
    'timestamp_to_seconds',
    
    # Hashing
    'calculate_hash',
    'verify_integrity',
    
    # JSON utilities
    'safe_json_loads',
    'safe_json_dumps',
    
    # Dictionary utilities
    'deep_merge_dict',
    'flatten_dict',
    
    # Async utilities
    'retry_async_operation',
    'run_with_timeout',
    
    # Performance
    'measure_time',
    
    # Formatting
    'format_file_size',
    'format_duration',
    
    # Validation
    'validate_config',
    'sanitize_string',
]
