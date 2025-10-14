                socktype=socktype
            )
            
            self.handler_logger.info(
                "Syslog handler initialized",
                address=address,
                facility=self.facility,
                socktype=socktype
            )
            
        except Exception as e:
            self.handler_logger.error("Failed to setup syslog handler", error=str(e))
            raise
    
    def _process_log_entries(self, entries: List[LogEntry]) -> None:
        """Send entries to syslog"""
        if not self._syslog_handler:
            raise RuntimeError("Syslog handler not initialized")
        
        try:
            for entry in entries:
                # Create syslog record
                syslog_record = logging.LogRecord(
                    name=entry.record.name,
                    level=entry.record.levelno,
                    pathname=entry.record.pathname,
                    lineno=entry.record.lineno,
                    msg=entry.formatted_message,
                    args=(),
                    exc_info=None
                )
                syslog_record.created = entry.record.created
                
                self._syslog_handler.emit(syslog_record)
            
        except Exception as e:
            self.handler_logger.error("Failed to send syslog entries", error=str(e))
            raise
    
    def close(self) -> None:
        """Close syslog handler"""
        super().close()
        
        if self._syslog_handler:
            self._syslog_handler.close()

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def create_file_handler(config: FileHandlerConfig, rotation_type: str = "size") -> BaseLogHandler:
    """Factory function to create file handlers"""
    
    if rotation_type.lower() == "time":
        return TimedRotatingFileHandler(config)
    else:
        return EnhancedRotatingFileHandler(config)

def create_remote_handler(config: RemoteHandlerConfig, handler_type: str = "http") -> BaseLogHandler:
    """Factory function to create remote handlers"""
    
    handler_types = {
        'http': RemoteHTTPHandler,
        'https': RemoteHTTPHandler,
        'syslog': SyslogHandler
    }
    
    handler_class = handler_types.get(handler_type.lower())
    if not handler_class:
        raise ValueError(f"Unknown remote handler type: {handler_type}")
    
    return handler_class(config)

def create_handler_from_config(handler_config: Dict[str, Any]) -> BaseLogHandler:
    """Create handler from configuration dictionary"""
    
    handler_type = handler_config.get('type', 'file').lower()
    
    if handler_type in ['file', 'rotating_file']:
        config = FileHandlerConfig(**handler_config)
        return EnhancedRotatingFileHandler(config)
    
    elif handler_type == 'timed_rotating_file':
        config = FileHandlerConfig(**handler_config)
        return TimedRotatingFileHandler(config)
    
    elif handler_type in ['http', 'https']:
        config = RemoteHandlerConfig(**handler_config)
        return RemoteHTTPHandler(config)
    
    elif handler_type == 'syslog':
        config = RemoteHandlerConfig(**handler_config)
        return SyslogHandler(config)
    
    else:
        raise ValueError(f"Unknown handler type: {handler_type}")

@track_performance
async def health_check_handlers(handlers: List[BaseLogHandler]) -> Dict[str, Any]:
    """Perform health check on all handlers"""
    
    health_status = {
        'overall_status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'handlers': {}
    }
    
    unhealthy_count = 0
    
    for handler in handlers:
        try:
            stats = handler.get_stats()
            
            # Determine handler health
            handler_healthy = (
                handler.config.enabled and
                stats['records_failed'] < stats['records_processed'] * 0.1  # Less than 10% failure rate
            )
            
            if not handler_healthy:
                unhealthy_count += 1
            
            health_status['handlers'][handler.config.name] = {
                'status': 'healthy' if handler_healthy else 'unhealthy',
                'stats': stats
            }
            
        except Exception as e:
            unhealthy_count += 1
            health_status['handlers'][handler.config.name] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Set overall status
    if unhealthy_count > 0:
        if unhealthy_count == len(handlers):
            health_status['overall_status'] = 'critical'
        else:
            health_status['overall_status'] = 'degraded'
    
    return health_status

# ===============================================================================
# MODULE INITIALIZATION
# ===============================================================================

def initialize_handlers(config: Dict[str, Any]) -> List[BaseLogHandler]:
    """Initialize handlers from configuration"""
    
    handlers = []
    
    for handler_name, handler_config in config.get('handlers', {}).items():
        try:
            handler_config['name'] = handler_name
            handler = create_handler_from_config(handler_config)
            handlers.append(handler)
            
            logger.info(
                "Handler initialized",
                handler_name=handler_name,
                handler_type=handler_config.get('type', 'file')
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize handler",
                handler_name=handler_name,
                error=str(e)
            )
    
    return handlers

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "BaseLogHandler",
    "EnhancedRotatingFileHandler",
    "TimedRotatingFileHandler", 
    "RemoteHTTPHandler",
    "SyslogHandler",
    "HandlerConfig",
    "FileHandlerConfig",
    "RemoteHandlerConfig",
    "LogEntry",
    "create_file_handler",
    "create_remote_handler",
    "create_handler_from_config",
    "health_check_handlers",
    "initialize_handlers"
]
        
