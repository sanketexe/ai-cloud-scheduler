"""
Structured logging service for the FinOps Platform.

This module provides centralized structured logging with correlation IDs,
contextual information, and JSON formatting for production environments.
"""

import logging
import json
import uuid
from typing import Dict, Any, Optional, Union
from datetime import datetime
from contextvars import ContextVar
from functools import wraps
import structlog
from structlog.stdlib import LoggerFactory
from structlog.processors import JSONRenderer, TimeStamper, add_log_level, StackInfoRenderer
from structlog.contextvars import merge_contextvars


# Context variables for request tracking
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default=None)
user_id_var: ContextVar[str] = ContextVar('user_id', default=None)
request_id_var: ContextVar[str] = ContextVar('request_id', default=None)


class CorrelationIDProcessor:
    """Processor to add correlation ID to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        correlation_id = correlation_id_var.get(None)
        if correlation_id:
            event_dict['correlation_id'] = correlation_id
        
        user_id = user_id_var.get(None)
        if user_id:
            event_dict['user_id'] = user_id
            
        request_id = request_id_var.get(None)
        if request_id:
            event_dict['request_id'] = request_id
            
        return event_dict


class LoggingService:
    """
    Centralized structured logging service with correlation IDs and context.
    
    Provides structured logging capabilities with automatic context injection,
    correlation ID tracking, and JSON formatting for production environments.
    """
    
    def __init__(self, log_level: str = "INFO", service_name: str = "finops-platform"):
        """
        Initialize the logging service.
        
        Args:
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            service_name: Name of the service for log identification
        """
        self.service_name = service_name
        self.log_level = getattr(logging, log_level.upper())
        
        # Configure structlog
        self._configure_structlog()
        
        # Get the structured logger
        self.logger = structlog.get_logger()
    
    def _configure_structlog(self):
        """Configure structlog with processors and formatters."""
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=None,
            level=self.log_level,
        )
        
        # Configure structlog
        structlog.configure(
            processors=[
                # Add log level to event dict
                add_log_level,
                # Add timestamp
                TimeStamper(fmt="iso"),
                # Add correlation ID and context
                CorrelationIDProcessor(),
                # Merge context variables
                merge_contextvars,
                # Add stack info for errors
                StackInfoRenderer(),
                # Render as JSON
                JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )
    
    def set_correlation_id(self, correlation_id: str = None) -> str:
        """
        Set correlation ID for request tracking.
        
        Args:
            correlation_id: Correlation ID to set. If None, generates a new UUID.
            
        Returns:
            The correlation ID that was set
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        correlation_id_var.set(correlation_id)
        return correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """
        Get the current correlation ID.
        
        Returns:
            Current correlation ID or None if not set
        """
        return correlation_id_var.get(None)
    
    def set_user_id(self, user_id: str) -> None:
        """
        Set user ID for request context.
        
        Args:
            user_id: User ID to set in context
        """
        user_id_var.set(user_id)
    
    def get_user_id(self) -> Optional[str]:
        """
        Get the current user ID.
        
        Returns:
            Current user ID or None if not set
        """
        return user_id_var.get(None)
    
    def set_request_id(self, request_id: str = None) -> str:
        """
        Set request ID for request tracking.
        
        Args:
            request_id: Request ID to set. If None, generates a new UUID.
            
        Returns:
            The request ID that was set
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        request_id_var.set(request_id)
        return request_id
    
    def get_request_id(self) -> Optional[str]:
        """
        Get the current request ID.
        
        Returns:
            Current request ID or None if not set
        """
        return request_id_var.get(None)
    
    def bind_context(self, **kwargs) -> structlog.BoundLogger:
        """
        Bind additional context to the logger.
        
        Args:
            **kwargs: Key-value pairs to bind to the logger
            
        Returns:
            Bound logger with additional context
        """
        return self.logger.bind(**kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log debug message.
        
        Args:
            message: Log message
            **kwargs: Additional context
        """
        self.logger.debug(message, service=self.service_name, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log info message.
        
        Args:
            message: Log message
            **kwargs: Additional context
        """
        self.logger.info(message, service=self.service_name, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log warning message.
        
        Args:
            message: Log message
            **kwargs: Additional context
        """
        self.logger.warning(message, service=self.service_name, **kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs) -> None:
        """
        Log error message.
        
        Args:
            message: Log message
            error: Exception object for additional context
            **kwargs: Additional context
        """
        if error:
            kwargs.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_details': getattr(error, 'details', {})
            })
            
        self.logger.error(message, service=self.service_name, **kwargs)
    
    def critical(self, message: str, error: Exception = None, **kwargs) -> None:
        """
        Log critical message.
        
        Args:
            message: Log message
            error: Exception object for additional context
            **kwargs: Additional context
        """
        if error:
            kwargs.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_details': getattr(error, 'details', {})
            })
            
        self.logger.critical(message, service=self.service_name, **kwargs)
    
    def audit(
        self, 
        action: str, 
        resource_type: str, 
        resource_id: str = None,
        user_id: str = None,
        **kwargs
    ) -> None:
        """
        Log audit event for security and compliance.
        
        Args:
            action: Action performed (CREATE, READ, UPDATE, DELETE, etc.)
            resource_type: Type of resource affected
            resource_id: ID of the resource (if applicable)
            user_id: User who performed the action
            **kwargs: Additional audit context
        """
        audit_data = {
            'event_type': 'audit',
            'action': action,
            'resource_type': resource_type,
            'service': self.service_name,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        if resource_id:
            audit_data['resource_id'] = resource_id
            
        if user_id:
            audit_data['user_id'] = user_id
        elif self.get_user_id():
            audit_data['user_id'] = self.get_user_id()
            
        audit_data.update(kwargs)
        
        self.logger.info("Audit event", **audit_data)
    
    def performance(
        self, 
        operation: str, 
        duration_ms: float, 
        **kwargs
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            **kwargs: Additional performance context
        """
        self.logger.info(
            "Performance metric",
            event_type='performance',
            operation=operation,
            duration_ms=duration_ms,
            service=self.service_name,
            **kwargs
        )
    
    def security(
        self, 
        event: str, 
        severity: str = "INFO",
        **kwargs
    ) -> None:
        """
        Log security event.
        
        Args:
            event: Security event description
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
            **kwargs: Additional security context
        """
        security_data = {
            'event_type': 'security',
            'security_event': event,
            'severity': severity,
            'service': self.service_name,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        security_data.update(kwargs)
        
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method("Security event", **security_data)


def log_function_call(logger: LoggingService = None):
    """
    Decorator to automatically log function calls with parameters and results.
    
    Args:
        logger: LoggingService instance. If None, creates a new one.
        
    Returns:
        Decorator function
    """
    if logger is None:
        logger = LoggingService()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log function entry
            logger.debug(
                f"Entering function {func_name}",
                function=func_name,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                logger.debug(
                    f"Function {func_name} completed successfully",
                    function=func_name,
                    result_type=type(result).__name__ if result is not None else None
                )
                
                return result
                
            except Exception as e:
                # Log function error
                logger.error(
                    f"Function {func_name} failed",
                    function=func_name,
                    error=e
                )
                raise
        
        return wrapper
    return decorator


def log_execution_time(logger: LoggingService = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: LoggingService instance. If None, creates a new one.
        
    Returns:
        Decorator function
    """
    if logger is None:
        logger = LoggingService()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            func_name = f"{func.__module__}.{func.__qualname__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.performance(
                    operation=func_name,
                    duration_ms=duration_ms,
                    status="success"
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.performance(
                    operation=func_name,
                    duration_ms=duration_ms,
                    status="error",
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator


# Global logging service instance
logging_service = LoggingService()


def get_logger() -> LoggingService:
    """
    Get the global logging service instance.
    
    Returns:
        Global LoggingService instance
    """
    return logging_service


def configure_logging(log_level: str = "INFO", service_name: str = "finops-platform"):
    """
    Configure the global logging service.
    
    Args:
        log_level: Minimum log level
        service_name: Service name for identification
    """
    global logging_service
    logging_service = LoggingService(log_level=log_level, service_name=service_name)