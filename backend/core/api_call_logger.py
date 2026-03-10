"""
API Call Logger for AWS API Troubleshooting
Validates: Requirements 4.5
"""

import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class APICallLog:
    """Log entry for an API call"""
    timestamp: str
    service: str
    operation: str
    parameters: Dict[str, Any]
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    response_size: Optional[int] = None
    retry_count: int = 0


class APICallLogger:
    """
    Logger for tracking AWS API calls for troubleshooting
    
    Maintains an in-memory log of recent API calls with details
    about timing, errors, and retry attempts.
    """
    
    def __init__(self, max_logs: int = 1000):
        self.max_logs = max_logs
        self.logs: deque = deque(maxlen=max_logs)
        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_duration_ms': 0,
            'retry_count': 0,
        }
    
    def log_call_start(
        self,
        service: str,
        operation: str,
        parameters: Dict[str, Any]
    ) -> float:
        """
        Log the start of an API call
        
        Args:
            service: AWS service name (e.g., 'cost-explorer', 'ec2')
            operation: Operation name (e.g., 'get_cost_and_usage')
            parameters: API call parameters
            
        Returns:
            Start time for duration calculation
        """
        start_time = time.time()
        
        logger.info(
            "AWS API call started",
            service=service,
            operation=operation,
            parameters=self._sanitize_parameters(parameters)
        )
        
        return start_time
    
    def log_call_end(
        self,
        service: str,
        operation: str,
        parameters: Dict[str, Any],
        start_time: float,
        success: bool = True,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        response_size: Optional[int] = None,
        retry_count: int = 0
    ):
        """
        Log the completion of an API call
        
        Args:
            service: AWS service name
            operation: Operation name
            parameters: API call parameters
            start_time: Start time from log_call_start
            success: Whether the call succeeded
            error_message: Error message if failed
            error_code: Error code if failed
            response_size: Size of response in bytes
            retry_count: Number of retries attempted
        """
        duration_ms = (time.time() - start_time) * 1000
        
        log_entry = APICallLog(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            service=service,
            operation=operation,
            parameters=self._sanitize_parameters(parameters),
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            error_code=error_code,
            response_size=response_size,
            retry_count=retry_count
        )
        
        self.logs.append(log_entry)
        
        # Update statistics
        self._stats['total_calls'] += 1
        if success:
            self._stats['successful_calls'] += 1
        else:
            self._stats['failed_calls'] += 1
        self._stats['total_duration_ms'] += duration_ms
        self._stats['retry_count'] += retry_count
        
        # Log to structured logger
        if success:
            logger.info(
                "AWS API call completed",
                service=service,
                operation=operation,
                duration_ms=round(duration_ms, 2),
                response_size=response_size,
                retry_count=retry_count
            )
        else:
            logger.error(
                "AWS API call failed",
                service=service,
                operation=operation,
                duration_ms=round(duration_ms, 2),
                error_code=error_code,
                error_message=error_message,
                retry_count=retry_count
            )
    
    def get_logs(
        self,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        failed_only: bool = False,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get API call logs with optional filtering
        
        Args:
            service: Filter by service name
            operation: Filter by operation name
            failed_only: Only return failed calls
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries as dictionaries
        """
        filtered_logs = []
        
        for log in self.logs:
            # Apply filters
            if service and log.service != service:
                continue
            if operation and log.operation != operation:
                continue
            if failed_only and log.success:
                continue
            
            filtered_logs.append(asdict(log))
        
        # Apply limit
        if limit:
            filtered_logs = filtered_logs[-limit:]
        
        return filtered_logs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get API call statistics
        
        Returns:
            Dictionary with call statistics
        """
        total_calls = self._stats['total_calls']
        
        return {
            'total_calls': total_calls,
            'successful_calls': self._stats['successful_calls'],
            'failed_calls': self._stats['failed_calls'],
            'success_rate': (
                self._stats['successful_calls'] / total_calls * 100
                if total_calls > 0 else 0
            ),
            'average_duration_ms': (
                self._stats['total_duration_ms'] / total_calls
                if total_calls > 0 else 0
            ),
            'total_retries': self._stats['retry_count'],
            'average_retries_per_call': (
                self._stats['retry_count'] / total_calls
                if total_calls > 0 else 0
            )
        }
    
    def get_error_summary(self) -> Dict[str, int]:
        """
        Get summary of errors by error code
        
        Returns:
            Dictionary mapping error codes to counts
        """
        error_counts: Dict[str, int] = {}
        
        for log in self.logs:
            if not log.success and log.error_code:
                error_counts[log.error_code] = error_counts.get(log.error_code, 0) + 1
        
        return error_counts
    
    def clear_logs(self):
        """Clear all logs and reset statistics"""
        self.logs.clear()
        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_duration_ms': 0,
            'retry_count': 0,
        }
        logger.info("API call logs cleared")
    
    def export_logs(self, filepath: str):
        """
        Export logs to JSON file
        
        Args:
            filepath: Path to export file
        """
        logs_data = {
            'logs': self.get_logs(),
            'statistics': self.get_statistics(),
            'error_summary': self.get_error_summary(),
            'exported_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2)
        
        logger.info("API call logs exported", filepath=filepath)
    
    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters to remove sensitive data
        
        Args:
            parameters: Original parameters
            
        Returns:
            Sanitized parameters
        """
        sanitized = {}
        sensitive_keys = {'AccessKeyId', 'SecretAccessKey', 'SessionToken', 'Password'}
        
        for key, value in parameters.items():
            if key in sensitive_keys:
                sanitized[key] = '***REDACTED***'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_parameters(value)
            else:
                sanitized[key] = value
        
        return sanitized


# Global logger instance
api_call_logger = APICallLogger()


def log_aws_api_call(service: str, operation: str):
    """
    Decorator for logging AWS API calls
    
    Usage:
        @log_aws_api_call('cost-explorer', 'get_cost_and_usage')
        def get_costs(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract parameters
            parameters = {
                'args': str(args)[:100],  # Truncate for brevity
                'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
            }
            
            start_time = api_call_logger.log_call_start(
                service=service,
                operation=operation,
                parameters=parameters
            )
            
            retry_count = 0
            try:
                result = func(*args, **kwargs)
                
                # Try to get response size
                response_size = None
                if hasattr(result, '__len__'):
                    try:
                        response_size = len(str(result))
                    except:
                        pass
                
                api_call_logger.log_call_end(
                    service=service,
                    operation=operation,
                    parameters=parameters,
                    start_time=start_time,
                    success=True,
                    response_size=response_size,
                    retry_count=retry_count
                )
                
                return result
                
            except Exception as e:
                error_code = getattr(e, 'response', {}).get('Error', {}).get('Code', 'Unknown')
                error_message = str(e)
                
                api_call_logger.log_call_end(
                    service=service,
                    operation=operation,
                    parameters=parameters,
                    start_time=start_time,
                    success=False,
                    error_message=error_message,
                    error_code=error_code,
                    retry_count=retry_count
                )
                
                raise
        
        return wrapper
    return decorator
