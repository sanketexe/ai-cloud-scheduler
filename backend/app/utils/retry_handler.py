"""
Retry Handler with Exponential Backoff for AWS API Calls
Validates: Requirements 4.4
"""

import time
import functools
from typing import Callable, Type, Tuple, Optional
from botocore.exceptions import ClientError
import structlog

logger = structlog.get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        backoff_multiplier: float = 2.0,
        retryable_errors: Tuple[str, ...] = (
            'ThrottlingException',
            'RequestLimitExceeded',
            'ServiceUnavailable',
            'InternalServerError',
            'RequestTimeout',
        )
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retryable_errors = retryable_errors


DEFAULT_RETRY_CONFIG = RetryConfig()


def is_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """
    Check if an error is retryable
    
    Args:
        error: The exception to check
        config: Retry configuration
        
    Returns:
        True if the error should be retried
    """
    if isinstance(error, ClientError):
        error_code = error.response.get('Error', {}).get('Code', '')
        return error_code in config.retryable_errors
    
    # Network errors are generally retryable
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    
    return False


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for retry attempt using exponential backoff
    
    Args:
        attempt: Current attempt number (1-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    delay = config.initial_delay * (config.backoff_multiplier ** (attempt - 1))
    return min(delay, config.max_delay)


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator to retry function with exponential backoff
    
    Args:
        config: Retry configuration (uses default if None)
        on_retry: Optional callback called on each retry
        
    Usage:
        @retry_with_backoff()
        def call_aws_api():
            ...
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_retries + 2):  # +1 for initial attempt, +1 for range
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry if this is the last attempt
                    if attempt > config.max_retries:
                        break
                    
                    # Check if error is retryable
                    if not is_retryable_error(e, config):
                        logger.warning(
                            "Non-retryable error encountered",
                            error_type=type(e).__name__,
                            error_message=str(e)
                        )
                        break
                    
                    # Calculate delay
                    delay = calculate_delay(attempt, config)
                    
                    logger.info(
                        "Retrying after error",
                        attempt=attempt,
                        max_retries=config.max_retries,
                        delay_seconds=delay,
                        error_type=type(e).__name__,
                        error_message=str(e)
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt, e)
                    
                    # Wait before retrying
                    time.sleep(delay)
            
            # All retries exhausted, raise the last exception
            logger.error(
                "All retry attempts exhausted",
                max_retries=config.max_retries,
                error_type=type(last_exception).__name__,
                error_message=str(last_exception)
            )
            raise last_exception
        
        return wrapper
    return decorator


def retry_with_rate_limit_handling(
    config: Optional[RetryConfig] = None
):
    """
    Decorator specifically for handling AWS rate limiting
    
    Respects Retry-After headers from AWS responses
    
    Usage:
        @retry_with_rate_limit_handling()
        def call_cost_explorer():
            ...
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_retries + 2):
                try:
                    return func(*args, **kwargs)
                    
                except ClientError as e:
                    last_exception = e
                    error_code = e.response.get('Error', {}).get('Code', '')
                    
                    # Don't retry if this is the last attempt
                    if attempt > config.max_retries:
                        break
                    
                    # Check if it's a rate limit error
                    is_rate_limit = error_code in ('ThrottlingException', 'RequestLimitExceeded')
                    
                    if not is_rate_limit and not is_retryable_error(e, config):
                        break
                    
                    # Use Retry-After header if available
                    if is_rate_limit:
                        retry_after = e.response.get('ResponseMetadata', {}).get('RetryAfter')
                        if retry_after:
                            delay = float(retry_after)
                        else:
                            delay = calculate_delay(attempt, config)
                    else:
                        delay = calculate_delay(attempt, config)
                    
                    logger.warning(
                        "Rate limit encountered, retrying",
                        attempt=attempt,
                        max_retries=config.max_retries,
                        delay_seconds=delay,
                        error_code=error_code
                    )
                    
                    time.sleep(delay)
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt > config.max_retries:
                        break
                    
                    if not is_retryable_error(e, config):
                        break
                    
                    delay = calculate_delay(attempt, config)
                    logger.info(
                        "Retrying after error",
                        attempt=attempt,
                        delay_seconds=delay,
                        error_type=type(e).__name__
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class RetryQueue:
    """
    Queue for managing retry requests
    
    Useful for rate-limited APIs where requests need to be queued
    and processed with delays.
    """
    
    def __init__(self, delay_between_requests: float = 0.5):
        self.delay_between_requests = delay_between_requests
        self.last_request_time = 0.0
    
    def execute_with_delay(self, func: Callable, *args, **kwargs):
        """
        Execute function with delay to respect rate limits
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        # Calculate time since last request
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Wait if needed
        if time_since_last < self.delay_between_requests:
            wait_time = self.delay_between_requests - time_since_last
            logger.debug(
                "Waiting before next request",
                wait_seconds=wait_time
            )
            time.sleep(wait_time)
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self.last_request_time = time.time()
            return result
        except Exception as e:
            self.last_request_time = time.time()
            raise


# Global retry queue for AWS API calls
aws_api_queue = RetryQueue(delay_between_requests=0.5)
