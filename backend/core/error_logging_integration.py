"""
Integration module for error handling and logging components.

This module provides a unified interface to set up and configure
all error handling and logging components for the FinOps Platform.
"""

from fastapi import FastAPI
from .logging_service import configure_logging, get_logger
from .exception_handlers import register_exception_handlers
from .middleware import setup_middleware


def setup_error_handling_and_logging(
    app: FastAPI,
    log_level: str = "INFO",
    service_name: str = "finops-platform",
    enable_request_logging: bool = True,
    enable_audit_logging: bool = True,
    enable_rate_limiting: bool = True
) -> None:
    """
    Set up comprehensive error handling and logging for the FastAPI application.
    
    This function configures:
    - Structured logging with correlation IDs
    - Global exception handlers
    - Request/response logging middleware
    - Audit logging middleware
    - Rate limiting middleware
    
    Args:
        app: FastAPI application instance
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service_name: Service name for log identification
        enable_request_logging: Whether to enable request/response logging
        enable_audit_logging: Whether to enable audit logging
        enable_rate_limiting: Whether to enable rate limiting
    """
    logger = get_logger()
    
    # Configure structured logging
    configure_logging(log_level=log_level, service_name=service_name)
    logger.info(
        "Configuring error handling and logging",
        log_level=log_level,
        service_name=service_name,
        request_logging=enable_request_logging,
        audit_logging=enable_audit_logging,
        rate_limiting=enable_rate_limiting
    )
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Set up middleware
    if enable_request_logging or enable_audit_logging or enable_rate_limiting:
        setup_middleware(app)
    
    logger.info("Error handling and logging setup completed successfully")


def get_application_logger():
    """
    Get the configured application logger.
    
    Returns:
        LoggingService instance
    """
    return get_logger()


# Example usage functions for demonstration
def example_logging_usage():
    """
    Example of how to use the logging service in application code.
    """
    logger = get_logger()
    
    # Set correlation ID for request tracking
    correlation_id = logger.set_correlation_id()
    logger.info("Processing request", operation="example_operation")
    
    # Set user context
    logger.set_user_id("user-123")
    
    # Log with additional context
    bound_logger = logger.bind_context(
        component="cost_calculator",
        provider="aws"
    )
    bound_logger.info("Calculating costs", resource_count=150)
    
    # Log performance metrics
    logger.performance(
        operation="cost_calculation",
        duration_ms=245.7,
        resource_count=150,
        provider="aws"
    )
    
    # Log audit events
    logger.audit(
        action="CREATE",
        resource_type="budget",
        resource_id="budget-456",
        amount=10000,
        currency="USD"
    )
    
    # Log security events
    logger.security(
        event="Failed login attempt",
        severity="WARNING",
        username="admin",
        ip_address="192.168.1.100"
    )


def example_exception_usage():
    """
    Example of how to use custom exceptions in application code.
    """
    from .exceptions import (
        CloudProviderException,
        ValidationException,
        ResourceNotFoundException,
        DatabaseException
    )
    
    # Cloud provider exception with context
    try:
        # Simulate cloud provider API call
        raise CloudProviderException(
            message="Failed to retrieve cost data from AWS",
            provider_type="aws",
            provider_id="provider-123",
            api_endpoint="/cost-explorer/get-cost-and-usage"
        ).add_context("retry_count", 3).add_context("last_error", "Timeout")
    except CloudProviderException as e:
        logger = get_logger()
        logger.error("Cloud provider error occurred", error=e)
    
    # Validation exception with field details
    try:
        raise ValidationException(
            message="Budget amount must be positive",
            field="amount",
            validation_errors={"amount": ["Must be greater than 0"]}
        )
    except ValidationException as e:
        logger = get_logger()
        logger.warning("Validation failed", error=e)
    
    # Resource not found exception
    try:
        raise ResourceNotFoundException(
            message="Budget not found",
            resource_type="budget",
            resource_id="budget-789"
        )
    except ResourceNotFoundException as e:
        logger = get_logger()
        logger.warning("Resource not found", error=e)


# Health check endpoint example
async def health_check_with_logging():
    """
    Example health check endpoint that demonstrates logging.
    """
    logger = get_logger()
    
    try:
        # Check database connection
        # db_status = await check_database_connection()
        db_status = True  # Placeholder
        
        # Check cache connection
        # cache_status = await check_cache_connection()
        cache_status = True  # Placeholder
        
        # Check external services
        # aws_status = await check_aws_connection()
        aws_status = True  # Placeholder
        
        health_status = {
            "status": "healthy" if all([db_status, cache_status, aws_status]) else "unhealthy",
            "database": "up" if db_status else "down",
            "cache": "up" if cache_status else "down",
            "aws": "up" if aws_status else "down"
        }
        
        logger.info("Health check completed", **health_status)
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=e)
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    # Example of setting up logging for testing
    from fastapi import FastAPI
    
    app = FastAPI(title="FinOps Platform")
    
    # Set up error handling and logging
    setup_error_handling_and_logging(
        app=app,
        log_level="DEBUG",
        service_name="finops-platform-dev"
    )
    
    # Run example usage
    example_logging_usage()
    example_exception_usage()