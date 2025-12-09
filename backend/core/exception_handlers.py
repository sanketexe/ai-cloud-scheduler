"""
Global exception handlers for FastAPI application.

This module provides comprehensive exception handling for all custom exceptions,
validation errors, and security-related exceptions with proper HTTP status codes
and structured error responses.
"""

from typing import Union
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError
import traceback

from .exceptions import (
    FinOpsException,
    CloudProviderException,
    CloudProviderConnectionException,
    CloudProviderAuthenticationException,
    CloudProviderRateLimitException,
    AuthenticationException,
    InvalidTokenException,
    InsufficientPermissionsException,
    ValidationException,
    SchemaValidationException,
    BusinessRuleException,
    DatabaseException,
    DatabaseConnectionException,
    DatabaseIntegrityException,
    CacheException,
    CacheConnectionException,
    ConfigurationException,
    ExternalServiceException,
    RateLimitException,
    ResourceNotFoundException,
    ConflictException
)
from .logging_service import get_logger


logger = get_logger()


async def finops_exception_handler(request: Request, exc: FinOpsException) -> JSONResponse:
    """
    Handle all FinOps custom exceptions.
    
    Args:
        request: FastAPI request object
        exc: FinOps exception instance
        
    Returns:
        JSON response with error details
    """
    # Set correlation ID from exception if available
    if exc.correlation_id:
        logger.set_correlation_id(exc.correlation_id)
    
    # Log the exception
    logger.error(
        f"FinOps exception occurred: {exc.message}",
        error_code=exc.error_code,
        error_details=exc.details,
        path=request.url.path,
        method=request.method,
        user_id=exc.user_id
    )
    
    # Determine HTTP status code based on exception type
    status_code = _get_status_code_for_exception(exc)
    
    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle FastAPI request validation errors.
    
    Args:
        request: FastAPI request object
        exc: Request validation error
        
    Returns:
        JSON response with validation error details
    """
    logger.warning(
        "Request validation failed",
        path=request.url.path,
        method=request.method,
        validation_errors=exc.errors()
    )
    
    # Format validation errors
    formatted_errors = []
    for error in exc.errors():
        formatted_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    error_response = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": {
                "validation_errors": formatted_errors
            },
            "timestamp": logger.logger.bind().info("", timestamp=True)["timestamp"]
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response
    )


async def response_validation_exception_handler(request: Request, exc: ResponseValidationError) -> JSONResponse:
    """
    Handle FastAPI response validation errors.
    
    Args:
        request: FastAPI request object
        exc: Response validation error
        
    Returns:
        JSON response with error details
    """
    logger.error(
        "Response validation failed - this indicates a server-side issue",
        path=request.url.path,
        method=request.method,
        validation_errors=exc.errors()
    )
    
    error_response = {
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Internal server error occurred",
            "details": {
                "message": "Response validation failed"
            },
            "timestamp": logger.logger.bind().info("", timestamp=True)["timestamp"]
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTP exceptions.
    
    Args:
        request: FastAPI request object
        exc: HTTP exception
        
    Returns:
        JSON response with error details
    """
    logger.warning(
        f"HTTP exception: {exc.detail}",
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method
    )
    
    error_response = {
        "error": {
            "code": _get_error_code_for_status(exc.status_code),
            "message": exc.detail,
            "details": {},
            "timestamp": logger.logger.bind().info("", timestamp=True)["timestamp"]
        }
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle Starlette HTTP exceptions.
    
    Args:
        request: FastAPI request object
        exc: Starlette HTTP exception
        
    Returns:
        JSON response with error details
    """
    logger.warning(
        f"Starlette HTTP exception: {exc.detail}",
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method
    )
    
    error_response = {
        "error": {
            "code": _get_error_code_for_status(exc.status_code),
            "message": exc.detail,
            "details": {},
            "timestamp": logger.logger.bind().info("", timestamp=True)["timestamp"]
        }
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all unhandled exceptions.
    
    Args:
        request: FastAPI request object
        exc: Unhandled exception
        
    Returns:
        JSON response with error details
    """
    # Log the full exception with stack trace
    logger.critical(
        f"Unhandled exception: {str(exc)}",
        exception_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        stack_trace=traceback.format_exc()
    )
    
    error_response = {
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "details": {
                "exception_type": type(exc).__name__
            },
            "timestamp": logger.logger.bind().info("", timestamp=True)["timestamp"]
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )


def _get_status_code_for_exception(exc: FinOpsException) -> int:
    """
    Get appropriate HTTP status code for FinOps exception.
    
    Args:
        exc: FinOps exception
        
    Returns:
        HTTP status code
    """
    # Authentication and authorization exceptions
    if isinstance(exc, InvalidTokenException):
        return status.HTTP_401_UNAUTHORIZED
    elif isinstance(exc, InsufficientPermissionsException):
        return status.HTTP_403_FORBIDDEN
    elif isinstance(exc, AuthenticationException):
        return status.HTTP_401_UNAUTHORIZED
    
    # Validation exceptions
    elif isinstance(exc, SchemaValidationException):
        return status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, BusinessRuleException):
        return status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, ValidationException):
        return status.HTTP_400_BAD_REQUEST
    
    # Resource exceptions
    elif isinstance(exc, ResourceNotFoundException):
        return status.HTTP_404_NOT_FOUND
    elif isinstance(exc, ConflictException):
        return status.HTTP_409_CONFLICT
    
    # Rate limiting
    elif isinstance(exc, RateLimitException):
        return status.HTTP_429_TOO_MANY_REQUESTS
    elif isinstance(exc, CloudProviderRateLimitException):
        return status.HTTP_429_TOO_MANY_REQUESTS
    
    # Cloud provider exceptions
    elif isinstance(exc, CloudProviderAuthenticationException):
        return status.HTTP_401_UNAUTHORIZED
    elif isinstance(exc, CloudProviderConnectionException):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, CloudProviderException):
        return status.HTTP_502_BAD_GATEWAY
    
    # Database exceptions
    elif isinstance(exc, DatabaseConnectionException):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, DatabaseIntegrityException):
        return status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, DatabaseException):
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Cache exceptions
    elif isinstance(exc, CacheConnectionException):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, CacheException):
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Configuration and external service exceptions
    elif isinstance(exc, ConfigurationException):
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    elif isinstance(exc, ExternalServiceException):
        return status.HTTP_502_BAD_GATEWAY
    
    # Default to internal server error
    else:
        return status.HTTP_500_INTERNAL_SERVER_ERROR


def _get_error_code_for_status(status_code: int) -> str:
    """
    Get error code for HTTP status code.
    
    Args:
        status_code: HTTP status code
        
    Returns:
        Error code string
    """
    status_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "UNPROCESSABLE_ENTITY",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT"
    }
    
    return status_code_map.get(status_code, "UNKNOWN_ERROR")


def register_exception_handlers(app):
    """
    Register all exception handlers with FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Register custom exception handlers
    app.add_exception_handler(FinOpsException, finops_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ResponseValidationError, response_validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    
    # Register general exception handler for all unhandled exceptions
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Exception handlers registered successfully")


# Security-specific exception handlers
async def security_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle security-related exceptions with enhanced logging.
    
    Args:
        request: FastAPI request object
        exc: Security exception
        
    Returns:
        JSON response with minimal error details for security
    """
    # Enhanced security logging
    logger.security(
        "Security exception occurred",
        severity="ERROR",
        exception_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        referer=request.headers.get("referer")
    )
    
    # Return minimal error information for security
    error_response = {
        "error": {
            "code": "SECURITY_ERROR",
            "message": "Access denied",
            "details": {},
            "timestamp": logger.logger.bind().info("", timestamp=True)["timestamp"]
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content=error_response
    )


# Rate limiting exception handler
async def rate_limit_exception_handler(request: Request, exc: RateLimitException) -> JSONResponse:
    """
    Handle rate limiting exceptions with retry-after header.
    
    Args:
        request: FastAPI request object
        exc: Rate limit exception
        
    Returns:
        JSON response with rate limit details
    """
    logger.warning(
        "Rate limit exceeded",
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else None,
        retry_after=exc.details.get('retry_after'),
        limit=exc.details.get('limit')
    )
    
    headers = {}
    if 'retry_after' in exc.details:
        headers['Retry-After'] = str(exc.details['retry_after'])
    
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=exc.to_dict(),
        headers=headers
    )