"""
Middleware components for the FinOps Platform.

This module provides request/response logging middleware, audit trail middleware,
and other cross-cutting concerns for the FastAPI application.
"""

import time
import json
import uuid
from typing import Callable, Dict, Any, Optional, Set
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import asyncio

from .logging_service import get_logger, correlation_id_var, user_id_var, request_id_var
from .exceptions import RateLimitException


logger = get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    
    Logs request details, response status, timing information,
    and provides correlation ID tracking across requests.
    """
    
    def __init__(
        self, 
        app: ASGIApp,
        exclude_paths: Set[str] = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1024
    ):
        """
        Initialize request logging middleware.
        
        Args:
            app: ASGI application
            exclude_paths: Set of paths to exclude from logging
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            max_body_size: Maximum body size to log (in bytes)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and response logging.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response object
        """
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate correlation and request IDs
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        
        # Set context variables
        correlation_id_var.set(correlation_id)
        request_id_var.set(request_id)
        
        # Extract user ID from request if available (from JWT token)
        user_id = await self._extract_user_id(request)
        if user_id:
            user_id_var.set(user_id)
        
        # Record start time
        start_time = time.time()
        
        # Log request
        await self._log_request(request, correlation_id, request_id, user_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            await self._log_response(request, response, duration_ms, correlation_id, request_id)
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger.error(
                "Request processing failed",
                path=request.url.path,
                method=request.method,
                duration_ms=duration_ms,
                error=e,
                correlation_id=correlation_id,
                request_id=request_id,
                user_id=user_id
            )
            
            raise
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """
        Extract user ID from request (e.g., from JWT token).
        
        Args:
            request: FastAPI request object
            
        Returns:
            User ID if available, None otherwise
        """
        # This would typically extract from JWT token
        # For now, return None - will be implemented with auth system
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            # TODO: Decode JWT token and extract user ID
            # This will be implemented when JWT auth is integrated
            pass
        
        return None
    
    async def _log_request(
        self, 
        request: Request, 
        correlation_id: str, 
        request_id: str,
        user_id: Optional[str]
    ) -> None:
        """
        Log incoming request details.
        
        Args:
            request: FastAPI request object
            correlation_id: Request correlation ID
            request_id: Unique request ID
            user_id: User ID if available
        """
        request_data = {
            "event_type": "request",
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
            "correlation_id": correlation_id,
            "request_id": request_id
        }
        
        if user_id:
            request_data["user_id"] = user_id
        
        # Log request body if enabled and not too large
        if self.log_request_body and request.headers.get("content-type"):
            try:
                body = await self._read_request_body(request)
                if body and len(body) <= self.max_body_size:
                    request_data["body"] = body[:self.max_body_size]
                elif body:
                    request_data["body_truncated"] = True
                    request_data["body_size"] = len(body)
            except Exception as e:
                logger.warning("Failed to read request body", error=e)
        
        logger.info("Incoming request", **request_data)
    
    async def _log_response(
        self, 
        request: Request, 
        response: Response, 
        duration_ms: float,
        correlation_id: str,
        request_id: str
    ) -> None:
        """
        Log outgoing response details.
        
        Args:
            request: FastAPI request object
            response: FastAPI response object
            duration_ms: Request processing duration in milliseconds
            correlation_id: Request correlation ID
            request_id: Unique request ID
        """
        response_data = {
            "event_type": "response",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "response_headers": dict(response.headers),
            "correlation_id": correlation_id,
            "request_id": request_id
        }
        
        # Log response body if enabled and not streaming
        if (self.log_response_body and 
            not isinstance(response, StreamingResponse) and 
            hasattr(response, 'body')):
            try:
                if response.body and len(response.body) <= self.max_body_size:
                    # Try to decode as JSON first, then as text
                    try:
                        response_data["body"] = json.loads(response.body.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        response_data["body"] = response.body.decode('utf-8', errors='ignore')[:self.max_body_size]
                elif response.body:
                    response_data["body_truncated"] = True
                    response_data["body_size"] = len(response.body)
            except Exception as e:
                logger.warning("Failed to read response body", error=e)
        
        # Determine log level based on status code
        if response.status_code >= 500:
            logger.error("Response sent", **response_data)
        elif response.status_code >= 400:
            logger.warning("Response sent", **response_data)
        else:
            logger.info("Response sent", **response_data)
    
    async def _read_request_body(self, request: Request) -> Optional[str]:
        """
        Read request body safely.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Request body as string or None
        """
        try:
            body = await request.body()
            if body:
                return body.decode('utf-8', errors='ignore')
        except Exception:
            pass
        return None


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware for audit logging of sensitive operations.
    
    Automatically logs operations that modify data or access
    sensitive resources for compliance and security purposes.
    """
    
    def __init__(
        self, 
        app: ASGIApp,
        audit_paths: Set[str] = None,
        sensitive_operations: Set[str] = None
    ):
        """
        Initialize audit middleware.
        
        Args:
            app: ASGI application
            audit_paths: Set of path patterns that require auditing
            sensitive_operations: Set of HTTP methods that require auditing
        """
        super().__init__(app)
        self.audit_paths = audit_paths or {
            "/api/v1/users",
            "/api/v1/cloud-providers",
            "/api/v1/budgets",
            "/api/v1/policies"
        }
        self.sensitive_operations = sensitive_operations or {"POST", "PUT", "PATCH", "DELETE"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process audit logging for sensitive operations.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response object
        """
        # Check if this request needs auditing
        needs_audit = (
            request.method in self.sensitive_operations or
            any(audit_path in request.url.path for audit_path in self.audit_paths)
        )
        
        if not needs_audit:
            return await call_next(request)
        
        # Extract user information
        user_id = user_id_var.get()
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Log audit event for successful operations
        if 200 <= response.status_code < 300:
            await self._log_audit_event(request, response, user_id, duration_ms)
        
        return response
    
    async def _log_audit_event(
        self, 
        request: Request, 
        response: Response, 
        user_id: Optional[str],
        duration_ms: float
    ) -> None:
        """
        Log audit event for sensitive operations.
        
        Args:
            request: FastAPI request object
            response: FastAPI response object
            user_id: User ID performing the operation
            duration_ms: Operation duration
        """
        # Determine resource type and action from path and method
        resource_type = self._extract_resource_type(request.url.path)
        action = self._map_method_to_action(request.method)
        
        # Extract resource ID if available
        resource_id = self._extract_resource_id(request.url.path)
        
        logger.audit(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
    
    def _extract_resource_type(self, path: str) -> str:
        """
        Extract resource type from request path.
        
        Args:
            path: Request path
            
        Returns:
            Resource type string
        """
        # Simple extraction based on path segments
        path_segments = path.strip('/').split('/')
        if len(path_segments) >= 3:  # /api/v1/resource-type
            return path_segments[2].replace('-', '_')
        return "unknown"
    
    def _extract_resource_id(self, path: str) -> Optional[str]:
        """
        Extract resource ID from request path.
        
        Args:
            path: Request path
            
        Returns:
            Resource ID if found, None otherwise
        """
        # Look for UUID-like patterns in path
        import re
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        match = re.search(uuid_pattern, path, re.IGNORECASE)
        return match.group(0) if match else None
    
    def _map_method_to_action(self, method: str) -> str:
        """
        Map HTTP method to audit action.
        
        Args:
            method: HTTP method
            
        Returns:
            Audit action string
        """
        method_map = {
            "GET": "READ",
            "POST": "CREATE",
            "PUT": "UPDATE",
            "PATCH": "UPDATE",
            "DELETE": "DELETE"
        }
        return method_map.get(method, method)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.
    
    Implements basic rate limiting based on client IP address
    to prevent abuse and ensure fair usage.
    """
    
    def __init__(
        self, 
        app: ASGIApp,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: ASGI application
            requests_per_minute: Maximum requests per minute per IP
            burst_size: Maximum burst requests allowed
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = 60  # Clean up old entries every minute
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process rate limiting for requests.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response object
            
        Raises:
            RateLimitException: If rate limit is exceeded
        """
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean up old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Check rate limit
        if await self._is_rate_limited(client_ip, current_time):
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path,
                method=request.method
            )
            
            raise RateLimitException(
                message=f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                retry_after=60,
                limit=self.requests_per_minute
            )
        
        # Update request count
        await self._update_request_count(client_ip, current_time)
        
        return await call_next(request)
    
    async def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """
        Check if client IP is rate limited.
        
        Args:
            client_ip: Client IP address
            current_time: Current timestamp
            
        Returns:
            True if rate limited, False otherwise
        """
        if client_ip not in self.request_counts:
            return False
        
        client_data = self.request_counts[client_ip]
        
        # Check requests in the last minute
        minute_ago = current_time - 60
        recent_requests = [
            timestamp for timestamp in client_data.get('requests', [])
            if timestamp > minute_ago
        ]
        
        return len(recent_requests) >= self.requests_per_minute
    
    async def _update_request_count(self, client_ip: str, current_time: float) -> None:
        """
        Update request count for client IP.
        
        Args:
            client_ip: Client IP address
            current_time: Current timestamp
        """
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {'requests': []}
        
        # Add current request timestamp
        self.request_counts[client_ip]['requests'].append(current_time)
        
        # Keep only requests from the last minute
        minute_ago = current_time - 60
        self.request_counts[client_ip]['requests'] = [
            timestamp for timestamp in self.request_counts[client_ip]['requests']
            if timestamp > minute_ago
        ]
    
    async def _cleanup_old_entries(self, current_time: float) -> None:
        """
        Clean up old rate limiting entries.
        
        Args:
            current_time: Current timestamp
        """
        minute_ago = current_time - 60
        
        # Remove clients with no recent requests
        clients_to_remove = []
        for client_ip, client_data in self.request_counts.items():
            recent_requests = [
                timestamp for timestamp in client_data.get('requests', [])
                if timestamp > minute_ago
            ]
            
            if not recent_requests:
                clients_to_remove.append(client_ip)
            else:
                client_data['requests'] = recent_requests
        
        for client_ip in clients_to_remove:
            del self.request_counts[client_ip]


def setup_middleware(app):
    """
    Set up all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Add rate limiting middleware (first to prevent abuse)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=100, burst_size=20)
    
    # Add audit middleware
    app.add_middleware(AuditMiddleware)
    
    # Add request logging middleware (last to capture all request details)
    app.add_middleware(
        RequestLoggingMiddleware,
        log_request_body=False,  # Set to True for debugging
        log_response_body=False,  # Set to True for debugging
        max_body_size=1024
    )
    
    logger.info("Middleware setup completed")