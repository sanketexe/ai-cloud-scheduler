"""
API Security Middleware and Decorators

This module provides FastAPI integration for the security framework including:
- Authentication middleware
- Authorization decorators
- Request/response security handling
- Rate limiting
- Security headers
"""

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, Callable
import time
from datetime import datetime
import json
from functools import wraps

from security_manager import security_manager, Permission
from security_framework import UserRole, SecurityEventType

# Security scheme for FastAPI
security_scheme = HTTPBearer(auto_error=False)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for FastAPI applications"""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.excluded_paths = {'/docs', '/redoc', '/openapi.json', '/health', '/'}
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Skip security for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Add security headers
        response = await call_next(request)
        
        # Add security headers to response
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # Log request for audit trail
        process_time = time.time() - start_time
        
        # Extract user info if available
        user_id = getattr(request.state, 'user_id', None)
        
        # Log the request
        security_manager.log_action(
            user_id=user_id,
            action=f"{request.method} {request.url.path}",
            resource_type="api_endpoint",
            resource_id=request.url.path,
            ip_address=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
            request_data={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "process_time": process_time
            },
            response_data={
                "status_code": response.status_code
            },
            success=response.status_code < 400
        )
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

class AuthenticationDependency:
    """FastAPI dependency for authentication"""
    
    def __init__(self, required: bool = True):
        self.required = required
    
    async def __call__(self, request: Request, 
                      credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)) -> Optional[Dict[str, Any]]:
        """Authenticate request and return user info"""
        
        # Extract IP and user agent
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Try session token authentication first
        if credentials and credentials.scheme.lower() == "bearer":
            session_info = security_manager.verify_session(credentials.credentials)
            if session_info:
                # Store user info in request state
                request.state.user_id = session_info['user_id']
                request.state.username = session_info['username']
                request.state.user_role = session_info['role']
                return session_info
        
        # Try API key authentication
        api_key = request.headers.get("X-API-Key")
        if api_key:
            api_key_obj = security_manager.verify_api_key(api_key, ip_address, user_agent)
            if api_key_obj:
                # Store API key info in request state
                request.state.user_id = api_key_obj.user_id
                request.state.api_key_id = api_key_obj.key_id
                request.state.api_permissions = [p.value for p in api_key_obj.permissions]
                
                # Get user info
                user = security_manager.auth_manager.users.get(api_key_obj.user_id)
                if user:
                    request.state.user_role = user.role.value
                    return {
                        'user_id': user.user_id,
                        'username': user.username,
                        'role': user.role.value,
                        'auth_type': 'api_key',
                        'api_key_name': api_key_obj.name
                    }
        
        # If authentication is required and failed
        if self.required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

# Create authentication dependencies
require_auth = AuthenticationDependency(required=True)
optional_auth = AuthenticationDependency(required=False)

def require_permission(permission: Permission):
    """Decorator to require specific permission for endpoints"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # Look in kwargs
                request = kwargs.get('request')
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            # Check if user is authenticated
            user_id = getattr(request.state, 'user_id', None)
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check permission
            if hasattr(request.state, 'api_permissions'):
                # API key authentication - check API key permissions
                if permission.value not in request.state.api_permissions:
                    security_manager.security_monitor.log_security_event(
                        SecurityEventType.PERMISSION_DENIED,
                        user_id,
                        request.client.host if request.client else "unknown",
                        request.headers.get("user-agent", ""),
                        {'required_permission': permission.value, 'auth_type': 'api_key'}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission required: {permission.value}"
                    )
            else:
                # Session authentication - check user role permissions
                if not security_manager.check_permission(user_id, permission):
                    security_manager.security_monitor.log_security_event(
                        SecurityEventType.PERMISSION_DENIED,
                        user_id,
                        request.client.host if request.client else "unknown",
                        request.headers.get("user-agent", ""),
                        {'required_permission': permission.value, 'auth_type': 'session'}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission required: {permission.value}"
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(role: UserRole):
    """Decorator to require specific role for endpoints"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                request = kwargs.get('request')
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            # Check if user is authenticated
            user_id = getattr(request.state, 'user_id', None)
            user_role = getattr(request.state, 'user_role', None)
            
            if not user_id or not user_role:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check role
            if user_role != role.value:
                security_manager.security_monitor.log_security_event(
                    SecurityEventType.PERMISSION_DENIED,
                    user_id,
                    request.client.host if request.client else "unknown",
                    request.headers.get("user-agent", ""),
                    {'required_role': role.value, 'user_role': user_role}
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {role.value}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Rate limiting decorator
class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self):
        self.requests = {}  # ip -> list of timestamps
    
    def is_allowed(self, ip: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
        """Check if request is allowed based on rate limits"""
        now = time.time()
        
        if ip not in self.requests:
            self.requests[ip] = []
        
        # Remove old requests outside the window
        self.requests[ip] = [req_time for req_time in self.requests[ip] 
                           if now - req_time < window_seconds]
        
        # Check if under limit
        if len(self.requests[ip]) >= max_requests:
            return False
        
        # Add current request
        self.requests[ip].append(now)
        return True

rate_limiter = RateLimiter()

def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """Decorator to add rate limiting to endpoints"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                request = kwargs.get('request')
            
            if request:
                ip = request.client.host if request.client else "unknown"
                
                if not rate_limiter.is_allowed(ip, max_requests, window_seconds):
                    # Log rate limit exceeded
                    user_id = getattr(request.state, 'user_id', None)
                    security_manager.security_monitor.log_security_event(
                        SecurityEventType.SUSPICIOUS_ACTIVITY,
                        user_id,
                        ip,
                        request.headers.get("user-agent", ""),
                        {
                            'event': 'rate_limit_exceeded',
                            'max_requests': max_requests,
                            'window_seconds': window_seconds
                        }
                    )
                    
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Utility functions for extracting request context
def get_request_context(request: Request) -> Dict[str, Any]:
    """Extract security context from request"""
    return {
        'user_id': getattr(request.state, 'user_id', None),
        'username': getattr(request.state, 'username', None),
        'user_role': getattr(request.state, 'user_role', None),
        'api_key_id': getattr(request.state, 'api_key_id', None),
        'ip_address': request.client.host if request.client else "unknown",
        'user_agent': request.headers.get("user-agent", "")
    }

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"

# Exception handlers for security-related errors
async def authentication_exception_handler(request: Request, exc: HTTPException):
    """Handle authentication exceptions"""
    context = get_request_context(request)
    
    security_manager.security_monitor.log_security_event(
        SecurityEventType.LOGIN_FAILURE,
        context['user_id'],
        context['ip_address'],
        context['user_agent'],
        {'error': str(exc.detail)}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

async def authorization_exception_handler(request: Request, exc: HTTPException):
    """Handle authorization exceptions"""
    context = get_request_context(request)
    
    security_manager.security_monitor.log_security_event(
        SecurityEventType.PERMISSION_DENIED,
        context['user_id'],
        context['ip_address'],
        context['user_agent'],
        {'error': str(exc.detail)}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

def setup_security_middleware(app: FastAPI):
    """Setup security middleware and exception handlers for FastAPI app"""
    
    # Add security middleware
    app.add_middleware(SecurityMiddleware)
    
    # Add exception handlers
    app.add_exception_handler(401, authentication_exception_handler)
    app.add_exception_handler(403, authorization_exception_handler)
    
    return app