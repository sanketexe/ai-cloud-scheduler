"""
Security API Endpoints

This module provides FastAPI endpoints for the security framework including:
- Authentication endpoints
- User management
- API key management
- Audit log access
- Compliance reporting
- Security monitoring dashboard
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

# Import security components
from security_manager import security_manager, Permission
from security_framework import UserRole, MFAMethod, SecurityEventType
from compliance_framework import ComplianceStandard, AuditEventType, generate_compliance_report
from security_monitoring import security_monitoring_system, process_security_event
from data_protection import data_protection_manager
from api_security import (
    require_auth, require_permission, require_role, 
    get_request_context, rate_limit, setup_security_middleware
)

# Pydantic models for API requests/responses
class LoginRequest(BaseModel):
    username: str
    password: str
    mfa_token: Optional[str] = None

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours
    user_info: Dict[str, Any]

class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str
    role: UserRole

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    mfa_enabled: bool
    created_at: str
    last_login: Optional[str]

class CreateAPIKeyRequest(BaseModel):
    name: str
    permissions: List[Permission]
    expires_in_days: Optional[int] = None

class APIKeyResponse(BaseModel):
    key_id: str
    name: str
    key: str  # Only returned on creation
    permissions: List[str]
    created_at: str
    expires_at: Optional[str]

class ComplianceReportRequest(BaseModel):
    standard: ComplianceStandard
    start_date: datetime
    end_date: datetime

class SecurityEventRequest(BaseModel):
    event_type: str
    user_id: Optional[str] = None
    ip_address: str
    user_agent: str
    details: Dict[str, Any] = Field(default_factory=dict)

def create_security_api(app: FastAPI) -> FastAPI:
    """Add security endpoints to FastAPI app"""
    
    # Setup security middleware
    setup_security_middleware(app)
    
    # Authentication endpoints
    @app.post("/api/auth/login", response_model=LoginResponse)
    async def login(request: LoginRequest, req: Request):
        """Authenticate user and return access token"""
        context = get_request_context(req)
        
        token = security_manager.authenticate_user(
            request.username,
            request.password,
            context['ip_address'],
            context['user_agent'],
            request.mfa_token
        )
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Get user info
        session_info = security_manager.verify_session(token)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )
        
        return LoginResponse(
            access_token=token,
            user_info=session_info
        )
    
    @app.post("/api/auth/logout")
    async def logout(req: Request, current_user: Dict = Depends(require_auth)):
        """Logout user and revoke session token"""
        context = get_request_context(req)
        
        # Extract token from Authorization header
        auth_header = req.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            security_manager.logout_user(token, context['ip_address'], context['user_agent'])
        
        return {"message": "Logged out successfully"}
    
    # User management endpoints
    @app.post("/api/users", response_model=UserResponse)
    @require_permission(Permission.USER_MANAGE)
    async def create_user(request: CreateUserRequest, req: Request, 
                         current_user: Dict = Depends(require_auth)):
        """Create new user (admin only)"""
        context = get_request_context(req)
        
        user = security_manager.create_user(
            request.username,
            request.email,
            request.password,
            request.role,
            current_user['user_id'],
            context['ip_address'],
            context['user_agent']
        )
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role.value,
            mfa_enabled=user.mfa_enabled,
            created_at=user.created_at.isoformat(),
            last_login=user.last_login.isoformat() if user.last_login else None
        )
    
    @app.get("/api/users", response_model=List[UserResponse])
    @require_permission(Permission.USER_MANAGE)
    async def list_users(current_user: Dict = Depends(require_auth)):
        """List all users (admin only)"""
        users_data = security_manager.list_users(current_user['user_id'])
        
        return [
            UserResponse(
                user_id=user['user_id'],
                username=user['username'],
                email=user['email'],
                role=user['role'],
                mfa_enabled=user['mfa_enabled'],
                created_at=user['created_at'],
                last_login=user['last_login']
            )
            for user in users_data
        ]
    
    @app.get("/api/users/me", response_model=UserResponse)
    async def get_current_user(current_user: Dict = Depends(require_auth)):
        """Get current user information"""
        user_info = security_manager.get_user_info(
            current_user['user_id'],
            current_user['user_id']
        )
        
        return UserResponse(
            user_id=user_info['user_id'],
            username=user_info['username'],
            email=user_info['email'],
            role=user_info['role'],
            mfa_enabled=user_info['mfa_enabled'],
            created_at=user_info['created_at'],
            last_login=user_info['last_login']
        )
    
    @app.post("/api/users/{user_id}/mfa/enable")
    async def enable_mfa(user_id: str, method: MFAMethod, req: Request,
                        current_user: Dict = Depends(require_auth)):
        """Enable MFA for user"""
        context = get_request_context(req)
        
        secret = security_manager.enable_user_mfa(
            user_id,
            method,
            current_user['user_id'],
            context['ip_address'],
            context['user_agent']
        )
        
        return {"message": "MFA enabled successfully", "secret": secret}
    
    # API Key management endpoints
    @app.post("/api/api-keys", response_model=APIKeyResponse)
    @require_permission(Permission.API_WRITE)
    async def create_api_key(request: CreateAPIKeyRequest, req: Request,
                            current_user: Dict = Depends(require_auth)):
        """Create new API key"""
        context = get_request_context(req)
        
        full_key, api_key = security_manager.create_api_key(
            current_user['user_id'],
            request.name,
            request.permissions,
            request.expires_in_days,
            current_user['user_id'],
            context['ip_address'],
            context['user_agent']
        )
        
        return APIKeyResponse(
            key_id=api_key.key_id,
            name=api_key.name,
            key=full_key,
            permissions=[p.value for p in api_key.permissions],
            created_at=api_key.created_at.isoformat(),
            expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None
        )
    
    @app.delete("/api/api-keys/{key_id}")
    @require_permission(Permission.API_WRITE)
    async def revoke_api_key(key_id: str, req: Request,
                            current_user: Dict = Depends(require_auth)):
        """Revoke API key"""
        context = get_request_context(req)
        
        success = security_manager.revoke_api_key(
            key_id,
            current_user['user_id'],
            current_user['user_id'],
            context['ip_address'],
            context['user_agent']
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        return {"message": "API key revoked successfully"}
    
    # Audit and compliance endpoints
    @app.get("/api/audit/logs")
    @require_permission(Permission.AUDIT_READ)
    async def get_audit_logs(
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100,
        current_user: Dict = Depends(require_auth)
    ):
        """Get audit logs with filtering"""
        
        logs = security_manager.get_audit_logs(
            user_id=user_id,
            resource_type=resource_type,
            start_time=start_time,
            end_time=end_time,
            requester_user_id=current_user['user_id'],
            limit=limit
        )
        
        return {"logs": logs, "count": len(logs)}
    
    @app.post("/api/compliance/reports")
    @require_permission(Permission.AUDIT_READ)
    async def generate_compliance_report_endpoint(
        request: ComplianceReportRequest,
        current_user: Dict = Depends(require_auth)
    ):
        """Generate compliance report"""
        
        report = security_manager.generate_compliance_report(
            request.standard.value,
            request.start_date,
            request.end_date,
            current_user['user_id']
        )
        
        return report
    
    # Security monitoring endpoints
    @app.get("/api/security/dashboard")
    @require_permission(Permission.AUDIT_READ)
    async def get_security_dashboard(current_user: Dict = Depends(require_auth)):
        """Get security monitoring dashboard"""
        
        dashboard_data = security_manager.get_security_dashboard(current_user['user_id'])
        monitoring_data = security_monitoring_system.get_security_dashboard()
        
        return {
            "security_events": dashboard_data,
            "monitoring": monitoring_data
        }
    
    @app.post("/api/security/events")
    @require_permission(Permission.API_WRITE)
    async def log_security_event(request: SecurityEventRequest, req: Request,
                                current_user: Dict = Depends(require_auth)):
        """Log a security event for monitoring"""
        context = get_request_context(req)
        
        event_data = {
            'event_type': request.event_type,
            'user_id': request.user_id or current_user['user_id'],
            'ip_address': request.ip_address or context['ip_address'],
            'user_agent': request.user_agent or context['user_agent'],
            'timestamp': datetime.now(),
            **request.details
        }
        
        incident_ids = process_security_event(event_data)
        
        return {
            "message": "Security event processed",
            "incident_ids": incident_ids
        }
    
    # Data protection endpoints
    @app.get("/api/security/encryption/status")
    @require_permission(Permission.SYSTEM_CONFIG)
    async def get_encryption_status(current_user: Dict = Depends(require_auth)):
        """Get encryption system status"""
        
        status = data_protection_manager.get_encryption_status()
        return status
    
    @app.post("/api/security/encryption/rotate-keys")
    @require_permission(Permission.SYSTEM_CONFIG)
    async def rotate_encryption_keys(req: Request, current_user: Dict = Depends(require_auth)):
        """Rotate encryption keys"""
        context = get_request_context(req)
        
        # Log the key rotation action
        security_manager.log_action(
            current_user['user_id'],
            "rotate_encryption_keys",
            "encryption",
            None,
            context['ip_address'],
            context['user_agent'],
            success=True
        )
        
        data_protection_manager.rotate_keys()
        
        return {"message": "Encryption keys rotated successfully"}
    
    # Health check endpoint
    @app.get("/api/security/health")
    async def security_health_check():
        """Security system health check"""
        
        return {
            "status": "healthy",
            "components": {
                "authentication": "active",
                "authorization": "active",
                "audit_logging": "active",
                "security_monitoring": security_monitoring_system.monitoring_active,
                "encryption": "active"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    return app

# Example usage function
def setup_security_api_example():
    """Example of how to setup the security API"""
    
    app = FastAPI(title="Cloud Intelligence Platform API", version="1.0.0")
    
    # Add security endpoints
    app = create_security_api(app)
    
    # Start security monitoring
    security_monitoring_system.start_monitoring()
    
    return app

if __name__ == "__main__":
    # Example usage
    app = setup_security_api_example()
    
    # In a real deployment, you would run with:
    # uvicorn security_api_endpoints:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
    print("Security API endpoints created. Use with FastAPI application.")
    print("Available endpoints:")
    print("- POST /api/auth/login - User authentication")
    print("- POST /api/auth/logout - User logout")
    print("- POST /api/users - Create user (admin)")
    print("- GET /api/users - List users (admin)")
    print("- GET /api/users/me - Get current user")
    print("- POST /api/users/{user_id}/mfa/enable - Enable MFA")
    print("- POST /api/api-keys - Create API key")
    print("- DELETE /api/api-keys/{key_id} - Revoke API key")
    print("- GET /api/audit/logs - Get audit logs")
    print("- POST /api/compliance/reports - Generate compliance report")
    print("- GET /api/security/dashboard - Security dashboard")
    print("- POST /api/security/events - Log security event")
    print("- GET /api/security/encryption/status - Encryption status")
    print("- POST /api/security/encryption/rotate-keys - Rotate keys")
    print("- GET /api/security/health - Health check")