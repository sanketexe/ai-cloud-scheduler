"""
Security Framework for Cloud Intelligence Platform

This module implements comprehensive security features including:
- Multi-factor authentication
- Role-based access control (RBAC)
- API key management
- Data encryption
- Audit logging
- Security monitoring
"""

import hashlib
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import json
import logging
from functools import wraps
import time
import re
from collections import defaultdict, deque
import threading
from contextlib import contextmanager

# Configure security logging
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
security_logger.addHandler(handler)

class UserRole(Enum):
    """User roles with hierarchical permissions"""
    ADMIN = "admin"
    MANAGER = "manager"
    ENGINEER = "engineer"
    VIEWER = "viewer"
    API_USER = "api_user"

class Permission(Enum):
    """Granular permissions for system resources"""
    # Workload permissions
    WORKLOAD_CREATE = "workload:create"
    WORKLOAD_READ = "workload:read"
    WORKLOAD_UPDATE = "workload:update"
    WORKLOAD_DELETE = "workload:delete"
    
    # VM permissions
    VM_CREATE = "vm:create"
    VM_READ = "vm:read"
    VM_UPDATE = "vm:update"
    VM_DELETE = "vm:delete"
    
    # Cost permissions
    COST_READ = "cost:read"
    COST_MANAGE = "cost:manage"
    BUDGET_CREATE = "budget:create"
    BUDGET_MANAGE = "budget:manage"
    
    # Performance permissions
    PERFORMANCE_READ = "performance:read"
    PERFORMANCE_MANAGE = "performance:manage"
    
    # System permissions
    SYSTEM_CONFIG = "system:config"
    USER_MANAGE = "user:manage"
    AUDIT_READ = "audit:read"
    
    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"

class MFAMethod(Enum):
    """Multi-factor authentication methods"""
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"
    EMAIL = "email"
    HARDWARE_TOKEN = "hardware_token"

class SecurityEventType(Enum):
    """Types of security events for monitoring"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    MFA_SUCCESS = "mfa_success"
    MFA_FAILURE = "mfa_failure"
    PERMISSION_DENIED = "permission_denied"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "config_change"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

@dataclass
class User:
    """User entity with authentication and authorization data"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: List[Permission] = field(default_factory=list)
    mfa_enabled: bool = False
    mfa_methods: List[MFAMethod] = field(default_factory=list)
    mfa_secret: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    locked_until: Optional[datetime] = None
    api_keys: List[str] = field(default_factory=list)
    session_tokens: List[str] = field(default_factory=list)

@dataclass
class APIKey:
    """API key for service-to-service authentication"""
    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: List[Permission]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    usage_count: int = 0
    rate_limit: int = 1000  # requests per hour

@dataclass
class SecurityEvent:
    """Security event for audit logging and monitoring"""
    event_id: str
    event_type: SecurityEventType
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    
@dataclass
class AuditLog:
    """Comprehensive audit log entry"""
    log_id: str
    user_id: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime = field(default_factory=datetime.now)
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

class EncryptionManager:
    """Handles data encryption and decryption"""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = os.environ.get('ENCRYPTION_KEY', Fernet.generate_key())
            if isinstance(self.master_key, str):
                self.master_key = self.master_key.encode()
        
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'cloud_intelligence_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.cipher_suite = Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data"""
        if isinstance(data, str):
            data = data.encode()
        encrypted_data = self.cipher_suite.encrypt(data)
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            security_logger.error(f"Decryption failed: {e}")
            raise ValueError("Invalid encrypted data")
    
    def encrypt_dict(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Encrypt specific fields in a dictionary"""
        encrypted_data = data.copy()
        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
        return encrypted_data
    
    def decrypt_dict(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Decrypt specific fields in a dictionary"""
        decrypted_data = data.copy()
        for field in sensitive_fields:
            if field in decrypted_data:
                decrypted_data[field] = self.decrypt(decrypted_data[field])
        return decrypted_data

class RolePermissionManager:
    """Manages role-based permissions"""
    
    def __init__(self):
        self.role_permissions = {
            UserRole.ADMIN: [
                # Full system access
                Permission.WORKLOAD_CREATE, Permission.WORKLOAD_READ, 
                Permission.WORKLOAD_UPDATE, Permission.WORKLOAD_DELETE,
                Permission.VM_CREATE, Permission.VM_READ, 
                Permission.VM_UPDATE, Permission.VM_DELETE,
                Permission.COST_READ, Permission.COST_MANAGE,
                Permission.BUDGET_CREATE, Permission.BUDGET_MANAGE,
                Permission.PERFORMANCE_READ, Permission.PERFORMANCE_MANAGE,
                Permission.SYSTEM_CONFIG, Permission.USER_MANAGE,
                Permission.AUDIT_READ, Permission.API_READ,
                Permission.API_WRITE, Permission.API_ADMIN
            ],
            UserRole.MANAGER: [
                # Management level access
                Permission.WORKLOAD_READ, Permission.WORKLOAD_UPDATE,
                Permission.VM_READ, Permission.VM_UPDATE,
                Permission.COST_READ, Permission.COST_MANAGE,
                Permission.BUDGET_CREATE, Permission.BUDGET_MANAGE,
                Permission.PERFORMANCE_READ, Permission.PERFORMANCE_MANAGE,
                Permission.API_READ, Permission.API_WRITE
            ],
            UserRole.ENGINEER: [
                # Engineering level access
                Permission.WORKLOAD_CREATE, Permission.WORKLOAD_READ,
                Permission.WORKLOAD_UPDATE, Permission.VM_READ,
                Permission.COST_READ, Permission.PERFORMANCE_READ,
                Permission.API_READ, Permission.API_WRITE
            ],
            UserRole.VIEWER: [
                # Read-only access
                Permission.WORKLOAD_READ, Permission.VM_READ,
                Permission.COST_READ, Permission.PERFORMANCE_READ,
                Permission.API_READ
            ],
            UserRole.API_USER: [
                # API-specific permissions
                Permission.API_READ, Permission.API_WRITE
            ]
        }
    
    def get_role_permissions(self, role: UserRole) -> List[Permission]:
        """Get permissions for a specific role"""
        return self.role_permissions.get(role, [])
    
    def has_permission(self, user_role: UserRole, permission: Permission) -> bool:
        """Check if a role has a specific permission"""
        role_permissions = self.get_role_permissions(user_role)
        return permission in role_permissions
    
    def add_permission_to_role(self, role: UserRole, permission: Permission):
        """Add permission to a role"""
        if role in self.role_permissions:
            if permission not in self.role_permissions[role]:
                self.role_permissions[role].append(permission)
    
    def remove_permission_from_role(self, role: UserRole, permission: Permission):
        """Remove permission from a role"""
        if role in self.role_permissions and permission in self.role_permissions[role]:
            self.role_permissions[role].remove(permission)

class AuthenticationManager:
    """Handles user authentication including MFA"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.jwt_secret = os.environ.get('JWT_SECRET', secrets.token_urlsafe(32))
        
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_user(self, username: str, email: str, password: str, role: UserRole) -> User:
        """Create a new user"""
        user_id = secrets.token_urlsafe(16)
        password_hash = self.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role
        )
        
        self.users[user_id] = user
        security_logger.info(f"User created: {username} with role {role.value}")
        return user
    
    def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[User]:
        """Authenticate user with username and password"""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            self._log_failed_attempt(username, ip_address)
            return None
        
        # Check if account is locked
        if user.account_locked and user.locked_until and datetime.now() < user.locked_until:
            security_logger.warning(f"Login attempt for locked account: {username}")
            return None
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            self._handle_failed_login(user, ip_address)
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.account_locked = False
        user.locked_until = None
        user.last_login = datetime.now()
        
        security_logger.info(f"User authenticated: {username}")
        return user
    
    def _log_failed_attempt(self, username: str, ip_address: str):
        """Log failed authentication attempt"""
        self.failed_attempts[ip_address].append(datetime.now())
        # Keep only attempts from last hour
        cutoff = datetime.now() - timedelta(hours=1)
        self.failed_attempts[ip_address] = [
            attempt for attempt in self.failed_attempts[ip_address] 
            if attempt > cutoff
        ]
        security_logger.warning(f"Failed login attempt for {username} from {ip_address}")
    
    def _handle_failed_login(self, user: User, ip_address: str):
        """Handle failed login attempt"""
        user.failed_login_attempts += 1
        self._log_failed_attempt(user.username, ip_address)
        
        # Lock account after 5 failed attempts
        if user.failed_login_attempts >= 5:
            user.account_locked = True
            user.locked_until = datetime.now() + timedelta(minutes=30)
            security_logger.warning(f"Account locked: {user.username}")
    
    def generate_mfa_secret(self) -> str:
        """Generate MFA secret for TOTP"""
        return secrets.token_urlsafe(32)
    
    def enable_mfa(self, user_id: str, method: MFAMethod) -> str:
        """Enable MFA for user"""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        user = self.users[user_id]
        user.mfa_enabled = True
        
        if method not in user.mfa_methods:
            user.mfa_methods.append(method)
        
        if method == MFAMethod.TOTP and not user.mfa_secret:
            user.mfa_secret = self.generate_mfa_secret()
        
        security_logger.info(f"MFA enabled for user {user.username} with method {method.value}")
        return user.mfa_secret or ""
    
    def verify_mfa_token(self, user_id: str, token: str, method: MFAMethod) -> bool:
        """Verify MFA token"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        if method == MFAMethod.TOTP:
            # In a real implementation, you would use a TOTP library
            # For now, we'll simulate verification
            return len(token) == 6 and token.isdigit()
        
        # Add other MFA method verifications here
        return False
    
    def create_session_token(self, user: User) -> str:
        """Create JWT session token"""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role.value,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        user.session_tokens.append(token)
        
        # Keep only last 5 tokens
        if len(user.session_tokens) > 5:
            user.session_tokens = user.session_tokens[-5:]
        
        return token
    
    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT session token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            if user_id in self.users and token in self.users[user_id].session_tokens:
                return payload
            
        except jwt.ExpiredSignatureError:
            security_logger.warning("Expired token used")
        except jwt.InvalidTokenError:
            security_logger.warning("Invalid token used")
        
        return None
    
    def revoke_session_token(self, user_id: str, token: str):
        """Revoke a session token"""
        if user_id in self.users and token in self.users[user_id].session_tokens:
            self.users[user_id].session_tokens.remove(token)
            security_logger.info(f"Session token revoked for user {self.users[user_id].username}")

class APIKeyManager:
    """Manages API keys for service-to-service authentication"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.api_keys: Dict[str, APIKey] = {}
        self.usage_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def generate_api_key(self, user_id: str, name: str, permissions: List[Permission], 
                        expires_in_days: Optional[int] = None) -> tuple[str, APIKey]:
        """Generate new API key"""
        key_id = secrets.token_urlsafe(16)
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=permissions,
            expires_at=expires_at
        )
        
        self.api_keys[key_id] = api_key
        
        # Return the raw key (only time it's available)
        full_key = f"{key_id}.{raw_key}"
        security_logger.info(f"API key created: {name} for user {user_id}")
        
        return full_key, api_key
    
    def verify_api_key(self, full_key: str) -> Optional[APIKey]:
        """Verify API key and return associated key object"""
        try:
            key_id, raw_key = full_key.split('.', 1)
            
            if key_id not in self.api_keys:
                return None
            
            api_key = self.api_keys[key_id]
            
            # Check if key is active
            if not api_key.is_active:
                return None
            
            # Check expiration
            if api_key.expires_at and datetime.now() > api_key.expires_at:
                api_key.is_active = False
                return None
            
            # Verify key hash
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            if key_hash != api_key.key_hash:
                return None
            
            # Check rate limit
            if not self._check_rate_limit(key_id):
                return None
            
            # Update usage
            api_key.last_used = datetime.now()
            api_key.usage_count += 1
            
            return api_key
            
        except ValueError:
            return None
    
    def _check_rate_limit(self, key_id: str) -> bool:
        """Check if API key is within rate limits"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Remove old usage records
        usage_times = self.usage_tracking[key_id]
        while usage_times and usage_times[0] < hour_ago:
            usage_times.popleft()
        
        api_key = self.api_keys[key_id]
        
        if len(usage_times) >= api_key.rate_limit:
            return False
        
        # Record this usage
        usage_times.append(now)
        return True
    
    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key"""
        if key_id in self.api_keys and self.api_keys[key_id].user_id == user_id:
            self.api_keys[key_id].is_active = False
            security_logger.info(f"API key revoked: {key_id}")
            return True
        return False
    
    def list_user_api_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user"""
        return [key for key in self.api_keys.values() if key.user_id == user_id]

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.audit_logs: List[AuditLog] = []
        self.sensitive_fields = ['password', 'token', 'key', 'secret']
    
    def log_action(self, user_id: Optional[str], action: str, resource_type: str,
                  resource_id: Optional[str], ip_address: str, user_agent: str,
                  request_data: Dict[str, Any] = None, response_data: Dict[str, Any] = None,
                  success: bool = True, error_message: Optional[str] = None):
        """Log user action for audit trail"""
        
        log_id = secrets.token_urlsafe(16)
        
        # Sanitize sensitive data
        safe_request_data = self._sanitize_data(request_data or {})
        safe_response_data = self._sanitize_data(response_data or {})
        
        audit_log = AuditLog(
            log_id=log_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_data=safe_request_data,
            response_data=safe_response_data,
            success=success,
            error_message=error_message
        )
        
        self.audit_logs.append(audit_log)
        
        # Log to security logger
        log_message = f"Action: {action}, Resource: {resource_type}:{resource_id}, User: {user_id}, Success: {success}"
        if success:
            security_logger.info(log_message)
        else:
            security_logger.warning(f"{log_message}, Error: {error_message}")
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive data from logs"""
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            else:
                sanitized[key] = value
        return sanitized
    
    def get_audit_logs(self, user_id: Optional[str] = None, 
                      resource_type: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      limit: int = 100) -> List[AuditLog]:
        """Retrieve audit logs with filtering"""
        filtered_logs = self.audit_logs
        
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
        
        if resource_type:
            filtered_logs = [log for log in filtered_logs if log.resource_type == resource_type]
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        # Sort by timestamp (newest first) and limit
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_logs[:limit]

class SecurityMonitor:
    """Security monitoring and threat detection"""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.threat_patterns = {
            'brute_force': {'max_attempts': 10, 'time_window': 300},  # 10 attempts in 5 minutes
            'suspicious_api_usage': {'max_requests': 1000, 'time_window': 3600},  # 1000 requests per hour
            'unusual_access_pattern': {'deviation_threshold': 2.0}
        }
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
    
    def log_security_event(self, event_type: SecurityEventType, user_id: Optional[str],
                          ip_address: str, user_agent: str, details: Dict[str, Any] = None):
        """Log security event"""
        event_id = secrets.token_urlsafe(16)
        risk_score = self._calculate_risk_score(event_type, user_id, ip_address, details or {})
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            risk_score=risk_score
        )
        
        self.security_events.append(event)
        
        # Check for threats
        self._analyze_threats(event)
        
        security_logger.info(f"Security event: {event_type.value}, Risk: {risk_score}")
    
    def _calculate_risk_score(self, event_type: SecurityEventType, user_id: Optional[str],
                            ip_address: str, details: Dict[str, Any]) -> float:
        """Calculate risk score for security event"""
        base_scores = {
            SecurityEventType.LOGIN_SUCCESS: 0.1,
            SecurityEventType.LOGIN_FAILURE: 0.5,
            SecurityEventType.MFA_FAILURE: 0.7,
            SecurityEventType.PERMISSION_DENIED: 0.6,
            SecurityEventType.SUSPICIOUS_ACTIVITY: 0.9
        }
        
        risk_score = base_scores.get(event_type, 0.3)
        
        # Increase risk for repeated failures from same IP
        recent_failures = [
            event for event in self.security_events[-100:]  # Last 100 events
            if event.ip_address == ip_address and 
            event.event_type in [SecurityEventType.LOGIN_FAILURE, SecurityEventType.MFA_FAILURE] and
            (datetime.now() - event.timestamp).seconds < 3600  # Last hour
        ]
        
        if len(recent_failures) > 3:
            risk_score = min(1.0, risk_score + 0.2 * len(recent_failures))
        
        return round(risk_score, 2)
    
    def _analyze_threats(self, event: SecurityEvent):
        """Analyze event for potential threats"""
        # Brute force detection
        if event.event_type in [SecurityEventType.LOGIN_FAILURE, SecurityEventType.MFA_FAILURE]:
            self._check_brute_force(event)
        
        # Suspicious API usage
        if event.event_type == SecurityEventType.DATA_ACCESS:
            self._check_api_abuse(event)
        
        # Unusual access patterns
        if event.user_id:
            self._check_unusual_patterns(event)
    
    def _check_brute_force(self, event: SecurityEvent):
        """Check for brute force attacks"""
        pattern = self.threat_patterns['brute_force']
        cutoff_time = datetime.now() - timedelta(seconds=pattern['time_window'])
        
        recent_failures = [
            e for e in self.security_events
            if e.ip_address == event.ip_address and
            e.event_type in [SecurityEventType.LOGIN_FAILURE, SecurityEventType.MFA_FAILURE] and
            e.timestamp > cutoff_time
        ]
        
        if len(recent_failures) >= pattern['max_attempts']:
            self._trigger_threat_alert('brute_force', event, {
                'failure_count': len(recent_failures),
                'time_window': pattern['time_window']
            })
    
    def _check_api_abuse(self, event: SecurityEvent):
        """Check for API abuse"""
        pattern = self.threat_patterns['suspicious_api_usage']
        cutoff_time = datetime.now() - timedelta(seconds=pattern['time_window'])
        
        recent_requests = [
            e for e in self.security_events
            if e.ip_address == event.ip_address and
            e.event_type == SecurityEventType.DATA_ACCESS and
            e.timestamp > cutoff_time
        ]
        
        if len(recent_requests) >= pattern['max_requests']:
            self._trigger_threat_alert('api_abuse', event, {
                'request_count': len(recent_requests),
                'time_window': pattern['time_window']
            })
    
    def _check_unusual_patterns(self, event: SecurityEvent):
        """Check for unusual access patterns"""
        if not event.user_id:
            return
        
        # Build user baseline if not exists
        if event.user_id not in self.user_baselines:
            self.user_baselines[event.user_id] = {
                'common_ips': set(),
                'common_user_agents': set(),
                'typical_hours': set(),
                'request_frequency': 0
            }
        
        baseline = self.user_baselines[event.user_id]
        
        # Check for unusual IP
        if event.ip_address not in baseline['common_ips']:
            if len(baseline['common_ips']) > 0:  # Only alert if we have baseline data
                self._trigger_threat_alert('unusual_ip', event, {
                    'new_ip': event.ip_address,
                    'known_ips': list(baseline['common_ips'])
                })
        
        # Update baseline
        baseline['common_ips'].add(event.ip_address)
        baseline['common_user_agents'].add(event.user_agent)
        baseline['typical_hours'].add(event.timestamp.hour)
        
        # Keep baselines manageable
        if len(baseline['common_ips']) > 10:
            baseline['common_ips'] = set(list(baseline['common_ips'])[-10:])
    
    def _trigger_threat_alert(self, threat_type: str, event: SecurityEvent, details: Dict[str, Any]):
        """Trigger threat alert"""
        alert_event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            user_id=event.user_id,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
            details={
                'threat_type': threat_type,
                'original_event_id': event.event_id,
                **details
            },
            risk_score=0.9
        )
        
        self.security_events.append(alert_event)
        security_logger.critical(f"THREAT DETECTED: {threat_type} from {event.ip_address}")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security monitoring dashboard data"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_events = [e for e in self.security_events if e.timestamp > last_24h]
        
        return {
            'total_events_24h': len(recent_events),
            'high_risk_events': len([e for e in recent_events if e.risk_score > 0.7]),
            'failed_logins': len([e for e in recent_events if e.event_type == SecurityEventType.LOGIN_FAILURE]),
            'successful_logins': len([e for e in recent_events if e.event_type == SecurityEventType.LOGIN_SUCCESS]),
            'permission_denials': len([e for e in recent_events if e.event_type == SecurityEventType.PERMISSION_DENIED]),
            'threat_alerts': len([e for e in recent_events if e.event_type == SecurityEventType.SUSPICIOUS_ACTIVITY]),
            'top_risk_ips': self._get_top_risk_ips(recent_events),
            'event_timeline': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'type': e.event_type.value,
                    'risk_score': e.risk_score,
                    'ip_address': e.ip_address
                }
                for e in sorted(recent_events, key=lambda x: x.timestamp, reverse=True)[:50]
            ]
        }
    
    def _get_top_risk_ips(self, events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Get top risk IP addresses"""
        ip_risks = defaultdict(list)
        
        for event in events:
            ip_risks[event.ip_address].append(event.risk_score)
        
        ip_scores = [
            {
                'ip_address': ip,
                'avg_risk_score': sum(scores) / len(scores),
                'event_count': len(scores),
                'max_risk_score': max(scores)
            }
            for ip, scores in ip_risks.items()
        ]
        
        return sorted(ip_scores, key=lambda x: x['avg_risk_score'], reverse=True)[:10]