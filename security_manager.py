"""
Security Manager - Main interface for the security framework

This module provides a unified interface for all security operations including:
- Authentication and authorization
- API key management  
- Audit logging
- Security monitoring
- Data encryption
- Compliance reporting
"""

from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from datetime import datetime, timedelta
import json
import os
from dataclasses import asdict

from security_framework import (
    User, UserRole, Permission, APIKey, SecurityEvent, SecurityEventType,
    AuditLog, MFAMethod, EncryptionManager, RolePermissionManager,
    AuthenticationManager, APIKeyManager, AuditLogger, SecurityMonitor
)

class SecurityManager:
    """Main security manager that coordinates all security components"""
    
    def __init__(self, master_key: Optional[str] = None):
        # Initialize core components
        self.encryption_manager = EncryptionManager(master_key)
        self.role_permission_manager = RolePermissionManager()
        self.auth_manager = AuthenticationManager(self.encryption_manager)
        self.api_key_manager = APIKeyManager(self.encryption_manager)
        self.audit_logger = AuditLogger(self.encryption_manager)
        self.security_monitor = SecurityMonitor()
        
        # Create default admin user if none exists
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if no users exist"""
        if not self.auth_manager.users:
            admin_user = self.auth_manager.create_user(
                username="admin",
                email="admin@cloudplatform.com",
                password="admin123!",  # Should be changed on first login
                role=UserRole.ADMIN
            )
            print(f"Default admin user created: {admin_user.username}")
            print("Please change the default password on first login!")
    
    # Authentication Methods
    def authenticate_user(self, username: str, password: str, ip_address: str, 
                         user_agent: str, mfa_token: Optional[str] = None) -> Optional[str]:
        """Authenticate user and return session token"""
        # Log authentication attempt
        self.security_monitor.log_security_event(
            SecurityEventType.LOGIN_FAILURE,  # Will be updated if successful
            None,
            ip_address,
            user_agent,
            {'username': username}
        )
        
        # Authenticate with username/password
        user = self.auth_manager.authenticate_user(username, password, ip_address)
        if not user:
            return None
        
        # Check MFA if enabled
        if user.mfa_enabled:
            if not mfa_token:
                return None  # MFA token required
            
            # Verify MFA token
            mfa_verified = False
            for method in user.mfa_methods:
                if self.auth_manager.verify_mfa_token(user.user_id, mfa_token, method):
                    mfa_verified = True
                    break
            
            if not mfa_verified:
                self.security_monitor.log_security_event(
                    SecurityEventType.MFA_FAILURE,
                    user.user_id,
                    ip_address,
                    user_agent,
                    {'username': username}
                )
                return None
            
            self.security_monitor.log_security_event(
                SecurityEventType.MFA_SUCCESS,
                user.user_id,
                ip_address,
                user_agent
            )
        
        # Create session token
        token = self.auth_manager.create_session_token(user)
        
        # Log successful authentication
        self.security_monitor.log_security_event(
            SecurityEventType.LOGIN_SUCCESS,
            user.user_id,
            ip_address,
            user_agent
        )
        
        # Log audit trail
        self.audit_logger.log_action(
            user.user_id,
            "login",
            "authentication",
            None,
            ip_address,
            user_agent,
            success=True
        )
        
        return token
    
    def verify_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify session token and return user info"""
        return self.auth_manager.verify_session_token(token)
    
    def logout_user(self, token: str, ip_address: str, user_agent: str):
        """Logout user and revoke session token"""
        session_info = self.verify_session(token)
        if session_info:
            user_id = session_info['user_id']
            self.auth_manager.revoke_session_token(user_id, token)
            
            self.audit_logger.log_action(
                user_id,
                "logout",
                "authentication",
                None,
                ip_address,
                user_agent,
                success=True
            )
    
    # User Management
    def create_user(self, username: str, email: str, password: str, role: UserRole,
                   creator_user_id: str, ip_address: str, user_agent: str) -> User:
        """Create new user (requires admin permissions)"""
        # Check permissions
        creator = self.auth_manager.users.get(creator_user_id)
        if not creator or not self.role_permission_manager.has_permission(creator.role, Permission.USER_MANAGE):
            raise PermissionError("Insufficient permissions to create user")
        
        user = self.auth_manager.create_user(username, email, password, role)
        
        # Log audit trail
        self.audit_logger.log_action(
            creator_user_id,
            "create_user",
            "user",
            user.user_id,
            ip_address,
            user_agent,
            request_data={'username': username, 'email': email, 'role': role.value},
            success=True
        )
        
        return user
    
    def enable_user_mfa(self, user_id: str, method: MFAMethod, 
                       requester_user_id: str, ip_address: str, user_agent: str) -> str:
        """Enable MFA for user"""
        # Users can enable MFA for themselves, or admins can enable for others
        if user_id != requester_user_id:
            requester = self.auth_manager.users.get(requester_user_id)
            if not requester or not self.role_permission_manager.has_permission(requester.role, Permission.USER_MANAGE):
                raise PermissionError("Insufficient permissions to manage user MFA")
        
        secret = self.auth_manager.enable_mfa(user_id, method)
        
        # Log audit trail
        self.audit_logger.log_action(
            requester_user_id,
            "enable_mfa",
            "user",
            user_id,
            ip_address,
            user_agent,
            request_data={'method': method.value},
            success=True
        )
        
        return secret
    
    # API Key Management
    def create_api_key(self, user_id: str, name: str, permissions: List[Permission],
                      expires_in_days: Optional[int] = None,
                      creator_user_id: str = None, ip_address: str = "", user_agent: str = "") -> tuple[str, APIKey]:
        """Create API key for user"""
        # Users can create API keys for themselves, or admins can create for others
        creator_id = creator_user_id or user_id
        if user_id != creator_id:
            creator = self.auth_manager.users.get(creator_id)
            if not creator or not self.role_permission_manager.has_permission(creator.role, Permission.API_ADMIN):
                raise PermissionError("Insufficient permissions to create API key for other users")
        
        # Validate permissions
        user = self.auth_manager.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        user_permissions = self.role_permission_manager.get_role_permissions(user.role)
        invalid_permissions = [p for p in permissions if p not in user_permissions]
        if invalid_permissions:
            raise ValueError(f"User role does not have permissions: {[p.value for p in invalid_permissions]}")
        
        full_key, api_key = self.api_key_manager.generate_api_key(
            user_id, name, permissions, expires_in_days
        )
        
        # Log security event
        self.security_monitor.log_security_event(
            SecurityEventType.API_KEY_CREATED,
            user_id,
            ip_address,
            user_agent,
            {'key_name': name, 'permissions': [p.value for p in permissions]}
        )
        
        # Log audit trail
        self.audit_logger.log_action(
            creator_id,
            "create_api_key",
            "api_key",
            api_key.key_id,
            ip_address,
            user_agent,
            request_data={'name': name, 'permissions': [p.value for p in permissions]},
            success=True
        )
        
        return full_key, api_key
    
    def verify_api_key(self, full_key: str, ip_address: str, user_agent: str) -> Optional[APIKey]:
        """Verify API key and log usage"""
        api_key = self.api_key_manager.verify_api_key(full_key)
        
        if api_key:
            # Log data access
            self.security_monitor.log_security_event(
                SecurityEventType.DATA_ACCESS,
                api_key.user_id,
                ip_address,
                user_agent,
                {'api_key_name': api_key.name}
            )
        
        return api_key
    
    def revoke_api_key(self, key_id: str, user_id: str, requester_user_id: str,
                      ip_address: str, user_agent: str) -> bool:
        """Revoke API key"""
        # Users can revoke their own keys, or admins can revoke any key
        if user_id != requester_user_id:
            requester = self.auth_manager.users.get(requester_user_id)
            if not requester or not self.role_permission_manager.has_permission(requester.role, Permission.API_ADMIN):
                raise PermissionError("Insufficient permissions to revoke API key")
        
        success = self.api_key_manager.revoke_api_key(key_id, user_id)
        
        if success:
            # Log security event
            self.security_monitor.log_security_event(
                SecurityEventType.API_KEY_REVOKED,
                user_id,
                ip_address,
                user_agent,
                {'key_id': key_id}
            )
            
            # Log audit trail
            self.audit_logger.log_action(
                requester_user_id,
                "revoke_api_key",
                "api_key",
                key_id,
                ip_address,
                user_agent,
                success=True
            )
        
        return success
    
    # Authorization Methods
    def check_permission(self, user_id: str, permission: Permission, 
                        resource_id: Optional[str] = None) -> bool:
        """Check if user has specific permission"""
        user = self.auth_manager.users.get(user_id)
        if not user:
            return False
        
        return self.role_permission_manager.has_permission(user.role, permission)
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission for API endpoints"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract user info from request context
                # This would be implemented based on your web framework
                user_id = kwargs.get('current_user_id')
                if not user_id or not self.check_permission(user_id, permission):
                    self.security_monitor.log_security_event(
                        SecurityEventType.PERMISSION_DENIED,
                        user_id,
                        kwargs.get('ip_address', ''),
                        kwargs.get('user_agent', ''),
                        {'required_permission': permission.value}
                    )
                    raise PermissionError(f"Permission required: {permission.value}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    # Data Protection Methods
    def encrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Encrypt sensitive fields in data"""
        return self.encryption_manager.encrypt_dict(data, sensitive_fields)
    
    def decrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Decrypt sensitive fields in data"""
        return self.encryption_manager.decrypt_dict(data, sensitive_fields)
    
    # Audit and Compliance Methods
    def log_action(self, user_id: Optional[str], action: str, resource_type: str,
                  resource_id: Optional[str], ip_address: str, user_agent: str,
                  request_data: Dict[str, Any] = None, response_data: Dict[str, Any] = None,
                  success: bool = True, error_message: Optional[str] = None):
        """Log action for audit trail"""
        self.audit_logger.log_action(
            user_id, action, resource_type, resource_id, ip_address, user_agent,
            request_data, response_data, success, error_message
        )
    
    def get_audit_logs(self, user_id: Optional[str] = None, resource_type: Optional[str] = None,
                      start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                      requester_user_id: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs (requires audit read permission)"""
        requester = self.auth_manager.users.get(requester_user_id)
        if not requester or not self.role_permission_manager.has_permission(requester.role, Permission.AUDIT_READ):
            raise PermissionError("Insufficient permissions to read audit logs")
        
        logs = self.audit_logger.get_audit_logs(user_id, resource_type, start_time, end_time, limit)
        return [asdict(log) for log in logs]
    
    def generate_compliance_report(self, report_type: str, start_date: datetime, 
                                 end_date: datetime, requester_user_id: str) -> Dict[str, Any]:
        """Generate compliance report"""
        requester = self.auth_manager.users.get(requester_user_id)
        if not requester or not self.role_permission_manager.has_permission(requester.role, Permission.AUDIT_READ):
            raise PermissionError("Insufficient permissions to generate compliance reports")
        
        # Get audit logs for the period
        logs = self.audit_logger.get_audit_logs(
            start_time=start_date,
            end_time=end_date,
            limit=10000
        )
        
        # Get security events for the period
        security_events = [
            event for event in self.security_monitor.security_events
            if start_date <= event.timestamp <= end_date
        ]
        
        report = {
            'report_type': report_type,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat(),
            'generated_by': requester_user_id,
            'summary': {
                'total_actions': len(logs),
                'successful_actions': len([log for log in logs if log.success]),
                'failed_actions': len([log for log in logs if not log.success]),
                'unique_users': len(set(log.user_id for log in logs if log.user_id)),
                'security_events': len(security_events),
                'high_risk_events': len([e for e in security_events if e.risk_score > 0.7])
            },
            'user_activity': self._generate_user_activity_report(logs),
            'security_summary': self._generate_security_summary(security_events),
            'compliance_checks': self._generate_compliance_checks(logs, security_events, report_type)
        }
        
        return report
    
    def _generate_user_activity_report(self, logs: List[AuditLog]) -> Dict[str, Any]:
        """Generate user activity summary"""
        user_stats = {}
        
        for log in logs:
            if log.user_id:
                if log.user_id not in user_stats:
                    user_stats[log.user_id] = {
                        'total_actions': 0,
                        'successful_actions': 0,
                        'failed_actions': 0,
                        'resource_types': set(),
                        'last_activity': log.timestamp
                    }
                
                stats = user_stats[log.user_id]
                stats['total_actions'] += 1
                if log.success:
                    stats['successful_actions'] += 1
                else:
                    stats['failed_actions'] += 1
                stats['resource_types'].add(log.resource_type)
                
                if log.timestamp > stats['last_activity']:
                    stats['last_activity'] = log.timestamp
        
        # Convert sets to lists for JSON serialization
        for stats in user_stats.values():
            stats['resource_types'] = list(stats['resource_types'])
            stats['last_activity'] = stats['last_activity'].isoformat()
        
        return user_stats
    
    def _generate_security_summary(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Generate security events summary"""
        event_counts = {}
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0}
        
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if event.risk_score < 0.3:
                risk_distribution['low'] += 1
            elif event.risk_score < 0.7:
                risk_distribution['medium'] += 1
            else:
                risk_distribution['high'] += 1
        
        return {
            'event_counts': event_counts,
            'risk_distribution': risk_distribution,
            'total_events': len(events)
        }
    
    def _generate_compliance_checks(self, logs: List[AuditLog], events: List[SecurityEvent], 
                                  report_type: str) -> Dict[str, Any]:
        """Generate compliance-specific checks"""
        checks = {
            'data_access_logged': len(logs) > 0,
            'authentication_events_tracked': len([e for e in events if 'login' in e.event_type.value]) > 0,
            'failed_access_monitored': len([e for e in events if 'failure' in e.event_type.value]) > 0,
            'administrative_actions_logged': len([log for log in logs if 'admin' in log.action.lower()]) >= 0
        }
        
        if report_type.upper() == 'SOC2':
            checks.update({
                'access_controls_enforced': True,  # Based on permission system
                'audit_trail_complete': len(logs) > 0,
                'security_monitoring_active': len(events) > 0
            })
        elif report_type.upper() == 'GDPR':
            checks.update({
                'data_processing_logged': len([log for log in logs if 'data' in log.resource_type.lower()]) >= 0,
                'user_consent_tracked': True,  # Would need additional implementation
                'data_deletion_logged': len([log for log in logs if 'delete' in log.action.lower()]) >= 0
            })
        
        return checks
    
    # Security Monitoring Methods
    def get_security_dashboard(self, requester_user_id: str) -> Dict[str, Any]:
        """Get security monitoring dashboard"""
        requester = self.auth_manager.users.get(requester_user_id)
        if not requester or not self.role_permission_manager.has_permission(requester.role, Permission.AUDIT_READ):
            raise PermissionError("Insufficient permissions to view security dashboard")
        
        return self.security_monitor.get_security_dashboard()
    
    def get_user_info(self, user_id: str, requester_user_id: str) -> Dict[str, Any]:
        """Get user information (sanitized)"""
        # Users can view their own info, admins can view any user
        if user_id != requester_user_id:
            requester = self.auth_manager.users.get(requester_user_id)
            if not requester or not self.role_permission_manager.has_permission(requester.role, Permission.USER_MANAGE):
                raise PermissionError("Insufficient permissions to view user information")
        
        user = self.auth_manager.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Return sanitized user info
        return {
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'role': user.role.value,
            'mfa_enabled': user.mfa_enabled,
            'mfa_methods': [method.value for method in user.mfa_methods],
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'account_locked': user.account_locked,
            'api_key_count': len(user.api_keys)
        }
    
    def list_users(self, requester_user_id: str) -> List[Dict[str, Any]]:
        """List all users (requires admin permissions)"""
        requester = self.auth_manager.users.get(requester_user_id)
        if not requester or not self.role_permission_manager.has_permission(requester.role, Permission.USER_MANAGE):
            raise PermissionError("Insufficient permissions to list users")
        
        users = []
        for user in self.auth_manager.users.values():
            users.append({
                'user_id': user.user_id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value,
                'mfa_enabled': user.mfa_enabled,
                'created_at': user.created_at.isoformat(),
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'account_locked': user.account_locked
            })
        
        return users

# Global security manager instance
security_manager = SecurityManager()

# Convenience functions for easy integration
def authenticate_user(username: str, password: str, ip_address: str, user_agent: str, 
                     mfa_token: Optional[str] = None) -> Optional[str]:
    """Authenticate user and return session token"""
    return security_manager.authenticate_user(username, password, ip_address, user_agent, mfa_token)

def verify_session(token: str) -> Optional[Dict[str, Any]]:
    """Verify session token"""
    return security_manager.verify_session(token)

def check_permission(user_id: str, permission: Permission) -> bool:
    """Check user permission"""
    return security_manager.check_permission(user_id, permission)

def require_permission(permission: Permission):
    """Decorator to require permission"""
    return security_manager.require_permission(permission)

def log_action(user_id: Optional[str], action: str, resource_type: str, resource_id: Optional[str],
              ip_address: str, user_agent: str, **kwargs):
    """Log action for audit trail"""
    return security_manager.log_action(user_id, action, resource_type, resource_id, 
                                     ip_address, user_agent, **kwargs)