"""
Participant Management System for Real-Time Collaborative FinOps Workspace
"""

import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Set
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.exc import IntegrityError

from .database import get_db_session
from .collaboration_models import (
    CollaborativeSession, SessionParticipant, SessionInvitation,
    ParticipantRole, ParticipantStatus, PermissionLevel, SessionStatus
)
from .models import User
from .redis_config import redis_manager
from .notification_service import get_notification_service

logger = logging.getLogger(__name__)

@dataclass
class ParticipantInfo:
    """Participant information"""
    participant_id: str
    user_id: str
    display_name: str
    role: ParticipantRole
    permission_level: PermissionLevel
    status: ParticipantStatus
    joined_at: datetime
    last_active: Optional[datetime] = None
    cursor_position: Optional[Dict[str, Any]] = None
    current_view: Optional[str] = None
    is_typing: bool = False

@dataclass
class PresenceUpdate:
    """Presence update information"""
    participant_id: str
    status: Optional[ParticipantStatus] = None
    cursor_position: Optional[Dict[str, Any]] = None
    current_view: Optional[str] = None
    is_typing: Optional[bool] = None
    last_active: Optional[datetime] = None

@dataclass
class InvitationRequest:
    """Invitation request parameters"""
    session_id: str
    email: str
    role: ParticipantRole = ParticipantRole.VIEWER
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    message: Optional[str] = None
    expires_in_hours: int = 24

@dataclass
class AuthenticationResult:
    """Authentication result"""
    success: bool
    user_id: Optional[str] = None
    permissions: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class AuthorizationResult:
    """Authorization result"""
    authorized: bool
    permissions: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class ParticipantManager:
    """
    Manages participant lifecycle, presence tracking, and permissions in collaborative sessions
    """
    
    def __init__(self):
        self.presence_cache: Dict[str, Dict[str, Any]] = {}  # participant_id -> presence data
        self.typing_timers: Dict[str, asyncio.Task] = {}  # participant_id -> timer task
        self.heartbeat_timers: Dict[str, asyncio.Task] = {}  # participant_id -> heartbeat timer
        self.presence_monitoring_active: Set[str] = set()  # active presence monitoring sessions
        
    async def track_user_presence(self, session_id: str, participant_id: str) -> bool:
        """
        Start tracking user presence in a session
        
        Args:
            session_id: ID of the session
            participant_id: ID of the participant
            
        Returns:
            bool: True if tracking started successfully
        """
        try:
            async with get_db_session() as db:
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.id == participant_id,
                    SessionParticipant.session_id == session_id
                ).first()
                
                if not participant:
                    logger.error(f"Participant {participant_id} not found in session {session_id}")
                    return False
                
                # Initialize presence cache with comprehensive tracking
                self.presence_cache[participant_id] = {
                    "session_id": session_id,
                    "user_id": str(participant.user_id),
                    "display_name": participant.display_name,
                    "role": participant.role.value,
                    "permission_level": participant.permission_level.value,
                    "status": participant.status.value,
                    "cursor_position": participant.cursor_position or {"x": 0, "y": 0, "element_id": None},
                    "current_view": participant.current_view or "dashboard",
                    "is_typing": participant.is_typing,
                    "last_active": datetime.utcnow(),
                    "last_heartbeat": datetime.utcnow(),
                    "connection_quality": "good",
                    "viewport": {"width": 1920, "height": 1080},  # Default viewport
                    "active_filters": {},
                    "selected_elements": []
                }
                
                # Start comprehensive presence monitoring
                await self._start_presence_monitoring(participant_id)
                await self._start_heartbeat_monitoring(participant_id)
                
                # Add to active monitoring set
                self.presence_monitoring_active.add(participant_id)
                
                # Broadcast initial presence to other participants
                await self._broadcast_presence_update(participant_id, self.presence_cache[participant_id])
                
                logger.info(f"Started comprehensive presence tracking for participant {participant_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error starting presence tracking: {e}")
            return False
    
    async def update_presence(self, participant_id: str, presence_update: PresenceUpdate) -> bool:
        """
        Update participant presence information with enhanced cursor and status tracking
        
        Args:
            participant_id: ID of the participant
            presence_update: Presence update data
            
        Returns:
            bool: True if update was successful
        """
        try:
            if participant_id not in self.presence_cache:
                logger.warning(f"Participant {participant_id} not in presence cache")
                return False
            
            # Update cache with enhanced tracking
            cache_data = self.presence_cache[participant_id]
            
            if presence_update.status:
                cache_data["status"] = presence_update.status.value
                
                # Handle status-specific logic
                if presence_update.status == ParticipantStatus.DISCONNECTED:
                    await self._handle_participant_disconnect(participant_id)
                elif presence_update.status == ParticipantStatus.ACTIVE:
                    await self._handle_participant_reconnect(participant_id)
            
            if presence_update.cursor_position is not None:
                # Enhanced cursor tracking with element identification
                cache_data["cursor_position"] = {
                    **presence_update.cursor_position,
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": cache_data["session_id"]
                }
                
                # Track cursor movement for analytics
                await self._track_cursor_movement(participant_id, presence_update.cursor_position)
            
            if presence_update.current_view is not None:
                cache_data["current_view"] = presence_update.current_view
                
                # Track view changes for session analytics
                await self._track_view_change(participant_id, presence_update.current_view)
            
            if presence_update.is_typing is not None:
                cache_data["is_typing"] = presence_update.is_typing
                
                # Enhanced typing indicator management
                if presence_update.is_typing:
                    await self._start_typing_timer(participant_id)
                else:
                    await self._stop_typing_timer(participant_id)
            
            # Update activity timestamps
            cache_data["last_active"] = datetime.utcnow()
            cache_data["last_heartbeat"] = datetime.utcnow()
            
            # Update database with enhanced presence data
            async with get_db_session() as db:
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.id == participant_id
                ).first()
                
                if participant:
                    if presence_update.status:
                        participant.status = presence_update.status
                    if presence_update.cursor_position is not None:
                        participant.cursor_position = cache_data["cursor_position"]
                    if presence_update.current_view is not None:
                        participant.current_view = presence_update.current_view
                    if presence_update.is_typing is not None:
                        participant.is_typing = presence_update.is_typing
                    
                    participant.last_active = datetime.utcnow()
                    participant.last_seen = datetime.utcnow()
                    db.commit()
            
            # Broadcast enhanced presence update
            await self._broadcast_presence_update(participant_id, cache_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating presence for participant {participant_id}: {e}")
            return False
    
    async def get_session_presence(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get presence information for all participants in a session
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of participant presence data
        """
        try:
            presence_data = []
            
            for participant_id, cache_data in self.presence_cache.items():
                if cache_data.get("session_id") == session_id:
                    presence_data.append({
                        "participant_id": participant_id,
                        "user_id": cache_data["user_id"],
                        "status": cache_data["status"],
                        "cursor_position": cache_data["cursor_position"],
                        "current_view": cache_data["current_view"],
                        "is_typing": cache_data["is_typing"],
                        "last_active": cache_data["last_active"].isoformat()
                    })
            
            return presence_data
            
        except Exception as e:
            logger.error(f"Error getting session presence: {e}")
            return []
    
    async def enforce_role_permissions(self, session_id: str, participant_id: str, 
                                     action: str, resource: Optional[str] = None,
                                     context: Optional[Dict[str, Any]] = None) -> AuthorizationResult:
        """
        Enforce role-based permissions for session actions with enhanced granular control
        
        Args:
            session_id: ID of the session
            participant_id: ID of the participant
            action: Action being attempted
            resource: Optional resource being accessed
            context: Additional context for permission evaluation
            
        Returns:
            AuthorizationResult: Authorization result
        """
        try:
            async with get_db_session() as db:
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.id == participant_id,
                    SessionParticipant.session_id == session_id
                ).first()
                
                if not participant:
                    return AuthorizationResult(
                        authorized=False,
                        error_message="Participant not found in session"
                    )
                
                # Get comprehensive permissions based on role and permission level
                permissions = self._get_role_permissions(participant.role, participant.permission_level)
                
                # Enhanced action validation with context awareness
                if not self._validate_action_permission(action, permissions, context):
                    return AuthorizationResult(
                        authorized=False,
                        error_message=f"Action '{action}' not allowed for role '{participant.role.value}' with permission level '{participant.permission_level.value}'"
                    )
                
                # Enhanced resource-specific permissions with data sensitivity levels
                if resource and not self._check_resource_access(permissions, resource, context):
                    return AuthorizationResult(
                        authorized=False,
                        error_message=f"Access to resource '{resource}' not allowed for current permission level"
                    )
                
                # Check time-based permissions (if applicable)
                if not await self._check_temporal_permissions(participant_id, action, context):
                    return AuthorizationResult(
                        authorized=False,
                        error_message="Action not allowed at this time due to temporal restrictions"
                    )
                
                # Log permission check for audit trail
                await self._log_permission_check(session_id, participant_id, action, resource, True)
                
                return AuthorizationResult(
                    authorized=True,
                    permissions=permissions
                )
                
        except Exception as e:
            logger.error(f"Error enforcing permissions: {e}")
            # Log failed permission check
            await self._log_permission_check(session_id, participant_id, action, resource, False, str(e))
            return AuthorizationResult(
                authorized=False,
                error_message=str(e)
            )
    
    async def authenticate_participant(self, session_id: str, user_id: str, 
                                    auth_token: Optional[str] = None,
                                    auth_context: Optional[Dict[str, Any]] = None) -> AuthenticationResult:
        """
        Authenticate a participant for session access with enhanced security
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            auth_token: Optional authentication token
            auth_context: Additional authentication context (IP, device info, etc.)
            
        Returns:
            AuthenticationResult: Authentication result
        """
        try:
            async with get_db_session() as db:
                # Enhanced user validation with security checks
                user = db.query(User).filter(
                    User.id == user_id, 
                    User.is_active == True
                ).first()
                
                if not user:
                    await self._log_authentication_attempt(session_id, user_id, False, "User not found or inactive")
                    return AuthenticationResult(
                        success=False,
                        error_message="User not found or inactive"
                    )
                
                # Validate session exists and is accessible
                session = db.query(CollaborativeSession).filter(
                    CollaborativeSession.id == session_id,
                    CollaborativeSession.status.in_([SessionStatus.CREATED, SessionStatus.ACTIVE])
                ).first()
                
                if not session:
                    await self._log_authentication_attempt(session_id, user_id, False, "Session not found or not active")
                    return AuthenticationResult(
                        success=False,
                        error_message="Session not found or not active"
                    )
                
                # Check if user has existing access to session
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    # Check for valid invitation with enhanced validation
                    invitation = db.query(SessionInvitation).filter(
                        SessionInvitation.session_id == session_id,
                        SessionInvitation.invited_user_id == user_id,
                        SessionInvitation.status == "pending",
                        SessionInvitation.expires_at > datetime.utcnow()
                    ).first()
                    
                    if not invitation:
                        await self._log_authentication_attempt(session_id, user_id, False, "No valid access to session")
                        return AuthenticationResult(
                            success=False,
                            error_message="No valid access to session"
                        )
                    
                    # Validate invitation token if provided
                    if auth_token and invitation.invitation_token != auth_token:
                        await self._log_authentication_attempt(session_id, user_id, False, "Invalid invitation token")
                        return AuthenticationResult(
                            success=False,
                            error_message="Invalid invitation token"
                        )
                
                # Enhanced security checks
                security_check_result = await self._perform_security_checks(user, session, auth_context)
                if not security_check_result.success:
                    await self._log_authentication_attempt(session_id, user_id, False, security_check_result.error_message)
                    return AuthenticationResult(
                        success=False,
                        error_message=security_check_result.error_message
                    )
                
                # Get comprehensive user permissions
                permissions = self._get_user_permissions(user, participant)
                
                # Add session-specific context to permissions
                permissions["session_context"] = {
                    "session_id": session_id,
                    "session_type": session.session_type.value,
                    "user_role": participant.role.value if participant else "invited",
                    "permission_level": participant.permission_level.value if participant else "read_only",
                    "authenticated_at": datetime.utcnow().isoformat()
                }
                
                await self._log_authentication_attempt(session_id, user_id, True, "Authentication successful")
                
                return AuthenticationResult(
                    success=True,
                    user_id=str(user.id),
                    permissions=permissions
                )
                
        except Exception as e:
            logger.error(f"Error authenticating participant: {e}")
            await self._log_authentication_attempt(session_id, user_id, False, str(e))
            return AuthenticationResult(
                success=False,
                error_message=str(e)
            )
    
    async def send_invitation(self, invitation_request: InvitationRequest, 
                            invited_by_user_id: str) -> bool:
        """
        Send invitation to join a collaborative session
        
        Args:
            invitation_request: Invitation request parameters
            invited_by_user_id: ID of user sending invitation
            
        Returns:
            bool: True if invitation was sent successfully
        """
        try:
            async with get_db_session() as db:
                # Validate session exists
                session = db.query(CollaborativeSession).filter(
                    CollaborativeSession.id == invitation_request.session_id
                ).first()
                
                if not session:
                    logger.error(f"Session {invitation_request.session_id} not found")
                    return False
                
                # Find or create user for email
                invited_user = db.query(User).filter(User.email == invitation_request.email).first()
                if not invited_user:
                    logger.error(f"User with email {invitation_request.email} not found")
                    return False
                
                # Check if invitation already exists
                existing_invitation = db.query(SessionInvitation).filter(
                    SessionInvitation.session_id == invitation_request.session_id,
                    SessionInvitation.invited_user_id == invited_user.id,
                    SessionInvitation.status == "pending"
                ).first()
                
                if existing_invitation:
                    logger.warning(f"Invitation already exists for user {invited_user.id}")
                    return False
                
                # Create invitation
                invitation_token = str(uuid.uuid4())
                expires_at = datetime.utcnow() + timedelta(hours=invitation_request.expires_in_hours)
                
                invitation = SessionInvitation(
                    session_id=invitation_request.session_id,
                    invited_user_id=invited_user.id,
                    invited_by=invited_by_user_id,
                    invitation_token=invitation_token,
                    email=invitation_request.email,
                    role=invitation_request.role,
                    permission_level=invitation_request.permission_level,
                    message=invitation_request.message,
                    expires_at=expires_at
                )
                
                db.add(invitation)
                db.commit()
                
                # Send notification
                await self._send_invitation_notification(invitation, session)
                
                logger.info(f"Sent invitation to {invitation_request.email} for session {invitation_request.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error sending invitation: {e}")
            return False
    
    async def accept_invitation(self, invitation_token: str, user_id: str) -> bool:
        """
        Accept a session invitation
        
        Args:
            invitation_token: Invitation token
            user_id: ID of the user accepting
            
        Returns:
            bool: True if invitation was accepted successfully
        """
        try:
            async with get_db_session() as db:
                invitation = db.query(SessionInvitation).filter(
                    SessionInvitation.invitation_token == invitation_token,
                    SessionInvitation.invited_user_id == user_id,
                    SessionInvitation.status == "pending",
                    SessionInvitation.expires_at > datetime.utcnow()
                ).first()
                
                if not invitation:
                    logger.error(f"Invalid or expired invitation token")
                    return False
                
                # Update invitation status
                invitation.status = "accepted"
                invitation.responded_at = datetime.utcnow()
                
                # Create participant record
                user = db.query(User).filter(User.id == user_id).first()
                participant = SessionParticipant(
                    session_id=invitation.session_id,
                    user_id=user_id,
                    display_name=f"{user.first_name} {user.last_name}",
                    role=invitation.role,
                    permission_level=invitation.permission_level,
                    status=ParticipantStatus.ACTIVE,
                    invited_by=invitation.invited_by,
                    invitation_sent_at=invitation.created_at,
                    invitation_accepted_at=datetime.utcnow()
                )
                
                db.add(participant)
                db.commit()
                
                logger.info(f"User {user_id} accepted invitation for session {invitation.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error accepting invitation: {e}")
            return False
    
    async def stop_presence_tracking(self, participant_id: str):
        """Stop tracking presence for a participant"""
        try:
            # Remove from active monitoring
            self.presence_monitoring_active.discard(participant_id)
            
            # Remove from cache
            if participant_id in self.presence_cache:
                del self.presence_cache[participant_id]
            
            # Stop all timers
            await self._stop_typing_timer(participant_id)
            await self._stop_heartbeat_monitoring(participant_id)
            
            logger.info(f"Stopped comprehensive presence tracking for participant {participant_id}")
            
        except Exception as e:
            logger.error(f"Error stopping presence tracking: {e}")
    
    def _get_role_permissions(self, role: ParticipantRole, permission_level: PermissionLevel) -> Dict[str, Any]:
        """Get comprehensive permissions for a role and permission level"""
        base_permissions = {
            "allowed_actions": ["view", "heartbeat", "update_presence"],
            "data_access": ["read"],
            "ui_access": ["dashboard", "reports"],
            "cost_data_access": {
                "view_summary": True,
                "view_details": False,
                "view_sensitive": False,
                "export_data": False
            },
            "collaboration_features": {
                "chat": False,
                "annotations": False,
                "screen_share": False,
                "voice_call": False
            },
            "session_management": {
                "invite_users": False,
                "manage_participants": False,
                "end_session": False,
                "modify_settings": False
            }
        }
        
        # Permission level enhancements
        if permission_level in [PermissionLevel.COMMENT, PermissionLevel.EDIT, PermissionLevel.ADMIN]:
            base_permissions["allowed_actions"].extend(["comment", "annotate", "chat"])
            base_permissions["collaboration_features"]["chat"] = True
            base_permissions["collaboration_features"]["annotations"] = True
        
        if permission_level in [PermissionLevel.EDIT, PermissionLevel.ADMIN]:
            base_permissions["allowed_actions"].extend([
                "edit_filters", "edit_views", "create_annotations", "modify_dashboard"
            ])
            base_permissions["data_access"].append("write")
            base_permissions["cost_data_access"]["view_details"] = True
            base_permissions["collaboration_features"]["screen_share"] = True
        
        if permission_level == PermissionLevel.ADMIN:
            base_permissions["allowed_actions"].extend([
                "invite_participants", "manage_permissions", "end_session", "export_data",
                "access_audit_logs", "manage_session_settings"
            ])
            base_permissions["data_access"].append("admin")
            base_permissions["ui_access"].append("admin_panel")
            base_permissions["cost_data_access"].update({
                "view_sensitive": True,
                "export_data": True
            })
            base_permissions["session_management"].update({
                "invite_users": True,
                "manage_participants": True,
                "modify_settings": True
            })
        
        # Role-specific enhancements
        if role == ParticipantRole.OWNER:
            base_permissions["allowed_actions"].extend([
                "delete_session", "transfer_ownership", "override_permissions"
            ])
            base_permissions["session_management"].update({
                "end_session": True,
                "invite_users": True,
                "manage_participants": True,
                "modify_settings": True
            })
            base_permissions["cost_data_access"]["view_sensitive"] = True
            base_permissions["collaboration_features"]["voice_call"] = True
        
        if role == ParticipantRole.MODERATOR:
            base_permissions["allowed_actions"].extend([
                "moderate_chat", "manage_annotations", "mute_participants"
            ])
            base_permissions["session_management"]["manage_participants"] = True
            base_permissions["collaboration_features"]["voice_call"] = True
        
        if role == ParticipantRole.APPROVER:
            base_permissions["allowed_actions"].extend([
                "approve_decisions", "view_approvals", "access_approval_queue"
            ])
            base_permissions["cost_data_access"]["view_details"] = True
        
        if role == ParticipantRole.ANALYST:
            base_permissions["allowed_actions"].extend([
                "create_reports", "advanced_filtering", "data_analysis"
            ])
            base_permissions["cost_data_access"]["view_details"] = True
            base_permissions["collaboration_features"]["screen_share"] = True
        
        return base_permissions
    
    def _check_resource_access(self, permissions: Dict[str, Any], resource: str, 
                              context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if permissions allow access to a specific resource with enhanced granularity"""
        data_access = permissions.get("data_access", [])
        cost_data_access = permissions.get("cost_data_access", {})
        
        # Enhanced resource classification and access control
        if resource.startswith("sensitive_"):
            return cost_data_access.get("view_sensitive", False) or "admin" in data_access
        
        elif resource.startswith("financial_"):
            # Financial data requires at least detail-level access
            return cost_data_access.get("view_details", False) or any(level in data_access for level in ["write", "admin"])
        
        elif resource.startswith("cost_details_"):
            return cost_data_access.get("view_details", False)
        
        elif resource.startswith("export_"):
            return cost_data_access.get("export_data", False)
        
        elif resource.startswith("audit_"):
            return "admin" in data_access
        
        elif resource.startswith("session_config_"):
            session_mgmt = permissions.get("session_management", {})
            return session_mgmt.get("modify_settings", False)
        
        elif resource.startswith("participant_"):
            session_mgmt = permissions.get("session_management", {})
            return session_mgmt.get("manage_participants", False)
        
        elif resource.startswith("collaboration_"):
            # Check specific collaboration feature access
            feature = resource.replace("collaboration_", "")
            collab_features = permissions.get("collaboration_features", {})
            return collab_features.get(feature, False)
        
        else:
            # Default resources require read access
            return "read" in data_access
    
    def _get_user_permissions(self, user: User, participant: Optional[SessionParticipant]) -> Dict[str, Any]:
        """Get user permissions for session access"""
        if participant:
            return self._get_role_permissions(participant.role, participant.permission_level)
        else:
            # Default permissions for invited users
            return self._get_role_permissions(ParticipantRole.VIEWER, PermissionLevel.READ_ONLY)
    
    async def _start_presence_monitoring(self, participant_id: str):
        """Start monitoring participant presence"""
        # This would typically involve setting up heartbeat monitoring
        # For now, we'll just log that monitoring started
        logger.debug(f"Started presence monitoring for participant {participant_id}")
    
    async def _start_typing_timer(self, participant_id: str):
        """Start typing indicator timer"""
        # Cancel existing timer
        await self._stop_typing_timer(participant_id)
        
        # Start new timer (5 seconds)
        async def typing_timeout():
            await asyncio.sleep(5)
            if participant_id in self.presence_cache:
                self.presence_cache[participant_id]["is_typing"] = False
                await self._broadcast_presence_update(participant_id, self.presence_cache[participant_id])
        
        self.typing_timers[participant_id] = asyncio.create_task(typing_timeout())
    
    async def _stop_typing_timer(self, participant_id: str):
        """Stop typing indicator timer"""
        if participant_id in self.typing_timers:
            self.typing_timers[participant_id].cancel()
            del self.typing_timers[participant_id]
    
    async def _broadcast_presence_update(self, participant_id: str, presence_data: Dict[str, Any]):
        """Broadcast presence update to session participants"""
        try:
            session_id = presence_data.get("session_id")
            if not session_id:
                return
            
            # Broadcast via Redis
            event_data = {
                "event_type": "presence_updated",
                "participant_id": participant_id,
                "presence_data": {
                    "status": presence_data["status"],
                    "cursor_position": presence_data["cursor_position"],
                    "current_view": presence_data["current_view"],
                    "is_typing": presence_data["is_typing"],
                    "last_active": presence_data["last_active"].isoformat()
                }
            }
            
            channel = f"session:{session_id}:presence"
            await redis_manager.publish(channel, event_data)
            
        except Exception as e:
            logger.error(f"Error broadcasting presence update: {e}")
    
    async def _send_invitation_notification(self, invitation: SessionInvitation, session: CollaborativeSession):
        """Send invitation notification"""
        try:
            # This would integrate with the notification service
            # For now, we'll just log the invitation
            logger.info(f"Sending invitation notification to {invitation.email} for session {session.session_name}")
            
            # In a real implementation, this would send an email with the invitation link
            # notification_service.send_email(
            #     to=invitation.email,
            #     subject=f"Invitation to join {session.session_name}",
            #     template="session_invitation",
            #     data={
            #         "session_name": session.session_name,
            #         "invitation_token": invitation.invitation_token,
            #         "expires_at": invitation.expires_at,
            #         "message": invitation.message
            #     }
            # )
            
        except Exception as e:
            logger.error(f"Error sending invitation notification: {e}")
    
    def _validate_action_permission(self, action: str, permissions: Dict[str, Any], 
                                   context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate if an action is allowed based on permissions and context"""
        allowed_actions = permissions.get("allowed_actions", [])
        
        # Basic action check
        if action not in allowed_actions:
            return False
        
        # Context-aware validation
        if context:
            # Check time-based restrictions
            if context.get("time_restricted") and not permissions.get("override_time_restrictions", False):
                return False
            
            # Check resource-specific action permissions
            if context.get("requires_elevated_access") and "admin" not in permissions.get("data_access", []):
                return False
        
        return True
    
    async def _check_temporal_permissions(self, participant_id: str, action: str, 
                                        context: Optional[Dict[str, Any]] = None) -> bool:
        """Check time-based permission restrictions"""
        try:
            # For now, always allow - in production this would check:
            # - Business hours restrictions
            # - Temporary permission elevations
            # - Time-limited access grants
            return True
        except Exception as e:
            logger.error(f"Error checking temporal permissions: {e}")
            return False
    
    async def _log_permission_check(self, session_id: str, participant_id: str, 
                                  action: str, resource: Optional[str], 
                                  success: bool, error_message: Optional[str] = None):
        """Log permission check for audit trail"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "participant_id": participant_id,
                "action": action,
                "resource": resource,
                "success": success,
                "error_message": error_message
            }
            
            # In production, this would write to audit log
            logger.info(f"Permission check: {log_entry}")
            
        except Exception as e:
            logger.error(f"Error logging permission check: {e}")
    
    async def _log_authentication_attempt(self, session_id: str, user_id: str, 
                                        success: bool, message: str):
        """Log authentication attempt for security monitoring"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "user_id": user_id,
                "success": success,
                "message": message
            }
            
            # In production, this would write to security log
            logger.info(f"Authentication attempt: {log_entry}")
            
        except Exception as e:
            logger.error(f"Error logging authentication attempt: {e}")
    
    async def _perform_security_checks(self, user: Any, session: Any, 
                                     auth_context: Optional[Dict[str, Any]]) -> AuthenticationResult:
        """Perform enhanced security checks"""
        try:
            # In production, this would include:
            # - IP address validation
            # - Device fingerprinting
            # - Rate limiting checks
            # - Suspicious activity detection
            
            return AuthenticationResult(success=True)
            
        except Exception as e:
            logger.error(f"Error performing security checks: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Security validation failed"
            )
    
    async def _handle_participant_disconnect(self, participant_id: str):
        """Handle participant disconnection"""
        try:
            if participant_id in self.presence_cache:
                cache_data = self.presence_cache[participant_id]
                cache_data["status"] = ParticipantStatus.DISCONNECTED.value
                cache_data["disconnected_at"] = datetime.utcnow().isoformat()
                
                # Stop monitoring timers
                await self._stop_typing_timer(participant_id)
                await self._stop_heartbeat_monitoring(participant_id)
                
                # Broadcast disconnect event
                await self._broadcast_presence_update(participant_id, cache_data)
                
        except Exception as e:
            logger.error(f"Error handling participant disconnect: {e}")
    
    async def _handle_participant_reconnect(self, participant_id: str):
        """Handle participant reconnection"""
        try:
            if participant_id in self.presence_cache:
                cache_data = self.presence_cache[participant_id]
                cache_data["status"] = ParticipantStatus.ACTIVE.value
                cache_data["reconnected_at"] = datetime.utcnow().isoformat()
                
                # Restart monitoring
                await self._start_heartbeat_monitoring(participant_id)
                
                # Broadcast reconnect event
                await self._broadcast_presence_update(participant_id, cache_data)
                
        except Exception as e:
            logger.error(f"Error handling participant reconnect: {e}")
    
    async def _track_cursor_movement(self, participant_id: str, cursor_position: Dict[str, Any]):
        """Track cursor movement for analytics"""
        try:
            # In production, this would store cursor movement data for:
            # - User behavior analytics
            # - Collaboration pattern analysis
            # - UI optimization insights
            
            logger.debug(f"Cursor movement tracked for participant {participant_id}: {cursor_position}")
            
        except Exception as e:
            logger.error(f"Error tracking cursor movement: {e}")
    
    async def _track_view_change(self, participant_id: str, view: str):
        """Track view changes for analytics"""
        try:
            # In production, this would store view change data for:
            # - Navigation pattern analysis
            # - Feature usage tracking
            # - User experience optimization
            
            logger.debug(f"View change tracked for participant {participant_id}: {view}")
            
        except Exception as e:
            logger.error(f"Error tracking view change: {e}")
    
    async def _start_heartbeat_monitoring(self, participant_id: str):
        """Start heartbeat monitoring for a participant"""
        try:
            # Cancel existing heartbeat timer
            await self._stop_heartbeat_monitoring(participant_id)
            
            async def heartbeat_monitor():
                while participant_id in self.presence_monitoring_active:
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                    if participant_id in self.presence_cache:
                        last_heartbeat = self.presence_cache[participant_id].get("last_heartbeat")
                        if last_heartbeat:
                            time_since_heartbeat = datetime.utcnow() - last_heartbeat
                            if time_since_heartbeat.total_seconds() > 60:  # 1 minute timeout
                                # Mark as idle
                                await self.update_presence(
                                    participant_id,
                                    PresenceUpdate(
                                        participant_id=participant_id,
                                        status=ParticipantStatus.IDLE
                                    )
                                )
            
            self.heartbeat_timers[participant_id] = asyncio.create_task(heartbeat_monitor())
            
        except Exception as e:
            logger.error(f"Error starting heartbeat monitoring: {e}")
    
    async def _stop_heartbeat_monitoring(self, participant_id: str):
        """Stop heartbeat monitoring for a participant"""
        try:
            if participant_id in self.heartbeat_timers:
                self.heartbeat_timers[participant_id].cancel()
                del self.heartbeat_timers[participant_id]
                
        except Exception as e:
            logger.error(f"Error stopping heartbeat monitoring: {e}")

# Global participant manager instance
participant_manager = ParticipantManager()