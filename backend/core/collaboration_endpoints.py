"""
API endpoints for Real-Time Collaborative FinOps Workspace
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from .database import get_db_session
from .auth import get_current_user
from .models import User
from .collaborative_session_manager import session_manager, SessionConfig, UserPermissions
from .participant_manager import participant_manager, InvitationRequest, PresenceUpdate
from .state_synchronization import sync_engine, SyncEventType
from .operational_transformation import ot_engine, Operation, OperationType
from .collaboration_models import (
    SessionType, ParticipantRole, ParticipantStatus, PermissionLevel, SessionStatus
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/collaboration", tags=["collaboration"])
security = HTTPBearer()

# Pydantic models for API

class SessionCreateRequest(BaseModel):
    """Request model for creating a session"""
    session_name: str = Field(..., min_length=1, max_length=200)
    session_type: SessionType = SessionType.GENERAL
    description: Optional[str] = Field(None, max_length=1000)
    max_participants: int = Field(50, ge=2, le=100)
    auto_save_interval: int = Field(30, ge=10, le=300)
    session_timeout: int = Field(3600, ge=300, le=86400)
    allow_anonymous: bool = False
    require_approval: bool = False
    permissions_template: str = "default"
    session_config: Optional[Dict[str, Any]] = None

class SessionJoinRequest(BaseModel):
    """Request model for joining a session"""
    role: ParticipantRole = ParticipantRole.VIEWER
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY

class SessionResponse(BaseModel):
    """Response model for session information"""
    id: str
    session_name: str
    session_type: str
    description: Optional[str]
    status: str
    max_participants: int
    current_participants: int
    created_by: str
    created_at: str
    started_at: Optional[str]
    last_activity: str

class ParticipantResponse(BaseModel):
    """Response model for participant information"""
    participant_id: str
    user_id: str
    display_name: str
    role: str
    permission_level: str
    status: str
    joined_at: str
    last_active: Optional[str]
    cursor_position: Optional[Dict[str, Any]]
    current_view: Optional[str]
    is_typing: bool

class PresenceUpdateRequest(BaseModel):
    """Request model for presence updates"""
    status: Optional[ParticipantStatus] = None
    cursor_position: Optional[Dict[str, Any]] = None
    current_view: Optional[str] = None
    is_typing: Optional[bool] = None

class StateUpdateRequest(BaseModel):
    """Request model for state updates"""
    operation_type: str = Field(..., min_length=1)
    target_path: str = Field(..., min_length=1)
    old_value: Optional[Any] = None
    new_value: Any
    operation_data: Optional[Dict[str, Any]] = None
    parent_version: Optional[int] = None

class CursorUpdateRequest(BaseModel):
    """Request model for cursor updates"""
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    viewport_width: int = Field(1920, ge=100)
    viewport_height: int = Field(1080, ge=100)

class ViewChangeRequest(BaseModel):
    """Request model for view changes"""
    view_type: str = Field(..., min_length=1)
    view_config: Dict[str, Any] = Field(default_factory=dict)

class FilterChangeRequest(BaseModel):
    """Request model for filter changes"""
    filter_path: str = Field(..., min_length=1)
    filter_value: Any

class InvitationSendRequest(BaseModel):
    """Request model for sending invitations"""
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    role: ParticipantRole = ParticipantRole.VIEWER
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    message: Optional[str] = Field(None, max_length=500)
    expires_in_hours: int = Field(24, ge=1, le=168)

# Session Management Endpoints

@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new collaborative session"""
    try:
        session_config = SessionConfig(
            session_name=request.session_name,
            session_type=request.session_type,
            description=request.description,
            max_participants=request.max_participants,
            auto_save_interval=request.auto_save_interval,
            session_timeout=request.session_timeout,
            allow_anonymous=request.allow_anonymous,
            require_approval=request.require_approval,
            permissions_template=request.permissions_template,
            session_config=request.session_config
        )
        
        session = await session_manager.create_session(str(current_user.id), session_config)
        
        # Get participant count
        participants = await session_manager.get_session_participants(str(session.id))
        
        return SessionResponse(
            id=str(session.id),
            session_name=session.session_name,
            session_type=session.session_type.value,
            description=session.description,
            status=session.status.value,
            max_participants=session.max_participants,
            current_participants=len(participants),
            created_by=str(session.created_by),
            created_at=session.created_at.isoformat(),
            started_at=session.started_at.isoformat() if session.started_at else None,
            last_activity=session.last_activity.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@router.post("/sessions/{session_id}/join")
async def join_session(
    session_id: str,
    request: SessionJoinRequest,
    current_user: User = Depends(get_current_user)
):
    """Join a collaborative session"""
    try:
        permissions = UserPermissions(
            role=request.role,
            permission_level=request.permission_level
        )
        
        result = await session_manager.join_session(session_id, str(current_user.id), permissions)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        # Start presence tracking and synchronization
        await participant_manager.track_user_presence(session_id, result.participant_id)
        await sync_engine.add_participant_to_sync(
            session_id, 
            str(current_user.id), 
            f"{current_user.first_name} {current_user.last_name}"
        )
        
        return {
            "success": True,
            "participant_id": result.participant_id,
            "session_state": result.session_state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining session: {e}")
        raise HTTPException(status_code=500, detail="Failed to join session")

@router.post("/sessions/{session_id}/leave")
async def leave_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Leave a collaborative session"""
    try:
        result = await session_manager.leave_session(session_id, str(current_user.id))
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        # Stop presence tracking and synchronization
        # Note: We need to get participant_id first
        participants = await session_manager.get_session_participants(session_id)
        for participant in participants:
            if participant["user_id"] == str(current_user.id):
                await participant_manager.stop_presence_tracking(participant["participant_id"])
                break
        
        await sync_engine.remove_participant_from_sync(session_id, str(current_user.id))
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error leaving session: {e}")
        raise HTTPException(status_code=500, detail="Failed to leave session")

@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get session information"""
    try:
        async with get_db_session() as db:
            from .collaboration_models import CollaborativeSession
            
            session = db.query(CollaborativeSession).filter(
                CollaborativeSession.id == session_id
            ).first()
            
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Check if user has access to session
            participants = await session_manager.get_session_participants(session_id)
            user_has_access = any(p["user_id"] == str(current_user.id) for p in participants)
            
            if not user_has_access:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return SessionResponse(
                id=str(session.id),
                session_name=session.session_name,
                session_type=session.session_type.value,
                description=session.description,
                status=session.status.value,
                max_participants=session.max_participants,
                current_participants=len(participants),
                created_by=str(session.created_by),
                created_at=session.created_at.isoformat(),
                started_at=session.started_at.isoformat() if session.started_at else None,
                last_activity=session.last_activity.isoformat()
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session")

@router.get("/sessions/{session_id}/participants", response_model=List[ParticipantResponse])
async def get_session_participants(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get session participants"""
    try:
        participants = await session_manager.get_session_participants(session_id)
        
        # Check if user has access to session
        user_has_access = any(p["user_id"] == str(current_user.id) for p in participants)
        if not user_has_access:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return [
            ParticipantResponse(
                participant_id=p["participant_id"],
                user_id=p["user_id"],
                display_name=p["display_name"],
                role=p["role"],
                permission_level=p["permission_level"],
                status=p["status"],
                joined_at=p["joined_at"],
                last_active=p["last_active"],
                cursor_position=p["cursor_position"],
                current_view=p["current_view"],
                is_typing=p["is_typing"]
            )
            for p in participants
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting participants: {e}")
        raise HTTPException(status_code=500, detail="Failed to get participants")

# Presence Management Endpoints

@router.post("/sessions/{session_id}/presence")
async def update_presence(
    session_id: str,
    request: PresenceUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update participant presence with real-time synchronization"""
    try:
        # Find participant ID
        participants = await session_manager.get_session_participants(session_id)
        participant_id = None
        
        for participant in participants:
            if participant["user_id"] == str(current_user.id):
                participant_id = participant["participant_id"]
                break
        
        if not participant_id:
            raise HTTPException(status_code=404, detail="Participant not found in session")
        
        # Update presence through participant manager
        presence_update = PresenceUpdate(
            participant_id=participant_id,
            status=request.status,
            cursor_position=request.cursor_position,
            current_view=request.current_view,
            is_typing=request.is_typing
        )
        
        success = await participant_manager.update_presence(participant_id, presence_update)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update presence")
        
        # Update through synchronization engine
        if request.status:
            await sync_engine.update_presence_indicator(
                session_id=session_id,
                user_id=str(current_user.id),
                status=request.status,
                current_view=request.current_view,
                is_typing=request.is_typing
            )
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating presence: {e}")
        raise HTTPException(status_code=500, detail="Failed to update presence")

@router.get("/sessions/{session_id}/presence")
async def get_session_presence(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get session presence information"""
    try:
        # Check access
        participants = await session_manager.get_session_participants(session_id)
        user_has_access = any(p["user_id"] == str(current_user.id) for p in participants)
        
        if not user_has_access:
            raise HTTPException(status_code=403, detail="Access denied")
        
        presence_data = await participant_manager.get_session_presence(session_id)
        return {"presence": presence_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting presence: {e}")
        raise HTTPException(status_code=500, detail="Failed to get presence")

# State Synchronization Endpoints

@router.post("/sessions/{session_id}/sync")
async def sync_session_state(
    session_id: str,
    request: StateUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Synchronize session state with operational transformation"""
    try:
        # Check permissions
        participants = await session_manager.get_session_participants(session_id)
        participant_id = None
        
        for participant in participants:
            if participant["user_id"] == str(current_user.id):
                participant_id = participant["participant_id"]
                break
        
        if not participant_id:
            raise HTTPException(status_code=404, detail="Participant not found in session")
        
        # Check authorization for the action
        auth_result = await participant_manager.enforce_role_permissions(
            session_id, participant_id, request.operation_type
        )
        
        if not auth_result.authorized:
            raise HTTPException(status_code=403, detail=auth_result.error_message)
        
        # Create operation
        operation = Operation(
            operation_type=OperationType(request.operation_type),
            target_path=request.target_path,
            old_value=request.old_value,
            new_value=request.new_value,
            metadata=request.operation_data or {},
            user_id=str(current_user.id),
            session_id=session_id,
            parent_version=request.parent_version
        )
        
        # Apply through synchronization engine
        success = await sync_engine.sync_state_update(session_id, str(current_user.id), operation)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to sync state")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing state: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync state")

@router.post("/sessions/{session_id}/cursor")
async def update_cursor_position(
    session_id: str,
    request: CursorUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update cursor position with real-time synchronization"""
    try:
        # Check if user is in session
        participants = await session_manager.get_session_participants(session_id)
        user_in_session = any(p["user_id"] == str(current_user.id) for p in participants)
        
        if not user_in_session:
            raise HTTPException(status_code=404, detail="User not in session")
        
        # Update cursor through synchronization engine
        success = await sync_engine.update_cursor_position(
            session_id=session_id,
            user_id=str(current_user.id),
            x=request.x,
            y=request.y,
            element_id=request.element_id,
            element_type=request.element_type
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update cursor")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating cursor: {e}")
        raise HTTPException(status_code=500, detail="Failed to update cursor")

@router.post("/sessions/{session_id}/view")
async def change_view(
    session_id: str,
    request: ViewChangeRequest,
    current_user: User = Depends(get_current_user)
):
    """Change dashboard view with synchronization"""
    try:
        # Check permissions
        participants = await session_manager.get_session_participants(session_id)
        participant_id = None
        
        for participant in participants:
            if participant["user_id"] == str(current_user.id):
                participant_id = participant["participant_id"]
                break
        
        if not participant_id:
            raise HTTPException(status_code=404, detail="Participant not found in session")
        
        # Check authorization
        auth_result = await participant_manager.enforce_role_permissions(
            session_id, participant_id, "edit_views"
        )
        
        if not auth_result.authorized:
            raise HTTPException(status_code=403, detail="Not authorized to change views")
        
        # Sync view change
        success = await sync_engine.sync_view_change(
            session_id=session_id,
            user_id=str(current_user.id),
            view_type=request.view_type,
            view_config=request.view_config
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to change view")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing view: {e}")
        raise HTTPException(status_code=500, detail="Failed to change view")

@router.post("/sessions/{session_id}/filters")
async def change_filter(
    session_id: str,
    request: FilterChangeRequest,
    current_user: User = Depends(get_current_user)
):
    """Change filters with real-time synchronization"""
    try:
        # Check permissions
        participants = await session_manager.get_session_participants(session_id)
        participant_id = None
        
        for participant in participants:
            if participant["user_id"] == str(current_user.id):
                participant_id = participant["participant_id"]
                break
        
        if not participant_id:
            raise HTTPException(status_code=404, detail="Participant not found in session")
        
        # Check authorization
        auth_result = await participant_manager.enforce_role_permissions(
            session_id, participant_id, "edit_filters"
        )
        
        if not auth_result.authorized:
            raise HTTPException(status_code=403, detail="Not authorized to change filters")
        
        # Sync filter change
        success = await sync_engine.sync_filter_change(
            session_id=session_id,
            user_id=str(current_user.id),
            filter_path=request.filter_path,
            filter_value=request.filter_value
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to change filter")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing filter: {e}")
        raise HTTPException(status_code=500, detail="Failed to change filter")

@router.get("/sessions/{session_id}/state")
async def get_session_state(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get complete synchronized session state"""
    try:
        # Check access
        participants = await session_manager.get_session_participants(session_id)
        user_has_access = any(p["user_id"] == str(current_user.id) for p in participants)
        
        if not user_has_access:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get synchronized state
        state = await sync_engine.get_session_state(session_id)
        
        return {"state": state}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session state: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session state")

@router.get("/sessions/{session_id}/metrics")
async def get_sync_metrics(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get synchronization performance metrics"""
    try:
        # Check access
        participants = await session_manager.get_session_participants(session_id)
        user_has_access = any(p["user_id"] == str(current_user.id) for p in participants)
        
        if not user_has_access:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get metrics
        metrics = await sync_engine.get_sync_performance_metrics(session_id)
        
        return {"metrics": metrics}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sync metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sync metrics")

# Invitation Management Endpoints

@router.post("/sessions/{session_id}/invitations")
async def send_invitation(
    session_id: str,
    request: InvitationSendRequest,
    current_user: User = Depends(get_current_user)
):
    """Send session invitation"""
    try:
        # Check permissions to invite
        participants = await session_manager.get_session_participants(session_id)
        participant_id = None
        
        for participant in participants:
            if participant["user_id"] == str(current_user.id):
                participant_id = participant["participant_id"]
                break
        
        if not participant_id:
            raise HTTPException(status_code=404, detail="Participant not found in session")
        
        auth_result = await participant_manager.enforce_role_permissions(
            session_id, participant_id, "invite_participants"
        )
        
        if not auth_result.authorized:
            raise HTTPException(status_code=403, detail="Not authorized to send invitations")
        
        invitation_request = InvitationRequest(
            session_id=session_id,
            email=request.email,
            role=request.role,
            permission_level=request.permission_level,
            message=request.message,
            expires_in_hours=request.expires_in_hours
        )
        
        success = await participant_manager.send_invitation(invitation_request, str(current_user.id))
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to send invitation")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending invitation: {e}")
        raise HTTPException(status_code=500, detail="Failed to send invitation")

@router.post("/invitations/{invitation_token}/accept")
async def accept_invitation(
    invitation_token: str,
    current_user: User = Depends(get_current_user)
):
    """Accept session invitation"""
    try:
        success = await participant_manager.accept_invitation(invitation_token, str(current_user.id))
        
        if not success:
            raise HTTPException(status_code=400, detail="Invalid or expired invitation")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error accepting invitation: {e}")
        raise HTTPException(status_code=500, detail="Failed to accept invitation")

# Session Management Endpoints

@router.get("/sessions")
async def list_user_sessions(
    status: Optional[SessionStatus] = Query(None),
    session_type: Optional[SessionType] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """List user's collaborative sessions"""
    try:
        async with get_db_session() as db:
            from .collaboration_models import CollaborativeSession, SessionParticipant
            
            query = db.query(CollaborativeSession).join(SessionParticipant).filter(
                SessionParticipant.user_id == current_user.id
            )
            
            if status:
                query = query.filter(CollaborativeSession.status == status)
            if session_type:
                query = query.filter(CollaborativeSession.session_type == session_type)
            
            sessions = query.order_by(CollaborativeSession.last_activity.desc()).offset(offset).limit(limit).all()
            
            session_list = []
            for session in sessions:
                participants = await session_manager.get_session_participants(str(session.id))
                session_list.append(SessionResponse(
                    id=str(session.id),
                    session_name=session.session_name,
                    session_type=session.session_type.value,
                    description=session.description,
                    status=session.status.value,
                    max_participants=session.max_participants,
                    current_participants=len(participants),
                    created_by=str(session.created_by),
                    created_at=session.created_at.isoformat(),
                    started_at=session.started_at.isoformat() if session.started_at else None,
                    last_activity=session.last_activity.isoformat()
                ))
            
            return {"sessions": session_list}
            
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")