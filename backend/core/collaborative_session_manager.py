"""
Collaborative Session Manager for Real-Time FinOps Workspace
"""

import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from sqlalchemy.exc import IntegrityError

from .database import get_db_session
from .collaboration_models import (
    CollaborativeSession, SessionParticipant, SessionStateUpdate,
    SessionInvitation, SessionType, ParticipantRole, ParticipantStatus,
    SessionStatus, PermissionLevel
)
from .models import User
from .redis_config import redis_manager

logger = logging.getLogger(__name__)

@dataclass
class SessionConfig:
    """Session configuration parameters"""
    session_name: str
    session_type: SessionType = SessionType.GENERAL
    description: Optional[str] = None
    max_participants: int = 50
    auto_save_interval: int = 30
    session_timeout: int = 3600
    allow_anonymous: bool = False
    require_approval: bool = False
    permissions_template: str = "default"
    session_config: Dict[str, Any] = None

@dataclass
class UserPermissions:
    """User permissions in a session"""
    role: ParticipantRole
    permission_level: PermissionLevel
    can_invite: bool = False
    can_moderate: bool = False
    can_edit_config: bool = False

@dataclass
class JoinResult:
    """Result of joining a session"""
    success: bool
    participant_id: Optional[str] = None
    session_state: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class LeaveResult:
    """Result of leaving a session"""
    success: bool
    error_message: Optional[str] = None

@dataclass
class SyncResult:
    """Result of state synchronization"""
    success: bool
    version: Optional[int] = None
    conflicts: List[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class BroadcastResult:
    """Result of event broadcasting"""
    success: bool
    delivered_count: int = 0
    failed_count: int = 0
    error_message: Optional[str] = None

@dataclass
class CollaborativeEvent:
    """Collaborative event for broadcasting"""
    event_type: str
    event_data: Dict[str, Any]
    sender_id: str
    target_participants: Optional[List[str]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class CollaborativeSessionManager:
    """
    Manages collaborative sessions, participant lifecycle, and real-time synchronization
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, Set[str]] = {}  # session_id -> set of participant_ids
        self.participant_sessions: Dict[str, str] = {}  # participant_id -> session_id
        self.session_locks: Dict[str, asyncio.Lock] = {}
        
    async def create_session(self, creator_id: str, session_config: SessionConfig) -> CollaborativeSession:
        """
        Create a new collaborative session
        
        Args:
            creator_id: ID of the user creating the session
            session_config: Session configuration parameters
            
        Returns:
            CollaborativeSession: The created session
            
        Raises:
            ValueError: If session configuration is invalid
            RuntimeError: If session creation fails
        """
        try:
            async with get_db_session() as db:
                # Validate creator exists
                creator = db.query(User).filter(User.id == creator_id).first()
                if not creator:
                    raise ValueError(f"Creator user {creator_id} not found")
                
                # Create session
                session = CollaborativeSession(
                    session_name=session_config.session_name,
                    session_type=session_config.session_type,
                    description=session_config.description,
                    max_participants=session_config.max_participants,
                    auto_save_interval=session_config.auto_save_interval,
                    session_timeout=session_config.session_timeout,
                    allow_anonymous=session_config.allow_anonymous,
                    require_approval=session_config.require_approval,
                    permissions_template=session_config.permissions_template,
                    session_config=session_config.session_config or {},
                    created_by=creator_id,
                    status=SessionStatus.CREATED
                )
                
                db.add(session)
                db.flush()  # Get the session ID
                
                # Add creator as owner participant
                owner_participant = SessionParticipant(
                    session_id=session.id,
                    user_id=creator_id,
                    display_name=f"{creator.first_name} {creator.last_name}",
                    role=ParticipantRole.OWNER,
                    permission_level=PermissionLevel.ADMIN,
                    status=ParticipantStatus.ACTIVE
                )
                
                db.add(owner_participant)
                db.commit()
                
                # Initialize session in Redis
                await self._initialize_session_cache(str(session.id))
                
                # Initialize session lock
                self.session_locks[str(session.id)] = asyncio.Lock()
                
                logger.info(f"Created collaborative session {session.id} by user {creator_id}")
                return session
                
        except IntegrityError as e:
            logger.error(f"Database integrity error creating session: {e}")
            raise RuntimeError(f"Failed to create session: {e}")
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise RuntimeError(f"Failed to create session: {e}")
    
    async def join_session(self, session_id: str, user_id: str, permissions: UserPermissions) -> JoinResult:
        """
        Join a collaborative session
        
        Args:
            session_id: ID of the session to join
            user_id: ID of the user joining
            permissions: User permissions for the session
            
        Returns:
            JoinResult: Result of the join operation
        """
        try:
            async with get_db_session() as db:
                # Validate session exists and is active
                session = db.query(CollaborativeSession).filter(
                    CollaborativeSession.id == session_id,
                    CollaborativeSession.status.in_([SessionStatus.CREATED, SessionStatus.ACTIVE])
                ).first()
                
                if not session:
                    return JoinResult(success=False, error_message="Session not found or not active")
                
                # Validate user exists
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return JoinResult(success=False, error_message="User not found")
                
                # Check if user is already in session
                existing_participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if existing_participant:
                    # Update existing participant status
                    existing_participant.status = ParticipantStatus.ACTIVE
                    existing_participant.last_active = datetime.utcnow()
                    existing_participant.last_seen = datetime.utcnow()
                    db.commit()
                    
                    participant_id = str(existing_participant.id)
                else:
                    # Check participant limit
                    active_count = db.query(SessionParticipant).filter(
                        SessionParticipant.session_id == session_id,
                        SessionParticipant.left_at.is_(None)
                    ).count()
                    
                    if active_count >= session.max_participants:
                        return JoinResult(success=False, error_message="Session is full")
                    
                    # Create new participant
                    participant = SessionParticipant(
                        session_id=session_id,
                        user_id=user_id,
                        display_name=f"{user.first_name} {user.last_name}",
                        role=permissions.role,
                        permission_level=permissions.permission_level,
                        status=ParticipantStatus.ACTIVE
                    )
                    
                    db.add(participant)
                    db.commit()
                    participant_id = str(participant.id)
                
                # Update session status to active if it was created
                if session.status == SessionStatus.CREATED:
                    session.status = SessionStatus.ACTIVE
                    session.started_at = datetime.utcnow()
                    db.commit()
                
                # Add to active sessions tracking
                if session_id not in self.active_sessions:
                    self.active_sessions[session_id] = set()
                self.active_sessions[session_id].add(participant_id)
                self.participant_sessions[participant_id] = session_id
                
                # Get current session state
                session_state = await self._get_session_state(session_id)
                
                # Broadcast join event
                join_event = CollaborativeEvent(
                    event_type="participant_joined",
                    event_data={
                        "participant_id": participant_id,
                        "user_id": user_id,
                        "display_name": f"{user.first_name} {user.last_name}",
                        "role": permissions.role.value
                    },
                    sender_id=user_id
                )
                await self.broadcast_event(session_id, join_event)
                
                logger.info(f"User {user_id} joined session {session_id}")
                return JoinResult(
                    success=True,
                    participant_id=participant_id,
                    session_state=session_state
                )
                
        except Exception as e:
            logger.error(f"Error joining session {session_id}: {e}")
            return JoinResult(success=False, error_message=str(e))
    
    async def leave_session(self, session_id: str, user_id: str) -> LeaveResult:
        """
        Leave a collaborative session
        
        Args:
            session_id: ID of the session to leave
            user_id: ID of the user leaving
            
        Returns:
            LeaveResult: Result of the leave operation
        """
        try:
            async with get_db_session() as db:
                # Find active participant
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return LeaveResult(success=False, error_message="Participant not found in session")
                
                # Update participant status
                participant.left_at = datetime.utcnow()
                participant.status = ParticipantStatus.DISCONNECTED
                db.commit()
                
                participant_id = str(participant.id)
                
                # Remove from active sessions tracking
                if session_id in self.active_sessions:
                    self.active_sessions[session_id].discard(participant_id)
                    if not self.active_sessions[session_id]:
                        # No active participants, mark session as paused
                        session = db.query(CollaborativeSession).filter(
                            CollaborativeSession.id == session_id
                        ).first()
                        if session:
                            session.status = SessionStatus.PAUSED
                            db.commit()
                
                if participant_id in self.participant_sessions:
                    del self.participant_sessions[participant_id]
                
                # Broadcast leave event
                leave_event = CollaborativeEvent(
                    event_type="participant_left",
                    event_data={
                        "participant_id": participant_id,
                        "user_id": user_id
                    },
                    sender_id=user_id
                )
                await self.broadcast_event(session_id, leave_event)
                
                logger.info(f"User {user_id} left session {session_id}")
                return LeaveResult(success=True)
                
        except Exception as e:
            logger.error(f"Error leaving session {session_id}: {e}")
            return LeaveResult(success=False, error_message=str(e))
    
    async def sync_state(self, session_id: str, state_update: Dict[str, Any]) -> SyncResult:
        """
        Synchronize session state across participants
        
        Args:
            session_id: ID of the session
            state_update: State update to apply
            
        Returns:
            SyncResult: Result of the synchronization
        """
        try:
            # Get session lock to prevent concurrent modifications
            if session_id not in self.session_locks:
                self.session_locks[session_id] = asyncio.Lock()
            
            async with self.session_locks[session_id]:
                async with get_db_session() as db:
                    # Create state update record
                    update_record = SessionStateUpdate(
                        session_id=session_id,
                        user_id=state_update.get("user_id"),
                        operation_id=state_update.get("operation_id", str(uuid.uuid4())),
                        operation_type=state_update.get("operation_type"),
                        target_path=state_update.get("target_path"),
                        old_value=state_update.get("old_value"),
                        new_value=state_update.get("new_value"),
                        operation_data=state_update.get("operation_data", {}),
                        version=await self._get_next_version(session_id)
                    )
                    
                    db.add(update_record)
                    db.commit()
                    
                    # Update session state in Redis
                    await self._update_session_state(session_id, state_update)
                    
                    # Broadcast state change
                    sync_event = CollaborativeEvent(
                        event_type="state_updated",
                        event_data={
                            "operation_id": update_record.operation_id,
                            "operation_type": update_record.operation_type,
                            "target_path": update_record.target_path,
                            "new_value": update_record.new_value,
                            "version": update_record.version
                        },
                        sender_id=state_update.get("user_id")
                    )
                    await self.broadcast_event(session_id, sync_event)
                    
                    return SyncResult(success=True, version=update_record.version)
                    
        except Exception as e:
            logger.error(f"Error syncing state for session {session_id}: {e}")
            return SyncResult(success=False, error_message=str(e))
    
    async def broadcast_event(self, session_id: str, event: CollaborativeEvent) -> BroadcastResult:
        """
        Broadcast an event to session participants
        
        Args:
            session_id: ID of the session
            event: Event to broadcast
            
        Returns:
            BroadcastResult: Result of the broadcast
        """
        try:
            if session_id not in self.active_sessions:
                return BroadcastResult(success=True, delivered_count=0)
            
            participants = self.active_sessions[session_id]
            if event.target_participants:
                participants = participants.intersection(set(event.target_participants))
            
            delivered_count = 0
            failed_count = 0
            
            # Broadcast via Redis pub/sub
            event_data = {
                "session_id": session_id,
                "event_type": event.event_type,
                "event_data": event.event_data,
                "sender_id": event.sender_id,
                "timestamp": event.timestamp.isoformat()
            }
            
            for participant_id in participants:
                try:
                    channel = f"session:{session_id}:participant:{participant_id}"
                    await redis_manager.publish(channel, event_data)
                    delivered_count += 1
                except Exception as e:
                    logger.error(f"Failed to deliver event to participant {participant_id}: {e}")
                    failed_count += 1
            
            return BroadcastResult(
                success=True,
                delivered_count=delivered_count,
                failed_count=failed_count
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting event to session {session_id}: {e}")
            return BroadcastResult(success=False, error_message=str(e))
    
    async def get_session_participants(self, session_id: str) -> List[Dict[str, Any]]:
        """Get list of active session participants"""
        try:
            async with get_db_session() as db:
                participants = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.left_at.is_(None)
                ).all()
                
                return [
                    {
                        "participant_id": str(p.id),
                        "user_id": str(p.user_id),
                        "display_name": p.display_name,
                        "role": p.role.value,
                        "permission_level": p.permission_level.value,
                        "status": p.status.value,
                        "joined_at": p.joined_at.isoformat(),
                        "last_active": p.last_active.isoformat() if p.last_active else None,
                        "cursor_position": p.cursor_position,
                        "current_view": p.current_view,
                        "is_typing": p.is_typing
                    }
                    for p in participants
                ]
        except Exception as e:
            logger.error(f"Error getting session participants: {e}")
            return []
    
    async def update_participant_presence(self, session_id: str, participant_id: str, 
                                        presence_data: Dict[str, Any]) -> bool:
        """Update participant presence information"""
        try:
            async with get_db_session() as db:
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.id == participant_id,
                    SessionParticipant.session_id == session_id
                ).first()
                
                if not participant:
                    return False
                
                # Update presence data
                if "cursor_position" in presence_data:
                    participant.cursor_position = presence_data["cursor_position"]
                if "current_view" in presence_data:
                    participant.current_view = presence_data["current_view"]
                if "is_typing" in presence_data:
                    participant.is_typing = presence_data["is_typing"]
                if "status" in presence_data:
                    participant.status = ParticipantStatus(presence_data["status"])
                
                participant.last_active = datetime.utcnow()
                participant.last_seen = datetime.utcnow()
                db.commit()
                
                # Broadcast presence update
                presence_event = CollaborativeEvent(
                    event_type="presence_updated",
                    event_data={
                        "participant_id": participant_id,
                        **presence_data
                    },
                    sender_id=str(participant.user_id)
                )
                await self.broadcast_event(session_id, presence_event)
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating participant presence: {e}")
            return False
    
    async def _initialize_session_cache(self, session_id: str):
        """Initialize session state in Redis cache"""
        try:
            cache_key = f"session:{session_id}:state"
            initial_state = {
                "dashboard_config": {},
                "active_filters": {},
                "annotations": [],
                "cursor_positions": {},
                "version": 0,
                "last_updated": datetime.utcnow().isoformat()
            }
            await redis_manager.set_json(cache_key, initial_state)
        except Exception as e:
            logger.error(f"Error initializing session cache: {e}")
    
    async def _get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get current session state from cache"""
        try:
            cache_key = f"session:{session_id}:state"
            state = await redis_manager.get_json(cache_key)
            return state or {}
        except Exception as e:
            logger.error(f"Error getting session state: {e}")
            return {}
    
    async def _update_session_state(self, session_id: str, state_update: Dict[str, Any]):
        """Update session state in cache"""
        try:
            cache_key = f"session:{session_id}:state"
            current_state = await self._get_session_state(session_id)
            
            # Apply state update based on operation type
            operation_type = state_update.get("operation_type")
            target_path = state_update.get("target_path")
            new_value = state_update.get("new_value")
            
            if operation_type == "filter_change":
                current_state["active_filters"][target_path] = new_value
            elif operation_type == "dashboard_config":
                current_state["dashboard_config"].update(new_value)
            elif operation_type == "cursor_position":
                current_state["cursor_positions"][state_update.get("user_id")] = new_value
            
            current_state["last_updated"] = datetime.utcnow().isoformat()
            await redis_manager.set_json(cache_key, current_state)
            
        except Exception as e:
            logger.error(f"Error updating session state: {e}")
    
    async def _get_next_version(self, session_id: str) -> int:
        """Get next version number for state updates"""
        try:
            async with get_db_session() as db:
                latest_update = db.query(SessionStateUpdate).filter(
                    SessionStateUpdate.session_id == session_id
                ).order_by(desc(SessionStateUpdate.version)).first()
                
                return (latest_update.version + 1) if latest_update else 1
        except Exception as e:
            logger.error(f"Error getting next version: {e}")
            return 1

# Global session manager instance
session_manager = CollaborativeSessionManager()