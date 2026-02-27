"""
Video Communication System for Real-Time Collaborative FinOps Workspace
Provides WebRTC-based video calls, screen sharing, and co-browsing capabilities
"""

import uuid
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.exc import IntegrityError

from .database import get_db_session
from .collaboration_models import (
    CollaborativeSession, SessionParticipant, ParticipantRole, PermissionLevel
)
from .models import User
from .redis_config import redis_manager
from .collaborative_session_manager import session_manager, CollaborativeEvent

logger = logging.getLogger(__name__)

class CallType(Enum):
    """Types of video calls"""
    AUDIO_ONLY = "audio_only"
    VIDEO = "video"
    SCREEN_SHARE = "screen_share"
    CO_BROWSE = "co_browse"

class CallStatus(Enum):
    """Status of video calls"""
    INITIATING = "initiating"
    RINGING = "ringing"
    CONNECTING = "connecting"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    ENDED = "ended"
    FAILED = "failed"

class ParticipantCallStatus(Enum):
    """Individual participant status in calls"""
    INVITED = "invited"
    JOINING = "joining"
    CONNECTED = "connected"
    AUDIO_MUTED = "audio_muted"
    VIDEO_DISABLED = "video_disabled"
    SCREEN_SHARING = "screen_sharing"
    DISCONNECTED = "disconnected"

class CallQuality(Enum):
    """Call quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    POOR = "poor"

@dataclass
class VideoCallSession:
    """Video call session data structure"""
    call_id: str
    session_id: str
    initiator_id: str
    call_type: CallType
    status: CallStatus
    participants: List[str]
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    recording_enabled: bool = False
    screen_sharing_active: bool = False
    quality_level: CallQuality = CallQuality.HIGH
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CallParticipant:
    """Call participant information"""
    participant_id: str
    user_id: str
    display_name: str
    status: ParticipantCallStatus
    joined_at: Optional[datetime] = None
    left_at: Optional[datetime] = None
    audio_enabled: bool = True
    video_enabled: bool = True
    screen_sharing: bool = False
    connection_quality: CallQuality = CallQuality.HIGH

@dataclass
class WebRTCOffer:
    """WebRTC offer/answer data"""
    call_id: str
    from_user_id: str
    to_user_id: str
    offer_type: str  # offer, answer, ice_candidate
    sdp_data: Optional[str] = None
    ice_candidate: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class CallResult:
    """Result of call operations"""
    success: bool
    call_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class SignalingResult:
    """Result of WebRTC signaling operations"""
    success: bool
    message_id: Optional[str] = None
    error_message: Optional[str] = None

class VideoCommunicationSystem:
    """
    Manages video communication features including WebRTC calls, screen sharing, and co-browsing
    """
    
    def __init__(self):
        self.active_calls: Dict[str, VideoCallSession] = {}  # call_id -> call session
        self.user_calls: Dict[str, str] = {}  # user_id -> call_id
        self.call_participants: Dict[str, List[CallParticipant]] = {}  # call_id -> participants
        self.signaling_channels: Dict[str, Set[str]] = {}  # call_id -> set of user_ids
        
    async def initiate_call(self, session_id: str, initiator_id: str, 
                          participants: List[str], call_type: CallType = CallType.VIDEO) -> CallResult:
        """
        Initiate a video call within a collaborative session
        
        Args:
            session_id: ID of the collaborative session
            initiator_id: ID of the user initiating the call
            participants: List of participant user IDs to invite
            call_type: Type of call (video, audio, screen share)
            
        Returns:
            CallResult: Result of call initiation
        """
        try:
            async with get_db_session() as db:
                # Validate session and initiator
                session = db.query(CollaborativeSession).filter(
                    CollaborativeSession.id == session_id
                ).first()
                
                if not session:
                    return CallResult(success=False, error_message="Session not found")
                
                initiator = db.query(User).filter(User.id == initiator_id).first()
                if not initiator:
                    return CallResult(success=False, error_message="Initiator not found")
                
                # Check if initiator is participant in session
                initiator_participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == initiator_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not initiator_participant:
                    return CallResult(success=False, error_message="Initiator not in session")
                
                # Check permissions for video calls
                if initiator_participant.permission_level == PermissionLevel.READ_ONLY:
                    return CallResult(success=False, error_message="No permission to initiate calls")
                
                # Validate all participants are in session
                valid_participants = []
                for participant_id in participants:
                    participant = db.query(SessionParticipant).filter(
                        SessionParticipant.session_id == session_id,
                        SessionParticipant.user_id == participant_id,
                        SessionParticipant.left_at.is_(None)
                    ).first()
                    
                    if participant:
                        valid_participants.append(participant_id)
                
                if not valid_participants:
                    return CallResult(success=False, error_message="No valid participants found")
                
                # Check if initiator is already in a call
                if initiator_id in self.user_calls:
                    return CallResult(success=False, error_message="User already in a call")
                
                # Create call session
                call_id = str(uuid.uuid4())
                call_session = VideoCallSession(
                    call_id=call_id,
                    session_id=session_id,
                    initiator_id=initiator_id,
                    call_type=call_type,
                    status=CallStatus.INITIATING,
                    participants=[initiator_id] + valid_participants
                )
                
                # Store call session
                self.active_calls[call_id] = call_session
                self.user_calls[initiator_id] = call_id
                self.call_participants[call_id] = []
                self.signaling_channels[call_id] = set([initiator_id] + valid_participants)
                
                # Add initiator as first participant
                initiator_call_participant = CallParticipant(
                    participant_id=str(initiator_participant.id),
                    user_id=initiator_id,
                    display_name=f"{initiator.first_name} {initiator.last_name}",
                    status=ParticipantCallStatus.CONNECTED,
                    joined_at=datetime.utcnow()
                )
                self.call_participants[call_id].append(initiator_call_participant)
                
                # Cache call information in Redis
                await self._cache_call_session(call_id, call_session)
                
                # Send call invitations to participants
                await self._send_call_invitations(session_id, call_id, valid_participants, call_type)
                
                # Update call status to ringing
                call_session.status = CallStatus.RINGING
                await self._cache_call_session(call_id, call_session)
                
                logger.info(f"Video call {call_id} initiated by user {initiator_id}")
                return CallResult(success=True, call_id=call_id)
                
        except Exception as e:
            logger.error(f"Error initiating call: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def join_call(self, call_id: str, user_id: str) -> CallResult:
        """
        Join an existing video call
        
        Args:
            call_id: ID of the call to join
            user_id: ID of the user joining
            
        Returns:
            CallResult: Result of join operation
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            
            # Check if user is invited to the call
            if user_id not in call_session.participants:
                return CallResult(success=False, error_message="User not invited to call")
            
            # Check if user is already in another call
            if user_id in self.user_calls and self.user_calls[user_id] != call_id:
                return CallResult(success=False, error_message="User already in another call")
            
            async with get_db_session() as db:
                # Get user information
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return CallResult(success=False, error_message="User not found")
                
                # Get participant information
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == call_session.session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return CallResult(success=False, error_message="User not in session")
                
                # Add user to call participants
                call_participant = CallParticipant(
                    participant_id=str(participant.id),
                    user_id=user_id,
                    display_name=f"{user.first_name} {user.last_name}",
                    status=ParticipantCallStatus.JOINING,
                    joined_at=datetime.utcnow()
                )
                
                self.call_participants[call_id].append(call_participant)
                self.user_calls[user_id] = call_id
                
                # Update call status if this is the first join
                if call_session.status == CallStatus.RINGING:
                    call_session.status = CallStatus.CONNECTING
                    call_session.started_at = datetime.utcnow()
                
                # Cache updated call session
                await self._cache_call_session(call_id, call_session)
                
                # Broadcast participant joined event
                join_event = CollaborativeEvent(
                    event_type="call_participant_joined",
                    event_data={
                        "call_id": call_id,
                        "user_id": user_id,
                        "display_name": call_participant.display_name,
                        "call_type": call_session.call_type.value
                    },
                    sender_id=user_id
                )
                
                await session_manager.broadcast_event(call_session.session_id, join_event)
                
                logger.info(f"User {user_id} joined call {call_id}")
                return CallResult(success=True, call_id=call_id)
                
        except Exception as e:
            logger.error(f"Error joining call: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def leave_call(self, call_id: str, user_id: str) -> CallResult:
        """
        Leave a video call
        
        Args:
            call_id: ID of the call to leave
            user_id: ID of the user leaving
            
        Returns:
            CallResult: Result of leave operation
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            
            # Find and update participant
            call_participants = self.call_participants.get(call_id, [])
            participant_found = False
            
            for participant in call_participants:
                if participant.user_id == user_id:
                    participant.status = ParticipantCallStatus.DISCONNECTED
                    participant.left_at = datetime.utcnow()
                    participant_found = True
                    break
            
            if not participant_found:
                return CallResult(success=False, error_message="User not in call")
            
            # Remove user from active calls tracking
            if user_id in self.user_calls:
                del self.user_calls[user_id]
            
            # Check if call should end (no active participants)
            active_participants = [
                p for p in call_participants 
                if p.status not in [ParticipantCallStatus.DISCONNECTED]
            ]
            
            if len(active_participants) <= 1:
                # End the call
                call_session.status = CallStatus.ENDED
                call_session.ended_at = datetime.utcnow()
                
                # Clean up call data
                await self._cleanup_call(call_id)
            
            # Cache updated call session
            await self._cache_call_session(call_id, call_session)
            
            # Broadcast participant left event
            leave_event = CollaborativeEvent(
                event_type="call_participant_left",
                event_data={
                    "call_id": call_id,
                    "user_id": user_id,
                    "call_ended": call_session.status == CallStatus.ENDED
                },
                sender_id=user_id
            )
            
            await session_manager.broadcast_event(call_session.session_id, leave_event)
            
            logger.info(f"User {user_id} left call {call_id}")
            return CallResult(success=True, call_id=call_id)
            
        except Exception as e:
            logger.error(f"Error leaving call: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def handle_webrtc_signaling(self, offer: WebRTCOffer) -> SignalingResult:
        """
        Handle WebRTC signaling messages (offers, answers, ICE candidates)
        
        Args:
            offer: WebRTC signaling data
            
        Returns:
            SignalingResult: Result of signaling operation
        """
        try:
            call_id = offer.call_id
            
            if call_id not in self.active_calls:
                return SignalingResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            
            # Validate users are in the call
            if (offer.from_user_id not in call_session.participants or 
                offer.to_user_id not in call_session.participants):
                return SignalingResult(success=False, error_message="Users not in call")
            
            # Generate message ID
            message_id = str(uuid.uuid4())
            
            # Store signaling message in Redis for real-time delivery
            signaling_data = {
                "message_id": message_id,
                "call_id": call_id,
                "from_user_id": offer.from_user_id,
                "to_user_id": offer.to_user_id,
                "offer_type": offer.offer_type,
                "sdp_data": offer.sdp_data,
                "ice_candidate": offer.ice_candidate,
                "timestamp": offer.timestamp.isoformat()
            }
            
            # Cache signaling message
            cache_key = f"call:{call_id}:signaling:{message_id}"
            await redis_manager.set_json(cache_key, signaling_data, expire=300)  # 5 minutes TTL
            
            # Send signaling message to target user
            signaling_event = CollaborativeEvent(
                event_type="webrtc_signaling",
                event_data=signaling_data,
                sender_id=offer.from_user_id,
                target_participants=[offer.to_user_id]
            )
            
            await session_manager.broadcast_event(call_session.session_id, signaling_event)
            
            # Update call status if this is the first successful signaling
            if call_session.status == CallStatus.CONNECTING and offer.offer_type == "answer":
                call_session.status = CallStatus.ACTIVE
                await self._cache_call_session(call_id, call_session)
            
            logger.debug(f"WebRTC signaling handled: {offer.offer_type} from {offer.from_user_id} to {offer.to_user_id}")
            return SignalingResult(success=True, message_id=message_id)
            
        except Exception as e:
            logger.error(f"Error handling WebRTC signaling: {e}")
            return SignalingResult(success=False, error_message=str(e))
    
    async def toggle_screen_sharing(self, call_id: str, user_id: str, 
                                  enable: bool) -> CallResult:
        """
        Toggle screen sharing for a user in a call
        
        Args:
            call_id: ID of the call
            user_id: ID of the user
            enable: Whether to enable or disable screen sharing
            
        Returns:
            CallResult: Result of operation
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            call_participants = self.call_participants.get(call_id, [])
            
            # Find participant
            participant = None
            for p in call_participants:
                if p.user_id == user_id:
                    participant = p
                    break
            
            if not participant:
                return CallResult(success=False, error_message="User not in call")
            
            # Update screen sharing status
            participant.screen_sharing = enable
            call_session.screen_sharing_active = any(p.screen_sharing for p in call_participants)
            
            # If enabling screen sharing, disable for other participants (only one can share at a time)
            if enable:
                for p in call_participants:
                    if p.user_id != user_id:
                        p.screen_sharing = False
            
            # Cache updated call session
            await self._cache_call_session(call_id, call_session)
            
            # Broadcast screen sharing event
            screen_share_event = CollaborativeEvent(
                event_type="screen_sharing_toggled",
                event_data={
                    "call_id": call_id,
                    "user_id": user_id,
                    "screen_sharing": enable
                },
                sender_id=user_id
            )
            
            await session_manager.broadcast_event(call_session.session_id, screen_share_event)
            
            logger.info(f"Screen sharing {'enabled' if enable else 'disabled'} for user {user_id} in call {call_id}")
            return CallResult(success=True, call_id=call_id)
            
        except Exception as e:
            logger.error(f"Error toggling screen sharing: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def start_co_browsing(self, call_id: str, user_id: str, 
                              dashboard_url: str) -> CallResult:
        """
        Start co-browsing session for synchronized dashboard viewing
        
        Args:
            call_id: ID of the call
            user_id: ID of the user starting co-browsing
            dashboard_url: URL of the dashboard to co-browse
            
        Returns:
            CallResult: Result of operation
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            call_participants = self.call_participants.get(call_id, [])
            
            # Find participant
            participant = None
            for p in call_participants:
                if p.user_id == user_id:
                    participant = p
                    break
            
            if not participant:
                return CallResult(success=False, error_message="User not in call")
            
            # Store co-browsing session info
            co_browse_session_id = str(uuid.uuid4())
            co_browse_data = {
                "session_id": co_browse_session_id,
                "call_id": call_id,
                "initiator_id": user_id,
                "dashboard_url": dashboard_url,
                "started_at": datetime.utcnow().isoformat(),
                "participants": [p.user_id for p in call_participants]
            }
            
            # Cache co-browsing session
            cache_key = f"call:{call_id}:cobrowse"
            await redis_manager.set_json(cache_key, co_browse_data, expire=3600)
            
            # Update call metadata
            call_session.metadata["co_browsing"] = {
                "active": True,
                "session_id": co_browse_session_id,
                "dashboard_url": dashboard_url,
                "initiator_id": user_id
            }
            
            # Cache updated call session
            await self._cache_call_session(call_id, call_session)
            
            # Broadcast co-browsing start event
            co_browse_event = CollaborativeEvent(
                event_type="co_browsing_started",
                event_data={
                    "call_id": call_id,
                    "co_browse_session_id": co_browse_session_id,
                    "initiator_id": user_id,
                    "dashboard_url": dashboard_url
                },
                sender_id=user_id
            )
            
            await session_manager.broadcast_event(call_session.session_id, co_browse_event)
            
            logger.info(f"Co-browsing started by user {user_id} in call {call_id}")
            return CallResult(success=True, call_id=call_id)
            
        except Exception as e:
            logger.error(f"Error starting co-browsing: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def sync_co_browse_action(self, call_id: str, user_id: str, 
                                  action_type: str, action_data: Dict[str, Any]) -> CallResult:
        """
        Synchronize co-browsing actions across participants
        
        Args:
            call_id: ID of the call
            user_id: ID of the user performing the action
            action_type: Type of action (navigate, scroll, click, filter, etc.)
            action_data: Action-specific data
            
        Returns:
            CallResult: Result of operation
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            
            # Check if co-browsing is active
            co_browsing_info = call_session.metadata.get("co_browsing", {})
            if not co_browsing_info.get("active", False):
                return CallResult(success=False, error_message="Co-browsing not active")
            
            # Validate user is in call
            call_participants = self.call_participants.get(call_id, [])
            user_in_call = any(p.user_id == user_id for p in call_participants)
            
            if not user_in_call:
                return CallResult(success=False, error_message="User not in call")
            
            # Create action event
            action_id = str(uuid.uuid4())
            sync_event = CollaborativeEvent(
                event_type="co_browse_action",
                event_data={
                    "action_id": action_id,
                    "call_id": call_id,
                    "user_id": user_id,
                    "action_type": action_type,
                    "action_data": action_data,
                    "timestamp": datetime.utcnow().isoformat()
                },
                sender_id=user_id
            )
            
            # Broadcast to all call participants except the sender
            other_participants = [p.user_id for p in call_participants if p.user_id != user_id]
            sync_event.target_participants = other_participants
            
            await session_manager.broadcast_event(call_session.session_id, sync_event)
            
            # Cache action for replay/history
            action_cache_key = f"call:{call_id}:cobrowse:actions"
            action_record = {
                "action_id": action_id,
                "user_id": user_id,
                "action_type": action_type,
                "action_data": action_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await redis_manager.lpush(action_cache_key, action_record)
            await redis_manager.ltrim(action_cache_key, 0, 99)  # Keep last 100 actions
            await redis_manager.expire(action_cache_key, 3600)  # 1 hour TTL
            
            logger.debug(f"Co-browse action {action_type} synced for user {user_id} in call {call_id}")
            return CallResult(success=True, call_id=call_id)
            
        except Exception as e:
            logger.error(f"Error syncing co-browse action: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def stop_co_browsing(self, call_id: str, user_id: str) -> CallResult:
        """
        Stop co-browsing session
        
        Args:
            call_id: ID of the call
            user_id: ID of the user stopping co-browsing
            
        Returns:
            CallResult: Result of operation
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            
            # Check if co-browsing is active
            co_browsing_info = call_session.metadata.get("co_browsing", {})
            if not co_browsing_info.get("active", False):
                return CallResult(success=False, error_message="Co-browsing not active")
            
            # Only initiator or moderators can stop co-browsing
            call_participants = self.call_participants.get(call_id, [])
            user_participant = None
            for p in call_participants:
                if p.user_id == user_id:
                    user_participant = p
                    break
            
            if not user_participant:
                return CallResult(success=False, error_message="User not in call")
            
            initiator_id = co_browsing_info.get("initiator_id")
            if user_id != initiator_id and user_id != call_session.initiator_id:
                return CallResult(success=False, error_message="Only initiator can stop co-browsing")
            
            # Update call metadata
            call_session.metadata["co_browsing"] = {
                "active": False,
                "ended_at": datetime.utcnow().isoformat()
            }
            
            # Cache updated call session
            await self._cache_call_session(call_id, call_session)
            
            # Clean up co-browsing cache
            cache_key = f"call:{call_id}:cobrowse"
            await redis_manager.delete(cache_key)
            
            # Broadcast co-browsing stop event
            co_browse_event = CollaborativeEvent(
                event_type="co_browsing_stopped",
                event_data={
                    "call_id": call_id,
                    "stopped_by": user_id
                },
                sender_id=user_id
            )
            
            await session_manager.broadcast_event(call_session.session_id, co_browse_event)
            
            logger.info(f"Co-browsing stopped by user {user_id} in call {call_id}")
            return CallResult(success=True, call_id=call_id)
            
        except Exception as e:
            logger.error(f"Error stopping co-browsing: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def update_call_quality(self, call_id: str, user_id: str, 
                                quality: CallQuality) -> CallResult:
        """
        Update call quality for adaptive streaming
        
        Args:
            call_id: ID of the call
            user_id: ID of the user
            quality: New quality level
            
        Returns:
            CallResult: Result of operation
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            call_participants = self.call_participants.get(call_id, [])
            
            # Find participant
            participant = None
            for p in call_participants:
                if p.user_id == user_id:
                    participant = p
                    break
            
            if not participant:
                return CallResult(success=False, error_message="User not in call")
            
            # Update quality
            participant.connection_quality = quality
            
            # Update overall call quality (lowest participant quality)
            min_quality = min(p.connection_quality for p in call_participants)
            call_session.quality_level = min_quality
            
            # Cache updated call session
            await self._cache_call_session(call_id, call_session)
            
            # Broadcast quality update event
            quality_event = CollaborativeEvent(
                event_type="call_quality_updated",
                event_data={
                    "call_id": call_id,
                    "user_id": user_id,
                    "quality": quality.value,
                    "overall_quality": min_quality.value
                },
                sender_id=user_id
            )
            
            await session_manager.broadcast_event(call_session.session_id, quality_event)
            
            return CallResult(success=True, call_id=call_id)
            
        except Exception as e:
            logger.error(f"Error updating call quality: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def handle_connection_issues(self, call_id: str, user_id: str, 
                                     issue_type: str, severity: str) -> CallResult:
        """
        Handle connection issues and implement fallback mechanisms
        
        Args:
            call_id: ID of the call
            user_id: ID of the user experiencing issues
            issue_type: Type of issue (network, audio, video, etc.)
            severity: Severity level (low, medium, high, critical)
            
        Returns:
            CallResult: Result of operation
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            call_participants = self.call_participants.get(call_id, [])
            
            # Find participant
            participant = None
            for p in call_participants:
                if p.user_id == user_id:
                    participant = p
                    break
            
            if not participant:
                return CallResult(success=False, error_message="User not in call")
            
            # Implement fallback strategies based on issue type and severity
            fallback_actions = []
            
            if issue_type == "network" and severity in ["high", "critical"]:
                # Reduce video quality or disable video
                if participant.video_enabled:
                    participant.video_enabled = False
                    fallback_actions.append("video_disabled")
                
                # Reduce audio quality
                participant.connection_quality = CallQuality.LOW
                fallback_actions.append("audio_quality_reduced")
                
            elif issue_type == "video" and severity in ["medium", "high", "critical"]:
                # Disable video, keep audio
                if participant.video_enabled:
                    participant.video_enabled = False
                    fallback_actions.append("video_disabled")
                
            elif issue_type == "audio" and severity in ["high", "critical"]:
                # Switch to text-only communication
                participant.audio_enabled = False
                fallback_actions.append("audio_disabled")
                fallback_actions.append("text_only_mode")
            
            # Update call session metadata with issue tracking
            if "connection_issues" not in call_session.metadata:
                call_session.metadata["connection_issues"] = []
            
            issue_record = {
                "user_id": user_id,
                "issue_type": issue_type,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
                "fallback_actions": fallback_actions
            }
            call_session.metadata["connection_issues"].append(issue_record)
            
            # Cache updated call session
            await self._cache_call_session(call_id, call_session)
            
            # Broadcast connection issue event
            issue_event = CollaborativeEvent(
                event_type="connection_issue_handled",
                event_data={
                    "call_id": call_id,
                    "user_id": user_id,
                    "issue_type": issue_type,
                    "severity": severity,
                    "fallback_actions": fallback_actions
                },
                sender_id=user_id
            )
            
            await session_manager.broadcast_event(call_session.session_id, issue_event)
            
            logger.info(f"Connection issue handled for user {user_id} in call {call_id}: {issue_type} ({severity})")
            return CallResult(success=True, call_id=call_id)
            
        except Exception as e:
            logger.error(f"Error handling connection issues: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def attempt_reconnection(self, call_id: str, user_id: str) -> CallResult:
        """
        Attempt to reconnect a user to a call after connection loss
        
        Args:
            call_id: ID of the call
            user_id: ID of the user attempting to reconnect
            
        Returns:
            CallResult: Result of reconnection attempt
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            call_participants = self.call_participants.get(call_id, [])
            
            # Find participant
            participant = None
            for p in call_participants:
                if p.user_id == user_id:
                    participant = p
                    break
            
            if not participant:
                return CallResult(success=False, error_message="User not in call")
            
            # Check if call is still active
            if call_session.status not in [CallStatus.ACTIVE, CallStatus.CONNECTING]:
                return CallResult(success=False, error_message="Call is not active")
            
            # Update participant status
            participant.status = ParticipantCallStatus.CONNECTING
            
            # Reset connection quality to medium for reconnection
            participant.connection_quality = CallQuality.MEDIUM
            
            # Cache updated call session
            await self._cache_call_session(call_id, call_session)
            
            # Broadcast reconnection attempt
            reconnect_event = CollaborativeEvent(
                event_type="participant_reconnecting",
                event_data={
                    "call_id": call_id,
                    "user_id": user_id,
                    "display_name": participant.display_name
                },
                sender_id=user_id
            )
            
            await session_manager.broadcast_event(call_session.session_id, reconnect_event)
            
            logger.info(f"Reconnection attempt for user {user_id} in call {call_id}")
            return CallResult(success=True, call_id=call_id)
            
        except Exception as e:
            logger.error(f"Error attempting reconnection: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def enable_audio_only_fallback(self, call_id: str, user_id: str) -> CallResult:
        """
        Enable audio-only fallback when video fails
        
        Args:
            call_id: ID of the call
            user_id: ID of the user
            
        Returns:
            CallResult: Result of operation
        """
        try:
            if call_id not in self.active_calls:
                return CallResult(success=False, error_message="Call not found")
            
            call_session = self.active_calls[call_id]
            call_participants = self.call_participants.get(call_id, [])
            
            # Find participant
            participant = None
            for p in call_participants:
                if p.user_id == user_id:
                    participant = p
                    break
            
            if not participant:
                return CallResult(success=False, error_message="User not in call")
            
            # Disable video, keep audio
            participant.video_enabled = False
            participant.audio_enabled = True
            participant.screen_sharing = False
            
            # Update call type if all participants are audio-only
            all_audio_only = all(not p.video_enabled for p in call_participants)
            if all_audio_only and call_session.call_type == CallType.VIDEO:
                call_session.call_type = CallType.AUDIO_ONLY
            
            # Cache updated call session
            await self._cache_call_session(call_id, call_session)
            
            # Broadcast fallback event
            fallback_event = CollaborativeEvent(
                event_type="audio_only_fallback",
                event_data={
                    "call_id": call_id,
                    "user_id": user_id,
                    "call_type": call_session.call_type.value
                },
                sender_id=user_id
            )
            
            await session_manager.broadcast_event(call_session.session_id, fallback_event)
            
            logger.info(f"Audio-only fallback enabled for user {user_id} in call {call_id}")
            return CallResult(success=True, call_id=call_id)
            
        except Exception as e:
            logger.error(f"Error enabling audio-only fallback: {e}")
            return CallResult(success=False, error_message=str(e))
    
    async def get_call_status(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a call"""
        try:
            if call_id not in self.active_calls:
                return None
            
            call_session = self.active_calls[call_id]
            call_participants = self.call_participants.get(call_id, [])
            
            return {
                "call_id": call_id,
                "session_id": call_session.session_id,
                "initiator_id": call_session.initiator_id,
                "call_type": call_session.call_type.value,
                "status": call_session.status.value,
                "started_at": call_session.started_at.isoformat() if call_session.started_at else None,
                "ended_at": call_session.ended_at.isoformat() if call_session.ended_at else None,
                "recording_enabled": call_session.recording_enabled,
                "screen_sharing_active": call_session.screen_sharing_active,
                "quality_level": call_session.quality_level.value,
                "participants": [
                    {
                        "participant_id": p.participant_id,
                        "user_id": p.user_id,
                        "display_name": p.display_name,
                        "status": p.status.value,
                        "joined_at": p.joined_at.isoformat() if p.joined_at else None,
                        "left_at": p.left_at.isoformat() if p.left_at else None,
                        "audio_enabled": p.audio_enabled,
                        "video_enabled": p.video_enabled,
                        "screen_sharing": p.screen_sharing,
                        "connection_quality": p.connection_quality.value
                    }
                    for p in call_participants
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting call status: {e}")
            return None
    
    async def get_user_active_call(self, user_id: str) -> Optional[str]:
        """Get the active call ID for a user"""
        return self.user_calls.get(user_id)
    
    async def _cache_call_session(self, call_id: str, call_session: VideoCallSession):
        """Cache call session in Redis"""
        try:
            cache_key = f"call:{call_id}:session"
            session_data = {
                "call_id": call_session.call_id,
                "session_id": call_session.session_id,
                "initiator_id": call_session.initiator_id,
                "call_type": call_session.call_type.value,
                "status": call_session.status.value,
                "participants": call_session.participants,
                "started_at": call_session.started_at.isoformat() if call_session.started_at else None,
                "ended_at": call_session.ended_at.isoformat() if call_session.ended_at else None,
                "recording_enabled": call_session.recording_enabled,
                "screen_sharing_active": call_session.screen_sharing_active,
                "quality_level": call_session.quality_level.value,
                "metadata": call_session.metadata
            }
            
            await redis_manager.set_json(cache_key, session_data, expire=3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Error caching call session: {e}")
    
    async def _send_call_invitations(self, session_id: str, call_id: str, 
                                   participants: List[str], call_type: CallType):
        """Send call invitations to participants"""
        try:
            invitation_event = CollaborativeEvent(
                event_type="call_invitation",
                event_data={
                    "call_id": call_id,
                    "call_type": call_type.value,
                    "session_id": session_id
                },
                sender_id="system",
                target_participants=participants
            )
            
            await session_manager.broadcast_event(session_id, invitation_event)
            
        except Exception as e:
            logger.error(f"Error sending call invitations: {e}")
    
    async def _cleanup_call(self, call_id: str):
        """Clean up call data when call ends"""
        try:
            # Remove from active calls
            if call_id in self.active_calls:
                del self.active_calls[call_id]
            
            # Remove participants
            if call_id in self.call_participants:
                del self.call_participants[call_id]
            
            # Remove signaling channels
            if call_id in self.signaling_channels:
                del self.signaling_channels[call_id]
            
            # Remove user call mappings
            users_to_remove = [user_id for user_id, cid in self.user_calls.items() if cid == call_id]
            for user_id in users_to_remove:
                del self.user_calls[user_id]
            
            # Clean up Redis cache
            cache_key = f"call:{call_id}:session"
            await redis_manager.delete(cache_key)
            
            logger.info(f"Cleaned up call {call_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up call: {e}")

# Global video communication system instance
video_communication = VideoCommunicationSystem()