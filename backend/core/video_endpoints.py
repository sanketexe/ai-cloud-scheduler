"""
API endpoints for Video Communication System
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
from .video_communication import (
    video_communication, VideoCallSession, CallParticipant, WebRTCOffer,
    CallType, CallStatus, ParticipantCallStatus, CallQuality
)
from .collaboration_models import ParticipantRole, PermissionLevel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/video", tags=["video"])
security = HTTPBearer()

# Pydantic models for API

class InitiateCallRequest(BaseModel):
    """Request model for initiating calls"""
    participants: List[str] = Field(..., min_items=1, max_items=50)
    call_type: str = Field("video", regex="^(audio_only|video|screen_share|co_browse)$")

    @validator('participants')
    def validate_participants(cls, v):
        # Validate UUID format for participant IDs
        for participant_id in v:
            try:
                UUID(participant_id)
            except ValueError:
                raise ValueError(f"Invalid participant ID format: {participant_id}")
        return v

class WebRTCSignalingRequest(BaseModel):
    """Request model for WebRTC signaling"""
    to_user_id: str = Field(..., regex="^[0-9a-f-]{36}$")
    offer_type: str = Field(..., regex="^(offer|answer|ice_candidate)$")
    sdp_data: Optional[str] = None
    ice_candidate: Optional[Dict[str, Any]] = None

    @validator('sdp_data')
    def validate_sdp_data(cls, v, values):
        if values.get('offer_type') in ['offer', 'answer'] and not v:
            raise ValueError("SDP data required for offer/answer")
        return v

    @validator('ice_candidate')
    def validate_ice_candidate(cls, v, values):
        if values.get('offer_type') == 'ice_candidate' and not v:
            raise ValueError("ICE candidate data required for ice_candidate type")
        return v

class ToggleScreenShareRequest(BaseModel):
    """Request model for toggling screen share"""
    enable: bool

class StartCoBrowsingRequest(BaseModel):
    """Request model for starting co-browsing"""
    dashboard_url: str = Field(..., min_length=1, max_length=500)

class CoBrowseActionRequest(BaseModel):
    """Request model for co-browsing actions"""
    action_type: str = Field(..., regex="^(navigate|scroll|click|filter|zoom|select)$")
    action_data: Dict[str, Any] = Field(..., description="Action-specific data")

class UpdateCallQualityRequest(BaseModel):
    """Request model for updating call quality"""
    quality: str = Field(..., regex="^(high|medium|low|poor)$")

class HandleConnectionIssueRequest(BaseModel):
    """Request model for handling connection issues"""
    issue_type: str = Field(..., regex="^(network|audio|video|signaling)$")
    severity: str = Field(..., regex="^(low|medium|high|critical)$")

class CallStatusResponse(BaseModel):
    """Response model for call status"""
    call_id: str
    session_id: str
    initiator_id: str
    call_type: str
    status: str
    started_at: Optional[str]
    ended_at: Optional[str]
    recording_enabled: bool
    screen_sharing_active: bool
    quality_level: str
    participants: List[Dict[str, Any]]

class CallParticipantResponse(BaseModel):
    """Response model for call participants"""
    participant_id: str
    user_id: str
    display_name: str
    status: str
    joined_at: Optional[str]
    left_at: Optional[str]
    audio_enabled: bool
    video_enabled: bool
    screen_sharing: bool
    connection_quality: str

# Call Management Endpoints

@router.post("/sessions/{session_id}/calls")
async def initiate_call(
    session_id: str,
    request: InitiateCallRequest,
    current_user: User = Depends(get_current_user)
):
    """Initiate a video call within a collaborative session"""
    try:
        result = await video_communication.initiate_call(
            session_id=session_id,
            initiator_id=str(current_user.id),
            participants=request.participants,
            call_type=CallType(request.call_type)
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "call_id": result.call_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating call: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate call")

@router.post("/calls/{call_id}/join")
async def join_call(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Join an existing video call"""
    try:
        result = await video_communication.join_call(
            call_id=call_id,
            user_id=str(current_user.id)
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "call_id": result.call_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining call: {e}")
        raise HTTPException(status_code=500, detail="Failed to join call")

@router.post("/calls/{call_id}/leave")
async def leave_call(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Leave a video call"""
    try:
        result = await video_communication.leave_call(
            call_id=call_id,
            user_id=str(current_user.id)
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "call_id": result.call_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error leaving call: {e}")
        raise HTTPException(status_code=500, detail="Failed to leave call")

@router.get("/calls/{call_id}/status", response_model=CallStatusResponse)
async def get_call_status(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get current status of a video call"""
    try:
        call_status = await video_communication.get_call_status(call_id)
        
        if not call_status:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Check if user is participant in the call
        user_is_participant = any(
            p["user_id"] == str(current_user.id) 
            for p in call_status["participants"]
        )
        
        if not user_is_participant:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return CallStatusResponse(**call_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get call status")

@router.get("/users/{user_id}/active-call")
async def get_user_active_call(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the active call for a user"""
    try:
        # Users can only check their own active call or if they're in the same session
        if user_id != str(current_user.id):
            # TODO: Add session-based permission check
            raise HTTPException(status_code=403, detail="Access denied")
        
        active_call_id = await video_communication.get_user_active_call(user_id)
        
        return {
            "user_id": user_id,
            "active_call_id": active_call_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user active call: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user active call")

# WebRTC Signaling Endpoints

@router.post("/calls/{call_id}/signaling")
async def handle_webrtc_signaling(
    call_id: str,
    request: WebRTCSignalingRequest,
    current_user: User = Depends(get_current_user)
):
    """Handle WebRTC signaling messages"""
    try:
        # Create WebRTC offer object
        offer = WebRTCOffer(
            call_id=call_id,
            from_user_id=str(current_user.id),
            to_user_id=request.to_user_id,
            offer_type=request.offer_type,
            sdp_data=request.sdp_data,
            ice_candidate=request.ice_candidate
        )
        
        result = await video_communication.handle_webrtc_signaling(offer)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "message_id": result.message_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling WebRTC signaling: {e}")
        raise HTTPException(status_code=500, detail="Failed to handle signaling")

# Screen Sharing and Co-browsing Endpoints

@router.post("/calls/{call_id}/screen-share")
async def toggle_screen_sharing(
    call_id: str,
    request: ToggleScreenShareRequest,
    current_user: User = Depends(get_current_user)
):
    """Toggle screen sharing for current user"""
    try:
        result = await video_communication.toggle_screen_sharing(
            call_id=call_id,
            user_id=str(current_user.id),
            enable=request.enable
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "screen_sharing": request.enable
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling screen sharing: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle screen sharing")

@router.post("/calls/{call_id}/co-browse/start")
async def start_co_browsing(
    call_id: str,
    request: StartCoBrowsingRequest,
    current_user: User = Depends(get_current_user)
):
    """Start co-browsing session for synchronized dashboard viewing"""
    try:
        result = await video_communication.start_co_browsing(
            call_id=call_id,
            user_id=str(current_user.id),
            dashboard_url=request.dashboard_url
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "call_id": result.call_id,
            "dashboard_url": request.dashboard_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting co-browsing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start co-browsing")

@router.post("/calls/{call_id}/co-browse/action")
async def sync_co_browse_action(
    call_id: str,
    request: CoBrowseActionRequest,
    current_user: User = Depends(get_current_user)
):
    """Synchronize co-browsing actions across participants"""
    try:
        result = await video_communication.sync_co_browse_action(
            call_id=call_id,
            user_id=str(current_user.id),
            action_type=request.action_type,
            action_data=request.action_data
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "action_type": request.action_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing co-browse action: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync co-browse action")

@router.post("/calls/{call_id}/co-browse/stop")
async def stop_co_browsing(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Stop co-browsing session"""
    try:
        result = await video_communication.stop_co_browsing(
            call_id=call_id,
            user_id=str(current_user.id)
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "call_id": result.call_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping co-browsing: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop co-browsing")

@router.get("/calls/{call_id}/co-browse/status")
async def get_co_browse_status(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get co-browsing session status"""
    try:
        call_status = await video_communication.get_call_status(call_id)
        
        if not call_status:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Check if user is participant in the call
        user_is_participant = any(
            p["user_id"] == str(current_user.id) 
            for p in call_status["participants"]
        )
        
        if not user_is_participant:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Extract co-browsing info from call metadata
        co_browsing_info = call_status.get("metadata", {}).get("co_browsing", {})
        
        return {
            "call_id": call_id,
            "co_browsing": {
                "active": co_browsing_info.get("active", False),
                "session_id": co_browsing_info.get("session_id"),
                "dashboard_url": co_browsing_info.get("dashboard_url"),
                "initiator_id": co_browsing_info.get("initiator_id"),
                "started_at": co_browsing_info.get("started_at"),
                "ended_at": co_browsing_info.get("ended_at")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting co-browse status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get co-browse status")

# Call Quality Management and Fallback Endpoints

@router.post("/calls/{call_id}/quality")
async def update_call_quality(
    call_id: str,
    request: UpdateCallQualityRequest,
    current_user: User = Depends(get_current_user)
):
    """Update call quality for adaptive streaming"""
    try:
        result = await video_communication.update_call_quality(
            call_id=call_id,
            user_id=str(current_user.id),
            quality=CallQuality(request.quality)
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "quality": request.quality
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating call quality: {e}")
        raise HTTPException(status_code=500, detail="Failed to update call quality")

@router.post("/calls/{call_id}/connection-issue")
async def handle_connection_issue(
    call_id: str,
    request: HandleConnectionIssueRequest,
    current_user: User = Depends(get_current_user)
):
    """Handle connection issues and implement fallback mechanisms"""
    try:
        result = await video_communication.handle_connection_issues(
            call_id=call_id,
            user_id=str(current_user.id),
            issue_type=request.issue_type,
            severity=request.severity
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "issue_handled": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling connection issue: {e}")
        raise HTTPException(status_code=500, detail="Failed to handle connection issue")

@router.post("/calls/{call_id}/reconnect")
async def attempt_reconnection(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Attempt to reconnect to a call after connection loss"""
    try:
        result = await video_communication.attempt_reconnection(
            call_id=call_id,
            user_id=str(current_user.id)
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "reconnecting": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error attempting reconnection: {e}")
        raise HTTPException(status_code=500, detail="Failed to attempt reconnection")

@router.post("/calls/{call_id}/audio-only-fallback")
async def enable_audio_only_fallback(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Enable audio-only fallback when video fails"""
    try:
        result = await video_communication.enable_audio_only_fallback(
            call_id=call_id,
            user_id=str(current_user.id)
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "audio_only": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling audio-only fallback: {e}")
        raise HTTPException(status_code=500, detail="Failed to enable audio-only fallback")

# Call Analytics and Monitoring Endpoints

@router.get("/sessions/{session_id}/call-history")
async def get_session_call_history(
    session_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """Get call history for a session"""
    try:
        # TODO: Implement call history storage and retrieval
        # For now, return empty list as this would require database storage
        
        return {
            "session_id": session_id,
            "calls": [],
            "total_count": 0,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error getting call history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get call history")

@router.get("/calls/{call_id}/metrics")
async def get_call_metrics(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get call quality and performance metrics"""
    try:
        call_status = await video_communication.get_call_status(call_id)
        
        if not call_status:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Check if user is participant in the call
        user_is_participant = any(
            p["user_id"] == str(current_user.id) 
            for p in call_status["participants"]
        )
        
        if not user_is_participant:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Calculate metrics
        total_participants = len(call_status["participants"])
        active_participants = len([
            p for p in call_status["participants"] 
            if p["status"] not in ["disconnected"]
        ])
        
        # Calculate call duration
        duration_seconds = 0
        if call_status["started_at"]:
            start_time = datetime.fromisoformat(call_status["started_at"])
            end_time = datetime.fromisoformat(call_status["ended_at"]) if call_status["ended_at"] else datetime.utcnow()
            duration_seconds = (end_time - start_time).total_seconds()
        
        # Quality distribution
        quality_counts = {}
        for participant in call_status["participants"]:
            quality = participant["connection_quality"]
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        return {
            "call_id": call_id,
            "metrics": {
                "total_participants": total_participants,
                "active_participants": active_participants,
                "duration_seconds": duration_seconds,
                "overall_quality": call_status["quality_level"],
                "screen_sharing_active": call_status["screen_sharing_active"],
                "recording_enabled": call_status["recording_enabled"],
                "quality_distribution": quality_counts
            },
            "participants": call_status["participants"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get call metrics")

# WebRTC Configuration Endpoints

@router.get("/webrtc/config")
async def get_webrtc_config(
    current_user: User = Depends(get_current_user)
):
    """Get WebRTC configuration for client"""
    try:
        # Return STUN/TURN server configuration
        # In production, this should come from environment variables
        config = {
            "iceServers": [
                {
                    "urls": ["stun:stun.l.google.com:19302"]
                },
                # Add TURN servers for production
                # {
                #     "urls": ["turn:your-turn-server.com:3478"],
                #     "username": "username",
                #     "credential": "password"
                # }
            ],
            "iceCandidatePoolSize": 10,
            "bundlePolicy": "max-bundle",
            "rtcpMuxPolicy": "require"
        }
        
        return {
            "webrtc_config": config,
            "user_id": str(current_user.id)
        }
        
    except Exception as e:
        logger.error(f"Error getting WebRTC config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get WebRTC config")

# Call Recording Endpoints (Placeholder)

@router.post("/calls/{call_id}/recording/start")
async def start_call_recording(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Start call recording (placeholder)"""
    try:
        # TODO: Implement call recording functionality
        # This would require integration with media recording services
        
        return {
            "success": False,
            "message": "Call recording not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Error starting call recording: {e}")
        raise HTTPException(status_code=500, detail="Failed to start recording")

@router.post("/calls/{call_id}/recording/stop")
async def stop_call_recording(
    call_id: str,
    current_user: User = Depends(get_current_user)
):
    """Stop call recording (placeholder)"""
    try:
        # TODO: Implement call recording functionality
        
        return {
            "success": False,
            "message": "Call recording not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Error stopping call recording: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop recording")