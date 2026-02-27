"""
API endpoints for Real-Time Communication Hub
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
from .communication_hub import (
    communication_hub, ChatMessage, Annotation, MessageType, AnnotationType
)
from .video_communication import video_communication
from .collaboration_models import ParticipantRole, PermissionLevel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/communication", tags=["communication"])
security = HTTPBearer()

# Pydantic models for API

class SendMessageRequest(BaseModel):
    """Request model for sending messages"""
    content: str = Field(..., min_length=1, max_length=5000)
    message_type: str = Field("text", regex="^(text|system|notification|mention|thread_reply)$")
    thread_id: Optional[str] = Field(None, regex="^[0-9a-f-]{36}$")
    reply_to_id: Optional[str] = Field(None, regex="^[0-9a-f-]{36}$")
    mentions: Optional[List[str]] = Field(default_factory=list)
    attachments: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

class CreateThreadRequest(BaseModel):
    """Request model for creating message threads"""
    initial_message: str = Field(..., min_length=1, max_length=5000)
    parent_message_id: str = Field(..., regex="^[0-9a-f-]{36}$")

class MessageSearchRequest(BaseModel):
    """Request model for message search"""
    query: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(20, ge=1, le=100)

class TypingIndicatorRequest(BaseModel):
    """Request model for typing indicators"""
    is_typing: bool

class CreateAnnotationRequest(BaseModel):
    """Request model for creating annotations"""
    annotation_type: str = Field("comment", regex="^(comment|highlight|arrow|sticky_note|drawing)$")
    content: str = Field(..., min_length=1, max_length=2000)
    target_element: Dict[str, Any] = Field(..., description="Target element information")
    position: Dict[str, Any] = Field(..., description="Position coordinates")
    styling: Optional[Dict[str, Any]] = Field(default_factory=dict)
    thread_id: Optional[str] = Field(None, regex="^[0-9a-f-]{36}$")
    parent_annotation_id: Optional[str] = Field(None, regex="^[0-9a-f-]{36}$")
    contextual_links: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    mentions: Optional[List[str]] = Field(default_factory=list)

    @validator('target_element')
    def validate_target_element(cls, v):
        required_fields = ['element_id', 'element_type']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"target_element must contain {field}")
        return v

    @validator('position')
    def validate_position(cls, v):
        required_fields = ['x', 'y']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"position must contain {field}")
        return v

class CreateAnnotationReplyRequest(BaseModel):
    """Request model for creating annotation replies"""
    content: str = Field(..., min_length=1, max_length=2000)
    mentions: Optional[List[str]] = Field(default_factory=list)

class AddContextualLinkRequest(BaseModel):
    """Request model for adding contextual links"""
    link_type: str = Field(..., regex="^(dashboard|report|resource|documentation)$")
    link_url: str = Field(..., min_length=1, max_length=500)
    link_title: str = Field(..., min_length=1, max_length=200)

class ArchiveAnnotationRequest(BaseModel):
    """Request model for archiving annotations"""
    archive_reason: Optional[str] = Field(None, max_length=500)

class MessageResponse(BaseModel):
    """Response model for messages"""
    message_id: str
    sender_id: str
    sender_name: str
    content: str
    message_type: str
    thread_id: Optional[str]
    reply_to_id: Optional[str]
    mentions: List[str]
    attachments: List[Dict[str, Any]]
    timestamp: str
    edited: bool
    edited_at: Optional[str]

class AnnotationResponse(BaseModel):
    """Response model for annotations"""
    annotation_id: str
    author_id: str
    author_name: str
    annotation_type: str
    content: str
    target_element: Dict[str, Any]
    position: Dict[str, Any]
    styling: Dict[str, Any]
    resolved: bool
    resolved_at: Optional[str]
    thread_id: Optional[str]
    parent_annotation_id: Optional[str]
    contextual_links: List[Dict[str, Any]]
    mentions: List[str]
    timestamp: str

# Chat Message Endpoints

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    current_user: User = Depends(get_current_user)
):
    """Send a chat message to a collaborative session"""
    try:
        # Create message object
        message = ChatMessage(
            message_id=str(UUID()),
            sender_id=str(current_user.id),
            content=request.content,
            message_type=MessageType(request.message_type),
            thread_id=request.thread_id,
            reply_to_id=request.reply_to_id,
            mentions=request.mentions,
            attachments=request.attachments
        )
        
        result = await communication_hub.send_message(session_id, message)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "message_id": result.message_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")

@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def get_message_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    thread_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """Get message history for a session"""
    try:
        messages = await communication_hub.get_message_history(
            session_id=session_id,
            user_id=str(current_user.id),
            limit=limit,
            offset=offset,
            thread_id=thread_id
        )
        
        return [
            MessageResponse(
                message_id=msg["message_id"],
                sender_id=msg["sender_id"],
                sender_name=msg["sender_name"],
                content=msg["content"],
                message_type=msg["message_type"],
                thread_id=msg["thread_id"],
                reply_to_id=msg["reply_to_id"],
                mentions=msg["mentions"],
                attachments=msg["attachments"],
                timestamp=msg["timestamp"],
                edited=msg["edited"],
                edited_at=msg["edited_at"]
            )
            for msg in messages
        ]
        
    except Exception as e:
        logger.error(f"Error getting message history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get message history")

@router.post("/sessions/{session_id}/messages/search")
async def search_messages(
    session_id: str,
    request: MessageSearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Search messages in a session"""
    try:
        messages = await communication_hub.search_messages(
            session_id=session_id,
            user_id=str(current_user.id),
            query=request.query,
            limit=request.limit
        )
        
        return {
            "results": messages,
            "total_count": len(messages)
        }
        
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to search messages")

@router.post("/sessions/{session_id}/threads")
async def create_thread(
    session_id: str,
    request: CreateThreadRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new message thread"""
    try:
        result = await communication_hub.create_thread(
            session_id=session_id,
            user_id=str(current_user.id),
            initial_message=request.initial_message,
            parent_message_id=request.parent_message_id
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "message_id": result.message_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating thread: {e}")
        raise HTTPException(status_code=500, detail="Failed to create thread")

@router.post("/sessions/{session_id}/typing")
async def update_typing_indicator(
    session_id: str,
    request: TypingIndicatorRequest,
    current_user: User = Depends(get_current_user)
):
    """Update typing indicator for current user"""
    try:
        success = await communication_hub.update_typing_indicator(
            session_id=session_id,
            user_id=str(current_user.id),
            is_typing=request.is_typing
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update typing indicator")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating typing indicator: {e}")
        raise HTTPException(status_code=500, detail="Failed to update typing indicator")

@router.get("/sessions/{session_id}/typing")
async def get_typing_users(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get list of users currently typing"""
    try:
        typing_users = await communication_hub.get_typing_users(session_id)
        
        return {
            "typing_users": typing_users
        }
        
    except Exception as e:
        logger.error(f"Error getting typing users: {e}")
        raise HTTPException(status_code=500, detail="Failed to get typing users")

# Annotation Endpoints

@router.post("/sessions/{session_id}/annotations")
async def create_annotation(
    session_id: str,
    request: CreateAnnotationRequest,
    current_user: User = Depends(get_current_user)
):
    """Create an annotation on dashboard elements"""
    try:
        # Create annotation object
        annotation = Annotation(
            annotation_id=str(UUID()),
            author_id=str(current_user.id),
            target_element=request.target_element,
            content=request.content,
            annotation_type=AnnotationType(request.annotation_type),
            position=request.position,
            styling=request.styling,
            thread_id=request.thread_id,
            parent_annotation_id=request.parent_annotation_id,
            contextual_links=request.contextual_links,
            mentions=request.mentions
        )
        
        result = await communication_hub.create_annotation(session_id, annotation)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "annotation_id": result.annotation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating annotation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create annotation")

@router.get("/sessions/{session_id}/annotations", response_model=List[AnnotationResponse])
async def get_session_annotations(
    session_id: str,
    element_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """Get annotations for a session or specific element"""
    try:
        annotations = await communication_hub.get_session_annotations(
            session_id=session_id,
            user_id=str(current_user.id),
            element_id=element_id
        )
        
        return [
            AnnotationResponse(
                annotation_id=ann["annotation_id"],
                author_id=ann["author_id"],
                author_name=ann["author_name"],
                annotation_type=ann["annotation_type"],
                content=ann["content"],
                target_element=ann["target_element"],
                position=ann["position"],
                styling=ann["styling"],
                resolved=ann["resolved"],
                resolved_at=ann["resolved_at"],
                thread_id=ann.get("thread_id"),
                parent_annotation_id=ann.get("parent_annotation_id"),
                contextual_links=ann.get("contextual_links", []),
                mentions=ann.get("mentions", []),
                timestamp=ann["timestamp"]
            )
            for ann in annotations
        ]
        
    except Exception as e:
        logger.error(f"Error getting annotations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get annotations")

@router.post("/sessions/{session_id}/annotations/{annotation_id}/resolve")
async def resolve_annotation(
    session_id: str,
    annotation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Resolve an annotation"""
    try:
        result = await communication_hub.resolve_annotation(
            session_id=session_id,
            annotation_id=annotation_id,
            user_id=str(current_user.id)
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "annotation_id": result.annotation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving annotation: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve annotation")

@router.post("/sessions/{session_id}/annotations/{annotation_id}/reply")
async def create_annotation_reply(
    session_id: str,
    annotation_id: str,
    request: CreateAnnotationReplyRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a reply to an annotation"""
    try:
        result = await communication_hub.create_annotation_reply(
            session_id=session_id,
            parent_annotation_id=annotation_id,
            reply_content=request.content,
            author_id=str(current_user.id),
            mentions=request.mentions
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "annotation_id": result.annotation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating annotation reply: {e}")
        raise HTTPException(status_code=500, detail="Failed to create annotation reply")

@router.post("/sessions/{session_id}/annotations/{annotation_id}/links")
async def add_contextual_link(
    session_id: str,
    annotation_id: str,
    request: AddContextualLinkRequest,
    current_user: User = Depends(get_current_user)
):
    """Add a contextual link to an annotation"""
    try:
        result = await communication_hub.add_contextual_link(
            session_id=session_id,
            annotation_id=annotation_id,
            link_type=request.link_type,
            link_url=request.link_url,
            link_title=request.link_title,
            user_id=str(current_user.id)
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "annotation_id": result.annotation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding contextual link: {e}")
        raise HTTPException(status_code=500, detail="Failed to add contextual link")

@router.post("/sessions/{session_id}/annotations/{annotation_id}/archive")
async def archive_annotation(
    session_id: str,
    annotation_id: str,
    request: ArchiveAnnotationRequest,
    current_user: User = Depends(get_current_user)
):
    """Archive an outdated annotation"""
    try:
        result = await communication_hub.archive_annotation(
            session_id=session_id,
            annotation_id=annotation_id,
            user_id=str(current_user.id),
            archive_reason=request.archive_reason
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        return {
            "success": True,
            "annotation_id": result.annotation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error archiving annotation: {e}")
        raise HTTPException(status_code=500, detail="Failed to archive annotation")

@router.get("/sessions/{session_id}/annotation-threads/{thread_id}")
async def get_annotation_thread(
    session_id: str,
    thread_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all annotations in a thread"""
    try:
        annotations = await communication_hub.get_annotation_thread(
            session_id=session_id,
            thread_id=thread_id,
            user_id=str(current_user.id)
        )
        
        return {
            "thread_id": thread_id,
            "annotations": annotations
        }
        
    except Exception as e:
        logger.error(f"Error getting annotation thread: {e}")
        raise HTTPException(status_code=500, detail="Failed to get annotation thread")

# Conversation Management Endpoints

@router.get("/sessions/{session_id}/conversations")
async def get_conversation_summary(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get conversation summary and statistics"""
    try:
        # Get message statistics
        messages = await communication_hub.get_message_history(
            session_id=session_id,
            user_id=str(current_user.id),
            limit=1000  # Get more for statistics
        )
        
        # Calculate statistics
        total_messages = len(messages)
        participants = set(msg["sender_id"] for msg in messages)
        threads = set(msg["thread_id"] for msg in messages if msg["thread_id"])
        mentions = sum(len(msg["mentions"]) for msg in messages)
        
        # Get recent activity (last 24 hours)
        from datetime import datetime, timedelta
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_messages = [
            msg for msg in messages 
            if datetime.fromisoformat(msg["timestamp"].replace('Z', '+00:00')) > cutoff_time
        ]
        
        return {
            "session_id": session_id,
            "statistics": {
                "total_messages": total_messages,
                "unique_participants": len(participants),
                "active_threads": len(threads),
                "total_mentions": mentions,
                "recent_activity": len(recent_messages)
            },
            "recent_messages": recent_messages[:10]  # Last 10 messages
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conversation summary")

@router.get("/sessions/{session_id}/activity")
async def get_communication_activity(
    session_id: str,
    hours: int = Query(24, ge=1, le=168),  # Last 1-168 hours
    current_user: User = Depends(get_current_user)
):
    """Get communication activity metrics"""
    try:
        # Get messages for the specified time period
        messages = await communication_hub.get_message_history(
            session_id=session_id,
            user_id=str(current_user.id),
            limit=10000  # Large limit for activity analysis
        )
        
        # Filter by time period
        from datetime import datetime, timedelta
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        filtered_messages = [
            msg for msg in messages 
            if datetime.fromisoformat(msg["timestamp"].replace('Z', '+00:00')) > cutoff_time
        ]
        
        # Calculate activity metrics
        activity_by_hour = {}
        activity_by_user = {}
        
        for msg in filtered_messages:
            # Activity by hour
            msg_time = datetime.fromisoformat(msg["timestamp"].replace('Z', '+00:00'))
            hour_key = msg_time.strftime("%Y-%m-%d %H:00")
            activity_by_hour[hour_key] = activity_by_hour.get(hour_key, 0) + 1
            
            # Activity by user
            user_id = msg["sender_id"]
            activity_by_user[user_id] = activity_by_user.get(user_id, 0) + 1
        
        return {
            "session_id": session_id,
            "time_period_hours": hours,
            "total_messages": len(filtered_messages),
            "activity_by_hour": activity_by_hour,
            "activity_by_user": activity_by_user,
            "peak_activity_hour": max(activity_by_hour.items(), key=lambda x: x[1])[0] if activity_by_hour else None
        }
        
    except Exception as e:
        logger.error(f"Error getting communication activity: {e}")
        raise HTTPException(status_code=500, detail="Failed to get communication activity")