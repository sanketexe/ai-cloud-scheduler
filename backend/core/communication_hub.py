"""
Real-Time Communication Hub for Collaborative FinOps Workspace
Provides integrated chat, video, and annotation capabilities
"""

import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.exc import IntegrityError

from .database import get_db_session
from .collaboration_models import (
    CollaborativeSession, SessionParticipant, SessionChatMessage, 
    SessionAnnotation, ParticipantRole, PermissionLevel
)
from .models import User
from .redis_config import redis_manager
from .collaborative_session_manager import session_manager, CollaborativeEvent

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of chat messages"""
    TEXT = "text"
    SYSTEM = "system"
    NOTIFICATION = "notification"
    MENTION = "mention"
    THREAD_REPLY = "thread_reply"

class AnnotationType(Enum):
    """Types of annotations"""
    COMMENT = "comment"
    HIGHLIGHT = "highlight"
    ARROW = "arrow"
    STICKY_NOTE = "sticky_note"
    DRAWING = "drawing"

@dataclass
class ChatMessage:
    """Chat message data structure"""
    message_id: str
    sender_id: str
    content: str
    message_type: MessageType = MessageType.TEXT
    thread_id: Optional[str] = None
    reply_to_id: Optional[str] = None
    mentions: List[str] = None
    attachments: List[Dict[str, Any]] = None
    timestamp: datetime = None
    edited: bool = False
    deleted: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.mentions is None:
            self.mentions = []
        if self.attachments is None:
            self.attachments = []

@dataclass
class Annotation:
    """Annotation data structure"""
    annotation_id: str
    author_id: str
    target_element: Dict[str, Any]
    content: str
    annotation_type: AnnotationType = AnnotationType.COMMENT
    position: Dict[str, Any] = None
    styling: Dict[str, Any] = None
    resolved: bool = False
    thread_id: Optional[str] = None
    parent_annotation_id: Optional[str] = None
    contextual_links: List[Dict[str, Any]] = None
    mentions: List[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.position is None:
            self.position = {}
        if self.styling is None:
            self.styling = {}
        if self.contextual_links is None:
            self.contextual_links = []
        if self.mentions is None:
            self.mentions = []

@dataclass
class MessageResult:
    """Result of message operations"""
    success: bool
    message_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class AnnotationResult:
    """Result of annotation operations"""
    success: bool
    annotation_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class NotificationResult:
    """Result of notification operations"""
    success: bool
    delivered_count: int = 0
    failed_count: int = 0
    error_message: Optional[str] = None

class CommunicationHub:
    """
    Manages real-time communication features including chat, annotations, and notifications
    """
    
    def __init__(self):
        self.active_threads: Dict[str, Set[str]] = {}  # thread_id -> set of participant_ids
        self.typing_indicators: Dict[str, Dict[str, datetime]] = {}  # session_id -> {user_id: timestamp}
        
    async def send_message(self, session_id: str, message: ChatMessage) -> MessageResult:
        """
        Send a chat message to a collaborative session
        
        Args:
            session_id: ID of the session
            message: Chat message to send
            
        Returns:
            MessageResult: Result of the send operation
        """
        try:
            async with get_db_session() as db:
                # Validate session and sender
                session = db.query(CollaborativeSession).filter(
                    CollaborativeSession.id == session_id
                ).first()
                
                if not session:
                    return MessageResult(success=False, error_message="Session not found")
                
                sender = db.query(User).filter(User.id == message.sender_id).first()
                if not sender:
                    return MessageResult(success=False, error_message="Sender not found")
                
                # Check if sender is participant in session
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == message.sender_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return MessageResult(success=False, error_message="Sender not in session")
                
                # Check permissions for messaging
                if participant.permission_level == PermissionLevel.READ_ONLY:
                    return MessageResult(success=False, error_message="No permission to send messages")
                
                # Process mentions and validate mentioned users
                processed_mentions = []
                if message.mentions:
                    for mentioned_user_id in message.mentions:
                        mentioned_participant = db.query(SessionParticipant).filter(
                            SessionParticipant.session_id == session_id,
                            SessionParticipant.user_id == mentioned_user_id,
                            SessionParticipant.left_at.is_(None)
                        ).first()
                        
                        if mentioned_participant:
                            processed_mentions.append(mentioned_user_id)
                
                # Create message record
                db_message = SessionChatMessage(
                    id=message.message_id,
                    session_id=session_id,
                    sender_id=message.sender_id,
                    content=message.content,
                    message_type=message.message_type.value,
                    thread_id=message.thread_id,
                    reply_to_id=message.reply_to_id,
                    mentions=processed_mentions,
                    attachments=message.attachments
                )
                
                db.add(db_message)
                db.commit()
                
                # Update session activity
                session.last_activity = datetime.utcnow()
                db.commit()
                
                # Store message in Redis for real-time access
                await self._cache_message(session_id, message)
                
                # Broadcast message to session participants
                message_event = CollaborativeEvent(
                    event_type="chat_message",
                    event_data={
                        "message_id": message.message_id,
                        "sender_id": message.sender_id,
                        "sender_name": f"{sender.first_name} {sender.last_name}",
                        "content": message.content,
                        "message_type": message.message_type.value,
                        "thread_id": message.thread_id,
                        "reply_to_id": message.reply_to_id,
                        "mentions": processed_mentions,
                        "attachments": message.attachments,
                        "timestamp": message.timestamp.isoformat()
                    },
                    sender_id=message.sender_id
                )
                
                await session_manager.broadcast_event(session_id, message_event)
                
                # Send mention notifications
                if processed_mentions:
                    await self._send_mention_notifications(session_id, message, processed_mentions)
                
                logger.info(f"Message {message.message_id} sent to session {session_id}")
                return MessageResult(success=True, message_id=message.message_id)
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return MessageResult(success=False, error_message=str(e))
    
    async def get_message_history(self, session_id: str, user_id: str, 
                                limit: int = 50, offset: int = 0,
                                thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get message history for a session
        
        Args:
            session_id: ID of the session
            user_id: ID of the requesting user
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            thread_id: Optional thread ID to filter messages
            
        Returns:
            List of message dictionaries
        """
        try:
            async with get_db_session() as db:
                # Verify user has access to session
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return []
                
                # Build query
                query = db.query(SessionChatMessage).filter(
                    SessionChatMessage.session_id == session_id,
                    SessionChatMessage.deleted == False
                )
                
                if thread_id:
                    query = query.filter(SessionChatMessage.thread_id == thread_id)
                
                messages = query.order_by(desc(SessionChatMessage.created_at)).offset(offset).limit(limit).all()
                
                # Convert to dictionaries with sender information
                result = []
                for msg in messages:
                    sender = db.query(User).filter(User.id == msg.sender_id).first()
                    result.append({
                        "message_id": str(msg.id),
                        "sender_id": str(msg.sender_id),
                        "sender_name": f"{sender.first_name} {sender.last_name}" if sender else "Unknown",
                        "content": msg.content,
                        "message_type": msg.message_type,
                        "thread_id": str(msg.thread_id) if msg.thread_id else None,
                        "reply_to_id": str(msg.reply_to_id) if msg.reply_to_id else None,
                        "mentions": msg.mentions or [],
                        "attachments": msg.attachments or [],
                        "timestamp": msg.created_at.isoformat(),
                        "edited": msg.edited,
                        "edited_at": msg.edited_at.isoformat() if msg.edited_at else None
                    })
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting message history: {e}")
            return []
    
    async def search_messages(self, session_id: str, user_id: str, 
                            query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search messages in a session
        
        Args:
            session_id: ID of the session
            user_id: ID of the requesting user
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching message dictionaries
        """
        try:
            async with get_db_session() as db:
                # Verify user has access to session
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return []
                
                # Search messages using full-text search
                messages = db.query(SessionChatMessage).filter(
                    SessionChatMessage.session_id == session_id,
                    SessionChatMessage.deleted == False,
                    SessionChatMessage.content.ilike(f"%{query}%")
                ).order_by(desc(SessionChatMessage.created_at)).limit(limit).all()
                
                # Convert to dictionaries with sender information
                result = []
                for msg in messages:
                    sender = db.query(User).filter(User.id == msg.sender_id).first()
                    result.append({
                        "message_id": str(msg.id),
                        "sender_id": str(msg.sender_id),
                        "sender_name": f"{sender.first_name} {sender.last_name}" if sender else "Unknown",
                        "content": msg.content,
                        "message_type": msg.message_type,
                        "thread_id": str(msg.thread_id) if msg.thread_id else None,
                        "timestamp": msg.created_at.isoformat(),
                        "relevance_score": self._calculate_relevance(msg.content, query)
                    })
                
                # Sort by relevance
                result.sort(key=lambda x: x["relevance_score"], reverse=True)
                return result
                
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return []
    
    async def create_thread(self, session_id: str, user_id: str, 
                          initial_message: str, parent_message_id: str) -> MessageResult:
        """
        Create a new message thread
        
        Args:
            session_id: ID of the session
            user_id: ID of the user creating the thread
            initial_message: Initial message content
            parent_message_id: ID of the message being replied to
            
        Returns:
            MessageResult: Result of thread creation
        """
        try:
            # Generate thread ID
            thread_id = str(uuid.uuid4())
            
            # Create initial thread message
            message = ChatMessage(
                message_id=str(uuid.uuid4()),
                sender_id=user_id,
                content=initial_message,
                message_type=MessageType.THREAD_REPLY,
                thread_id=thread_id,
                reply_to_id=parent_message_id
            )
            
            result = await self.send_message(session_id, message)
            
            if result.success:
                # Track thread
                if thread_id not in self.active_threads:
                    self.active_threads[thread_id] = set()
                self.active_threads[thread_id].add(user_id)
                
                # Broadcast thread creation
                thread_event = CollaborativeEvent(
                    event_type="thread_created",
                    event_data={
                        "thread_id": thread_id,
                        "parent_message_id": parent_message_id,
                        "creator_id": user_id
                    },
                    sender_id=user_id
                )
                
                await session_manager.broadcast_event(session_id, thread_event)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            return MessageResult(success=False, error_message=str(e))
    
    async def update_typing_indicator(self, session_id: str, user_id: str, 
                                    is_typing: bool) -> bool:
        """
        Update typing indicator for a user
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            is_typing: Whether the user is typing
            
        Returns:
            bool: Success status
        """
        try:
            if session_id not in self.typing_indicators:
                self.typing_indicators[session_id] = {}
            
            if is_typing:
                self.typing_indicators[session_id][user_id] = datetime.utcnow()
            else:
                self.typing_indicators[session_id].pop(user_id, None)
            
            # Broadcast typing indicator
            typing_event = CollaborativeEvent(
                event_type="typing_indicator",
                event_data={
                    "user_id": user_id,
                    "is_typing": is_typing
                },
                sender_id=user_id
            )
            
            await session_manager.broadcast_event(session_id, typing_event)
            return True
            
        except Exception as e:
            logger.error(f"Error updating typing indicator: {e}")
            return False
    
    async def get_typing_users(self, session_id: str) -> List[str]:
        """Get list of users currently typing"""
        try:
            if session_id not in self.typing_indicators:
                return []
            
            # Clean up old typing indicators (older than 5 seconds)
            cutoff_time = datetime.utcnow() - timedelta(seconds=5)
            typing_users = []
            
            for user_id, timestamp in list(self.typing_indicators[session_id].items()):
                if timestamp > cutoff_time:
                    typing_users.append(user_id)
                else:
                    del self.typing_indicators[session_id][user_id]
            
            return typing_users
            
        except Exception as e:
            logger.error(f"Error getting typing users: {e}")
            return []
    
    async def create_annotation(self, session_id: str, annotation: Annotation) -> AnnotationResult:
        """
        Create an annotation on dashboard elements
        
        Args:
            session_id: ID of the session
            annotation: Annotation to create
            
        Returns:
            AnnotationResult: Result of annotation creation
        """
        try:
            async with get_db_session() as db:
                # Validate session and author
                session = db.query(CollaborativeSession).filter(
                    CollaborativeSession.id == session_id
                ).first()
                
                if not session:
                    return AnnotationResult(success=False, error_message="Session not found")
                
                author = db.query(User).filter(User.id == annotation.author_id).first()
                if not author:
                    return AnnotationResult(success=False, error_message="Author not found")
                
                # Check if author is participant in session
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == annotation.author_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return AnnotationResult(success=False, error_message="Author not in session")
                
                # Check permissions for annotations
                if participant.permission_level == PermissionLevel.READ_ONLY:
                    return AnnotationResult(success=False, error_message="No permission to create annotations")
                
                # Process mentions and validate mentioned users
                processed_mentions = []
                if annotation.mentions:
                    for mentioned_user_id in annotation.mentions:
                        mentioned_participant = db.query(SessionParticipant).filter(
                            SessionParticipant.session_id == session_id,
                            SessionParticipant.user_id == mentioned_user_id,
                            SessionParticipant.left_at.is_(None)
                        ).first()
                        
                        if mentioned_participant:
                            processed_mentions.append(mentioned_user_id)
                
                # Validate parent annotation if this is a reply
                if annotation.parent_annotation_id:
                    parent_annotation = db.query(SessionAnnotation).filter(
                        SessionAnnotation.id == annotation.parent_annotation_id,
                        SessionAnnotation.session_id == session_id
                    ).first()
                    
                    if not parent_annotation:
                        return AnnotationResult(success=False, error_message="Parent annotation not found")
                    
                    # Use parent's thread_id or create new thread
                    if not annotation.thread_id:
                        annotation.thread_id = parent_annotation.thread_id or str(uuid.uuid4())
                
                # Create annotation record
                db_annotation = SessionAnnotation(
                    id=annotation.annotation_id,
                    session_id=session_id,
                    author_id=annotation.author_id,
                    annotation_type=annotation.annotation_type.value,
                    content=annotation.content,
                    target_element_id=annotation.target_element.get("element_id", ""),
                    target_element_type=annotation.target_element.get("element_type", ""),
                    position=annotation.position,
                    styling=annotation.styling,
                    thread_id=annotation.thread_id,
                    parent_annotation_id=annotation.parent_annotation_id,
                    contextual_links=annotation.contextual_links,
                    mentions=processed_mentions
                )
                
                db.add(db_annotation)
                db.commit()
                
                # Update session activity
                session.last_activity = datetime.utcnow()
                db.commit()
                
                # Broadcast annotation to session participants
                annotation_event = CollaborativeEvent(
                    event_type="annotation_created",
                    event_data={
                        "annotation_id": annotation.annotation_id,
                        "author_id": annotation.author_id,
                        "author_name": f"{author.first_name} {author.last_name}",
                        "annotation_type": annotation.annotation_type.value,
                        "content": annotation.content,
                        "target_element": annotation.target_element,
                        "position": annotation.position,
                        "styling": annotation.styling,
                        "thread_id": annotation.thread_id,
                        "parent_annotation_id": annotation.parent_annotation_id,
                        "contextual_links": annotation.contextual_links,
                        "mentions": processed_mentions,
                        "timestamp": annotation.timestamp.isoformat()
                    },
                    sender_id=annotation.author_id
                )
                
                await session_manager.broadcast_event(session_id, annotation_event)
                
                # Send mention notifications for annotations
                if processed_mentions:
                    await self._send_annotation_mention_notifications(session_id, annotation, processed_mentions)
                
                logger.info(f"Annotation {annotation.annotation_id} created in session {session_id}")
                return AnnotationResult(success=True, annotation_id=annotation.annotation_id)
                
        except Exception as e:
            logger.error(f"Error creating annotation: {e}")
            return AnnotationResult(success=False, error_message=str(e))
    
    async def create_annotation_reply(self, session_id: str, parent_annotation_id: str,
                                   reply_content: str, author_id: str,
                                   mentions: List[str] = None) -> AnnotationResult:
        """
        Create a reply to an existing annotation
        
        Args:
            session_id: ID of the session
            parent_annotation_id: ID of the parent annotation
            reply_content: Content of the reply
            author_id: ID of the user creating the reply
            mentions: List of mentioned user IDs
            
        Returns:
            AnnotationResult: Result of reply creation
        """
        try:
            async with get_db_session() as db:
                # Get parent annotation
                parent_annotation = db.query(SessionAnnotation).filter(
                    SessionAnnotation.id == parent_annotation_id,
                    SessionAnnotation.session_id == session_id
                ).first()
                
                if not parent_annotation:
                    return AnnotationResult(success=False, error_message="Parent annotation not found")
                
                # Create reply annotation
                reply_annotation = Annotation(
                    annotation_id=str(uuid.uuid4()),
                    author_id=author_id,
                    target_element={"element_id": parent_annotation.target_element_id, 
                                  "element_type": parent_annotation.target_element_type},
                    content=reply_content,
                    annotation_type=AnnotationType.COMMENT,
                    position=parent_annotation.position,
                    thread_id=parent_annotation.thread_id or str(uuid.uuid4()),
                    parent_annotation_id=parent_annotation_id,
                    mentions=mentions or []
                )
                
                return await self.create_annotation(session_id, reply_annotation)
                
        except Exception as e:
            logger.error(f"Error creating annotation reply: {e}")
            return AnnotationResult(success=False, error_message=str(e))
    
    async def add_contextual_link(self, session_id: str, annotation_id: str,
                                link_type: str, link_url: str, link_title: str,
                                user_id: str) -> AnnotationResult:
        """
        Add a contextual link to an annotation
        
        Args:
            session_id: ID of the session
            annotation_id: ID of the annotation
            link_type: Type of link (dashboard, report, resource, etc.)
            link_url: URL of the link
            link_title: Display title for the link
            user_id: ID of the user adding the link
            
        Returns:
            AnnotationResult: Result of operation
        """
        try:
            async with get_db_session() as db:
                # Find annotation
                annotation = db.query(SessionAnnotation).filter(
                    SessionAnnotation.id == annotation_id,
                    SessionAnnotation.session_id == session_id
                ).first()
                
                if not annotation:
                    return AnnotationResult(success=False, error_message="Annotation not found")
                
                # Check permissions
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return AnnotationResult(success=False, error_message="User not in session")
                
                # Add contextual link
                if not annotation.contextual_links:
                    annotation.contextual_links = []
                
                new_link = {
                    "link_id": str(uuid.uuid4()),
                    "link_type": link_type,
                    "url": link_url,
                    "title": link_title,
                    "added_by": user_id,
                    "added_at": datetime.utcnow().isoformat()
                }
                
                annotation.contextual_links.append(new_link)
                db.commit()
                
                # Broadcast link addition
                link_event = CollaborativeEvent(
                    event_type="annotation_link_added",
                    event_data={
                        "annotation_id": annotation_id,
                        "link": new_link
                    },
                    sender_id=user_id
                )
                
                await session_manager.broadcast_event(session_id, link_event)
                
                return AnnotationResult(success=True, annotation_id=annotation_id)
                
        except Exception as e:
            logger.error(f"Error adding contextual link: {e}")
            return AnnotationResult(success=False, error_message=str(e))
    
    async def archive_annotation(self, session_id: str, annotation_id: str,
                               user_id: str, archive_reason: str = None) -> AnnotationResult:
        """
        Archive an outdated annotation while preserving audit trail
        
        Args:
            session_id: ID of the session
            annotation_id: ID of the annotation to archive
            user_id: ID of the user archiving the annotation
            archive_reason: Optional reason for archiving
            
        Returns:
            AnnotationResult: Result of archival
        """
        try:
            async with get_db_session() as db:
                # Find annotation
                annotation = db.query(SessionAnnotation).filter(
                    SessionAnnotation.id == annotation_id,
                    SessionAnnotation.session_id == session_id
                ).first()
                
                if not annotation:
                    return AnnotationResult(success=False, error_message="Annotation not found")
                
                # Check permissions
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return AnnotationResult(success=False, error_message="User not in session")
                
                # Only author or moderators can archive annotations
                if (annotation.author_id != user_id and 
                    participant.role not in [ParticipantRole.OWNER, ParticipantRole.MODERATOR]):
                    return AnnotationResult(success=False, error_message="No permission to archive annotation")
                
                # Mark as archived
                annotation.archived = True
                annotation.archived_at = datetime.utcnow()
                annotation.archived_by = user_id
                annotation.archive_reason = archive_reason
                db.commit()
                
                # Broadcast archival
                archive_event = CollaborativeEvent(
                    event_type="annotation_archived",
                    event_data={
                        "annotation_id": annotation_id,
                        "archived_by": user_id,
                        "archive_reason": archive_reason
                    },
                    sender_id=user_id
                )
                
                await session_manager.broadcast_event(session_id, archive_event)
                
                return AnnotationResult(success=True, annotation_id=annotation_id)
                
        except Exception as e:
            logger.error(f"Error archiving annotation: {e}")
            return AnnotationResult(success=False, error_message=str(e))
    
    async def get_annotation_thread(self, session_id: str, thread_id: str,
                                  user_id: str) -> List[Dict[str, Any]]:
        """
        Get all annotations in a thread
        
        Args:
            session_id: ID of the session
            thread_id: ID of the thread
            user_id: ID of the requesting user
            
        Returns:
            List of annotation dictionaries in thread order
        """
        try:
            async with get_db_session() as db:
                # Verify user has access to session
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return []
                
                # Get thread annotations
                annotations = db.query(SessionAnnotation).filter(
                    SessionAnnotation.session_id == session_id,
                    SessionAnnotation.thread_id == thread_id,
                    SessionAnnotation.archived == False
                ).order_by(SessionAnnotation.created_at).all()
                
                # Convert to dictionaries with author information
                result = []
                for annotation in annotations:
                    author = db.query(User).filter(User.id == annotation.author_id).first()
                    result.append({
                        "annotation_id": str(annotation.id),
                        "author_id": str(annotation.author_id),
                        "author_name": f"{author.first_name} {author.last_name}" if author else "Unknown",
                        "annotation_type": annotation.annotation_type,
                        "content": annotation.content,
                        "target_element": {
                            "element_id": annotation.target_element_id,
                            "element_type": annotation.target_element_type
                        },
                        "position": annotation.position,
                        "styling": annotation.styling,
                        "thread_id": str(annotation.thread_id) if annotation.thread_id else None,
                        "parent_annotation_id": str(annotation.parent_annotation_id) if annotation.parent_annotation_id else None,
                        "contextual_links": annotation.contextual_links or [],
                        "mentions": annotation.mentions or [],
                        "resolved": annotation.resolved,
                        "resolved_at": annotation.resolved_at.isoformat() if annotation.resolved_at else None,
                        "timestamp": annotation.created_at.isoformat()
                    })
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting annotation thread: {e}")
            return []
    
    async def get_session_annotations(self, session_id: str, user_id: str,
                                    element_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get annotations for a session or specific element
        
        Args:
            session_id: ID of the session
            user_id: ID of the requesting user
            element_id: Optional element ID to filter annotations
            
        Returns:
            List of annotation dictionaries
        """
        try:
            async with get_db_session() as db:
                # Verify user has access to session
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return []
                
                # Build query
                query = db.query(SessionAnnotation).filter(
                    SessionAnnotation.session_id == session_id
                )
                
                if element_id:
                    query = query.filter(SessionAnnotation.target_element_id == element_id)
                
                annotations = query.order_by(SessionAnnotation.created_at).all()
                
                # Convert to dictionaries with author information
                result = []
                for annotation in annotations:
                    author = db.query(User).filter(User.id == annotation.author_id).first()
                    result.append({
                        "annotation_id": str(annotation.id),
                        "author_id": str(annotation.author_id),
                        "author_name": f"{author.first_name} {author.last_name}" if author else "Unknown",
                        "annotation_type": annotation.annotation_type,
                        "content": annotation.content,
                        "target_element": {
                            "element_id": annotation.target_element_id,
                            "element_type": annotation.target_element_type
                        },
                        "position": annotation.position,
                        "styling": annotation.styling,
                        "thread_id": str(annotation.thread_id) if annotation.thread_id else None,
                        "parent_annotation_id": str(annotation.parent_annotation_id) if annotation.parent_annotation_id else None,
                        "contextual_links": annotation.contextual_links or [],
                        "mentions": annotation.mentions or [],
                        "resolved": annotation.resolved,
                        "resolved_at": annotation.resolved_at.isoformat() if annotation.resolved_at else None,
                        "archived": annotation.archived,
                        "archived_at": annotation.archived_at.isoformat() if annotation.archived_at else None,
                        "timestamp": annotation.created_at.isoformat()
                    })
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting session annotations: {e}")
            return []
    
    async def resolve_annotation(self, session_id: str, annotation_id: str, 
                               user_id: str) -> AnnotationResult:
        """
        Resolve an annotation
        
        Args:
            session_id: ID of the session
            annotation_id: ID of the annotation to resolve
            user_id: ID of the user resolving the annotation
            
        Returns:
            AnnotationResult: Result of resolution
        """
        try:
            async with get_db_session() as db:
                # Find annotation
                annotation = db.query(SessionAnnotation).filter(
                    SessionAnnotation.id == annotation_id,
                    SessionAnnotation.session_id == session_id
                ).first()
                
                if not annotation:
                    return AnnotationResult(success=False, error_message="Annotation not found")
                
                # Check permissions
                participant = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                    SessionParticipant.left_at.is_(None)
                ).first()
                
                if not participant:
                    return AnnotationResult(success=False, error_message="User not in session")
                
                # Only author or moderators can resolve annotations
                if (annotation.author_id != user_id and 
                    participant.role not in [ParticipantRole.OWNER, ParticipantRole.MODERATOR]):
                    return AnnotationResult(success=False, error_message="No permission to resolve annotation")
                
                # Mark as resolved
                annotation.resolved = True
                annotation.resolved_at = datetime.utcnow()
                annotation.resolved_by = user_id
                db.commit()
                
                # Broadcast resolution
                resolve_event = CollaborativeEvent(
                    event_type="annotation_resolved",
                    event_data={
                        "annotation_id": annotation_id,
                        "resolved_by": user_id
                    },
                    sender_id=user_id
                )
                
                await session_manager.broadcast_event(session_id, resolve_event)
                
                return AnnotationResult(success=True, annotation_id=annotation_id)
                
        except Exception as e:
            logger.error(f"Error resolving annotation: {e}")
            return AnnotationResult(success=False, error_message=str(e))
    
    async def _cache_message(self, session_id: str, message: ChatMessage):
        """Cache message in Redis for real-time access"""
        try:
            cache_key = f"session:{session_id}:messages"
            message_data = {
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "content": message.content,
                "message_type": message.message_type.value,
                "thread_id": message.thread_id,
                "reply_to_id": message.reply_to_id,
                "mentions": message.mentions,
                "attachments": message.attachments,
                "timestamp": message.timestamp.isoformat()
            }
            
            # Add to message list (keep last 100 messages)
            await redis_manager.lpush(cache_key, message_data)
            await redis_manager.ltrim(cache_key, 0, 99)
            await redis_manager.expire(cache_key, 3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Error caching message: {e}")
    
    async def _send_mention_notifications(self, session_id: str, message: ChatMessage, 
                                        mentioned_users: List[str]):
        """Send notifications for mentioned users"""
        try:
            for mentioned_user_id in mentioned_users:
                notification_event = CollaborativeEvent(
                    event_type="mention_notification",
                    event_data={
                        "message_id": message.message_id,
                        "sender_id": message.sender_id,
                        "content": message.content,
                        "mentioned_user_id": mentioned_user_id,
                        "session_id": session_id
                    },
                    sender_id=message.sender_id,
                    target_participants=[mentioned_user_id]
                )
                
                await session_manager.broadcast_event(session_id, notification_event)
                
        except Exception as e:
            logger.error(f"Error sending mention notifications: {e}")
    
    async def _send_annotation_mention_notifications(self, session_id: str, annotation: Annotation, 
                                                   mentioned_users: List[str]):
        """Send notifications for mentioned users in annotations"""
        try:
            for mentioned_user_id in mentioned_users:
                notification_event = CollaborativeEvent(
                    event_type="annotation_mention_notification",
                    event_data={
                        "annotation_id": annotation.annotation_id,
                        "author_id": annotation.author_id,
                        "content": annotation.content,
                        "target_element": annotation.target_element,
                        "mentioned_user_id": mentioned_user_id,
                        "session_id": session_id
                    },
                    sender_id=annotation.author_id,
                    target_participants=[mentioned_user_id]
                )
                
                await session_manager.broadcast_event(session_id, notification_event)
                
        except Exception as e:
            logger.error(f"Error sending annotation mention notifications: {e}")
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score for search results"""
        try:
            content_lower = content.lower()
            query_lower = query.lower()
            
            # Simple relevance scoring
            if query_lower == content_lower:
                return 1.0
            elif query_lower in content_lower:
                return 0.8
            else:
                # Count matching words
                query_words = query_lower.split()
                content_words = content_lower.split()
                matches = sum(1 for word in query_words if word in content_words)
                return matches / len(query_words) if query_words else 0.0
                
        except Exception:
            return 0.0

# Global communication hub instance
communication_hub = CommunicationHub()