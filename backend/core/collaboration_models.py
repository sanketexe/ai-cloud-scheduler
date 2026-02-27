"""
SQLAlchemy models for Real-Time Collaborative FinOps Workspace
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Dict, Any, List, Optional

from sqlalchemy import (
    Column, String, DateTime, Boolean, Text, Integer, Numeric, 
    ForeignKey, Index, JSON, Enum, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from .database import Base

class SessionType(PyEnum):
    """Types of collaborative sessions"""
    COST_ANALYSIS = "cost_analysis"
    BUDGET_PLANNING = "budget_planning"
    OPTIMIZATION_REVIEW = "optimization_review"
    COMPLIANCE_AUDIT = "compliance_audit"
    GENERAL = "general"

class ParticipantRole(PyEnum):
    """Participant roles in collaborative sessions"""
    OWNER = "owner"
    MODERATOR = "moderator"
    ANALYST = "analyst"
    VIEWER = "viewer"
    APPROVER = "approver"

class ParticipantStatus(PyEnum):
    """Participant status in sessions"""
    ACTIVE = "active"
    IDLE = "idle"
    DISCONNECTED = "disconnected"
    AWAY = "away"

class SessionStatus(PyEnum):
    """Session lifecycle status"""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ARCHIVED = "archived"

class PermissionLevel(PyEnum):
    """Permission levels for session participants"""
    READ_ONLY = "read_only"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"

class CollaborativeSession(Base):
    """Collaborative session management"""
    __tablename__ = "collaborative_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_name = Column(String(200), nullable=False)
    session_type = Column(Enum(SessionType), nullable=False, default=SessionType.GENERAL)
    description = Column(Text)
    status = Column(Enum(SessionStatus), nullable=False, default=SessionStatus.CREATED)
    
    # Session configuration
    max_participants = Column(Integer, nullable=False, default=50)
    auto_save_interval = Column(Integer, nullable=False, default=30)  # seconds
    session_timeout = Column(Integer, nullable=False, default=3600)  # seconds
    allow_anonymous = Column(Boolean, nullable=False, default=False)
    require_approval = Column(Boolean, nullable=False, default=False)
    
    # Session metadata
    session_config = Column(JSONB, default={})
    shared_state = Column(JSONB, default={})
    permissions_template = Column(String(100), default="default")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    
    # Owner
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    created_by_user = relationship("User")
    participants = relationship("SessionParticipant", back_populates="session", cascade="all, delete-orphan")
    state_updates = relationship("SessionStateUpdate", back_populates="session", cascade="all, delete-orphan")
    chat_messages = relationship("SessionChatMessage", back_populates="session", cascade="all, delete-orphan")
    annotations = relationship("SessionAnnotation", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_sessions_status_type', 'status', 'session_type'),
        Index('ix_sessions_created_by', 'created_by'),
        Index('ix_sessions_last_activity', 'last_activity'),
    )
    
    def __repr__(self):
        return f"<CollaborativeSession(name='{self.session_name}', status='{self.status.value}')>"

class SessionParticipant(Base):
    """Session participant management"""
    __tablename__ = "session_participants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaborative_sessions.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Participant details
    display_name = Column(String(100), nullable=False)
    role = Column(Enum(ParticipantRole), nullable=False, default=ParticipantRole.VIEWER)
    permission_level = Column(Enum(PermissionLevel), nullable=False, default=PermissionLevel.READ_ONLY)
    status = Column(Enum(ParticipantStatus), nullable=False, default=ParticipantStatus.ACTIVE)
    
    # Session participation
    joined_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    left_at = Column(DateTime(timezone=True))
    last_active = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now())
    
    # Presence and cursor tracking
    cursor_position = Column(JSONB, default={})
    current_view = Column(String(200))
    is_typing = Column(Boolean, nullable=False, default=False)
    
    # Invitation details
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    invitation_sent_at = Column(DateTime(timezone=True))
    invitation_accepted_at = Column(DateTime(timezone=True))
    
    # Relationships
    session = relationship("CollaborativeSession", back_populates="participants")
    user = relationship("User", foreign_keys=[user_id])
    invited_by_user = relationship("User", foreign_keys=[invited_by])
    
    # Indexes
    __table_args__ = (
        Index('ix_participants_session_user', 'session_id', 'user_id'),
        Index('ix_participants_session_status', 'session_id', 'status'),
        Index('ix_participants_last_active', 'last_active'),
        UniqueConstraint('session_id', 'user_id', name='uq_session_participant'),
    )
    
    def __repr__(self):
        return f"<SessionParticipant(session_id='{self.session_id}', user_id='{self.user_id}', role='{self.role.value}')>"

class SessionStateUpdate(Base):
    """Session state synchronization tracking"""
    __tablename__ = "session_state_updates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaborative_sessions.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Operation details
    operation_id = Column(String(255), nullable=False, index=True)
    operation_type = Column(String(50), nullable=False)  # filter_change, view_change, annotation_add, etc.
    target_path = Column(String(500), nullable=False)  # Path to the changed element
    
    # State changes
    old_value = Column(JSONB)
    new_value = Column(JSONB)
    operation_data = Column(JSONB, default={})
    
    # Operational transformation
    version = Column(Integer, nullable=False, default=1)
    parent_version = Column(Integer)
    conflict_resolved = Column(Boolean, nullable=False, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    applied_at = Column(DateTime(timezone=True))
    
    # Relationships
    session = relationship("CollaborativeSession", back_populates="state_updates")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('ix_state_updates_session_version', 'session_id', 'version'),
        Index('ix_state_updates_operation', 'operation_id'),
        Index('ix_state_updates_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<SessionStateUpdate(session_id='{self.session_id}', operation='{self.operation_type}', version={self.version})>"

class SessionChatMessage(Base):
    """Session chat messages"""
    __tablename__ = "session_chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaborative_sessions.id"), nullable=False)
    sender_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Message content
    content = Column(Text, nullable=False)
    message_type = Column(String(50), nullable=False, default="text")  # text, system, notification
    thread_id = Column(UUID(as_uuid=True))  # For threaded conversations
    reply_to_id = Column(UUID(as_uuid=True), ForeignKey("session_chat_messages.id"))
    
    # Mentions and attachments
    mentions = Column(JSONB, default=[])  # List of mentioned user IDs
    attachments = Column(JSONB, default=[])  # List of attachment metadata
    
    # Message metadata
    edited = Column(Boolean, nullable=False, default=False)
    edited_at = Column(DateTime(timezone=True))
    deleted = Column(Boolean, nullable=False, default=False)
    deleted_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    session = relationship("CollaborativeSession", back_populates="chat_messages")
    sender = relationship("User")
    reply_to = relationship("SessionChatMessage", remote_side=[id])
    
    # Indexes
    __table_args__ = (
        Index('ix_chat_messages_session_created', 'session_id', 'created_at'),
        Index('ix_chat_messages_thread', 'thread_id'),
        Index('ix_chat_messages_sender', 'sender_id'),
    )
    
    def __repr__(self):
        return f"<SessionChatMessage(session_id='{self.session_id}', sender_id='{self.sender_id}')>"

class SessionAnnotation(Base):
    """Session annotations on dashboards and data"""
    __tablename__ = "session_annotations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaborative_sessions.id"), nullable=False)
    author_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Annotation details
    annotation_type = Column(String(50), nullable=False, default="comment")  # comment, highlight, arrow, etc.
    content = Column(Text, nullable=False)
    
    # Target element
    target_element_id = Column(String(255), nullable=False)
    target_element_type = Column(String(100), nullable=False)  # chart, table, metric, etc.
    
    # Position and styling
    position = Column(JSONB, nullable=False)  # x, y coordinates and dimensions
    styling = Column(JSONB, default={})  # Color, size, etc.
    
    # Threading support
    thread_id = Column(UUID(as_uuid=True))  # For threaded conversations
    parent_annotation_id = Column(UUID(as_uuid=True), ForeignKey("session_annotations.id"))  # For replies
    
    # Contextual links and mentions
    contextual_links = Column(JSONB, default=[])  # Links to dashboards, reports, etc.
    mentions = Column(JSONB, default=[])  # Mentioned user IDs
    
    # Annotation metadata
    resolved = Column(Boolean, nullable=False, default=False)
    resolved_at = Column(DateTime(timezone=True))
    resolved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Archival support
    archived = Column(Boolean, nullable=False, default=False)
    archived_at = Column(DateTime(timezone=True))
    archived_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    archive_reason = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    session = relationship("CollaborativeSession", back_populates="annotations")
    author = relationship("User", foreign_keys=[author_id])
    resolved_by_user = relationship("User", foreign_keys=[resolved_by])
    archived_by_user = relationship("User", foreign_keys=[archived_by])
    parent_annotation = relationship("SessionAnnotation", remote_side=[id])
    
    # Indexes
    __table_args__ = (
        Index('ix_annotations_session_element', 'session_id', 'target_element_id'),
        Index('ix_annotations_author', 'author_id'),
        Index('ix_annotations_resolved', 'resolved'),
        Index('ix_annotations_thread', 'thread_id'),
        Index('ix_annotations_archived', 'archived'),
    )
    
    def __repr__(self):
        return f"<SessionAnnotation(session_id='{self.session_id}', type='{self.annotation_type}')>"

class SessionInvitation(Base):
    """Session invitation management"""
    __tablename__ = "session_invitations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaborative_sessions.id"), nullable=False)
    invited_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Invitation details
    invitation_token = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), nullable=False)
    role = Column(Enum(ParticipantRole), nullable=False, default=ParticipantRole.VIEWER)
    permission_level = Column(Enum(PermissionLevel), nullable=False, default=PermissionLevel.READ_ONLY)
    
    # Invitation status
    status = Column(String(50), nullable=False, default="pending")  # pending, accepted, declined, expired
    message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    responded_at = Column(DateTime(timezone=True))
    
    # Relationships
    session = relationship("CollaborativeSession")
    invited_user = relationship("User", foreign_keys=[invited_user_id])
    invited_by_user = relationship("User", foreign_keys=[invited_by])
    
    # Indexes
    __table_args__ = (
        Index('ix_invitations_session_status', 'session_id', 'status'),
        Index('ix_invitations_token', 'invitation_token'),
        Index('ix_invitations_expires_at', 'expires_at'),
        UniqueConstraint('session_id', 'invited_user_id', name='uq_session_invitation'),
    )
    
    def __repr__(self):
        return f"<SessionInvitation(session_id='{self.session_id}', email='{self.email}', status='{self.status}')>"