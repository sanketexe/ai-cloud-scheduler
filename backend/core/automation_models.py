"""
SQLAlchemy models for Automated Cost Optimization
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Dict, Any, List, Optional

from sqlalchemy import (
    Column, String, DateTime, Boolean, Text, Numeric, 
    ForeignKey, Index, JSON, Enum, Integer
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from .database import Base


class AutomationLevel(PyEnum):
    """Automation policy levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class ActionType(PyEnum):
    """Types of optimization actions"""
    STOP_INSTANCE = "stop_instance"
    TERMINATE_INSTANCE = "terminate_instance"
    RESIZE_INSTANCE = "resize_instance"
    DELETE_VOLUME = "delete_volume"
    UPGRADE_STORAGE = "upgrade_storage"
    RELEASE_ELASTIC_IP = "release_elastic_ip"
    DELETE_LOAD_BALANCER = "delete_load_balancer"
    CLEANUP_SECURITY_GROUP = "cleanup_security_group"


class ActionStatus(PyEnum):
    """Status of optimization actions"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class RiskLevel(PyEnum):
    """Risk levels for actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalStatus(PyEnum):
    """Approval workflow status"""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AutomationPolicy(Base):
    """Automation policy configuration"""
    __tablename__ = "automation_policies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    automation_level = Column(Enum(AutomationLevel), nullable=False)
    enabled_actions = Column(JSONB, nullable=False, default=list)  # List of ActionType values
    approval_required_actions = Column(JSONB, nullable=False, default=list)
    blocked_actions = Column(JSONB, nullable=False, default=list)
    resource_filters = Column(JSONB, nullable=False, default=dict)
    time_restrictions = Column(JSONB, nullable=False, default=dict)
    safety_overrides = Column(JSONB, nullable=False, default=dict)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    optimization_actions = relationship("OptimizationAction", back_populates="policy")
    
    def __repr__(self):
        return f"<AutomationPolicy(name='{self.name}', level='{self.automation_level.value}')>"


class OptimizationAction(Base):
    """Individual optimization actions"""
    __tablename__ = "optimization_actions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    action_type = Column(Enum(ActionType), nullable=False)
    resource_id = Column(String(255), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False)
    estimated_monthly_savings = Column(Numeric(precision=12, scale=4), nullable=False)
    actual_savings = Column(Numeric(precision=12, scale=4))
    risk_level = Column(Enum(RiskLevel), nullable=False)
    requires_approval = Column(Boolean, nullable=False, default=False)
    approval_status = Column(Enum(ApprovalStatus), nullable=False, default=ApprovalStatus.NOT_REQUIRED)
    scheduled_execution_time = Column(DateTime(timezone=True))
    execution_started_at = Column(DateTime(timezone=True))
    execution_completed_at = Column(DateTime(timezone=True))
    safety_checks_passed = Column(Boolean, nullable=False, default=False)
    rollback_plan = Column(JSONB, nullable=False, default=dict)
    execution_status = Column(Enum(ActionStatus), nullable=False, default=ActionStatus.PENDING)
    error_message = Column(Text)
    resource_metadata = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    policy_id = Column(UUID(as_uuid=True), ForeignKey("automation_policies.id"), nullable=False)
    
    # Relationships
    policy = relationship("AutomationPolicy", back_populates="optimization_actions")
    audit_logs = relationship("AutomationAuditLog", back_populates="action")
    
    # Indexes
    __table_args__ = (
        Index('ix_optimization_actions_resource', 'resource_id', 'resource_type'),
        Index('ix_optimization_actions_status', 'execution_status'),
        Index('ix_optimization_actions_scheduled', 'scheduled_execution_time'),
    )
    
    @validates('estimated_monthly_savings')
    def validate_savings(self, key, savings):
        """Validate savings amount is non-negative"""
        if savings < 0:
            raise ValueError("Estimated savings cannot be negative")
        return savings
    
    def __repr__(self):
        return f"<OptimizationAction(type='{self.action_type.value}', resource='{self.resource_id}')>"


class AutomationAuditLog(Base):
    """Immutable audit log for automation actions"""
    __tablename__ = "automation_audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    action_id = Column(UUID(as_uuid=True), ForeignKey("optimization_actions.id"), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)  # 'created', 'scheduled', 'executed', 'failed', 'rolled_back'
    event_data = Column(JSONB, nullable=False, default=dict)
    user_context = Column(JSONB, nullable=False, default=dict)
    system_context = Column(JSONB, nullable=False, default=dict)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    correlation_id = Column(String(255), index=True)
    
    # Relationships
    action = relationship("OptimizationAction", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('ix_automation_audit_logs_action_event', 'action_id', 'event_type'),
        Index('ix_automation_audit_logs_timestamp', 'timestamp'),
        Index('ix_automation_audit_logs_correlation', 'correlation_id'),
    )
    
    def __repr__(self):
        return f"<AutomationAuditLog(action_id='{self.action_id}', event='{self.event_type}')>"


class ActionApproval(Base):
    """Action approval workflow"""
    __tablename__ = "action_approvals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    action_id = Column(UUID(as_uuid=True), ForeignKey("optimization_actions.id"), nullable=False, index=True)
    requested_by = Column(String(255), nullable=False)  # System or user identifier
    requested_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    approved_at = Column(DateTime(timezone=True))
    rejection_reason = Column(Text)
    approval_status = Column(Enum(ApprovalStatus), nullable=False, default=ApprovalStatus.PENDING)
    expires_at = Column(DateTime(timezone=True))
    
    # Indexes
    __table_args__ = (
        Index('ix_action_approvals_action', 'action_id'),
        Index('ix_action_approvals_status', 'approval_status'),
    )
    
    def __repr__(self):
        return f"<ActionApproval(action_id='{self.action_id}', status='{self.approval_status.value}')>"


class SafetyCheckResult(Base):
    """Results of safety checks for actions"""
    __tablename__ = "safety_check_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    action_id = Column(UUID(as_uuid=True), ForeignKey("optimization_actions.id"), nullable=False, index=True)
    check_name = Column(String(100), nullable=False)
    check_result = Column(Boolean, nullable=False)
    check_details = Column(JSONB, nullable=False, default=dict)
    checked_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('ix_safety_check_results_action', 'action_id'),
        Index('ix_safety_check_results_name_result', 'check_name', 'check_result'),
    )
    
    def __repr__(self):
        return f"<SafetyCheckResult(action_id='{self.action_id}', check='{self.check_name}', result={self.check_result})>"