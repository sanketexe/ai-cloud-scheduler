"""
SQLAlchemy models for FinOps Platform
"""

import uuid
from datetime import datetime, date
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Dict, Any, List, Optional

from sqlalchemy import (
    Column, String, DateTime, Boolean, Text, Date, Numeric, 
    ForeignKey, Index, JSON, Enum, Integer, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from .database import Base

class UserRole(PyEnum):
    """User roles for role-based access control"""
    ADMIN = "admin"
    FINANCE_MANAGER = "finance_manager"
    ANALYST = "analyst"
    VIEWER = "viewer"

class ProviderType(PyEnum):
    """Supported cloud provider types"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    OTHER = "other"

class BudgetType(PyEnum):
    """Budget types"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    PROJECT = "project"

class AlertType(PyEnum):
    """Budget alert types"""
    THRESHOLD = "threshold"
    FORECAST = "forecast"
    ANOMALY = "anomaly"

class RecommendationType(PyEnum):
    """Optimization recommendation types"""
    RIGHTSIZING = "rightsizing"
    RESERVED_INSTANCE = "reserved_instance"
    UNUSED_RESOURCE = "unused_resource"
    UNDERUTILIZED = "underutilized"

class RecommendationStatus(PyEnum):
    """Recommendation status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"

class RiskLevel(PyEnum):
    """Risk levels for recommendations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class BaseModel(Base):
    """Base model with common fields"""
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class User(BaseModel):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.VIEWER)
    is_active = Column(Boolean, nullable=False, default=True)
    last_login = Column(DateTime(timezone=True))
    password_changed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    cloud_providers = relationship("CloudProvider", back_populates="created_by_user")
    budgets = relationship("Budget", back_populates="created_by_user")
    audit_logs = relationship("AuditLog", back_populates="user")
    acknowledged_alerts = relationship("BudgetAlert", back_populates="acknowledged_by_user")
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    def __repr__(self):
        return f"<User(email='{self.email}', role='{self.role.value}')>"

class CloudProvider(BaseModel):
    """Cloud provider configuration and credentials"""
    __tablename__ = "cloud_providers"
    
    name = Column(String(100), nullable=False)
    provider_type = Column(Enum(ProviderType), nullable=False)
    credentials_encrypted = Column(Text, nullable=False)  # JSON encrypted credentials
    is_active = Column(Boolean, nullable=False, default=True)
    last_sync = Column(DateTime(timezone=True))
    sync_frequency_hours = Column(Integer, default=24)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    created_by_user = relationship("User", back_populates="cloud_providers")
    cost_data = relationship("CostData", back_populates="provider")
    optimization_recommendations = relationship("OptimizationRecommendation", back_populates="provider")
    
    # Indexes
    __table_args__ = (
        Index('ix_cloud_providers_type_active', 'provider_type', 'is_active'),
    )
    
    def __repr__(self):
        return f"<CloudProvider(name='{self.name}', type='{self.provider_type.value}')>"

class CostData(BaseModel):
    """Cost data from cloud providers"""
    __tablename__ = "cost_data"
    
    provider_id = Column(UUID(as_uuid=True), ForeignKey("cloud_providers.id"), nullable=False, index=True)
    resource_id = Column(String(255), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)
    service_name = Column(String(100), nullable=False, index=True)
    cost_amount = Column(Numeric(precision=12, scale=4), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    cost_date = Column(Date, nullable=False, index=True)
    usage_quantity = Column(Numeric(precision=12, scale=4))
    usage_unit = Column(String(50))
    tags = Column(JSONB, default={})
    resource_metadata = Column(JSONB, default={})
    
    # Relationships
    provider = relationship("CloudProvider", back_populates="cost_data")
    
    # Indexes for time-series queries
    __table_args__ = (
        Index('ix_cost_data_provider_date', 'provider_id', 'cost_date'),
        Index('ix_cost_data_resource_date', 'resource_id', 'cost_date'),
        Index('ix_cost_data_service_date', 'service_name', 'cost_date'),
        Index('ix_cost_data_tags_gin', 'tags', postgresql_using='gin'),
    )
    
    def __repr__(self):
        return f"<CostData(resource='{self.resource_id}', cost={self.cost_amount}, date='{self.cost_date}')>"

class Budget(BaseModel):
    """Budget management"""
    __tablename__ = "budgets"
    
    name = Column(String(200), nullable=False)
    amount = Column(Numeric(precision=12, scale=2), nullable=False)
    budget_type = Column(Enum(BudgetType), nullable=False)
    scope_filters = Column(JSONB, nullable=False)  # Filters for cost data
    alert_thresholds = Column(JSONB, nullable=False)  # List of threshold percentages
    notification_emails = Column(JSONB, default=[])  # List of email addresses
    start_date = Column(Date, nullable=False)
    end_date = Column(Date)
    is_active = Column(Boolean, nullable=False, default=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    created_by_user = relationship("User", back_populates="budgets")
    alerts = relationship("BudgetAlert", back_populates="budget")
    
    # Indexes
    __table_args__ = (
        Index('ix_budgets_active_dates', 'is_active', 'start_date', 'end_date'),
    )
    
    @validates('amount')
    def validate_amount(self, key, amount):
        """Validate budget amount is positive"""
        if amount <= 0:
            raise ValueError("Budget amount must be positive")
        return amount
    
    def __repr__(self):
        return f"<Budget(name='{self.name}', amount={self.amount}, type='{self.budget_type.value}')>"

class BudgetAlert(BaseModel):
    """Budget alerts and notifications"""
    __tablename__ = "budget_alerts"
    
    budget_id = Column(UUID(as_uuid=True), ForeignKey("budgets.id"), nullable=False, index=True)
    threshold_percentage = Column(Integer, nullable=False)
    current_spend = Column(Numeric(precision=12, scale=2), nullable=False)
    budget_amount = Column(Numeric(precision=12, scale=2), nullable=False)
    alert_type = Column(Enum(AlertType), nullable=False, default=AlertType.THRESHOLD)
    message = Column(Text)
    acknowledged = Column(Boolean, nullable=False, default=False)
    acknowledged_at = Column(DateTime(timezone=True))
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Relationships
    budget = relationship("Budget", back_populates="alerts")
    acknowledged_by_user = relationship("User", back_populates="acknowledged_alerts")
    
    # Indexes
    __table_args__ = (
        Index('ix_budget_alerts_budget_ack', 'budget_id', 'acknowledged'),
    )
    
    def __repr__(self):
        return f"<BudgetAlert(budget_id='{self.budget_id}', threshold={self.threshold_percentage}%)>"

class OptimizationRecommendation(BaseModel):
    """Cost optimization recommendations"""
    __tablename__ = "optimization_recommendations"
    
    provider_id = Column(UUID(as_uuid=True), ForeignKey("cloud_providers.id"), nullable=False, index=True)
    resource_id = Column(String(255), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False)
    recommendation_type = Column(Enum(RecommendationType), nullable=False)
    current_cost = Column(Numeric(precision=12, scale=4), nullable=False)
    optimized_cost = Column(Numeric(precision=12, scale=4), nullable=False)
    potential_savings = Column(Numeric(precision=12, scale=4), nullable=False)
    confidence_score = Column(Numeric(precision=3, scale=2), nullable=False)  # 0.00 to 1.00
    risk_level = Column(Enum(RiskLevel), nullable=False)
    recommendation_text = Column(Text, nullable=False)
    implementation_details = Column(JSONB, default={})
    status = Column(Enum(RecommendationStatus), nullable=False, default=RecommendationStatus.PENDING)
    valid_until = Column(DateTime(timezone=True))
    
    # Relationships
    provider = relationship("CloudProvider", back_populates="optimization_recommendations")
    
    # Indexes
    __table_args__ = (
        Index('ix_optimization_provider_status', 'provider_id', 'status'),
        Index('ix_optimization_type_savings', 'recommendation_type', 'potential_savings'),
    )
    
    @validates('confidence_score')
    def validate_confidence_score(self, key, score):
        """Validate confidence score is between 0 and 1"""
        if not 0 <= score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
        return score
    
    def __repr__(self):
        return f"<OptimizationRecommendation(resource='{self.resource_id}', savings={self.potential_savings})>"

class AuditLog(BaseModel):
    """Audit log for security and compliance"""
    __tablename__ = "audit_logs"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255))
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    session_id = Column(String(255))
    correlation_id = Column(String(255), index=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('ix_audit_logs_user_action', 'user_id', 'action'),
        Index('ix_audit_logs_resource', 'resource_type', 'resource_id'),
        Index('ix_audit_logs_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<AuditLog(user_id='{self.user_id}', action='{self.action}', resource='{self.resource_type}')>"

class SystemConfiguration(BaseModel):
    """System configuration and settings"""
    __tablename__ = "system_configurations"
    
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(JSONB, nullable=False)
    description = Column(Text)
    is_encrypted = Column(Boolean, nullable=False, default=False)
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    def __repr__(self):
        return f"<SystemConfiguration(key='{self.key}')>"