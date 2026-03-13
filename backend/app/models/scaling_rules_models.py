"""
SQLAlchemy models for Auto-Scaling Rules Engine

Stores user-defined scaling rules and their execution history.
Users pre-configure rules that define:
  - Which metric to watch (CPU, storage, connections, etc.)
  - Threshold conditions (e.g., > 85%)
  - What scaling action to take (increase storage, resize instance, etc.)
  - Safety limits (max storage, cooldown period, etc.)
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, DateTime, Boolean, Text, Numeric,
    ForeignKey, Index, Integer, Float
)
from sqlalchemy.types import JSON as JSONB, Uuid as UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.database.database import Base


class ServiceType(PyEnum):
    """AWS service types that can be auto-scaled"""
    EC2 = "ec2"
    EBS = "ebs"
    RDS = "rds"
    ASG = "asg"


class ThresholdOperator(PyEnum):
    """Comparison operators for threshold evaluation"""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"


class ScalingDirection(PyEnum):
    """Direction of scaling action"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


class RuleExecutionStatus(PyEnum):
    """Status of a rule execution"""
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    COOLDOWN_SKIPPED = "cooldown_skipped"
    LIMIT_REACHED = "limit_reached"
    DRY_RUN = "dry_run"


class ScalingRule(Base):
    """
    User-defined auto-scaling rule.

    Example rule: "If EBS volume usage > 85% for 3 consecutive checks (5 min apart),
    increase storage by 4 GB, up to a maximum of 100 GB. Wait 30 min between scaling."
    """
    __tablename__ = "scaling_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    # What to watch
    service_type = Column(String(20), nullable=False)  # ec2, ebs, rds, asg
    resource_filter = Column(JSONB, nullable=False, default=dict)
    # e.g. {"resource_ids": ["vol-xxx"], "tags": {"Environment": "production"}}

    # Metric & threshold
    metric_namespace = Column(String(100), nullable=False)  # e.g. "AWS/EBS", "AWS/EC2"
    metric_name = Column(String(100), nullable=False)       # e.g. "VolumeQueueLength", "CPUUtilization"
    metric_dimension_name = Column(String(100), nullable=False)  # e.g. "VolumeId", "InstanceId"
    metric_statistic = Column(String(20), nullable=False, default="Average")  # Average, Maximum, Sum
    threshold_operator = Column(String(10), nullable=False)  # gt, lt, gte, lte
    threshold_value = Column(Float, nullable=False)
    evaluation_periods = Column(Integer, nullable=False, default=3)
    evaluation_interval_seconds = Column(Integer, nullable=False, default=300)  # 5 min

    # What action to take
    scaling_direction = Column(String(20), nullable=False, default="scale_up")
    scaling_action = Column(JSONB, nullable=False)
    # EBS example:  {"action": "increase_storage", "amount_gb": 4}
    # EC2 example:  {"action": "resize_instance", "target_instance_type": "m5.xlarge"}
    # RDS example:  {"action": "resize_db_instance", "target_db_instance_class": "db.m5.large"}

    # Safety limits
    max_scaling_limit = Column(JSONB, nullable=False, default=dict)
    # EBS: {"max_size_gb": 100}  EC2: {"max_instance_type": "m5.4xlarge"}  RDS: {"max_instance_class": "db.m5.4xlarge"}
    cooldown_seconds = Column(Integer, nullable=False, default=1800)  # 30 min

    # State
    is_enabled = Column(Boolean, nullable=False, default=True)
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    trigger_count = Column(Integer, nullable=False, default=0)
    total_cost_impact = Column(Numeric(precision=12, scale=4), nullable=False, default=0)

    # Metadata
    created_by = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    executions = relationship("ScalingRuleExecution", back_populates="rule", order_by="desc(ScalingRuleExecution.triggered_at)")

    __table_args__ = (
        Index('ix_scaling_rules_service', 'service_type'),
        Index('ix_scaling_rules_enabled', 'is_enabled'),
    )

    def __repr__(self):
        return f"<ScalingRule(name='{self.name}', service='{self.service_type}', metric='{self.metric_name}')>"


class ScalingRuleExecution(Base):
    """
    Immutable log of every scaling rule execution.
    Records what happened, what was changed, and the result.
    """
    __tablename__ = "scaling_rule_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id = Column(UUID(as_uuid=True), ForeignKey("scaling_rules.id"), nullable=False, index=True)
    resource_id = Column(String(255), nullable=False)
    triggered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # What triggered it
    metric_value_at_trigger = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=False)

    # What was done
    action_taken = Column(JSONB, nullable=False)
    # e.g. {"action": "increase_storage", "amount_gb": 4, "api_call": "ec2.modify_volume"}
    previous_state = Column(JSONB, nullable=False, default=dict)
    # e.g. {"size_gb": 20, "volume_type": "gp3"}
    new_state = Column(JSONB, nullable=False, default=dict)
    # e.g. {"size_gb": 24, "volume_type": "gp3"}

    # Result
    status = Column(String(30), nullable=False)  # success, failed, rolled_back, etc.
    error_message = Column(Text, nullable=True)
    cost_impact = Column(Numeric(precision=12, scale=4), nullable=True)
    execution_duration_ms = Column(Integer, nullable=True)

    # Relationships
    rule = relationship("ScalingRule", back_populates="executions")

    __table_args__ = (
        Index('ix_scaling_rule_executions_rule_status', 'rule_id', 'status'),
        Index('ix_scaling_rule_executions_triggered', 'triggered_at'),
        Index('ix_scaling_rule_executions_resource', 'resource_id'),
    )

    def __repr__(self):
        return f"<ScalingRuleExecution(rule_id='{self.rule_id}', resource='{self.resource_id}', status='{self.status}')>"
