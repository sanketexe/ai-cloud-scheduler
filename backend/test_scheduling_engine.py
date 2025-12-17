"""
Test for SchedulingEngine functionality
"""

import pytest
from datetime import datetime, timedelta, time
from decimal import Decimal
import uuid

from core.scheduling_engine import (
    SchedulingEngine, BusinessHours, MaintenanceWindow, BlackoutPeriod,
    SchedulingContext, MaintenanceWindowType, SchedulePriority
)
from core.automation_models import (
    OptimizationAction, AutomationPolicy, ActionType, RiskLevel,
    ActionStatus, ApprovalStatus, AutomationLevel
)


def test_scheduling_engine_initialization():
    """Test that SchedulingEngine initializes correctly"""
    engine = SchedulingEngine()
    assert engine is not None
    assert engine.default_timezone == "UTC"


def test_calculate_optimal_execution_time_low_risk():
    """Test scheduling for low-risk actions"""
    engine = SchedulingEngine()
    
    # Create a low-risk action
    action = OptimizationAction(
        id=uuid.uuid4(),
        action_type=ActionType.RELEASE_ELASTIC_IP,
        resource_id="eip-12345",
        resource_type="elastic_ip",
        estimated_monthly_savings=Decimal("10.00"),
        risk_level=RiskLevel.LOW,
        requires_approval=False,
        approval_status=ApprovalStatus.NOT_REQUIRED,
        scheduled_execution_time=None,
        safety_checks_passed=True,
        rollback_plan={},
        execution_status=ActionStatus.PENDING,
        resource_metadata={},
        policy_id=uuid.uuid4()
    )
    
    # Create a simple policy
    policy = AutomationPolicy(
        id=uuid.uuid4(),
        name="Test Policy",
        automation_level=AutomationLevel.BALANCED,
        enabled_actions=["release_elastic_ip"],
        approval_required_actions=[],
        blocked_actions=[],
        resource_filters={},
        time_restrictions={},
        safety_overrides={},
        created_by=uuid.uuid4()
    )
    
    # Calculate execution time
    execution_time = engine.calculate_optimal_execution_time(action, policy)
    
    # Should be scheduled soon for low-risk actions
    now = datetime.utcnow()
    assert execution_time > now
    assert execution_time <= now + timedelta(hours=1)


def test_calculate_optimal_execution_time_high_risk():
    """Test scheduling for high-risk actions"""
    engine = SchedulingEngine()
    
    # Create a high-risk action
    action = OptimizationAction(
        id=uuid.uuid4(),
        action_type=ActionType.TERMINATE_INSTANCE,
        resource_id="i-12345",
        resource_type="ec2_instance",
        estimated_monthly_savings=Decimal("100.00"),
        risk_level=RiskLevel.HIGH,
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING,
        scheduled_execution_time=None,
        safety_checks_passed=True,
        rollback_plan={},
        execution_status=ActionStatus.PENDING,
        resource_metadata={},
        policy_id=uuid.uuid4()
    )
    
    # Create a policy with business hours
    policy = AutomationPolicy(
        id=uuid.uuid4(),
        name="Test Policy",
        automation_level=AutomationLevel.CONSERVATIVE,
        enabled_actions=["terminate_instance"],
        approval_required_actions=["terminate_instance"],
        blocked_actions=[],
        resource_filters={},
        time_restrictions={
            "business_hours": {
                "timezone": "UTC",
                "start": "09:00",
                "end": "17:00",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                "enabled": True
            }
        },
        safety_overrides={},
        created_by=uuid.uuid4()
    )
    
    # Calculate execution time
    execution_time = engine.calculate_optimal_execution_time(action, policy)
    
    # Should be scheduled with more delay for high-risk actions
    now = datetime.utcnow()
    assert execution_time > now + timedelta(hours=2)


def test_business_hours_constraint():
    """Test that business hours constraints are applied correctly"""
    engine = SchedulingEngine()
    
    # Create business hours configuration
    business_hours = BusinessHours(
        timezone="UTC",
        start_time=time(9, 0),
        end_time=time(17, 0),
        business_days=["monday", "tuesday", "wednesday", "thursday", "friday"],
        enabled=True
    )
    
    # Test during business hours (assuming it's a weekday at 12:00 UTC)
    test_time = datetime(2024, 1, 15, 12, 0)  # Monday at 12:00 UTC
    is_business_hours = engine._is_business_hours(test_time, business_hours)
    assert is_business_hours is True
    
    # Test outside business hours
    test_time = datetime(2024, 1, 15, 20, 0)  # Monday at 20:00 UTC
    is_business_hours = engine._is_business_hours(test_time, business_hours)
    assert is_business_hours is False
    
    # Test weekend
    test_time = datetime(2024, 1, 13, 12, 0)  # Saturday at 12:00 UTC
    is_business_hours = engine._is_business_hours(test_time, business_hours)
    assert is_business_hours is False


def test_emergency_override():
    """Test emergency override functionality"""
    engine = SchedulingEngine()
    
    # Create a high-risk action
    action = OptimizationAction(
        id=uuid.uuid4(),
        action_type=ActionType.TERMINATE_INSTANCE,
        resource_id="i-12345",
        resource_type="ec2_instance",
        estimated_monthly_savings=Decimal("100.00"),
        risk_level=RiskLevel.HIGH,
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING,
        scheduled_execution_time=None,
        safety_checks_passed=True,
        rollback_plan={},
        execution_status=ActionStatus.PENDING,
        resource_metadata={},
        policy_id=uuid.uuid4()
    )
    
    # Create scheduling context with emergency override
    context = SchedulingContext(
        current_time=datetime.utcnow(),
        business_hours=None,
        maintenance_windows=[],
        blackout_periods=[],
        resource_patterns={},
        emergency_override=True,
        override_reason="Critical cost emergency",
        override_authorized_by="admin@company.com"
    )
    
    # Schedule with emergency override
    execution_time = engine._schedule_emergency_action(action, context)
    
    # Should be scheduled very soon
    now = datetime.utcnow()
    assert execution_time > now
    assert execution_time <= now + timedelta(minutes=5)


def test_batch_scheduling():
    """Test batch scheduling functionality"""
    engine = SchedulingEngine()
    
    # Create multiple actions with different risk levels
    actions = []
    for i in range(3):
        action = OptimizationAction(
            id=uuid.uuid4(),
            action_type=ActionType.STOP_INSTANCE if i == 0 else ActionType.RELEASE_ELASTIC_IP,
            resource_id=f"resource-{i}",
            resource_type="ec2_instance" if i == 0 else "elastic_ip",
            estimated_monthly_savings=Decimal("10.00"),
            risk_level=RiskLevel.LOW if i < 2 else RiskLevel.HIGH,
            requires_approval=False,
            approval_status=ApprovalStatus.NOT_REQUIRED,
            scheduled_execution_time=None,
            safety_checks_passed=True,
            rollback_plan={},
            execution_status=ActionStatus.PENDING,
            resource_metadata={},
            policy_id=uuid.uuid4()
        )
        actions.append(action)
    
    # Create policy
    policy = AutomationPolicy(
        id=uuid.uuid4(),
        name="Test Policy",
        automation_level=AutomationLevel.BALANCED,
        enabled_actions=["stop_instance", "release_elastic_ip"],
        approval_required_actions=[],
        blocked_actions=[],
        resource_filters={},
        time_restrictions={},
        safety_overrides={},
        created_by=uuid.uuid4()
    )
    
    # Schedule batch
    scheduled_actions = engine.schedule_actions_batch(actions, policy)
    
    # Verify all actions were scheduled
    assert len(scheduled_actions) == 3
    
    # Verify execution times are staggered
    execution_times = [action.scheduled_execution_time for action in scheduled_actions]
    execution_times.sort()
    
    # Times should be different (staggered)
    for i in range(1, len(execution_times)):
        assert execution_times[i] > execution_times[i-1]


if __name__ == "__main__":
    # Run basic tests
    test_scheduling_engine_initialization()
    test_calculate_optimal_execution_time_low_risk()
    test_calculate_optimal_execution_time_high_risk()
    test_business_hours_constraint()
    test_emergency_override()
    test_batch_scheduling()
    
    print("All scheduling engine tests passed!")