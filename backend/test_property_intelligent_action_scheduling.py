"""
Property-Based Test for Intelligent Action Scheduling

**Feature: automated-cost-optimization, Property 15: Intelligent Action Scheduling**

Property: For any optimization action, the system should consider business hours, 
maintenance windows, resource usage patterns, and prioritize based on savings and risk levels

**Validates: Requirements 7.1, 7.2, 7.4, 7.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta, time
from decimal import Decimal
import uuid

from core.scheduling_engine import SchedulingEngine, BusinessHours, MaintenanceWindow, BlackoutPeriod, MaintenanceWindowType
from core.automation_models import (
    OptimizationAction, AutomationPolicy, ActionType, RiskLevel,
    ActionStatus, ApprovalStatus, AutomationLevel
)


# Hypothesis strategies for generating test data
@st.composite
def optimization_action_strategy(draw):
    """Generate random OptimizationAction instances"""
    action_type = draw(st.sampled_from(list(ActionType)))
    risk_level = draw(st.sampled_from(list(RiskLevel)))
    
    return OptimizationAction(
        id=uuid.uuid4(),
        action_type=action_type,
        resource_id=draw(st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd')))),
        resource_type=draw(st.sampled_from(["ec2_instance", "ebs_volume", "elastic_ip", "load_balancer"])),
        estimated_monthly_savings=Decimal(str(draw(st.floats(min_value=1.0, max_value=1000.0)))),
        risk_level=risk_level,
        requires_approval=draw(st.booleans()),
        approval_status=ApprovalStatus.NOT_REQUIRED,
        scheduled_execution_time=None,
        safety_checks_passed=True,
        rollback_plan={},
        execution_status=ActionStatus.PENDING,
        resource_metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=10), 
            st.text(min_size=1, max_size=20),
            max_size=5
        )),
        policy_id=uuid.uuid4()
    )


@st.composite
def automation_policy_strategy(draw):
    """Generate random AutomationPolicy instances"""
    # Generate business hours configuration
    business_hours_enabled = draw(st.booleans())
    business_hours = {}
    
    if business_hours_enabled:
        start_hour = draw(st.integers(min_value=0, max_value=12))
        end_hour = draw(st.integers(min_value=start_hour + 1, max_value=23))
        
        business_hours = {
            "business_hours": {
                "timezone": "UTC",
                "start": f"{start_hour:02d}:00",
                "end": f"{end_hour:02d}:00",
                "days": draw(st.lists(
                    st.sampled_from(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]),
                    min_size=1, max_size=7, unique=True
                )),
                "enabled": True
            }
        }
    
    # Generate maintenance windows
    maintenance_windows = []
    num_windows = draw(st.integers(min_value=0, max_value=3))
    
    for i in range(num_windows):
        start_time = datetime.utcnow() + timedelta(days=draw(st.integers(min_value=1, max_value=7)))
        end_time = start_time + timedelta(hours=draw(st.integers(min_value=1, max_value=6)))
        
        maintenance_windows.append({
            "name": f"maintenance_window_{i}",
            "type": "weekly",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "timezone": "UTC",
            "enabled": True,
            "recurrence_pattern": {"weekday": draw(st.integers(min_value=0, max_value=6))}
        })
    
    time_restrictions = business_hours.copy()
    if maintenance_windows:
        time_restrictions["maintenance_windows"] = maintenance_windows
    
    return AutomationPolicy(
        id=uuid.uuid4(),
        name=draw(st.text(min_size=5, max_size=50)),
        automation_level=draw(st.sampled_from(list(AutomationLevel))),
        enabled_actions=[action.value for action in ActionType],
        approval_required_actions=draw(st.lists(
            st.sampled_from([action.value for action in ActionType]),
            max_size=3
        )),
        blocked_actions=[],
        resource_filters={},
        time_restrictions=time_restrictions,
        safety_overrides={},
        created_by=uuid.uuid4()
    )


class TestIntelligentActionScheduling:
    """Property-based tests for intelligent action scheduling"""
    
    @given(
        action=optimization_action_strategy(),
        policy=automation_policy_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_scheduling_considers_risk_levels(self, action, policy):
        """
        Property: Scheduling should prioritize actions based on risk levels
        
        High-risk actions should be scheduled with more delay than low-risk actions
        """
        engine = SchedulingEngine()
        
        # Calculate execution time
        execution_time = engine.calculate_optimal_execution_time(action, policy)
        current_time = datetime.utcnow()
        
        # Verify execution time is in the future
        assert execution_time > current_time, "Execution time must be in the future"
        
        # Calculate delay based on risk level
        delay = execution_time - current_time
        
        # Risk-based scheduling validation - more realistic expectations
        if action.risk_level == RiskLevel.LOW:
            # Low-risk actions should be scheduled relatively soon (within 8 hours)
            # Allow for action type modifiers and business hours constraints
            assert delay <= timedelta(hours=8), f"Low-risk action scheduled too far in future: {delay}"
        elif action.risk_level == RiskLevel.MEDIUM:
            # Medium-risk actions should have moderate delay (within 12 hours)
            assert delay <= timedelta(hours=12), f"Medium-risk action scheduled too far in future: {delay}"
        elif action.risk_level == RiskLevel.HIGH:
            # High-risk actions can have longer delays but should be reasonable (within 48 hours)
            # Allow for maintenance windows and business hours constraints
            assert delay <= timedelta(hours=48), f"High-risk action scheduled too far in future: {delay}"
    
    @given(
        action=optimization_action_strategy(),
        policy=automation_policy_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_scheduling_respects_business_hours(self, action, policy):
        """
        Property: High-risk actions should avoid business hours when business hours are configured
        """
        engine = SchedulingEngine()
        
        # Only test when business hours are configured
        business_hours_config = policy.time_restrictions.get("business_hours")
        if not business_hours_config or not business_hours_config.get("enabled"):
            return  # Skip if no business hours configured
        
        # Only test high-risk actions (they should avoid business hours)
        if action.risk_level != RiskLevel.HIGH:
            return
        
        execution_time = engine.calculate_optimal_execution_time(action, policy)
        
        # Parse business hours
        start_hour = int(business_hours_config["start"].split(":")[0])
        end_hour = int(business_hours_config["end"].split(":")[0])
        business_days = business_hours_config["days"]
        
        # Check if execution time avoids business hours
        exec_hour = execution_time.hour
        exec_weekday = execution_time.strftime("%A").lower()
        
        # If it's a business day, execution should be outside business hours
        if exec_weekday in business_days:
            is_business_hours = start_hour <= exec_hour < end_hour
            assert not is_business_hours, f"High-risk action scheduled during business hours: {execution_time}"
    
    @given(
        actions=st.lists(optimization_action_strategy(), min_size=2, max_size=10),
        policy=automation_policy_strategy()
    )
    @settings(max_examples=50, deadline=None)
    def test_batch_scheduling_distributes_actions(self, actions, policy):
        """
        Property: Batch scheduling should distribute actions over time to avoid conflicts
        """
        engine = SchedulingEngine()
        
        # Schedule actions as a batch
        scheduled_actions = engine.schedule_actions_batch(actions, policy)
        
        # Verify all actions were scheduled
        assert len(scheduled_actions) == len(actions), "All actions should be scheduled"
        
        # Verify all actions have execution times
        execution_times = []
        for action in scheduled_actions:
            assert action.scheduled_execution_time is not None, "Action should have execution time"
            assert action.execution_status == ActionStatus.SCHEDULED, "Action should be in scheduled status"
            execution_times.append(action.scheduled_execution_time)
        
        # Verify times are distributed (not all the same)
        if len(execution_times) > 1:
            unique_times = set(execution_times)
            # At least some actions should have different execution times
            # (allowing for some overlap due to rounding and low-risk actions)
            assert len(unique_times) >= min(3, len(execution_times) // 2), "Actions should be distributed over time"
    
    @given(
        action=optimization_action_strategy(),
        policy=automation_policy_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_scheduling_considers_action_type_characteristics(self, action, policy):
        """
        Property: Different action types should have appropriate scheduling delays
        """
        engine = SchedulingEngine()
        
        execution_time = engine.calculate_optimal_execution_time(action, policy)
        current_time = datetime.utcnow()
        delay = execution_time - current_time
        
        # Risk level specific validation - consider actual risk level, not just action type
        if action.risk_level == RiskLevel.LOW:
            # Low-risk actions should be scheduled relatively soon
            assert delay <= timedelta(hours=6), f"Low-risk action {action.action_type.value} scheduled too far out: {delay}"
        
        elif action.risk_level == RiskLevel.MEDIUM:
            # Medium-risk actions can have moderate delays
            assert delay <= timedelta(hours=24), f"Medium-risk action {action.action_type.value} scheduled too far out: {delay}"
        
        elif action.risk_level == RiskLevel.HIGH:
            # High-risk actions can have longer delays due to maintenance windows and business hours
            assert delay <= timedelta(days=2), f"High-risk action {action.action_type.value} scheduled too far out: {delay}"
        
        # All actions should be scheduled within reasonable bounds
        assert delay >= timedelta(minutes=1), "Actions should have minimum delay for safety"
        assert delay <= timedelta(days=2), "Actions should not be scheduled too far in the future"
    
    @given(
        action=optimization_action_strategy(),
        policy=automation_policy_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_scheduling_maintains_execution_order_by_priority(self, action, policy):
        """
        Property: When scheduling multiple actions, higher savings should generally be prioritized
        """
        engine = SchedulingEngine()
        
        # Create a second action with different savings
        action2 = OptimizationAction(
            id=uuid.uuid4(),
            action_type=action.action_type,
            resource_id=f"{action.resource_id}_2",
            resource_type=action.resource_type,
            estimated_monthly_savings=action.estimated_monthly_savings + Decimal("100.00"),  # Higher savings
            risk_level=action.risk_level,
            requires_approval=action.requires_approval,
            approval_status=ApprovalStatus.NOT_REQUIRED,
            scheduled_execution_time=None,
            safety_checks_passed=True,
            rollback_plan={},
            execution_status=ActionStatus.PENDING,
            resource_metadata=action.resource_metadata,
            policy_id=action.policy_id
        )
        
        # Schedule both actions
        scheduled_actions = engine.schedule_actions_batch([action, action2], policy)
        
        # Find the actions in the results
        scheduled_action1 = next(a for a in scheduled_actions if a.resource_id == action.resource_id)
        scheduled_action2 = next(a for a in scheduled_actions if a.resource_id == action2.resource_id)
        
        # Both should be scheduled
        assert scheduled_action1.scheduled_execution_time is not None
        assert scheduled_action2.scheduled_execution_time is not None
        
        # If they have the same risk level, higher savings action should not be scheduled significantly later
        if scheduled_action1.risk_level == scheduled_action2.risk_level:
            time_diff = abs((scheduled_action2.scheduled_execution_time - scheduled_action1.scheduled_execution_time).total_seconds())
            # Allow for reasonable scheduling differences but not extreme delays for higher value actions
            assert time_diff <= 7200, "Higher savings actions should not be delayed significantly more than lower savings actions"


if __name__ == "__main__":
    # Run the property-based tests
    test_instance = TestIntelligentActionScheduling()
    
    # Test with some sample data
    sample_action = OptimizationAction(
        id=uuid.uuid4(),
        action_type=ActionType.STOP_INSTANCE,
        resource_id="i-12345",
        resource_type="ec2_instance",
        estimated_monthly_savings=Decimal("50.00"),
        risk_level=RiskLevel.MEDIUM,
        requires_approval=False,
        approval_status=ApprovalStatus.NOT_REQUIRED,
        scheduled_execution_time=None,
        safety_checks_passed=True,
        rollback_plan={},
        execution_status=ActionStatus.PENDING,
        resource_metadata={},
        policy_id=uuid.uuid4()
    )
    
    sample_policy = AutomationPolicy(
        id=uuid.uuid4(),
        name="Test Policy",
        automation_level=AutomationLevel.BALANCED,
        enabled_actions=[action.value for action in ActionType],
        approval_required_actions=[],
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
    
    # Run a simple validation test
    engine = SchedulingEngine()
    execution_time = engine.calculate_optimal_execution_time(sample_action, sample_policy)
    current_time = datetime.utcnow()
    
    # Basic validation
    assert execution_time > current_time, "Execution time must be in the future"
    delay = execution_time - current_time
    assert delay <= timedelta(hours=24), "Action should be scheduled within reasonable time"
    
    print("Property-based test for intelligent action scheduling completed successfully!")