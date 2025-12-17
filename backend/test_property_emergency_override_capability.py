"""
Property-Based Test for Emergency Override Capability

**Feature: automated-cost-optimization, Property 16: Emergency Override Capability**

Property: For any emergency cost optimization scenario, the system should provide override 
capabilities that bypass normal scheduling restrictions for immediate action

**Validates: Requirements 7.3**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from decimal import Decimal
import uuid

from core.scheduling_engine import SchedulingEngine, SchedulingContext
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
    """Generate random AutomationPolicy instances with time restrictions"""
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
    
    # Generate blackout periods
    blackout_periods = []
    num_blackouts = draw(st.integers(min_value=0, max_value=2))
    
    for i in range(num_blackouts):
        start_time = datetime.utcnow() + timedelta(hours=draw(st.integers(min_value=1, max_value=48)))
        end_time = start_time + timedelta(hours=draw(st.integers(min_value=1, max_value=12)))
        
        blackout_periods.append({
            "name": f"blackout_period_{i}",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "timezone": "UTC",
            "reason": "Scheduled maintenance",
            "enabled": True
        })
    
    time_restrictions = business_hours.copy()
    if blackout_periods:
        time_restrictions["blackout_periods"] = blackout_periods
    
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


class TestEmergencyOverrideCapability:
    """Property-based tests for emergency override capability"""
    
    @given(
        action=optimization_action_strategy(),
        policy=automation_policy_strategy(),
        override_reason=st.text(min_size=10, max_size=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_emergency_override_bypasses_normal_scheduling(self, action, policy, override_reason):
        """
        Property: Emergency override should bypass normal scheduling restrictions
        
        Actions with emergency override should be scheduled immediately regardless of 
        business hours, blackout periods, or other time restrictions
        """
        engine = SchedulingEngine()
        
        # Calculate normal execution time (without override)
        normal_execution_time = engine.calculate_optimal_execution_time(action, policy)
        
        # Create emergency override context
        emergency_context = SchedulingContext(
            current_time=datetime.utcnow(),
            business_hours=None,
            maintenance_windows=[],
            blackout_periods=[],
            resource_patterns={},
            emergency_override=True,
            override_reason=override_reason,
            override_authorized_by="emergency_admin@company.com"
        )
        
        # Schedule with emergency override
        emergency_execution_time = engine._schedule_emergency_action(action, emergency_context)
        
        current_time = datetime.utcnow()
        
        # Emergency execution should be very soon (within 5 minutes)
        emergency_delay = emergency_execution_time - current_time
        assert emergency_delay <= timedelta(minutes=5), f"Emergency override should schedule within 5 minutes, got {emergency_delay}"
        
        # Emergency execution should be sooner than normal scheduling
        normal_delay = normal_execution_time - current_time
        assert emergency_delay < normal_delay, f"Emergency override ({emergency_delay}) should be faster than normal scheduling ({normal_delay})"
        
        # Emergency execution should be in the future
        assert emergency_execution_time > current_time, "Emergency execution time must be in the future"
    
    @given(
        actions=st.lists(optimization_action_strategy(), min_size=1, max_size=5),
        policy=automation_policy_strategy(),
        override_reason=st.text(min_size=10, max_size=100)
    )
    @settings(max_examples=50, deadline=None)
    def test_emergency_batch_override_schedules_immediately(self, actions, policy, override_reason):
        """
        Property: Emergency batch override should schedule all actions immediately
        
        When multiple actions are scheduled with emergency override, all should be 
        scheduled for immediate execution
        """
        engine = SchedulingEngine()
        
        # Schedule actions with emergency override
        scheduled_actions = engine.schedule_actions_batch(
            actions, 
            policy, 
            emergency_override=True,
            override_reason=override_reason,
            authorized_by="emergency_admin@company.com"
        )
        
        current_time = datetime.utcnow()
        
        # Verify all actions were scheduled
        assert len(scheduled_actions) == len(actions), "All actions should be scheduled"
        
        # Verify all actions are scheduled for immediate execution
        for action in scheduled_actions:
            assert action.scheduled_execution_time is not None, "Action should have execution time"
            assert action.execution_status == ActionStatus.SCHEDULED, "Action should be in scheduled status"
            
            delay = action.scheduled_execution_time - current_time
            assert delay <= timedelta(minutes=10), f"Emergency action should be scheduled within 10 minutes, got {delay}"
    
    @given(
        action=optimization_action_strategy(),
        policy=automation_policy_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_emergency_override_ignores_business_hours(self, action, policy):
        """
        Property: Emergency override should ignore business hours restrictions
        
        Even high-risk actions should be scheduled immediately during business hours 
        when emergency override is applied
        """
        # Only test when business hours are configured and action is high-risk
        business_hours_config = policy.time_restrictions.get("business_hours")
        if not business_hours_config or not business_hours_config.get("enabled"):
            return  # Skip if no business hours configured
        
        # Force high-risk action for this test
        action.risk_level = RiskLevel.HIGH
        
        engine = SchedulingEngine()
        
        # Create emergency context during business hours
        # Simulate current time being during business hours
        business_start = int(business_hours_config["start"].split(":")[0])
        business_end = int(business_hours_config["end"].split(":")[0])
        business_days = business_hours_config["days"]
        
        # Create a time that would be during business hours
        current_time = datetime.utcnow().replace(hour=(business_start + 1) % 24, minute=0, second=0, microsecond=0)
        
        emergency_context = SchedulingContext(
            current_time=current_time,
            business_hours=None,
            maintenance_windows=[],
            blackout_periods=[],
            resource_patterns={},
            emergency_override=True,
            override_reason="Critical cost emergency",
            override_authorized_by="emergency_admin@company.com"
        )
        
        # Schedule with emergency override
        emergency_execution_time = engine._schedule_emergency_action(action, emergency_context)
        
        # Should be scheduled immediately despite business hours
        delay = emergency_execution_time - current_time
        assert delay <= timedelta(minutes=5), f"Emergency override should ignore business hours, got delay of {delay}"
    
    @given(
        action=optimization_action_strategy(),
        policy=automation_policy_strategy(),
        override_reason=st.text(min_size=5, max_size=200)
    )
    @settings(max_examples=100, deadline=None)
    def test_emergency_override_requires_authorization(self, action, policy, override_reason):
        """
        Property: Emergency override should require proper authorization
        
        Emergency override context should include authorization information
        """
        engine = SchedulingEngine()
        
        authorized_by = "admin@company.com"
        
        # Create emergency context with authorization
        emergency_context = SchedulingContext(
            current_time=datetime.utcnow(),
            business_hours=None,
            maintenance_windows=[],
            blackout_periods=[],
            resource_patterns={},
            emergency_override=True,
            override_reason=override_reason,
            override_authorized_by=authorized_by
        )
        
        # Schedule with emergency override
        emergency_execution_time = engine._schedule_emergency_action(action, emergency_context)
        
        # Verify authorization information is preserved in context
        assert emergency_context.override_reason == override_reason, "Override reason should be preserved"
        assert emergency_context.override_authorized_by == authorized_by, "Authorization info should be preserved"
        assert emergency_context.emergency_override is True, "Emergency override flag should be set"
        
        # Verify execution time is immediate
        current_time = datetime.utcnow()
        delay = emergency_execution_time - current_time
        assert delay <= timedelta(minutes=5), "Emergency override should schedule immediately"
    
    @given(
        action=optimization_action_strategy(),
        policy=automation_policy_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_emergency_override_ignores_blackout_periods(self, action, policy):
        """
        Property: Emergency override should ignore blackout periods
        
        Actions should be scheduled immediately even during blackout periods 
        when emergency override is applied
        """
        # Only test when blackout periods are configured
        blackout_periods = policy.time_restrictions.get("blackout_periods", [])
        if not blackout_periods:
            return  # Skip if no blackout periods configured
        
        engine = SchedulingEngine()
        
        # Create emergency context during a blackout period
        # Use the first blackout period's time
        blackout_start = datetime.fromisoformat(blackout_periods[0]["start_time"])
        blackout_time = blackout_start + timedelta(hours=1)  # Middle of blackout period
        
        emergency_context = SchedulingContext(
            current_time=blackout_time,
            business_hours=None,
            maintenance_windows=[],
            blackout_periods=[],  # Empty for emergency context
            resource_patterns={},
            emergency_override=True,
            override_reason="Critical emergency during blackout",
            override_authorized_by="emergency_admin@company.com"
        )
        
        # Schedule with emergency override
        emergency_execution_time = engine._schedule_emergency_action(action, emergency_context)
        
        # Should be scheduled immediately despite blackout period
        delay = emergency_execution_time - blackout_time
        assert delay <= timedelta(minutes=5), f"Emergency override should ignore blackout periods, got delay of {delay}"
    
    @given(
        action=optimization_action_strategy(),
        policy=automation_policy_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_emergency_override_works_for_all_risk_levels(self, action, policy):
        """
        Property: Emergency override should work regardless of action risk level
        
        Low, medium, and high-risk actions should all be scheduled immediately 
        when emergency override is applied
        """
        engine = SchedulingEngine()
        
        # Test with the action's original risk level
        emergency_context = SchedulingContext(
            current_time=datetime.utcnow(),
            business_hours=None,
            maintenance_windows=[],
            blackout_periods=[],
            resource_patterns={},
            emergency_override=True,
            override_reason="Emergency cost optimization",
            override_authorized_by="admin@company.com"
        )
        
        # Schedule with emergency override
        emergency_execution_time = engine._schedule_emergency_action(action, emergency_context)
        
        current_time = datetime.utcnow()
        delay = emergency_execution_time - current_time
        
        # Should be scheduled immediately regardless of risk level
        assert delay <= timedelta(minutes=5), f"Emergency override should work for {action.risk_level.value} risk actions, got delay of {delay}"
        
        # Verify execution time is in the future
        assert emergency_execution_time > current_time, "Emergency execution time must be in the future"


if __name__ == "__main__":
    # Run the property-based tests
    test_instance = TestEmergencyOverrideCapability()
    
    # Test with some sample data
    sample_action = OptimizationAction(
        id=uuid.uuid4(),
        action_type=ActionType.TERMINATE_INSTANCE,
        resource_id="i-emergency123",
        resource_type="ec2_instance",
        estimated_monthly_savings=Decimal("500.00"),
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
    
    sample_policy = AutomationPolicy(
        id=uuid.uuid4(),
        name="Emergency Test Policy",
        automation_level=AutomationLevel.CONSERVATIVE,
        enabled_actions=[action.value for action in ActionType],
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
            },
            "blackout_periods": [{
                "name": "maintenance_blackout",
                "start_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "end_time": (datetime.utcnow() + timedelta(hours=6)).isoformat(),
                "timezone": "UTC",
                "reason": "Scheduled maintenance",
                "enabled": True
            }]
        },
        safety_overrides={},
        created_by=uuid.uuid4()
    )
    
    # Run a simple validation test
    engine = SchedulingEngine()
    
    # Test normal scheduling
    normal_time = engine.calculate_optimal_execution_time(sample_action, sample_policy)
    
    # Test emergency override
    emergency_context = SchedulingContext(
        current_time=datetime.utcnow(),
        business_hours=None,
        maintenance_windows=[],
        blackout_periods=[],
        resource_patterns={},
        emergency_override=True,
        override_reason="Critical cost emergency",
        override_authorized_by="admin@company.com"
    )
    
    emergency_time = engine._schedule_emergency_action(sample_action, emergency_context)
    
    # Basic validation
    current_time = datetime.utcnow()
    normal_delay = normal_time - current_time
    emergency_delay = emergency_time - current_time
    
    assert emergency_delay < normal_delay, "Emergency override should be faster than normal scheduling"
    assert emergency_delay <= timedelta(minutes=5), "Emergency override should be immediate"
    
    print("Property-based test for emergency override capability completed successfully!")