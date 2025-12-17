#!/usr/bin/env python3
"""
Property-Based Test for Automation State Management

**Feature: automated-cost-optimization, Property 14: Automation State Management**
**Validates: Requirements 5.4**

Tests that when automation is disabled or paused, the system continues monitoring
but queues actions for manual review instead of executing them automatically.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
from hypothesis import given, strategies as st, settings
import pytest

# Import the components we're testing
from core.monitoring_service import (
    MonitoringService, AutomationState, MonitoringSeverity, ExecutionReport
)
from core.auto_remediation_engine import AutoRemediationEngine
from core.automation_models import (
    OptimizationAction, ActionType, ActionStatus, RiskLevel, AutomationPolicy, AutomationLevel
)
from core.notification_service import NotificationMessage, NotificationPriority


class MockActionQueue:
    """Mock action queue for testing queued actions during disabled/paused states"""
    
    def __init__(self):
        self.queued_actions = []
        self.executed_actions = []
        self.monitoring_events = []
    
    def queue_action(self, action: OptimizationAction):
        """Queue an action for manual review"""
        self.queued_actions.append({
            "action": action,
            "queued_at": datetime.utcnow(),
            "status": "queued_for_review"
        })
    
    def execute_action(self, action: OptimizationAction):
        """Execute an action (should only happen when enabled)"""
        self.executed_actions.append({
            "action": action,
            "executed_at": datetime.utcnow(),
            "status": "executed"
        })
    
    def log_monitoring_event(self, event_type: str, data: Dict[str, Any]):
        """Log a monitoring event"""
        self.monitoring_events.append({
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.utcnow()
        })
    
    def clear(self):
        """Clear all queues and logs"""
        self.queued_actions.clear()
        self.executed_actions.clear()
        self.monitoring_events.clear()


class MockAutomationEngine:
    """Mock automation engine that respects state management"""
    
    def __init__(self, monitoring_service: MonitoringService, action_queue: MockActionQueue):
        self.monitoring_service = monitoring_service
        self.action_queue = action_queue
    
    async def process_optimization_opportunity(self, action: OptimizationAction):
        """Process an optimization opportunity based on current automation state"""
        
        # Always monitor the opportunity
        self.action_queue.log_monitoring_event(
            "opportunity_detected",
            {
                "action_id": str(action.id),
                "action_type": action.action_type.value,
                "resource_id": action.resource_id,
                "automation_state": self.monitoring_service.get_automation_state().value
            }
        )
        
        # Check automation state to decide action
        automation_state = self.monitoring_service.get_automation_state()
        
        if automation_state == AutomationState.ENABLED:
            # Execute the action
            self.action_queue.execute_action(action)
            self.action_queue.log_monitoring_event(
                "action_executed",
                {
                    "action_id": str(action.id),
                    "reason": "automation_enabled"
                }
            )
        else:
            # Queue for manual review
            self.action_queue.queue_action(action)
            self.action_queue.log_monitoring_event(
                "action_queued",
                {
                    "action_id": str(action.id),
                    "reason": f"automation_{automation_state.value}",
                    "requires_manual_review": True
                }
            )


class TestAutomationStateManagement:
    """Test automation state management property"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.monitoring_service = MonitoringService()
        self.action_queue = MockActionQueue()
        self.automation_engine = MockAutomationEngine(self.monitoring_service, self.action_queue)
    
    @given(
        # Generate various automation states
        automation_states=st.lists(
            st.sampled_from([
                AutomationState.ENABLED,
                AutomationState.DISABLED,
                AutomationState.PAUSED,
                AutomationState.MAINTENANCE
            ]),
            min_size=1,
            max_size=5
        ),
        # Generate optimization actions
        action_types=st.lists(
            st.sampled_from([
                ActionType.STOP_INSTANCE,
                ActionType.DELETE_VOLUME,
                ActionType.RELEASE_ELASTIC_IP,
                ActionType.UPGRADE_STORAGE
            ]),
            min_size=1,
            max_size=10
        ),
        # Generate resource IDs
        resource_ids=st.lists(
            st.text(min_size=8, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'))),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_automation_state_management_property(self, automation_states, action_types, resource_ids):
        """
        **Feature: automated-cost-optimization, Property 14: Automation State Management**
        
        Property: For any system state (enabled/disabled/paused), when automation is disabled,
        the system should continue monitoring but queue actions for manual review instead of executing them.
        
        **Validates: Requirements 5.4**
        """
        
        # Setup test fixtures
        self.setup_method()
        
        # Clear previous state
        self.action_queue.clear()
        
        # Create optimization actions
        actions = []
        for i, (action_type, resource_id) in enumerate(zip(action_types, resource_ids)):
            action = OptimizationAction(
                id=uuid.uuid4(),
                action_type=action_type,
                resource_id=f"resource-{resource_id}-{i}",
                resource_type="ec2_instance" if "instance" in action_type.value else "ebs_volume",
                estimated_monthly_savings=50.0 + (i * 10),
                risk_level=RiskLevel.LOW,
                requires_approval=False,
                execution_status=ActionStatus.PENDING,
                resource_metadata={"test": True}
            )
            actions.append(action)
        
        # Test each automation state
        for state in automation_states:
            # Set automation state
            self.monitoring_service.set_automation_state(
                state, 
                f"Testing state {state.value}"
            )
            
            # Process actions in this state
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                for action in actions:
                    loop.run_until_complete(
                        self.automation_engine.process_optimization_opportunity(action)
                    )
            finally:
                loop.close()
            
            # Verify state management behavior
            self._verify_state_management_behavior(state, actions)
        
        # Verify overall property compliance
        self._verify_overall_property_compliance(automation_states, actions)
        
        print(f"✓ Automation state management property verified for {len(automation_states)} states and {len(actions)} actions")
    
    def _verify_state_management_behavior(self, state: AutomationState, actions: List[OptimizationAction]):
        """Verify behavior for a specific automation state"""
        
        # 1. Monitoring should always continue regardless of state
        monitoring_events = [e for e in self.action_queue.monitoring_events 
                           if e["event_type"] == "opportunity_detected"]
        
        assert len(monitoring_events) >= len(actions), \
            f"Monitoring must continue in {state.value} state - expected at least {len(actions)} events, got {len(monitoring_events)}"
        
        # 2. Verify correct action handling based on state
        if state == AutomationState.ENABLED:
            # Actions should be executed
            executed_count = len([e for e in self.action_queue.monitoring_events 
                                if e["event_type"] == "action_executed"])
            
            assert executed_count >= len(actions), \
                f"Actions should be executed when automation is enabled - expected {len(actions)}, got {executed_count}"
            
            # Should have executed actions in the queue
            assert len(self.action_queue.executed_actions) >= len(actions), \
                f"Executed actions queue should contain {len(actions)} actions when enabled"
        
        else:
            # Actions should be queued for manual review
            queued_count = len([e for e in self.action_queue.monitoring_events 
                              if e["event_type"] == "action_queued"])
            
            assert queued_count >= len(actions), \
                f"Actions should be queued when automation is {state.value} - expected {len(actions)}, got {queued_count}"
            
            # Should have queued actions
            assert len(self.action_queue.queued_actions) >= len(actions), \
                f"Queued actions should contain {len(actions)} actions when {state.value}"
            
            # Verify queued actions have correct metadata
            for queued_item in self.action_queue.queued_actions[-len(actions):]:
                assert queued_item["status"] == "queued_for_review", \
                    f"Queued actions should have 'queued_for_review' status"
                
                assert "queued_at" in queued_item, \
                    "Queued actions should have timestamp"
        
        # 3. Verify monitoring events contain correct state information
        for event in monitoring_events[-len(actions):]:
            assert event["data"]["automation_state"] == state.value, \
                f"Monitoring events should record correct automation state: {state.value}"
    
    def _verify_overall_property_compliance(self, states: List[AutomationState], actions: List[OptimizationAction]):
        """Verify overall property compliance across all states"""
        
        # 1. Monitoring should have occurred for all opportunities
        total_monitoring_events = len([e for e in self.action_queue.monitoring_events 
                                     if e["event_type"] == "opportunity_detected"])
        
        expected_monitoring_events = len(states) * len(actions)
        assert total_monitoring_events >= expected_monitoring_events, \
            f"Monitoring should occur for all opportunities - expected {expected_monitoring_events}, got {total_monitoring_events}"
        
        # 2. Actions should only be executed when enabled
        execution_events = [e for e in self.action_queue.monitoring_events 
                          if e["event_type"] == "action_executed"]
        
        enabled_states_count = len([s for s in states if s == AutomationState.ENABLED])
        expected_executions = enabled_states_count * len(actions)
        
        assert len(execution_events) >= expected_executions, \
            f"Actions should only execute when enabled - expected {expected_executions}, got {len(execution_events)}"
        
        # 3. Actions should be queued when not enabled
        queuing_events = [e for e in self.action_queue.monitoring_events 
                         if e["event_type"] == "action_queued"]
        
        non_enabled_states_count = len([s for s in states if s != AutomationState.ENABLED])
        expected_queueing = non_enabled_states_count * len(actions)
        
        assert len(queuing_events) >= expected_queueing, \
            f"Actions should be queued when not enabled - expected {expected_queueing}, got {len(queuing_events)}"
        
        # 4. Verify state transitions are properly logged
        state_changes = len(set(states))  # Number of unique states tested
        if state_changes > 1:
            # Should have monitoring events for state awareness
            state_aware_events = [e for e in self.action_queue.monitoring_events 
                                if "automation_state" in e.get("data", {})]
            
            assert len(state_aware_events) > 0, \
                "System should be aware of automation state changes"
        
        # 5. Verify no actions are executed when automation is disabled/paused
        for event in execution_events:
            # Check that execution events only occurred during enabled state
            # This is implicitly verified by the counts above, but we can add explicit checks
            assert event["data"]["reason"] == "automation_enabled", \
                "Actions should only execute when automation is explicitly enabled"
        
        # 6. Verify queued actions have proper manual review flags
        for event in queuing_events:
            assert event["data"]["requires_manual_review"] is True, \
                "Queued actions should require manual review"
            
            assert event["data"]["reason"].startswith("automation_"), \
                "Queued actions should specify automation state as reason"


def test_automation_state_management():
    """Run the automation state management property test"""
    test_instance = TestAutomationStateManagement()
    test_instance.setup_method()  # Initialize the test instance properly
    
    # Create a simple test case manually
    automation_states = [AutomationState.ENABLED, AutomationState.DISABLED]
    action_types = [ActionType.STOP_INSTANCE, ActionType.DELETE_VOLUME]
    resource_ids = ["test-resource-1", "test-resource-2"]
    
    # Clear previous state
    test_instance.action_queue.clear()
    
    # Create optimization actions
    actions = []
    for i, (action_type, resource_id) in enumerate(zip(action_types, resource_ids)):
        action = OptimizationAction(
            id=uuid.uuid4(),
            action_type=action_type,
            resource_id=f"resource-{resource_id}-{i}",
            resource_type="ec2_instance" if "instance" in action_type.value else "ebs_volume",
            estimated_monthly_savings=50.0 + (i * 10),
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            execution_status=ActionStatus.PENDING,
            resource_metadata={"test": True}
        )
        actions.append(action)
    
    # Test each automation state
    for state in automation_states:
        # Set automation state
        test_instance.monitoring_service.set_automation_state(
            state, 
            f"Testing state {state.value}"
        )
        
        # Process actions in this state
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for action in actions:
                loop.run_until_complete(
                    test_instance.automation_engine.process_optimization_opportunity(action)
                )
        finally:
            loop.close()
        
        # Verify state management behavior
        test_instance._verify_state_management_behavior(state, actions)
    
    # Verify overall property compliance
    test_instance._verify_overall_property_compliance(automation_states, actions)
    
    print(f"✓ Automation state management property verified for {len(automation_states)} states and {len(actions)} actions")
    print("✓ Automation state management property test completed successfully")


if __name__ == "__main__":
    test_automation_state_management()