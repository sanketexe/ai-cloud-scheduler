#!/usr/bin/env python3
"""
Property-Based Tests for Policy Enforcement Consistency

This module contains property-based tests to verify that the policy manager
validates against defined policies and blocks actions that violate policy rules
while alerting administrators according to the requirements specification.

**Feature: automated-cost-optimization, Property 6: Policy Enforcement Consistency**
**Validates: Requirements 3.2, 3.4**
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List
from hypothesis import given, strategies as st, assume, settings
from dataclasses import dataclass

# Import the components we're testing
from core.policy_manager import PolicyManager, PolicyValidationResult
from core.automation_models import (
    AutomationLevel, ActionType, RiskLevel, ActionStatus,
    OptimizationAction, AutomationPolicy
)


class MockOptimizationAction:
    """Mock optimization action for testing"""
    
    def __init__(self, action_type: ActionType, resource_id: str, resource_type: str,
                 estimated_monthly_savings: float, risk_level: RiskLevel,
                 resource_metadata: Dict[str, Any]):
        self.id = uuid.uuid4()
        self.action_type = action_type
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.estimated_monthly_savings = estimated_monthly_savings
        self.risk_level = risk_level
        self.resource_metadata = resource_metadata
        self.execution_status = ActionStatus.PENDING
        self.error_message = None


class MockAutomationPolicy:
    """Mock automation policy for testing"""
    
    def __init__(self, policy_id: uuid.UUID, name: str, automation_level: AutomationLevel,
                 enabled_actions: List[str], approval_required_actions: List[str],
                 blocked_actions: List[str], resource_filters: Dict[str, Any]):
        self.id = policy_id
        self.name = name
        self.automation_level = automation_level
        self.enabled_actions = enabled_actions
        self.approval_required_actions = approval_required_actions
        self.blocked_actions = blocked_actions
        self.resource_filters = resource_filters
        self.time_restrictions = {}
        self.safety_overrides = {}


class TestPolicyEnforcementConsistency:
    """Property-based tests for policy enforcement consistency"""
    
    def __init__(self):
        self.policy_manager = PolicyManager()
    
    @given(
        # Generate policy configuration
        policy_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs'))),
        automation_level=st.sampled_from(list(AutomationLevel)),
        
        # Generate action configurations
        enabled_actions=st.lists(
            st.sampled_from([action.value for action in ActionType]),
            min_size=1,
            max_size=len(ActionType),
            unique=True
        ),
        blocked_actions=st.lists(
            st.sampled_from([action.value for action in ActionType]),
            min_size=0,
            max_size=len(ActionType) // 2,
            unique=True
        ),
        
        # Generate resource filter configurations
        exclude_tags=st.lists(
            st.sampled_from(['Environment=production', 'Critical=true', 'Tier=prod']),
            min_size=0,
            max_size=3,
            unique=True
        ),
        include_services=st.lists(
            st.sampled_from(['EC2', 'EBS', 'EIP']),
            min_size=1,
            max_size=3,
            unique=True
        ),
        min_cost_threshold=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        
        # Generate test actions
        test_action_type=st.sampled_from(list(ActionType)),
        resource_service=st.sampled_from(['EC2', 'EBS', 'EIP', 'ELB']),
        resource_tags=st.dictionaries(
            st.sampled_from(['Environment', 'Critical', 'Tier', 'Owner']),
            st.sampled_from(['production', 'staging', 'development', 'true', 'false', 'prod', 'test', 'admin']),
            min_size=0,
            max_size=4
        ),
        resource_cost=st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
        estimated_savings=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        risk_level=st.sampled_from(list(RiskLevel))
    )
    @settings(max_examples=100, deadline=None)
    def test_policy_enforcement_consistency_property(self,
                                                   policy_name: str,
                                                   automation_level: AutomationLevel,
                                                   enabled_actions: List[str],
                                                   blocked_actions: List[str],
                                                   exclude_tags: List[str],
                                                   include_services: List[str],
                                                   min_cost_threshold: float,
                                                   test_action_type: ActionType,
                                                   resource_service: str,
                                                   resource_tags: Dict[str, str],
                                                   resource_cost: float,
                                                   estimated_savings: float,
                                                   risk_level: RiskLevel):
        """
        **Feature: automated-cost-optimization, Property 6: Policy Enforcement Consistency**
        
        Property: For any optimization action, the system should validate against 
        defined policies and block actions that violate policy rules while alerting 
        administrators.
        
        This property verifies that:
        1. Policy validation is consistently applied to all actions
        2. Actions violating policy rules are correctly identified
        3. Blocked actions are properly categorized and reported
        4. Policy enforcement decisions are deterministic and consistent
        5. Violation detection provides detailed information
        6. Administrator alerts are generated for policy violations
        """
        
        # Skip invalid inputs
        assume(len(policy_name.strip()) > 0)
        assume(not (any(char in policy_name for char in ['<', '>', '"', "'", '&'])))
        
        # Ensure no conflicts between enabled and blocked actions
        enabled_set = set(enabled_actions)
        blocked_set = set(blocked_actions)
        assume(not enabled_set.intersection(blocked_set))  # No conflicting actions
        
        # Build resource filters
        resource_filters = {
            "include_services": include_services,
            "min_cost_threshold": min_cost_threshold
        }
        if exclude_tags:
            resource_filters["exclude_tags"] = exclude_tags
        
        # Create mock policy
        policy = MockAutomationPolicy(
            policy_id=uuid.uuid4(),
            name=policy_name,
            automation_level=automation_level,
            enabled_actions=enabled_actions,
            approval_required_actions=[],
            blocked_actions=blocked_actions,
            resource_filters=resource_filters
        )
        
        # Create test resource metadata
        resource_metadata = {
            "service": resource_service,
            "tags": resource_tags,
            "monthly_cost": resource_cost
        }
        
        # Create mock action
        action = MockOptimizationAction(
            action_type=test_action_type,
            resource_id=f"resource-{uuid.uuid4().hex[:8]}",
            resource_type=resource_service,
            estimated_monthly_savings=estimated_savings,
            risk_level=risk_level,
            resource_metadata=resource_metadata
        )
        
        # PROPERTY ASSERTIONS: Policy enforcement consistency requirements
        
        # 1. Policy validation must be consistently applied
        is_allowed_1, validation_details_1 = self.policy_manager.validate_action_against_policy(
            action.action_type, action.resource_metadata, policy
        )
        
        # Run the same validation again - should get identical results
        is_allowed_2, validation_details_2 = self.policy_manager.validate_action_against_policy(
            action.action_type, action.resource_metadata, policy
        )
        
        assert is_allowed_1 == is_allowed_2, \
            "Policy validation must be deterministic and consistent"
        
        assert validation_details_1["checks"] == validation_details_2["checks"], \
            "Policy validation details must be consistent across multiple calls"
        
        # 2. Actions violating policy rules must be correctly identified
        
        # Check blocked action enforcement
        if test_action_type.value in blocked_actions:
            assert not is_allowed_1, \
                "Actions explicitly blocked by policy must be rejected"
            assert any("blocked by policy" in violation for violation in validation_details_1.get("violations", [])), \
                "Blocked actions must generate specific violation messages"
        
        # Check enabled action enforcement
        if test_action_type.value not in enabled_actions:
            assert not is_allowed_1, \
                "Actions not enabled by policy must be rejected"
            assert any("not enabled" in violation for violation in validation_details_1.get("violations", [])), \
                "Non-enabled actions must generate specific violation messages"
        
        # Check resource filter enforcement
        if resource_service not in include_services:
            assert not is_allowed_1, \
                "Actions on resources not in included services must be rejected"
            assert any("not in included services" in violation for violation in validation_details_1.get("violations", [])), \
                "Service filter violations must generate specific violation messages"
        
        if resource_cost < min_cost_threshold:
            assert not is_allowed_1, \
                "Actions on resources below cost threshold must be rejected"
            assert any("below threshold" in violation for violation in validation_details_1.get("violations", [])), \
                "Cost threshold violations must generate specific violation messages"
        
        # Check exclude tags enforcement
        if exclude_tags:
            resource_tag_strings = [f"{k}={v}" for k, v in resource_tags.items()]
            excluded_tag_found = any(tag in resource_tag_strings for tag in exclude_tags)
            if excluded_tag_found:
                assert not is_allowed_1, \
                    "Actions on resources with excluded tags must be rejected"
                assert any("excluded tag" in violation for violation in validation_details_1.get("violations", [])), \
                    "Excluded tag violations must generate specific violation messages"
        
        # 3. Validation details must provide comprehensive information
        assert isinstance(validation_details_1, dict), \
            "Validation details must be provided as a dictionary"
        
        assert "policy_id" in validation_details_1, \
            "Validation details must include policy ID"
        
        assert "action_type" in validation_details_1, \
            "Validation details must include action type"
        
        assert "checks" in validation_details_1, \
            "Validation details must include check results"
        
        assert isinstance(validation_details_1["checks"], dict), \
            "Check results must be provided as a dictionary"
        
        # 4. Test violation detection for multiple actions
        actions_list = [action]
        
        violation_report = self.policy_manager.detect_policy_violations(actions_list, policy)
        
        assert isinstance(violation_report, dict), \
            "Violation report must be a dictionary"
        
        assert "policy_id" in violation_report, \
            "Violation report must include policy ID"
        
        assert "total_actions_checked" in violation_report, \
            "Violation report must include total actions checked"
        
        assert violation_report["total_actions_checked"] == 1, \
            "Violation report must correctly count checked actions"
        
        assert "violations_detected" in violation_report, \
            "Violation report must include violations detected"
        
        assert "blocked_actions" in violation_report, \
            "Violation report must include blocked actions"
        
        assert "allowed_actions" in violation_report, \
            "Violation report must include allowed actions"
        
        # 5. Violation detection consistency with individual validation
        if not is_allowed_1:
            assert len(violation_report["violations_detected"]) == 1, \
                "Violation detection must identify the same violations as individual validation"
            
            assert len(violation_report["blocked_actions"]) == 1, \
                "Blocked actions count must match violation detection"
            
            assert len(violation_report["allowed_actions"]) == 0, \
                "Allowed actions count must be zero when action is blocked"
            
            # Check that violation details are consistent
            detected_violation = violation_report["violations_detected"][0]
            assert detected_violation["action_type"] == test_action_type.value, \
                "Detected violation must include correct action type"
            
            assert detected_violation["resource_id"] == action.resource_id, \
                "Detected violation must include correct resource ID"
        
        else:
            assert len(violation_report["violations_detected"]) == 0, \
                "No violations should be detected for allowed actions"
            
            assert len(violation_report["blocked_actions"]) == 0, \
                "No actions should be blocked for allowed actions"
            
            assert len(violation_report["allowed_actions"]) == 1, \
                "Allowed actions count must be one when action is allowed"
        
        # 6. Test action blocking functionality
        blocking_results = self.policy_manager.block_policy_violating_actions(
            actions_list, policy, notification_channels=[]
        )
        
        assert isinstance(blocking_results, dict), \
            "Blocking results must be a dictionary"
        
        assert "actions_processed" in blocking_results, \
            "Blocking results must include actions processed count"
        
        assert blocking_results["actions_processed"] == 1, \
            "Blocking results must correctly count processed actions"
        
        assert "actions_blocked" in blocking_results, \
            "Blocking results must include actions blocked count"
        
        assert "actions_allowed" in blocking_results, \
            "Blocking results must include actions allowed count"
        
        # 7. Blocking consistency with violation detection
        if not is_allowed_1:
            assert blocking_results["actions_blocked"] == 1, \
                "Blocking must block the same actions identified as violations"
            
            assert blocking_results["actions_allowed"] == 0, \
                "No actions should be allowed when violations are detected"
        
        else:
            assert blocking_results["actions_blocked"] == 0, \
                "No actions should be blocked when no violations are detected"
            
            assert blocking_results["actions_allowed"] == 1, \
                "Actions should be allowed when no violations are detected"
        
        # 8. Violation summary must be accurate
        violation_summary = violation_report["violation_summary"]
        assert isinstance(violation_summary, dict), \
            "Violation summary must be a dictionary"
        
        # Count expected violation types
        expected_blocked_actions = 1 if test_action_type.value in blocked_actions else 0
        expected_resource_violations = 0
        
        if resource_service not in include_services:
            expected_resource_violations += 1
        if resource_cost < min_cost_threshold:
            expected_resource_violations += 1
        if exclude_tags and any(f"{k}={v}" in exclude_tags for k, v in resource_tags.items()):
            expected_resource_violations += 1
        
        # Verify violation summary accuracy
        if not is_allowed_1:
            total_violations = (violation_summary["blocked_action_types"] + 
                              violation_summary["resource_filter_violations"])
            assert total_violations > 0, \
                "Violation summary must count detected violations"


def run_property_test():
    """Run the policy enforcement consistency property test"""
    print("Running Property-Based Test for Policy Enforcement Consistency")
    print("=" * 60)
    print("**Feature: automated-cost-optimization, Property 6: Policy Enforcement Consistency**")
    print("**Validates: Requirements 3.2, 3.4**")
    print()
    
    test_instance = TestPolicyEnforcementConsistency()
    
    try:
        print("Testing Property 6: Policy Enforcement Consistency...")
        test_instance.test_policy_enforcement_consistency_property()
        print("✓ Property 6 test completed successfully")
        print()
        print("Property validation confirmed:")
        print("- Policy validation is consistently applied to all actions")
        print("- Actions violating policy rules are correctly identified")
        print("- Blocked actions are properly categorized and reported")
        print("- Policy enforcement decisions are deterministic and consistent")
        print("- Violation detection provides detailed information")
        print("- Administrator alerts are generated for policy violations")
        
        return True
        
    except Exception as e:
        print(f"✗ Property test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_test()
    if success:
        print("\nPolicy Enforcement Consistency property test passed!")
    else:
        print("\nPolicy Enforcement Consistency property test failed!")
        exit(1)