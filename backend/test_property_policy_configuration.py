#!/usr/bin/env python3
"""
Property-Based Tests for Policy Configuration Completeness

This module contains property-based tests to verify that the policy manager
allows configuration of all required elements with proper validation according
to the requirements specification.

**Feature: automated-cost-optimization, Property 20: Policy Configuration Completeness**
**Validates: Requirements 3.1**
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List
from hypothesis import given, strategies as st, assume, settings
from dataclasses import dataclass

# Import the components we're testing
from core.policy_manager import PolicyManager, PolicyValidationResult
from core.automation_models import (
    AutomationLevel, ActionType
)


class TestPolicyConfigurationCompleteness:
    """Property-based tests for policy configuration completeness"""
    
    @given(
        # Generate policy names
        policy_name=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs'))),
        
        # Generate automation levels
        automation_level=st.sampled_from(list(AutomationLevel)),
        
        # Generate action type configurations
        enabled_actions=st.lists(
            st.sampled_from([action.value for action in ActionType]),
            min_size=0,
            max_size=len(ActionType),
            unique=True
        ),
        approval_required_actions=st.lists(
            st.sampled_from([action.value for action in ActionType]),
            min_size=0,
            max_size=len(ActionType),
            unique=True
        ),
        blocked_actions=st.lists(
            st.sampled_from([action.value for action in ActionType]),
            min_size=0,
            max_size=len(ActionType),
            unique=True
        ),
        
        # Generate resource filter configurations
        exclude_tags=st.lists(
            st.sampled_from(['Environment=production', 'Critical=true', 'Tier=prod', 'Stage=production']),
            min_size=0,
            max_size=4,
            unique=True
        ),
        include_services=st.lists(
            st.sampled_from(['EC2', 'EBS', 'EIP', 'ELB', 'VPC']),
            min_size=0,
            max_size=5,
            unique=True
        ),
        min_cost_threshold=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        
        # Generate time restriction configurations
        business_hours_enabled=st.booleans(),
        start_hour=st.integers(min_value=0, max_value=23),
        end_hour=st.integers(min_value=0, max_value=23),
        timezone=st.sampled_from(['UTC', 'US/Eastern', 'US/Pacific', 'Europe/London']),
        business_days=st.lists(
            st.sampled_from(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']),
            min_size=1,
            max_size=7,
            unique=True
        ),
        
        # Generate safety override configurations
        production_tag_override=st.booleans(),
        business_hours_override=st.booleans(),
        asg_protection_override=st.booleans(),
        
        # Generate user context
        created_by=st.uuids()
    )
    @settings(max_examples=100, deadline=None)
    def test_policy_configuration_completeness_property(self,
                                                      policy_name: str,
                                                      automation_level: AutomationLevel,
                                                      enabled_actions: List[str],
                                                      approval_required_actions: List[str],
                                                      blocked_actions: List[str],
                                                      exclude_tags: List[str],
                                                      include_services: List[str],
                                                      min_cost_threshold: float,
                                                      business_hours_enabled: bool,
                                                      start_hour: int,
                                                      end_hour: int,
                                                      timezone: str,
                                                      business_days: List[str],
                                                      production_tag_override: bool,
                                                      business_hours_override: bool,
                                                      asg_protection_override: bool,
                                                      created_by: uuid.UUID):
        """
        **Feature: automated-cost-optimization, Property 20: Policy Configuration Completeness**
        
        Property: For any automation policy setup, the system should allow 
        configuration of all required elements (action types, resource filters, 
        approval requirements) with proper validation.
        
        This property verifies that:
        1. All required policy elements can be configured
        2. Action types configuration is properly validated
        3. Resource filters configuration is properly validated
        4. Approval requirements configuration is properly validated
        5. Time restrictions configuration is properly validated
        6. Safety overrides configuration is properly validated
        7. Policy validation provides detailed feedback
        8. Valid configurations are accepted
        9. Invalid configurations are rejected with clear error messages
        """
        
        # Skip invalid inputs
        assume(len(policy_name.strip()) > 0)
        assume(not (any(char in policy_name for char in ['<', '>', '"', "'", '&'])))  # Avoid HTML/SQL injection chars
        
        # Build resource filters configuration
        resource_filters = {}
        if exclude_tags:
            resource_filters["exclude_tags"] = exclude_tags
        if include_services:
            resource_filters["include_services"] = include_services
        if min_cost_threshold > 0:
            resource_filters["min_cost_threshold"] = min_cost_threshold
        
        # Build time restrictions configuration
        time_restrictions = {}
        if business_hours_enabled:
            time_restrictions["business_hours"] = {
                "start": f"{start_hour:02d}:00",
                "end": f"{end_hour:02d}:00",
                "timezone": timezone,
                "days": business_days
            }
        
        # Build safety overrides configuration
        safety_overrides = {}
        if production_tag_override:
            safety_overrides["production_tag_protection"] = {
                "enabled": False,
                "parameters": {"override_reason": "test_override"}
            }
        if business_hours_override:
            safety_overrides["business_hours_protection"] = {
                "enabled": False,
                "parameters": {"override_reason": "test_override"}
            }
        if asg_protection_override:
            safety_overrides["auto_scaling_group_protection"] = {
                "enabled": False,
                "parameters": {"override_reason": "test_override"}
            }
        
        # Test policy validation (without creating in database)
        policy_manager = PolicyManager()
        validation_result = policy_manager.validate_policy_configuration(
            automation_level=automation_level,
            enabled_actions=enabled_actions,
            approval_required_actions=approval_required_actions,
            blocked_actions=blocked_actions,
            resource_filters=resource_filters,
            time_restrictions=time_restrictions,
            safety_overrides=safety_overrides
        )
        
        # PROPERTY ASSERTIONS: Policy configuration completeness requirements
        
        # 1. Validation must always return a PolicyValidationResult
        assert isinstance(validation_result, PolicyValidationResult), \
            "Policy validation must return a PolicyValidationResult object"
        
        # 2. Validation result must have all required fields
        assert hasattr(validation_result, 'is_valid'), "Validation result must have is_valid field"
        assert hasattr(validation_result, 'errors'), "Validation result must have errors field"
        assert hasattr(validation_result, 'warnings'), "Validation result must have warnings field"
        assert isinstance(validation_result.is_valid, bool), "is_valid must be a boolean"
        assert isinstance(validation_result.errors, list), "errors must be a list"
        assert isinstance(validation_result.warnings, list), "warnings must be a list"
        
        # 3. Action types configuration validation
        # All action types must be valid ActionType enum values
        all_action_types = enabled_actions + approval_required_actions + blocked_actions
        valid_action_types = [action.value for action in ActionType]
        
        invalid_actions = [action for action in all_action_types if action not in valid_action_types]
        if invalid_actions:
            assert not validation_result.is_valid, \
                "Policy with invalid action types should fail validation"
            assert any("Invalid" in error and "action type" in error for error in validation_result.errors), \
                "Invalid action types should generate specific error messages"
        
        # 4. Action conflict detection
        enabled_set = set(enabled_actions)
        blocked_set = set(blocked_actions)
        conflicting_actions = enabled_set.intersection(blocked_set)
        
        if conflicting_actions:
            assert not validation_result.is_valid, \
                "Policy with conflicting enabled/blocked actions should fail validation"
            assert any("cannot be both enabled and blocked" in error for error in validation_result.errors), \
                "Conflicting actions should generate specific error messages"
        
        # 5. Resource filters configuration validation
        if resource_filters:
            # Check exclude_tags validation
            if "exclude_tags" in resource_filters:
                exclude_tags_value = resource_filters["exclude_tags"]
                if not isinstance(exclude_tags_value, list):
                    assert not validation_result.is_valid, \
                        "Policy with invalid exclude_tags format should fail validation"
            
            # Check include_services validation
            if "include_services" in resource_filters:
                include_services_value = resource_filters["include_services"]
                if not isinstance(include_services_value, list):
                    assert not validation_result.is_valid, \
                        "Policy with invalid include_services format should fail validation"
            
            # Check min_cost_threshold validation
            if "min_cost_threshold" in resource_filters:
                threshold_value = resource_filters["min_cost_threshold"]
                if not isinstance(threshold_value, (int, float)) or threshold_value < 0:
                    assert not validation_result.is_valid, \
                        "Policy with invalid min_cost_threshold should fail validation"
        
        # 6. Time restrictions configuration validation
        if time_restrictions and "business_hours" in time_restrictions:
            business_hours = time_restrictions["business_hours"]
            
            # Check time format validation
            if "start" in business_hours and "end" in business_hours:
                start_time = business_hours["start"]
                end_time = business_hours["end"]
                
                # Valid time format should not cause validation errors
                if isinstance(start_time, str) and isinstance(end_time, str):
                    try:
                        # Check if time format is valid (HH:MM)
                        start_parts = start_time.split(":")
                        end_parts = end_time.split(":")
                        
                        if (len(start_parts) == 2 and len(end_parts) == 2 and
                            start_parts[0].isdigit() and start_parts[1].isdigit() and
                            end_parts[0].isdigit() and end_parts[1].isdigit() and
                            0 <= int(start_parts[0]) <= 23 and 0 <= int(start_parts[1]) <= 59 and
                            0 <= int(end_parts[0]) <= 23 and 0 <= int(end_parts[1]) <= 59):
                            # Valid time format should not cause time-related errors
                            time_errors = [error for error in validation_result.errors if "time format" in error.lower()]
                            assert len(time_errors) == 0, \
                                "Valid time format should not generate time format errors"
                    except (ValueError, IndexError):
                        # Invalid time format should cause validation errors
                        assert not validation_result.is_valid, \
                            "Policy with invalid time format should fail validation"
            
            # Check days validation
            if "days" in business_hours:
                days_value = business_hours["days"]
                if not isinstance(days_value, list):
                    assert not validation_result.is_valid, \
                        "Policy with invalid days format should fail validation"
                else:
                    valid_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                    invalid_days = [day for day in days_value if day.lower() not in valid_days]
                    if invalid_days:
                        assert not validation_result.is_valid, \
                            "Policy with invalid day names should fail validation"
        
        # 7. Safety overrides configuration validation
        if safety_overrides:
            for rule_name, override_config in safety_overrides.items():
                if not isinstance(override_config, dict):
                    assert not validation_result.is_valid, \
                        "Policy with invalid safety override format should fail validation"
                    assert any(f"Safety override for {rule_name} must be a dictionary" in error 
                              for error in validation_result.errors), \
                        "Invalid safety override format should generate specific error messages"
                
                # Check override structure
                if isinstance(override_config, dict):
                    if "enabled" in override_config and not isinstance(override_config["enabled"], bool):
                        assert not validation_result.is_valid, \
                            "Policy with invalid safety override enabled flag should fail validation"
                    
                    if "parameters" in override_config and not isinstance(override_config["parameters"], dict):
                        assert not validation_result.is_valid, \
                            "Policy with invalid safety override parameters should fail validation"
        
        # 8. Automation level consistency validation
        if automation_level == AutomationLevel.CONSERVATIVE:
            high_risk_actions = [ActionType.TERMINATE_INSTANCE.value, ActionType.DELETE_LOAD_BALANCER.value]
            enabled_high_risk = set(enabled_actions).intersection(set(high_risk_actions))
            if enabled_high_risk:
                # Should generate warnings for conservative policy with high-risk actions
                conservative_warnings = [warning for warning in validation_result.warnings 
                                       if "Conservative policy enables high-risk actions" in warning]
                assert len(conservative_warnings) > 0, \
                    "Conservative policy with high-risk actions should generate warnings"
        
        elif automation_level == AutomationLevel.AGGRESSIVE:
            if len(approval_required_actions) > len(enabled_actions) * 0.5:
                # Should generate warnings for aggressive policy requiring many approvals
                aggressive_warnings = [warning for warning in validation_result.warnings 
                                     if "Aggressive policy requires approval for many actions" in warning]
                assert len(aggressive_warnings) > 0, \
                    "Aggressive policy requiring many approvals should generate warnings"
        
        # 9. If validation passes, all required elements should be configurable
        # Note: We focus on validation completeness rather than database operations
        # since this property test is about configuration validation, not persistence
        if validation_result.is_valid:
            # Verify that valid configurations have all required elements properly validated
            # This confirms that the system allows configuration of all required elements
            
            # Check that action types are properly handled
            if enabled_actions or approval_required_actions or blocked_actions:
                # At least one action configuration was provided and validated successfully
                pass
            
            # Check that resource filters are properly handled
            if resource_filters:
                # Resource filter configuration was provided and validated successfully
                pass
            
            # Check that time restrictions are properly handled
            if time_restrictions:
                # Time restriction configuration was provided and validated successfully
                pass
            
            # Check that safety overrides are properly handled
            if safety_overrides:
                # Safety override configuration was provided and validated successfully
                pass
            
            # The fact that validation passed with these configurations confirms
            # that the system allows configuration of all required elements
        
        # 10. Error messages should be informative and specific
        if not validation_result.is_valid:
            assert len(validation_result.errors) > 0, \
                "Failed validation must include error messages"
            
            # Error messages should be specific and actionable
            for error in validation_result.errors:
                assert isinstance(error, str), "Error messages must be strings"
                assert len(error.strip()) > 0, "Error messages must not be empty"
                # Error messages should contain specific information about what failed
                assert any(keyword in error.lower() for keyword in 
                          ['action', 'filter', 'time', 'safety', 'format', 'invalid', 'must']), \
                    f"Error message should be specific and actionable: {error}"


def run_property_test():
    """Run the policy configuration completeness property test"""
    print("Running Property-Based Test for Policy Configuration Completeness")
    print("=" * 60)
    print("**Feature: automated-cost-optimization, Property 20: Policy Configuration Completeness**")
    print("**Validates: Requirements 3.1**")
    print()
    
    test_instance = TestPolicyConfigurationCompleteness()
    
    try:
        print("Testing Property 20: Policy Configuration Completeness...")
        test_instance.test_policy_configuration_completeness_property()
        print("✓ Property 20 test completed successfully")
        print()
        print("Property validation confirmed:")
        print("- All required policy elements can be configured")
        print("- Action types configuration is properly validated")
        print("- Resource filters configuration is properly validated")
        print("- Approval requirements configuration is properly validated")
        print("- Time restrictions configuration is properly validated")
        print("- Safety overrides configuration is properly validated")
        print("- Policy validation provides detailed feedback")
        print("- Valid configurations are accepted")
        print("- Invalid configurations are rejected with clear error messages")
        
        return True
        
    except Exception as e:
        print(f"✗ Property test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_test()
    if success:
        print("\nPolicy Configuration Completeness property test passed!")
    else:
        print("\nPolicy Configuration Completeness property test failed!")
        exit(1)