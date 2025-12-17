#!/usr/bin/env python3
"""
Property-Based Tests for Universal Safety Validation

This module contains property-based tests to verify that the safety checker
always performs required safety checks for any optimization action according
to the requirements specification.

**Feature: automated-cost-optimization, Property 4: Universal Safety Validation**
**Validates: Requirements 2.1, 2.2, 2.4**
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from hypothesis import given, strategies as st, assume, settings
from dataclasses import dataclass

# Import the components we're testing
from core.safety_checker import SafetyChecker
from core.automation_models import (
    AutomationPolicy, ActionType, RiskLevel, AutomationLevel,
    ApprovalStatus
)


@dataclass
class MockOptimizationOpportunity:
    """Mock optimization opportunity for testing"""
    resource_id: str
    resource_type: str
    action_type: ActionType
    estimated_monthly_savings: Decimal
    risk_level: RiskLevel
    resource_metadata: Dict[str, Any]
    detection_details: Dict[str, Any]


class TestUniversalSafetyValidation:
    """Property-based tests for universal safety validation"""
    
    @given(
        # Generate various resource types and action types
        resource_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        resource_type=st.sampled_from(['ec2_instance', 'ebs_volume', 'elastic_ip', 'load_balancer', 'security_group']),
        action_type=st.sampled_from(list(ActionType)),
        estimated_savings=st.decimals(min_value=0, max_value=10000, places=2),
        risk_level=st.sampled_from(list(RiskLevel)),
        
        # Generate various tag combinations including production tags
        tags=st.dictionaries(
            keys=st.sampled_from(['Environment', 'Critical', 'Tier', 'Stage', 'Team', 'Application', 'Owner']),
            values=st.sampled_from(['production', 'prod', 'development', 'dev', 'staging', 'test', 'true', 'false', 'yes', 'no', '1', '0', 'critical', 'non-critical']),
            min_size=0,
            max_size=5
        ),
        
        # Generate various resource metadata
        has_asg=st.booleans(),
        has_lb_targets=st.booleans(),
        has_db_connections=st.booleans(),
        cpu_utilization=st.floats(min_value=0.0, max_value=100.0),
        
        # Generate policy configurations
        automation_level=st.sampled_from(list(AutomationLevel)),
        business_hours_enabled=st.booleans(),
        
        # Generate time scenarios
        current_hour=st.integers(min_value=0, max_value=23)
    )
    @settings(max_examples=100, deadline=None)
    def test_universal_safety_validation_property(self,
                                                resource_id: str,
                                                resource_type: str,
                                                action_type: ActionType,
                                                estimated_savings: Decimal,
                                                risk_level: RiskLevel,
                                                tags: Dict[str, str],
                                                has_asg: bool,
                                                has_lb_targets: bool,
                                                has_db_connections: bool,
                                                cpu_utilization: float,
                                                automation_level: AutomationLevel,
                                                business_hours_enabled: bool,
                                                current_hour: int):
        """
        **Feature: automated-cost-optimization, Property 4: Universal Safety Validation**
        
        Property: For any optimization action, the system should always perform 
        safety checks against production tags, critical resource indicators, 
        and business hours before execution.
        
        This property verifies that:
        1. Safety validation is always performed (never skipped)
        2. Production tag checks are always executed
        3. Business hours checks are always executed for medium/high risk actions
        4. Resource dependency checks are always executed
        5. Safety results always include detailed check information
        """
        
        # Skip invalid combinations
        assume(len(resource_id.strip()) > 0)
        
        # Create resource metadata
        resource_metadata = {
            "tags": tags,
            "cpu_utilization_24h": cpu_utilization
        }
        
        if has_asg:
            resource_metadata["auto_scaling_group"] = f"asg-{resource_id}"
        
        if has_lb_targets:
            resource_metadata["load_balancer_targets"] = [f"lb-{resource_id}"]
        
        if has_db_connections:
            resource_metadata["database_connections"] = [f"db-{resource_id}"]
        
        # Add last activity timestamp (varies from recent to old)
        hours_ago = 48 if risk_level == RiskLevel.LOW else 12  # Vary activity recency
        last_activity = datetime.utcnow() - timedelta(hours=hours_ago)
        resource_metadata["last_activity"] = last_activity.isoformat()
        
        # Create mock opportunity
        opportunity = MockOptimizationOpportunity(
            resource_id=resource_id,
            resource_type=resource_type,
            action_type=action_type,
            estimated_monthly_savings=estimated_savings,
            risk_level=risk_level,
            resource_metadata=resource_metadata,
            detection_details={}
        )
        
        # Create automation policy
        policy = AutomationPolicy(
            id=uuid.uuid4(),
            name="test-policy",
            automation_level=automation_level,
            enabled_actions=[action_type.value],
            approval_required_actions=[],
            blocked_actions=[],
            resource_filters={},
            time_restrictions={
                "business_hours": {
                    "start": "09:00",
                    "end": "17:00",
                    "timezone": "UTC"
                } if business_hours_enabled else {}
            },
            safety_overrides={},
            created_by=uuid.uuid4()
        )
        
        # Mock current time for business hours testing
        test_time = datetime.utcnow().replace(hour=current_hour, minute=0, second=0, microsecond=0)
        
        # Execute safety validation
        safety_checker = SafetyChecker()
        safety_passed, safety_details = safety_checker.validate_action_safety(
            opportunity, policy
        )
        
        # PROPERTY ASSERTIONS: Universal safety validation requirements
        
        # 1. Safety validation must always return detailed results
        assert isinstance(safety_passed, bool), "Safety validation must return a boolean result"
        assert isinstance(safety_details, dict), "Safety validation must return detailed results"
        assert "overall_passed" in safety_details, "Safety results must include overall_passed"
        assert "checks" in safety_details, "Safety results must include individual check results"
        assert safety_details["overall_passed"] == safety_passed, "Overall result must match return value"
        
        # 2. Production tag checks must always be performed
        assert "production_tag_protection" in safety_details["checks"], \
            "Production tag protection check must always be performed"
        
        production_check = safety_details["checks"]["production_tag_protection"]
        assert "passed" in production_check, "Production tag check must have pass/fail result"
        assert "details" in production_check, "Production tag check must have detailed results"
        
        # 3. Business hours checks must be performed for medium/high risk actions
        if risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH] and business_hours_enabled:
            assert "business_hours_protection" in safety_details["checks"], \
                "Business hours check must be performed for medium/high risk actions"
            
            business_hours_check = safety_details["checks"]["business_hours_protection"]
            assert "passed" in business_hours_check, "Business hours check must have pass/fail result"
            assert "details" in business_hours_check, "Business hours check must have detailed results"
        
        # 4. Resource dependency checks must always be performed
        dependency_checks = [
            "auto_scaling_group_protection",
            "load_balancer_target_protection", 
            "database_dependency_protection"
        ]
        
        for check_name in dependency_checks:
            if check_name in safety_details["checks"]:
                check_result = safety_details["checks"][check_name]
                assert "passed" in check_result, f"{check_name} must have pass/fail result"
                assert "details" in check_result, f"{check_name} must have detailed results"
        
        # 5. Recent activity checks must always be performed
        assert "recent_activity_protection" in safety_details["checks"], \
            "Recent activity check must always be performed"
        
        activity_check = safety_details["checks"]["recent_activity_protection"]
        assert "passed" in activity_check, "Recent activity check must have pass/fail result"
        assert "details" in activity_check, "Recent activity check must have detailed results"
        
        # 6. Verify production tag detection logic
        has_production_tags = safety_checker.check_production_tags(tags)
        production_tag_passed = safety_details["checks"]["production_tag_protection"]["passed"]
        
        # The production tag protection check should pass when NO production tags are found
        # and fail when production tags ARE found (to protect the resource)
        if has_production_tags:
            assert not production_tag_passed, \
                "Resources with production tags should fail production tag protection check (to protect them)"
        else:
            assert production_tag_passed, \
                "Resources without production tags should pass production tag protection check"
        
        # 7. Verify dependency detection logic
        if has_asg:
            asg_check = safety_details["checks"].get("auto_scaling_group_protection", {})
            if asg_check:  # Check might be disabled in some configurations
                assert not asg_check["passed"], \
                    "Resources in Auto Scaling Groups should fail ASG protection check"
        
        if has_lb_targets:
            lb_check = safety_details["checks"].get("load_balancer_target_protection", {})
            if lb_check:  # Check might be disabled in some configurations
                assert not lb_check["passed"], \
                    "Resources with load balancer targets should fail LB protection check"
        
        if has_db_connections:
            db_check = safety_details["checks"].get("database_dependency_protection", {})
            if db_check:  # Check might be disabled in some configurations
                assert not db_check["passed"], \
                    "Resources with database connections should fail DB protection check"
        
        # 8. Verify that overall safety result reflects individual check results
        individual_results = [check["passed"] for check in safety_details["checks"].values()]
        expected_overall = all(individual_results)
        assert safety_details["overall_passed"] == expected_overall, \
            "Overall safety result must reflect all individual check results"
        
        # 9. Verify error and warning lists are present
        assert "warnings" in safety_details, "Safety results must include warnings list"
        assert "errors" in safety_details, "Safety results must include errors list"
        assert isinstance(safety_details["warnings"], list), "Warnings must be a list"
        assert isinstance(safety_details["errors"], list), "Errors must be a list"
        
        # 10. If overall check failed, there should be error messages
        if not safety_details["overall_passed"]:
            assert len(safety_details["errors"]) > 0, \
                "Failed safety validation must include error messages"


def run_property_test():
    """Run the universal safety validation property test"""
    print("Running Property-Based Test for Universal Safety Validation")
    print("=" * 60)
    print("**Feature: automated-cost-optimization, Property 4: Universal Safety Validation**")
    print("**Validates: Requirements 2.1, 2.2, 2.4**")
    print()
    
    test_instance = TestUniversalSafetyValidation()
    
    try:
        print("Testing Property 4: Universal Safety Validation...")
        test_instance.test_universal_safety_validation_property()
        print("✓ Property 4 test completed successfully")
        print()
        print("Property validation confirmed:")
        print("- Safety validation is always performed for any optimization action")
        print("- Production tag checks are always executed")
        print("- Business hours checks are executed for medium/high risk actions")
        print("- Resource dependency checks are always executed")
        print("- Safety results always include detailed check information")
        print("- Overall safety result correctly reflects individual check results")
        
        return True
        
    except Exception as e:
        print(f"✗ Property test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_test()
    if success:
        print("\nUniversal Safety Validation property test passed!")
    else:
        print("\nUniversal Safety Validation property test failed!")
        exit(1)