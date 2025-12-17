"""
Test script to verify the core automation infrastructure works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from decimal import Decimal
import uuid

# Import the core automation components
from core.auto_remediation_engine import AutoRemediationEngine, OptimizationOpportunity
from core.safety_checker import SafetyChecker
from core.action_engine import ActionEngine
from core.rollback_manager import RollbackManager
from core.automation_audit_logger import AutomationAuditLogger
from core.policy_manager import PolicyManager
from core.automation_models import (
    AutomationLevel, ActionType, RiskLevel, AutomationPolicy
)

def test_core_infrastructure():
    """Test that all core components can be instantiated and basic methods work"""
    
    print("Testing Core Automation Infrastructure...")
    
    # Test 1: Component instantiation
    print("\n1. Testing component instantiation...")
    
    try:
        engine = AutoRemediationEngine()
        safety_checker = SafetyChecker()
        action_engine = ActionEngine()
        rollback_manager = RollbackManager()
        audit_logger = AutomationAuditLogger()
        policy_manager = PolicyManager()
        
        print("✓ All components instantiated successfully")
    except Exception as e:
        print(f"✗ Component instantiation failed: {e}")
        return False
    
    # Test 2: Safety Checker functionality
    print("\n2. Testing SafetyChecker...")
    
    try:
        # Test production tag checking
        production_tags = {"Environment": "production", "Team": "backend"}
        non_production_tags = {"Environment": "development", "Team": "frontend"}
        
        has_prod_tags = safety_checker.check_production_tags(production_tags)
        has_no_prod_tags = safety_checker.check_production_tags(non_production_tags)
        
        assert has_prod_tags == True, "Should detect production tags"
        assert has_no_prod_tags == False, "Should not detect production tags"
        
        print("✓ SafetyChecker production tag detection works")
        
        # Test business hours verification
        business_hours = {"start": "09:00", "end": "17:00"}
        test_time = datetime(2024, 1, 15, 14, 30)  # 2:30 PM
        
        in_business_hours = safety_checker.verify_business_hours(business_hours, test_time)
        assert in_business_hours == True, "Should be in business hours"
        
        print("✓ SafetyChecker business hours verification works")
        
        # Test risk assessment
        resource_metadata = {
            "tags": {"Environment": "development"},
            "instance_type": "t3.micro"
        }
        
        risk_level = safety_checker.assess_action_risk(ActionType.STOP_INSTANCE, resource_metadata)
        assert risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH], "Should return valid risk level"
        
        print("✓ SafetyChecker risk assessment works")
        
    except Exception as e:
        print(f"✗ SafetyChecker test failed: {e}")
        return False
    
    # Test 3: ActionEngine functionality
    print("\n3. Testing ActionEngine...")
    
    try:
        # Test batch operations
        instance_ids = ["i-1234567890abcdef0", "i-0987654321fedcba0"]
        results = action_engine.stop_unused_instances(instance_ids)
        
        assert len(results) == 2, "Should return results for all instances"
        assert all(isinstance(success, bool) for success in results.values()), "Results should be boolean"
        
        print("✓ ActionEngine batch operations work")
        
        # Test volume operations
        volume_ids = ["vol-1234567890abcdef0"]
        volume_results = action_engine.delete_unattached_volumes(volume_ids)
        
        assert len(volume_results) == 1, "Should return result for volume"
        
        print("✓ ActionEngine volume operations work")
        
    except Exception as e:
        print(f"✗ ActionEngine test failed: {e}")
        return False
    
    # Test 4: RollbackManager functionality
    print("\n4. Testing RollbackManager...")
    
    try:
        # Create a test opportunity
        opportunity = OptimizationOpportunity(
            resource_id="i-1234567890abcdef0",
            resource_type="ec2_instance",
            action_type=ActionType.STOP_INSTANCE,
            estimated_monthly_savings=Decimal("50.00"),
            risk_level=RiskLevel.LOW,
            resource_metadata={"instance_type": "t3.micro", "tags": {}},
            detection_details={}
        )
        
        # Test rollback plan creation
        rollback_plan = rollback_manager.create_rollback_plan(opportunity)
        
        assert "rollback_steps" in rollback_plan, "Rollback plan should have steps"
        assert "estimated_rollback_cost" in rollback_plan, "Rollback plan should have cost estimate"
        
        print("✓ RollbackManager rollback plan creation works")
        
    except Exception as e:
        print(f"✗ RollbackManager test failed: {e}")
        return False
    
    # Test 5: PolicyManager functionality
    print("\n5. Testing PolicyManager...")
    
    try:
        # Test policy validation
        validation_result = policy_manager.validate_policy_configuration(
            automation_level=AutomationLevel.BALANCED,
            enabled_actions=[ActionType.STOP_INSTANCE.value, ActionType.RELEASE_ELASTIC_IP.value],
            approval_required_actions=[ActionType.STOP_INSTANCE.value],
            blocked_actions=[ActionType.TERMINATE_INSTANCE.value],
            resource_filters={"min_cost_threshold": 10.0},
            time_restrictions={"business_hours": {"start": "09:00", "end": "17:00"}},
            safety_overrides={}
        )
        
        assert validation_result.is_valid == True, "Valid policy should pass validation"
        
        print("✓ PolicyManager policy validation works")
        
        # Test dry run simulation
        opportunities = [opportunity]  # Use the opportunity from rollback test
        
        # Create a mock policy object for dry run
        class MockPolicy:
            def __init__(self):
                self.id = uuid.uuid4()
                self.name = "Test Policy"
                self.enabled_actions = [ActionType.STOP_INSTANCE.value]
                self.approval_required_actions = []
                self.blocked_actions = []
                self.resource_filters = {}
                self.time_restrictions = {}
                self.safety_overrides = {}
        
        mock_policy = MockPolicy()
        
        dry_run_results = policy_manager.simulate_dry_run([opportunity], mock_policy)
        
        assert "total_opportunities" in dry_run_results, "Dry run should report total opportunities"
        assert dry_run_results["total_opportunities"] == 1, "Should process one opportunity"
        
        print("✓ PolicyManager dry run simulation works")
        
    except Exception as e:
        print(f"✗ PolicyManager test failed: {e}")
        return False
    
    # Test 6: AuditLogger functionality
    print("\n6. Testing AutomationAuditLogger...")
    
    try:
        # Generate correlation ID
        correlation_id = audit_logger.generate_correlation_id()
        assert correlation_id.startswith("auto-"), "Correlation ID should have correct prefix"
        
        print("✓ AutomationAuditLogger correlation ID generation works")
        
        # Test integrity hash calculation
        test_data = {"action": "test", "resource": "test-resource"}
        hash1 = audit_logger._calculate_integrity_hash(test_data)
        hash2 = audit_logger._calculate_integrity_hash(test_data)
        
        assert hash1 == hash2, "Same data should produce same hash"
        assert len(hash1) == 64, "SHA256 hash should be 64 characters"
        
        print("✓ AutomationAuditLogger integrity hash calculation works")
        
    except Exception as e:
        print(f"✗ AutomationAuditLogger test failed: {e}")
        return False
    
    # Test 7: AutoRemediationEngine integration
    print("\n7. Testing AutoRemediationEngine integration...")
    
    try:
        # Test opportunity detection (should return empty list since no real AWS integration)
        class MockPolicy:
            def __init__(self):
                self.id = uuid.uuid4()
                self.enabled_actions = [ActionType.STOP_INSTANCE.value]
                self.time_restrictions = {}
                self.safety_overrides = {}
        
        mock_policy = MockPolicy()
        opportunities = engine.detect_optimization_opportunities(mock_policy)
        
        assert isinstance(opportunities, list), "Should return list of opportunities"
        
        print("✓ AutoRemediationEngine opportunity detection works")
        
        # Test safety validation
        safety_passed, safety_details = engine.validate_safety_requirements(opportunity, mock_policy)
        
        assert isinstance(safety_passed, bool), "Safety validation should return boolean"
        assert isinstance(safety_details, dict), "Safety details should be dictionary"
        
        print("✓ AutoRemediationEngine safety validation works")
        
    except Exception as e:
        print(f"✗ AutoRemediationEngine test failed: {e}")
        return False
    
    print("\n🎉 All core infrastructure tests passed!")
    print("\nCore automation infrastructure is working correctly:")
    print("- AutoRemediationEngine: ✓")
    print("- SafetyChecker: ✓") 
    print("- ActionEngine: ✓")
    print("- RollbackManager: ✓")
    print("- PolicyManager: ✓")
    print("- AutomationAuditLogger: ✓")
    
    return True

if __name__ == "__main__":
    success = test_core_infrastructure()
    if success:
        print("\n✅ Core automation infrastructure setup completed successfully!")
    else:
        print("\n❌ Core automation infrastructure setup failed!")
        sys.exit(1)