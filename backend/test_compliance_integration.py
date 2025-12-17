#!/usr/bin/env python3
"""
Integration test for ComplianceManager with automation models
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

from core.compliance_manager import ComplianceManager, RetentionPolicy, AnonymizationRule
from core.compliance_framework import ComplianceStandard, DataClassification
from core.automation_models import ActionType, RiskLevel, ActionStatus


def test_compliance_manager_integration():
    """Test ComplianceManager integration with automation models"""
    
    print("Testing ComplianceManager integration...")
    
    # Create a mock database session
    class MockDB:
        def query(self, model_class):
            return MockQuery()
        def commit(self):
            pass
        def rollback(self):
            pass
    
    class MockQuery:
        def filter(self, *args):
            return self
        def order_by(self, *args):
            return self
        def first(self):
            return None
        def all(self):
            return []
    
    # Initialize ComplianceManager
    compliance_manager = ComplianceManager(db_session=MockDB())
    
    # Test 1: Create retention policy
    policy = RetentionPolicy(
        policy_id="test_policy",
        name="Test Policy",
        data_types=["optimization_action"],
        retention_period_days=365,
        compliance_standards=[ComplianceStandard.SOC2]
    )
    
    result = compliance_manager.add_retention_policy(policy)
    assert result, "Should be able to add retention policy"
    
    # Test 2: Create anonymization rule
    rule = AnonymizationRule(
        rule_id="test_rule",
        field_pattern=r"user_id",
        anonymization_method="hash",
        applies_to_data_types=["optimization_action"]
    )
    
    result = compliance_manager.add_anonymization_rule(rule)
    assert result, "Should be able to add anonymization rule"
    
    # Test 3: Test data anonymization
    test_data = {
        "user_id": "user123",
        "resource_id": "i-1234567890abcdef0",
        "action_type": ActionType.STOP_INSTANCE.value
    }
    
    from core.compliance_manager import AnonymizationLevel
    anonymized = compliance_manager.anonymize_data(
        test_data, "optimization_action", AnonymizationLevel.PARTIAL
    )
    
    assert anonymized is not None, "Anonymization should return data"
    assert anonymized["user_id"] != test_data["user_id"], "User ID should be anonymized"
    
    # Test 4: Validate configuration
    validation = compliance_manager.validate_compliance_configuration()
    assert validation["status"] in ["VALID", "INCOMPLETE"], "Should return validation status"
    
    print("✓ All integration tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_compliance_manager_integration()
        print("\nComplianceManager integration test successful!")
    except Exception as e:
        print(f"\nIntegration test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)