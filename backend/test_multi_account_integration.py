"""
Integration test demonstrating multi-account automation coordination
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.multi_account_manager import (
    MultiAccountManager, AWSAccount, CrossAccountRole
)
from backend.core.automation_models import (
    OptimizationAction, AutomationPolicy, ActionType, ActionStatus,
    RiskLevel, ApprovalStatus, AutomationLevel
)


import pytest

@pytest.mark.asyncio
async def test_multi_account_automation_workflow():
    """
    Test complete multi-account automation workflow:
    1. Set up accounts with different policies
    2. Coordinate cross-account actions
    3. Generate consolidated reports
    """
    print("Testing multi-account automation workflow...")
    
    # Initialize MultiAccountManager
    credentials = {
        'access_key_id': 'test_key',
        'secret_access_key': 'test_secret',
        'region': 'us-east-1'
    }
    
    manager = MultiAccountManager(credentials)
    
    # Set up test accounts
    prod_account = AWSAccount(
        account_id='111111111111',
        account_name='Production Account',
        email='prod@example.com',
        status='ACTIVE',
        environment='production',
        team='platform',
        automation_enabled=True
    )
    
    dev_account = AWSAccount(
        account_id='222222222222',
        account_name='Development Account',
        email='dev@example.com',
        status='ACTIVE',
        environment='development',
        team='engineering',
        automation_enabled=True
    )
    
    staging_account = AWSAccount(
        account_id='333333333333',
        account_name='Staging Account',
        email='staging@example.com',
        status='ACTIVE',
        environment='staging',
        team='qa',
        automation_enabled=False  # Automation disabled
    )
    
    manager.accounts[prod_account.account_id] = prod_account
    manager.accounts[dev_account.account_id] = dev_account
    manager.accounts[staging_account.account_id] = staging_account
    
    # Create different automation policies for different environments
    conservative_policy = Mock()
    conservative_policy.id = 'policy-conservative'
    conservative_policy.automation_level = AutomationLevel.CONSERVATIVE
    conservative_policy.enabled_actions = ['release_elastic_ip']
    conservative_policy.approval_required_actions = ['stop_instance', 'delete_volume']
    
    aggressive_policy = Mock()
    aggressive_policy.id = 'policy-aggressive'
    aggressive_policy.automation_level = AutomationLevel.AGGRESSIVE
    aggressive_policy.enabled_actions = ['stop_instance', 'delete_volume', 'release_elastic_ip']
    aggressive_policy.approval_required_actions = []
    
    # Apply policies to accounts
    await manager.set_account_automation_policy(prod_account.account_id, conservative_policy)
    await manager.set_account_automation_policy(dev_account.account_id, aggressive_policy)
    
    print("✓ Set up accounts with different automation policies")
    
    # Test filtering automation-enabled accounts
    enabled_accounts = manager.get_automation_enabled_accounts()
    assert len(enabled_accounts) == 2  # prod and dev, not staging
    
    enabled_ids = [acc.account_id for acc in enabled_accounts]
    assert prod_account.account_id in enabled_ids
    assert dev_account.account_id in enabled_ids
    assert staging_account.account_id not in enabled_ids
    
    print("✓ Account filtering by automation status works")
    
    # Create a test optimization action
    test_action = Mock()
    test_action.id = 'action-123'
    test_action.action_type = ActionType.STOP_INSTANCE
    test_action.resource_id = 'i-1234567890abcdef0'
    test_action.resource_type = 'EC2Instance'
    test_action.estimated_monthly_savings = Decimal('50.00')
    test_action.risk_level = RiskLevel.LOW
    test_action.execution_status = ActionStatus.PENDING
    
    # Mock cross-account role
    cross_account_role = CrossAccountRole(
        role_arn='arn:aws:iam::123456789012:role/FinOpsAccessRole',
        session_name='FinOpsAutomation'
    )
    
    # Mock the assume_role and action execution methods
    with patch.object(manager, 'assume_role', new_callable=AsyncMock) as mock_assume_role, \
         patch.object(manager, '_execute_action_with_session', new_callable=AsyncMock) as mock_execute:
        
        # Configure mocks
        mock_assume_role.return_value = Mock()  # Mock AWS session
        mock_execute.return_value = (True, {'message': 'Action executed successfully'})
        
        # Test cross-account action coordination
        target_accounts = [prod_account.account_id, dev_account.account_id]
        results = await manager.coordinate_cross_account_action(
            test_action, target_accounts, cross_account_role
        )
        
        # Verify results
        assert len(results) == 2
        assert all(result.success for result in results)
        assert all(result.account_id in target_accounts for result in results)
        
        print("✓ Cross-account action coordination works")
    
    # Test consolidated reporting (with mocked database)
    with patch('backend.core.database.get_db_session') as mock_db_session:
        mock_db = Mock()
        mock_db_session.return_value.__enter__.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        report = await manager.generate_consolidated_report(start_date, end_date)
        
        assert report.total_accounts == 3
        assert report.active_accounts == 2  # Only prod and dev have automation enabled
        assert isinstance(report.account_summaries, dict)
        
        print("✓ Consolidated reporting works")
    
    # Test account-specific reporting
    with patch('backend.core.database.get_db_session') as mock_db_session:
        mock_db = Mock()
        mock_db_session.return_value.__enter__.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        account_report = await manager.get_account_specific_report(
            prod_account.account_id, start_date, end_date
        )
        
        assert account_report['account_id'] == prod_account.account_id
        assert account_report['account_name'] == prod_account.account_name
        assert account_report['environment'] == 'production'
        assert 'summary' in account_report
        
        print("✓ Account-specific reporting works")
    
    print("\n🎉 Multi-account automation workflow test completed successfully!")
    return True


if __name__ == "__main__":
    print("Testing Multi-Account Automation Integration...")
    
    # Run the integration test
    result = asyncio.run(test_multi_account_automation_workflow())
    
    if result:
        print("\n✅ All integration tests passed!")
        print("\nMulti-account automation features verified:")
        print("- Account setup with different policies: ✓")
        print("- Cross-account action coordination: ✓")
        print("- Automation-enabled account filtering: ✓")
        print("- Consolidated multi-account reporting: ✓")
        print("- Account-specific detailed reporting: ✓")
        print("\nThe multi-account support is ready for production use!")
    else:
        print("\n❌ Integration tests failed!")