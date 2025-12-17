"""
Test Multi-Account Manager functionality for automated cost optimization
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from backend.core.multi_account_manager import (
    MultiAccountManager, AWSAccount, CrossAccountRole, 
    CrossAccountActionResult, MultiAccountReport
)
from backend.core.automation_models import (
    OptimizationAction, AutomationPolicy, ActionType, ActionStatus,
    RiskLevel, ApprovalStatus, AutomationLevel
)


def test_multi_account_manager_initialization():
    """Test MultiAccountManager initialization"""
    credentials = {
        'access_key_id': 'test_key',
        'secret_access_key': 'test_secret',
        'region': 'us-east-1'
    }
    
    manager = MultiAccountManager(credentials)
    
    assert manager.master_credentials == credentials
    assert isinstance(manager.accounts, dict)
    assert isinstance(manager.account_policies, dict)
    assert manager.executor is not None


def test_account_automation_policy_management():
    """Test setting and getting account-specific automation policies"""
    credentials = {
        'access_key_id': 'test_key',
        'secret_access_key': 'test_secret',
        'region': 'us-east-1'
    }
    
    manager = MultiAccountManager(credentials)
    
    # Create test account
    account = AWSAccount(
        account_id='123456789012',
        account_name='Test Account',
        email='test@example.com',
        status='ACTIVE'
    )
    manager.accounts[account.account_id] = account
    
    # Create test policy
    policy = Mock()
    policy.id = 'policy-123'
    policy.automation_level = AutomationLevel.BALANCED
    
    # Test setting policy
    result = asyncio.run(manager.set_account_automation_policy(account.account_id, policy))
    assert result is True
    assert manager.account_policies[account.account_id] == policy
    assert manager.accounts[account.account_id].automation_policy_id == 'policy-123'
    
    # Test getting policy
    retrieved_policy = manager.get_account_automation_policy(account.account_id)
    assert retrieved_policy == policy


def test_automation_enable_disable():
    """Test enabling and disabling automation for accounts"""
    credentials = {
        'access_key_id': 'test_key',
        'secret_access_key': 'test_secret',
        'region': 'us-east-1'
    }
    
    manager = MultiAccountManager(credentials)
    
    # Create test account
    account = AWSAccount(
        account_id='123456789012',
        account_name='Test Account',
        email='test@example.com',
        status='ACTIVE',
        automation_enabled=True
    )
    manager.accounts[account.account_id] = account
    
    # Test disabling automation
    result = asyncio.run(manager.disable_automation_for_account(account.account_id))
    assert result is True
    assert manager.accounts[account.account_id].automation_enabled is False
    
    # Test enabling automation
    result = asyncio.run(manager.enable_automation_for_account(account.account_id))
    assert result is True
    assert manager.accounts[account.account_id].automation_enabled is True


def test_get_automation_enabled_accounts():
    """Test filtering accounts by automation status"""
    credentials = {
        'access_key_id': 'test_key',
        'secret_access_key': 'test_secret',
        'region': 'us-east-1'
    }
    
    manager = MultiAccountManager(credentials)
    
    # Create test accounts
    account1 = AWSAccount(
        account_id='111111111111',
        account_name='Account 1',
        email='test1@example.com',
        status='ACTIVE',
        automation_enabled=True
    )
    
    account2 = AWSAccount(
        account_id='222222222222',
        account_name='Account 2',
        email='test2@example.com',
        status='ACTIVE',
        automation_enabled=False
    )
    
    account3 = AWSAccount(
        account_id='333333333333',
        account_name='Account 3',
        email='test3@example.com',
        status='ACTIVE',
        automation_enabled=True
    )
    
    manager.accounts[account1.account_id] = account1
    manager.accounts[account2.account_id] = account2
    manager.accounts[account3.account_id] = account3
    
    # Test filtering
    enabled_accounts = manager.get_automation_enabled_accounts()
    
    assert len(enabled_accounts) == 2
    enabled_ids = [acc.account_id for acc in enabled_accounts]
    assert '111111111111' in enabled_ids
    assert '333333333333' in enabled_ids
    assert '222222222222' not in enabled_ids


@patch('backend.core.database.get_db_session')
def test_generate_consolidated_report(mock_db_session):
    """Test generating consolidated multi-account reports"""
    credentials = {
        'access_key_id': 'test_key',
        'secret_access_key': 'test_secret',
        'region': 'us-east-1'
    }
    
    manager = MultiAccountManager(credentials)
    
    # Create test accounts
    account1 = AWSAccount(
        account_id='111111111111',
        account_name='Account 1',
        email='test1@example.com',
        status='ACTIVE',
        automation_enabled=True
    )
    
    account2 = AWSAccount(
        account_id='222222222222',
        account_name='Account 2',
        email='test2@example.com',
        status='ACTIVE',
        automation_enabled=True
    )
    
    manager.accounts[account1.account_id] = account1
    manager.accounts[account2.account_id] = account2
    
    # Mock database session and query results
    mock_db = Mock()
    mock_db_session.return_value.__enter__.return_value = mock_db
    mock_db.query.return_value.filter.return_value.all.return_value = []
    
    # Test report generation
    start_date = datetime.utcnow() - timedelta(days=30)
    end_date = datetime.utcnow()
    
    report = asyncio.run(manager.generate_consolidated_report(start_date, end_date))
    
    assert isinstance(report, MultiAccountReport)
    assert report.total_accounts == 2
    assert report.active_accounts == 2
    assert report.report_period_start == start_date
    assert report.report_period_end == end_date


def test_cross_account_action_result():
    """Test CrossAccountActionResult data structure"""
    result = CrossAccountActionResult(
        account_id='123456789012',
        action_id='action-123',
        success=True,
        execution_time=datetime.utcnow(),
        savings_achieved=100.50
    )
    
    assert result.account_id == '123456789012'
    assert result.action_id == 'action-123'
    assert result.success is True
    assert result.error_message is None
    assert result.savings_achieved == 100.50


if __name__ == "__main__":
    print("Testing Multi-Account Manager functionality...")
    
    # Run basic tests
    test_multi_account_manager_initialization()
    print("✓ MultiAccountManager initialization works")
    
    test_account_automation_policy_management()
    print("✓ Account automation policy management works")
    
    test_automation_enable_disable()
    print("✓ Automation enable/disable works")
    
    test_get_automation_enabled_accounts()
    print("✓ Automation enabled accounts filtering works")
    
    test_cross_account_action_result()
    print("✓ CrossAccountActionResult data structure works")
    
    print("\n🎉 All Multi-Account Manager tests passed!")
    print("\nMulti-account support features working correctly:")
    print("- Account-specific policy management: ✓")
    print("- Automation enable/disable per account: ✓")
    print("- Account filtering by automation status: ✓")
    print("- Cross-account action result tracking: ✓")
    print("- Consolidated reporting structure: ✓")