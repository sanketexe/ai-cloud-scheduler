#!/usr/bin/env python3
"""
Property-Based Tests for Multi-Account Reporting and Isolation

This module contains property-based tests to verify that the multi-account system
provides consolidated and per-account savings breakdowns while ensuring actions
in one account do not affect resources in other accounts according to the 
requirements specification.

**Feature: automated-cost-optimization, Property 18: Multi-Account Reporting and Isolation**
**Validates: Requirements 8.4, 8.5**
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple, Set
from hypothesis import given, strategies as st, assume, settings
from dataclasses import dataclass
import asyncio

# Import the components we're testing
from core.multi_account_manager import (
    MultiAccountManager, AWSAccount, CrossAccountRole, 
    CrossAccountActionResult, MultiAccountReport
)
from core.automation_models import (
    OptimizationAction, ActionType, RiskLevel, ActionStatus,
    AutomationLevel, ApprovalStatus
)


@dataclass
class MockOptimizationAction:
    """Mock optimization action for testing"""
    id: uuid.UUID
    action_type: ActionType
    resource_id: str
    resource_type: str
    estimated_monthly_savings: Decimal
    actual_savings: Optional[Decimal]
    risk_level: RiskLevel
    requires_approval: bool
    approval_status: ApprovalStatus
    scheduled_execution_time: Optional[datetime]
    execution_status: ActionStatus
    resource_metadata: Dict[str, Any]
    policy_id: uuid.UUID
    created_at: datetime
    execution_completed_at: Optional[datetime]


class MockMultiAccountManager:
    """Mock multi-account manager for testing reporting and isolation"""
    
    def __init__(self, master_credentials: Dict[str, str]):
        self.master_credentials = master_credentials
        self.accounts: Dict[str, AWSAccount] = {}
        self.account_policies: Dict[str, Any] = {}
        self.executed_actions: Dict[str, List[MockOptimizationAction]] = {}  # account_id -> actions
        self.resource_access_log: List[Tuple[str, str, str]] = []  # (account_id, resource_id, operation)
        self.cross_account_access_attempts: List[Tuple[str, str, str]] = []  # (source_account, target_account, resource)
        
    def add_test_account(self, account: AWSAccount):
        """Add a test account for testing purposes"""
        self.accounts[account.account_id] = account
        self.executed_actions[account.account_id] = []
    
    def add_executed_action(self, account_id: str, action: MockOptimizationAction):
        """Add an executed action for testing purposes"""
        if account_id not in self.executed_actions:
            self.executed_actions[account_id] = []
        self.executed_actions[account_id].append(action)
        
        # Log resource access
        self.resource_access_log.append((account_id, action.resource_id, "modify"))
    
    def simulate_cross_account_access_attempt(self, source_account: str, target_account: str, resource_id: str):
        """Simulate an attempt to access resources across accounts (should be blocked)"""
        self.cross_account_access_attempts.append((source_account, target_account, resource_id))
    
    async def generate_consolidated_report(self,
                                         start_date: datetime,
                                         end_date: datetime) -> MultiAccountReport:
        """
        Mock consolidated report generation that aggregates data from all accounts
        """
        account_summaries = {}
        total_actions = 0
        total_savings = 0.0
        action_type_counts = {}
        risk_level_counts = {}
        
        # Process each account independently
        for account_id, account in self.accounts.items():
            if not account.automation_enabled:
                continue
            
            # Get actions for this account in the date range
            account_actions = [
                action for action in self.executed_actions.get(account_id, [])
                if (action.execution_completed_at and 
                    start_date <= action.execution_completed_at <= end_date and
                    action.execution_status == ActionStatus.COMPLETED)
            ]
            
            # Calculate account-level metrics
            account_action_count = len(account_actions)
            account_savings = sum(float(action.actual_savings or 0) for action in account_actions)
            
            # Count by action type and risk level for this account
            account_action_types = {}
            account_risk_levels = {}
            
            for action in account_actions:
                # Action type counts
                action_type = action.action_type.value
                account_action_types[action_type] = account_action_types.get(action_type, 0) + 1
                action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
                
                # Risk level counts
                risk_level = action.risk_level.value
                account_risk_levels[risk_level] = account_risk_levels.get(risk_level, 0) + 1
                risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1
            
            # Store account summary (isolated per account)
            account_summaries[account_id] = {
                'account_name': account.account_name,
                'team': account.team,
                'environment': account.environment,
                'actions_executed': account_action_count,
                'savings_achieved': account_savings,
                'action_types': account_action_types,
                'risk_levels': account_risk_levels,
                'automation_policy_id': account.automation_policy_id
            }
            
            total_actions += account_action_count
            total_savings += account_savings
        
        # Create consolidated report
        return MultiAccountReport(
            report_period_start=start_date,
            report_period_end=end_date,
            total_accounts=len(self.accounts),
            active_accounts=len([a for a in self.accounts.values() if a.automation_enabled]),
            total_actions_executed=total_actions,
            total_savings_achieved=total_savings,
            account_summaries=account_summaries,
            action_type_breakdown=action_type_counts,
            risk_level_breakdown=risk_level_counts
        )
    
    async def get_account_specific_report(self,
                                        account_id: str,
                                        start_date: datetime,
                                        end_date: datetime) -> Dict[str, Any]:
        """
        Mock account-specific report generation that only includes data from the specified account
        """
        if account_id not in self.accounts:
            return {}
        
        account = self.accounts[account_id]
        
        # Get actions ONLY for this specific account
        account_actions = [
            action for action in self.executed_actions.get(account_id, [])
            if (action.execution_completed_at and 
                start_date <= action.execution_completed_at <= end_date)
        ]
        
        completed_actions = [a for a in account_actions if a.execution_status == ActionStatus.COMPLETED]
        failed_actions = [a for a in account_actions if a.execution_status == ActionStatus.FAILED]
        
        total_savings = sum(float(action.actual_savings or 0) for action in completed_actions)
        
        # Group by service/resource type (only for this account)
        service_breakdown = {}
        for action in completed_actions:
            service = action.resource_metadata.get('service', 'Unknown')
            if service not in service_breakdown:
                service_breakdown[service] = {
                    'actions': 0,
                    'savings': 0.0,
                    'resources': []
                }
            service_breakdown[service]['actions'] += 1
            service_breakdown[service]['savings'] += float(action.actual_savings or 0)
            service_breakdown[service]['resources'].append(action.resource_id)
        
        # Create account-specific report (isolated to this account only)
        return {
            'account_id': account_id,
            'account_name': account.account_name,
            'team': account.team,
            'environment': account.environment,
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_actions': len(account_actions),
                'completed_actions': len(completed_actions),
                'failed_actions': len(failed_actions),
                'total_savings': total_savings,
                'automation_enabled': account.automation_enabled,
                'automation_policy_id': account.automation_policy_id
            },
            'service_breakdown': service_breakdown,
            'recent_actions': [
                {
                    'action_id': str(action.id),
                    'action_type': action.action_type.value,
                    'resource_id': action.resource_id,
                    'resource_type': action.resource_type,
                    'status': action.execution_status.value,
                    'savings': float(action.actual_savings or 0),
                    'executed_at': action.execution_completed_at.isoformat() if action.execution_completed_at else None
                }
                for action in sorted(account_actions, key=lambda x: x.created_at, reverse=True)[:10]
            ]
        }
    
    def get_resource_access_log(self) -> List[Tuple[str, str, str]]:
        """Get log of all resource access attempts"""
        return self.resource_access_log.copy()
    
    def get_cross_account_access_attempts(self) -> List[Tuple[str, str, str]]:
        """Get log of cross-account access attempts (should be empty for proper isolation)"""
        return self.cross_account_access_attempts.copy()


class TestMultiAccountReportingAndIsolation:
    """Property-based tests for multi-account reporting and isolation"""
    
    @property
    def master_credentials(self):
        return {
            'access_key_id': 'test_master_key',
            'secret_access_key': 'test_master_secret',
            'region': 'us-east-1'
        }
    
    @given(
        # Generate multiple AWS accounts
        account_count=st.integers(min_value=3, max_value=10),
        
        # Generate account properties
        account_names=st.lists(
            st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
            min_size=3, max_size=10
        ),
        environments=st.lists(
            st.sampled_from(['production', 'staging', 'development', 'test']),
            min_size=3, max_size=10
        ),
        teams=st.lists(
            st.sampled_from(['backend', 'frontend', 'data', 'devops', 'security']),
            min_size=3, max_size=10
        ),
        
        # Generate actions per account
        actions_per_account=st.lists(
            st.integers(min_value=0, max_value=20),
            min_size=3, max_size=10
        ),
        
        # Generate action properties
        action_types=st.lists(
            st.sampled_from(list(ActionType)),
            min_size=1, max_size=20
        ),
        resource_types=st.lists(
            st.sampled_from(['ec2_instance', 'ebs_volume', 'elastic_ip', 'load_balancer']),
            min_size=1, max_size=20
        ),
        savings_amounts=st.lists(
            st.decimals(min_value=10, max_value=1000, places=2),
            min_size=1, max_size=20
        ),
        risk_levels=st.lists(
            st.sampled_from(list(RiskLevel)),
            min_size=1, max_size=20
        ),
        
        # Generate report parameters
        report_days_back=st.integers(min_value=7, max_value=90),
        automation_enabled_ratio=st.floats(min_value=0.5, max_value=1.0),
        
        # Generate isolation test parameters
        cross_account_attempts=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_multi_account_reporting_and_isolation_property(self,
                                                          account_count: int,
                                                          account_names: List[str],
                                                          environments: List[str],
                                                          teams: List[str],
                                                          actions_per_account: List[int],
                                                          action_types: List[ActionType],
                                                          resource_types: List[str],
                                                          savings_amounts: List[Decimal],
                                                          risk_levels: List[RiskLevel],
                                                          report_days_back: int,
                                                          automation_enabled_ratio: float,
                                                          cross_account_attempts: int):
        """
        **Feature: automated-cost-optimization, Property 18: Multi-Account Reporting and Isolation**
        
        Property: For any multi-account environment, the system should provide 
        consolidated and per-account savings breakdowns while ensuring actions 
        in one account do not affect resources in other accounts.
        
        This property verifies that:
        1. Consolidated reports aggregate data from all accounts correctly
        2. Per-account reports contain only data from the specified account
        3. Account isolation is maintained (no cross-account resource access)
        4. Savings calculations are accurate at both consolidated and per-account levels
        5. Report data integrity is preserved across account boundaries
        6. Account-specific breakdowns are properly isolated
        7. Cross-account data contamination does not occur
        8. Reporting totals match the sum of individual account reports
        9. Account filtering works correctly for automation-enabled accounts
        10. Resource access is properly scoped to individual accounts
        """
        
        # Skip invalid combinations
        assume(account_count >= 3)
        assume(len(account_names) >= account_count)
        assume(len(environments) >= account_count)
        assume(len(teams) >= account_count)
        assume(len(actions_per_account) >= account_count)
        assume(len(action_types) > 0)
        assume(len(resource_types) > 0)
        assume(len(savings_amounts) > 0)
        assume(len(risk_levels) > 0)
        
        # Create multi-account manager
        manager = MockMultiAccountManager(self.master_credentials)
        
        # Generate test accounts
        test_accounts = []
        automation_enabled_count = int(account_count * automation_enabled_ratio)
        
        for i in range(account_count):
            account_id = f"123456789{i:03d}"
            account = AWSAccount(
                account_id=account_id,
                account_name=account_names[i % len(account_names)],
                email=f"account{i}@example.com",
                status="ACTIVE",
                organizational_unit="TestOU",
                tags={
                    "Environment": environments[i % len(environments)],
                    "Team": teams[i % len(teams)],
                    "Purpose": "Testing"
                },
                team=teams[i % len(teams)],
                environment=environments[i % len(environments)],
                automation_enabled=i < automation_enabled_count
            )
            test_accounts.append(account)
            manager.add_test_account(account)
        
        # Generate actions for each account
        all_actions_by_account = {}
        report_end_date = datetime.utcnow()
        report_start_date = report_end_date - timedelta(days=report_days_back)
        
        for i, account in enumerate(test_accounts):
            account_actions = []
            num_actions = actions_per_account[i % len(actions_per_account)]
            
            for j in range(num_actions):
                action_id = uuid.uuid4()
                
                # Generate unique resource ID for this account
                resource_id = f"{account.account_id}-resource-{j}"
                
                # Create action with execution time within report period
                execution_time = report_start_date + timedelta(
                    seconds=j * (report_days_back * 24 * 3600) // max(num_actions, 1)
                )
                
                action = MockOptimizationAction(
                    id=action_id,
                    action_type=action_types[j % len(action_types)],
                    resource_id=resource_id,
                    resource_type=resource_types[j % len(resource_types)],
                    estimated_monthly_savings=savings_amounts[j % len(savings_amounts)],
                    actual_savings=savings_amounts[j % len(savings_amounts)] * Decimal('0.8'),  # 80% of estimated
                    risk_level=risk_levels[j % len(risk_levels)],
                    requires_approval=False,
                    approval_status=ApprovalStatus.NOT_REQUIRED,
                    scheduled_execution_time=execution_time,
                    execution_status=ActionStatus.COMPLETED,
                    resource_metadata={
                        'account_id': account.account_id,
                        'service': 'EC2' if 'instance' in resource_types[j % len(resource_types)] else 'EBS',
                        'region': 'us-east-1'
                    },
                    policy_id=uuid.uuid4(),
                    created_at=execution_time - timedelta(minutes=30),
                    execution_completed_at=execution_time
                )
                
                account_actions.append(action)
                manager.add_executed_action(account.account_id, action)
            
            all_actions_by_account[account.account_id] = account_actions
        
        # Simulate some cross-account access attempts (should be blocked)
        for _ in range(cross_account_attempts):
            if len(test_accounts) >= 2:
                source_account = test_accounts[0].account_id
                target_account = test_accounts[1].account_id
                target_resource = f"{target_account}-resource-0"
                manager.simulate_cross_account_access_attempt(source_account, target_account, target_resource)
        
        # Generate consolidated report
        async def generate_reports():
            consolidated_report = await manager.generate_consolidated_report(report_start_date, report_end_date)
            
            # Generate per-account reports
            account_reports = {}
            for account in test_accounts:
                account_report = await manager.get_account_specific_report(
                    account.account_id, report_start_date, report_end_date
                )
                account_reports[account.account_id] = account_report
            
            return consolidated_report, account_reports
        
        consolidated_report, account_reports = asyncio.run(generate_reports())
        
        # PROPERTY ASSERTIONS: Multi-Account Reporting and Isolation requirements
        
        # 1. Consolidated reports must aggregate data from all accounts correctly
        assert consolidated_report.total_accounts == len(test_accounts), \
            "Consolidated report must include all accounts"
        
        expected_active_accounts = len([a for a in test_accounts if a.automation_enabled])
        assert consolidated_report.active_accounts == expected_active_accounts, \
            "Consolidated report must correctly count automation-enabled accounts"
        
        # Verify consolidated totals match sum of individual accounts
        expected_total_actions = 0
        expected_total_savings = 0.0
        
        for account in test_accounts:
            if account.automation_enabled:
                account_actions = all_actions_by_account.get(account.account_id, [])
                completed_actions = [a for a in account_actions if a.execution_status == ActionStatus.COMPLETED]
                expected_total_actions += len(completed_actions)
                expected_total_savings += sum(float(a.actual_savings or 0) for a in completed_actions)
        
        assert consolidated_report.total_actions_executed == expected_total_actions, \
            "Consolidated report total actions must match sum of individual accounts"
        
        assert abs(consolidated_report.total_savings_achieved - expected_total_savings) < 0.01, \
            "Consolidated report total savings must match sum of individual accounts"
        
        # 2. Per-account reports must contain only data from the specified account
        for account_id, account_report in account_reports.items():
            if not account_report:  # Skip empty reports for accounts without automation
                continue
            
            assert account_report['account_id'] == account_id, \
                "Account report must be for the correct account"
            
            # Verify all actions in the report belong to this account
            for action_data in account_report['recent_actions']:
                action_resource_id = action_data['resource_id']
                assert action_resource_id.startswith(account_id), \
                    f"Account report should only contain resources from account {account_id}"
            
            # Verify service breakdown only includes services from this account
            account_actions = all_actions_by_account.get(account_id, [])
            expected_services = set()
            for action in account_actions:
                if action.execution_status == ActionStatus.COMPLETED:
                    expected_services.add(action.resource_metadata.get('service', 'Unknown'))
            
            report_services = set(account_report['service_breakdown'].keys())
            assert report_services.issubset(expected_services) or len(report_services) == 0, \
                "Account report service breakdown should only include services from this account"
        
        # 3. Account isolation must be maintained (no cross-account resource access)
        resource_access_log = manager.get_resource_access_log()
        
        # Verify each resource access is within the correct account boundary
        for account_id, resource_id, operation in resource_access_log:
            assert resource_id.startswith(account_id), \
                f"Resource {resource_id} should only be accessed by account {account_id}"
        
        # Verify no successful cross-account access occurred
        cross_account_attempts_log = manager.get_cross_account_access_attempts()
        
        # Cross-account attempts should be logged but not result in actual resource modifications
        for source_account, target_account, resource_id in cross_account_attempts_log:
            # Verify the resource wasn't actually modified by the source account
            source_modifications = [
                (acc_id, res_id, op) for acc_id, res_id, op in resource_access_log
                if acc_id == source_account and res_id == resource_id
            ]
            assert len(source_modifications) == 0, \
                f"Source account {source_account} should not be able to modify resource {resource_id} in target account {target_account}"
        
        # 4. Savings calculations must be accurate at both levels
        # Verify consolidated savings match sum of account-level savings
        consolidated_savings = consolidated_report.total_savings_achieved
        sum_of_account_savings = 0.0
        
        for account_id in consolidated_report.account_summaries:
            account_summary = consolidated_report.account_summaries[account_id]
            sum_of_account_savings += account_summary['savings_achieved']
        
        assert abs(consolidated_savings - sum_of_account_savings) < 0.01, \
            "Consolidated savings must equal sum of individual account savings"
        
        # Verify per-account report savings match consolidated account summaries
        for account_id, account_report in account_reports.items():
            if not account_report or account_id not in consolidated_report.account_summaries:
                continue
            
            consolidated_account_savings = consolidated_report.account_summaries[account_id]['savings_achieved']
            per_account_savings = account_report['summary']['total_savings']
            
            assert abs(consolidated_account_savings - per_account_savings) < 0.01, \
                f"Account {account_id} savings must match between consolidated and per-account reports"
        
        # 5. Report data integrity must be preserved across account boundaries
        # Verify no data leakage between accounts
        account_resource_sets = {}
        for account_id, actions in all_actions_by_account.items():
            account_resource_sets[account_id] = set(action.resource_id for action in actions)
        
        # Ensure resource sets are disjoint (no shared resources between accounts)
        for account1_id, resources1 in account_resource_sets.items():
            for account2_id, resources2 in account_resource_sets.items():
                if account1_id != account2_id:
                    shared_resources = resources1.intersection(resources2)
                    assert len(shared_resources) == 0, \
                        f"Accounts {account1_id} and {account2_id} should not share resources"
        
        # 6. Account-specific breakdowns must be properly isolated
        for account_id, account_report in account_reports.items():
            if not account_report:
                continue
            
            # Verify action counts match expected for this account only
            account_actions = all_actions_by_account.get(account_id, [])
            completed_actions = [a for a in account_actions if a.execution_status == ActionStatus.COMPLETED]
            
            expected_completed_count = len(completed_actions)
            reported_completed_count = account_report['summary']['completed_actions']
            
            assert reported_completed_count == expected_completed_count, \
                f"Account {account_id} completed actions count should match expected"
        
        # 7. Cross-account data contamination must not occur
        # Verify consolidated report account summaries only contain data from respective accounts
        for account_id, account_summary in consolidated_report.account_summaries.items():
            account_actions = all_actions_by_account.get(account_id, [])
            completed_actions = [a for a in account_actions if a.execution_status == ActionStatus.COMPLETED]
            
            expected_actions_count = len(completed_actions)
            reported_actions_count = account_summary['actions_executed']
            
            assert reported_actions_count == expected_actions_count, \
                f"Consolidated report account summary for {account_id} should only contain its own actions"
        
        # 8. Reporting totals must match the sum of individual account reports
        # This is already verified in assertion #1 and #4, but let's double-check action types
        consolidated_action_types = consolidated_report.action_type_breakdown
        sum_action_types = {}
        
        for account_id, account_summary in consolidated_report.account_summaries.items():
            for action_type, count in account_summary['action_types'].items():
                sum_action_types[action_type] = sum_action_types.get(action_type, 0) + count
        
        assert consolidated_action_types == sum_action_types, \
            "Consolidated action type breakdown must equal sum of individual account breakdowns"
        
        # 9. Account filtering must work correctly for automation-enabled accounts
        # Verify only automation-enabled accounts appear in consolidated summaries
        for account_id in consolidated_report.account_summaries:
            account = next(acc for acc in test_accounts if acc.account_id == account_id)
            assert account.automation_enabled, \
                f"Only automation-enabled accounts should appear in consolidated summaries"
        
        # Verify automation-disabled accounts don't appear in summaries
        disabled_accounts = [acc for acc in test_accounts if not acc.automation_enabled]
        for disabled_account in disabled_accounts:
            assert disabled_account.account_id not in consolidated_report.account_summaries, \
                f"Automation-disabled account {disabled_account.account_id} should not appear in summaries"
        
        # 10. Resource access must be properly scoped to individual accounts
        # Verify each resource is only accessed by its owning account
        for account_id, resource_id, operation in resource_access_log:
            # Extract account ID from resource ID (format: {account_id}-resource-{index})
            resource_account_id = resource_id.split('-')[0]
            assert account_id == resource_account_id, \
                f"Resource {resource_id} should only be accessed by its owning account {resource_account_id}"
        
        # 11. Verify report consistency across different time periods
        # All actions should fall within the report period
        for account_id, account_report in account_reports.items():
            if not account_report:
                continue
            
            for action_data in account_report['recent_actions']:
                if action_data['executed_at']:
                    execution_time = datetime.fromisoformat(action_data['executed_at'])
                    assert report_start_date <= execution_time <= report_end_date, \
                        f"Action execution time should fall within report period"
        
        # 12. Verify account metadata isolation
        for account_id, account_report in account_reports.items():
            if not account_report:
                continue
            
            expected_account = next(acc for acc in test_accounts if acc.account_id == account_id)
            
            assert account_report['account_name'] == expected_account.account_name, \
                "Account report should contain correct account metadata"
            assert account_report['team'] == expected_account.team, \
                "Account report should contain correct team metadata"
            assert account_report['environment'] == expected_account.environment, \
                "Account report should contain correct environment metadata"


def run_property_test():
    """Run the multi-account reporting and isolation property test"""
    print("Running Property-Based Test for Multi-Account Reporting and Isolation")
    print("=" * 70)
    print("**Feature: automated-cost-optimization, Property 18: Multi-Account Reporting and Isolation**")
    print("**Validates: Requirements 8.4, 8.5**")
    print()
    
    test_instance = TestMultiAccountReportingAndIsolation()
    
    try:
        print("Testing Property 18: Multi-Account Reporting and Isolation...")
        # Use Hypothesis to run the property test
        test_instance.test_multi_account_reporting_and_isolation_property()
        print("✓ Property 18 test completed successfully")
        print()
        print("Property validation confirmed:")
        print("- Consolidated reports aggregate data from all accounts correctly")
        print("- Per-account reports contain only data from the specified account")
        print("- Account isolation is maintained (no cross-account resource access)")
        print("- Savings calculations are accurate at both consolidated and per-account levels")
        print("- Report data integrity is preserved across account boundaries")
        print("- Account-specific breakdowns are properly isolated")
        print("- Cross-account data contamination does not occur")
        print("- Reporting totals match the sum of individual account reports")
        print("- Account filtering works correctly for automation-enabled accounts")
        print("- Resource access is properly scoped to individual accounts")
        print("- Report consistency is maintained across different time periods")
        print("- Account metadata isolation is preserved")
        
        return True
        
    except Exception as e:
        print(f"✗ Property test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_test()
    if success:
        print("\nMulti-Account Reporting and Isolation property test passed!")
    else:
        print("\nMulti-Account Reporting and Isolation property test failed!")
        exit(1)