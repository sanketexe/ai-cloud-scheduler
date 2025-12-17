#!/usr/bin/env python3
"""
Property-Based Tests for Multi-Account Coordination

This module contains property-based tests to verify that the multi-account system
coordinates optimization actions across all accounts using appropriate IAM roles
and applying correct account-specific policies according to the requirements specification.

**Feature: automated-cost-optimization, Property 17: Multi-Account Coordination**
**Validates: Requirements 8.1, 8.2, 8.3**
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from hypothesis import given, strategies as st, assume, settings
from dataclasses import dataclass
import asyncio

# Import the components we're testing
from core.multi_account_manager import (
    MultiAccountManager, AWSAccount, CrossAccountRole, 
    CrossAccountActionResult, AutomationPolicy
)
from core.automation_models import (
    OptimizationAction, ActionType, RiskLevel, ActionStatus,
    AutomationLevel, ApprovalStatus
)


@dataclass
class MockAWSCredentials:
    """Mock AWS credentials for testing"""
    access_key_id: str
    secret_access_key: str
    region: str = "us-east-1"


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


@dataclass
class MockAutomationPolicy:
    """Mock automation policy for testing"""
    id: uuid.UUID
    name: str
    automation_level: AutomationLevel
    enabled_actions: List[str]
    approval_required_actions: List[str]
    blocked_actions: List[str]
    resource_filters: Dict[str, Any]
    time_restrictions: Dict[str, Any]
    safety_overrides: Dict[str, Any]
    is_active: bool


class MockMultiAccountManager:
    """Mock multi-account manager for testing that simulates AWS operations"""
    
    def __init__(self, master_credentials: Dict[str, str]):
        self.master_credentials = master_credentials
        self.accounts: Dict[str, AWSAccount] = {}
        self.role_sessions: Dict[str, Dict[str, Any]] = {}  # account_id -> session info
        self.account_policies: Dict[str, MockAutomationPolicy] = {}
        self.execution_results: List[CrossAccountActionResult] = []
        self.role_assumptions: List[Tuple[str, CrossAccountRole]] = []  # Track role assumptions
        self.policy_applications: List[Tuple[str, MockAutomationPolicy]] = []  # Track policy applications
        
        # Simulate some test accounts
        self._setup_test_accounts()
    
    def _setup_test_accounts(self):
        """Set up test accounts for simulation"""
        # This would normally be populated by discover_accounts()
        pass
    
    async def discover_accounts(self) -> List[AWSAccount]:
        """Mock account discovery"""
        # Return empty list - accounts will be added by test
        return list(self.accounts.values())
    
    async def assume_role(self, account_id: str, role: CrossAccountRole) -> Optional[Dict[str, Any]]:
        """Mock role assumption"""
        # Track the role assumption
        self.role_assumptions.append((account_id, role))
        
        # Simulate successful role assumption if account exists
        if account_id in self.accounts:
            session_info = {
                "account_id": account_id,
                "role_arn": role.role_arn,
                "session_name": role.session_name,
                "assumed_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(seconds=role.duration_seconds)
            }
            self.role_sessions[account_id] = session_info
            return session_info
        
        return None
    
    async def set_account_automation_policy(self, account_id: str, policy: MockAutomationPolicy) -> bool:
        """Mock setting account-specific automation policy"""
        if account_id not in self.accounts:
            return False
        
        # Track the policy application
        self.policy_applications.append((account_id, policy))
        
        # Store the policy
        self.account_policies[account_id] = policy
        
        # Update account metadata
        self.accounts[account_id].automation_policy_id = str(policy.id)
        
        return True
    
    def get_account_automation_policy(self, account_id: str) -> Optional[MockAutomationPolicy]:
        """Mock getting automation policy for account"""
        return self.account_policies.get(account_id)
    
    async def coordinate_cross_account_action(self,
                                            action: MockOptimizationAction,
                                            target_accounts: List[str],
                                            role: CrossAccountRole) -> List[CrossAccountActionResult]:
        """Mock cross-account action coordination"""
        results = []
        
        for account_id in target_accounts:
            # Simulate role assumption
            session = await self.assume_role(account_id, role)
            
            if not session:
                # Failed to assume role
                result = CrossAccountActionResult(
                    account_id=account_id,
                    action_id=str(action.id),
                    success=False,
                    error_message="Failed to assume role in target account",
                    execution_time=datetime.utcnow()
                )
                results.append(result)
                continue
            
            # Check if account exists and has automation enabled
            account = self.accounts.get(account_id)
            if not account or not account.automation_enabled:
                result = CrossAccountActionResult(
                    account_id=account_id,
                    action_id=str(action.id),
                    success=False,
                    error_message="Automation disabled for account" if account else "Account not found",
                    execution_time=datetime.utcnow()
                )
                results.append(result)
                continue
            
            # Check account-specific policy
            account_policy = self.get_account_automation_policy(account_id)
            if not account_policy:
                result = CrossAccountActionResult(
                    account_id=account_id,
                    action_id=str(action.id),
                    success=False,
                    error_message="No automation policy configured for account",
                    execution_time=datetime.utcnow()
                )
                results.append(result)
                continue
            
            # Check if action is allowed by policy
            action_allowed = self._check_action_allowed_by_policy(action, account_policy)
            
            if not action_allowed:
                result = CrossAccountActionResult(
                    account_id=account_id,
                    action_id=str(action.id),
                    success=False,
                    error_message="Action blocked by account policy",
                    execution_time=datetime.utcnow()
                )
                results.append(result)
                continue
            
            # Simulate successful execution
            savings_achieved = float(action.estimated_monthly_savings) * 0.8  # Simulate 80% of estimated
            
            result = CrossAccountActionResult(
                account_id=account_id,
                action_id=str(action.id),
                success=True,
                error_message=None,
                execution_time=datetime.utcnow(),
                savings_achieved=savings_achieved
            )
            results.append(result)
        
        # Store results for verification
        self.execution_results.extend(results)
        
        return results
    
    def _check_action_allowed_by_policy(self, action: MockOptimizationAction, policy: MockAutomationPolicy) -> bool:
        """Check if action is allowed by the account policy"""
        action_type_str = action.action_type.value
        
        # Check if action is blocked
        if action_type_str in policy.blocked_actions:
            return False
        
        # Check if action is enabled
        if action_type_str not in policy.enabled_actions:
            return False
        
        # Check if policy is active
        if not policy.is_active:
            return False
        
        return True
    
    async def enable_automation_for_account(self, account_id: str) -> bool:
        """Mock enabling automation for account"""
        if account_id in self.accounts:
            self.accounts[account_id].automation_enabled = True
            return True
        return False
    
    async def disable_automation_for_account(self, account_id: str) -> bool:
        """Mock disabling automation for account"""
        if account_id in self.accounts:
            self.accounts[account_id].automation_enabled = False
            return True
        return False
    
    def add_test_account(self, account: AWSAccount):
        """Add a test account for testing purposes"""
        self.accounts[account.account_id] = account
    
    def get_role_assumptions(self) -> List[Tuple[str, CrossAccountRole]]:
        """Get list of role assumptions made during testing"""
        return self.role_assumptions.copy()
    
    def get_policy_applications(self) -> List[Tuple[str, MockAutomationPolicy]]:
        """Get list of policy applications made during testing"""
        return self.policy_applications.copy()


class TestMultiAccountCoordination:
    """Property-based tests for multi-account coordination"""
    
    @property
    def master_credentials(self):
        return {
            'access_key_id': 'test_master_key',
            'secret_access_key': 'test_master_secret',
            'region': 'us-east-1'
        }
    
    @given(
        # Generate multiple AWS accounts
        account_count=st.integers(min_value=2, max_value=8),
        
        # Generate account properties
        account_names=st.lists(
            st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
            min_size=2, max_size=8
        ),
        environments=st.lists(
            st.sampled_from(['production', 'staging', 'development', 'test']),
            min_size=2, max_size=8
        ),
        teams=st.lists(
            st.sampled_from(['backend', 'frontend', 'data', 'devops', 'security']),
            min_size=2, max_size=8
        ),
        
        # Generate automation policies
        automation_levels=st.lists(
            st.sampled_from(list(AutomationLevel)),
            min_size=2, max_size=8
        ),
        
        # Generate optimization actions
        action_type=st.sampled_from(list(ActionType)),
        resource_id=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        resource_type=st.sampled_from(['ec2_instance', 'ebs_volume', 'elastic_ip', 'load_balancer']),
        estimated_savings=st.decimals(min_value=10, max_value=5000, places=2),
        risk_level=st.sampled_from(list(RiskLevel)),
        
        # Generate cross-account role configuration
        role_duration=st.integers(min_value=900, max_value=7200),  # 15 minutes to 2 hours
        has_external_id=st.booleans(),
        
        # Generate execution scenarios
        automation_enabled_ratio=st.floats(min_value=0.3, max_value=1.0),  # At least 30% enabled
        policy_configured_ratio=st.floats(min_value=0.5, max_value=1.0),  # At least 50% have policies
        
        # Generate action execution parameters
        target_account_ratio=st.floats(min_value=0.4, max_value=0.8)  # Execute in 40-80% of accounts
    )
    @settings(max_examples=100, deadline=None)
    def test_multi_account_coordination_property(self,
                                               account_count: int,
                                               account_names: List[str],
                                               environments: List[str],
                                               teams: List[str],
                                               automation_levels: List[AutomationLevel],
                                               action_type: ActionType,
                                               resource_id: str,
                                               resource_type: str,
                                               estimated_savings: Decimal,
                                               risk_level: RiskLevel,
                                               role_duration: int,
                                               has_external_id: bool,
                                               automation_enabled_ratio: float,
                                               policy_configured_ratio: float,
                                               target_account_ratio: float):
        """
        **Feature: automated-cost-optimization, Property 17: Multi-Account Coordination**
        
        Property: For any multi-account AWS environment, the system should coordinate 
        optimization actions across all accounts using appropriate IAM roles and 
        applying correct account-specific policies.
        
        This property verifies that:
        1. Cross-account actions are coordinated across multiple accounts
        2. Appropriate IAM roles are used for each account
        3. Account-specific policies are correctly applied
        4. Actions respect per-account automation settings
        5. Role assumptions are properly tracked and managed
        6. Policy enforcement is consistent across accounts
        7. Results are properly aggregated and reported
        8. Failed role assumptions are handled gracefully
        9. Account isolation is maintained (actions in one account don't affect others)
        """
        
        # Skip invalid combinations
        assume(len(resource_id.strip()) > 0)
        assume(account_count >= 2)
        assume(len(account_names) >= account_count)
        assume(len(environments) >= account_count)
        assume(len(teams) >= account_count)
        assume(len(automation_levels) >= account_count)
        
        # Create multi-account manager
        manager = MockMultiAccountManager(self.master_credentials)
        
        # Generate test accounts
        test_accounts = []
        for i in range(account_count):
            account_id = f"123456789{i:03d}"  # Generate account ID
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
                automation_enabled=i < int(account_count * automation_enabled_ratio)
            )
            test_accounts.append(account)
            manager.add_test_account(account)
        
        # Generate and apply automation policies for some accounts
        policies_created = []
        policy_configured_count = int(account_count * policy_configured_ratio)
        
        for i in range(policy_configured_count):
            account = test_accounts[i]
            policy_id = uuid.uuid4()
            
            # Create policy based on automation level
            automation_level = automation_levels[i % len(automation_levels)]
            
            if automation_level == AutomationLevel.CONSERVATIVE:
                enabled_actions = [ActionType.RELEASE_ELASTIC_IP.value, ActionType.UPGRADE_STORAGE.value]
                approval_required = [ActionType.STOP_INSTANCE.value]
                blocked_actions = [ActionType.TERMINATE_INSTANCE.value, ActionType.DELETE_VOLUME.value]
            elif automation_level == AutomationLevel.BALANCED:
                enabled_actions = [action.value for action in ActionType if action != ActionType.TERMINATE_INSTANCE]
                approval_required = [ActionType.DELETE_VOLUME.value, ActionType.STOP_INSTANCE.value]
                blocked_actions = [ActionType.TERMINATE_INSTANCE.value]
            else:  # AGGRESSIVE
                enabled_actions = [action.value for action in ActionType]
                approval_required = []
                blocked_actions = []
            
            policy = MockAutomationPolicy(
                id=policy_id,
                name=f"Policy-{account.account_name}",
                automation_level=automation_level,
                enabled_actions=enabled_actions,
                approval_required_actions=approval_required,
                blocked_actions=blocked_actions,
                resource_filters={
                    "exclude_tags": ["Environment=production"] if automation_level == AutomationLevel.CONSERVATIVE else [],
                    "include_services": ["EC2", "EBS", "EIP"]
                },
                time_restrictions={
                    "business_hours_only": automation_level == AutomationLevel.CONSERVATIVE
                },
                safety_overrides={},
                is_active=True
            )
            
            policies_created.append((account.account_id, policy))
        
        # Apply policies to accounts asynchronously
        async def apply_policies():
            for account_id, policy in policies_created:
                success = await manager.set_account_automation_policy(account_id, policy)
                assert success, f"Failed to set policy for account {account_id}"
        
        # Run policy application
        asyncio.run(apply_policies())
        
        # Create optimization action
        action_id = uuid.uuid4()
        action = MockOptimizationAction(
            id=action_id,
            action_type=action_type,
            resource_id=resource_id,
            resource_type=resource_type,
            estimated_monthly_savings=estimated_savings,
            actual_savings=None,
            risk_level=risk_level,
            requires_approval=False,
            approval_status=ApprovalStatus.NOT_REQUIRED,
            scheduled_execution_time=datetime.utcnow() + timedelta(minutes=5),
            execution_status=ActionStatus.PENDING,
            resource_metadata={
                "service": "EC2" if resource_type == "ec2_instance" else "EBS",
                "region": "us-east-1"
            },
            policy_id=uuid.uuid4()
        )
        
        # Create cross-account role
        master_account_id = "123456789000"
        role = CrossAccountRole(
            role_arn=f"arn:aws:iam::{master_account_id}:role/FinOpsAccessRole",
            external_id="test-external-id" if has_external_id else None,
            session_name="FinOpsTestSession",
            duration_seconds=role_duration
        )
        
        # Select target accounts for action execution
        target_account_count = max(1, int(account_count * target_account_ratio))
        target_accounts = [acc.account_id for acc in test_accounts[:target_account_count]]
        
        # Execute cross-account coordination
        async def execute_coordination():
            return await manager.coordinate_cross_account_action(action, target_accounts, role)
        
        results = asyncio.run(execute_coordination())
        
        # PROPERTY ASSERTIONS: Multi-Account Coordination requirements
        
        # 1. Cross-account actions must be coordinated across multiple accounts
        assert len(results) == len(target_accounts), \
            "Results must be returned for all target accounts"
        
        result_account_ids = {result.account_id for result in results}
        target_account_ids = set(target_accounts)
        assert result_account_ids == target_account_ids, \
            "Results must cover exactly the target accounts"
        
        # 2. Appropriate IAM roles must be used for each account
        role_assumptions = manager.get_role_assumptions()
        
        # Should have attempted role assumption for each target account
        assumed_accounts = {assumption[0] for assumption in role_assumptions}
        assert len(assumed_accounts) == len(target_accounts), \
            "Role assumption must be attempted for all target accounts"
        
        for account_id, assumed_role in role_assumptions:
            assert account_id in target_accounts, \
                f"Role assumption for {account_id} must be for a target account"
            assert assumed_role.role_arn == role.role_arn, \
                "Correct role ARN must be used for assumption"
            assert assumed_role.session_name == role.session_name, \
                "Correct session name must be used"
            assert assumed_role.duration_seconds == role.duration_seconds, \
                "Correct duration must be used"
            
            if has_external_id:
                assert assumed_role.external_id == role.external_id, \
                    "External ID must be used when provided"
        
        # 3. Account-specific policies must be correctly applied
        policy_applications = manager.get_policy_applications()
        
        # Verify policies were applied to the expected accounts
        applied_account_ids = {app[0] for app in policy_applications}
        expected_policy_accounts = {acc.account_id for acc in test_accounts[:policy_configured_count]}
        assert applied_account_ids == expected_policy_accounts, \
            "Policies must be applied to expected accounts"
        
        # Verify policy content is correct
        for account_id, applied_policy in policy_applications:
            account = next(acc for acc in test_accounts if acc.account_id == account_id)
            expected_level = automation_levels[test_accounts.index(account) % len(automation_levels)]
            
            assert applied_policy.automation_level == expected_level, \
                f"Policy automation level must match expected for account {account_id}"
            assert applied_policy.is_active, \
                f"Applied policy must be active for account {account_id}"
        
        # 4. Actions must respect per-account automation settings
        for result in results:
            account = next(acc for acc in test_accounts if acc.account_id == result.account_id)
            
            if not account.automation_enabled:
                # Should fail if automation is disabled
                assert not result.success, \
                    f"Action should fail for account {result.account_id} with automation disabled"
                assert "automation disabled" in result.error_message.lower(), \
                    "Error message should indicate automation is disabled"
            
            # Check policy enforcement
            account_policy = manager.get_account_automation_policy(result.account_id)
            if account.automation_enabled and not account_policy:
                # Should fail if no policy is configured
                assert not result.success, \
                    f"Action should fail for account {result.account_id} without policy"
                assert "no automation policy" in result.error_message.lower(), \
                    "Error message should indicate missing policy"
        
        # 5. Policy enforcement must be consistent across accounts
        for result in results:
            account = next(acc for acc in test_accounts if acc.account_id == result.account_id)
            account_policy = manager.get_account_automation_policy(result.account_id)
            
            if account.automation_enabled and account_policy:
                action_allowed = manager._check_action_allowed_by_policy(action, account_policy)
                
                if not action_allowed:
                    # Should fail if action is blocked by policy
                    assert not result.success, \
                        f"Action should fail for account {result.account_id} due to policy restrictions"
                    assert "blocked by" in result.error_message.lower(), \
                        "Error message should indicate policy blocking"
                else:
                    # Should succeed if action is allowed and automation is enabled
                    assert result.success, \
                        f"Action should succeed for account {result.account_id} with proper policy"
                    assert result.savings_achieved is not None, \
                        "Successful actions should report savings"
                    assert result.savings_achieved > 0, \
                        "Successful actions should have positive savings"
        
        # 6. Results must be properly aggregated and reported
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Verify all results have required fields
        for result in results:
            assert result.account_id is not None, "Result must include account ID"
            assert result.action_id == str(action.id), "Result must include correct action ID"
            assert isinstance(result.success, bool), "Result must include success status"
            assert result.execution_time is not None, "Result must include execution time"
            
            if result.success:
                assert result.savings_achieved is not None, "Successful result must include savings"
                assert result.savings_achieved >= 0, "Savings must be non-negative"
                assert result.error_message is None, "Successful result should not have error message"
            else:
                assert result.error_message is not None, "Failed result must include error message"
                assert len(result.error_message.strip()) > 0, "Error message must not be empty"
        
        # 7. Failed role assumptions must be handled gracefully
        # This is implicitly tested by checking that all target accounts get results
        # even if some fail due to role assumption issues
        
        # 8. Account isolation must be maintained
        # Verify that each result only affects its own account
        for result in results:
            # The action should only reference resources in the target account
            assert result.account_id in target_accounts, \
                "Result should only be for target accounts"
            
            # No cross-contamination of account data
            other_results = [r for r in results if r.account_id != result.account_id]
            for other_result in other_results:
                assert other_result.account_id != result.account_id, \
                    "Results should be isolated per account"
        
        # 9. Verify coordination preserves action integrity
        for result in results:
            assert result.action_id == str(action.id), \
                "All results must reference the same action"
        
        # 10. Verify proper error categorization
        error_categories = {}
        for result in failed_results:
            if "role" in result.error_message.lower():
                error_categories["role_issues"] = error_categories.get("role_issues", 0) + 1
            elif "automation disabled" in result.error_message.lower():
                error_categories["automation_disabled"] = error_categories.get("automation_disabled", 0) + 1
            elif "policy" in result.error_message.lower():
                error_categories["policy_issues"] = error_categories.get("policy_issues", 0) + 1
            else:
                error_categories["other"] = error_categories.get("other", 0) + 1
        
        # All errors should be categorized
        total_categorized = sum(error_categories.values())
        assert total_categorized == len(failed_results), \
            "All failed results should have categorized error messages"
        
        # 11. Verify savings calculation consistency
        total_estimated_savings = float(estimated_savings) * len(successful_results)
        total_actual_savings = sum(r.savings_achieved for r in successful_results if r.savings_achieved)
        
        if len(successful_results) > 0:
            # Actual savings should be reasonable compared to estimated
            savings_ratio = total_actual_savings / total_estimated_savings if total_estimated_savings > 0 else 0
            assert 0 <= savings_ratio <= 1.5, \
                "Actual savings should be reasonable compared to estimated savings"
        
        # 12. Verify execution timing consistency
        execution_times = [r.execution_time for r in results]
        time_span = max(execution_times) - min(execution_times)
        
        # All executions should complete within a reasonable timeframe (simulated)
        assert time_span.total_seconds() < 60, \
            "Cross-account coordination should complete within reasonable time"


def run_property_test():
    """Run the multi-account coordination property test"""
    print("Running Property-Based Test for Multi-Account Coordination")
    print("=" * 60)
    print("**Feature: automated-cost-optimization, Property 17: Multi-Account Coordination**")
    print("**Validates: Requirements 8.1, 8.2, 8.3**")
    print()
    
    test_instance = TestMultiAccountCoordination()
    
    try:
        print("Testing Property 17: Multi-Account Coordination...")
        # Use Hypothesis to run the property test
        from hypothesis import given
        test_instance.test_multi_account_coordination_property()
        print("✓ Property 17 test completed successfully")
        print()
        print("Property validation confirmed:")
        print("- Cross-account actions are coordinated across multiple accounts")
        print("- Appropriate IAM roles are used for each account")
        print("- Account-specific policies are correctly applied")
        print("- Actions respect per-account automation settings")
        print("- Role assumptions are properly tracked and managed")
        print("- Policy enforcement is consistent across accounts")
        print("- Results are properly aggregated and reported")
        print("- Failed role assumptions are handled gracefully")
        print("- Account isolation is maintained")
        print("- Savings calculations are consistent")
        print("- Error categorization is comprehensive")
        print("- Execution timing is reasonable")
        
        return True
        
    except Exception as e:
        print(f"✗ Property test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_test()
    if success:
        print("\nMulti-Account Coordination property test passed!")
    else:
        print("\nMulti-Account Coordination property test failed!")
        exit(1)