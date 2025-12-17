"""
Multi-Account Manager for Enterprise AWS Environments

Handles cross-account access using IAM roles, manages multiple team accounts,
and provides centralized cost visibility across the organization.

Extended for Automated Cost Optimization:
- Cross-account action coordination
- Account-specific policy application
- Consolidated and per-account reporting
"""

import boto3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import structlog

from .automation_models import (
    OptimizationAction, AutomationPolicy, ActionType, ActionStatus,
    RiskLevel, ApprovalStatus
)

logger = structlog.get_logger(__name__)


@dataclass
class AWSAccount:
    """Represents an AWS account in the organization"""
    account_id: str
    account_name: str
    email: str
    status: str
    organizational_unit: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    team: Optional[str] = None
    cost_center: Optional[str] = None
    environment: Optional[str] = None  # prod, dev, staging
    automation_policy_id: Optional[str] = None  # Account-specific automation policy
    automation_enabled: bool = True  # Whether automation is enabled for this account


@dataclass
class CrossAccountActionResult:
    """Result of a cross-account automation action"""
    account_id: str
    action_id: str
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None
    savings_achieved: Optional[float] = None


@dataclass
class MultiAccountReport:
    """Consolidated reporting across multiple accounts"""
    report_period_start: datetime
    report_period_end: datetime
    total_accounts: int
    active_accounts: int
    total_actions_executed: int
    total_savings_achieved: float
    account_summaries: Dict[str, Dict[str, Any]]
    action_type_breakdown: Dict[str, int]
    risk_level_breakdown: Dict[str, int]


@dataclass
class CrossAccountRole:
    """Configuration for cross-account IAM role"""
    role_arn: str
    external_id: Optional[str] = None
    session_name: str = "FinOpsSession"
    duration_seconds: int = 3600


class MultiAccountManager:
    """
    Manages multiple AWS accounts for enterprise FinOps.
    Supports AWS Organizations and cross-account role assumption.
    
    Extended for Automated Cost Optimization:
    - Cross-account action coordination
    - Account-specific policy application
    - Consolidated and per-account reporting
    """
    
    def __init__(self, master_credentials: Dict[str, str]):
        """
        Initialize with master account credentials.
        
        Args:
            master_credentials: Dict with access_key_id, secret_access_key, region
        """
        self.master_credentials = master_credentials
        self.accounts: Dict[str, AWSAccount] = {}
        self.role_sessions: Dict[str, boto3.Session] = {}
        self.account_policies: Dict[str, AutomationPolicy] = {}  # Account-specific policies
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize AWS clients
        self._master_session = None
        self._organizations_client = None
        self._sts_client = None
    
    def _get_master_session(self) -> boto3.Session:
        """Get master account session"""
        if not self._master_session:
            self._master_session = boto3.Session(
                aws_access_key_id=self.master_credentials['access_key_id'],
                aws_secret_access_key=self.master_credentials['secret_access_key'],
                region_name=self.master_credentials.get('region', 'us-east-1')
            )
        return self._master_session
    
    def _get_organizations_client(self):
        """Get AWS Organizations client"""
        if not self._organizations_client:
            session = self._get_master_session()
            self._organizations_client = session.client('organizations')
        return self._organizations_client
    
    def _get_sts_client(self):
        """Get STS client for role assumption"""
        if not self._sts_client:
            session = self._get_master_session()
            self._sts_client = session.client('sts')
        return self._sts_client
    
    async def discover_accounts(self) -> List[AWSAccount]:
        """
        Discover all accounts in the AWS Organization.
        
        Returns:
            List of AWS accounts
        """
        try:
            def _discover():
                org_client = self._get_organizations_client()
                accounts = []
                
                # Get all accounts
                paginator = org_client.get_paginator('list_accounts')
                for page in paginator.paginate():
                    for account_data in page['Accounts']:
                        # Get account tags
                        try:
                            tags_response = org_client.list_tags_for_resource(
                                ResourceId=account_data['Id']
                            )
                            tags = {tag['Key']: tag['Value'] for tag in tags_response.get('Tags', [])}
                        except:
                            tags = {}
                        
                        # Get organizational unit
                        try:
                            parents = org_client.list_parents(ChildId=account_data['Id'])
                            ou_id = parents['Parents'][0]['Id'] if parents['Parents'] else None
                            
                            if ou_id and ou_id.startswith('ou-'):
                                ou_info = org_client.describe_organizational_unit(
                                    OrganizationalUnitId=ou_id
                                )
                                ou_name = ou_info['OrganizationalUnit']['Name']
                            else:
                                ou_name = 'Root'
                        except:
                            ou_name = 'Unknown'
                        
                        account = AWSAccount(
                            account_id=account_data['Id'],
                            account_name=account_data['Name'],
                            email=account_data['Email'],
                            status=account_data['Status'],
                            organizational_unit=ou_name,
                            tags=tags,
                            team=tags.get('Team'),
                            cost_center=tags.get('CostCenter'),
                            environment=tags.get('Environment')
                        )
                        accounts.append(account)
                        self.accounts[account.account_id] = account
                
                return accounts
            
            loop = asyncio.get_event_loop()
            accounts = await loop.run_in_executor(self.executor, _discover)
            
            logger.info("Discovered AWS accounts",
                       count=len(accounts),
                       active=len([a for a in accounts if a.status == 'ACTIVE']))
            
            return accounts
            
        except Exception as e:
            logger.error("Failed to discover accounts", error=str(e))
            return []
    
    async def assume_role(self, 
                         account_id: str, 
                         role: CrossAccountRole) -> Optional[boto3.Session]:
        """
        Assume a role in a member account.
        
        Args:
            account_id: Target account ID
            role: Cross-account role configuration
            
        Returns:
            boto3.Session with assumed role credentials or None
        """
        try:
            def _assume():
                sts_client = self._get_sts_client()
                
                assume_role_params = {
                    'RoleArn': role.role_arn,
                    'RoleSessionName': role.session_name,
                    'DurationSeconds': role.duration_seconds
                }
                
                if role.external_id:
                    assume_role_params['ExternalId'] = role.external_id
                
                response = sts_client.assume_role(**assume_role_params)
                
                credentials = response['Credentials']
                
                # Create session with temporary credentials
                session = boto3.Session(
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken'],
                    region_name=self.master_credentials.get('region', 'us-east-1')
                )
                
                return session
            
            loop = asyncio.get_event_loop()
            session = await loop.run_in_executor(self.executor, _assume)
            
            # Cache the session
            self.role_sessions[account_id] = session
            
            logger.info("Assumed role in account",
                       account_id=account_id,
                       role_arn=role.role_arn)
            
            return session
            
        except Exception as e:
            logger.error("Failed to assume role",
                        account_id=account_id,
                        role_arn=role.role_arn,
                        error=str(e))
            return None
    
    async def get_account_cost_data(self,
                                   account_id: str,
                                   start_date: datetime,
                                   end_date: datetime,
                                   role: Optional[CrossAccountRole] = None) -> List[Dict[str, Any]]:
        """
        Get cost data for a specific account.
        
        Args:
            account_id: Account ID
            start_date: Start date for cost data
            end_date: End date for cost data
            role: Optional cross-account role (if not using master account)
            
        Returns:
            List of cost data records
        """
        try:
            def _get_costs():
                # Use assumed role session if provided, otherwise use master
                if role:
                    session = self.role_sessions.get(account_id)
                    if not session:
                        # Need to assume role first
                        return []
                else:
                    session = self._get_master_session()
                
                ce_client = session.client('ce')
                
                response = ce_client.get_cost_and_usage(
                    TimePeriod={
                        'Start': start_date.strftime('%Y-%m-%d'),
                        'End': end_date.strftime('%Y-%m-%d')
                    },
                    Granularity='DAILY',
                    Metrics=['BlendedCost', 'UsageQuantity'],
                    GroupBy=[
                        {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                    ],
                    Filter={
                        'Dimensions': {
                            'Key': 'LINKED_ACCOUNT',
                            'Values': [account_id]
                        }
                    }
                )
                
                cost_data = []
                for result in response['ResultsByTime']:
                    result_date = datetime.strptime(result['TimePeriod']['Start'], '%Y-%m-%d')
                    
                    for group in result['Groups']:
                        service_name = group['Keys'][0] if group['Keys'] else 'Unknown'
                        cost_amount = float(group['Metrics']['BlendedCost']['Amount'])
                        
                        if cost_amount > 0:
                            cost_data.append({
                                'account_id': account_id,
                                'date': result_date,
                                'service': service_name,
                                'cost': cost_amount,
                                'currency': 'USD'
                            })
                
                return cost_data
            
            loop = asyncio.get_event_loop()
            cost_data = await loop.run_in_executor(self.executor, _get_costs)
            
            logger.info("Retrieved cost data for account",
                       account_id=account_id,
                       records=len(cost_data))
            
            return cost_data
            
        except Exception as e:
            logger.error("Failed to get account cost data",
                        account_id=account_id,
                        error=str(e))
            return []
    
    async def get_all_accounts_cost_data(self,
                                        start_date: datetime,
                                        end_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get cost data for all accounts in parallel.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict mapping account_id to cost data
        """
        tasks = []
        for account_id in self.accounts.keys():
            task = self.get_account_cost_data(account_id, start_date, end_date)
            tasks.append((account_id, task))
        
        results = {}
        for account_id, task in tasks:
            cost_data = await task
            results[account_id] = cost_data
        
        total_records = sum(len(data) for data in results.values())
        logger.info("Retrieved cost data for all accounts",
                   accounts=len(results),
                   total_records=total_records)
        
        return results
    
    def get_accounts_by_team(self, team: str) -> List[AWSAccount]:
        """Get all accounts belonging to a specific team"""
        return [
            account for account in self.accounts.values()
            if account.team == team
        ]
    
    def get_accounts_by_cost_center(self, cost_center: str) -> List[AWSAccount]:
        """Get all accounts belonging to a specific cost center"""
        return [
            account for account in self.accounts.values()
            if account.cost_center == cost_center
        ]
    
    def get_accounts_by_environment(self, environment: str) -> List[AWSAccount]:
        """Get all accounts in a specific environment"""
        return [
            account for account in self.accounts.values()
            if account.environment == environment
        ]
    
    async def tag_account(self, account_id: str, tags: Dict[str, str]) -> bool:
        """
        Add tags to an AWS account.
        
        Args:
            account_id: Account ID
            tags: Tags to add
            
        Returns:
            True if successful
        """
        try:
            def _tag():
                org_client = self._get_organizations_client()
                
                tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
                
                org_client.tag_resource(
                    ResourceId=account_id,
                    Tags=tag_list
                )
                
                return True
            
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(self.executor, _tag)
            
            # Update local cache
            if account_id in self.accounts:
                self.accounts[account_id].tags.update(tags)
                if 'Team' in tags:
                    self.accounts[account_id].team = tags['Team']
                if 'CostCenter' in tags:
                    self.accounts[account_id].cost_center = tags['CostCenter']
                if 'Environment' in tags:
                    self.accounts[account_id].environment = tags['Environment']
            
            logger.info("Tagged account",
                       account_id=account_id,
                       tags=tags)
            
            return success
            
        except Exception as e:
            logger.error("Failed to tag account",
                        account_id=account_id,
                        error=str(e))
            return False
    
    def generate_cross_account_role_template(self, 
                                            master_account_id: str,
                                            external_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate CloudFormation template for cross-account IAM role.
        
        Args:
            master_account_id: Master/management account ID
            external_id: Optional external ID for additional security
            
        Returns:
            CloudFormation template as dict
        """
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": f"arn:aws:iam::{master_account_id}:root"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        if external_id:
            trust_policy["Statement"][0]["Condition"] = {
                "StringEquals": {
                    "sts:ExternalId": external_id
                }
            }
        
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "FinOps Cross-Account Access Role",
            "Resources": {
                "FinOpsRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "RoleName": "FinOpsAccessRole",
                        "AssumeRolePolicyDocument": trust_policy,
                        "ManagedPolicyArns": [
                            "arn:aws:iam::aws:policy/ReadOnlyAccess",
                            "arn:aws:iam::aws:policy/AWSSupportAccess"
                        ],
                        "Policies": [
                            {
                                "PolicyName": "FinOpsCostExplorerAccess",
                                "PolicyDocument": {
                                    "Version": "2012-10-17",
                                    "Statement": [
                                        {
                                            "Effect": "Allow",
                                            "Action": [
                                                "ce:*",
                                                "cur:*",
                                                "pricing:*",
                                                "budgets:*"
                                            ],
                                            "Resource": "*"
                                        }
                                    ]
                                }
                            }
                        ],
                        "Tags": [
                            {"Key": "Purpose", "Value": "FinOps"},
                            {"Key": "ManagedBy", "Value": "FinOpsPlatform"}
                        ]
                    }
                }
            },
            "Outputs": {
                "RoleArn": {
                    "Description": "ARN of the FinOps access role",
                    "Value": {"Fn::GetAtt": ["FinOpsRole", "Arn"]}
                }
            }
        }
        
        return template

    async def set_account_automation_policy(self, 
                                          account_id: str, 
                                          policy: AutomationPolicy) -> bool:
        """
        Set account-specific automation policy.
        
        Args:
            account_id: Target account ID
            policy: Automation policy to apply
            
        Returns:
            True if successful
        """
        try:
            if account_id not in self.accounts:
                logger.error("Account not found", account_id=account_id)
                return False
            
            # Store the policy for this account
            self.account_policies[account_id] = policy
            
            # Update account metadata
            self.accounts[account_id].automation_policy_id = str(policy.id)
            
            logger.info("Set automation policy for account",
                       account_id=account_id,
                       policy_id=str(policy.id),
                       automation_level=policy.automation_level.value)
            
            return True
            
        except Exception as e:
            logger.error("Failed to set account automation policy",
                        account_id=account_id,
                        error=str(e))
            return False

    def get_account_automation_policy(self, account_id: str) -> Optional[AutomationPolicy]:
        """
        Get automation policy for a specific account.
        
        Args:
            account_id: Account ID
            
        Returns:
            AutomationPolicy or None if not set
        """
        return self.account_policies.get(account_id)

    async def coordinate_cross_account_action(self,
                                            action: OptimizationAction,
                                            target_accounts: List[str],
                                            role: CrossAccountRole) -> List[CrossAccountActionResult]:
        """
        Coordinate execution of an optimization action across multiple accounts.
        
        Args:
            action: Optimization action to execute
            target_accounts: List of account IDs to execute in
            role: Cross-account role configuration
            
        Returns:
            List of execution results per account
        """
        logger.info("Starting cross-account action coordination",
                   action_id=str(action.id),
                   action_type=action.action_type.value,
                   target_accounts=target_accounts)
        
        results = []
        
        # Execute action in each account concurrently
        tasks = []
        for account_id in target_accounts:
            task = self._execute_action_in_account(action, account_id, role)
            tasks.append((account_id, task))
        
        # Wait for all executions to complete
        for account_id, task in tasks:
            try:
                result = await task
                results.append(result)
                
                logger.info("Cross-account action completed",
                           account_id=account_id,
                           action_id=str(action.id),
                           success=result.success)
                           
            except Exception as e:
                error_result = CrossAccountActionResult(
                    account_id=account_id,
                    action_id=str(action.id),
                    success=False,
                    error_message=str(e),
                    execution_time=datetime.utcnow()
                )
                results.append(error_result)
                
                logger.error("Cross-account action failed",
                           account_id=account_id,
                           action_id=str(action.id),
                           error=str(e))
        
        # Log summary
        successful = len([r for r in results if r.success])
        total_savings = sum(r.savings_achieved or 0 for r in results if r.success)
        
        logger.info("Cross-account action coordination completed",
                   action_id=str(action.id),
                   total_accounts=len(target_accounts),
                   successful_accounts=successful,
                   total_savings=total_savings)
        
        return results

    async def _execute_action_in_account(self,
                                       action: OptimizationAction,
                                       account_id: str,
                                       role: CrossAccountRole) -> CrossAccountActionResult:
        """
        Execute an optimization action in a specific account.
        
        Args:
            action: Optimization action to execute
            account_id: Target account ID
            role: Cross-account role configuration
            
        Returns:
            CrossAccountActionResult
        """
        start_time = datetime.utcnow()
        
        try:
            # Assume role in target account
            session = await self.assume_role(account_id, role)
            if not session:
                return CrossAccountActionResult(
                    account_id=account_id,
                    action_id=str(action.id),
                    success=False,
                    error_message="Failed to assume role in target account",
                    execution_time=start_time
                )
            
            # Get account-specific policy
            account_policy = self.get_account_automation_policy(account_id)
            if not account_policy:
                return CrossAccountActionResult(
                    account_id=account_id,
                    action_id=str(action.id),
                    success=False,
                    error_message="No automation policy configured for account",
                    execution_time=start_time
                )
            
            # Check if account has automation enabled
            account = self.accounts.get(account_id)
            if not account or not account.automation_enabled:
                return CrossAccountActionResult(
                    account_id=account_id,
                    action_id=str(action.id),
                    success=False,
                    error_message="Automation disabled for account",
                    execution_time=start_time
                )
            
            # Execute the action using the assumed role session
            success, execution_details = await self._execute_action_with_session(
                action, session, account_policy
            )
            
            # Calculate savings (simplified - would integrate with actual cost calculation)
            savings_achieved = float(action.estimated_monthly_savings) if success else 0.0
            
            return CrossAccountActionResult(
                account_id=account_id,
                action_id=str(action.id),
                success=success,
                error_message=execution_details.get('error') if not success else None,
                execution_time=datetime.utcnow(),
                savings_achieved=savings_achieved
            )
            
        except Exception as e:
            return CrossAccountActionResult(
                account_id=account_id,
                action_id=str(action.id),
                success=False,
                error_message=str(e),
                execution_time=datetime.utcnow()
            )

    async def _execute_action_with_session(self,
                                         action: OptimizationAction,
                                         session: boto3.Session,
                                         policy: AutomationPolicy) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute an optimization action using a specific AWS session.
        
        Args:
            action: Optimization action to execute
            session: AWS session with appropriate credentials
            policy: Account-specific automation policy
            
        Returns:
            Tuple of (success, execution_details)
        """
        try:
            # Import here to avoid circular imports
            from .action_engine import ActionEngine
            from .safety_checker import SafetyChecker
            
            # Create action engine with the cross-account session
            action_engine = ActionEngine(aws_session=session)
            safety_checker = SafetyChecker()
            
            # Perform safety checks with account-specific policy
            safety_passed, safety_details = safety_checker.validate_action_safety(action, policy)
            
            if not safety_passed:
                return False, {
                    'error': 'Safety checks failed',
                    'safety_details': safety_details
                }
            
            # Execute the action
            success, execution_details = action_engine.execute_action(action)
            
            return success, execution_details
            
        except Exception as e:
            return False, {'error': str(e)}

    async def generate_consolidated_report(self,
                                         start_date: datetime,
                                         end_date: datetime) -> MultiAccountReport:
        """
        Generate consolidated automation report across all accounts.
        
        Args:
            start_date: Report period start
            end_date: Report period end
            
        Returns:
            MultiAccountReport with consolidated data
        """
        logger.info("Generating consolidated multi-account report",
                   start_date=start_date.isoformat(),
                   end_date=end_date.isoformat())
        
        try:
            # Import here to avoid circular imports
            from .database import get_db_session
            from sqlalchemy import and_, func
            
            account_summaries = {}
            total_actions = 0
            total_savings = 0.0
            action_type_counts = {}
            risk_level_counts = {}
            
            with get_db_session() as db:
                # Get actions for each account in the date range
                for account_id, account in self.accounts.items():
                    if not account.automation_enabled:
                        continue
                    
                    # Query actions for this account in the date range
                    actions = db.query(OptimizationAction).filter(
                        and_(
                            OptimizationAction.execution_completed_at >= start_date,
                            OptimizationAction.execution_completed_at <= end_date,
                            OptimizationAction.execution_status == ActionStatus.COMPLETED,
                            OptimizationAction.resource_metadata['account_id'].astext == account_id
                        )
                    ).all()
                    
                    # Calculate account-level metrics
                    account_actions = len(actions)
                    account_savings = sum(float(action.actual_savings or 0) for action in actions)
                    
                    # Count by action type and risk level
                    account_action_types = {}
                    account_risk_levels = {}
                    
                    for action in actions:
                        # Action type counts
                        action_type = action.action_type.value
                        account_action_types[action_type] = account_action_types.get(action_type, 0) + 1
                        action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
                        
                        # Risk level counts
                        risk_level = action.risk_level.value
                        account_risk_levels[risk_level] = account_risk_levels.get(risk_level, 0) + 1
                        risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1
                    
                    # Store account summary
                    account_summaries[account_id] = {
                        'account_name': account.account_name,
                        'team': account.team,
                        'environment': account.environment,
                        'actions_executed': account_actions,
                        'savings_achieved': account_savings,
                        'action_types': account_action_types,
                        'risk_levels': account_risk_levels,
                        'automation_policy_id': account.automation_policy_id
                    }
                    
                    total_actions += account_actions
                    total_savings += account_savings
            
            # Create consolidated report
            report = MultiAccountReport(
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
            
            logger.info("Generated consolidated report",
                       total_accounts=report.total_accounts,
                       active_accounts=report.active_accounts,
                       total_actions=total_actions,
                       total_savings=total_savings)
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate consolidated report", error=str(e))
            # Return empty report on error
            return MultiAccountReport(
                report_period_start=start_date,
                report_period_end=end_date,
                total_accounts=0,
                active_accounts=0,
                total_actions_executed=0,
                total_savings_achieved=0.0,
                account_summaries={},
                action_type_breakdown={},
                risk_level_breakdown={}
            )

    async def get_account_specific_report(self,
                                        account_id: str,
                                        start_date: datetime,
                                        end_date: datetime) -> Dict[str, Any]:
        """
        Generate detailed report for a specific account.
        
        Args:
            account_id: Target account ID
            start_date: Report period start
            end_date: Report period end
            
        Returns:
            Detailed account report
        """
        try:
            if account_id not in self.accounts:
                logger.error("Account not found for report", account_id=account_id)
                return {}
            
            account = self.accounts[account_id]
            
            # Import here to avoid circular imports
            from .database import get_db_session
            from sqlalchemy import and_
            
            with get_db_session() as db:
                # Get all actions for this account
                actions = db.query(OptimizationAction).filter(
                    and_(
                        OptimizationAction.execution_completed_at >= start_date,
                        OptimizationAction.execution_completed_at <= end_date,
                        OptimizationAction.resource_metadata['account_id'].astext == account_id
                    )
                ).all()
                
                # Calculate detailed metrics
                completed_actions = [a for a in actions if a.execution_status == ActionStatus.COMPLETED]
                failed_actions = [a for a in actions if a.execution_status == ActionStatus.FAILED]
                
                total_savings = sum(float(action.actual_savings or 0) for action in completed_actions)
                
                # Group by service/resource type
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
                
                # Create detailed report
                report = {
                    'account_id': account_id,
                    'account_name': account.account_name,
                    'team': account.team,
                    'environment': account.environment,
                    'report_period': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'summary': {
                        'total_actions': len(actions),
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
                        for action in sorted(actions, key=lambda x: x.created_at, reverse=True)[:10]
                    ]
                }
                
                logger.info("Generated account-specific report",
                           account_id=account_id,
                           total_actions=len(actions),
                           total_savings=total_savings)
                
                return report
                
        except Exception as e:
            logger.error("Failed to generate account-specific report",
                        account_id=account_id,
                        error=str(e))
            return {}

    async def enable_automation_for_account(self, account_id: str) -> bool:
        """
        Enable automation for a specific account.
        
        Args:
            account_id: Account ID
            
        Returns:
            True if successful
        """
        try:
            if account_id not in self.accounts:
                logger.error("Account not found", account_id=account_id)
                return False
            
            self.accounts[account_id].automation_enabled = True
            
            logger.info("Enabled automation for account", account_id=account_id)
            return True
            
        except Exception as e:
            logger.error("Failed to enable automation for account",
                        account_id=account_id,
                        error=str(e))
            return False

    async def disable_automation_for_account(self, account_id: str) -> bool:
        """
        Disable automation for a specific account.
        
        Args:
            account_id: Account ID
            
        Returns:
            True if successful
        """
        try:
            if account_id not in self.accounts:
                logger.error("Account not found", account_id=account_id)
                return False
            
            self.accounts[account_id].automation_enabled = False
            
            logger.info("Disabled automation for account", account_id=account_id)
            return True
            
        except Exception as e:
            logger.error("Failed to disable automation for account",
                        account_id=account_id,
                        error=str(e))
            return False

    def get_automation_enabled_accounts(self) -> List[AWSAccount]:
        """
        Get all accounts with automation enabled.
        
        Returns:
            List of accounts with automation enabled
        """
        return [
            account for account in self.accounts.values()
            if account.automation_enabled
        ]

    async def validate_cross_account_permissions(self, 
                                               account_id: str,
                                               role: CrossAccountRole) -> Dict[str, bool]:
        """
        Validate that cross-account permissions are properly configured.
        
        Args:
            account_id: Target account ID
            role: Cross-account role configuration
            
        Returns:
            Dict of permission checks and their results
        """
        try:
            session = await self.assume_role(account_id, role)
            if not session:
                return {
                    'assume_role': False,
                    'ec2_access': False,
                    'cost_explorer_access': False,
                    'iam_read_access': False
                }
            
            checks = {}
            
            # Test EC2 access
            try:
                ec2 = session.client('ec2')
                ec2.describe_instances(MaxResults=1)
                checks['ec2_access'] = True
            except:
                checks['ec2_access'] = False
            
            # Test Cost Explorer access
            try:
                ce = session.client('ce')
                ce.get_cost_and_usage(
                    TimePeriod={
                        'Start': (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d'),
                        'End': datetime.utcnow().strftime('%Y-%m-%d')
                    },
                    Granularity='DAILY',
                    Metrics=['BlendedCost']
                )
                checks['cost_explorer_access'] = True
            except:
                checks['cost_explorer_access'] = False
            
            # Test IAM read access
            try:
                iam = session.client('iam')
                iam.list_roles(MaxItems=1)
                checks['iam_read_access'] = True
            except:
                checks['iam_read_access'] = False
            
            checks['assume_role'] = True
            
            logger.info("Validated cross-account permissions",
                       account_id=account_id,
                       checks=checks)
            
            return checks
            
        except Exception as e:
            logger.error("Failed to validate cross-account permissions",
                        account_id=account_id,
                        error=str(e))
            return {
                'assume_role': False,
                'ec2_access': False,
                'cost_explorer_access': False,
                'iam_read_access': False
            }


# Helper function to create manager instance
def create_multi_account_manager(master_credentials: Dict[str, str]) -> MultiAccountManager:
    """
    Create a multi-account manager instance.
    
    Args:
        master_credentials: Master account credentials
        
    Returns:
        MultiAccountManager instance
    """
    return MultiAccountManager(master_credentials)
