"""
Multi-Account Manager for Enterprise AWS Environments

Handles cross-account access using IAM roles, manages multiple team accounts,
and provides centralized cost visibility across the organization.
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
