"""
Enterprise Cloud Provider Integration Layer
Handles multi-account AWS environments for large companies
"""

import json
import boto3
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
import asyncio
from concurrent.futures import ThreadPoolExecutor
import structlog

from .models import CloudProvider, CostData, ProviderType
from .encryption import EncryptionService

logger = structlog.get_logger(__name__)

class CloudProviderAdapter(ABC):
    """Abstract base class for cloud provider adapters"""
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if credentials and connection are valid"""
        pass
    
    @abstractmethod
    async def get_accounts(self) -> List[Dict[str, Any]]:
        """Get all accounts accessible with these credentials"""
        pass
    
    @abstractmethod
    async def get_cost_and_usage(self, 
                                start_date: date, 
                                end_date: date,
                                accounts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get cost and usage data for specified date range"""
        pass
    
    @abstractmethod
    async def get_resources(self, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover resources in the account(s)"""
        pass
    
    @abstractmethod
    async def get_rightsizing_recommendations(self) -> List[Dict[str, Any]]:
        """Get rightsizing recommendations"""
        pass

class AWSCredentials:
    """AWS credentials container"""
    
    def __init__(self, 
                 access_key_id: str,
                 secret_access_key: str,
                 region: str = 'us-east-1',
                 role_arn: Optional[str] = None,
                 external_id: Optional[str] = None):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.role_arn = role_arn
        self.external_id = external_id
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for boto3"""
        creds = {
            'aws_access_key_id': self.access_key_id,
            'aws_secret_access_key': self.secret_access_key,
            'region_name': self.region
        }
        return creds
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'AWSCredentials':
        """Create from dictionary"""
        return cls(
            access_key_id=data['access_key_id'],
            secret_access_key=data['secret_access_key'],
            region=data.get('region', 'us-east-1'),
            role_arn=data.get('role_arn'),
            external_id=data.get('external_id')
        )

class AWSCostExplorerAdapter(CloudProviderAdapter):
    """Enterprise AWS Cost Explorer integration for multi-account environments"""
    
    def __init__(self, credentials: AWSCredentials):
        self.credentials = credentials
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize AWS clients
        self._cost_explorer_client = None
        self._organizations_client = None
        self._ec2_client = None
        self._cloudwatch_client = None
        self._sts_client = None
    
    def _get_cost_explorer_client(self):
        """Get Cost Explorer client (lazy initialization)"""
        if not self._cost_explorer_client:
            session = boto3.Session(**self.credentials.to_dict())
            self._cost_explorer_client = session.client('ce')
        return self._cost_explorer_client
    
    def _get_organizations_client(self):
        """Get Organizations client for multi-account management"""
        if not self._organizations_client:
            session = boto3.Session(**self.credentials.to_dict())
            self._organizations_client = session.client('organizations')
        return self._organizations_client
    
    def _get_ec2_client(self):
        """Get EC2 client for resource discovery"""
        if not self._ec2_client:
            session = boto3.Session(**self.credentials.to_dict())
            self._ec2_client = session.client('ec2')
        return self._ec2_client
    
    def _get_sts_client(self):
        """Get STS client for account information"""
        if not self._sts_client:
            session = boto3.Session(**self.credentials.to_dict())
            self._sts_client = session.client('sts')
        return self._sts_client
    
    async def test_connection(self) -> bool:
        """Test AWS credentials and permissions"""
        try:
            def _test():
                # Test basic AWS access
                sts = self._get_sts_client()
                identity = sts.get_caller_identity()
                
                # Test Cost Explorer access
                ce = self._get_cost_explorer_client()
                end_date = date.today()
                start_date = end_date - timedelta(days=1)
                
                ce.get_cost_and_usage(
                    TimePeriod={
                        'Start': start_date.strftime('%Y-%m-%d'),
                        'End': end_date.strftime('%Y-%m-%d')
                    },
                    Granularity='DAILY',
                    Metrics=['BlendedCost']
                )
                
                return True, identity
            
            loop = asyncio.get_event_loop()
            success, identity = await loop.run_in_executor(self.executor, _test)
            
            logger.info("AWS connection test successful", 
                       account_id=identity.get('Account'),
                       user_id=identity.get('UserId'))
            return success
            
        except Exception as e:
            logger.error("AWS connection test failed", error=str(e))
            return False
    
    async def get_accounts(self) -> List[Dict[str, Any]]:
        """Get all AWS accounts in the organization"""
        try:
            def _get_accounts():
                accounts = []
                
                # Try to get organization accounts (if master account)
                try:
                    org_client = self._get_organizations_client()
                    paginator = org_client.get_paginator('list_accounts')
                    
                    for page in paginator.paginate():
                        for account in page['Accounts']:
                            accounts.append({
                                'account_id': account['Id'],
                                'account_name': account['Name'],
                                'email': account['Email'],
                                'status': account['Status'],
                                'joined_method': account.get('JoinedMethod', 'UNKNOWN')
                            })
                except Exception as org_error:
                    logger.warning("Could not access AWS Organizations", error=str(org_error))
                    
                    # Fallback: get current account only
                    sts = self._get_sts_client()
                    identity = sts.get_caller_identity()
                    accounts.append({
                        'account_id': identity['Account'],
                        'account_name': f"Account {identity['Account']}",
                        'email': 'unknown@company.com',
                        'status': 'ACTIVE',
                        'joined_method': 'SINGLE_ACCOUNT'
                    })
                
                return accounts
            
            loop = asyncio.get_event_loop()
            accounts = await loop.run_in_executor(self.executor, _get_accounts)
            
            logger.info("Retrieved AWS accounts", count=len(accounts))
            return accounts
            
        except Exception as e:
            logger.error("Failed to get AWS accounts", error=str(e))
            return []
    
    async def get_cost_and_usage(self, 
                                start_date: date, 
                                end_date: date,
                                accounts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get comprehensive cost and usage data for enterprise environments"""
        try:
            def _get_cost_data():
                ce_client = self._get_cost_explorer_client()
                cost_data = []
                
                # Base parameters
                time_period = {
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                }
                
                # Get cost by service and account
                response = ce_client.get_cost_and_usage(
                    TimePeriod=time_period,
                    Granularity='DAILY',
                    Metrics=['BlendedCost', 'UsageQuantity'],
                    GroupBy=[
                        {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                        {'Type': 'DIMENSION', 'Key': 'LINKED_ACCOUNT'}
                    ],
                    Filter={
                        'Dimensions': {
                            'Key': 'LINKED_ACCOUNT',
                            'Values': accounts if accounts else []
                        }
                    } if accounts else None
                )
                
                # Process results
                for result in response['ResultsByTime']:
                    result_date = datetime.strptime(result['TimePeriod']['Start'], '%Y-%m-%d').date()
                    
                    for group in result['Groups']:
                        service_name = group['Keys'][0] if group['Keys'][0] else 'Unknown'
                        account_id = group['Keys'][1] if len(group['Keys']) > 1 else 'Unknown'
                        
                        # Extract cost and usage
                        blended_cost = group['Metrics']['BlendedCost']
                        usage_quantity = group['Metrics'].get('UsageQuantity', {})
                        
                        cost_amount = Decimal(blended_cost['Amount']) if blended_cost['Amount'] else Decimal('0')
                        usage_amount = Decimal(usage_quantity.get('Amount', '0')) if usage_quantity.get('Amount') else Decimal('0')
                        
                        if cost_amount > 0:  # Only include non-zero costs
                            cost_data.append({
                                'date': result_date,
                                'account_id': account_id,
                                'service_name': service_name,
                                'cost_amount': cost_amount,
                                'currency': blended_cost['Unit'],
                                'usage_quantity': usage_amount,
                                'usage_unit': usage_quantity.get('Unit', ''),
                                'cost_type': 'BlendedCost'
                            })
                
                # Get additional cost breakdown by tags (for team attribution)
                try:
                    tag_response = ce_client.get_cost_and_usage(
                        TimePeriod=time_period,
                        Granularity='DAILY',
                        Metrics=['BlendedCost'],
                        GroupBy=[
                            {'Type': 'TAG', 'Key': 'Team'},
                            {'Type': 'TAG', 'Key': 'Project'},
                            {'Type': 'TAG', 'Key': 'Environment'}
                        ]
                    )
                    
                    # Process tag-based costs
                    for result in tag_response['ResultsByTime']:
                        result_date = datetime.strptime(result['TimePeriod']['Start'], '%Y-%m-%d').date()
                        
                        for group in result['Groups']:
                            if group['Keys'] and any(key for key in group['Keys'] if key):
                                team = group['Keys'][0] if group['Keys'][0] else 'Untagged'
                                project = group['Keys'][1] if len(group['Keys']) > 1 and group['Keys'][1] else 'Untagged'
                                environment = group['Keys'][2] if len(group['Keys']) > 2 and group['Keys'][2] else 'Untagged'
                                
                                blended_cost = group['Metrics']['BlendedCost']
                                cost_amount = Decimal(blended_cost['Amount']) if blended_cost['Amount'] else Decimal('0')
                                
                                if cost_amount > 0:
                                    # Find matching cost data and add tags
                                    for cost_item in cost_data:
                                        if cost_item['date'] == result_date:
                                            if 'tags' not in cost_item:
                                                cost_item['tags'] = {}
                                            cost_item['tags'].update({
                                                'Team': team,
                                                'Project': project,
                                                'Environment': environment
                                            })
                
                except Exception as tag_error:
                    logger.warning("Could not retrieve tag-based cost data", error=str(tag_error))
                
                return cost_data
            
            loop = asyncio.get_event_loop()
            cost_data = await loop.run_in_executor(self.executor, _get_cost_data)
            
            logger.info("Retrieved AWS cost data", 
                       start_date=start_date.isoformat(),
                       end_date=end_date.isoformat(),
                       records=len(cost_data))
            
            return cost_data
            
        except Exception as e:
            logger.error("Failed to get AWS cost data", error=str(e))
            return []
    
    async def get_resources(self, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover AWS resources across accounts"""
        try:
            def _get_resources():
                resources = []
                ec2_client = self._get_ec2_client()
                
                # Get EC2 instances
                try:
                    paginator = ec2_client.get_paginator('describe_instances')
                    for page in paginator.paginate():
                        for reservation in page['Reservations']:
                            for instance in reservation['Instances']:
                                # Extract tags
                                tags = {}
                                if 'Tags' in instance:
                                    tags = {tag['Key']: tag['Value'] for tag in instance['Tags']}
                                
                                resources.append({
                                    'resource_id': instance['InstanceId'],
                                    'resource_type': 'EC2Instance',
                                    'service_name': 'Amazon Elastic Compute Cloud - Compute',
                                    'region': instance['Placement']['AvailabilityZone'][:-1],
                                    'state': instance['State']['Name'],
                                    'instance_type': instance['InstanceType'],
                                    'launch_time': instance['LaunchTime'],
                                    'tags': tags,
                                    'metadata': {
                                        'vpc_id': instance.get('VpcId'),
                                        'subnet_id': instance.get('SubnetId'),
                                        'security_groups': [sg['GroupId'] for sg in instance.get('SecurityGroups', [])],
                                        'platform': instance.get('Platform', 'linux')
                                    }
                                })
                except Exception as ec2_error:
                    logger.warning("Could not retrieve EC2 instances", error=str(ec2_error))
                
                # Get RDS instances
                try:
                    rds_client = boto3.Session(**self.credentials.to_dict()).client('rds')
                    paginator = rds_client.get_paginator('describe_db_instances')
                    
                    for page in paginator.paginate():
                        for db_instance in page['DBInstances']:
                            # Get tags
                            tags = {}
                            try:
                                tag_response = rds_client.list_tags_for_resource(
                                    ResourceName=db_instance['DBInstanceArn']
                                )
                                tags = {tag['Key']: tag['Value'] for tag in tag_response['TagList']}
                            except:
                                pass
                            
                            resources.append({
                                'resource_id': db_instance['DBInstanceIdentifier'],
                                'resource_type': 'RDSInstance',
                                'service_name': 'Amazon Relational Database Service',
                                'region': db_instance['AvailabilityZone'][:-1] if db_instance.get('AvailabilityZone') else 'unknown',
                                'state': db_instance['DBInstanceStatus'],
                                'instance_type': db_instance['DBInstanceClass'],
                                'launch_time': db_instance['InstanceCreateTime'],
                                'tags': tags,
                                'metadata': {
                                    'engine': db_instance['Engine'],
                                    'engine_version': db_instance['EngineVersion'],
                                    'allocated_storage': db_instance['AllocatedStorage'],
                                    'multi_az': db_instance['MultiAZ']
                                }
                            })
                except Exception as rds_error:
                    logger.warning("Could not retrieve RDS instances", error=str(rds_error))
                
                return resources
            
            loop = asyncio.get_event_loop()
            resources = await loop.run_in_executor(self.executor, _get_resources)
            
            logger.info("Discovered AWS resources", count=len(resources))
            return resources
            
        except Exception as e:
            logger.error("Failed to discover AWS resources", error=str(e))
            return []
    
    async def get_rightsizing_recommendations(self) -> List[Dict[str, Any]]:
        """Get AWS rightsizing recommendations"""
        try:
            def _get_recommendations():
                ce_client = self._get_cost_explorer_client()
                recommendations = []
                
                try:
                    response = ce_client.get_rightsizing_recommendation(
                        Service='AmazonEC2',
                        Configuration={
                            'BenefitsConsidered': True,
                            'RecommendationTarget': 'SAME_INSTANCE_FAMILY'
                        }
                    )
                    
                    for rec in response.get('RightsizingRecommendations', []):
                        current_instance = rec.get('CurrentInstance', {})
                        recommendations.append({
                            'resource_id': current_instance.get('ResourceId', ''),
                            'resource_type': 'EC2Instance',
                            'recommendation_type': 'rightsizing',
                            'current_instance_type': current_instance.get('InstanceType', ''),
                            'recommended_instance_type': rec.get('RightsizingType', ''),
                            'current_cost': Decimal(str(current_instance.get('MonthlyCost', '0'))),
                            'recommended_cost': Decimal(str(rec.get('TargetInstances', [{}])[0].get('EstimatedMonthlyCost', '0'))),
                            'potential_savings': Decimal(str(current_instance.get('MonthlyCost', '0'))) - Decimal(str(rec.get('TargetInstances', [{}])[0].get('EstimatedMonthlyCost', '0'))),
                            'confidence': 'HIGH',
                            'details': rec
                        })
                
                except Exception as rec_error:
                    logger.warning("Could not retrieve rightsizing recommendations", error=str(rec_error))
                
                return recommendations
            
            loop = asyncio.get_event_loop()
            recommendations = await loop.run_in_executor(self.executor, _get_recommendations)
            
            logger.info("Retrieved AWS rightsizing recommendations", count=len(recommendations))
            return recommendations
            
        except Exception as e:
            logger.error("Failed to get AWS rightsizing recommendations", error=str(e))
            return []

class CloudProviderService:
    """Enterprise cloud provider management service"""
    
    def __init__(self, encryption_service: EncryptionService):
        self.encryption_service = encryption_service
        self.adapters: Dict[UUID, CloudProviderAdapter] = {}
    
    def _create_adapter(self, provider: CloudProvider) -> CloudProviderAdapter:
        """Create appropriate adapter for provider type"""
        # Decrypt credentials
        credentials_data = self.encryption_service.decrypt(provider.credentials_encrypted)
        credentials_dict = json.loads(credentials_data)
        
        if provider.provider_type == ProviderType.AWS:
            aws_creds = AWSCredentials.from_dict(credentials_dict)
            return AWSCostExplorerAdapter(aws_creds)
        else:
            raise ValueError(f"Unsupported provider type: {provider.provider_type}")
    
    async def register_provider(self, 
                               name: str,
                               provider_type: ProviderType,
                               credentials: Dict[str, str],
                               created_by: UUID) -> CloudProvider:
        """Register a new cloud provider"""
        # Test credentials first
        if provider_type == ProviderType.AWS:
            aws_creds = AWSCredentials.from_dict(credentials)
            adapter = AWSCostExplorerAdapter(aws_creds)
            
            if not await adapter.test_connection():
                raise ValueError("Invalid AWS credentials or insufficient permissions")
        
        # Encrypt credentials
        credentials_json = json.dumps(credentials)
        encrypted_credentials = self.encryption_service.encrypt(credentials_json)
        
        # Create provider record (this would use repository in real implementation)
        provider = CloudProvider(
            name=name,
            provider_type=provider_type,
            credentials_encrypted=encrypted_credentials,
            created_by=created_by
        )
        
        # Cache adapter
        adapter = self._create_adapter(provider)
        self.adapters[provider.id] = adapter
        
        logger.info("Cloud provider registered", 
                   provider_id=provider.id,
                   provider_type=provider_type.value,
                   name=name)
        
        return provider
    
    async def get_adapter(self, provider_id: UUID) -> CloudProviderAdapter:
        """Get adapter for provider"""
        if provider_id not in self.adapters:
            # Load from database and create adapter
            # This would use repository in real implementation
            pass
        
        return self.adapters.get(provider_id)
    
    async def sync_cost_data(self, 
                            provider_id: UUID,
                            start_date: date,
                            end_date: date) -> List[CostData]:
        """Sync cost data from cloud provider"""
        adapter = await self.get_adapter(provider_id)
        if not adapter:
            raise ValueError(f"No adapter found for provider {provider_id}")
        
        # Get cost data from provider
        raw_cost_data = await adapter.get_cost_and_usage(start_date, end_date)
        
        # Convert to CostData models
        cost_data_records = []
        for item in raw_cost_data:
            cost_record = CostData(
                provider_id=provider_id,
                resource_id=item.get('resource_id', f"{item['service_name']}-{item['account_id']}"),
                resource_type=item.get('resource_type', 'Service'),
                service_name=item['service_name'],
                cost_amount=item['cost_amount'],
                currency=item['currency'],
                cost_date=item['date'],
                usage_quantity=item.get('usage_quantity'),
                usage_unit=item.get('usage_unit'),
                tags=item.get('tags', {}),
                metadata={
                    'account_id': item.get('account_id'),
                    'cost_type': item.get('cost_type'),
                    'region': item.get('region')
                }
            )
            cost_data_records.append(cost_record)
        
        logger.info("Synced cost data", 
                   provider_id=provider_id,
                   records=len(cost_data_records),
                   date_range=f"{start_date} to {end_date}")
        
        return cost_data_records