# cloud_api_integration.py
"""
Cloud Provider API Integration Layer

This module provides unified interfaces for interacting with different cloud provider APIs
including billing, resource management, and monitoring services.
"""

import time
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from urllib.parse import urljoin
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from google.cloud import billing_v1, compute_v1, monitoring_v3
from google.oauth2 import service_account
from azure.identity import ClientSecretCredential
from azure.mgmt.consumption import ConsumptionManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.monitor import MonitorManagementClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIServiceType(Enum):
    """Types of API services"""
    BILLING = "billing"
    RESOURCES = "resources"
    MONITORING = "monitoring"
    PRICING = "pricing"


class RetryStrategy(Enum):
    """Retry strategies for API calls"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"


@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    response_time_ms: float = 0
    retry_count: int = 0
    
    def is_success(self) -> bool:
        return self.success and self.error_message is None


@dataclass
class RetryConfig:
    """Configuration for API retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * attempt
        else:  # FIXED_INTERVAL
            delay = self.base_delay
        
        return min(delay, self.max_delay)


class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 service_type: Optional[APIServiceType] = None):
        super().__init__(message)
        self.status_code = status_code
        self.service_type = service_type


class BaseCloudAPIAdapter(ABC):
    """Abstract base class for cloud provider API adapters"""
    
    def __init__(self, credentials: Dict[str, str], retry_config: RetryConfig = None):
        self.credentials = credentials
        self.retry_config = retry_config or RetryConfig()
        self._session = None
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the cloud provider"""
        pass
    
    @abstractmethod
    def test_connection(self) -> APIResponse:
        """Test API connection"""
        pass
    
    @abstractmethod
    def get_billing_data(self, start_date: datetime, end_date: datetime) -> APIResponse:
        """Get billing data for date range"""
        pass
    
    @abstractmethod
    def list_resources(self, resource_type: str = None) -> APIResponse:
        """List cloud resources"""
        pass
    
    @abstractmethod
    def get_resource_metrics(self, resource_id: str, metric_name: str,
                           start_time: datetime, end_time: datetime) -> APIResponse:
        """Get metrics for a specific resource"""
        pass
    
    def _make_request_with_retry(self, request_func, *args, **kwargs) -> APIResponse:
        """Make API request with retry logic"""
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                response = request_func(*args, **kwargs)
                response_time = (time.time() - start_time) * 1000
                
                return APIResponse(
                    success=True,
                    data=response,
                    response_time_ms=response_time,
                    retry_count=attempt
                )
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.calculate_delay(attempt)
                    logger.warning(f"API request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"API request failed after {attempt + 1} attempts: {e}")
        
        response_time = (time.time() - start_time) * 1000
        return APIResponse(
            success=False,
            error_message=str(last_exception),
            response_time_ms=response_time,
            retry_count=self.retry_config.max_retries
        )


class AWSAPIAdapter(BaseCloudAPIAdapter):
    """AWS API adapter for billing, resources, and monitoring"""
    
    def __init__(self, credentials: Dict[str, str], region: str = "us-east-1",
                 retry_config: RetryConfig = None):
        super().__init__(credentials, retry_config)
        self.region = region
        self._ce_client = None
        self._ec2_client = None
        self._cloudwatch_client = None
    
    def authenticate(self) -> bool:
        """Authenticate with AWS"""
        try:
            # Initialize AWS clients
            session = boto3.Session(
                aws_access_key_id=self.credentials.get("access_key_id"),
                aws_secret_access_key=self.credentials.get("secret_access_key"),
                aws_session_token=self.credentials.get("session_token"),
                region_name=self.region
            )
            
            self._ce_client = session.client('ce')  # Cost Explorer
            self._ec2_client = session.client('ec2')
            self._cloudwatch_client = session.client('cloudwatch')
            
            return True
            
        except Exception as e:
            logger.error(f"AWS authentication failed: {e}")
            return False
    
    def test_connection(self) -> APIResponse:
        """Test AWS API connection"""
        if not self.authenticate():
            return APIResponse(success=False, error_message="Authentication failed")
        
        def _test_request():
            # Test with a simple EC2 describe regions call
            return self._ec2_client.describe_regions()
        
        return self._make_request_with_retry(_test_request)
    
    def get_billing_data(self, start_date: datetime, end_date: datetime) -> APIResponse:
        """Get AWS billing data using Cost Explorer"""
        if not self._ce_client:
            if not self.authenticate():
                return APIResponse(success=False, error_message="Authentication failed")
        
        def _get_costs():
            response = self._ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost', 'UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'LINKED_ACCOUNT'}
                ]
            )
            return response
        
        return self._make_request_with_retry(_get_costs)
    
    def list_resources(self, resource_type: str = None) -> APIResponse:
        """List AWS resources"""
        if not self._ec2_client:
            if not self.authenticate():
                return APIResponse(success=False, error_message="Authentication failed")
        
        def _list_instances():
            if resource_type == "instances" or resource_type is None:
                return self._ec2_client.describe_instances()
            elif resource_type == "volumes":
                return self._ec2_client.describe_volumes()
            elif resource_type == "snapshots":
                return self._ec2_client.describe_snapshots(OwnerIds=['self'])
            else:
                raise ValueError(f"Unsupported resource type: {resource_type}")
        
        return self._make_request_with_retry(_list_instances)
    
    def get_resource_metrics(self, resource_id: str, metric_name: str,
                           start_time: datetime, end_time: datetime) -> APIResponse:
        """Get CloudWatch metrics for AWS resource"""
        if not self._cloudwatch_client:
            if not self.authenticate():
                return APIResponse(success=False, error_message="Authentication failed")
        
        def _get_metrics():
            response = self._cloudwatch_client.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName=metric_name,
                Dimensions=[
                    {
                        'Name': 'InstanceId',
                        'Value': resource_id
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=['Average', 'Maximum']
            )
            return response
        
        return self._make_request_with_retry(_get_metrics)


class GCPAPIAdapter(BaseCloudAPIAdapter):
    """GCP API adapter for billing, resources, and monitoring"""
    
    def __init__(self, credentials: Dict[str, str], project_id: str = None,
                 retry_config: RetryConfig = None):
        super().__init__(credentials, retry_config)
        self.project_id = project_id or credentials.get("project_id")
        self._billing_client = None
        self._compute_client = None
        self._monitoring_client = None
        self._credentials_obj = None
    
    def authenticate(self) -> bool:
        """Authenticate with GCP"""
        try:
            # Parse service account key
            service_account_info = json.loads(self.credentials.get("service_account_key", "{}"))
            
            self._credentials_obj = service_account.Credentials.from_service_account_info(
                service_account_info
            )
            
            # Initialize GCP clients
            self._billing_client = billing_v1.CloudBillingClient(credentials=self._credentials_obj)
            self._compute_client = compute_v1.InstancesClient(credentials=self._credentials_obj)
            self._monitoring_client = monitoring_v3.MetricServiceClient(credentials=self._credentials_obj)
            
            return True
            
        except Exception as e:
            logger.error(f"GCP authentication failed: {e}")
            return False
    
    def test_connection(self) -> APIResponse:
        """Test GCP API connection"""
        if not self.authenticate():
            return APIResponse(success=False, error_message="Authentication failed")
        
        def _test_request():
            # Test with a simple compute zones list
            request = compute_v1.ListZonesRequest(project=self.project_id)
            zones = list(self._compute_client.list(request=request))
            return {"zones_count": len(zones)}
        
        return self._make_request_with_retry(_test_request)
    
    def get_billing_data(self, start_date: datetime, end_date: datetime) -> APIResponse:
        """Get GCP billing data"""
        if not self._billing_client:
            if not self.authenticate():
                return APIResponse(success=False, error_message="Authentication failed")
        
        def _get_billing():
            # Note: This is a simplified example. Real implementation would use BigQuery
            # to query billing export data
            billing_accounts = list(self._billing_client.list_billing_accounts())
            return {"billing_accounts": len(billing_accounts), "message": "Use BigQuery for detailed billing data"}
        
        return self._make_request_with_retry(_get_billing)
    
    def list_resources(self, resource_type: str = None) -> APIResponse:
        """List GCP resources"""
        if not self._compute_client:
            if not self.authenticate():
                return APIResponse(success=False, error_message="Authentication failed")
        
        def _list_instances():
            if resource_type == "instances" or resource_type is None:
                request = compute_v1.AggregatedListInstancesRequest(project=self.project_id)
                instances = self._compute_client.aggregated_list(request=request)
                return {"instances": dict(instances)}
            else:
                raise ValueError(f"Unsupported resource type: {resource_type}")
        
        return self._make_request_with_retry(_list_instances)
    
    def get_resource_metrics(self, resource_id: str, metric_name: str,
                           start_time: datetime, end_time: datetime) -> APIResponse:
        """Get GCP monitoring metrics"""
        if not self._monitoring_client:
            if not self.authenticate():
                return APIResponse(success=False, error_message="Authentication failed")
        
        def _get_metrics():
            project_name = f"projects/{self.project_id}"
            
            interval = monitoring_v3.TimeInterval({
                "end_time": {"seconds": int(end_time.timestamp())},
                "start_time": {"seconds": int(start_time.timestamp())},
            })
            
            request = monitoring_v3.ListTimeSeriesRequest({
                "name": project_name,
                "filter": f'metric.type="compute.googleapis.com/instance/{metric_name}"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            })
            
            results = self._monitoring_client.list_time_series(request=request)
            return {"time_series": list(results)}
        
        return self._make_request_with_retry(_get_metrics)


class AzureAPIAdapter(BaseCloudAPIAdapter):
    """Azure API adapter for billing, resources, and monitoring"""
    
    def __init__(self, credentials: Dict[str, str], subscription_id: str = None,
                 retry_config: RetryConfig = None):
        super().__init__(credentials, retry_config)
        self.subscription_id = subscription_id or credentials.get("subscription_id")
        self._credential_obj = None
        self._consumption_client = None
        self._resource_client = None
        self._monitor_client = None
    
    def authenticate(self) -> bool:
        """Authenticate with Azure"""
        try:
            self._credential_obj = ClientSecretCredential(
                tenant_id=self.credentials.get("tenant_id"),
                client_id=self.credentials.get("client_id"),
                client_secret=self.credentials.get("client_secret")
            )
            
            # Initialize Azure clients
            self._consumption_client = ConsumptionManagementClient(
                self._credential_obj, self.subscription_id
            )
            self._resource_client = ResourceManagementClient(
                self._credential_obj, self.subscription_id
            )
            self._monitor_client = MonitorManagementClient(
                self._credential_obj, self.subscription_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Azure authentication failed: {e}")
            return False
    
    def test_connection(self) -> APIResponse:
        """Test Azure API connection"""
        if not self.authenticate():
            return APIResponse(success=False, error_message="Authentication failed")
        
        def _test_request():
            # Test with resource groups list
            resource_groups = list(self._resource_client.resource_groups.list())
            return {"resource_groups_count": len(resource_groups)}
        
        return self._make_request_with_retry(_test_request)
    
    def get_billing_data(self, start_date: datetime, end_date: datetime) -> APIResponse:
        """Get Azure billing data"""
        if not self._consumption_client:
            if not self.authenticate():
                return APIResponse(success=False, error_message="Authentication failed")
        
        def _get_usage():
            usage_details = self._consumption_client.usage_details.list(
                scope=f"/subscriptions/{self.subscription_id}",
                filter=f"properties/usageStart ge '{start_date.isoformat()}' and properties/usageEnd le '{end_date.isoformat()}'"
            )
            return {"usage_details": list(usage_details)}
        
        return self._make_request_with_retry(_get_usage)
    
    def list_resources(self, resource_type: str = None) -> APIResponse:
        """List Azure resources"""
        if not self._resource_client:
            if not self.authenticate():
                return APIResponse(success=False, error_message="Authentication failed")
        
        def _list_resources():
            if resource_type:
                resources = self._resource_client.resources.list(
                    filter=f"resourceType eq '{resource_type}'"
                )
            else:
                resources = self._resource_client.resources.list()
            
            return {"resources": list(resources)}
        
        return self._make_request_with_retry(_list_resources)
    
    def get_resource_metrics(self, resource_id: str, metric_name: str,
                           start_time: datetime, end_time: datetime) -> APIResponse:
        """Get Azure monitoring metrics"""
        if not self._monitor_client:
            if not self.authenticate():
                return APIResponse(success=False, error_message="Authentication failed")
        
        def _get_metrics():
            metrics = self._monitor_client.metrics.list(
                resource_uri=resource_id,
                timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
                metricnames=metric_name,
                aggregation="Average"
            )
            return {"metrics": metrics}
        
        return self._make_request_with_retry(_get_metrics)


class UnifiedCloudAPIManager:
    """Unified manager for all cloud provider API adapters"""
    
    def __init__(self):
        self.adapters: Dict[str, BaseCloudAPIAdapter] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
    
    def register_adapter(self, provider_name: str, adapter: BaseCloudAPIAdapter):
        """Register a cloud provider adapter"""
        self.adapters[provider_name] = adapter
        self.circuit_breakers[provider_name] = {
            "failure_count": 0,
            "last_failure": None,
            "state": "closed"  # closed, open, half-open
        }
        logger.info(f"Registered adapter for provider: {provider_name}")
    
    def get_adapter(self, provider_name: str) -> Optional[BaseCloudAPIAdapter]:
        """Get adapter for a specific provider"""
        return self.adapters.get(provider_name)
    
    def test_all_connections(self) -> Dict[str, APIResponse]:
        """Test connections for all registered providers"""
        results = {}
        
        for provider_name, adapter in self.adapters.items():
            if self._is_circuit_breaker_open(provider_name):
                results[provider_name] = APIResponse(
                    success=False,
                    error_message="Circuit breaker is open"
                )
                continue
            
            try:
                result = adapter.test_connection()
                results[provider_name] = result
                
                if result.success:
                    self._reset_circuit_breaker(provider_name)
                else:
                    self._record_failure(provider_name)
                    
            except Exception as e:
                self._record_failure(provider_name)
                results[provider_name] = APIResponse(
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def get_billing_data_unified(self, provider_name: str, start_date: datetime,
                               end_date: datetime) -> APIResponse:
        """Get billing data with circuit breaker protection"""
        if self._is_circuit_breaker_open(provider_name):
            return APIResponse(
                success=False,
                error_message="Circuit breaker is open for this provider"
            )
        
        adapter = self.get_adapter(provider_name)
        if not adapter:
            return APIResponse(
                success=False,
                error_message=f"No adapter found for provider: {provider_name}"
            )
        
        try:
            result = adapter.get_billing_data(start_date, end_date)
            
            if result.success:
                self._reset_circuit_breaker(provider_name)
            else:
                self._record_failure(provider_name)
            
            return result
            
        except Exception as e:
            self._record_failure(provider_name)
            return APIResponse(
                success=False,
                error_message=str(e)
            )
    
    def _is_circuit_breaker_open(self, provider_name: str) -> bool:
        """Check if circuit breaker is open for provider"""
        breaker = self.circuit_breakers.get(provider_name, {})
        
        if breaker.get("state") == "open":
            # Check if we should try half-open
            last_failure = breaker.get("last_failure")
            if last_failure and datetime.now() - last_failure > timedelta(minutes=5):
                breaker["state"] = "half-open"
                return False
            return True
        
        return False
    
    def _record_failure(self, provider_name: str):
        """Record API failure for circuit breaker"""
        breaker = self.circuit_breakers[provider_name]
        breaker["failure_count"] += 1
        breaker["last_failure"] = datetime.now()
        
        # Open circuit breaker after 5 failures
        if breaker["failure_count"] >= 5:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for provider: {provider_name}")
    
    def _reset_circuit_breaker(self, provider_name: str):
        """Reset circuit breaker after successful call"""
        breaker = self.circuit_breakers[provider_name]
        breaker["failure_count"] = 0
        breaker["last_failure"] = None
        breaker["state"] = "closed"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "providers": {}
        }
        
        for provider_name, breaker in self.circuit_breakers.items():
            status["providers"][provider_name] = {
                "state": breaker["state"],
                "failure_count": breaker["failure_count"],
                "last_failure": breaker["last_failure"].isoformat() if breaker["last_failure"] else None
            }
        
        return status


# Factory function to create appropriate adapter
def create_cloud_adapter(provider_type: str, credentials: Dict[str, str],
                        **kwargs) -> BaseCloudAPIAdapter:
    """Factory function to create cloud provider adapter"""
    
    if provider_type.lower() == "aws":
        return AWSAPIAdapter(
            credentials=credentials,
            region=kwargs.get("region", "us-east-1"),
            retry_config=kwargs.get("retry_config")
        )
    elif provider_type.lower() == "gcp":
        return GCPAPIAdapter(
            credentials=credentials,
            project_id=kwargs.get("project_id"),
            retry_config=kwargs.get("retry_config")
        )
    elif provider_type.lower() == "azure":
        return AzureAPIAdapter(
            credentials=credentials,
            subscription_id=kwargs.get("subscription_id"),
            retry_config=kwargs.get("retry_config")
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize unified API manager
    api_manager = UnifiedCloudAPIManager()
    
    # Example AWS configuration
    aws_credentials = {
        "access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    }
    
    # Create and register AWS adapter
    aws_adapter = create_cloud_adapter("aws", aws_credentials, region="us-east-1")
    api_manager.register_adapter("aws-production", aws_adapter)
    
    # Test connections
    connection_results = api_manager.test_all_connections()
    print("Connection Test Results:")
    for provider, result in connection_results.items():
        print(f"  {provider}: {'✓' if result.success else '✗'} {result.error_message or 'Connected'}")
    
    # Get health status
    health_status = api_manager.get_health_status()
    print(f"\nHealth Status: {health_status}")
    
    # Example billing data request
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    billing_result = api_manager.get_billing_data_unified("aws-production", start_date, end_date)
    print(f"\nBilling Data Request: {'✓' if billing_result.success else '✗'} {billing_result.error_message or 'Success'}")