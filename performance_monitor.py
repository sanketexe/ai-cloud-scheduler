# performance_monitor.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import statistics
import json
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    import math

from enhanced_models import (
    EnhancedVirtualMachine, PerformanceMetrics, HealthStatus, 
    SeverityLevel, UtilizationTrends
)


class MetricType(Enum):
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    OTHER = "other"


class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


class TrendDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class CloudResource:
    """Represents a cloud resource for monitoring"""
    resource_id: str
    resource_type: str  # e.g., "ec2_instance", "gcp_instance", "azure_vm"
    provider: CloudProvider
    region: str
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MetricsData:
    """Container for collected metrics data"""
    resource_id: str
    metrics: Dict[MetricType, List[Tuple[datetime, float]]]  # (timestamp, value) pairs
    collection_period_start: datetime
    collection_period_end: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceCapacity:
    """Represents resource capacity configuration"""
    cpu_cores: int
    memory_gb: float
    disk_gb: float
    network_bandwidth_mbps: float
    instance_type: str


@dataclass
class ScalingRecommendation:
    """Recommendation for resource scaling"""
    resource_id: str
    current_capacity: ResourceCapacity
    recommended_capacity: ResourceCapacity
    scaling_direction: ScalingDirection
    confidence_score: float  # 0-1
    estimated_cost_impact: float
    rationale: str
    urgency: SeverityLevel
    created_at: datetime = field(default_factory=datetime.now)


class CloudMonitoringAPI(ABC):
    """Abstract base class for cloud provider monitoring APIs"""
    
    @abstractmethod
    async def get_metrics(self, resource_ids: List[str], 
                         metric_types: List[MetricType],
                         start_time: datetime, end_time: datetime) -> Dict[str, MetricsData]:
        """Retrieve metrics from cloud provider monitoring service"""
        pass
    
    @abstractmethod
    async def get_resource_info(self, resource_ids: List[str]) -> Dict[str, CloudResource]:
        """Get resource information and metadata"""
        pass


class AWSCloudWatchAPI(CloudMonitoringAPI):
    """AWS CloudWatch integration for metrics collection"""
    
    def __init__(self, access_key: str, secret_key: str, region: str = "us-east-1"):
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.logger = logging.getLogger(f"{__name__}.AWSCloudWatchAPI")
    
    async def get_metrics(self, resource_ids: List[str], 
                         metric_types: List[MetricType],
                         start_time: datetime, end_time: datetime) -> Dict[str, MetricsData]:
        """Retrieve metrics from CloudWatch"""
        self.logger.info(f"Fetching AWS metrics for {len(resource_ids)} resources")
        
        metrics_data = {}
        
        for resource_id in resource_ids:
            # Simulate CloudWatch API calls
            metrics = {}
            
            for metric_type in metric_types:
                # Generate mock time series data
                time_points = []
                current_time = start_time
                
                while current_time <= end_time:
                    value = self._generate_mock_metric_value(metric_type, current_time)
                    time_points.append((current_time, value))
                    current_time += timedelta(minutes=5)  # 5-minute intervals
                
                metrics[metric_type] = time_points
            
            metrics_data[resource_id] = MetricsData(
                resource_id=resource_id,
                metrics=metrics,
                collection_period_start=start_time,
                collection_period_end=end_time,
                metadata={"provider": "aws", "region": self.region}
            )
        
        return metrics_data
    
    async def get_resource_info(self, resource_ids: List[str]) -> Dict[str, CloudResource]:
        """Get EC2 instance information"""
        resources = {}
        
        for resource_id in resource_ids:
            resources[resource_id] = CloudResource(
                resource_id=resource_id,
                resource_type="ec2_instance",
                provider=CloudProvider.AWS,
                region=self.region,
                tags={"Environment": "production", "Team": "backend"}
            )
        
        return resources
    
    def _generate_mock_metric_value(self, metric_type: MetricType, timestamp: datetime) -> float:
        """Generate realistic mock metric values"""
        import random
        
        # Add some time-based patterns
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        base_values = {
            MetricType.CPU_UTILIZATION: 45.0,
            MetricType.MEMORY_UTILIZATION: 60.0,
            MetricType.DISK_IO: 100.0,
            MetricType.NETWORK_IO: 50.0,
            MetricType.RESPONSE_TIME: 150.0,
            MetricType.THROUGHPUT: 1000.0,
            MetricType.ERROR_RATE: 0.5,
            MetricType.AVAILABILITY: 99.9
        }
        
        base_value = base_values.get(metric_type, 50.0)
        
        # Add daily patterns (higher during business hours)
        if 9 <= hour <= 17 and day_of_week < 5:  # Business hours, weekdays
            multiplier = 1.3
        elif 22 <= hour or hour <= 6:  # Night hours
            multiplier = 0.7
        else:
            multiplier = 1.0
        
        # Add random variation
        variation = random.uniform(0.8, 1.2)
        
        value = base_value * multiplier * variation
        
        # Ensure values are within reasonable bounds
        if metric_type in [MetricType.CPU_UTILIZATION, MetricType.MEMORY_UTILIZATION]:
            value = max(0, min(100, value))
        elif metric_type == MetricType.ERROR_RATE:
            value = max(0, min(10, value))
        elif metric_type == MetricType.AVAILABILITY:
            value = max(95, min(100, value))
        
        return round(value, 2)


class GCPMonitoringAPI(CloudMonitoringAPI):
    """Google Cloud Monitoring integration"""
    
    def __init__(self, project_id: str, credentials_path: str):
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.logger = logging.getLogger(f"{__name__}.GCPMonitoringAPI")
    
    async def get_metrics(self, resource_ids: List[str], 
                         metric_types: List[MetricType],
                         start_time: datetime, end_time: datetime) -> Dict[str, MetricsData]:
        """Retrieve metrics from Google Cloud Monitoring"""
        self.logger.info(f"Fetching GCP metrics for {len(resource_ids)} resources")
        # Similar implementation to AWS but with GCP-specific mock data
        return {}
    
    async def get_resource_info(self, resource_ids: List[str]) -> Dict[str, CloudResource]:
        """Get Compute Engine instance information"""
        return {}


class AzureMonitorAPI(CloudMonitoringAPI):
    """Azure Monitor integration"""
    
    def __init__(self, subscription_id: str, tenant_id: str, client_id: str, client_secret: str):
        self.subscription_id = subscription_id
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.logger = logging.getLogger(f"{__name__}.AzureMonitorAPI")
    
    async def get_metrics(self, resource_ids: List[str], 
                         metric_types: List[MetricType],
                         start_time: datetime, end_time: datetime) -> Dict[str, MetricsData]:
        """Retrieve metrics from Azure Monitor"""
        self.logger.info(f"Fetching Azure metrics for {len(resource_ids)} resources")
        return {}
    
    async def get_resource_info(self, resource_ids: List[str]) -> Dict[str, CloudResource]:
        """Get Azure VM information"""
        return {}


class MetricsCollector:
    """Centralized metrics collection from multiple cloud providers"""
    
    def __init__(self):
        self.monitoring_apis: Dict[CloudProvider, CloudMonitoringAPI] = {}
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.metrics_cache: Dict[str, MetricsData] = {}
        self.cache_ttl_minutes = 5
    
    def register_monitoring_api(self, provider: CloudProvider, api: CloudMonitoringAPI):
        """Register a monitoring API for a cloud provider"""
        self.monitoring_apis[provider] = api
        self.logger.info(f"Registered monitoring API for {provider.value}")
    
    async def collect_metrics(self, resources: List[CloudResource], 
                             metric_types: List[MetricType],
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Dict[str, MetricsData]:
        """Collect metrics from all resources across providers"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()
        
        # Group resources by provider
        resources_by_provider = {}
        for resource in resources:
            if resource.provider not in resources_by_provider:
                resources_by_provider[resource.provider] = []
            resources_by_provider[resource.provider].append(resource)
        
        all_metrics = {}
        tasks = []
        
        # Collect metrics from each provider
        for provider, provider_resources in resources_by_provider.items():
            if provider in self.monitoring_apis:
                api = self.monitoring_apis[provider]
                resource_ids = [r.resource_id for r in provider_resources]
                
                task = api.get_metrics(resource_ids, metric_types, start_time, end_time)
                tasks.append((provider, task))
        
        # Execute all collection tasks concurrently
        if tasks:
            for provider, task in tasks:
                try:
                    provider_metrics = await task
                    all_metrics.update(provider_metrics)
                    self.logger.info(f"Collected metrics from {provider.value}: {len(provider_metrics)} resources")
                except Exception as e:
                    self.logger.error(f"Error collecting metrics from {provider.value}: {e}")
        
        # Cache the results
        for resource_id, metrics_data in all_metrics.items():
            cache_key = f"{resource_id}_{start_time.isoformat()}_{end_time.isoformat()}"
            self.metrics_cache[cache_key] = metrics_data
        
        self.logger.info(f"Total metrics collected: {len(all_metrics)} resources")
        return all_metrics
    
    def normalize_metrics(self, metrics_data: Dict[str, MetricsData]) -> Dict[str, MetricsData]:
        """Normalize metrics across different cloud providers"""
        normalized_metrics = {}
        
        for resource_id, data in metrics_data.items():
            normalized_data = MetricsData(
                resource_id=resource_id,
                metrics={},
                collection_period_start=data.collection_period_start,
                collection_period_end=data.collection_period_end,
                metadata=data.metadata
            )
            
            # Normalize metric values and units
            for metric_type, time_series in data.metrics.items():
                normalized_series = []
                
                for timestamp, value in time_series:
                    # Apply provider-specific normalization
                    normalized_value = self._normalize_metric_value(
                        metric_type, value, data.metadata.get("provider", "unknown")
                    )
                    normalized_series.append((timestamp, normalized_value))
                
                normalized_data.metrics[metric_type] = normalized_series
            
            normalized_metrics[resource_id] = normalized_data
        
        return normalized_metrics
    
    def _normalize_metric_value(self, metric_type: MetricType, value: float, provider: str) -> float:
        """Normalize metric values across providers"""
        # Provider-specific normalization rules
        if provider == "aws":
            # AWS CloudWatch specific normalizations
            if metric_type == MetricType.DISK_IO:
                # Convert from bytes/sec to MB/sec if needed
                return value / (1024 * 1024) if value > 1000000 else value
        elif provider == "gcp":
            # GCP specific normalizations
            pass
        elif provider == "azure":
            # Azure specific normalizations
            pass
        
        return value
    
    def store_metrics(self, metrics_data: Dict[str, MetricsData], 
                     storage_backend: str = "memory") -> bool:
        """Store metrics data in time-series database"""
        try:
            if storage_backend == "memory":
                # Simple in-memory storage for demonstration
                for resource_id, data in metrics_data.items():
                    storage_key = f"metrics_{resource_id}_{datetime.now().isoformat()}"
                    # In a real implementation, this would write to InfluxDB, TimescaleDB, etc.
                    self.logger.debug(f"Stored metrics for {resource_id}: {len(data.metrics)} metric types")
            
            self.logger.info(f"Successfully stored metrics for {len(metrics_data)} resources")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")
            return False
    
    def retrieve_metrics(self, resource_ids: List[str], 
                        metric_types: List[MetricType],
                        start_time: datetime, end_time: datetime) -> Dict[str, MetricsData]:
        """Retrieve stored metrics data"""
        # Check cache first
        cached_metrics = {}
        
        for resource_id in resource_ids:
            cache_key = f"{resource_id}_{start_time.isoformat()}_{end_time.isoformat()}"
            if cache_key in self.metrics_cache:
                # Check if cache is still valid
                cache_age = datetime.now() - self.metrics_cache[cache_key].collection_period_end
                if cache_age.total_seconds() / 60 < self.cache_ttl_minutes:
                    cached_metrics[resource_id] = self.metrics_cache[cache_key]
        
        if cached_metrics:
            self.logger.info(f"Retrieved {len(cached_metrics)} resources from cache")
        
        return cached_metrics