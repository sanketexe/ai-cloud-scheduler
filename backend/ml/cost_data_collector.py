"""
Cost Data Collector for AI Anomaly Detection

Enhanced AWS cost data ingestion system that collects detailed cost and usage data
for machine learning analysis. Provides high-granularity data collection with
proper normalization and enrichment for ML feature engineering.
"""

import asyncio
import boto3
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date
from decimal import Decimal
from dataclasses import dataclass, field
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CostDataPoint:
    """Individual cost data point for ML processing"""
    timestamp: datetime
    account_id: str
    service: str
    region: str
    resource_id: Optional[str]
    cost_amount: Decimal
    usage_amount: Optional[Decimal]
    usage_unit: Optional[str]
    currency: str = "USD"
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceMetrics:
    """Service-specific usage and performance metrics"""
    service: str
    timestamp: datetime
    metrics: Dict[str, float]
    resource_count: int
    utilization_stats: Dict[str, float]


@dataclass
class ResourceMetadata:
    """Resource metadata for cost attribution"""
    resource_id: str
    resource_type: str
    service: str
    region: str
    tags: Dict[str, str]
    creation_time: datetime
    last_modified: datetime
    configuration: Dict[str, Any]


class CostDataCollector:
    """
    Enhanced cost data collector for ML anomaly detection.
    
    Collects detailed cost and usage data from multiple AWS APIs,
    normalizes the data format, and enriches with contextual metadata
    for machine learning feature engineering.
    """
    
    def __init__(self, aws_session: boto3.Session = None):
        self.session = aws_session or boto3.Session()
        
        # Initialize AWS clients
        self.cost_explorer = self.session.client('ce')
        self.cloudwatch = self.session.client('cloudwatch')
        self.config = self.session.client('config')
        self.billing = self.session.client('cur')  # Cost and Usage Reports
        
        # Data collection settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.batch_size = 1000
        
        # Service-specific metric configurations
        self.service_metrics_config = {
            'EC2': {
                'metrics': ['CPUUtilization', 'NetworkIn', 'NetworkOut', 'DiskReadOps', 'DiskWriteOps'],
                'dimensions': ['InstanceId', 'InstanceType'],
                'statistics': ['Average', 'Maximum', 'Sum']
            },
            'S3': {
                'metrics': ['BucketSizeBytes', 'NumberOfObjects'],
                'dimensions': ['BucketName', 'StorageType'],
                'statistics': ['Average', 'Sum']
            },
            'RDS': {
                'metrics': ['CPUUtilization', 'DatabaseConnections', 'ReadLatency', 'WriteLatency'],
                'dimensions': ['DBInstanceIdentifier', 'DBClusterIdentifier'],
                'statistics': ['Average', 'Maximum']
            },
            'Lambda': {
                'metrics': ['Invocations', 'Duration', 'Errors', 'Throttles'],
                'dimensions': ['FunctionName'],
                'statistics': ['Sum', 'Average']
            }
        }
    
    async def collect_hourly_costs(self, 
                                  start_date: date, 
                                  end_date: date,
                                  account_ids: Optional[List[str]] = None,
                                  services: Optional[List[str]] = None) -> List[CostDataPoint]:
        """
        Collect detailed hourly cost data for ML analysis.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            account_ids: Optional list of account IDs to filter
            services: Optional list of services to filter
            
        Returns:
            List of normalized cost data points
        """
        logger.info(
            "Starting hourly cost data collection",
            start_date=start_date,
            end_date=end_date,
            accounts=len(account_ids) if account_ids else "all",
            services=len(services) if services else "all"
        )
        
        try:
            # Build Cost Explorer query
            query_params = {
                'TimePeriod': {
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                'Granularity': 'HOURLY',
                'Metrics': ['BlendedCost', 'UsageQuantity'],
                'GroupBy': [
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'},
                    {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
                ]
            }
            
            # Add filters if specified
            if account_ids or services:
                query_params['Filter'] = self._build_cost_filter(account_ids, services)
            
            # Execute query with pagination
            cost_data = []
            next_token = None
            
            while True:
                if next_token:
                    query_params['NextPageToken'] = next_token
                
                response = await self._execute_with_retry(
                    self.cost_explorer.get_cost_and_usage,
                    **query_params
                )
                
                # Process response
                for result in response.get('ResultsByTime', []):
                    timestamp = datetime.fromisoformat(result['TimePeriod']['Start'])
                    
                    for group in result.get('Groups', []):
                        cost_point = self._parse_cost_group(group, timestamp)
                        if cost_point:
                            cost_data.append(cost_point)
                
                # Check for more pages
                next_token = response.get('NextPageToken')
                if not next_token:
                    break
            
            logger.info(
                "Hourly cost data collection completed",
                data_points=len(cost_data),
                time_range=f"{start_date} to {end_date}"
            )
            
            return cost_data
            
        except Exception as e:
            logger.error("Failed to collect hourly cost data", error=str(e))
            raise
    
    async def collect_service_metrics(self,
                                    service: str,
                                    start_time: datetime,
                                    end_time: datetime,
                                    region: str = 'us-east-1') -> List[ServiceMetrics]:
        """
        Collect service-specific usage and performance metrics.
        
        Args:
            service: AWS service name (EC2, S3, RDS, Lambda)
            start_time: Start time for metrics collection
            end_time: End time for metrics collection
            region: AWS region for metrics collection
            
        Returns:
            List of service metrics with utilization statistics
        """
        logger.info(
            "Collecting service metrics",
            service=service,
            region=region,
            time_range=f"{start_time} to {end_time}"
        )
        
        try:
            if service not in self.service_metrics_config:
                logger.warning("Service metrics not configured", service=service)
                return []
            
            config = self.service_metrics_config[service]
            metrics_data = []
            
            # Collect each metric for the service
            for metric_name in config['metrics']:
                metric_data = await self._collect_cloudwatch_metric(
                    service, metric_name, start_time, end_time, region, config
                )
                metrics_data.extend(metric_data)
            
            # Group by timestamp and calculate utilization stats
            grouped_metrics = self._group_metrics_by_timestamp(metrics_data)
            
            service_metrics = []
            for timestamp, metrics in grouped_metrics.items():
                utilization_stats = self._calculate_utilization_stats(metrics, service)
                
                service_metric = ServiceMetrics(
                    service=service,
                    timestamp=timestamp,
                    metrics=metrics,
                    resource_count=len(set(m.get('resource_id') for m in metrics.values() if m.get('resource_id'))),
                    utilization_stats=utilization_stats
                )
                service_metrics.append(service_metric)
            
            logger.info(
                "Service metrics collection completed",
                service=service,
                metric_points=len(service_metrics)
            )
            
            return service_metrics
            
        except Exception as e:
            logger.error("Failed to collect service metrics", service=service, error=str(e))
            raise
    
    async def collect_resource_metadata(self,
                                      service: str,
                                      region: str = 'us-east-1') -> List[ResourceMetadata]:
        """
        Collect resource metadata for cost attribution and context.
        
        Args:
            service: AWS service name
            region: AWS region
            
        Returns:
            List of resource metadata with tags and configuration
        """
        logger.info("Collecting resource metadata", service=service, region=region)
        
        try:
            # Use AWS Config to get resource inventory
            resource_types = self._get_resource_types_for_service(service)
            all_metadata = []
            
            for resource_type in resource_types:
                resources = await self._get_resources_by_type(resource_type, region)
                
                for resource in resources:
                    metadata = await self._enrich_resource_metadata(resource, service)
                    if metadata:
                        all_metadata.append(metadata)
            
            logger.info(
                "Resource metadata collection completed",
                service=service,
                resources=len(all_metadata)
            )
            
            return all_metadata
            
        except Exception as e:
            logger.error("Failed to collect resource metadata", service=service, error=str(e))
            raise
    
    def normalize_cost_data(self, cost_data: List[CostDataPoint]) -> pd.DataFrame:
        """
        Normalize cost data into ML-ready format.
        
        Args:
            cost_data: List of cost data points
            
        Returns:
            Normalized pandas DataFrame for ML processing
        """
        logger.info("Normalizing cost data for ML", data_points=len(cost_data))
        
        try:
            # Convert to DataFrame
            df_data = []
            for point in cost_data:
                row = {
                    'timestamp': point.timestamp,
                    'account_id': point.account_id,
                    'service': point.service,
                    'region': point.region,
                    'resource_id': point.resource_id,
                    'cost_amount': float(point.cost_amount),
                    'usage_amount': float(point.usage_amount) if point.usage_amount else 0.0,
                    'usage_unit': point.usage_unit,
                    'currency': point.currency
                }
                
                # Add tags as separate columns
                for tag_key, tag_value in point.tags.items():
                    row[f'tag_{tag_key}'] = tag_value
                
                # Add metadata
                for meta_key, meta_value in point.metadata.items():
                    row[f'meta_{meta_key}'] = meta_value
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Data normalization steps
            if not df.empty:
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Handle missing values
                df['cost_amount'] = df['cost_amount'].fillna(0.0)
                df['usage_amount'] = df['usage_amount'].fillna(0.0)
                
                # Add time-based features
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['day_of_month'] = df['timestamp'].dt.day
                df['month'] = df['timestamp'].dt.month
                
                # Add derived features
                df['cost_per_unit'] = df['cost_amount'] / (df['usage_amount'] + 1e-10)  # Avoid division by zero
                
                logger.info(
                    "Cost data normalization completed",
                    rows=len(df),
                    columns=len(df.columns)
                )
            
            return df
            
        except Exception as e:
            logger.error("Failed to normalize cost data", error=str(e))
            raise
    
    async def _execute_with_retry(self, func, **kwargs):
        """Execute AWS API call with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return func(**kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                
                logger.warning(
                    "API call failed, retrying",
                    attempt=attempt + 1,
                    error=str(e)
                )
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    def _build_cost_filter(self, account_ids: Optional[List[str]], services: Optional[List[str]]) -> Dict[str, Any]:
        """Build Cost Explorer filter"""
        filters = []
        
        if account_ids:
            filters.append({
                'Dimensions': {
                    'Key': 'LINKED_ACCOUNT',
                    'Values': account_ids
                }
            })
        
        if services:
            filters.append({
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': services
                }
            })
        
        if len(filters) == 1:
            return filters[0]
        elif len(filters) > 1:
            return {'And': filters}
        else:
            return {}
    
    def _parse_cost_group(self, group: Dict[str, Any], timestamp: datetime) -> Optional[CostDataPoint]:
        """Parse Cost Explorer group into CostDataPoint"""
        try:
            keys = group.get('Keys', [])
            if len(keys) < 3:
                return None
            
            service = keys[0] if keys[0] != 'NoService' else 'Unknown'
            region = keys[1] if keys[1] != 'NoRegion' else 'global'
            usage_type = keys[2]
            
            metrics = group.get('Metrics', {})
            cost_amount = Decimal(metrics.get('BlendedCost', {}).get('Amount', '0'))
            usage_amount = Decimal(metrics.get('UsageQuantity', {}).get('Amount', '0'))
            usage_unit = metrics.get('UsageQuantity', {}).get('Unit')
            
            # Extract account ID from context (would be available in real implementation)
            account_id = 'default'  # Placeholder
            
            return CostDataPoint(
                timestamp=timestamp,
                account_id=account_id,
                service=service,
                region=region,
                resource_id=None,  # Not available in Cost Explorer groups
                cost_amount=cost_amount,
                usage_amount=usage_amount,
                usage_unit=usage_unit,
                metadata={'usage_type': usage_type}
            )
            
        except Exception as e:
            logger.warning("Failed to parse cost group", group=group, error=str(e))
            return None
    
    async def _collect_cloudwatch_metric(self,
                                       service: str,
                                       metric_name: str,
                                       start_time: datetime,
                                       end_time: datetime,
                                       region: str,
                                       config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect CloudWatch metric data"""
        try:
            namespace = f'AWS/{service}'
            
            response = await self._execute_with_retry(
                self.cloudwatch.get_metric_statistics,
                Namespace=namespace,
                MetricName=metric_name,
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=config['statistics']
            )
            
            metrics = []
            for datapoint in response.get('Datapoints', []):
                metric_data = {
                    'metric_name': metric_name,
                    'timestamp': datapoint['Timestamp'],
                    'value': datapoint.get('Average', datapoint.get('Sum', 0)),
                    'unit': datapoint.get('Unit', ''),
                    'service': service,
                    'region': region
                }
                metrics.append(metric_data)
            
            return metrics
            
        except Exception as e:
            logger.warning("Failed to collect CloudWatch metric", metric=metric_name, error=str(e))
            return []
    
    def _group_metrics_by_timestamp(self, metrics_data: List[Dict[str, Any]]) -> Dict[datetime, Dict[str, Any]]:
        """Group metrics by timestamp"""
        grouped = {}
        
        for metric in metrics_data:
            timestamp = metric['timestamp']
            if timestamp not in grouped:
                grouped[timestamp] = {}
            
            metric_name = metric['metric_name']
            grouped[timestamp][metric_name] = metric
        
        return grouped
    
    def _calculate_utilization_stats(self, metrics: Dict[str, Any], service: str) -> Dict[str, float]:
        """Calculate utilization statistics for service metrics"""
        stats = {}
        
        if service == 'EC2':
            cpu_metric = metrics.get('CPUUtilization')
            if cpu_metric:
                stats['avg_cpu_utilization'] = cpu_metric.get('value', 0)
                stats['cpu_efficiency'] = min(cpu_metric.get('value', 0) / 80.0, 1.0)  # 80% target
        
        elif service == 'S3':
            size_metric = metrics.get('BucketSizeBytes')
            objects_metric = metrics.get('NumberOfObjects')
            if size_metric and objects_metric:
                stats['storage_efficiency'] = objects_metric.get('value', 0) / max(size_metric.get('value', 1), 1)
        
        elif service == 'RDS':
            cpu_metric = metrics.get('CPUUtilization')
            connections_metric = metrics.get('DatabaseConnections')
            if cpu_metric:
                stats['db_cpu_utilization'] = cpu_metric.get('value', 0)
            if connections_metric:
                stats['connection_utilization'] = connections_metric.get('value', 0)
        
        elif service == 'Lambda':
            invocations_metric = metrics.get('Invocations')
            errors_metric = metrics.get('Errors')
            if invocations_metric and errors_metric:
                total_invocations = invocations_metric.get('value', 0)
                total_errors = errors_metric.get('value', 0)
                stats['error_rate'] = total_errors / max(total_invocations, 1)
                stats['success_rate'] = 1 - stats['error_rate']
        
        return stats
    
    def _get_resource_types_for_service(self, service: str) -> List[str]:
        """Get AWS Config resource types for a service"""
        service_resource_types = {
            'EC2': ['AWS::EC2::Instance', 'AWS::EC2::Volume', 'AWS::EC2::SecurityGroup'],
            'S3': ['AWS::S3::Bucket'],
            'RDS': ['AWS::RDS::DBInstance', 'AWS::RDS::DBCluster'],
            'Lambda': ['AWS::Lambda::Function']
        }
        
        return service_resource_types.get(service, [])
    
    async def _get_resources_by_type(self, resource_type: str, region: str) -> List[Dict[str, Any]]:
        """Get resources by type from AWS Config"""
        try:
            response = await self._execute_with_retry(
                self.config.list_discovered_resources,
                resourceType=resource_type,
                limit=100  # Pagination would be needed for production
            )
            
            return response.get('resourceIdentifiers', [])
            
        except Exception as e:
            logger.warning("Failed to get resources by type", resource_type=resource_type, error=str(e))
            return []
    
    async def _enrich_resource_metadata(self, resource: Dict[str, Any], service: str) -> Optional[ResourceMetadata]:
        """Enrich resource with metadata and tags"""
        try:
            resource_id = resource.get('resourceId')
            resource_type = resource.get('resourceType')
            
            if not resource_id or not resource_type:
                return None
            
            # Get detailed resource configuration
            config_response = await self._execute_with_retry(
                self.config.get_resource_config_history,
                resourceType=resource_type,
                resourceId=resource_id,
                limit=1
            )
            
            config_items = config_response.get('configurationItems', [])
            if not config_items:
                return None
            
            config_item = config_items[0]
            
            return ResourceMetadata(
                resource_id=resource_id,
                resource_type=resource_type,
                service=service,
                region=config_item.get('awsRegion', 'unknown'),
                tags=config_item.get('tags', {}),
                creation_time=config_item.get('resourceCreationTime', datetime.utcnow()),
                last_modified=config_item.get('configurationItemCaptureTime', datetime.utcnow()),
                configuration=config_item.get('configuration', {})
            )
            
        except Exception as e:
            logger.warning("Failed to enrich resource metadata", resource=resource, error=str(e))
            return None