"""
Resource-Level Cost Attribution Engine

Provides detailed cost attribution to specific resources, enabling
drill-down analysis from service-level anomalies to individual resources.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ResourceCostAttribution:
    """Cost attribution for a specific resource"""
    resource_id: str
    resource_type: str
    service: str
    account_id: str
    region: str
    cost_amount: float
    cost_percentage: float  # Percentage of total service cost
    usage_metrics: Dict[str, Any]
    tags: Dict[str, str]
    attribution_confidence: float
    cost_drivers: List[str]
    optimization_opportunities: List[str]

@dataclass
class CostBreakdown:
    """Hierarchical cost breakdown"""
    total_cost: float
    service_breakdown: Dict[str, float]
    resource_breakdown: Dict[str, List[ResourceCostAttribution]]
    unattributed_cost: float
    attribution_accuracy: float

class ResourceAttributionEngine:
    """
    Attributes costs to specific resources with high granularity.
    
    Enables drill-down from service-level anomalies to individual
    resources, providing detailed cost attribution and optimization insights.
    """
    
    def __init__(self):
        self.attribution_rules = self._initialize_attribution_rules()
        self.cost_drivers = self._initialize_cost_drivers()
    
    def _initialize_attribution_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize service-specific attribution rules"""
        return {
            'ec2': {
                'primary_metrics': ['instance_hours', 'cpu_utilization', 'network_io'],
                'cost_factors': {
                    'instance_type': 0.7,
                    'usage_hours': 0.2,
                    'data_transfer': 0.1
                },
                'attribution_method': 'instance_based'
            },
            's3': {
                'primary_metrics': ['storage_gb', 'requests', 'data_transfer'],
                'cost_factors': {
                    'storage_size': 0.6,
                    'request_count': 0.25,
                    'data_transfer': 0.15
                },
                'attribution_method': 'bucket_based'
            },
            'rds': {
                'primary_metrics': ['instance_hours', 'storage_gb', 'io_operations'],
                'cost_factors': {
                    'instance_class': 0.8,
                    'storage_size': 0.15,
                    'io_operations': 0.05
                },
                'attribution_method': 'instance_based'
            },
            'lambda': {
                'primary_metrics': ['invocations', 'duration_ms', 'memory_mb'],
                'cost_factors': {
                    'invocations': 0.4,
                    'duration': 0.4,
                    'memory_allocation': 0.2
                },
                'attribution_method': 'function_based'
            }
        }
    
    def _initialize_cost_drivers(self) -> Dict[str, List[str]]:
        """Initialize cost driver patterns for each service"""
        return {
            'ec2': [
                'instance_type_upgrade',
                'increased_usage_hours',
                'data_transfer_spike',
                'ebs_volume_expansion',
                'elastic_ip_usage'
            ],
            's3': [
                'storage_growth',
                'request_pattern_change',
                'data_transfer_increase',
                'storage_class_change',
                'lifecycle_policy_impact'
            ],
            'rds': [
                'instance_scaling',
                'storage_expansion',
                'backup_retention_increase',
                'multi_az_enablement',
                'read_replica_addition'
            ],
            'lambda': [
                'invocation_increase',
                'execution_duration_increase',
                'memory_allocation_increase',
                'concurrent_execution_spike'
            ]
        }
    
    async def attribute_service_costs(
        self,
        service: str,
        account_id: str,
        cost_data: Dict[str, Any],
        resource_data: List[Dict[str, Any]],
        time_period: Tuple[datetime, datetime]
    ) -> CostBreakdown:
        """
        Attribute service costs to individual resources
        
        Args:
            service: AWS service name (ec2, s3, rds, lambda)
            account_id: AWS account ID
            cost_data: Service-level cost data
            resource_data: Individual resource usage data
            time_period: Time period for attribution
        
        Returns:
            CostBreakdown with detailed resource attribution
        """
        try:
            logger.info(f"Starting cost attribution for {service} in account {account_id}")
            
            # Get attribution rules for service
            rules = self.attribution_rules.get(service.lower(), {})
            if not rules:
                logger.warning(f"No attribution rules for service: {service}")
                return self._create_fallback_breakdown(cost_data, resource_data)
            
            # Perform service-specific attribution
            if service.lower() == 'ec2':
                attributions = await self._attribute_ec2_costs(cost_data, resource_data, rules)
            elif service.lower() == 's3':
                attributions = await self._attribute_s3_costs(cost_data, resource_data, rules)
            elif service.lower() == 'rds':
                attributions = await self._attribute_rds_costs(cost_data, resource_data, rules)
            elif service.lower() == 'lambda':
                attributions = await self._attribute_lambda_costs(cost_data, resource_data, rules)
            else:
                attributions = await self._attribute_generic_costs(cost_data, resource_data, rules)
            
            # Calculate breakdown
            breakdown = self._calculate_cost_breakdown(service, cost_data, attributions)
            
            logger.info(f"Cost attribution completed for {service}: {len(attributions)} resources")
            return breakdown
            
        except Exception as e:
            logger.error(f"Error in cost attribution: {e}")
            return self._create_fallback_breakdown(cost_data, resource_data)
    
    async def _attribute_ec2_costs(
        self,
        cost_data: Dict[str, Any],
        resource_data: List[Dict[str, Any]],
        rules: Dict[str, Any]
    ) -> List[ResourceCostAttribution]:
        """Attribute EC2 costs to individual instances"""
        
        attributions = []
        total_cost = cost_data.get('total_cost', 0)
        
        # Calculate total weighted usage
        total_weighted_usage = 0
        for resource in resource_data:
            instance_type = resource.get('instance_type', 't3.micro')
            hours = resource.get('usage_hours', 0)
            
            # Get instance type cost weight (simplified pricing model)
            type_weight = self._get_instance_type_weight(instance_type)
            weighted_usage = hours * type_weight
            total_weighted_usage += weighted_usage
        
        # Attribute costs proportionally
        for resource in resource_data:
            instance_id = resource.get('instance_id', 'unknown')
            instance_type = resource.get('instance_type', 't3.micro')
            hours = resource.get('usage_hours', 0)
            
            # Calculate resource cost
            type_weight = self._get_instance_type_weight(instance_type)
            weighted_usage = hours * type_weight
            
            if total_weighted_usage > 0:
                resource_cost = total_cost * (weighted_usage / total_weighted_usage)
                cost_percentage = (resource_cost / total_cost * 100) if total_cost > 0 else 0
            else:
                resource_cost = 0
                cost_percentage = 0
            
            # Identify cost drivers
            cost_drivers = []
            if 'xlarge' in instance_type or 'metal' in instance_type:
                cost_drivers.append('high_performance_instance_type')
            if hours > 720:  # More than 30 days
                cost_drivers.append('continuous_operation')
            
            # Generate optimization opportunities
            optimization_opportunities = []
            cpu_utilization = resource.get('avg_cpu_utilization', 0)
            if cpu_utilization < 20:
                optimization_opportunities.append('Consider downsizing instance type')
            if hours > 168 and not resource.get('reserved_instance', False):
                optimization_opportunities.append('Consider Reserved Instance for cost savings')
            
            attribution = ResourceCostAttribution(
                resource_id=instance_id,
                resource_type='ec2_instance',
                service='ec2',
                account_id=resource.get('account_id', ''),
                region=resource.get('region', ''),
                cost_amount=resource_cost,
                cost_percentage=cost_percentage,
                usage_metrics={
                    'instance_type': instance_type,
                    'usage_hours': hours,
                    'avg_cpu_utilization': cpu_utilization,
                    'network_io_gb': resource.get('network_io_gb', 0)
                },
                tags=resource.get('tags', {}),
                attribution_confidence=0.9,
                cost_drivers=cost_drivers,
                optimization_opportunities=optimization_opportunities
            )
            
            attributions.append(attribution)
        
        return attributions
    
    async def _attribute_s3_costs(
        self,
        cost_data: Dict[str, Any],
        resource_data: List[Dict[str, Any]],
        rules: Dict[str, Any]
    ) -> List[ResourceCostAttribution]:
        """Attribute S3 costs to individual buckets"""
        
        attributions = []
        total_cost = cost_data.get('total_cost', 0)
        
        # Calculate total storage and requests
        total_storage = sum(r.get('storage_gb', 0) for r in resource_data)
        total_requests = sum(r.get('request_count', 0) for r in resource_data)
        
        for resource in resource_data:
            bucket_name = resource.get('bucket_name', 'unknown')
            storage_gb = resource.get('storage_gb', 0)
            requests = resource.get('request_count', 0)
            
            # Calculate cost attribution based on storage and requests
            storage_cost_ratio = (storage_gb / total_storage) if total_storage > 0 else 0
            request_cost_ratio = (requests / total_requests) if total_requests > 0 else 0
            
            # Weighted attribution (60% storage, 40% requests)
            cost_ratio = (storage_cost_ratio * 0.6) + (request_cost_ratio * 0.4)
            resource_cost = total_cost * cost_ratio
            cost_percentage = (resource_cost / total_cost * 100) if total_cost > 0 else 0
            
            # Identify cost drivers
            cost_drivers = []
            if storage_gb > 1000:  # > 1TB
                cost_drivers.append('large_storage_volume')
            if requests > 1000000:  # > 1M requests
                cost_drivers.append('high_request_volume')
            
            # Generate optimization opportunities
            optimization_opportunities = []
            storage_class = resource.get('storage_class', 'STANDARD')
            if storage_class == 'STANDARD' and storage_gb > 100:
                optimization_opportunities.append('Consider Intelligent Tiering or IA storage class')
            
            attribution = ResourceCostAttribution(
                resource_id=bucket_name,
                resource_type='s3_bucket',
                service='s3',
                account_id=resource.get('account_id', ''),
                region=resource.get('region', ''),
                cost_amount=resource_cost,
                cost_percentage=cost_percentage,
                usage_metrics={
                    'storage_gb': storage_gb,
                    'request_count': requests,
                    'storage_class': storage_class,
                    'data_transfer_gb': resource.get('data_transfer_gb', 0)
                },
                tags=resource.get('tags', {}),
                attribution_confidence=0.85,
                cost_drivers=cost_drivers,
                optimization_opportunities=optimization_opportunities
            )
            
            attributions.append(attribution)
        
        return attributions
    
    async def _attribute_rds_costs(
        self,
        cost_data: Dict[str, Any],
        resource_data: List[Dict[str, Any]],
        rules: Dict[str, Any]
    ) -> List[ResourceCostAttribution]:
        """Attribute RDS costs to individual database instances"""
        
        attributions = []
        total_cost = cost_data.get('total_cost', 0)
        
        # Calculate total weighted usage
        total_weighted_usage = 0
        for resource in resource_data:
            instance_class = resource.get('instance_class', 'db.t3.micro')
            hours = resource.get('usage_hours', 0)
            
            class_weight = self._get_rds_class_weight(instance_class)
            weighted_usage = hours * class_weight
            total_weighted_usage += weighted_usage
        
        for resource in resource_data:
            db_instance_id = resource.get('db_instance_id', 'unknown')
            instance_class = resource.get('instance_class', 'db.t3.micro')
            hours = resource.get('usage_hours', 0)
            
            # Calculate resource cost
            class_weight = self._get_rds_class_weight(instance_class)
            weighted_usage = hours * class_weight
            
            if total_weighted_usage > 0:
                resource_cost = total_cost * (weighted_usage / total_weighted_usage)
                cost_percentage = (resource_cost / total_cost * 100) if total_cost > 0 else 0
            else:
                resource_cost = 0
                cost_percentage = 0
            
            # Identify cost drivers
            cost_drivers = []
            if 'xlarge' in instance_class:
                cost_drivers.append('large_instance_class')
            if resource.get('multi_az', False):
                cost_drivers.append('multi_az_deployment')
            
            # Generate optimization opportunities
            optimization_opportunities = []
            cpu_utilization = resource.get('avg_cpu_utilization', 0)
            if cpu_utilization < 30:
                optimization_opportunities.append('Consider downsizing instance class')
            
            attribution = ResourceCostAttribution(
                resource_id=db_instance_id,
                resource_type='rds_instance',
                service='rds',
                account_id=resource.get('account_id', ''),
                region=resource.get('region', ''),
                cost_amount=resource_cost,
                cost_percentage=cost_percentage,
                usage_metrics={
                    'instance_class': instance_class,
                    'usage_hours': hours,
                    'allocated_storage_gb': resource.get('allocated_storage_gb', 0),
                    'avg_cpu_utilization': cpu_utilization
                },
                tags=resource.get('tags', {}),
                attribution_confidence=0.9,
                cost_drivers=cost_drivers,
                optimization_opportunities=optimization_opportunities
            )
            
            attributions.append(attribution)
        
        return attributions
    
    async def _attribute_lambda_costs(
        self,
        cost_data: Dict[str, Any],
        resource_data: List[Dict[str, Any]],
        rules: Dict[str, Any]
    ) -> List[ResourceCostAttribution]:
        """Attribute Lambda costs to individual functions"""
        
        attributions = []
        total_cost = cost_data.get('total_cost', 0)
        
        # Calculate total GB-seconds (invocations * duration * memory)
        total_gb_seconds = 0
        for resource in resource_data:
            invocations = resource.get('invocations', 0)
            avg_duration_ms = resource.get('avg_duration_ms', 0)
            memory_mb = resource.get('memory_mb', 128)
            
            gb_seconds = invocations * (avg_duration_ms / 1000) * (memory_mb / 1024)
            total_gb_seconds += gb_seconds
        
        for resource in resource_data:
            function_name = resource.get('function_name', 'unknown')
            invocations = resource.get('invocations', 0)
            avg_duration_ms = resource.get('avg_duration_ms', 0)
            memory_mb = resource.get('memory_mb', 128)
            
            # Calculate resource cost based on GB-seconds
            gb_seconds = invocations * (avg_duration_ms / 1000) * (memory_mb / 1024)
            
            if total_gb_seconds > 0:
                resource_cost = total_cost * (gb_seconds / total_gb_seconds)
                cost_percentage = (resource_cost / total_cost * 100) if total_cost > 0 else 0
            else:
                resource_cost = 0
                cost_percentage = 0
            
            # Identify cost drivers
            cost_drivers = []
            if invocations > 1000000:  # > 1M invocations
                cost_drivers.append('high_invocation_volume')
            if avg_duration_ms > 10000:  # > 10 seconds
                cost_drivers.append('long_execution_duration')
            if memory_mb > 1024:  # > 1GB
                cost_drivers.append('high_memory_allocation')
            
            # Generate optimization opportunities
            optimization_opportunities = []
            if avg_duration_ms > 5000:
                optimization_opportunities.append('Optimize function performance to reduce duration')
            if memory_mb > 512 and avg_duration_ms < 1000:
                optimization_opportunities.append('Consider reducing memory allocation')
            
            attribution = ResourceCostAttribution(
                resource_id=function_name,
                resource_type='lambda_function',
                service='lambda',
                account_id=resource.get('account_id', ''),
                region=resource.get('region', ''),
                cost_amount=resource_cost,
                cost_percentage=cost_percentage,
                usage_metrics={
                    'invocations': invocations,
                    'avg_duration_ms': avg_duration_ms,
                    'memory_mb': memory_mb,
                    'gb_seconds': gb_seconds
                },
                tags=resource.get('tags', {}),
                attribution_confidence=0.95,
                cost_drivers=cost_drivers,
                optimization_opportunities=optimization_opportunities
            )
            
            attributions.append(attribution)
        
        return attributions
    
    async def _attribute_generic_costs(
        self,
        cost_data: Dict[str, Any],
        resource_data: List[Dict[str, Any]],
        rules: Dict[str, Any]
    ) -> List[ResourceCostAttribution]:
        """Generic cost attribution for unsupported services"""
        
        attributions = []
        total_cost = cost_data.get('total_cost', 0)
        resource_count = len(resource_data)
        
        if resource_count == 0:
            return attributions
        
        # Equal distribution for generic services
        cost_per_resource = total_cost / resource_count
        
        for resource in resource_data:
            resource_id = resource.get('resource_id', 'unknown')
            
            attribution = ResourceCostAttribution(
                resource_id=resource_id,
                resource_type='generic_resource',
                service=cost_data.get('service', 'unknown'),
                account_id=resource.get('account_id', ''),
                region=resource.get('region', ''),
                cost_amount=cost_per_resource,
                cost_percentage=(100 / resource_count),
                usage_metrics=resource.get('metrics', {}),
                tags=resource.get('tags', {}),
                attribution_confidence=0.5,
                cost_drivers=['generic_usage'],
                optimization_opportunities=['Review service usage patterns']
            )
            
            attributions.append(attribution)
        
        return attributions
    
    def _get_instance_type_weight(self, instance_type: str) -> float:
        """Get relative cost weight for EC2 instance type"""
        # Simplified instance type weights (relative to t3.micro = 1.0)
        weights = {
            't3.nano': 0.5, 't3.micro': 1.0, 't3.small': 2.0, 't3.medium': 4.0,
            't3.large': 8.0, 't3.xlarge': 16.0, 't3.2xlarge': 32.0,
            'm5.large': 10.0, 'm5.xlarge': 20.0, 'm5.2xlarge': 40.0,
            'c5.large': 9.0, 'c5.xlarge': 18.0, 'c5.2xlarge': 36.0,
            'r5.large': 12.0, 'r5.xlarge': 24.0, 'r5.2xlarge': 48.0
        }
        return weights.get(instance_type, 4.0)  # Default weight
    
    def _get_rds_class_weight(self, instance_class: str) -> float:
        """Get relative cost weight for RDS instance class"""
        weights = {
            'db.t3.micro': 1.0, 'db.t3.small': 2.0, 'db.t3.medium': 4.0,
            'db.t3.large': 8.0, 'db.t3.xlarge': 16.0, 'db.t3.2xlarge': 32.0,
            'db.m5.large': 12.0, 'db.m5.xlarge': 24.0, 'db.m5.2xlarge': 48.0,
            'db.r5.large': 15.0, 'db.r5.xlarge': 30.0, 'db.r5.2xlarge': 60.0
        }
        return weights.get(instance_class, 8.0)  # Default weight
    
    def _calculate_cost_breakdown(
        self,
        service: str,
        cost_data: Dict[str, Any],
        attributions: List[ResourceCostAttribution]
    ) -> CostBreakdown:
        """Calculate hierarchical cost breakdown"""
        
        total_cost = cost_data.get('total_cost', 0)
        attributed_cost = sum(attr.cost_amount for attr in attributions)
        unattributed_cost = max(0, total_cost - attributed_cost)
        
        attribution_accuracy = (attributed_cost / total_cost) if total_cost > 0 else 0
        
        return CostBreakdown(
            total_cost=total_cost,
            service_breakdown={service: total_cost},
            resource_breakdown={service: attributions},
            unattributed_cost=unattributed_cost,
            attribution_accuracy=attribution_accuracy
        )
    
    def _create_fallback_breakdown(
        self,
        cost_data: Dict[str, Any],
        resource_data: List[Dict[str, Any]]
    ) -> CostBreakdown:
        """Create fallback breakdown when attribution fails"""
        
        total_cost = cost_data.get('total_cost', 0)
        service = cost_data.get('service', 'unknown')
        
        return CostBreakdown(
            total_cost=total_cost,
            service_breakdown={service: total_cost},
            resource_breakdown={service: []},
            unattributed_cost=total_cost,
            attribution_accuracy=0.0
        )
    
    async def get_top_cost_resources(
        self,
        attributions: List[ResourceCostAttribution],
        limit: int = 10
    ) -> List[ResourceCostAttribution]:
        """Get top cost-contributing resources"""
        
        sorted_attributions = sorted(
            attributions,
            key=lambda x: x.cost_amount,
            reverse=True
        )
        
        return sorted_attributions[:limit]
    
    async def get_optimization_opportunities(
        self,
        attributions: List[ResourceCostAttribution]
    ) -> Dict[str, List[str]]:
        """Aggregate optimization opportunities by category"""
        
        opportunities = defaultdict(list)
        
        for attribution in attributions:
            for opportunity in attribution.optimization_opportunities:
                opportunities[attribution.service].append({
                    'resource_id': attribution.resource_id,
                    'opportunity': opportunity,
                    'potential_savings': attribution.cost_amount * 0.1  # Estimated 10% savings
                })
        
        return dict(opportunities)