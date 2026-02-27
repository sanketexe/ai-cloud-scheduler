"""
Service-Specific Anomaly Analysis

Provides specialized anomaly detection for different AWS services with
service-specific patterns, thresholds, and analysis capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceType(Enum):
    EC2 = "ec2"
    S3 = "s3"
    RDS = "rds"
    LAMBDA = "lambda"
    EBS = "ebs"
    CLOUDFRONT = "cloudfront"
    ELB = "elb"

@dataclass
class ServiceAnomalyPattern:
    """Service-specific anomaly pattern definition"""
    service_type: ServiceType
    pattern_name: str
    threshold_multiplier: float
    baseline_period_hours: int
    seasonal_adjustment: bool
    cost_impact_weight: float
    confidence_threshold: float

@dataclass
class ServiceAnomalyResult:
    """Result of service-specific anomaly analysis"""
    service_type: ServiceType
    resource_id: str
    anomaly_type: str
    confidence_score: float
    cost_impact: float
    baseline_cost: float
    actual_cost: float
    deviation_percentage: float
    contributing_factors: List[str]
    service_specific_metrics: Dict[str, Any]
    recommendations: List[str]

class ServiceSpecificAnalyzer:
    """
    Analyzes anomalies with service-specific intelligence and patterns.
    
    Each AWS service has unique cost patterns, scaling behaviors, and
    anomaly characteristics that require specialized analysis.
    """
    
    def __init__(self):
        self.service_patterns = self._initialize_service_patterns()
        self.service_analyzers = {
            ServiceType.EC2: self._analyze_ec2_anomaly,
            ServiceType.S3: self._analyze_s3_anomaly,
            ServiceType.RDS: self._analyze_rds_anomaly,
            ServiceType.LAMBDA: self._analyze_lambda_anomaly,
            ServiceType.EBS: self._analyze_ebs_anomaly,
            ServiceType.CLOUDFRONT: self._analyze_cloudfront_anomaly,
            ServiceType.ELB: self._analyze_elb_anomaly,
        }
    
    def _initialize_service_patterns(self) -> Dict[ServiceType, List[ServiceAnomalyPattern]]:
        """Initialize service-specific anomaly patterns"""
        return {
            ServiceType.EC2: [
                ServiceAnomalyPattern(
                    service_type=ServiceType.EC2,
                    pattern_name="instance_scaling_spike",
                    threshold_multiplier=2.0,
                    baseline_period_hours=168,  # 1 week
                    seasonal_adjustment=True,
                    cost_impact_weight=1.5,
                    confidence_threshold=0.8
                ),
                ServiceAnomalyPattern(
                    service_type=ServiceType.EC2,
                    pattern_name="instance_type_change",
                    threshold_multiplier=1.5,
                    baseline_period_hours=72,  # 3 days
                    seasonal_adjustment=False,
                    cost_impact_weight=2.0,
                    confidence_threshold=0.9
                ),
            ],
            ServiceType.S3: [
                ServiceAnomalyPattern(
                    service_type=ServiceType.S3,
                    pattern_name="storage_growth_spike",
                    threshold_multiplier=1.8,
                    baseline_period_hours=720,  # 30 days
                    seasonal_adjustment=True,
                    cost_impact_weight=1.2,
                    confidence_threshold=0.7
                ),
                ServiceAnomalyPattern(
                    service_type=ServiceType.S3,
                    pattern_name="request_pattern_anomaly",
                    threshold_multiplier=3.0,
                    baseline_period_hours=24,  # 1 day
                    seasonal_adjustment=True,
                    cost_impact_weight=1.8,
                    confidence_threshold=0.85
                ),
            ],
            ServiceType.RDS: [
                ServiceAnomalyPattern(
                    service_type=ServiceType.RDS,
                    pattern_name="instance_scaling",
                    threshold_multiplier=2.5,
                    baseline_period_hours=168,  # 1 week
                    seasonal_adjustment=False,
                    cost_impact_weight=2.2,
                    confidence_threshold=0.9
                ),
            ],
            ServiceType.LAMBDA: [
                ServiceAnomalyPattern(
                    service_type=ServiceType.LAMBDA,
                    pattern_name="invocation_spike",
                    threshold_multiplier=4.0,
                    baseline_period_hours=24,  # 1 day
                    seasonal_adjustment=True,
                    cost_impact_weight=1.0,
                    confidence_threshold=0.75
                ),
            ],
        }
    
    async def analyze_service_anomaly(
        self,
        service_type: str,
        resource_id: str,
        cost_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> ServiceAnomalyResult:
        """
        Analyze anomaly with service-specific intelligence
        
        Args:
            service_type: AWS service type (ec2, s3, rds, lambda, etc.)
            resource_id: Specific resource identifier
            cost_data: Current cost data point
            historical_data: Historical cost data for baseline
            context: Additional context (tags, metadata, etc.)
        
        Returns:
            ServiceAnomalyResult with detailed analysis
        """
        try:
            # Convert service type
            service_enum = ServiceType(service_type.lower())
            
            # Get service-specific analyzer
            analyzer = self.service_analyzers.get(service_enum)
            if not analyzer:
                logger.warning(f"No specific analyzer for service: {service_type}")
                return self._generic_analysis(service_type, resource_id, cost_data, historical_data)
            
            # Run service-specific analysis
            result = await analyzer(resource_id, cost_data, historical_data, context)
            
            logger.info(f"Service-specific analysis completed for {service_type}:{resource_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in service-specific analysis: {e}")
            return self._generic_analysis(service_type, resource_id, cost_data, historical_data)
    
    async def _analyze_ec2_anomaly(
        self,
        resource_id: str,
        cost_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> ServiceAnomalyResult:
        """Analyze EC2-specific anomaly patterns"""
        
        # Calculate baseline metrics
        baseline_cost = np.mean([d.get('cost', 0) for d in historical_data[-168:]])  # 1 week
        actual_cost = cost_data.get('cost', 0)
        deviation = ((actual_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        # EC2-specific factors
        contributing_factors = []
        service_metrics = {}
        recommendations = []
        
        # Instance count changes
        current_instances = cost_data.get('instance_count', 0)
        historical_instances = np.mean([d.get('instance_count', 0) for d in historical_data[-24:]])
        
        if current_instances > historical_instances * 1.5:
            contributing_factors.append("Significant instance count increase")
            recommendations.append("Review auto-scaling policies and instance launch patterns")
            service_metrics['instance_scaling_factor'] = current_instances / historical_instances
        
        # Instance type changes
        current_instance_types = cost_data.get('instance_types', [])
        if any('xlarge' in itype or 'metal' in itype for itype in current_instance_types):
            contributing_factors.append("High-cost instance types detected")
            recommendations.append("Evaluate if large instance types are necessary")
            service_metrics['high_cost_instances'] = len([t for t in current_instance_types if 'xlarge' in t])
        
        # Spot vs On-Demand analysis
        spot_percentage = cost_data.get('spot_percentage', 0)
        if spot_percentage < 30 and deviation > 50:
            contributing_factors.append("Low Spot instance usage during cost spike")
            recommendations.append("Consider increasing Spot instance usage for cost optimization")
            service_metrics['spot_usage'] = spot_percentage
        
        # Calculate confidence score
        confidence_score = min(0.95, abs(deviation) / 100 + 0.5)
        if len(contributing_factors) > 2:
            confidence_score += 0.1
        
        return ServiceAnomalyResult(
            service_type=ServiceType.EC2,
            resource_id=resource_id,
            anomaly_type="ec2_scaling_anomaly" if current_instances > historical_instances * 1.2 else "ec2_cost_anomaly",
            confidence_score=confidence_score,
            cost_impact=actual_cost - baseline_cost,
            baseline_cost=baseline_cost,
            actual_cost=actual_cost,
            deviation_percentage=deviation,
            contributing_factors=contributing_factors,
            service_specific_metrics=service_metrics,
            recommendations=recommendations
        )
    
    async def _analyze_s3_anomaly(
        self,
        resource_id: str,
        cost_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> ServiceAnomalyResult:
        """Analyze S3-specific anomaly patterns"""
        
        baseline_cost = np.mean([d.get('cost', 0) for d in historical_data[-720:]])  # 30 days
        actual_cost = cost_data.get('cost', 0)
        deviation = ((actual_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        contributing_factors = []
        service_metrics = {}
        recommendations = []
        
        # Storage growth analysis
        current_storage = cost_data.get('storage_gb', 0)
        historical_storage = np.mean([d.get('storage_gb', 0) for d in historical_data[-168:]])
        
        if current_storage > historical_storage * 1.3:
            contributing_factors.append("Rapid storage growth detected")
            recommendations.append("Review data lifecycle policies and implement archiving")
            service_metrics['storage_growth_rate'] = (current_storage - historical_storage) / historical_storage
        
        # Request pattern analysis
        current_requests = cost_data.get('request_count', 0)
        historical_requests = np.mean([d.get('request_count', 0) for d in historical_data[-24:]])
        
        if current_requests > historical_requests * 2.0:
            contributing_factors.append("Unusual request pattern spike")
            recommendations.append("Investigate application behavior and implement request optimization")
            service_metrics['request_spike_factor'] = current_requests / historical_requests if historical_requests > 0 else 0
        
        # Data transfer analysis
        transfer_cost = cost_data.get('transfer_cost', 0)
        total_cost = cost_data.get('cost', 0)
        transfer_percentage = (transfer_cost / total_cost * 100) if total_cost > 0 else 0
        
        if transfer_percentage > 40:
            contributing_factors.append("High data transfer costs")
            recommendations.append("Optimize data transfer patterns and consider CloudFront")
            service_metrics['transfer_cost_percentage'] = transfer_percentage
        
        confidence_score = min(0.95, abs(deviation) / 100 + 0.6)
        
        return ServiceAnomalyResult(
            service_type=ServiceType.S3,
            resource_id=resource_id,
            anomaly_type="s3_storage_anomaly" if current_storage > historical_storage * 1.2 else "s3_request_anomaly",
            confidence_score=confidence_score,
            cost_impact=actual_cost - baseline_cost,
            baseline_cost=baseline_cost,
            actual_cost=actual_cost,
            deviation_percentage=deviation,
            contributing_factors=contributing_factors,
            service_specific_metrics=service_metrics,
            recommendations=recommendations
        )
    
    async def _analyze_rds_anomaly(
        self,
        resource_id: str,
        cost_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> ServiceAnomalyResult:
        """Analyze RDS-specific anomaly patterns"""
        
        baseline_cost = np.mean([d.get('cost', 0) for d in historical_data[-168:]])
        actual_cost = cost_data.get('cost', 0)
        deviation = ((actual_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        contributing_factors = []
        service_metrics = {}
        recommendations = []
        
        # Instance size changes
        current_instance_class = cost_data.get('instance_class', '')
        if 'xlarge' in current_instance_class or 'metal' in current_instance_class:
            contributing_factors.append("Large RDS instance class detected")
            recommendations.append("Evaluate if large instance class is necessary")
            service_metrics['instance_class'] = current_instance_class
        
        # Storage growth
        current_storage = cost_data.get('allocated_storage', 0)
        historical_storage = np.mean([d.get('allocated_storage', 0) for d in historical_data[-168:]])
        
        if current_storage > historical_storage * 1.2:
            contributing_factors.append("Database storage growth")
            recommendations.append("Review database growth patterns and archiving strategies")
            service_metrics['storage_growth'] = (current_storage - historical_storage) / historical_storage
        
        confidence_score = min(0.95, abs(deviation) / 100 + 0.7)
        
        return ServiceAnomalyResult(
            service_type=ServiceType.RDS,
            resource_id=resource_id,
            anomaly_type="rds_scaling_anomaly",
            confidence_score=confidence_score,
            cost_impact=actual_cost - baseline_cost,
            baseline_cost=baseline_cost,
            actual_cost=actual_cost,
            deviation_percentage=deviation,
            contributing_factors=contributing_factors,
            service_specific_metrics=service_metrics,
            recommendations=recommendations
        )
    
    async def _analyze_lambda_anomaly(
        self,
        resource_id: str,
        cost_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> ServiceAnomalyResult:
        """Analyze Lambda-specific anomaly patterns"""
        
        baseline_cost = np.mean([d.get('cost', 0) for d in historical_data[-24:]])  # 1 day baseline
        actual_cost = cost_data.get('cost', 0)
        deviation = ((actual_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        contributing_factors = []
        service_metrics = {}
        recommendations = []
        
        # Invocation spike analysis
        current_invocations = cost_data.get('invocations', 0)
        historical_invocations = np.mean([d.get('invocations', 0) for d in historical_data[-24:]])
        
        if current_invocations > historical_invocations * 3.0:
            contributing_factors.append("Significant invocation spike")
            recommendations.append("Investigate trigger sources and implement throttling if needed")
            service_metrics['invocation_spike_factor'] = current_invocations / historical_invocations if historical_invocations > 0 else 0
        
        # Duration analysis
        avg_duration = cost_data.get('avg_duration_ms', 0)
        if avg_duration > 10000:  # > 10 seconds
            contributing_factors.append("Long-running function executions")
            recommendations.append("Optimize function performance and consider timeout adjustments")
            service_metrics['avg_duration_ms'] = avg_duration
        
        confidence_score = min(0.95, abs(deviation) / 100 + 0.5)
        
        return ServiceAnomalyResult(
            service_type=ServiceType.LAMBDA,
            resource_id=resource_id,
            anomaly_type="lambda_invocation_anomaly",
            confidence_score=confidence_score,
            cost_impact=actual_cost - baseline_cost,
            baseline_cost=baseline_cost,
            actual_cost=actual_cost,
            deviation_percentage=deviation,
            contributing_factors=contributing_factors,
            service_specific_metrics=service_metrics,
            recommendations=recommendations
        )
    
    async def _analyze_ebs_anomaly(self, resource_id: str, cost_data: Dict[str, Any], 
                                 historical_data: List[Dict[str, Any]], context: Dict[str, Any]) -> ServiceAnomalyResult:
        """Analyze EBS-specific anomaly patterns"""
        return self._generic_analysis("ebs", resource_id, cost_data, historical_data)
    
    async def _analyze_cloudfront_anomaly(self, resource_id: str, cost_data: Dict[str, Any], 
                                        historical_data: List[Dict[str, Any]], context: Dict[str, Any]) -> ServiceAnomalyResult:
        """Analyze CloudFront-specific anomaly patterns"""
        return self._generic_analysis("cloudfront", resource_id, cost_data, historical_data)
    
    async def _analyze_elb_anomaly(self, resource_id: str, cost_data: Dict[str, Any], 
                                 historical_data: List[Dict[str, Any]], context: Dict[str, Any]) -> ServiceAnomalyResult:
        """Analyze ELB-specific anomaly patterns"""
        return self._generic_analysis("elb", resource_id, cost_data, historical_data)
    
    def _generic_analysis(
        self,
        service_type: str,
        resource_id: str,
        cost_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> ServiceAnomalyResult:
        """Generic analysis for unsupported services"""
        
        baseline_cost = np.mean([d.get('cost', 0) for d in historical_data[-168:]])
        actual_cost = cost_data.get('cost', 0)
        deviation = ((actual_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        return ServiceAnomalyResult(
            service_type=ServiceType.EC2,  # Default
            resource_id=resource_id,
            anomaly_type="generic_cost_anomaly",
            confidence_score=0.6,
            cost_impact=actual_cost - baseline_cost,
            baseline_cost=baseline_cost,
            actual_cost=actual_cost,
            deviation_percentage=deviation,
            contributing_factors=["Generic cost deviation detected"],
            service_specific_metrics={},
            recommendations=["Review service usage patterns and optimization opportunities"]
        )
    
    async def get_service_patterns(self, service_type: str) -> List[ServiceAnomalyPattern]:
        """Get anomaly patterns for a specific service"""
        try:
            service_enum = ServiceType(service_type.lower())
            return self.service_patterns.get(service_enum, [])
        except ValueError:
            return []