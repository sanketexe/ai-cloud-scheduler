# waste_detection_engine.py
"""
Advanced Waste Detection and Resource Optimization Engine

This module provides comprehensive waste detection capabilities including:
- Resource utilization analysis with industry benchmarking
- Intelligent waste identification for unused, idle, and oversized resources
- Optimization recommendations with savings calculations
- Risk assessment for optimization actions
- Automated optimization tracking and success measurement
"""

import json
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import math

try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from performance_monitor import (
    MetricType, MetricsData, CloudResource, ResourceCapacity,
    ScalingRecommendation, CloudProvider
)
from resource_discovery import ResourceMetadata, ResourceType, ResourceState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UtilizationLevel(Enum):
    """Resource utilization levels"""
    UNUSED = "unused"           # 0-5% utilization
    UNDERUTILIZED = "underutilized"  # 5-30% utilization
    OPTIMAL = "optimal"         # 30-70% utilization
    HIGH = "high"              # 70-85% utilization
    OVERUTILIZED = "overutilized"    # 85-100% utilization


class WasteType(Enum):
    """Types of resource waste"""
    UNUSED_RESOURCE = "unused_resource"
    UNDERUTILIZED_RESOURCE = "underutilized_resource"
    OVERSIZED_RESOURCE = "oversized_resource"
    IDLE_RESOURCE = "idle_resource"
    ORPHANED_RESOURCE = "orphaned_resource"
    ZOMBIE_RESOURCE = "zombie_resource"


class OptimizationType(Enum):
    """Types of optimization actions"""
    TERMINATE = "terminate"
    DOWNSIZE = "downsize"
    RIGHTSIZE = "rightsize"
    SCHEDULE = "schedule"
    MIGRATE = "migrate"
    CONSOLIDATE = "consolidate"


class RiskLevel(Enum):
    """Risk levels for optimization actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UtilizationMetrics:
    """Comprehensive utilization metrics for a resource"""
    resource_id: str
    measurement_period: Tuple[datetime, datetime]
    
    # Core utilization metrics
    cpu_utilization: Dict[str, float]  # avg, min, max, p95, p99
    memory_utilization: Dict[str, float]
    storage_utilization: Dict[str, float]
    network_utilization: Dict[str, float]
    
    # Derived metrics
    overall_utilization: float
    utilization_level: UtilizationLevel
    efficiency_score: float  # 0-100
    waste_score: float      # 0-100, higher = more waste
    
    # Pattern analysis
    usage_patterns: Dict[str, Any]
    peak_hours: List[int]  # Hours of day with peak usage
    idle_periods: List[Tuple[datetime, datetime]]
    
    # Benchmarking
    industry_benchmark: Optional[float] = None
    peer_comparison: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "resource_id": self.resource_id,
            "measurement_period": [
                self.measurement_period[0].isoformat(),
                self.measurement_period[1].isoformat()
            ],
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "storage_utilization": self.storage_utilization,
            "network_utilization": self.network_utilization,
            "overall_utilization": self.overall_utilization,
            "utilization_level": self.utilization_level.value,
            "efficiency_score": self.efficiency_score,
            "waste_score": self.waste_score,
            "usage_patterns": self.usage_patterns,
            "peak_hours": self.peak_hours,
            "idle_periods": [
                [period[0].isoformat(), period[1].isoformat()]
                for period in self.idle_periods
            ],
            "industry_benchmark": self.industry_benchmark,
            "peer_comparison": self.peer_comparison
        }


@dataclass
class IndustryBenchmark:
    """Industry benchmark data for resource types"""
    resource_type: str
    provider: CloudProvider
    
    # Benchmark utilization ranges
    optimal_cpu_range: Tuple[float, float]  # (min, max) for optimal utilization
    optimal_memory_range: Tuple[float, float]
    optimal_storage_range: Tuple[float, float]
    
    # Industry averages
    average_cpu_utilization: float
    average_memory_utilization: float
    average_efficiency_score: float
    
    # Cost efficiency metrics
    cost_per_unit_performance: float
    typical_waste_percentage: float
    
    last_updated: datetime = field(default_factory=datetime.now)


class ResourceAnalyzer:
    """
    Comprehensive resource utilization analysis system with industry benchmarking
    
    Provides detailed utilization metrics collection, pattern analysis,
    efficiency scoring, and benchmarking against industry standards.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ResourceAnalyzer")
        self.benchmarks: Dict[str, IndustryBenchmark] = {}
        self.utilization_history: Dict[str, List[UtilizationMetrics]] = {}
        self._load_industry_benchmarks()
    
    def analyze_resource_utilization(self, 
                                   resources: List[CloudResource],
                                   metrics_data: Dict[str, MetricsData],
                                   analysis_period_days: int = 30) -> Dict[str, UtilizationMetrics]:
        """
        Analyze comprehensive utilization metrics for resources
        
        Args:
            resources: List of cloud resources to analyze
            metrics_data: Historical metrics data for resources
            analysis_period_days: Period for analysis in days
            
        Returns:
            Dictionary mapping resource_id to UtilizationMetrics
        """
        self.logger.info(f"Analyzing utilization for {len(resources)} resources over {analysis_period_days} days")
        
        utilization_results = {}
        
        for resource in resources:
            if resource.resource_id not in metrics_data:
                self.logger.warning(f"No metrics data available for resource {resource.resource_id}")
                continue
            
            try:
                metrics = self._analyze_single_resource(
                    resource, 
                    metrics_data[resource.resource_id],
                    analysis_period_days
                )
                
                if metrics:
                    utilization_results[resource.resource_id] = metrics
                    
                    # Store in history
                    if resource.resource_id not in self.utilization_history:
                        self.utilization_history[resource.resource_id] = []
                    
                    self.utilization_history[resource.resource_id].append(metrics)
                    
                    # Keep history manageable (last 90 days of daily analyses)
                    if len(self.utilization_history[resource.resource_id]) > 90:
                        self.utilization_history[resource.resource_id] = \
                            self.utilization_history[resource.resource_id][-90:]
                            
            except Exception as e:
                self.logger.error(f"Error analyzing resource {resource.resource_id}: {str(e)}")
                continue
        
        self.logger.info(f"Completed utilization analysis for {len(utilization_results)} resources")
        return utilization_results
    
    def _analyze_single_resource(self, 
                               resource: CloudResource,
                               metrics_data: MetricsData,
                               analysis_period_days: int) -> Optional[UtilizationMetrics]:
        """Analyze utilization for a single resource"""
        
        # Calculate measurement period
        end_time = datetime.now()
        start_time = end_time - timedelta(days=analysis_period_days)
        measurement_period = (start_time, end_time)
        
        # Extract and calculate utilization metrics
        cpu_stats = self._calculate_metric_statistics(
            metrics_data.metrics.get(MetricType.CPU_UTILIZATION, []),
            start_time, end_time
        )
        
        memory_stats = self._calculate_metric_statistics(
            metrics_data.metrics.get(MetricType.MEMORY_UTILIZATION, []),
            start_time, end_time
        )
        
        # For storage and network, we need to handle different metric types
        storage_stats = self._calculate_storage_utilization(metrics_data, start_time, end_time)
        network_stats = self._calculate_network_utilization(metrics_data, start_time, end_time)
        
        # Calculate overall utilization (weighted average)
        overall_utilization = self._calculate_overall_utilization(
            cpu_stats, memory_stats, storage_stats, network_stats
        )
        
        # Determine utilization level
        utilization_level = self._determine_utilization_level(overall_utilization)
        
        # Calculate efficiency and waste scores
        efficiency_score = self._calculate_efficiency_score(
            resource, cpu_stats, memory_stats, storage_stats, network_stats
        )
        waste_score = 100 - efficiency_score
        
        # Analyze usage patterns
        usage_patterns = self._analyze_usage_patterns(metrics_data, start_time, end_time)
        peak_hours = self._identify_peak_hours(metrics_data, start_time, end_time)
        idle_periods = self._identify_idle_periods(metrics_data, start_time, end_time)
        
        # Get benchmark comparisons
        industry_benchmark = self._get_industry_benchmark_score(resource, overall_utilization)
        peer_comparison = self._get_peer_comparison_score(resource, overall_utilization)
        
        return UtilizationMetrics(
            resource_id=resource.resource_id,
            measurement_period=measurement_period,
            cpu_utilization=cpu_stats,
            memory_utilization=memory_stats,
            storage_utilization=storage_stats,
            network_utilization=network_stats,
            overall_utilization=overall_utilization,
            utilization_level=utilization_level,
            efficiency_score=efficiency_score,
            waste_score=waste_score,
            usage_patterns=usage_patterns,
            peak_hours=peak_hours,
            idle_periods=idle_periods,
            industry_benchmark=industry_benchmark,
            peer_comparison=peer_comparison
        )
    
    def _calculate_metric_statistics(self, 
                                   metric_data: List[Tuple[datetime, float]],
                                   start_time: datetime,
                                   end_time: datetime) -> Dict[str, float]:
        """Calculate comprehensive statistics for a metric"""
        
        # Filter data to analysis period
        filtered_data = [
            value for timestamp, value in metric_data
            if start_time <= timestamp <= end_time
        ]
        
        if not filtered_data:
            return {
                "avg": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0,
                "std": 0.0, "sample_count": 0
            }
        
        # Calculate statistics
        avg = statistics.mean(filtered_data)
        min_val = min(filtered_data)
        max_val = max(filtered_data)
        
        # Calculate percentiles
        sorted_data = sorted(filtered_data)
        n = len(sorted_data)
        
        p50 = sorted_data[int(n * 0.5)] if n > 0 else 0.0
        p95 = sorted_data[int(n * 0.95)] if n > 0 else 0.0
        p99 = sorted_data[int(n * 0.99)] if n > 0 else 0.0
        
        std = statistics.stdev(filtered_data) if len(filtered_data) > 1 else 0.0
        
        return {
            "avg": round(avg, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "std": round(std, 2),
            "sample_count": len(filtered_data)
        }
    
    def _calculate_storage_utilization(self, 
                                     metrics_data: MetricsData,
                                     start_time: datetime,
                                     end_time: datetime) -> Dict[str, float]:
        """Calculate storage utilization metrics"""
        
        # Look for disk I/O metrics as proxy for storage utilization
        disk_io_data = metrics_data.metrics.get(MetricType.DISK_IO, [])
        
        if not disk_io_data:
            # Return default values if no data available
            return {
                "avg": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0,
                "std": 0.0, "sample_count": 0
            }
        
        return self._calculate_metric_statistics(disk_io_data, start_time, end_time)
    
    def _calculate_network_utilization(self, 
                                     metrics_data: MetricsData,
                                     start_time: datetime,
                                     end_time: datetime) -> Dict[str, float]:
        """Calculate network utilization metrics"""
        
        # Look for network I/O metrics
        network_io_data = metrics_data.metrics.get(MetricType.NETWORK_IO, [])
        
        if not network_io_data:
            return {
                "avg": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0,
                "std": 0.0, "sample_count": 0
            }
        
        return self._calculate_metric_statistics(network_io_data, start_time, end_time)
    
    def _calculate_overall_utilization(self, 
                                     cpu_stats: Dict[str, float],
                                     memory_stats: Dict[str, float],
                                     storage_stats: Dict[str, float],
                                     network_stats: Dict[str, float]) -> float:
        """Calculate weighted overall utilization score"""
        
        # Weights for different resource types (can be configured)
        weights = {
            "cpu": 0.4,
            "memory": 0.3,
            "storage": 0.2,
            "network": 0.1
        }
        
        # Use average utilization for overall calculation
        cpu_util = cpu_stats.get("avg", 0.0)
        memory_util = memory_stats.get("avg", 0.0)
        storage_util = storage_stats.get("avg", 0.0)
        network_util = network_stats.get("avg", 0.0)
        
        overall = (
            cpu_util * weights["cpu"] +
            memory_util * weights["memory"] +
            storage_util * weights["storage"] +
            network_util * weights["network"]
        )
        
        return round(overall, 2)
    
    def _determine_utilization_level(self, overall_utilization: float) -> UtilizationLevel:
        """Determine utilization level based on overall utilization"""
        
        if overall_utilization <= 5.0:
            return UtilizationLevel.UNUSED
        elif overall_utilization <= 30.0:
            return UtilizationLevel.UNDERUTILIZED
        elif overall_utilization <= 70.0:
            return UtilizationLevel.OPTIMAL
        elif overall_utilization <= 85.0:
            return UtilizationLevel.HIGH
        else:
            return UtilizationLevel.OVERUTILIZED
    
    def _calculate_efficiency_score(self, 
                                  resource: CloudResource,
                                  cpu_stats: Dict[str, float],
                                  memory_stats: Dict[str, float],
                                  storage_stats: Dict[str, float],
                                  network_stats: Dict[str, float]) -> float:
        """Calculate efficiency score (0-100) based on utilization patterns"""
        
        # Get benchmark for this resource type
        benchmark_key = f"{resource.provider.value}_{resource.resource_type}"
        benchmark = self.benchmarks.get(benchmark_key)
        
        if not benchmark:
            # Use generic scoring if no benchmark available
            return self._calculate_generic_efficiency_score(cpu_stats, memory_stats)
        
        # Calculate efficiency based on how close utilization is to optimal ranges
        cpu_efficiency = self._calculate_range_efficiency(
            cpu_stats.get("avg", 0.0),
            benchmark.optimal_cpu_range
        )
        
        memory_efficiency = self._calculate_range_efficiency(
            memory_stats.get("avg", 0.0),
            benchmark.optimal_memory_range
        )
        
        storage_efficiency = self._calculate_range_efficiency(
            storage_stats.get("avg", 0.0),
            benchmark.optimal_storage_range
        )
        
        # Weighted average efficiency
        efficiency = (
            cpu_efficiency * 0.4 +
            memory_efficiency * 0.3 +
            storage_efficiency * 0.3
        )
        
        return round(efficiency, 2)
    
    def _calculate_generic_efficiency_score(self, 
                                          cpu_stats: Dict[str, float],
                                          memory_stats: Dict[str, float]) -> float:
        """Calculate generic efficiency score without benchmarks"""
        
        cpu_util = cpu_stats.get("avg", 0.0)
        memory_util = memory_stats.get("avg", 0.0)
        
        # Optimal range is 30-70% for most resources
        optimal_min, optimal_max = 30.0, 70.0
        
        cpu_efficiency = self._calculate_range_efficiency(cpu_util, (optimal_min, optimal_max))
        memory_efficiency = self._calculate_range_efficiency(memory_util, (optimal_min, optimal_max))
        
        return round((cpu_efficiency + memory_efficiency) / 2, 2)
    
    def _calculate_range_efficiency(self, value: float, optimal_range: Tuple[float, float]) -> float:
        """Calculate efficiency based on how close value is to optimal range"""
        
        min_optimal, max_optimal = optimal_range
        
        if min_optimal <= value <= max_optimal:
            # Value is in optimal range
            return 100.0
        elif value < min_optimal:
            # Underutilized - efficiency decreases as value approaches 0
            if value == 0:
                return 0.0
            return max(0.0, (value / min_optimal) * 100.0)
        else:
            # Overutilized - efficiency decreases as value exceeds optimal
            excess = value - max_optimal
            penalty = min(excess, 30.0)  # Cap penalty at 30 points
            return max(0.0, 100.0 - (penalty / 30.0) * 100.0)
    
    def _analyze_usage_patterns(self, 
                              metrics_data: MetricsData,
                              start_time: datetime,
                              end_time: datetime) -> Dict[str, Any]:
        """Analyze usage patterns and seasonality"""
        
        patterns = {
            "has_daily_pattern": False,
            "has_weekly_pattern": False,
            "peak_to_average_ratio": 1.0,
            "variability_score": 0.0,
            "predictability_score": 0.0
        }
        
        # Analyze CPU utilization patterns
        cpu_data = metrics_data.metrics.get(MetricType.CPU_UTILIZATION, [])
        if not cpu_data:
            return patterns
        
        # Filter to analysis period
        filtered_data = [
            (timestamp, value) for timestamp, value in cpu_data
            if start_time <= timestamp <= end_time
        ]
        
        if len(filtered_data) < 24:  # Need at least 24 hours of data
            return patterns
        
        values = [value for _, value in filtered_data]
        
        # Calculate variability
        if len(values) > 1:
            avg_value = statistics.mean(values)
            std_value = statistics.stdev(values)
            patterns["variability_score"] = round((std_value / avg_value) * 100 if avg_value > 0 else 0, 2)
            
            # Calculate peak to average ratio
            max_value = max(values)
            patterns["peak_to_average_ratio"] = round(max_value / avg_value if avg_value > 0 else 1.0, 2)
        
        # Simple pattern detection (would be more sophisticated in production)
        patterns["has_daily_pattern"] = patterns["variability_score"] > 20.0
        patterns["has_weekly_pattern"] = len(filtered_data) >= 168  # 7 days of hourly data
        
        # Predictability based on variability (inverse relationship)
        patterns["predictability_score"] = round(max(0, 100 - patterns["variability_score"]), 2)
        
        return patterns
    
    def _identify_peak_hours(self, 
                           metrics_data: MetricsData,
                           start_time: datetime,
                           end_time: datetime) -> List[int]:
        """Identify peak usage hours (0-23)"""
        
        cpu_data = metrics_data.metrics.get(MetricType.CPU_UTILIZATION, [])
        if not cpu_data:
            return []
        
        # Group data by hour of day
        hourly_averages = {}
        for timestamp, value in cpu_data:
            if start_time <= timestamp <= end_time:
                hour = timestamp.hour
                if hour not in hourly_averages:
                    hourly_averages[hour] = []
                hourly_averages[hour].append(value)
        
        # Calculate average for each hour
        hour_avgs = {}
        for hour, values in hourly_averages.items():
            hour_avgs[hour] = statistics.mean(values)
        
        if not hour_avgs:
            return []
        
        # Find hours with above-average utilization
        overall_avg = statistics.mean(hour_avgs.values())
        peak_hours = [
            hour for hour, avg in hour_avgs.items()
            if avg > overall_avg * 1.2  # 20% above average
        ]
        
        return sorted(peak_hours)
    
    def _identify_idle_periods(self, 
                             metrics_data: MetricsData,
                             start_time: datetime,
                             end_time: datetime) -> List[Tuple[datetime, datetime]]:
        """Identify periods of idle usage (very low utilization)"""
        
        cpu_data = metrics_data.metrics.get(MetricType.CPU_UTILIZATION, [])
        if not cpu_data:
            return []
        
        idle_threshold = 5.0  # Consider < 5% as idle
        min_idle_duration = timedelta(hours=2)  # Minimum 2 hours to be considered idle period
        
        idle_periods = []
        current_idle_start = None
        
        for timestamp, value in cpu_data:
            if start_time <= timestamp <= end_time:
                if value < idle_threshold:
                    if current_idle_start is None:
                        current_idle_start = timestamp
                else:
                    if current_idle_start is not None:
                        idle_duration = timestamp - current_idle_start
                        if idle_duration >= min_idle_duration:
                            idle_periods.append((current_idle_start, timestamp))
                        current_idle_start = None
        
        # Handle case where idle period extends to end of analysis
        if current_idle_start is not None:
            idle_duration = end_time - current_idle_start
            if idle_duration >= min_idle_duration:
                idle_periods.append((current_idle_start, end_time))
        
        return idle_periods
    
    def _get_industry_benchmark_score(self, resource: CloudResource, utilization: float) -> Optional[float]:
        """Get industry benchmark comparison score"""
        
        benchmark_key = f"{resource.provider.value}_{resource.resource_type}"
        benchmark = self.benchmarks.get(benchmark_key)
        
        if not benchmark:
            return None
        
        # Compare to industry average
        industry_avg = benchmark.average_cpu_utilization
        if industry_avg == 0:
            return None
        
        # Return percentage difference from industry average
        difference = ((utilization - industry_avg) / industry_avg) * 100
        return round(difference, 2)
    
    def _get_peer_comparison_score(self, resource: CloudResource, utilization: float) -> Optional[float]:
        """Get peer comparison score based on similar resources"""
        
        # Find similar resources in history
        similar_resources = []
        for resource_id, history in self.utilization_history.items():
            if resource_id != resource.resource_id and history:
                # Use most recent utilization data
                recent_utilization = history[-1].overall_utilization
                similar_resources.append(recent_utilization)
        
        if len(similar_resources) < 3:  # Need at least 3 peers for comparison
            return None
        
        peer_avg = statistics.mean(similar_resources)
        if peer_avg == 0:
            return None
        
        # Return percentage difference from peer average
        difference = ((utilization - peer_avg) / peer_avg) * 100
        return round(difference, 2)
    
    def _load_industry_benchmarks(self):
        """Load industry benchmark data"""
        
        # Default benchmarks (in production, these would be loaded from external sources)
        default_benchmarks = {
            "aws_ec2_instance": IndustryBenchmark(
                resource_type="ec2_instance",
                provider=CloudProvider.AWS,
                optimal_cpu_range=(30.0, 70.0),
                optimal_memory_range=(40.0, 80.0),
                optimal_storage_range=(20.0, 80.0),
                average_cpu_utilization=45.0,
                average_memory_utilization=60.0,
                average_efficiency_score=65.0,
                cost_per_unit_performance=0.10,
                typical_waste_percentage=35.0
            ),
            "gcp_compute_instance": IndustryBenchmark(
                resource_type="compute_instance",
                provider=CloudProvider.GCP,
                optimal_cpu_range=(35.0, 75.0),
                optimal_memory_range=(40.0, 80.0),
                optimal_storage_range=(25.0, 85.0),
                average_cpu_utilization=48.0,
                average_memory_utilization=62.0,
                average_efficiency_score=67.0,
                cost_per_unit_performance=0.09,
                typical_waste_percentage=32.0
            ),
            "azure_virtual_machine": IndustryBenchmark(
                resource_type="virtual_machine",
                provider=CloudProvider.AZURE,
                optimal_cpu_range=(32.0, 72.0),
                optimal_memory_range=(38.0, 78.0),
                optimal_storage_range=(22.0, 82.0),
                average_cpu_utilization=46.0,
                average_memory_utilization=58.0,
                average_efficiency_score=64.0,
                cost_per_unit_performance=0.11,
                typical_waste_percentage=36.0
            )
        }
        
        self.benchmarks.update(default_benchmarks)
        self.logger.info(f"Loaded {len(self.benchmarks)} industry benchmarks")
    
    def get_utilization_summary(self, resource_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get summary of utilization analysis results"""
        
        if resource_ids is None:
            # Get all resources with recent utilization data
            resource_ids = list(self.utilization_history.keys())
        
        if not resource_ids:
            return {"error": "No utilization data available"}
        
        # Get most recent utilization data for each resource
        recent_utilizations = []
        for resource_id in resource_ids:
            if resource_id in self.utilization_history and self.utilization_history[resource_id]:
                recent_utilizations.append(self.utilization_history[resource_id][-1])
        
        if not recent_utilizations:
            return {"error": "No recent utilization data available"}
        
        # Calculate summary statistics
        overall_utils = [u.overall_utilization for u in recent_utilizations]
        efficiency_scores = [u.efficiency_score for u in recent_utilizations]
        waste_scores = [u.waste_score for u in recent_utilizations]
        
        # Count by utilization level
        level_counts = {}
        for level in UtilizationLevel:
            level_counts[level.value] = sum(
                1 for u in recent_utilizations 
                if u.utilization_level == level
            )
        
        return {
            "total_resources_analyzed": len(recent_utilizations),
            "average_utilization": round(statistics.mean(overall_utils), 2),
            "average_efficiency_score": round(statistics.mean(efficiency_scores), 2),
            "average_waste_score": round(statistics.mean(waste_scores), 2),
            "utilization_distribution": level_counts,
            "resources_needing_attention": sum([
                level_counts.get("unused", 0),
                level_counts.get("underutilized", 0),
                level_counts.get("overutilized", 0)
            ]),
            "optimization_potential": round(statistics.mean(waste_scores), 2),
            "analysis_timestamp": datetime.now().isoformat()
        }


@dataclass
class WasteItem:
    """Represents a detected waste item"""
    resource_id: str
    waste_type: WasteType
    severity: RiskLevel
    
    # Waste details
    current_utilization: float
    expected_utilization: float
    waste_percentage: float
    
    # Cost impact
    current_monthly_cost: float
    wasted_monthly_cost: float
    potential_savings: float
    
    # Detection details
    detection_confidence: float  # 0-1
    detection_method: str
    evidence: Dict[str, Any]
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    resource_metadata: Optional[ResourceMetadata] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "resource_id": self.resource_id,
            "waste_type": self.waste_type.value,
            "severity": self.severity.value,
            "current_utilization": self.current_utilization,
            "expected_utilization": self.expected_utilization,
            "waste_percentage": self.waste_percentage,
            "current_monthly_cost": self.current_monthly_cost,
            "wasted_monthly_cost": self.wasted_monthly_cost,
            "potential_savings": self.potential_savings,
            "detection_confidence": self.detection_confidence,
            "detection_method": self.detection_method,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat(),
            "resource_metadata": self.resource_metadata.to_dict() if self.resource_metadata else None
        }


@dataclass
class WasteDetectionConfig:
    """Configuration for waste detection algorithms"""
    
    # Unused resource thresholds
    unused_cpu_threshold: float = 5.0      # % CPU utilization
    unused_memory_threshold: float = 10.0   # % Memory utilization
    unused_network_threshold: float = 1.0   # % Network utilization
    unused_duration_hours: int = 168        # 7 days
    
    # Underutilized resource thresholds
    underutilized_cpu_threshold: float = 30.0
    underutilized_memory_threshold: float = 40.0
    underutilized_duration_hours: int = 72  # 3 days
    
    # Idle resource detection
    idle_threshold: float = 2.0             # % utilization
    idle_duration_hours: int = 48           # 2 days
    
    # Zombie resource detection
    zombie_no_activity_hours: int = 336     # 14 days
    zombie_no_connections_hours: int = 168  # 7 days
    
    # Detection confidence thresholds
    min_confidence_threshold: float = 0.7
    high_confidence_threshold: float = 0.9


class WasteIdentifier:
    """
    Intelligent waste identification engine with multiple detection algorithms
    
    Identifies various types of waste including unused, underutilized, idle,
    and zombie resources using configurable thresholds and ML-based detection.
    """
    
    def __init__(self, config: Optional[WasteDetectionConfig] = None):
        self.logger = logging.getLogger(f"{__name__}.WasteIdentifier")
        self.config = config or WasteDetectionConfig()
        self.detection_history: List[WasteItem] = []
        
        # Initialize ML models if available
        self.anomaly_detector = None
        if SKLEARN_AVAILABLE:
            self._initialize_ml_models()
    
    def identify_waste(self, 
                      utilization_data: Dict[str, UtilizationMetrics],
                      resource_metadata: Dict[str, ResourceMetadata],
                      cost_data: Optional[Dict[str, float]] = None) -> List[WasteItem]:
        """
        Identify waste across all provided resources
        
        Args:
            utilization_data: Resource utilization metrics
            resource_metadata: Resource metadata for context
            cost_data: Optional cost data for savings calculations
            
        Returns:
            List of detected waste items
        """
        self.logger.info(f"Identifying waste across {len(utilization_data)} resources")
        
        waste_items = []
        
        for resource_id, utilization in utilization_data.items():
            metadata = resource_metadata.get(resource_id)
            monthly_cost = cost_data.get(resource_id, 0.0) if cost_data else 0.0
            
            # Run all waste detection algorithms
            resource_waste = self._detect_resource_waste(
                utilization, metadata, monthly_cost
            )
            
            waste_items.extend(resource_waste)
        
        # Store in history
        self.detection_history.extend(waste_items)
        
        # Keep history manageable (last 1000 detections)
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        # Sort by potential savings (highest first)
        waste_items.sort(key=lambda x: x.potential_savings, reverse=True)
        
        self.logger.info(f"Identified {len(waste_items)} waste items with total potential savings of ${sum(item.potential_savings for item in waste_items):.2f}/month")
        
        return waste_items
    
    def _detect_resource_waste(self, 
                             utilization: UtilizationMetrics,
                             metadata: Optional[ResourceMetadata],
                             monthly_cost: float) -> List[WasteItem]:
        """Detect waste for a single resource using multiple algorithms"""
        
        waste_items = []
        
        # 1. Unused resource detection
        unused_waste = self._detect_unused_resource(utilization, metadata, monthly_cost)
        if unused_waste:
            waste_items.append(unused_waste)
        
        # 2. Underutilized resource detection
        underutilized_waste = self._detect_underutilized_resource(utilization, metadata, monthly_cost)
        if underutilized_waste:
            waste_items.append(underutilized_waste)
        
        # 3. Idle resource detection
        idle_waste = self._detect_idle_resource(utilization, metadata, monthly_cost)
        if idle_waste:
            waste_items.append(idle_waste)
        
        # 4. Zombie resource detection
        zombie_waste = self._detect_zombie_resource(utilization, metadata, monthly_cost)
        if zombie_waste:
            waste_items.append(zombie_waste)
        
        # 5. ML-based anomaly detection (if available)
        if self.anomaly_detector and SKLEARN_AVAILABLE:
            anomaly_waste = self._detect_anomalous_waste(utilization, metadata, monthly_cost)
            if anomaly_waste:
                waste_items.append(anomaly_waste)
        
        return waste_items
    
    def _detect_unused_resource(self, 
                              utilization: UtilizationMetrics,
                              metadata: Optional[ResourceMetadata],
                              monthly_cost: float) -> Optional[WasteItem]:
        """Detect completely unused resources based on zero utilization patterns"""
        
        cpu_avg = utilization.cpu_utilization.get("avg", 0.0)
        memory_avg = utilization.memory_utilization.get("avg", 0.0)
        network_avg = utilization.network_utilization.get("avg", 0.0)
        
        # Check if resource meets unused criteria
        is_unused = (
            cpu_avg <= self.config.unused_cpu_threshold and
            memory_avg <= self.config.unused_memory_threshold and
            network_avg <= self.config.unused_network_threshold
        )
        
        if not is_unused:
            return None
        
        # Check duration - resource must be unused for minimum period
        measurement_duration = (
            utilization.measurement_period[1] - utilization.measurement_period[0]
        ).total_seconds() / 3600  # Convert to hours
        
        if measurement_duration < self.config.unused_duration_hours:
            return None
        
        # Calculate confidence based on consistency of low utilization
        cpu_max = utilization.cpu_utilization.get("max", 0.0)
        memory_max = utilization.memory_utilization.get("max", 0.0)
        
        # Higher confidence if max utilization is also very low
        confidence = 0.9 if (cpu_max <= 10.0 and memory_max <= 15.0) else 0.8
        
        # Calculate waste percentage (nearly 100% for unused resources)
        waste_percentage = 95.0
        wasted_cost = monthly_cost * (waste_percentage / 100.0)
        
        evidence = {
            "cpu_avg": cpu_avg,
            "memory_avg": memory_avg,
            "network_avg": network_avg,
            "cpu_max": cpu_max,
            "memory_max": memory_max,
            "measurement_duration_hours": measurement_duration,
            "idle_periods_count": len(utilization.idle_periods)
        }
        
        return WasteItem(
            resource_id=utilization.resource_id,
            waste_type=WasteType.UNUSED_RESOURCE,
            severity=RiskLevel.HIGH,
            current_utilization=utilization.overall_utilization,
            expected_utilization=0.0,
            waste_percentage=waste_percentage,
            current_monthly_cost=monthly_cost,
            wasted_monthly_cost=wasted_cost,
            potential_savings=wasted_cost,
            detection_confidence=confidence,
            detection_method="threshold_based_unused",
            evidence=evidence,
            resource_metadata=metadata
        )
    
    def _detect_underutilized_resource(self, 
                                     utilization: UtilizationMetrics,
                                     metadata: Optional[ResourceMetadata],
                                     monthly_cost: float) -> Optional[WasteItem]:
        """Detect underutilized resources with configurable thresholds"""
        
        cpu_avg = utilization.cpu_utilization.get("avg", 0.0)
        memory_avg = utilization.memory_utilization.get("avg", 0.0)
        
        # Check if resource is underutilized
        is_underutilized = (
            cpu_avg <= self.config.underutilized_cpu_threshold and
            memory_avg <= self.config.underutilized_memory_threshold and
            utilization.overall_utilization <= self.config.underutilized_cpu_threshold
        )
        
        if not is_underutilized:
            return None
        
        # Skip if already detected as unused (avoid duplicate detection)
        if (cpu_avg <= self.config.unused_cpu_threshold and 
            memory_avg <= self.config.unused_memory_threshold):
            return None
        
        # Check duration
        measurement_duration = (
            utilization.measurement_period[1] - utilization.measurement_period[0]
        ).total_seconds() / 3600
        
        if measurement_duration < self.config.underutilized_duration_hours:
            return None
        
        # Calculate confidence based on utilization consistency
        cpu_std = utilization.cpu_utilization.get("std", 0.0)
        memory_std = utilization.memory_utilization.get("std", 0.0)
        
        # Lower standard deviation = more consistent underutilization = higher confidence
        variability = (cpu_std + memory_std) / 2
        confidence = max(0.6, 0.9 - (variability / 50.0))  # Adjust based on variability
        
        # Calculate waste percentage based on how far below optimal utilization
        optimal_utilization = 50.0  # Assume 50% is optimal
        current_utilization = utilization.overall_utilization
        waste_percentage = ((optimal_utilization - current_utilization) / optimal_utilization) * 100
        waste_percentage = max(0, min(waste_percentage, 80))  # Cap at 80%
        
        wasted_cost = monthly_cost * (waste_percentage / 100.0)
        
        evidence = {
            "cpu_avg": cpu_avg,
            "memory_avg": memory_avg,
            "overall_utilization": current_utilization,
            "cpu_std": cpu_std,
            "memory_std": memory_std,
            "variability_score": utilization.usage_patterns.get("variability_score", 0),
            "measurement_duration_hours": measurement_duration
        }
        
        return WasteItem(
            resource_id=utilization.resource_id,
            waste_type=WasteType.UNDERUTILIZED_RESOURCE,
            severity=RiskLevel.MEDIUM,
            current_utilization=current_utilization,
            expected_utilization=optimal_utilization,
            waste_percentage=waste_percentage,
            current_monthly_cost=monthly_cost,
            wasted_monthly_cost=wasted_cost,
            potential_savings=wasted_cost * 0.6,  # Conservative savings estimate
            detection_confidence=confidence,
            detection_method="threshold_based_underutilized",
            evidence=evidence,
            resource_metadata=metadata
        )
    
    def _detect_idle_resource(self, 
                            utilization: UtilizationMetrics,
                            metadata: Optional[ResourceMetadata],
                            monthly_cost: float) -> Optional[WasteItem]:
        """Detect resources with extended idle periods"""
        
        # Check if resource has significant idle periods
        if not utilization.idle_periods:
            return None
        
        # Calculate total idle time
        total_idle_hours = sum(
            (end - start).total_seconds() / 3600
            for start, end in utilization.idle_periods
        )
        
        # Check if idle time exceeds threshold
        if total_idle_hours < self.config.idle_duration_hours:
            return None
        
        # Calculate idle percentage of total measurement period
        total_measurement_hours = (
            utilization.measurement_period[1] - utilization.measurement_period[0]
        ).total_seconds() / 3600
        
        idle_percentage = (total_idle_hours / total_measurement_hours) * 100
        
        # Only flag if idle time is significant (>20% of total time)
        if idle_percentage < 20.0:
            return None
        
        # Calculate confidence based on idle period consistency
        avg_idle_duration = total_idle_hours / len(utilization.idle_periods)
        confidence = min(0.9, 0.6 + (avg_idle_duration / 24.0) * 0.3)  # Higher confidence for longer idle periods
        
        # Calculate waste based on idle percentage
        waste_percentage = min(idle_percentage * 0.8, 70.0)  # Conservative estimate, cap at 70%
        wasted_cost = monthly_cost * (waste_percentage / 100.0)
        
        evidence = {
            "total_idle_hours": total_idle_hours,
            "idle_percentage": idle_percentage,
            "idle_periods_count": len(utilization.idle_periods),
            "avg_idle_duration_hours": avg_idle_duration,
            "longest_idle_period_hours": max(
                (end - start).total_seconds() / 3600
                for start, end in utilization.idle_periods
            )
        }
        
        return WasteItem(
            resource_id=utilization.resource_id,
            waste_type=WasteType.IDLE_RESOURCE,
            severity=RiskLevel.MEDIUM,
            current_utilization=utilization.overall_utilization,
            expected_utilization=utilization.overall_utilization * (1 + idle_percentage / 100),
            waste_percentage=waste_percentage,
            current_monthly_cost=monthly_cost,
            wasted_monthly_cost=wasted_cost,
            potential_savings=wasted_cost * 0.5,  # Conservative savings for scheduling optimization
            detection_confidence=confidence,
            detection_method="idle_period_analysis",
            evidence=evidence,
            resource_metadata=metadata
        )
    
    def _detect_zombie_resource(self, 
                              utilization: UtilizationMetrics,
                              metadata: Optional[ResourceMetadata],
                              monthly_cost: float) -> Optional[WasteItem]:
        """Detect zombie resources (running but not serving any purpose)"""
        
        # Zombie detection criteria:
        # 1. Very low utilization for extended period
        # 2. No network activity (suggests no external connections)
        # 3. Resource is in running state but appears abandoned
        
        cpu_avg = utilization.cpu_utilization.get("avg", 0.0)
        network_avg = utilization.network_utilization.get("avg", 0.0)
        
        # Check for zombie characteristics
        is_potential_zombie = (
            cpu_avg <= 3.0 and  # Very low CPU
            network_avg <= 0.5   # Almost no network activity
        )
        
        if not is_potential_zombie:
            return None
        
        # Check if resource has been in this state for extended period
        measurement_duration = (
            utilization.measurement_period[1] - utilization.measurement_period[0]
        ).total_seconds() / 3600
        
        if measurement_duration < self.config.zombie_no_activity_hours:
            return None
        
        # Additional zombie indicators
        zombie_indicators = 0
        evidence = {
            "cpu_avg": cpu_avg,
            "network_avg": network_avg,
            "measurement_duration_hours": measurement_duration
        }
        
        # Check for consistent low utilization
        cpu_max = utilization.cpu_utilization.get("max", 0.0)
        if cpu_max <= 10.0:
            zombie_indicators += 1
            evidence["cpu_max_very_low"] = True
        
        # Check for lack of usage patterns (suggests no regular activity)
        variability = utilization.usage_patterns.get("variability_score", 0)
        if variability <= 5.0:  # Very low variability suggests no real activity
            zombie_indicators += 1
            evidence["no_usage_patterns"] = True
        
        # Check if resource state suggests it should be active
        if metadata and metadata.state == ResourceState.RUNNING:
            zombie_indicators += 1
            evidence["running_but_inactive"] = True
        
        # Need at least 2 zombie indicators for detection
        if zombie_indicators < 2:
            return None
        
        # Calculate confidence based on number of indicators and duration
        confidence = min(0.9, 0.5 + (zombie_indicators * 0.15) + (measurement_duration / (24 * 30)) * 0.2)
        
        # Zombie resources are typically 90%+ waste
        waste_percentage = 90.0
        wasted_cost = monthly_cost * (waste_percentage / 100.0)
        
        evidence["zombie_indicators_count"] = zombie_indicators
        
        return WasteItem(
            resource_id=utilization.resource_id,
            waste_type=WasteType.ZOMBIE_RESOURCE,
            severity=RiskLevel.HIGH,
            current_utilization=utilization.overall_utilization,
            expected_utilization=0.0,
            waste_percentage=waste_percentage,
            current_monthly_cost=monthly_cost,
            wasted_monthly_cost=wasted_cost,
            potential_savings=wasted_cost,
            detection_confidence=confidence,
            detection_method="zombie_pattern_analysis",
            evidence=evidence,
            resource_metadata=metadata
        )
    
    def _detect_anomalous_waste(self, 
                              utilization: UtilizationMetrics,
                              metadata: Optional[ResourceMetadata],
                              monthly_cost: float) -> Optional[WasteItem]:
        """Detect waste using ML-based anomaly detection"""
        
        if not SKLEARN_AVAILABLE or not self.anomaly_detector:
            return None
        
        try:
            # Prepare features for anomaly detection
            features = [
                utilization.cpu_utilization.get("avg", 0.0),
                utilization.memory_utilization.get("avg", 0.0),
                utilization.storage_utilization.get("avg", 0.0),
                utilization.network_utilization.get("avg", 0.0),
                utilization.efficiency_score,
                utilization.usage_patterns.get("variability_score", 0),
                len(utilization.idle_periods),
                len(utilization.peak_hours)
            ]
            
            # Reshape for sklearn
            features_array = np.array(features).reshape(1, -1)
            
            # Predict anomaly (-1 for anomaly, 1 for normal)
            anomaly_score = self.anomaly_detector.predict(features_array)[0]
            
            if anomaly_score == 1:  # Normal resource, no anomaly detected
                return None
            
            # Calculate anomaly confidence (decision function gives distance from boundary)
            decision_score = self.anomaly_detector.decision_function(features_array)[0]
            confidence = min(0.9, max(0.6, abs(decision_score) / 2.0))
            
            # Estimate waste based on how anomalous the resource is
            waste_percentage = min(60.0, abs(decision_score) * 20)  # Conservative estimate
            wasted_cost = monthly_cost * (waste_percentage / 100.0)
            
            evidence = {
                "anomaly_score": float(anomaly_score),
                "decision_score": float(decision_score),
                "features_analyzed": features,
                "ml_model": "isolation_forest"
            }
            
            return WasteItem(
                resource_id=utilization.resource_id,
                waste_type=WasteType.UNDERUTILIZED_RESOURCE,  # Default to underutilized for ML detection
                severity=RiskLevel.MEDIUM,
                current_utilization=utilization.overall_utilization,
                expected_utilization=50.0,  # Assume optimal is 50%
                waste_percentage=waste_percentage,
                current_monthly_cost=monthly_cost,
                wasted_monthly_cost=wasted_cost,
                potential_savings=wasted_cost * 0.4,  # Conservative ML-based savings
                detection_confidence=confidence,
                detection_method="ml_anomaly_detection",
                evidence=evidence,
                resource_metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in ML-based waste detection for {utilization.resource_id}: {str(e)}")
            return None
    
    def _initialize_ml_models(self):
        """Initialize ML models for anomaly detection"""
        
        try:
            # Initialize Isolation Forest for anomaly detection
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # Expect 10% of resources to be anomalous
                random_state=42,
                n_estimators=100
            )
            
            # In production, you would train this on historical data
            # For now, we'll use synthetic training data
            self._train_anomaly_detector()
            
            self.logger.info("Initialized ML models for waste detection")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {str(e)}")
            self.anomaly_detector = None
    
    def _train_anomaly_detector(self):
        """Train the anomaly detector with synthetic data"""
        
        if not self.anomaly_detector:
            return
        
        # Generate synthetic training data representing normal resource utilization
        np.random.seed(42)
        n_samples = 1000
        
        # Normal resources (80% of data)
        normal_samples = int(n_samples * 0.8)
        normal_data = []
        
        for _ in range(normal_samples):
            # Generate realistic normal utilization patterns
            cpu = np.random.normal(45, 15)  # Mean 45%, std 15%
            memory = np.random.normal(55, 20)  # Mean 55%, std 20%
            storage = np.random.normal(40, 25)  # Mean 40%, std 25%
            network = np.random.normal(30, 20)  # Mean 30%, std 20%
            efficiency = np.random.normal(70, 15)  # Mean 70%, std 15%
            variability = np.random.normal(25, 10)  # Mean 25%, std 10%
            idle_periods = np.random.poisson(2)  # Average 2 idle periods
            peak_hours = np.random.poisson(8)  # Average 8 peak hours
            
            # Ensure values are within reasonable bounds
            cpu = max(0, min(100, cpu))
            memory = max(0, min(100, memory))
            storage = max(0, min(100, storage))
            network = max(0, min(100, network))
            efficiency = max(0, min(100, efficiency))
            variability = max(0, min(100, variability))
            idle_periods = max(0, min(20, idle_periods))
            peak_hours = max(0, min(24, peak_hours))
            
            normal_data.append([cpu, memory, storage, network, efficiency, variability, idle_periods, peak_hours])
        
        # Anomalous resources (20% of data)
        anomalous_samples = n_samples - normal_samples
        anomalous_data = []
        
        for _ in range(anomalous_samples):
            # Generate anomalous patterns (very low or very high utilization)
            if np.random.random() < 0.7:  # 70% are underutilized anomalies
                cpu = np.random.uniform(0, 10)  # Very low CPU
                memory = np.random.uniform(0, 15)  # Very low memory
                storage = np.random.uniform(0, 20)  # Low storage
                network = np.random.uniform(0, 5)  # Very low network
                efficiency = np.random.uniform(0, 30)  # Low efficiency
                variability = np.random.uniform(0, 10)  # Low variability
                idle_periods = np.random.poisson(10)  # Many idle periods
                peak_hours = np.random.poisson(2)  # Few peak hours
            else:  # 30% are overutilized anomalies
                cpu = np.random.uniform(85, 100)  # Very high CPU
                memory = np.random.uniform(85, 100)  # Very high memory
                storage = np.random.uniform(80, 100)  # High storage
                network = np.random.uniform(70, 100)  # High network
                efficiency = np.random.uniform(20, 50)  # Low efficiency due to overutilization
                variability = np.random.uniform(40, 80)  # High variability
                idle_periods = 0  # No idle periods
                peak_hours = np.random.poisson(20)  # Many peak hours
            
            anomalous_data.append([cpu, memory, storage, network, efficiency, variability, idle_periods, peak_hours])
        
        # Combine and train
        training_data = np.array(normal_data + anomalous_data)
        
        # Fit the model
        self.anomaly_detector.fit(training_data)
        
        self.logger.info(f"Trained anomaly detector with {len(training_data)} samples")
    
    def get_waste_summary(self, waste_items: Optional[List[WasteItem]] = None) -> Dict[str, Any]:
        """Get summary of waste detection results"""
        
        if waste_items is None:
            waste_items = self.detection_history
        
        if not waste_items:
            return {"error": "No waste items to summarize"}
        
        # Calculate summary statistics
        total_potential_savings = sum(item.potential_savings for item in waste_items)
        total_wasted_cost = sum(item.wasted_monthly_cost for item in waste_items)
        
        # Count by waste type
        waste_type_counts = {}
        waste_type_savings = {}
        for waste_type in WasteType:
            type_items = [item for item in waste_items if item.waste_type == waste_type]
            waste_type_counts[waste_type.value] = len(type_items)
            waste_type_savings[waste_type.value] = sum(item.potential_savings for item in type_items)
        
        # Count by severity
        severity_counts = {}
        for severity in RiskLevel:
            severity_counts[severity.value] = sum(
                1 for item in waste_items if item.severity == severity
            )
        
        # Calculate average confidence
        avg_confidence = statistics.mean([item.detection_confidence for item in waste_items])
        
        # Find top waste items
        top_waste_items = sorted(waste_items, key=lambda x: x.potential_savings, reverse=True)[:10]
        
        return {
            "total_waste_items": len(waste_items),
            "total_potential_monthly_savings": round(total_potential_savings, 2),
            "total_monthly_waste": round(total_wasted_cost, 2),
            "waste_by_type": {
                "counts": waste_type_counts,
                "savings": {k: round(v, 2) for k, v in waste_type_savings.items()}
            },
            "waste_by_severity": severity_counts,
            "average_detection_confidence": round(avg_confidence, 3),
            "top_waste_opportunities": [
                {
                    "resource_id": item.resource_id,
                    "waste_type": item.waste_type.value,
                    "potential_savings": round(item.potential_savings, 2),
                    "confidence": round(item.detection_confidence, 3)
                }
                for item in top_waste_items
            ],
            "analysis_timestamp": datetime.now().isoformat()
        }

@dataclass
class RightSizingRecommendation:
    """Recommendation for right-sizing a resource"""
    resource_id: str
    current_capacity: ResourceCapacity
    recommended_capacity: ResourceCapacity
    
    # Sizing analysis
    oversizing_percentage: float
    utilization_headroom: float  # How much capacity is unused
    
    # Cost impact
    current_monthly_cost: float
    recommended_monthly_cost: float
    monthly_savings: float
    annual_savings: float
    
    # Risk assessment
    risk_level: RiskLevel
    confidence_score: float
    
    # Supporting data
    analysis_period_days: int
    peak_utilization: Dict[str, float]  # Peak utilization for each metric
    recommendation_rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "resource_id": self.resource_id,
            "current_capacity": {
                "cpu_cores": self.current_capacity.cpu_cores,
                "memory_gb": self.current_capacity.memory_gb,
                "disk_gb": self.current_capacity.disk_gb,
                "network_bandwidth_mbps": self.current_capacity.network_bandwidth_mbps,
                "instance_type": self.current_capacity.instance_type
            },
            "recommended_capacity": {
                "cpu_cores": self.recommended_capacity.cpu_cores,
                "memory_gb": self.recommended_capacity.memory_gb,
                "disk_gb": self.recommended_capacity.disk_gb,
                "network_bandwidth_mbps": self.recommended_capacity.network_bandwidth_mbps,
                "instance_type": self.recommended_capacity.instance_type
            },
            "oversizing_percentage": self.oversizing_percentage,
            "utilization_headroom": self.utilization_headroom,
            "current_monthly_cost": self.current_monthly_cost,
            "recommended_monthly_cost": self.recommended_monthly_cost,
            "monthly_savings": self.monthly_savings,
            "annual_savings": self.annual_savings,
            "risk_level": self.risk_level.value,
            "confidence_score": self.confidence_score,
            "analysis_period_days": self.analysis_period_days,
            "peak_utilization": self.peak_utilization,
            "recommendation_rationale": self.recommendation_rationale
        }


class OversizedResourceDetector:
    """
    Oversized resource detection system for right-sizing analysis
    
    Analyzes compute instances, storage volumes, and network resources
    to identify oversized configurations and provide right-sizing recommendations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.OversizedResourceDetector")
        self.instance_types = self._load_instance_type_catalog()
        self.pricing_data = self._load_pricing_data()
    
    def detect_oversized_resources(self, 
                                 utilization_data: Dict[str, UtilizationMetrics],
                                 resource_metadata: Dict[str, ResourceMetadata],
                                 capacity_data: Dict[str, ResourceCapacity],
                                 cost_data: Optional[Dict[str, float]] = None) -> List[RightSizingRecommendation]:
        """
        Detect oversized resources and generate right-sizing recommendations
        
        Args:
            utilization_data: Resource utilization metrics
            resource_metadata: Resource metadata
            capacity_data: Current resource capacity configurations
            cost_data: Optional cost data for savings calculations
            
        Returns:
            List of right-sizing recommendations
        """
        self.logger.info(f"Analyzing {len(utilization_data)} resources for oversizing")
        
        recommendations = []
        
        for resource_id, utilization in utilization_data.items():
            metadata = resource_metadata.get(resource_id)
            capacity = capacity_data.get(resource_id)
            monthly_cost = cost_data.get(resource_id, 0.0) if cost_data else 0.0
            
            if not capacity:
                self.logger.warning(f"No capacity data available for resource {resource_id}")
                continue
            
            # Analyze different resource types
            if metadata and metadata.resource_type == ResourceType.COMPUTE_INSTANCE:
                recommendation = self._analyze_compute_instance_sizing(
                    resource_id, utilization, metadata, capacity, monthly_cost
                )
            elif metadata and metadata.resource_type == ResourceType.STORAGE_VOLUME:
                recommendation = self._analyze_storage_volume_sizing(
                    resource_id, utilization, metadata, capacity, monthly_cost
                )
            elif metadata and metadata.resource_type == ResourceType.LOAD_BALANCER:
                recommendation = self._analyze_load_balancer_sizing(
                    resource_id, utilization, metadata, capacity, monthly_cost
                )
            else:
                # Generic analysis for other resource types
                recommendation = self._analyze_generic_resource_sizing(
                    resource_id, utilization, metadata, capacity, monthly_cost
                )
            
            if recommendation:
                recommendations.append(recommendation)
        
        # Sort by potential savings (highest first)
        recommendations.sort(key=lambda x: x.monthly_savings, reverse=True)
        
        total_savings = sum(rec.monthly_savings for rec in recommendations)
        self.logger.info(f"Generated {len(recommendations)} right-sizing recommendations with total potential savings of ${total_savings:.2f}/month")
        
        return recommendations
    
    def _analyze_compute_instance_sizing(self, 
                                       resource_id: str,
                                       utilization: UtilizationMetrics,
                                       metadata: ResourceMetadata,
                                       capacity: ResourceCapacity,
                                       monthly_cost: float) -> Optional[RightSizingRecommendation]:
        """Analyze compute instance for right-sizing opportunities"""
        
        # Get peak utilization values (use P95 to account for spikes)
        cpu_peak = utilization.cpu_utilization.get("p95", 0.0)
        memory_peak = utilization.memory_utilization.get("p95", 0.0)
        
        # Calculate required capacity with safety buffer (20% headroom)
        safety_buffer = 1.2
        required_cpu_percent = cpu_peak * safety_buffer
        required_memory_percent = memory_peak * safety_buffer
        
        # Convert percentages to actual capacity requirements
        required_cpu_cores = math.ceil((required_cpu_percent / 100.0) * capacity.cpu_cores)
        required_memory_gb = math.ceil((required_memory_percent / 100.0) * capacity.memory_gb)
        
        # Ensure minimum viable capacity
        required_cpu_cores = max(1, required_cpu_cores)
        required_memory_gb = max(1, required_memory_gb)
        
        # Check if current capacity is oversized
        cpu_oversized = capacity.cpu_cores > required_cpu_cores
        memory_oversized = capacity.memory_gb > required_memory_gb
        
        if not (cpu_oversized or memory_oversized):
            return None  # Resource is appropriately sized
        
        # Find optimal instance type
        recommended_instance = self._find_optimal_instance_type(
            required_cpu_cores, required_memory_gb, capacity.instance_type
        )
        
        if not recommended_instance:
            return None  # No suitable smaller instance found
        
        # Calculate oversizing percentage
        cpu_oversizing = ((capacity.cpu_cores - required_cpu_cores) / capacity.cpu_cores) * 100
        memory_oversizing = ((capacity.memory_gb - required_memory_gb) / capacity.memory_gb) * 100
        overall_oversizing = max(cpu_oversizing, memory_oversizing)
        
        # Calculate utilization headroom
        cpu_headroom = 100 - cpu_peak
        memory_headroom = 100 - memory_peak
        overall_headroom = max(cpu_headroom, memory_headroom)
        
        # Calculate cost savings
        current_cost = monthly_cost
        recommended_cost = self._estimate_instance_cost(recommended_instance)
        monthly_savings = max(0, current_cost - recommended_cost)
        
        # Assess risk level
        risk_level = self._assess_rightsizing_risk(
            cpu_peak, memory_peak, utilization.usage_patterns
        )
        
        # Calculate confidence score
        confidence = self._calculate_rightsizing_confidence(
            utilization, overall_oversizing, risk_level
        )
        
        # Generate rationale
        rationale = self._generate_compute_rationale(
            capacity, recommended_instance, cpu_peak, memory_peak, overall_oversizing
        )
        
        return RightSizingRecommendation(
            resource_id=resource_id,
            current_capacity=capacity,
            recommended_capacity=ResourceCapacity(
                cpu_cores=recommended_instance["cpu_cores"],
                memory_gb=recommended_instance["memory_gb"],
                disk_gb=capacity.disk_gb,  # Keep same disk size
                network_bandwidth_mbps=recommended_instance.get("network_bandwidth_mbps", capacity.network_bandwidth_mbps),
                instance_type=recommended_instance["instance_type"]
            ),
            oversizing_percentage=overall_oversizing,
            utilization_headroom=overall_headroom,
            current_monthly_cost=current_cost,
            recommended_monthly_cost=recommended_cost,
            monthly_savings=monthly_savings,
            annual_savings=monthly_savings * 12,
            risk_level=risk_level,
            confidence_score=confidence,
            analysis_period_days=30,  # Default analysis period
            peak_utilization={
                "cpu_p95": cpu_peak,
                "memory_p95": memory_peak
            },
            recommendation_rationale=rationale
        )
    
    def _analyze_storage_volume_sizing(self, 
                                     resource_id: str,
                                     utilization: UtilizationMetrics,
                                     metadata: ResourceMetadata,
                                     capacity: ResourceCapacity,
                                     monthly_cost: float) -> Optional[RightSizingRecommendation]:
        """Analyze storage volume for right-sizing opportunities"""
        
        # Get storage utilization (use average as storage doesn't spike like CPU)
        storage_utilization = utilization.storage_utilization.get("avg", 0.0)
        storage_peak = utilization.storage_utilization.get("max", 0.0)
        
        # Calculate required storage with growth buffer (30% for storage)
        growth_buffer = 1.3
        required_storage_percent = storage_peak * growth_buffer
        required_storage_gb = math.ceil((required_storage_percent / 100.0) * capacity.disk_gb)
        
        # Ensure minimum viable storage (10GB minimum)
        required_storage_gb = max(10, required_storage_gb)
        
        # Check if current storage is oversized (need at least 20% oversizing to recommend)
        if capacity.disk_gb <= required_storage_gb * 1.2:
            return None
        
        # Calculate oversizing percentage
        oversizing_percentage = ((capacity.disk_gb - required_storage_gb) / capacity.disk_gb) * 100
        
        # Calculate cost savings (storage cost is typically proportional to size)
        storage_cost_per_gb = monthly_cost / capacity.disk_gb if capacity.disk_gb > 0 else 0
        recommended_cost = required_storage_gb * storage_cost_per_gb
        monthly_savings = max(0, monthly_cost - recommended_cost)
        
        # Storage right-sizing is generally low risk
        risk_level = RiskLevel.LOW
        
        # Higher confidence for storage as it's more predictable
        confidence = min(0.9, 0.7 + (oversizing_percentage / 100.0))
        
        rationale = f"Storage volume is {oversizing_percentage:.1f}% oversized. Current utilization is {storage_utilization:.1f}% with peak at {storage_peak:.1f}%. Recommended size: {required_storage_gb}GB (down from {capacity.disk_gb}GB)."
        
        return RightSizingRecommendation(
            resource_id=resource_id,
            current_capacity=capacity,
            recommended_capacity=ResourceCapacity(
                cpu_cores=capacity.cpu_cores,
                memory_gb=capacity.memory_gb,
                disk_gb=required_storage_gb,
                network_bandwidth_mbps=capacity.network_bandwidth_mbps,
                instance_type=capacity.instance_type
            ),
            oversizing_percentage=oversizing_percentage,
            utilization_headroom=100 - storage_peak,
            current_monthly_cost=monthly_cost,
            recommended_monthly_cost=recommended_cost,
            monthly_savings=monthly_savings,
            annual_savings=monthly_savings * 12,
            risk_level=risk_level,
            confidence_score=confidence,
            analysis_period_days=30,
            peak_utilization={
                "storage_avg": storage_utilization,
                "storage_peak": storage_peak
            },
            recommendation_rationale=rationale
        )
    
    def _analyze_load_balancer_sizing(self, 
                                    resource_id: str,
                                    utilization: UtilizationMetrics,
                                    metadata: ResourceMetadata,
                                    capacity: ResourceCapacity,
                                    monthly_cost: float) -> Optional[RightSizingRecommendation]:
        """Analyze load balancer for right-sizing opportunities"""
        
        # Get network utilization metrics
        network_avg = utilization.network_utilization.get("avg", 0.0)
        network_peak = utilization.network_utilization.get("p95", 0.0)
        
        # Calculate required bandwidth with safety buffer (50% for network spikes)
        safety_buffer = 1.5
        required_bandwidth_percent = network_peak * safety_buffer
        required_bandwidth_mbps = math.ceil((required_bandwidth_percent / 100.0) * capacity.network_bandwidth_mbps)
        
        # Ensure minimum viable bandwidth (10 Mbps minimum)
        required_bandwidth_mbps = max(10, required_bandwidth_mbps)
        
        # Check if current bandwidth is oversized (need at least 30% oversizing)
        if capacity.network_bandwidth_mbps <= required_bandwidth_mbps * 1.3:
            return None
        
        # Calculate oversizing percentage
        oversizing_percentage = ((capacity.network_bandwidth_mbps - required_bandwidth_mbps) / capacity.network_bandwidth_mbps) * 100
        
        # Estimate cost savings (network costs are often tiered)
        cost_reduction_factor = min(0.5, oversizing_percentage / 100.0)  # Cap at 50% savings
        recommended_cost = monthly_cost * (1 - cost_reduction_factor)
        monthly_savings = monthly_cost - recommended_cost
        
        # Network right-sizing has medium risk due to traffic spikes
        risk_level = RiskLevel.MEDIUM if network_peak > 70 else RiskLevel.LOW
        
        # Confidence based on traffic patterns
        traffic_variability = utilization.usage_patterns.get("variability_score", 0)
        confidence = max(0.6, 0.9 - (traffic_variability / 100.0))
        
        rationale = f"Load balancer bandwidth is {oversizing_percentage:.1f}% oversized. Average utilization is {network_avg:.1f}% with P95 at {network_peak:.1f}%. Recommended bandwidth: {required_bandwidth_mbps} Mbps (down from {capacity.network_bandwidth_mbps} Mbps)."
        
        return RightSizingRecommendation(
            resource_id=resource_id,
            current_capacity=capacity,
            recommended_capacity=ResourceCapacity(
                cpu_cores=capacity.cpu_cores,
                memory_gb=capacity.memory_gb,
                disk_gb=capacity.disk_gb,
                network_bandwidth_mbps=required_bandwidth_mbps,
                instance_type=capacity.instance_type
            ),
            oversizing_percentage=oversizing_percentage,
            utilization_headroom=100 - network_peak,
            current_monthly_cost=monthly_cost,
            recommended_monthly_cost=recommended_cost,
            monthly_savings=monthly_savings,
            annual_savings=monthly_savings * 12,
            risk_level=risk_level,
            confidence_score=confidence,
            analysis_period_days=30,
            peak_utilization={
                "network_avg": network_avg,
                "network_p95": network_peak
            },
            recommendation_rationale=rationale
        )
    
    def _analyze_generic_resource_sizing(self, 
                                       resource_id: str,
                                       utilization: UtilizationMetrics,
                                       metadata: Optional[ResourceMetadata],
                                       capacity: ResourceCapacity,
                                       monthly_cost: float) -> Optional[RightSizingRecommendation]:
        """Generic analysis for other resource types"""
        
        # Use overall utilization for generic analysis
        overall_utilization = utilization.overall_utilization
        
        # Consider resource oversized if utilization is consistently low
        if overall_utilization > 40.0:  # Not oversized if utilization > 40%
            return None
        
        # Calculate potential downsizing based on utilization
        target_utilization = 60.0  # Target 60% utilization
        sizing_factor = overall_utilization / target_utilization
        
        # Conservative approach - don't recommend more than 50% reduction
        sizing_factor = max(0.5, sizing_factor)
        
        oversizing_percentage = (1 - sizing_factor) * 100
        
        # Estimate cost savings proportional to sizing reduction
        recommended_cost = monthly_cost * sizing_factor
        monthly_savings = monthly_cost - recommended_cost
        
        # Generic resources have medium risk
        risk_level = RiskLevel.MEDIUM
        
        # Lower confidence for generic analysis
        confidence = 0.6
        
        rationale = f"Resource shows consistently low utilization ({overall_utilization:.1f}%). Generic right-sizing analysis suggests {oversizing_percentage:.1f}% reduction potential."
        
        return RightSizingRecommendation(
            resource_id=resource_id,
            current_capacity=capacity,
            recommended_capacity=ResourceCapacity(
                cpu_cores=max(1, int(capacity.cpu_cores * sizing_factor)),
                memory_gb=max(1, capacity.memory_gb * sizing_factor),
                disk_gb=max(10, capacity.disk_gb * sizing_factor),
                network_bandwidth_mbps=max(10, capacity.network_bandwidth_mbps * sizing_factor),
                instance_type=f"rightsized_{capacity.instance_type}"
            ),
            oversizing_percentage=oversizing_percentage,
            utilization_headroom=100 - overall_utilization,
            current_monthly_cost=monthly_cost,
            recommended_monthly_cost=recommended_cost,
            monthly_savings=monthly_savings,
            annual_savings=monthly_savings * 12,
            risk_level=risk_level,
            confidence_score=confidence,
            analysis_period_days=30,
            peak_utilization={
                "overall_utilization": overall_utilization
            },
            recommendation_rationale=rationale
        )
    
    def _find_optimal_instance_type(self, 
                                  required_cpu: int,
                                  required_memory: float,
                                  current_instance_type: str) -> Optional[Dict[str, Any]]:
        """Find optimal instance type for required capacity"""
        
        # Get current instance family (e.g., 'm5' from 'm5.large')
        current_family = current_instance_type.split('.')[0] if '.' in current_instance_type else current_instance_type
        
        # Look for instances in the same family first
        suitable_instances = []
        
        for instance_type, specs in self.instance_types.items():
            if (specs["cpu_cores"] >= required_cpu and 
                specs["memory_gb"] >= required_memory and
                specs["cpu_cores"] < self.instance_types.get(current_instance_type, {}).get("cpu_cores", float('inf'))):
                
                # Prefer same family instances
                instance_family = instance_type.split('.')[0] if '.' in instance_type else instance_type
                family_match = instance_family == current_family
                
                suitable_instances.append({
                    "instance_type": instance_type,
                    "cpu_cores": specs["cpu_cores"],
                    "memory_gb": specs["memory_gb"],
                    "network_bandwidth_mbps": specs.get("network_bandwidth_mbps", 1000),
                    "cost_per_hour": specs.get("cost_per_hour", 0.1),
                    "family_match": family_match
                })
        
        if not suitable_instances:
            return None
        
        # Sort by family match first, then by cost efficiency
        suitable_instances.sort(key=lambda x: (not x["family_match"], x["cost_per_hour"]))
        
        return suitable_instances[0]
    
    def _assess_rightsizing_risk(self, 
                               cpu_peak: float,
                               memory_peak: float,
                               usage_patterns: Dict[str, Any]) -> RiskLevel:
        """Assess risk level for right-sizing recommendation"""
        
        # High utilization = higher risk
        if cpu_peak > 80 or memory_peak > 85:
            return RiskLevel.HIGH
        
        # High variability = higher risk
        variability = usage_patterns.get("variability_score", 0)
        if variability > 50:
            return RiskLevel.MEDIUM
        
        # Unpredictable patterns = higher risk
        predictability = usage_patterns.get("predictability_score", 100)
        if predictability < 60:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _calculate_rightsizing_confidence(self, 
                                        utilization: UtilizationMetrics,
                                        oversizing_percentage: float,
                                        risk_level: RiskLevel) -> float:
        """Calculate confidence score for right-sizing recommendation"""
        
        base_confidence = 0.7
        
        # Higher oversizing = higher confidence
        oversizing_bonus = min(0.2, oversizing_percentage / 100.0)
        
        # More data samples = higher confidence
        sample_count = utilization.cpu_utilization.get("sample_count", 0)
        sample_bonus = min(0.1, sample_count / 1000.0)
        
        # Lower risk = higher confidence
        risk_penalty = {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: -0.1,
            RiskLevel.HIGH: -0.2,
            RiskLevel.CRITICAL: -0.3
        }.get(risk_level, 0.0)
        
        confidence = base_confidence + oversizing_bonus + sample_bonus + risk_penalty
        
        return max(0.5, min(0.95, confidence))
    
    def _generate_compute_rationale(self, 
                                  current_capacity: ResourceCapacity,
                                  recommended_instance: Dict[str, Any],
                                  cpu_peak: float,
                                  memory_peak: float,
                                  oversizing_percentage: float) -> str:
        """Generate human-readable rationale for compute right-sizing"""
        
        return (f"Instance {current_capacity.instance_type} is {oversizing_percentage:.1f}% oversized. "
                f"Peak utilization: CPU {cpu_peak:.1f}%, Memory {memory_peak:.1f}%. "
                f"Recommended instance: {recommended_instance['instance_type']} "
                f"({recommended_instance['cpu_cores']} vCPUs, {recommended_instance['memory_gb']:.1f}GB RAM) "
                f"provides adequate capacity with 20% safety buffer.")
    
    def _estimate_instance_cost(self, instance_specs: Dict[str, Any]) -> float:
        """Estimate monthly cost for an instance type"""
        
        hourly_cost = instance_specs.get("cost_per_hour", 0.1)
        hours_per_month = 24 * 30  # Assume 30 days per month
        
        return hourly_cost * hours_per_month
    
    def _load_instance_type_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load instance type catalog with specifications"""
        
        # Simplified instance catalog (in production, this would be loaded from cloud provider APIs)
        return {
            # AWS EC2 instances (simplified)
            "t3.nano": {"cpu_cores": 2, "memory_gb": 0.5, "cost_per_hour": 0.0052},
            "t3.micro": {"cpu_cores": 2, "memory_gb": 1, "cost_per_hour": 0.0104},
            "t3.small": {"cpu_cores": 2, "memory_gb": 2, "cost_per_hour": 0.0208},
            "t3.medium": {"cpu_cores": 2, "memory_gb": 4, "cost_per_hour": 0.0416},
            "t3.large": {"cpu_cores": 2, "memory_gb": 8, "cost_per_hour": 0.0832},
            "m5.large": {"cpu_cores": 2, "memory_gb": 8, "cost_per_hour": 0.096},
            "m5.xlarge": {"cpu_cores": 4, "memory_gb": 16, "cost_per_hour": 0.192},
            "m5.2xlarge": {"cpu_cores": 8, "memory_gb": 32, "cost_per_hour": 0.384},
            "m5.4xlarge": {"cpu_cores": 16, "memory_gb": 64, "cost_per_hour": 0.768},
            "c5.large": {"cpu_cores": 2, "memory_gb": 4, "cost_per_hour": 0.085},
            "c5.xlarge": {"cpu_cores": 4, "memory_gb": 8, "cost_per_hour": 0.17},
            "c5.2xlarge": {"cpu_cores": 8, "memory_gb": 16, "cost_per_hour": 0.34},
            "r5.large": {"cpu_cores": 2, "memory_gb": 16, "cost_per_hour": 0.126},
            "r5.xlarge": {"cpu_cores": 4, "memory_gb": 32, "cost_per_hour": 0.252},
            
            # GCP instances (simplified)
            "n1-standard-1": {"cpu_cores": 1, "memory_gb": 3.75, "cost_per_hour": 0.0475},
            "n1-standard-2": {"cpu_cores": 2, "memory_gb": 7.5, "cost_per_hour": 0.095},
            "n1-standard-4": {"cpu_cores": 4, "memory_gb": 15, "cost_per_hour": 0.19},
            "n1-standard-8": {"cpu_cores": 8, "memory_gb": 30, "cost_per_hour": 0.38},
            
            # Azure instances (simplified)
            "Standard_B1s": {"cpu_cores": 1, "memory_gb": 1, "cost_per_hour": 0.0104},
            "Standard_B2s": {"cpu_cores": 2, "memory_gb": 4, "cost_per_hour": 0.0416},
            "Standard_D2s_v3": {"cpu_cores": 2, "memory_gb": 8, "cost_per_hour": 0.096},
            "Standard_D4s_v3": {"cpu_cores": 4, "memory_gb": 16, "cost_per_hour": 0.192}
        }
    
    def _load_pricing_data(self) -> Dict[str, Dict[str, float]]:
        """Load pricing data for different resource types"""
        
        # Simplified pricing data (in production, this would be loaded from cloud provider APIs)
        return {
            "storage": {
                "gp2_per_gb_month": 0.10,  # AWS GP2 EBS
                "gp3_per_gb_month": 0.08,  # AWS GP3 EBS
                "standard_per_gb_month": 0.045,  # AWS Standard EBS
                "ssd_per_gb_month": 0.17   # Premium SSD
            },
            "network": {
                "data_transfer_per_gb": 0.09,
                "load_balancer_per_hour": 0.025
            }
        }
@dataclass

class OptimizationRecommendation:
    """Comprehensive optimization recommendation with detailed savings analysis"""
    resource_id: str
    optimization_type: OptimizationType
    priority: int  # 1-10, higher = more important
    
    # Current state
    current_configuration: Dict[str, Any]
    current_monthly_cost: float
    current_utilization: float
    
    # Recommended state
    recommended_configuration: Dict[str, Any]
    recommended_monthly_cost: float
    expected_utilization: float
    
    # Financial impact
    monthly_savings: float
    annual_savings: float
    implementation_cost: float
    payback_period_months: float
    roi_percentage: float  # Return on investment
    net_present_value: float  # NPV over 3 years
    
    # Risk and confidence
    risk_level: RiskLevel
    confidence_score: float
    business_impact: str
    
    # Implementation details
    implementation_effort: str  # LOW, MEDIUM, HIGH
    estimated_downtime_minutes: int
    rollback_plan: str
    prerequisites: List[str]
    
    # Supporting analysis
    cost_benefit_analysis: Dict[str, Any]
    sensitivity_analysis: Dict[str, float]
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "resource_id": self.resource_id,
            "optimization_type": self.optimization_type.value,
            "priority": self.priority,
            "current_configuration": self.current_configuration,
            "current_monthly_cost": self.current_monthly_cost,
            "current_utilization": self.current_utilization,
            "recommended_configuration": self.recommended_configuration,
            "recommended_monthly_cost": self.recommended_monthly_cost,
            "expected_utilization": self.expected_utilization,
            "monthly_savings": self.monthly_savings,
            "annual_savings": self.annual_savings,
            "implementation_cost": self.implementation_cost,
            "payback_period_months": self.payback_period_months,
            "roi_percentage": self.roi_percentage,
            "net_present_value": self.net_present_value,
            "risk_level": self.risk_level.value,
            "confidence_score": self.confidence_score,
            "business_impact": self.business_impact,
            "implementation_effort": self.implementation_effort,
            "estimated_downtime_minutes": self.estimated_downtime_minutes,
            "rollback_plan": self.rollback_plan,
            "prerequisites": self.prerequisites,
            "cost_benefit_analysis": self.cost_benefit_analysis,
            "sensitivity_analysis": self.sensitivity_analysis,
            "created_at": self.created_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None
        }


class OptimizationCalculator:
    """
    Optimization recommendation and savings calculation engine
    
    Generates detailed optimization recommendations with comprehensive
    cost-benefit analysis, ROI calculations, and payback period estimates.
    """
    
    def __init__(self, discount_rate: float = 0.08):
        self.logger = logging.getLogger(f"{__name__}.OptimizationCalculator")
        self.discount_rate = discount_rate  # For NPV calculations
        self.recommendations_history: List[OptimizationRecommendation] = []
    
    def generate_optimization_recommendations(self, 
                                            waste_items: List[WasteItem],
                                            rightsizing_recommendations: List[RightSizingRecommendation],
                                            resource_metadata: Dict[str, ResourceMetadata]) -> List[OptimizationRecommendation]:
        """
        Generate comprehensive optimization recommendations from waste detection and right-sizing analysis
        
        Args:
            waste_items: Detected waste items
            rightsizing_recommendations: Right-sizing recommendations
            resource_metadata: Resource metadata for context
            
        Returns:
            List of optimization recommendations sorted by priority
        """
        self.logger.info(f"Generating optimization recommendations from {len(waste_items)} waste items and {len(rightsizing_recommendations)} right-sizing recommendations")
        
        recommendations = []
        
        # Process waste items
        for waste_item in waste_items:
            recommendation = self._create_waste_optimization_recommendation(waste_item, resource_metadata)
            if recommendation:
                recommendations.append(recommendation)
        
        # Process right-sizing recommendations
        for rightsizing in rightsizing_recommendations:
            recommendation = self._create_rightsizing_optimization_recommendation(rightsizing, resource_metadata)
            if recommendation:
                recommendations.append(recommendation)
        
        # Calculate priorities and sort
        recommendations = self._prioritize_recommendations(recommendations)
        
        # Store in history
        self.recommendations_history.extend(recommendations)
        
        # Keep history manageable
        if len(self.recommendations_history) > 500:
            self.recommendations_history = self.recommendations_history[-500:]
        
        total_monthly_savings = sum(rec.monthly_savings for rec in recommendations)
        total_annual_savings = sum(rec.annual_savings for rec in recommendations)
        
        self.logger.info(f"Generated {len(recommendations)} optimization recommendations with total potential savings of ${total_monthly_savings:.2f}/month (${total_annual_savings:.2f}/year)")
        
        return recommendations
    
    def _create_waste_optimization_recommendation(self, 
                                                waste_item: WasteItem,
                                                resource_metadata: Dict[str, ResourceMetadata]) -> Optional[OptimizationRecommendation]:
        """Create optimization recommendation from waste item"""
        
        metadata = resource_metadata.get(waste_item.resource_id)
        
        # Determine optimization type based on waste type
        optimization_type = self._map_waste_to_optimization_type(waste_item.waste_type)
        
        # Calculate implementation cost
        implementation_cost = self._estimate_implementation_cost(optimization_type, waste_item.current_monthly_cost)
        
        # Calculate payback period
        payback_months = implementation_cost / waste_item.potential_savings if waste_item.potential_savings > 0 else float('inf')
        
        # Calculate ROI
        annual_savings = waste_item.potential_savings * 12
        roi_percentage = ((annual_savings - implementation_cost) / implementation_cost) * 100 if implementation_cost > 0 else float('inf')
        
        # Calculate NPV (3-year horizon)
        npv = self._calculate_npv(waste_item.potential_savings, implementation_cost, 36)
        
        # Determine business impact
        business_impact = self._assess_business_impact(waste_item, metadata)
        
        # Get implementation details
        implementation_details = self._get_implementation_details(optimization_type, waste_item)
        
        # Cost-benefit analysis
        cost_benefit = self._perform_cost_benefit_analysis(
            waste_item.potential_savings, implementation_cost, waste_item.detection_confidence
        )
        
        # Sensitivity analysis
        sensitivity = self._perform_sensitivity_analysis(waste_item.potential_savings, implementation_cost)
        
        return OptimizationRecommendation(
            resource_id=waste_item.resource_id,
            optimization_type=optimization_type,
            priority=0,  # Will be calculated later
            current_configuration={
                "utilization": waste_item.current_utilization,
                "monthly_cost": waste_item.current_monthly_cost,
                "waste_type": waste_item.waste_type.value
            },
            current_monthly_cost=waste_item.current_monthly_cost,
            current_utilization=waste_item.current_utilization,
            recommended_configuration={
                "expected_utilization": waste_item.expected_utilization,
                "optimization_action": optimization_type.value
            },
            recommended_monthly_cost=waste_item.current_monthly_cost - waste_item.potential_savings,
            expected_utilization=waste_item.expected_utilization,
            monthly_savings=waste_item.potential_savings,
            annual_savings=annual_savings,
            implementation_cost=implementation_cost,
            payback_period_months=payback_months,
            roi_percentage=roi_percentage,
            net_present_value=npv,
            risk_level=waste_item.severity,
            confidence_score=waste_item.detection_confidence,
            business_impact=business_impact,
            implementation_effort=implementation_details["effort"],
            estimated_downtime_minutes=implementation_details["downtime"],
            rollback_plan=implementation_details["rollback"],
            prerequisites=implementation_details["prerequisites"],
            cost_benefit_analysis=cost_benefit,
            sensitivity_analysis=sensitivity,
            valid_until=datetime.now() + timedelta(days=30)  # Recommendations valid for 30 days
        )
    
    def _create_rightsizing_optimization_recommendation(self, 
                                                      rightsizing: RightSizingRecommendation,
                                                      resource_metadata: Dict[str, ResourceMetadata]) -> Optional[OptimizationRecommendation]:
        """Create optimization recommendation from right-sizing recommendation"""
        
        metadata = resource_metadata.get(rightsizing.resource_id)
        
        # Implementation cost for right-sizing (typically involves instance replacement)
        implementation_cost = self._estimate_rightsizing_implementation_cost(rightsizing)
        
        # Calculate payback period
        payback_months = implementation_cost / rightsizing.monthly_savings if rightsizing.monthly_savings > 0 else float('inf')
        
        # Calculate ROI
        roi_percentage = ((rightsizing.annual_savings - implementation_cost) / implementation_cost) * 100 if implementation_cost > 0 else float('inf')
        
        # Calculate NPV
        npv = self._calculate_npv(rightsizing.monthly_savings, implementation_cost, 36)
        
        # Business impact assessment
        business_impact = self._assess_rightsizing_business_impact(rightsizing, metadata)
        
        # Implementation details for right-sizing
        implementation_details = self._get_rightsizing_implementation_details(rightsizing)
        
        # Cost-benefit analysis
        cost_benefit = self._perform_cost_benefit_analysis(
            rightsizing.monthly_savings, implementation_cost, rightsizing.confidence_score
        )
        
        # Sensitivity analysis
        sensitivity = self._perform_sensitivity_analysis(rightsizing.monthly_savings, implementation_cost)
        
        return OptimizationRecommendation(
            resource_id=rightsizing.resource_id,
            optimization_type=OptimizationType.RIGHTSIZE,
            priority=0,  # Will be calculated later
            current_configuration={
                "instance_type": rightsizing.current_capacity.instance_type,
                "cpu_cores": rightsizing.current_capacity.cpu_cores,
                "memory_gb": rightsizing.current_capacity.memory_gb,
                "utilization": rightsizing.utilization_headroom
            },
            current_monthly_cost=rightsizing.current_monthly_cost,
            current_utilization=100 - rightsizing.utilization_headroom,
            recommended_configuration={
                "instance_type": rightsizing.recommended_capacity.instance_type,
                "cpu_cores": rightsizing.recommended_capacity.cpu_cores,
                "memory_gb": rightsizing.recommended_capacity.memory_gb
            },
            recommended_monthly_cost=rightsizing.recommended_monthly_cost,
            expected_utilization=rightsizing.utilization_headroom + 20,  # Expect higher utilization after right-sizing
            monthly_savings=rightsizing.monthly_savings,
            annual_savings=rightsizing.annual_savings,
            implementation_cost=implementation_cost,
            payback_period_months=payback_months,
            roi_percentage=roi_percentage,
            net_present_value=npv,
            risk_level=rightsizing.risk_level,
            confidence_score=rightsizing.confidence_score,
            business_impact=business_impact,
            implementation_effort=implementation_details["effort"],
            estimated_downtime_minutes=implementation_details["downtime"],
            rollback_plan=implementation_details["rollback"],
            prerequisites=implementation_details["prerequisites"],
            cost_benefit_analysis=cost_benefit,
            sensitivity_analysis=sensitivity,
            valid_until=datetime.now() + timedelta(days=30)
        )
    
    def _map_waste_to_optimization_type(self, waste_type: WasteType) -> OptimizationType:
        """Map waste type to optimization action"""
        
        mapping = {
            WasteType.UNUSED_RESOURCE: OptimizationType.TERMINATE,
            WasteType.ZOMBIE_RESOURCE: OptimizationType.TERMINATE,
            WasteType.UNDERUTILIZED_RESOURCE: OptimizationType.DOWNSIZE,
            WasteType.IDLE_RESOURCE: OptimizationType.SCHEDULE,
            WasteType.ORPHANED_RESOURCE: OptimizationType.TERMINATE,
            WasteType.OVERSIZED_RESOURCE: OptimizationType.RIGHTSIZE
        }
        
        return mapping.get(waste_type, OptimizationType.RIGHTSIZE)
    
    def _estimate_implementation_cost(self, optimization_type: OptimizationType, monthly_cost: float) -> float:
        """Estimate implementation cost for optimization action"""
        
        # Cost factors based on optimization type
        cost_factors = {
            OptimizationType.TERMINATE: 0.0,  # No cost to terminate
            OptimizationType.DOWNSIZE: monthly_cost * 0.1,  # 10% of monthly cost
            OptimizationType.RIGHTSIZE: monthly_cost * 0.15,  # 15% of monthly cost
            OptimizationType.SCHEDULE: monthly_cost * 0.05,  # 5% of monthly cost
            OptimizationType.MIGRATE: monthly_cost * 0.25,  # 25% of monthly cost
            OptimizationType.CONSOLIDATE: monthly_cost * 0.20  # 20% of monthly cost
        }
        
        return cost_factors.get(optimization_type, monthly_cost * 0.1)
    
    def _estimate_rightsizing_implementation_cost(self, rightsizing: RightSizingRecommendation) -> float:
        """Estimate implementation cost for right-sizing"""
        
        # Base cost for instance replacement
        base_cost = rightsizing.current_monthly_cost * 0.1
        
        # Additional cost based on risk level
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 1.5,
            RiskLevel.HIGH: 2.0,
            RiskLevel.CRITICAL: 3.0
        }
        
        multiplier = risk_multipliers.get(rightsizing.risk_level, 1.0)
        
        return base_cost * multiplier
    
    def _calculate_npv(self, monthly_savings: float, implementation_cost: float, months: int) -> float:
        """Calculate Net Present Value over specified months"""
        
        if monthly_savings <= 0:
            return -implementation_cost
        
        # Calculate NPV using monthly discount rate
        monthly_discount_rate = self.discount_rate / 12
        
        npv = -implementation_cost  # Initial investment (negative)
        
        for month in range(1, months + 1):
            discounted_savings = monthly_savings / ((1 + monthly_discount_rate) ** month)
            npv += discounted_savings
        
        return npv
    
    def _assess_business_impact(self, waste_item: WasteItem, metadata: Optional[ResourceMetadata]) -> str:
        """Assess business impact of optimization action"""
        
        if waste_item.waste_type in [WasteType.UNUSED_RESOURCE, WasteType.ZOMBIE_RESOURCE]:
            return "LOW - Resource appears unused, minimal business impact expected"
        
        if waste_item.waste_type == WasteType.UNDERUTILIZED_RESOURCE:
            if waste_item.current_utilization < 10:
                return "LOW - Very low utilization, minimal impact expected"
            else:
                return "MEDIUM - Some utilization present, monitor after optimization"
        
        if waste_item.waste_type == WasteType.IDLE_RESOURCE:
            return "MEDIUM - Resource has idle periods, scheduling optimization recommended"
        
        return "MEDIUM - Standard optimization with monitoring recommended"
    
    def _assess_rightsizing_business_impact(self, rightsizing: RightSizingRecommendation, metadata: Optional[ResourceMetadata]) -> str:
        """Assess business impact of right-sizing action"""
        
        if rightsizing.risk_level == RiskLevel.LOW:
            return "LOW - Right-sizing has minimal risk, no significant business impact expected"
        elif rightsizing.risk_level == RiskLevel.MEDIUM:
            return "MEDIUM - Monitor performance after right-sizing, rollback plan available"
        elif rightsizing.risk_level == RiskLevel.HIGH:
            return "HIGH - Careful monitoring required, consider gradual implementation"
        else:
            return "CRITICAL - High risk optimization, extensive testing and monitoring required"
    
    def _get_implementation_details(self, optimization_type: OptimizationType, waste_item: WasteItem) -> Dict[str, Any]:
        """Get implementation details for optimization action"""
        
        details = {
            OptimizationType.TERMINATE: {
                "effort": "LOW",
                "downtime": 5,
                "rollback": "Restore from backup or recreate resource if needed",
                "prerequisites": ["Confirm resource is truly unused", "Backup critical data"]
            },
            OptimizationType.DOWNSIZE: {
                "effort": "MEDIUM",
                "downtime": 15,
                "rollback": "Scale back up to original size",
                "prerequisites": ["Performance baseline", "Monitoring setup", "Change window"]
            },
            OptimizationType.RIGHTSIZE: {
                "effort": "MEDIUM",
                "downtime": 30,
                "rollback": "Revert to original instance type",
                "prerequisites": ["Performance testing", "Backup", "Change approval"]
            },
            OptimizationType.SCHEDULE: {
                "effort": "LOW",
                "downtime": 0,
                "rollback": "Remove scheduling configuration",
                "prerequisites": ["Usage pattern analysis", "Business hour confirmation"]
            },
            OptimizationType.MIGRATE: {
                "effort": "HIGH",
                "downtime": 60,
                "rollback": "Migrate back to original location",
                "prerequisites": ["Migration testing", "Network configuration", "DNS updates"]
            },
            OptimizationType.CONSOLIDATE: {
                "effort": "HIGH",
                "downtime": 45,
                "rollback": "Separate back to individual resources",
                "prerequisites": ["Compatibility testing", "Capacity planning", "Performance validation"]
            }
        }
        
        return details.get(optimization_type, {
            "effort": "MEDIUM",
            "downtime": 20,
            "rollback": "Revert configuration changes",
            "prerequisites": ["Testing", "Monitoring", "Approval"]
        })
    
    def _get_rightsizing_implementation_details(self, rightsizing: RightSizingRecommendation) -> Dict[str, Any]:
        """Get implementation details for right-sizing"""
        
        if rightsizing.risk_level == RiskLevel.LOW:
            return {
                "effort": "LOW",
                "downtime": 10,
                "rollback": "Revert to original instance type within 24 hours",
                "prerequisites": ["Backup", "Change window"]
            }
        elif rightsizing.risk_level == RiskLevel.MEDIUM:
            return {
                "effort": "MEDIUM",
                "downtime": 20,
                "rollback": "Immediate revert capability, performance monitoring",
                "prerequisites": ["Performance baseline", "Load testing", "Monitoring setup"]
            }
        else:
            return {
                "effort": "HIGH",
                "downtime": 45,
                "rollback": "Comprehensive rollback plan with performance validation",
                "prerequisites": ["Extensive testing", "Stakeholder approval", "Emergency procedures"]
            }
    
    def _perform_cost_benefit_analysis(self, monthly_savings: float, implementation_cost: float, confidence: float) -> Dict[str, Any]:
        """Perform detailed cost-benefit analysis"""
        
        # Calculate various financial metrics
        annual_savings = monthly_savings * 12
        
        # Risk-adjusted savings (based on confidence)
        risk_adjusted_monthly = monthly_savings * confidence
        risk_adjusted_annual = risk_adjusted_monthly * 12
        
        # Break-even analysis
        break_even_months = implementation_cost / monthly_savings if monthly_savings > 0 else float('inf')
        
        # 3-year total benefit
        three_year_savings = monthly_savings * 36
        three_year_net_benefit = three_year_savings - implementation_cost
        
        return {
            "annual_gross_savings": round(annual_savings, 2),
            "risk_adjusted_annual_savings": round(risk_adjusted_annual, 2),
            "implementation_cost": round(implementation_cost, 2),
            "break_even_months": round(break_even_months, 1),
            "three_year_net_benefit": round(three_year_net_benefit, 2),
            "benefit_cost_ratio": round(three_year_savings / implementation_cost if implementation_cost > 0 else float('inf'), 2),
            "confidence_factor": confidence
        }
    
    def _perform_sensitivity_analysis(self, monthly_savings: float, implementation_cost: float) -> Dict[str, float]:
        """Perform sensitivity analysis on savings estimates"""
        
        # Test different scenarios
        scenarios = {
            "pessimistic": 0.7,  # 30% lower savings
            "realistic": 1.0,    # Expected savings
            "optimistic": 1.3    # 30% higher savings
        }
        
        sensitivity = {}
        
        for scenario, factor in scenarios.items():
            adjusted_savings = monthly_savings * factor
            adjusted_annual = adjusted_savings * 12
            adjusted_npv = self._calculate_npv(adjusted_savings, implementation_cost, 36)
            adjusted_payback = implementation_cost / adjusted_savings if adjusted_savings > 0 else float('inf')
            
            sensitivity[f"{scenario}_monthly_savings"] = round(adjusted_savings, 2)
            sensitivity[f"{scenario}_annual_savings"] = round(adjusted_annual, 2)
            sensitivity[f"{scenario}_npv"] = round(adjusted_npv, 2)
            sensitivity[f"{scenario}_payback_months"] = round(adjusted_payback, 1)
        
        return sensitivity
    
    def _prioritize_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Calculate priority scores and sort recommendations"""
        
        if not recommendations:
            return recommendations
        
        # Calculate priority scores based on multiple factors
        for rec in recommendations:
            priority_score = self._calculate_priority_score(rec)
            rec.priority = priority_score
        
        # Sort by priority (highest first)
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations
    
    def _calculate_priority_score(self, recommendation: OptimizationRecommendation) -> int:
        """Calculate priority score (1-10) for recommendation"""
        
        score = 5  # Base score
        
        # Factor 1: Monthly savings (higher savings = higher priority)
        if recommendation.monthly_savings > 1000:
            score += 2
        elif recommendation.monthly_savings > 500:
            score += 1
        elif recommendation.monthly_savings < 50:
            score -= 1
        
        # Factor 2: Payback period (shorter payback = higher priority)
        if recommendation.payback_period_months < 3:
            score += 2
        elif recommendation.payback_period_months < 6:
            score += 1
        elif recommendation.payback_period_months > 12:
            score -= 1
        
        # Factor 3: Risk level (lower risk = higher priority)
        risk_adjustments = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 0,
            RiskLevel.HIGH: -1,
            RiskLevel.CRITICAL: -2
        }
        score += risk_adjustments.get(recommendation.risk_level, 0)
        
        # Factor 4: Confidence (higher confidence = higher priority)
        if recommendation.confidence_score > 0.9:
            score += 1
        elif recommendation.confidence_score < 0.7:
            score -= 1
        
        # Factor 5: Implementation effort (lower effort = higher priority)
        if recommendation.implementation_effort == "LOW":
            score += 1
        elif recommendation.implementation_effort == "HIGH":
            score -= 1
        
        # Ensure score is within valid range
        return max(1, min(10, score))
    
    def get_optimization_summary(self, recommendations: Optional[List[OptimizationRecommendation]] = None) -> Dict[str, Any]:
        """Get summary of optimization recommendations"""
        
        if recommendations is None:
            recommendations = self.recommendations_history
        
        if not recommendations:
            return {"error": "No optimization recommendations available"}
        
        # Calculate totals
        total_monthly_savings = sum(rec.monthly_savings for rec in recommendations)
        total_annual_savings = sum(rec.annual_savings for rec in recommendations)
        total_implementation_cost = sum(rec.implementation_cost for rec in recommendations)
        
        # Count by optimization type
        type_counts = {}
        type_savings = {}
        for opt_type in OptimizationType:
            type_recs = [rec for rec in recommendations if rec.optimization_type == opt_type]
            type_counts[opt_type.value] = len(type_recs)
            type_savings[opt_type.value] = sum(rec.monthly_savings for rec in type_recs)
        
        # Count by priority
        priority_distribution = {}
        for i in range(1, 11):
            priority_distribution[f"priority_{i}"] = sum(
                1 for rec in recommendations if rec.priority == i
            )
        
        # Calculate average metrics
        avg_payback = statistics.mean([
            rec.payback_period_months for rec in recommendations 
            if rec.payback_period_months != float('inf')
        ]) if any(rec.payback_period_months != float('inf') for rec in recommendations) else 0
        
        avg_roi = statistics.mean([
            rec.roi_percentage for rec in recommendations 
            if rec.roi_percentage != float('inf')
        ]) if any(rec.roi_percentage != float('inf') for rec in recommendations) else 0
        
        # Top recommendations
        top_recommendations = sorted(recommendations, key=lambda x: x.monthly_savings, reverse=True)[:5]
        
        return {
            "total_recommendations": len(recommendations),
            "total_monthly_savings": round(total_monthly_savings, 2),
            "total_annual_savings": round(total_annual_savings, 2),
            "total_implementation_cost": round(total_implementation_cost, 2),
            "net_first_year_benefit": round(total_annual_savings - total_implementation_cost, 2),
            "average_payback_months": round(avg_payback, 1),
            "average_roi_percentage": round(avg_roi, 1),
            "optimization_by_type": {
                "counts": type_counts,
                "monthly_savings": {k: round(v, 2) for k, v in type_savings.items()}
            },
            "priority_distribution": priority_distribution,
            "top_opportunities": [
                {
                    "resource_id": rec.resource_id,
                    "optimization_type": rec.optimization_type.value,
                    "monthly_savings": round(rec.monthly_savings, 2),
                    "priority": rec.priority,
                    "payback_months": round(rec.payback_period_months, 1)
                }
                for rec in top_recommendations
            ],
            "analysis_timestamp": datetime.now().isoformat()
        }
@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for optimization actions"""
    resource_id: str
    optimization_type: OptimizationType
    overall_risk_level: RiskLevel
    risk_score: float  # 0-100, higher = more risky
    
    # Risk categories
    technical_risk: RiskLevel
    business_risk: RiskLevel
    financial_risk: RiskLevel
    operational_risk: RiskLevel
    
    # Risk factors
    risk_factors: List[str]
    mitigation_strategies: List[str]
    safety_checks: List[str]
    
    # Impact analysis
    potential_impact: str
    blast_radius: str  # How many systems/users could be affected
    recovery_time_estimate: str
    
    # Recommendations
    recommended_approach: str
    testing_requirements: List[str]
    monitoring_requirements: List[str]
    rollback_triggers: List[str]
    
    # Metadata
    assessed_at: datetime = field(default_factory=datetime.now)
    assessor: str = "automated_risk_assessor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "resource_id": self.resource_id,
            "optimization_type": self.optimization_type.value,
            "overall_risk_level": self.overall_risk_level.value,
            "risk_score": self.risk_score,
            "technical_risk": self.technical_risk.value,
            "business_risk": self.business_risk.value,
            "financial_risk": self.financial_risk.value,
            "operational_risk": self.operational_risk.value,
            "risk_factors": self.risk_factors,
            "mitigation_strategies": self.mitigation_strategies,
            "safety_checks": self.safety_checks,
            "potential_impact": self.potential_impact,
            "blast_radius": self.blast_radius,
            "recovery_time_estimate": self.recovery_time_estimate,
            "recommended_approach": self.recommended_approach,
            "testing_requirements": self.testing_requirements,
            "monitoring_requirements": self.monitoring_requirements,
            "rollback_triggers": self.rollback_triggers,
            "assessed_at": self.assessed_at.isoformat(),
            "assessor": self.assessor
        }


@dataclass
class SafetyCheck:
    """Individual safety check for optimization action"""
    check_name: str
    check_type: str  # PRE_IMPLEMENTATION, POST_IMPLEMENTATION, CONTINUOUS
    description: str
    success_criteria: str
    failure_action: str
    automated: bool
    priority: RiskLevel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "check_type": self.check_type,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "failure_action": self.failure_action,
            "automated": self.automated,
            "priority": self.priority.value
        }


class RiskAssessor:
    """
    Risk assessment and safety checks system for optimization actions
    
    Evaluates technical, business, financial, and operational risks
    for optimization recommendations and provides comprehensive
    safety checks and mitigation strategies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RiskAssessor")
        self.risk_history: List[RiskAssessment] = []
        self.safety_check_templates = self._load_safety_check_templates()
        self.risk_thresholds = self._load_risk_thresholds()
    
    def assess_optimization_risk(self, 
                               recommendation: OptimizationRecommendation,
                               resource_metadata: Optional[ResourceMetadata] = None,
                               utilization_data: Optional[UtilizationMetrics] = None) -> RiskAssessment:
        """
        Perform comprehensive risk assessment for optimization recommendation
        
        Args:
            recommendation: Optimization recommendation to assess
            resource_metadata: Resource metadata for context
            utilization_data: Utilization metrics for risk analysis
            
        Returns:
            Comprehensive risk assessment
        """
        self.logger.info(f"Assessing risk for optimization {recommendation.optimization_type.value} on resource {recommendation.resource_id}")
        
        # Assess different risk categories
        technical_risk = self._assess_technical_risk(recommendation, resource_metadata, utilization_data)
        business_risk = self._assess_business_risk(recommendation, resource_metadata)
        financial_risk = self._assess_financial_risk(recommendation)
        operational_risk = self._assess_operational_risk(recommendation, resource_metadata)
        
        # Calculate overall risk
        overall_risk, risk_score = self._calculate_overall_risk(
            technical_risk, business_risk, financial_risk, operational_risk
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            recommendation, resource_metadata, utilization_data
        )
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(
            recommendation, overall_risk, risk_factors
        )
        
        # Generate safety checks
        safety_checks = self._generate_safety_checks(recommendation, overall_risk)
        
        # Assess impact and blast radius
        impact_analysis = self._assess_impact_and_blast_radius(
            recommendation, resource_metadata
        )
        
        # Generate recommendations
        approach_recommendations = self._generate_approach_recommendations(
            recommendation, overall_risk, risk_factors
        )
        
        risk_assessment = RiskAssessment(
            resource_id=recommendation.resource_id,
            optimization_type=recommendation.optimization_type,
            overall_risk_level=overall_risk,
            risk_score=risk_score,
            technical_risk=technical_risk,
            business_risk=business_risk,
            financial_risk=financial_risk,
            operational_risk=operational_risk,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            safety_checks=safety_checks,
            potential_impact=impact_analysis["potential_impact"],
            blast_radius=impact_analysis["blast_radius"],
            recovery_time_estimate=impact_analysis["recovery_time"],
            recommended_approach=approach_recommendations["approach"],
            testing_requirements=approach_recommendations["testing"],
            monitoring_requirements=approach_recommendations["monitoring"],
            rollback_triggers=approach_recommendations["rollback_triggers"]
        )
        
        # Store in history
        self.risk_history.append(risk_assessment)
        
        # Keep history manageable
        if len(self.risk_history) > 200:
            self.risk_history = self.risk_history[-200:]
        
        self.logger.info(f"Risk assessment completed: {overall_risk.value} risk (score: {risk_score:.1f})")
        
        return risk_assessment
    
    def _assess_technical_risk(self, 
                             recommendation: OptimizationRecommendation,
                             metadata: Optional[ResourceMetadata],
                             utilization: Optional[UtilizationMetrics]) -> RiskLevel:
        """Assess technical risk of optimization action"""
        
        risk_factors = []
        
        # High utilization increases technical risk
        if recommendation.current_utilization > 80:
            risk_factors.append("high_current_utilization")
        
        # Optimization type risk assessment
        high_risk_optimizations = [OptimizationType.MIGRATE, OptimizationType.CONSOLIDATE]
        if recommendation.optimization_type in high_risk_optimizations:
            risk_factors.append("high_risk_optimization_type")
        
        # Resource type considerations
        if metadata:
            critical_resource_types = [ResourceType.DATABASE, ResourceType.LOAD_BALANCER]
            if metadata.resource_type in critical_resource_types:
                risk_factors.append("critical_resource_type")
        
        # Utilization pattern analysis
        if utilization:
            variability = utilization.usage_patterns.get("variability_score", 0)
            if variability > 60:
                risk_factors.append("high_utilization_variability")
            
            if len(utilization.peak_hours) > 16:  # More than 16 peak hours suggests always-on workload
                risk_factors.append("always_on_workload")
        
        # Confidence level
        if recommendation.confidence_score < 0.7:
            risk_factors.append("low_confidence_recommendation")
        
        # Determine risk level based on factors
        if len(risk_factors) >= 3:
            return RiskLevel.HIGH
        elif len(risk_factors) >= 2:
            return RiskLevel.MEDIUM
        elif len(risk_factors) >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.LOW
    
    def _assess_business_risk(self, 
                            recommendation: OptimizationRecommendation,
                            metadata: Optional[ResourceMetadata]) -> RiskLevel:
        """Assess business risk of optimization action"""
        
        risk_factors = []
        
        # Resource criticality based on tags
        if metadata and metadata.tags:
            environment = metadata.tags.get("environment", "").lower()
            if environment in ["production", "prod"]:
                risk_factors.append("production_environment")
            
            criticality = metadata.tags.get("criticality", "").lower()
            if criticality in ["critical", "high"]:
                risk_factors.append("high_criticality")
            
            # Check for business-critical tags
            business_tags = ["customer-facing", "revenue-generating", "compliance-required"]
            for tag_key, tag_value in metadata.tags.items():
                if any(bt in tag_value.lower() for bt in business_tags):
                    risk_factors.append("business_critical_resource")
                    break
        
        # Optimization type business impact
        if recommendation.optimization_type == OptimizationType.TERMINATE:
            risk_factors.append("resource_termination")
        
        # High savings might indicate critical resource
        if recommendation.monthly_savings > 1000:
            risk_factors.append("high_cost_resource")
        
        # Determine risk level
        if len(risk_factors) >= 3:
            return RiskLevel.HIGH
        elif len(risk_factors) >= 2:
            return RiskLevel.MEDIUM
        elif len(risk_factors) >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.LOW
    
    def _assess_financial_risk(self, recommendation: OptimizationRecommendation) -> RiskLevel:
        """Assess financial risk of optimization action"""
        
        risk_factors = []
        
        # High implementation cost
        if recommendation.implementation_cost > recommendation.monthly_savings * 6:
            risk_factors.append("high_implementation_cost")
        
        # Long payback period
        if recommendation.payback_period_months > 12:
            risk_factors.append("long_payback_period")
        
        # Low or negative ROI
        if recommendation.roi_percentage < 50:
            risk_factors.append("low_roi")
        
        # Negative NPV
        if recommendation.net_present_value < 0:
            risk_factors.append("negative_npv")
        
        # Determine risk level
        if len(risk_factors) >= 3:
            return RiskLevel.HIGH
        elif len(risk_factors) >= 2:
            return RiskLevel.MEDIUM
        elif len(risk_factors) >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.LOW
    
    def _assess_operational_risk(self, 
                               recommendation: OptimizationRecommendation,
                               metadata: Optional[ResourceMetadata]) -> RiskLevel:
        """Assess operational risk of optimization action"""
        
        risk_factors = []
        
        # High downtime requirement
        if recommendation.estimated_downtime_minutes > 30:
            risk_factors.append("high_downtime_requirement")
        
        # High implementation effort
        if recommendation.implementation_effort == "HIGH":
            risk_factors.append("high_implementation_effort")
        
        # Resource age (newer resources might be less stable for optimization)
        if metadata and metadata.created_date:
            resource_age_days = (datetime.now() - metadata.created_date).days
            if resource_age_days < 30:
                risk_factors.append("new_resource")
        
        # Complex rollback requirements
        if "complex" in recommendation.rollback_plan.lower() or "extensive" in recommendation.rollback_plan.lower():
            risk_factors.append("complex_rollback")
        
        # Multiple prerequisites
        if len(recommendation.prerequisites) > 3:
            risk_factors.append("many_prerequisites")
        
        # Determine risk level
        if len(risk_factors) >= 3:
            return RiskLevel.HIGH
        elif len(risk_factors) >= 2:
            return RiskLevel.MEDIUM
        elif len(risk_factors) >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.LOW
    
    def _calculate_overall_risk(self, 
                              technical: RiskLevel,
                              business: RiskLevel,
                              financial: RiskLevel,
                              operational: RiskLevel) -> Tuple[RiskLevel, float]:
        """Calculate overall risk level and score"""
        
        # Convert risk levels to numeric scores
        risk_scores = {
            RiskLevel.LOW: 25,
            RiskLevel.MEDIUM: 50,
            RiskLevel.HIGH: 75,
            RiskLevel.CRITICAL: 100
        }
        
        # Weighted average (business risk has highest weight)
        weights = {
            "technical": 0.25,
            "business": 0.40,
            "financial": 0.20,
            "operational": 0.15
        }
        
        weighted_score = (
            risk_scores[technical] * weights["technical"] +
            risk_scores[business] * weights["business"] +
            risk_scores[financial] * weights["financial"] +
            risk_scores[operational] * weights["operational"]
        )
        
        # Convert back to risk level
        if weighted_score >= 80:
            overall_risk = RiskLevel.CRITICAL
        elif weighted_score >= 60:
            overall_risk = RiskLevel.HIGH
        elif weighted_score >= 35:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW
        
        return overall_risk, weighted_score
    
    def _identify_risk_factors(self, 
                             recommendation: OptimizationRecommendation,
                             metadata: Optional[ResourceMetadata],
                             utilization: Optional[UtilizationMetrics]) -> List[str]:
        """Identify specific risk factors for the optimization"""
        
        factors = []
        
        # Resource-specific factors
        if metadata:
            if metadata.state == ResourceState.RUNNING:
                factors.append("Resource is currently active and running")
            
            if metadata.tags.get("environment") == "production":
                factors.append("Production environment resource")
            
            if metadata.resource_type in [ResourceType.DATABASE, ResourceType.LOAD_BALANCER]:
                factors.append("Critical infrastructure component")
        
        # Utilization-specific factors
        if utilization:
            if utilization.overall_utilization > 70:
                factors.append("High current utilization may indicate active usage")
            
            if len(utilization.peak_hours) > 12:
                factors.append("Extended peak usage periods suggest consistent demand")
            
            variability = utilization.usage_patterns.get("variability_score", 0)
            if variability > 50:
                factors.append("High utilization variability makes prediction difficult")
        
        # Optimization-specific factors
        if recommendation.optimization_type == OptimizationType.TERMINATE:
            factors.append("Resource termination is irreversible without backup")
        
        if recommendation.estimated_downtime_minutes > 0:
            factors.append(f"Requires {recommendation.estimated_downtime_minutes} minutes of downtime")
        
        if recommendation.payback_period_months > 6:
            factors.append("Long payback period increases financial risk")
        
        if recommendation.confidence_score < 0.8:
            factors.append("Lower confidence in optimization recommendation")
        
        return factors
    
    def _generate_mitigation_strategies(self, 
                                      recommendation: OptimizationRecommendation,
                                      risk_level: RiskLevel,
                                      risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation strategies"""
        
        strategies = []
        
        # Base strategies for all optimizations
        strategies.append("Implement comprehensive monitoring before and after optimization")
        strategies.append("Prepare detailed rollback plan with clear triggers")
        
        # Risk-level specific strategies
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            strategies.append("Conduct thorough testing in non-production environment")
            strategies.append("Implement gradual rollout with checkpoints")
            strategies.append("Obtain stakeholder approval before implementation")
            strategies.append("Schedule implementation during low-usage periods")
        
        if risk_level == RiskLevel.CRITICAL:
            strategies.append("Implement emergency rollback procedures")
            strategies.append("Have technical team on standby during implementation")
            strategies.append("Consider phased implementation approach")
        
        # Optimization-type specific strategies
        if recommendation.optimization_type == OptimizationType.TERMINATE:
            strategies.append("Verify resource is truly unused through extended monitoring")
            strategies.append("Create backup or snapshot before termination")
            strategies.append("Implement soft deletion with recovery period")
        
        if recommendation.optimization_type in [OptimizationType.RIGHTSIZE, OptimizationType.DOWNSIZE]:
            strategies.append("Monitor performance metrics closely after resize")
            strategies.append("Implement automatic scaling if utilization exceeds thresholds")
        
        # Factor-specific strategies
        for factor in risk_factors:
            if "production" in factor.lower():
                strategies.append("Coordinate with production support team")
            if "high utilization" in factor.lower():
                strategies.append("Implement performance alerting with tight thresholds")
            if "variability" in factor.lower():
                strategies.append("Use conservative sizing with additional headroom")
        
        return list(set(strategies))  # Remove duplicates
    
    def _generate_safety_checks(self, 
                              recommendation: OptimizationRecommendation,
                              risk_level: RiskLevel) -> List[str]:
        """Generate safety checks for optimization"""
        
        checks = []
        
        # Pre-implementation checks
        checks.append("Verify resource backup/snapshot is available")
        checks.append("Confirm monitoring systems are operational")
        checks.append("Validate rollback procedures are tested")
        checks.append("Check for any scheduled maintenance windows")
        
        # Implementation checks
        checks.append("Monitor resource health during optimization")
        checks.append("Verify dependent services remain operational")
        checks.append("Confirm expected cost changes are reflected")
        
        # Post-implementation checks
        checks.append("Monitor performance metrics for 24-48 hours")
        checks.append("Verify application functionality is unaffected")
        checks.append("Confirm cost savings are realized")
        checks.append("Check for any error rate increases")
        
        # Risk-level specific checks
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            checks.append("Conduct end-to-end functionality testing")
            checks.append("Verify business process continuity")
            checks.append("Monitor customer impact metrics")
            checks.append("Check compliance requirements are maintained")
        
        # Optimization-type specific checks
        if recommendation.optimization_type == OptimizationType.TERMINATE:
            checks.append("Confirm no active connections or dependencies")
            checks.append("Verify data retention requirements are met")
        
        if recommendation.optimization_type in [OptimizationType.RIGHTSIZE, OptimizationType.DOWNSIZE]:
            checks.append("Monitor CPU and memory utilization trends")
            checks.append("Check for performance degradation indicators")
        
        return checks
    
    def _assess_impact_and_blast_radius(self, 
                                      recommendation: OptimizationRecommendation,
                                      metadata: Optional[ResourceMetadata]) -> Dict[str, str]:
        """Assess potential impact and blast radius of optimization"""
        
        # Determine potential impact
        if recommendation.optimization_type == OptimizationType.TERMINATE:
            potential_impact = "Complete service interruption if resource is actually in use"
        elif recommendation.optimization_type in [OptimizationType.RIGHTSIZE, OptimizationType.DOWNSIZE]:
            potential_impact = "Performance degradation or service slowdown"
        elif recommendation.optimization_type == OptimizationType.SCHEDULE:
            potential_impact = "Service unavailability during scheduled downtime"
        else:
            potential_impact = "Temporary service disruption during implementation"
        
        # Determine blast radius
        if metadata and metadata.resource_type == ResourceType.LOAD_BALANCER:
            blast_radius = "Multiple services and potentially many users"
        elif metadata and metadata.resource_type == ResourceType.DATABASE:
            blast_radius = "Applications dependent on database, potentially high user impact"
        elif metadata and metadata.tags.get("environment") == "production":
            blast_radius = "Production users and dependent services"
        else:
            blast_radius = "Limited to specific service or application"
        
        # Estimate recovery time
        if recommendation.estimated_downtime_minutes > 60:
            recovery_time = "1-4 hours including rollback and validation"
        elif recommendation.estimated_downtime_minutes > 30:
            recovery_time = "30-60 minutes for rollback and verification"
        else:
            recovery_time = "5-15 minutes for quick rollback"
        
        return {
            "potential_impact": potential_impact,
            "blast_radius": blast_radius,
            "recovery_time": recovery_time
        }
    
    def _generate_approach_recommendations(self, 
                                         recommendation: OptimizationRecommendation,
                                         risk_level: RiskLevel,
                                         risk_factors: List[str]) -> Dict[str, Any]:
        """Generate recommended approach based on risk assessment"""
        
        if risk_level == RiskLevel.LOW:
            approach = "Standard implementation with basic monitoring"
            testing = ["Basic functionality testing", "Performance baseline comparison"]
            monitoring = ["Standard performance metrics", "Cost tracking"]
            rollback_triggers = ["Performance degradation > 20%", "Error rate increase > 5%"]
        
        elif risk_level == RiskLevel.MEDIUM:
            approach = "Careful implementation with enhanced monitoring and staged rollout"
            testing = ["Comprehensive functionality testing", "Load testing", "Performance validation"]
            monitoring = ["Enhanced performance monitoring", "Business metrics tracking", "User experience monitoring"]
            rollback_triggers = ["Performance degradation > 15%", "Error rate increase > 3%", "User complaints"]
        
        elif risk_level == RiskLevel.HIGH:
            approach = "Cautious implementation with extensive testing and gradual rollout"
            testing = ["Full regression testing", "Stress testing", "Disaster recovery testing", "Business continuity validation"]
            monitoring = ["Real-time performance monitoring", "Business impact monitoring", "Customer satisfaction tracking"]
            rollback_triggers = ["Performance degradation > 10%", "Error rate increase > 2%", "Business metric impact", "Customer escalations"]
        
        else:  # CRITICAL
            approach = "Highly controlled implementation with maximum safety measures"
            testing = ["Complete test suite execution", "Production-like environment testing", "Chaos engineering validation", "Business process verification"]
            monitoring = ["Continuous real-time monitoring", "Automated alerting", "Business dashboard monitoring", "Executive reporting"]
            rollback_triggers = ["Any performance degradation", "Any error rate increase", "Any business impact", "Any customer complaints"]
        
        return {
            "approach": approach,
            "testing": testing,
            "monitoring": monitoring,
            "rollback_triggers": rollback_triggers
        }
    
    def _load_safety_check_templates(self) -> Dict[str, List[SafetyCheck]]:
        """Load safety check templates for different optimization types"""
        
        # This would typically be loaded from a configuration file or database
        return {
            "terminate": [
                SafetyCheck("Resource Usage Verification", "PRE_IMPLEMENTATION", 
                          "Verify resource has no active connections or usage", 
                          "Zero active connections for 7+ days", "Abort termination", True, RiskLevel.HIGH),
                SafetyCheck("Backup Verification", "PRE_IMPLEMENTATION",
                          "Ensure all critical data is backed up",
                          "Backup completed and verified", "Create backup before proceeding", True, RiskLevel.CRITICAL),
                SafetyCheck("Dependency Check", "PRE_IMPLEMENTATION",
                          "Check for dependent resources or services",
                          "No active dependencies found", "Resolve dependencies first", True, RiskLevel.HIGH)
            ],
            "rightsize": [
                SafetyCheck("Performance Baseline", "PRE_IMPLEMENTATION",
                          "Establish performance baseline metrics",
                          "Baseline metrics captured for 24+ hours", "Extend monitoring period", True, RiskLevel.MEDIUM),
                SafetyCheck("Capacity Validation", "PRE_IMPLEMENTATION",
                          "Validate new capacity meets requirements",
                          "New capacity > peak usage + 20% buffer", "Increase capacity allocation", True, RiskLevel.HIGH),
                SafetyCheck("Performance Monitoring", "POST_IMPLEMENTATION",
                          "Monitor performance after resize",
                          "Performance within 5% of baseline", "Rollback if degraded", True, RiskLevel.HIGH)
            ]
        }
    
    def _load_risk_thresholds(self) -> Dict[str, float]:
        """Load risk assessment thresholds"""
        
        return {
            "high_utilization_threshold": 80.0,
            "high_variability_threshold": 60.0,
            "long_payback_threshold": 12.0,  # months
            "low_confidence_threshold": 0.7,
            "high_downtime_threshold": 30.0,  # minutes
            "high_cost_threshold": 1000.0  # monthly cost
        }
    
    def get_risk_summary(self, assessments: Optional[List[RiskAssessment]] = None) -> Dict[str, Any]:
        """Get summary of risk assessments"""
        
        if assessments is None:
            assessments = self.risk_history
        
        if not assessments:
            return {"error": "No risk assessments available"}
        
        # Count by risk level
        risk_level_counts = {}
        for level in RiskLevel:
            risk_level_counts[level.value] = sum(
                1 for assessment in assessments 
                if assessment.overall_risk_level == level
            )
        
        # Average risk score
        avg_risk_score = statistics.mean([a.risk_score for a in assessments])
        
        # Most common risk factors
        all_factors = []
        for assessment in assessments:
            all_factors.extend(assessment.risk_factors)
        
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        top_risk_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # High-risk optimizations
        high_risk_optimizations = [
            {
                "resource_id": a.resource_id,
                "optimization_type": a.optimization_type.value,
                "risk_level": a.overall_risk_level.value,
                "risk_score": a.risk_score
            }
            for a in assessments 
            if a.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
        
        return {
            "total_assessments": len(assessments),
            "average_risk_score": round(avg_risk_score, 1),
            "risk_distribution": risk_level_counts,
            "top_risk_factors": [{"factor": factor, "count": count} for factor, count in top_risk_factors],
            "high_risk_optimizations": high_risk_optimizations[:10],  # Top 10
            "recommendations": {
                "proceed_with_caution": risk_level_counts.get("high", 0) + risk_level_counts.get("critical", 0),
                "safe_to_proceed": risk_level_counts.get("low", 0),
                "needs_review": risk_level_counts.get("medium", 0)
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
class WasteDetectionEngine:
    """
    Main waste detection and resource optimization engine
    
    Orchestrates all components to provide comprehensive waste detection,
    optimization recommendations, and risk assessment capabilities.
    """
    
    def __init__(self, config: Optional[WasteDetectionConfig] = None):
        self.logger = logging.getLogger(f"{__name__}.WasteDetectionEngine")
        
        # Initialize components
        self.resource_analyzer = ResourceAnalyzer()
        self.waste_identifier = WasteIdentifier(config)
        self.oversized_detector = OversizedResourceDetector()
        self.optimization_calculator = OptimizationCalculator()
        self.risk_assessor = RiskAssessor()
        
        self.logger.info("Waste Detection Engine initialized successfully")
    
    def analyze_and_optimize(self, 
                           resources: List[CloudResource],
                           metrics_data: Dict[str, MetricsData],
                           resource_metadata: Dict[str, ResourceMetadata],
                           capacity_data: Dict[str, ResourceCapacity],
                           cost_data: Optional[Dict[str, float]] = None,
                           analysis_period_days: int = 30) -> Dict[str, Any]:
        """
        Perform comprehensive waste detection and optimization analysis
        
        Args:
            resources: List of cloud resources to analyze
            metrics_data: Historical metrics data
            resource_metadata: Resource metadata
            capacity_data: Resource capacity configurations
            cost_data: Optional cost data for savings calculations
            analysis_period_days: Analysis period in days
            
        Returns:
            Comprehensive analysis results with recommendations
        """
        self.logger.info(f"Starting comprehensive waste detection analysis for {len(resources)} resources")
        
        try:
            # Step 1: Analyze resource utilization
            self.logger.info("Step 1: Analyzing resource utilization...")
            utilization_results = self.resource_analyzer.analyze_resource_utilization(
                resources, metrics_data, analysis_period_days
            )
            
            # Step 2: Identify waste
            self.logger.info("Step 2: Identifying waste...")
            waste_items = self.waste_identifier.identify_waste(
                utilization_results, resource_metadata, cost_data
            )
            
            # Step 3: Detect oversized resources
            self.logger.info("Step 3: Detecting oversized resources...")
            rightsizing_recommendations = self.oversized_detector.detect_oversized_resources(
                utilization_results, resource_metadata, capacity_data, cost_data
            )
            
            # Step 4: Generate optimization recommendations
            self.logger.info("Step 4: Generating optimization recommendations...")
            optimization_recommendations = self.optimization_calculator.generate_optimization_recommendations(
                waste_items, rightsizing_recommendations, resource_metadata
            )
            
            # Step 5: Assess risks
            self.logger.info("Step 5: Assessing risks...")
            risk_assessments = []
            for recommendation in optimization_recommendations:
                risk_assessment = self.risk_assessor.assess_optimization_risk(
                    recommendation, 
                    resource_metadata.get(recommendation.resource_id),
                    utilization_results.get(recommendation.resource_id)
                )
                risk_assessments.append(risk_assessment)
            
            # Generate comprehensive summary
            summary = self._generate_comprehensive_summary(
                utilization_results, waste_items, rightsizing_recommendations,
                optimization_recommendations, risk_assessments
            )
            
            self.logger.info("Comprehensive waste detection analysis completed successfully")
            
            return {
                "summary": summary,
                "utilization_analysis": {
                    "results": {k: v.to_dict() for k, v in utilization_results.items()},
                    "summary": self.resource_analyzer.get_utilization_summary()
                },
                "waste_detection": {
                    "waste_items": [item.to_dict() for item in waste_items],
                    "summary": self.waste_identifier.get_waste_summary(waste_items)
                },
                "rightsizing_analysis": {
                    "recommendations": [rec.to_dict() for rec in rightsizing_recommendations],
                    "total_monthly_savings": sum(rec.monthly_savings for rec in rightsizing_recommendations)
                },
                "optimization_recommendations": {
                    "recommendations": [rec.to_dict() for rec in optimization_recommendations],
                    "summary": self.optimization_calculator.get_optimization_summary(optimization_recommendations)
                },
                "risk_assessments": {
                    "assessments": [assessment.to_dict() for assessment in risk_assessments],
                    "summary": self.risk_assessor.get_risk_summary(risk_assessments)
                },
                "analysis_metadata": {
                    "analysis_period_days": analysis_period_days,
                    "resources_analyzed": len(resources),
                    "completed_at": datetime.now().isoformat(),
                    "engine_version": "1.0.0"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during waste detection analysis: {str(e)}")
            raise
    
    def get_quick_wins(self, 
                      optimization_recommendations: List[OptimizationRecommendation],
                      max_risk_level: RiskLevel = RiskLevel.MEDIUM,
                      min_monthly_savings: float = 100.0,
                      max_payback_months: float = 6.0) -> List[OptimizationRecommendation]:
        """
        Identify quick win optimization opportunities
        
        Args:
            optimization_recommendations: List of optimization recommendations
            max_risk_level: Maximum acceptable risk level
            min_monthly_savings: Minimum monthly savings threshold
            max_payback_months: Maximum payback period in months
            
        Returns:
            Filtered list of quick win recommendations
        """
        risk_level_values = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        
        max_risk_value = risk_level_values[max_risk_level]
        
        quick_wins = [
            rec for rec in optimization_recommendations
            if (risk_level_values[rec.risk_level] <= max_risk_value and
                rec.monthly_savings >= min_monthly_savings and
                rec.payback_period_months <= max_payback_months)
        ]
        
        # Sort by priority and monthly savings
        quick_wins.sort(key=lambda x: (x.priority, x.monthly_savings), reverse=True)
        
        self.logger.info(f"Identified {len(quick_wins)} quick win opportunities")
        
        return quick_wins
    
    def simulate_optimization_impact(self, 
                                   recommendations: List[OptimizationRecommendation],
                                   implementation_timeline_months: int = 12) -> Dict[str, Any]:
        """
        Simulate the financial impact of implementing optimization recommendations
        
        Args:
            recommendations: List of optimization recommendations
            implementation_timeline_months: Timeline for implementation in months
            
        Returns:
            Simulation results with financial projections
        """
        self.logger.info(f"Simulating optimization impact for {len(recommendations)} recommendations over {implementation_timeline_months} months")
        
        # Sort recommendations by priority for implementation order
        sorted_recommendations = sorted(recommendations, key=lambda x: x.priority, reverse=True)
        
        # Simulate monthly implementation
        monthly_results = []
        cumulative_savings = 0
        cumulative_costs = 0
        
        for month in range(implementation_timeline_months):
            month_savings = 0
            month_costs = 0
            implementations_this_month = []
            
            # Implement recommendations based on priority and capacity
            # Assume we can implement 5-10 recommendations per month
            max_implementations = min(10, len(sorted_recommendations))
            
            for i in range(min(max_implementations, len(sorted_recommendations))):
                rec = sorted_recommendations.pop(0)
                
                # Implementation cost occurs in the implementation month
                month_costs += rec.implementation_cost
                
                # Savings start accruing from the next month
                if month > 0:  # No savings in the first month
                    month_savings += rec.monthly_savings
                
                implementations_this_month.append({
                    "resource_id": rec.resource_id,
                    "optimization_type": rec.optimization_type.value,
                    "monthly_savings": rec.monthly_savings,
                    "implementation_cost": rec.implementation_cost
                })
            
            cumulative_savings += month_savings
            cumulative_costs += month_costs
            
            monthly_results.append({
                "month": month + 1,
                "implementations": len(implementations_this_month),
                "monthly_savings": round(month_savings, 2),
                "monthly_costs": round(month_costs, 2),
                "cumulative_savings": round(cumulative_savings, 2),
                "cumulative_costs": round(cumulative_costs, 2),
                "net_benefit": round(cumulative_savings - cumulative_costs, 2),
                "implementations_detail": implementations_this_month
            })
        
        # Calculate final metrics
        total_potential_savings = sum(rec.monthly_savings for rec in recommendations) * implementation_timeline_months
        total_implementation_costs = sum(rec.implementation_cost for rec in recommendations)
        net_benefit = cumulative_savings - cumulative_costs
        roi = (net_benefit / total_implementation_costs) * 100 if total_implementation_costs > 0 else 0
        
        return {
            "simulation_summary": {
                "timeline_months": implementation_timeline_months,
                "total_recommendations": len(recommendations),
                "total_potential_monthly_savings": round(sum(rec.monthly_savings for rec in recommendations), 2),
                "total_implementation_costs": round(total_implementation_costs, 2),
                "cumulative_savings": round(cumulative_savings, 2),
                "net_benefit": round(net_benefit, 2),
                "roi_percentage": round(roi, 1),
                "break_even_month": next((m["month"] for m in monthly_results if m["net_benefit"] > 0), None)
            },
            "monthly_progression": monthly_results,
            "recommendations_not_implemented": len(sorted_recommendations),
            "simulation_timestamp": datetime.now().isoformat()
        }
    
    def _generate_comprehensive_summary(self, 
                                      utilization_results: Dict[str, UtilizationMetrics],
                                      waste_items: List[WasteItem],
                                      rightsizing_recommendations: List[RightSizingRecommendation],
                                      optimization_recommendations: List[OptimizationRecommendation],
                                      risk_assessments: List[RiskAssessment]) -> Dict[str, Any]:
        """Generate comprehensive summary of analysis results"""
        
        # Calculate totals
        total_monthly_savings = sum(rec.monthly_savings for rec in optimization_recommendations)
        total_annual_savings = sum(rec.annual_savings for rec in optimization_recommendations)
        total_implementation_costs = sum(rec.implementation_cost for rec in optimization_recommendations)
        
        # Risk distribution
        risk_distribution = {}
        for level in RiskLevel:
            risk_distribution[level.value] = sum(
                1 for assessment in risk_assessments 
                if assessment.overall_risk_level == level
            )
        
        # Quick wins (low risk, high savings, short payback)
        quick_wins = self.get_quick_wins(optimization_recommendations)
        
        # Top opportunities
        top_opportunities = sorted(
            optimization_recommendations, 
            key=lambda x: x.monthly_savings, 
            reverse=True
        )[:10]
        
        return {
            "executive_summary": {
                "total_resources_analyzed": len(utilization_results),
                "waste_items_identified": len(waste_items),
                "optimization_opportunities": len(optimization_recommendations),
                "total_monthly_savings_potential": round(total_monthly_savings, 2),
                "total_annual_savings_potential": round(total_annual_savings, 2),
                "total_implementation_investment": round(total_implementation_costs, 2),
                "net_first_year_benefit": round(total_annual_savings - total_implementation_costs, 2),
                "quick_wins_available": len(quick_wins),
                "average_payback_months": round(
                    statistics.mean([
                        rec.payback_period_months for rec in optimization_recommendations 
                        if rec.payback_period_months != float('inf')
                    ]), 1
                ) if any(rec.payback_period_months != float('inf') for rec in optimization_recommendations) else 0
            },
            "risk_overview": {
                "risk_distribution": risk_distribution,
                "high_risk_optimizations": risk_distribution.get("high", 0) + risk_distribution.get("critical", 0),
                "safe_optimizations": risk_distribution.get("low", 0),
                "recommendations_needing_review": risk_distribution.get("medium", 0)
            },
            "quick_wins": {
                "count": len(quick_wins),
                "monthly_savings": round(sum(qw.monthly_savings for qw in quick_wins), 2),
                "average_payback_months": round(
                    statistics.mean([qw.payback_period_months for qw in quick_wins]), 1
                ) if quick_wins else 0
            },
            "top_opportunities": [
                {
                    "resource_id": opp.resource_id,
                    "optimization_type": opp.optimization_type.value,
                    "monthly_savings": round(opp.monthly_savings, 2),
                    "risk_level": opp.risk_level.value,
                    "payback_months": round(opp.payback_period_months, 1)
                }
                for opp in top_opportunities
            ],
            "utilization_insights": {
                "average_utilization": round(
                    statistics.mean([u.overall_utilization for u in utilization_results.values()]), 1
                ),
                "underutilized_resources": sum(
                    1 for u in utilization_results.values() 
                    if u.utilization_level in [UtilizationLevel.UNUSED, UtilizationLevel.UNDERUTILIZED]
                ),
                "optimization_potential_score": round(
                    statistics.mean([u.waste_score for u in utilization_results.values()]), 1
                )
            }
        }


# Example usage and testing functions
def create_sample_data():
    """Create sample data for testing the waste detection engine"""
    
    # Sample resources
    resources = [
        CloudResource(
            resource_id="i-1234567890abcdef0",
            resource_type="ec2_instance",
            provider=CloudProvider.AWS,
            region="us-east-1",
            tags={"environment": "production", "team": "backend"}
        ),
        CloudResource(
            resource_id="i-0987654321fedcba0",
            resource_type="ec2_instance", 
            provider=CloudProvider.AWS,
            region="us-east-1",
            tags={"environment": "development", "team": "frontend"}
        )
    ]
    
    # Sample metrics data
    now = datetime.now()
    metrics_data = {}
    
    for resource in resources:
        # Generate sample CPU and memory utilization data
        cpu_data = []
        memory_data = []
        
        for i in range(720):  # 30 days of hourly data
            timestamp = now - timedelta(hours=i)
            
            if resource.resource_id == "i-1234567890abcdef0":
                # High utilization resource
                cpu_util = 75 + (i % 24) * 2 + (i % 168) * 0.5  # Daily and weekly patterns
                memory_util = 80 + (i % 24) * 1.5
            else:
                # Low utilization resource (waste candidate)
                cpu_util = 5 + (i % 24) * 0.5
                memory_util = 10 + (i % 24) * 0.3
            
            cpu_data.append((timestamp, min(100, max(0, cpu_util))))
            memory_data.append((timestamp, min(100, max(0, memory_util))))
        
        metrics_data[resource.resource_id] = MetricsData(
            resource_id=resource.resource_id,
            metrics={
                MetricType.CPU_UTILIZATION: cpu_data,
                MetricType.MEMORY_UTILIZATION: memory_data,
                MetricType.DISK_IO: [(ts, 20.0) for ts, _ in cpu_data],
                MetricType.NETWORK_IO: [(ts, 15.0) for ts, _ in cpu_data]
            },
            collection_period_start=now - timedelta(days=30),
            collection_period_end=now
        )
    
    # Sample resource metadata
    resource_metadata = {
        "i-1234567890abcdef0": ResourceMetadata(
            resource_id="i-1234567890abcdef0",
            resource_type=ResourceType.COMPUTE_INSTANCE,
            provider_name="aws",
            provider_type="ec2",
            region="us-east-1",
            name="prod-backend-server",
            state=ResourceState.RUNNING,
            tags={"environment": "production", "team": "backend", "criticality": "high"}
        ),
        "i-0987654321fedcba0": ResourceMetadata(
            resource_id="i-0987654321fedcba0",
            resource_type=ResourceType.COMPUTE_INSTANCE,
            provider_name="aws",
            provider_type="ec2",
            region="us-east-1",
            name="dev-frontend-server",
            state=ResourceState.RUNNING,
            tags={"environment": "development", "team": "frontend", "criticality": "low"}
        )
    }
    
    # Sample capacity data
    capacity_data = {
        "i-1234567890abcdef0": ResourceCapacity(
            cpu_cores=4,
            memory_gb=16,
            disk_gb=100,
            network_bandwidth_mbps=1000,
            instance_type="m5.xlarge"
        ),
        "i-0987654321fedcba0": ResourceCapacity(
            cpu_cores=2,
            memory_gb=8,
            disk_gb=50,
            network_bandwidth_mbps=500,
            instance_type="t3.large"
        )
    }
    
    # Sample cost data
    cost_data = {
        "i-1234567890abcdef0": 192.0,  # $192/month
        "i-0987654321fedcba0": 83.2    # $83.2/month
    }
    
    return resources, metrics_data, resource_metadata, capacity_data, cost_data


if __name__ == "__main__":
    # Example usage
    print("Waste Detection Engine - Example Usage")
    print("=" * 50)
    
    # Create sample data
    resources, metrics_data, resource_metadata, capacity_data, cost_data = create_sample_data()
    
    # Initialize engine
    engine = WasteDetectionEngine()
    
    # Run comprehensive analysis
    results = engine.analyze_and_optimize(
        resources=resources,
        metrics_data=metrics_data,
        resource_metadata=resource_metadata,
        capacity_data=capacity_data,
        cost_data=cost_data,
        analysis_period_days=30
    )
    
    # Print summary
    summary = results["summary"]["executive_summary"]
    print(f"Resources Analyzed: {summary['total_resources_analyzed']}")
    print(f"Waste Items Found: {summary['waste_items_identified']}")
    print(f"Optimization Opportunities: {summary['optimization_opportunities']}")
    print(f"Potential Monthly Savings: ${summary['total_monthly_savings_potential']:.2f}")
    print(f"Potential Annual Savings: ${summary['total_annual_savings_potential']:.2f}")
    print(f"Quick Wins Available: {summary['quick_wins_available']}")
    
    # Get quick wins
    quick_wins = engine.get_quick_wins(results["optimization_recommendations"]["recommendations"])
    print(f"\nQuick Win Opportunities: {len(quick_wins)}")
    
    # Simulate implementation
    simulation = engine.simulate_optimization_impact(
        results["optimization_recommendations"]["recommendations"],
        implementation_timeline_months=12
    )
    
    sim_summary = simulation["simulation_summary"]
    print(f"\n12-Month Implementation Simulation:")
    print(f"Net Benefit: ${sim_summary['net_benefit']:.2f}")
    print(f"ROI: {sim_summary['roi_percentage']:.1f}%")
    print(f"Break-even Month: {sim_summary['break_even_month']}")