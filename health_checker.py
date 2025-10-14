# health_checker.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import statistics
import asyncio

from performance_monitor import MetricType, MetricsData, CloudResource
from enhanced_models import HealthStatus, SeverityLevel


@dataclass
class HealthCheck:
    """Defines a health check configuration"""
    check_id: str
    name: str
    check_type: str  # "metric_threshold", "availability", "response_time", "custom"
    metric_type: Optional[MetricType] = None
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    enabled: bool = True
    interval_minutes: int = 5
    timeout_seconds: int = 30
    failure_threshold: int = 3  # Number of consecutive failures before marking unhealthy
    recovery_threshold: int = 2  # Number of consecutive successes before marking healthy
    resource_filters: Dict[str, List[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HealthCheckResult:
    """Result of a health check execution"""
    check_id: str
    resource_id: str
    status: HealthStatus
    message: str
    execution_time: datetime
    response_time_ms: Optional[float] = None
    metric_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceHealth:
    """Overall health status of a resource"""
    resource_id: str
    overall_status: HealthStatus
    last_updated: datetime
    check_results: Dict[str, HealthCheckResult] = field(default_factory=dict)
    health_score: float = 100.0  # 0-100, higher is better
    uptime_percentage: float = 100.0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    recovery_time: Optional[datetime] = None


class HealthChecker:
    """Monitors resource health and availability"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.HealthChecker")
        self.health_checks: Dict[str, HealthCheck] = {}
        self.resource_health: Dict[str, ResourceHealth] = {}
        self.check_history: List[HealthCheckResult] = []
        self.failure_counters: Dict[str, int] = {}  # Track consecutive failures per resource/check
        self.success_counters: Dict[str, int] = {}  # Track consecutive successes per resource/check
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a new health check configuration"""
        self.health_checks[health_check.check_id] = health_check
        self.logger.info(f"Added health check: {health_check.name} ({health_check.check_id})")
    
    def remove_health_check(self, check_id: str):
        """Remove a health check configuration"""
        if check_id in self.health_checks:
            del self.health_checks[check_id]
            self.logger.info(f"Removed health check: {check_id}")
    
    async def execute_health_checks(self, resources: List[CloudResource], 
                                   metrics_data: Dict[str, MetricsData]) -> Dict[str, ResourceHealth]:
        """Execute all health checks for the given resources"""
        self.logger.info(f"Executing health checks for {len(resources)} resources")
        
        # Execute checks for each resource
        for resource in resources:
            await self._check_resource_health(resource, metrics_data.get(resource.resource_id))
        
        # Update overall health status
        self._update_overall_health_status()
        
        self.logger.info(f"Health checks completed for {len(self.resource_health)} resources")
        return self.resource_health
    
    async def _check_resource_health(self, resource: CloudResource, 
                                   metrics_data: Optional[MetricsData]):
        """Execute health checks for a specific resource"""
        resource_id = resource.resource_id
        
        # Initialize resource health if not exists
        if resource_id not in self.resource_health:
            self.resource_health[resource_id] = ResourceHealth(
                resource_id=resource_id,
                overall_status=HealthStatus.UNKNOWN,
                last_updated=datetime.now()
            )
        
        resource_health = self.resource_health[resource_id]
        check_results = {}
        
        # Execute applicable health checks
        for check_id, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            # Check if this health check applies to this resource
            if not self._resource_matches_filters(resource, health_check.resource_filters):
                continue
            
            # Execute the health check
            result = await self._execute_single_health_check(
                health_check, resource, metrics_data
            )
            
            if result:
                check_results[check_id] = result
                self.check_history.append(result)
        
        # Update resource health based on check results
        resource_health.check_results = check_results
        resource_health.last_updated = datetime.now()
        
        # Calculate overall health status and score
        self._calculate_resource_health_status(resource_health)
        
        # Keep history manageable
        if len(self.check_history) > 10000:
            self.check_history = self.check_history[-5000:]
    
    async def _execute_single_health_check(self, health_check: HealthCheck, 
                                         resource: CloudResource,
                                         metrics_data: Optional[MetricsData]) -> Optional[HealthCheckResult]:
        """Execute a single health check"""
        try:
            start_time = datetime.now()
            
            if health_check.check_type == "metric_threshold":
                result = self._check_metric_threshold(health_check, resource, metrics_data)
            elif health_check.check_type == "availability":
                result = await self._check_availability(health_check, resource)
            elif health_check.check_type == "response_time":
                result = await self._check_response_time(health_check, resource)
            elif health_check.check_type == "custom":
                result = await self._check_custom(health_check, resource, metrics_data)
            else:
                self.logger.warning(f"Unknown health check type: {health_check.check_type}")
                return None
            
            execution_time = datetime.now()
            response_time = (execution_time - start_time).total_seconds() * 1000  # ms
            
            if result:
                result.execution_time = execution_time
                result.response_time_ms = response_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing health check {health_check.check_id}: {e}")
            return HealthCheckResult(
                check_id=health_check.check_id,
                resource_id=resource.resource_id,
                status=HealthStatus.UNKNOWN,
                message=f"Health check execution failed: {str(e)}",
                execution_time=datetime.now()
            )
    
    def _check_metric_threshold(self, health_check: HealthCheck, 
                               resource: CloudResource,
                               metrics_data: Optional[MetricsData]) -> Optional[HealthCheckResult]:
        """Check if metric values are within healthy thresholds"""
        if not metrics_data or not health_check.metric_type:
            return HealthCheckResult(
                check_id=health_check.check_id,
                resource_id=resource.resource_id,
                status=HealthStatus.UNKNOWN,
                message="No metrics data available",
                execution_time=datetime.now()
            )
        
        if health_check.metric_type not in metrics_data.metrics:
            return HealthCheckResult(
                check_id=health_check.check_id,
                resource_id=resource.resource_id,
                status=HealthStatus.UNKNOWN,
                message=f"Metric {health_check.metric_type.value} not available",
                execution_time=datetime.now()
            )
        
        time_series = metrics_data.metrics[health_check.metric_type]
        if not time_series:
            return HealthCheckResult(
                check_id=health_check.check_id,
                resource_id=resource.resource_id,
                status=HealthStatus.UNKNOWN,
                message="No metric data points available",
                execution_time=datetime.now()
            )
        
        # Get latest metric value
        latest_timestamp, latest_value = time_series[-1]
        
        # Check thresholds
        status = HealthStatus.HEALTHY
        message = f"{health_check.metric_type.value}: {latest_value}"
        
        if health_check.threshold_min is not None and latest_value < health_check.threshold_min:
            status = HealthStatus.UNHEALTHY
            message += f" (below minimum threshold {health_check.threshold_min})"
        elif health_check.threshold_max is not None and latest_value > health_check.threshold_max:
            status = HealthStatus.UNHEALTHY
            message += f" (above maximum threshold {health_check.threshold_max})"
        elif (health_check.threshold_min is not None and 
              latest_value < health_check.threshold_min * 1.1):  # Warning zone
            status = HealthStatus.WARNING
            message += f" (approaching minimum threshold)"
        elif (health_check.threshold_max is not None and 
              latest_value > health_check.threshold_max * 0.9):  # Warning zone
            status = HealthStatus.WARNING
            message += f" (approaching maximum threshold)"
        
        return HealthCheckResult(
            check_id=health_check.check_id,
            resource_id=resource.resource_id,
            status=status,
            message=message,
            execution_time=datetime.now(),
            metric_value=latest_value
        )
    
    async def _check_availability(self, health_check: HealthCheck, 
                                resource: CloudResource) -> HealthCheckResult:
        """Check resource availability (simplified implementation)"""
        # In a real implementation, this would ping the resource or check its status
        # For now, we'll simulate availability check
        
        try:
            # Simulate network check with timeout
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Mock availability check - assume 99% uptime
            import random
            is_available = random.random() > 0.01  # 99% availability
            
            if is_available:
                return HealthCheckResult(
                    check_id=health_check.check_id,
                    resource_id=resource.resource_id,
                    status=HealthStatus.HEALTHY,
                    message="Resource is available",
                    execution_time=datetime.now()
                )
            else:
                return HealthCheckResult(
                    check_id=health_check.check_id,
                    resource_id=resource.resource_id,
                    status=HealthStatus.UNHEALTHY,
                    message="Resource is not responding",
                    execution_time=datetime.now()
                )
                
        except asyncio.TimeoutError:
            return HealthCheckResult(
                check_id=health_check.check_id,
                resource_id=resource.resource_id,
                status=HealthStatus.UNHEALTHY,
                message="Availability check timed out",
                execution_time=datetime.now()
            )
    
    async def _check_response_time(self, health_check: HealthCheck, 
                                 resource: CloudResource) -> HealthCheckResult:
        """Check resource response time"""
        start_time = datetime.now()
        
        try:
            # Simulate response time check
            await asyncio.sleep(0.05)  # Simulate request
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000  # ms
            
            # Determine status based on response time
            if response_time < 100:  # < 100ms
                status = HealthStatus.HEALTHY
                message = f"Response time: {response_time:.1f}ms (excellent)"
            elif response_time < 500:  # < 500ms
                status = HealthStatus.HEALTHY
                message = f"Response time: {response_time:.1f}ms (good)"
            elif response_time < 1000:  # < 1s
                status = HealthStatus.WARNING
                message = f"Response time: {response_time:.1f}ms (slow)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Response time: {response_time:.1f}ms (very slow)"
            
            return HealthCheckResult(
                check_id=health_check.check_id,
                resource_id=resource.resource_id,
                status=status,
                message=message,
                execution_time=datetime.now(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_id=health_check.check_id,
                resource_id=resource.resource_id,
                status=HealthStatus.UNHEALTHY,
                message=f"Response time check failed: {str(e)}",
                execution_time=datetime.now()
            )
    
    async def _check_custom(self, health_check: HealthCheck, 
                          resource: CloudResource,
                          metrics_data: Optional[MetricsData]) -> HealthCheckResult:
        """Execute custom health check logic"""
        # Placeholder for custom health check implementations
        # In a real system, this would allow pluggable health check logic
        
        return HealthCheckResult(
            check_id=health_check.check_id,
            resource_id=resource.resource_id,
            status=HealthStatus.HEALTHY,
            message="Custom health check passed",
            execution_time=datetime.now()
        )
    
    def _resource_matches_filters(self, resource: CloudResource, 
                                filters: Dict[str, List[str]]) -> bool:
        """Check if resource matches the filter criteria"""
        if not filters:
            return True  # No filters means match all
        
        # Check resource type filter
        if "resource_types" in filters:
            if resource.resource_type not in filters["resource_types"]:
                return False
        
        # Check provider filter
        if "providers" in filters:
            if resource.provider.value not in filters["providers"]:
                return False
        
        # Check region filter
        if "regions" in filters:
            if resource.region not in filters["regions"]:
                return False
        
        # Check tag filters
        if "tags" in filters:
            for required_tag in filters["tags"]:
                if ":" in required_tag:
                    key, value = required_tag.split(":", 1)
                    if resource.tags.get(key) != value:
                        return False
                else:
                    if required_tag not in resource.tags:
                        return False
        
        return True
    
    def _calculate_resource_health_status(self, resource_health: ResourceHealth):
        """Calculate overall health status and score for a resource"""
        if not resource_health.check_results:
            resource_health.overall_status = HealthStatus.UNKNOWN
            resource_health.health_score = 0.0
            return
        
        # Count status types
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in resource_health.check_results.values():
            status_counts[result.status] += 1
        
        total_checks = len(resource_health.check_results)
        
        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            resource_health.overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.WARNING] > 0:
            resource_health.overall_status = HealthStatus.WARNING
        elif status_counts[HealthStatus.HEALTHY] > 0:
            resource_health.overall_status = HealthStatus.HEALTHY
        else:
            resource_health.overall_status = HealthStatus.UNKNOWN
        
        # Calculate health score (0-100)
        healthy_weight = 100
        warning_weight = 60
        unhealthy_weight = 0
        unknown_weight = 30
        
        weighted_score = (
            status_counts[HealthStatus.HEALTHY] * healthy_weight +
            status_counts[HealthStatus.WARNING] * warning_weight +
            status_counts[HealthStatus.UNHEALTHY] * unhealthy_weight +
            status_counts[HealthStatus.UNKNOWN] * unknown_weight
        )
        
        resource_health.health_score = weighted_score / total_checks
        
        # Update failure tracking
        resource_id = resource_health.resource_id
        
        if resource_health.overall_status == HealthStatus.UNHEALTHY:
            resource_health.consecutive_failures += 1
            resource_health.last_failure_time = datetime.now()
            resource_health.recovery_time = None
        else:
            if resource_health.consecutive_failures > 0:
                resource_health.recovery_time = datetime.now()
            resource_health.consecutive_failures = 0
    
    def _update_overall_health_status(self):
        """Update overall health status for all resources"""
        for resource_health in self.resource_health.values():
            # Calculate uptime percentage based on recent history
            self._calculate_uptime_percentage(resource_health)
    
    def _calculate_uptime_percentage(self, resource_health: ResourceHealth, 
                                   period_hours: int = 24):
        """Calculate uptime percentage for a resource over the specified period"""
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        
        # Get recent check results for this resource
        recent_results = [
            result for result in self.check_history
            if (result.resource_id == resource_health.resource_id and 
                result.execution_time >= cutoff_time)
        ]
        
        if not recent_results:
            return  # No recent data
        
        # Count healthy vs unhealthy checks
        healthy_count = len([
            r for r in recent_results 
            if r.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
        ])
        
        total_count = len(recent_results)
        uptime_percentage = (healthy_count / total_count) * 100 if total_count > 0 else 100
        
        resource_health.uptime_percentage = uptime_percentage
    
    def get_resource_health(self, resource_id: str) -> Optional[ResourceHealth]:
        """Get health status for a specific resource"""
        return self.resource_health.get(resource_id)
    
    def get_unhealthy_resources(self) -> List[ResourceHealth]:
        """Get list of resources with unhealthy status"""
        return [
            health for health in self.resource_health.values()
            if health.overall_status == HealthStatus.UNHEALTHY
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary across all resources"""
        if not self.resource_health:
            return {
                'total_resources': 0,
                'by_status': {},
                'average_health_score': 0.0,
                'average_uptime': 0.0
            }
        
        # Count by status
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        health_scores = []
        uptime_percentages = []
        
        for health in self.resource_health.values():
            status_counts[health.overall_status] += 1
            health_scores.append(health.health_score)
            uptime_percentages.append(health.uptime_percentage)
        
        return {
            'total_resources': len(self.resource_health),
            'by_status': {status.value: count for status, count in status_counts.items() if count > 0},
            'average_health_score': statistics.mean(health_scores) if health_scores else 0.0,
            'average_uptime': statistics.mean(uptime_percentages) if uptime_percentages else 0.0,
            'resources_with_issues': len([
                h for h in self.resource_health.values() 
                if h.overall_status in [HealthStatus.WARNING, HealthStatus.UNHEALTHY]
            ])
        }
    
    def create_default_health_checks(self) -> List[HealthCheck]:
        """Create a set of default health checks"""
        default_checks = [
            HealthCheck(
                check_id="cpu_utilization_check",
                name="CPU Utilization Health Check",
                check_type="metric_threshold",
                metric_type=MetricType.CPU_UTILIZATION,
                threshold_max=95.0,
                interval_minutes=5
            ),
            HealthCheck(
                check_id="memory_utilization_check",
                name="Memory Utilization Health Check",
                check_type="metric_threshold",
                metric_type=MetricType.MEMORY_UTILIZATION,
                threshold_max=95.0,
                interval_minutes=5
            ),
            HealthCheck(
                check_id="error_rate_check",
                name="Error Rate Health Check",
                check_type="metric_threshold",
                metric_type=MetricType.ERROR_RATE,
                threshold_max=5.0,
                interval_minutes=2
            ),
            HealthCheck(
                check_id="availability_check",
                name="Resource Availability Check",
                check_type="availability",
                interval_minutes=1,
                timeout_seconds=10
            ),
            HealthCheck(
                check_id="response_time_check",
                name="Response Time Health Check",
                check_type="response_time",
                interval_minutes=5,
                timeout_seconds=30
            )
        ]
        
        # Add all default checks
        for check in default_checks:
            self.add_health_check(check)
        
        self.logger.info(f"Created {len(default_checks)} default health checks")
        return default_checks