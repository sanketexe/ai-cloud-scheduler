"""
Performance Metrics Collection and Monitoring

Tracks and analyzes performance metrics for the multi-cloud cost comparison engine,
including API response times, cache performance, and system resource utilization.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import statistics
import structlog

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    API_RESPONSE_TIME = "api_response_time"
    CACHE_HIT_RATE = "cache_hit_rate"
    COMPARISON_PROCESSING_TIME = "comparison_processing_time"
    PARALLEL_EFFICIENCY = "parallel_efficiency"
    DATABASE_QUERY_TIME = "database_query_time"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric"""
    metric_name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    median_value: float
    p95_value: float
    p99_value: float
    std_deviation: float
    time_range: Tuple[datetime, datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'metric_name': self.metric_name,
            'count': self.count,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'avg_value': self.avg_value,
            'median_value': self.median_value,
            'p95_value': self.p95_value,
            'p99_value': self.p99_value,
            'std_deviation': self.std_deviation,
            'time_range': {
                'start': self.time_range[0].isoformat(),
                'end': self.time_range[1].isoformat()
            }
        }


@dataclass
class PerformanceAlert:
    """Performance alert configuration and state"""
    metric_name: str
    threshold_value: float
    comparison_operator: str  # 'gt', 'lt', 'eq'
    window_minutes: int
    alert_enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


class MetricBuffer:
    """Circular buffer for storing metric points"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.total_points = 0
    
    def add_point(self, point: MetricPoint):
        """Add a metric point to the buffer"""
        self.buffer.append(point)
        self.total_points += 1
    
    def get_points(self, since: datetime = None, until: datetime = None) -> List[MetricPoint]:
        """Get metric points within time range"""
        points = list(self.buffer)
        
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        if until:
            points = [p for p in points if p.timestamp <= until]
        
        return points
    
    def get_values(self, since: datetime = None, until: datetime = None) -> List[float]:
        """Get metric values within time range"""
        points = self.get_points(since, until)
        return [p.value for p in points]
    
    def clear(self):
        """Clear all points from buffer"""
        self.buffer.clear()


class PerformanceMetrics:
    """
    Performance metrics collection and analysis system.
    
    Tracks various performance metrics, provides statistical analysis,
    and supports alerting on performance thresholds.
    """
    
    def __init__(self, buffer_size: int = 10000, retention_hours: int = 24):
        self.buffer_size = buffer_size
        self.retention_hours = retention_hours
        
        # Metric storage
        self.metrics: Dict[str, MetricBuffer] = defaultdict(
            lambda: MetricBuffer(buffer_size)
        )
        
        # Performance alerts
        self.alerts: Dict[str, PerformanceAlert] = {}
        
        # System state tracking
        self.start_time = datetime.utcnow()
        self.last_cleanup = datetime.utcnow()
        
        # Performance thresholds (can be configured)
        self.default_thresholds = {
            MetricType.API_RESPONSE_TIME.value: 5000.0,  # 5 seconds
            MetricType.COMPARISON_PROCESSING_TIME.value: 30000.0,  # 30 seconds
            MetricType.CACHE_HIT_RATE.value: 0.8,  # 80%
            MetricType.ERROR_RATE.value: 0.05,  # 5%
            MetricType.PARALLEL_EFFICIENCY.value: 2.0,  # 2x improvement
        }
        
        # Initialize default alerts
        self._setup_default_alerts()
    
    def track_api_response_time(self, provider: str, duration: float):
        """
        Track API response time for a provider.
        
        Args:
            provider: Cloud provider name (aws, gcp, azure)
            duration: Response time in milliseconds
        """
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=duration,
            labels={'provider': provider},
            metadata={'metric_type': MetricType.API_RESPONSE_TIME.value}
        )
        
        metric_key = f"{MetricType.API_RESPONSE_TIME.value}:{provider}"
        self.metrics[metric_key].add_point(point)
        
        # Also track overall API response time
        overall_key = MetricType.API_RESPONSE_TIME.value
        self.metrics[overall_key].add_point(point)
        
        # Check alerts
        self._check_alert(overall_key, duration)
        
        logger.debug(
            "API response time tracked",
            provider=provider,
            duration_ms=duration
        )
    
    def track_cache_hit_rate(self, cache_type: str, hit_rate: float):
        """
        Track cache hit rate for different cache types.
        
        Args:
            cache_type: Type of cache (memory, redis, etc.)
            hit_rate: Hit rate as a decimal (0.0 to 1.0)
        """
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=hit_rate,
            labels={'cache_type': cache_type},
            metadata={'metric_type': MetricType.CACHE_HIT_RATE.value}
        )
        
        metric_key = f"{MetricType.CACHE_HIT_RATE.value}:{cache_type}"
        self.metrics[metric_key].add_point(point)
        
        # Also track overall cache hit rate
        overall_key = MetricType.CACHE_HIT_RATE.value
        self.metrics[overall_key].add_point(point)
        
        # Check alerts (alert if hit rate is too low)
        self._check_alert(overall_key, hit_rate, invert_threshold=True)
        
        logger.debug(
            "Cache hit rate tracked",
            cache_type=cache_type,
            hit_rate=hit_rate
        )
    
    def track_comparison_processing_time(self, workload_complexity: int, duration: float):
        """
        Track cost comparison processing time.
        
        Args:
            workload_complexity: Complexity score of the workload (1-10)
            duration: Processing time in milliseconds
        """
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=duration,
            labels={'complexity': str(workload_complexity)},
            metadata={
                'metric_type': MetricType.COMPARISON_PROCESSING_TIME.value,
                'workload_complexity': workload_complexity
            }
        )
        
        metric_key = MetricType.COMPARISON_PROCESSING_TIME.value
        self.metrics[metric_key].add_point(point)
        
        # Track by complexity level
        complexity_key = f"{metric_key}:complexity_{workload_complexity}"
        self.metrics[complexity_key].add_point(point)
        
        # Check alerts
        self._check_alert(metric_key, duration)
        
        logger.debug(
            "Comparison processing time tracked",
            complexity=workload_complexity,
            duration_ms=duration
        )
    
    def track_parallel_efficiency(self, provider_count: int, efficiency_ratio: float):
        """
        Track parallel processing efficiency.
        
        Args:
            provider_count: Number of providers processed in parallel
            efficiency_ratio: Efficiency ratio (speedup factor)
        """
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=efficiency_ratio,
            labels={'provider_count': str(provider_count)},
            metadata={'metric_type': MetricType.PARALLEL_EFFICIENCY.value}
        )
        
        metric_key = MetricType.PARALLEL_EFFICIENCY.value
        self.metrics[metric_key].add_point(point)
        
        logger.debug(
            "Parallel efficiency tracked",
            provider_count=provider_count,
            efficiency_ratio=efficiency_ratio
        )
    
    def track_database_query_time(self, query_type: str, duration: float):
        """
        Track database query performance.
        
        Args:
            query_type: Type of database query
            duration: Query execution time in milliseconds
        """
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=duration,
            labels={'query_type': query_type},
            metadata={'metric_type': MetricType.DATABASE_QUERY_TIME.value}
        )
        
        metric_key = f"{MetricType.DATABASE_QUERY_TIME.value}:{query_type}"
        self.metrics[metric_key].add_point(point)
        
        # Also track overall database performance
        overall_key = MetricType.DATABASE_QUERY_TIME.value
        self.metrics[overall_key].add_point(point)
        
        logger.debug(
            "Database query time tracked",
            query_type=query_type,
            duration_ms=duration
        )
    
    def track_memory_usage(self, component: str, usage_mb: float):
        """
        Track memory usage for different components.
        
        Args:
            component: Component name (cache, engine, etc.)
            usage_mb: Memory usage in megabytes
        """
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=usage_mb,
            labels={'component': component},
            metadata={'metric_type': MetricType.MEMORY_USAGE.value}
        )
        
        metric_key = f"{MetricType.MEMORY_USAGE.value}:{component}"
        self.metrics[metric_key].add_point(point)
        
        logger.debug(
            "Memory usage tracked",
            component=component,
            usage_mb=usage_mb
        )
    
    def track_error_rate(self, component: str, error_rate: float):
        """
        Track error rate for different components.
        
        Args:
            component: Component name
            error_rate: Error rate as a decimal (0.0 to 1.0)
        """
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=error_rate,
            labels={'component': component},
            metadata={'metric_type': MetricType.ERROR_RATE.value}
        )
        
        metric_key = f"{MetricType.ERROR_RATE.value}:{component}"
        self.metrics[metric_key].add_point(point)
        
        # Check alerts
        self._check_alert(MetricType.ERROR_RATE.value, error_rate)
        
        logger.debug(
            "Error rate tracked",
            component=component,
            error_rate=error_rate
        )
    
    def track_throughput(self, operation: str, requests_per_second: float):
        """
        Track throughput for different operations.
        
        Args:
            operation: Operation name
            requests_per_second: Throughput in requests per second
        """
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=requests_per_second,
            labels={'operation': operation},
            metadata={'metric_type': MetricType.THROUGHPUT.value}
        )
        
        metric_key = f"{MetricType.THROUGHPUT.value}:{operation}"
        self.metrics[metric_key].add_point(point)
        
        logger.debug(
            "Throughput tracked",
            operation=operation,
            rps=requests_per_second
        )
    
    def get_metric_summary(self, metric_name: str, hours: int = 1) -> Optional[MetricSummary]:
        """
        Get statistical summary for a metric over the specified time period.
        
        Args:
            metric_name: Name of the metric
            hours: Number of hours to analyze
            
        Returns:
            MetricSummary with statistical analysis
        """
        if metric_name not in self.metrics:
            return None
        
        since = datetime.utcnow() - timedelta(hours=hours)
        values = self.metrics[metric_name].get_values(since=since)
        
        if not values:
            return None
        
        # Calculate statistics
        sorted_values = sorted(values)
        count = len(values)
        
        min_value = min(values)
        max_value = max(values)
        avg_value = statistics.mean(values)
        median_value = statistics.median(values)
        
        # Calculate percentiles
        p95_index = int(0.95 * count)
        p99_index = int(0.99 * count)
        p95_value = sorted_values[min(p95_index, count - 1)]
        p99_value = sorted_values[min(p99_index, count - 1)]
        
        # Calculate standard deviation
        std_deviation = statistics.stdev(values) if count > 1 else 0.0
        
        # Get time range
        points = self.metrics[metric_name].get_points(since=since)
        time_range = (
            min(p.timestamp for p in points),
            max(p.timestamp for p in points)
        )
        
        return MetricSummary(
            metric_name=metric_name,
            count=count,
            min_value=min_value,
            max_value=max_value,
            avg_value=avg_value,
            median_value=median_value,
            p95_value=p95_value,
            p99_value=p99_value,
            std_deviation=std_deviation,
            time_range=time_range
        )
    
    def get_all_metrics_summary(self, hours: int = 1) -> Dict[str, MetricSummary]:
        """Get summary for all tracked metrics"""
        summaries = {}
        
        for metric_name in self.metrics.keys():
            summary = self.get_metric_summary(metric_name, hours)
            if summary:
                summaries[metric_name] = summary
        
        return summaries
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive performance data for dashboard display.
        
        Returns:
            Dictionary with performance metrics and summaries
        """
        dashboard_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_hours': (datetime.utcnow() - self.start_time).total_seconds() / 3600,
            'metrics_summary': {},
            'recent_alerts': [],
            'system_health': {}
        }
        
        # Get summaries for key metrics
        key_metrics = [
            MetricType.API_RESPONSE_TIME.value,
            MetricType.CACHE_HIT_RATE.value,
            MetricType.COMPARISON_PROCESSING_TIME.value,
            MetricType.PARALLEL_EFFICIENCY.value,
            MetricType.ERROR_RATE.value
        ]
        
        for metric in key_metrics:
            summary = self.get_metric_summary(metric, hours=1)
            if summary:
                dashboard_data['metrics_summary'][metric] = summary.to_dict()
        
        # Get recent alerts
        recent_alerts = []
        for alert_name, alert in self.alerts.items():
            if alert.last_triggered and alert.last_triggered > datetime.utcnow() - timedelta(hours=24):
                recent_alerts.append({
                    'metric_name': alert.metric_name,
                    'threshold_value': alert.threshold_value,
                    'last_triggered': alert.last_triggered.isoformat(),
                    'trigger_count': alert.trigger_count
                })
        
        dashboard_data['recent_alerts'] = recent_alerts
        
        # Calculate system health score
        dashboard_data['system_health'] = self._calculate_system_health()
        
        return dashboard_data
    
    def add_performance_alert(self, metric_name: str, threshold_value: float, 
                            comparison_operator: str = 'gt', window_minutes: int = 5):
        """
        Add a performance alert for a metric.
        
        Args:
            metric_name: Name of the metric to monitor
            threshold_value: Threshold value for alerting
            comparison_operator: 'gt' (greater than), 'lt' (less than), 'eq' (equal)
            window_minutes: Time window for alert evaluation
        """
        alert = PerformanceAlert(
            metric_name=metric_name,
            threshold_value=threshold_value,
            comparison_operator=comparison_operator,
            window_minutes=window_minutes
        )
        
        self.alerts[metric_name] = alert
        
        logger.info(
            "Performance alert added",
            metric_name=metric_name,
            threshold=threshold_value,
            operator=comparison_operator
        )
    
    def _setup_default_alerts(self):
        """Setup default performance alerts"""
        for metric_name, threshold in self.default_thresholds.items():
            if metric_name == MetricType.CACHE_HIT_RATE.value:
                # Alert if cache hit rate is too low
                self.add_performance_alert(metric_name, threshold, 'lt')
            elif metric_name == MetricType.ERROR_RATE.value:
                # Alert if error rate is too high
                self.add_performance_alert(metric_name, threshold, 'gt')
            else:
                # Alert if response time/processing time is too high
                self.add_performance_alert(metric_name, threshold, 'gt')
    
    def _check_alert(self, metric_name: str, value: float, invert_threshold: bool = False):
        """Check if a metric value triggers an alert"""
        if metric_name not in self.alerts:
            return
        
        alert = self.alerts[metric_name]
        if not alert.alert_enabled:
            return
        
        # Check threshold
        triggered = False
        
        if alert.comparison_operator == 'gt':
            triggered = value > alert.threshold_value
        elif alert.comparison_operator == 'lt':
            triggered = value < alert.threshold_value
        elif alert.comparison_operator == 'eq':
            triggered = abs(value - alert.threshold_value) < 0.001  # Float comparison
        
        # Invert for metrics where lower is worse (like cache hit rate)
        if invert_threshold:
            triggered = not triggered
        
        if triggered:
            alert.last_triggered = datetime.utcnow()
            alert.trigger_count += 1
            
            logger.warning(
                "Performance alert triggered",
                metric_name=metric_name,
                value=value,
                threshold=alert.threshold_value,
                operator=alert.comparison_operator,
                trigger_count=alert.trigger_count
            )
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        health_score = 100.0
        health_factors = {}
        
        # Check API response times
        api_summary = self.get_metric_summary(MetricType.API_RESPONSE_TIME.value, hours=1)
        if api_summary and api_summary.avg_value > 2000:  # 2 seconds
            penalty = min(20, (api_summary.avg_value - 2000) / 100)
            health_score -= penalty
            health_factors['api_response_time'] = f"Average response time: {api_summary.avg_value:.0f}ms"
        
        # Check cache hit rate
        cache_summary = self.get_metric_summary(MetricType.CACHE_HIT_RATE.value, hours=1)
        if cache_summary and cache_summary.avg_value < 0.8:  # 80%
            penalty = (0.8 - cache_summary.avg_value) * 25
            health_score -= penalty
            health_factors['cache_hit_rate'] = f"Cache hit rate: {cache_summary.avg_value:.1%}"
        
        # Check error rate
        error_summary = self.get_metric_summary(MetricType.ERROR_RATE.value, hours=1)
        if error_summary and error_summary.avg_value > 0.01:  # 1%
            penalty = error_summary.avg_value * 500  # Heavy penalty for errors
            health_score -= penalty
            health_factors['error_rate'] = f"Error rate: {error_summary.avg_value:.1%}"
        
        # Ensure health score is between 0 and 100
        health_score = max(0, min(100, health_score))
        
        return {
            'score': health_score,
            'status': self._get_health_status(health_score),
            'factors': health_factors
        }
    
    def _get_health_status(self, score: float) -> str:
        """Get health status based on score"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        elif score >= 40:
            return "poor"
        else:
            return "critical"
    
    async def cleanup_old_metrics(self):
        """Clean up old metric data beyond retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        cleaned_count = 0
        for metric_name, buffer in self.metrics.items():
            # Get points to keep
            points_to_keep = [
                point for point in buffer.buffer 
                if point.timestamp > cutoff_time
            ]
            
            # Clear and repopulate buffer
            old_count = len(buffer.buffer)
            buffer.clear()
            
            for point in points_to_keep:
                buffer.add_point(point)
            
            cleaned_count += old_count - len(points_to_keep)
        
        self.last_cleanup = datetime.utcnow()
        
        logger.info(
            "Metrics cleanup completed",
            cleaned_points=cleaned_count,
            retention_hours=self.retention_hours
        )
    
    def export_metrics(self, format: str = 'json') -> Dict[str, Any]:
        """
        Export all metrics data for external analysis.
        
        Args:
            format: Export format ('json' supported)
            
        Returns:
            Exported metrics data
        """
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'system_info': {
                'start_time': self.start_time.isoformat(),
                'uptime_hours': (datetime.utcnow() - self.start_time).total_seconds() / 3600,
                'buffer_size': self.buffer_size,
                'retention_hours': self.retention_hours
            },
            'metrics': {},
            'summaries': {}
        }
        
        # Export raw metrics
        for metric_name, buffer in self.metrics.items():
            points_data = []
            for point in buffer.buffer:
                points_data.append({
                    'timestamp': point.timestamp.isoformat(),
                    'value': point.value,
                    'labels': point.labels,
                    'metadata': point.metadata
                })
            
            export_data['metrics'][metric_name] = {
                'total_points': buffer.total_points,
                'current_points': len(buffer.buffer),
                'points': points_data
            }
        
        # Export summaries
        export_data['summaries'] = {
            name: summary.to_dict() 
            for name, summary in self.get_all_metrics_summary(hours=24).items()
        }
        
        return export_data