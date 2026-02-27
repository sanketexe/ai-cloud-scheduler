"""
Predictive Maintenance System for Infrastructure Health

This system provides AI-powered predictive maintenance capabilities for cloud infrastructure,
including health metrics analysis, maintenance scheduling, degradation detection, and
effectiveness tracking to prevent cost overruns and performance issues.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import asyncio
from collections import defaultdict, deque
import statistics
import numpy as np
from pathlib import Path
import sqlite3
from contextlib import contextmanager

try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Infrastructure health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"

class MaintenanceType(Enum):
    """Types of maintenance actions"""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"

class MaintenancePriority(Enum):
    """Maintenance priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DegradationPattern(Enum):
    """Types of degradation patterns"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SUDDEN = "sudden"
    CYCLICAL = "cyclical"
    IRREGULAR = "irregular"

@dataclass
class HealthMetric:
    """Infrastructure health metric data point"""
    resource_id: str
    metric_name: str
    value: float
    timestamp: datetime
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthAssessment:
    """Comprehensive health assessment for a resource"""
    resource_id: str
    resource_type: str
    overall_status: HealthStatus
    health_score: float  # 0-100, higher is better
    metrics: List[HealthMetric]
    issues_detected: List[str]
    recommendations: List[str]
    assessed_at: datetime
    next_assessment: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation with timing and details"""
    recommendation_id: str
    resource_id: str
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    recommended_window: Tuple[datetime, datetime]
    estimated_duration: timedelta
    description: str
    expected_benefits: List[str]
    risks_if_delayed: List[str]
    cost_estimate: Optional[float] = None
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DegradationAlert:
    """Alert for detected resource degradation"""
    alert_id: str
    resource_id: str
    degradation_pattern: DegradationPattern
    severity: HealthStatus
    detected_at: datetime
    predicted_failure_time: Optional[datetime]
    confidence: float
    description: str
    recommended_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MaintenanceOutcome:
    """Record of maintenance action outcome"""
    outcome_id: str
    recommendation_id: str
    resource_id: str
    executed_at: datetime
    duration: timedelta
    success: bool
    improvements_observed: List[str]
    issues_resolved: List[str]
    cost_actual: Optional[float] = None
    effectiveness_score: float = 0.0
    notes: str = ""

class HealthMetricsAnalyzer:
    """Analyzes infrastructure health metrics and detects issues"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.metric_history: Dict[str, List[HealthMetric]] = defaultdict(list)
        self.baseline_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.health_thresholds: Dict[str, Dict[str, float]] = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for health metrics analyzer"""
        return {
            "analysis_window_hours": 24,
            "baseline_training_days": 14,
            "anomaly_sensitivity": 0.1,
            "health_score_weights": {
                "cpu_utilization": 0.2,
                "memory_utilization": 0.2,
                "disk_utilization": 0.15,
                "network_latency": 0.15,
                "error_rate": 0.15,
                "availability": 0.15
            },
            "default_thresholds": {
                "cpu_utilization": {"warning": 70.0, "critical": 90.0},
                "memory_utilization": {"warning": 80.0, "critical": 95.0},
                "disk_utilization": {"warning": 85.0, "critical": 95.0},
                "network_latency": {"warning": 100.0, "critical": 500.0},
                "error_rate": {"warning": 1.0, "critical": 5.0},
                "availability": {"warning": 99.0, "critical": 95.0}
            }
        }
    
    async def analyze_resource_health(self, resource_id: str, 
                                    metrics: List[HealthMetric]) -> HealthAssessment:
        """Analyze health of a specific resource"""
        logger.info(f"Analyzing health for resource: {resource_id}")
        
        # Store metrics in history
        self.metric_history[resource_id].extend(metrics)
        
        # Calculate health score
        health_score = await self._calculate_health_score(resource_id, metrics)
        
        # Determine overall status
        overall_status = self._determine_health_status(health_score, metrics)
        
        # Detect issues
        issues = await self._detect_health_issues(resource_id, metrics)
        
        # Generate recommendations
        recommendations = await self._generate_health_recommendations(
            resource_id, metrics, issues
        )
        
        # Determine next assessment time
        next_assessment = self._calculate_next_assessment_time(overall_status)
        
        assessment = HealthAssessment(
            resource_id=resource_id,
            resource_type=self._infer_resource_type(resource_id),
            overall_status=overall_status,
            health_score=health_score,
            metrics=metrics,
            issues_detected=issues,
            recommendations=recommendations,
            assessed_at=datetime.now(),
            next_assessment=next_assessment
        )
        
        logger.info(f"Health assessment completed for {resource_id}: "
                   f"Status={overall_status.value}, Score={health_score:.1f}")
        
        return assessment
    
    async def _calculate_health_score(self, resource_id: str, 
                                    metrics: List[HealthMetric]) -> float:
        """Calculate overall health score for a resource"""
        if not metrics:
            return 0.0
        
        weights = self.config["health_score_weights"]
        weighted_scores = []
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.metric_name].append(metric)
        
        for metric_name, metric_list in metric_groups.items():
            if metric_name not in weights:
                continue
                
            # Calculate metric score (0-100)
            metric_score = await self._calculate_metric_score(metric_name, metric_list)
            weight = weights[metric_name]
            weighted_scores.append(metric_score * weight)
        
        # Calculate weighted average
        if weighted_scores:
            return sum(weighted_scores) / sum(weights[name] for name in metric_groups.keys() if name in weights)
        
        return 50.0  # Default neutral score
    
    async def _calculate_metric_score(self, metric_name: str, 
                                    metrics: List[HealthMetric]) -> float:
        """Calculate score for a specific metric"""
        if not metrics:
            return 50.0
        
        # Get latest metric value
        latest_metric = max(metrics, key=lambda m: m.timestamp)
        value = latest_metric.value
        
        # Get thresholds
        thresholds = self.config["default_thresholds"].get(metric_name, {})
        warning_threshold = thresholds.get("warning", 80.0)
        critical_threshold = thresholds.get("critical", 95.0)
        
        # Calculate score based on thresholds
        if metric_name in ["cpu_utilization", "memory_utilization", "disk_utilization", "error_rate"]:
            # Higher values are worse
            if value >= critical_threshold:
                return 0.0
            elif value >= warning_threshold:
                return 30.0 * (1 - (value - warning_threshold) / (critical_threshold - warning_threshold))
            else:
                return 100.0 * (1 - value / warning_threshold)
        
        elif metric_name == "availability":
            # Higher values are better
            if value >= warning_threshold:
                return 100.0
            elif value >= critical_threshold:
                return 50.0 + 50.0 * (value - critical_threshold) / (warning_threshold - critical_threshold)
            else:
                return 50.0 * value / critical_threshold
        
        elif metric_name == "network_latency":
            # Lower values are better
            if value >= critical_threshold:
                return 0.0
            elif value >= warning_threshold:
                return 30.0 * (1 - (value - warning_threshold) / (critical_threshold - warning_threshold))
            else:
                return 100.0 * (1 - value / warning_threshold)
        
        return 50.0  # Default neutral score
    
    def _determine_health_status(self, health_score: float, 
                               metrics: List[HealthMetric]) -> HealthStatus:
        """Determine overall health status based on score and metrics"""
        # Check for critical metrics first
        for metric in metrics:
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                return HealthStatus.CRITICAL
        
        # Check for warning metrics
        warning_count = 0
        for metric in metrics:
            if metric.threshold_warning and metric.value >= metric.threshold_warning:
                warning_count += 1
        
        # Determine status based on health score and warning count
        if health_score >= 80 and warning_count == 0:
            return HealthStatus.HEALTHY
        elif health_score >= 60 and warning_count <= 1:
            return HealthStatus.WARNING
        elif health_score >= 30:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.CRITICAL
    
    async def _detect_health_issues(self, resource_id: str, 
                                  metrics: List[HealthMetric]) -> List[str]:
        """Detect specific health issues from metrics"""
        issues = []
        
        for metric in metrics:
            # Check threshold violations
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                issues.append(f"Critical threshold exceeded for {metric.metric_name}: "
                            f"{metric.value:.2f} >= {metric.threshold_critical:.2f}")
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                issues.append(f"Warning threshold exceeded for {metric.metric_name}: "
                            f"{metric.value:.2f} >= {metric.threshold_warning:.2f}")
        
        # Check for anomalies using historical data
        if resource_id in self.metric_history:
            anomalies = await self._detect_metric_anomalies(resource_id, metrics)
            issues.extend(anomalies)
        
        return issues
    
    async def _detect_metric_anomalies(self, resource_id: str, 
                                     current_metrics: List[HealthMetric]) -> List[str]:
        """Detect anomalies in metrics using historical data"""
        anomalies = []
        
        if not SKLEARN_AVAILABLE:
            return anomalies
        
        historical_metrics = self.metric_history[resource_id]
        
        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in historical_metrics:
            metric_groups[metric.metric_name].append(metric.value)
        
        for current_metric in current_metrics:
            metric_name = current_metric.metric_name
            if metric_name not in metric_groups or len(metric_groups[metric_name]) < 10:
                continue
            
            # Calculate statistical anomaly
            historical_values = metric_groups[metric_name][-100:]  # Last 100 values
            mean_val = statistics.mean(historical_values)
            std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
            
            if std_val > 0:
                z_score = abs(current_metric.value - mean_val) / std_val
                if z_score > 3:  # 3-sigma rule
                    anomalies.append(f"Anomalous {metric_name} detected: "
                                   f"{current_metric.value:.2f} (z-score: {z_score:.2f})")
        
        return anomalies
    
    async def _generate_health_recommendations(self, resource_id: str, 
                                             metrics: List[HealthMetric], 
                                             issues: List[str]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        # Analyze metrics for specific recommendations
        for metric in metrics:
            if metric.metric_name == "cpu_utilization" and metric.value > 80:
                recommendations.append("Consider scaling up CPU resources or optimizing workload")
            elif metric.metric_name == "memory_utilization" and metric.value > 85:
                recommendations.append("Increase memory allocation or optimize memory usage")
            elif metric.metric_name == "disk_utilization" and metric.value > 90:
                recommendations.append("Add storage capacity or implement data archival")
            elif metric.metric_name == "error_rate" and metric.value > 2:
                recommendations.append("Investigate and fix underlying errors causing high error rate")
            elif metric.metric_name == "network_latency" and metric.value > 200:
                recommendations.append("Optimize network configuration or consider regional deployment")
        
        # General recommendations based on issues
        if len(issues) > 3:
            recommendations.append("Schedule comprehensive health check and maintenance")
        
        if not recommendations:
            recommendations.append("Continue monitoring - no immediate action required")
        
        return recommendations
    
    def _calculate_next_assessment_time(self, status: HealthStatus) -> datetime:
        """Calculate when next health assessment should occur"""
        now = datetime.now()
        
        if status == HealthStatus.CRITICAL:
            return now + timedelta(hours=1)
        elif status == HealthStatus.DEGRADED:
            return now + timedelta(hours=4)
        elif status == HealthStatus.WARNING:
            return now + timedelta(hours=12)
        else:
            return now + timedelta(hours=24)
    
    def _infer_resource_type(self, resource_id: str) -> str:
        """Infer resource type from resource ID"""
        # Simple heuristic based on resource ID patterns
        if "ec2" in resource_id.lower():
            return "compute"
        elif "rds" in resource_id.lower():
            return "database"
        elif "s3" in resource_id.lower():
            return "storage"
        elif "elb" in resource_id.lower() or "alb" in resource_id.lower():
            return "load_balancer"
        else:
            return "unknown"

class DegradationDetector:
    """Detects resource degradation patterns and predicts failures"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.degradation_models: Dict[str, Any] = {}
        self.pattern_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for degradation detector"""
        return {
            "prediction_horizon_hours": 72,
            "min_data_points": 20,
            "degradation_threshold": 0.15,  # 15% degradation
            "pattern_detection_window": 168,  # 1 week in hours
            "confidence_threshold": 0.7
        }
    
    async def detect_degradation(self, resource_id: str, 
                               health_history: List[HealthAssessment]) -> Optional[DegradationAlert]:
        """Detect degradation patterns in resource health"""
        if len(health_history) < self.config["min_data_points"]:
            return None
        
        logger.info(f"Analyzing degradation patterns for resource: {resource_id}")
        
        # Extract health scores over time
        time_series = [(assessment.assessed_at, assessment.health_score) 
                      for assessment in health_history]
        time_series.sort(key=lambda x: x[0])
        
        # Detect degradation pattern
        pattern = await self._identify_degradation_pattern(time_series)
        
        if pattern["type"] == DegradationPattern.LINEAR:
            return await self._handle_linear_degradation(resource_id, time_series, pattern)
        elif pattern["type"] == DegradationPattern.EXPONENTIAL:
            return await self._handle_exponential_degradation(resource_id, time_series, pattern)
        elif pattern["type"] == DegradationPattern.SUDDEN:
            return await self._handle_sudden_degradation(resource_id, time_series, pattern)
        
        return None
    
    async def _identify_degradation_pattern(self, time_series: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Identify the type of degradation pattern"""
        if len(time_series) < 5:
            return {"type": None, "confidence": 0.0}
        
        # Extract values and calculate trends
        values = [score for _, score in time_series]
        
        # Check for sudden drops
        for i in range(1, len(values)):
            drop_percentage = (values[i-1] - values[i]) / values[i-1] if values[i-1] > 0 else 0
            if drop_percentage > 0.3:  # 30% sudden drop
                return {
                    "type": DegradationPattern.SUDDEN,
                    "confidence": 0.9,
                    "drop_point": i,
                    "drop_percentage": drop_percentage
                }
        
        # Check for linear degradation
        if SKLEARN_AVAILABLE and len(values) >= 10:
            # Use linear regression to detect trend
            X = np.arange(len(values)).reshape(-1, 1)
            y = np.array(values)
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            r_squared = model.score(X, y)
            
            if slope < -0.5 and r_squared > 0.7:  # Significant negative trend
                return {
                    "type": DegradationPattern.LINEAR,
                    "confidence": r_squared,
                    "slope": slope,
                    "model": model
                }
            elif slope < -1.0 and r_squared > 0.6:  # Exponential-like trend
                return {
                    "type": DegradationPattern.EXPONENTIAL,
                    "confidence": r_squared,
                    "slope": slope
                }
        
        # Simple statistical check for degradation
        recent_avg = statistics.mean(values[-5:]) if len(values) >= 5 else values[-1]
        overall_avg = statistics.mean(values)
        
        if recent_avg < overall_avg * (1 - self.config["degradation_threshold"]):
            return {
                "type": DegradationPattern.LINEAR,
                "confidence": 0.6,
                "recent_avg": recent_avg,
                "overall_avg": overall_avg
            }
        
        return {"type": None, "confidence": 0.0}
    
    async def _handle_linear_degradation(self, resource_id: str, 
                                       time_series: List[Tuple[datetime, float]], 
                                       pattern: Dict[str, Any]) -> DegradationAlert:
        """Handle linear degradation pattern"""
        # Predict failure time based on linear trend
        predicted_failure_time = None
        confidence = pattern["confidence"]
        
        if "model" in pattern and confidence > self.config["confidence_threshold"]:
            # Use the linear model to predict when health score reaches critical level (20)
            model = pattern["model"]
            current_time_index = len(time_series) - 1
            
            # Find when score will reach 20
            critical_score = 20.0
            current_score = time_series[-1][1]
            
            if pattern["slope"] < 0:  # Degrading
                time_to_critical = (current_score - critical_score) / abs(pattern["slope"])
                predicted_failure_time = time_series[-1][0] + timedelta(hours=time_to_critical)
                
                # Ensure prediction is in the future
                if predicted_failure_time <= datetime.now():
                    predicted_failure_time = datetime.now() + timedelta(hours=1)  # At least 1 hour from now
        
        return DegradationAlert(
            alert_id=str(uuid.uuid4()),
            resource_id=resource_id,
            degradation_pattern=DegradationPattern.LINEAR,
            severity=HealthStatus.WARNING if confidence < 0.8 else HealthStatus.CRITICAL,
            detected_at=datetime.now(),
            predicted_failure_time=predicted_failure_time,
            confidence=confidence,
            description=f"Linear degradation detected with slope {pattern.get('slope', 'unknown')}",
            recommended_actions=[
                "Schedule preventive maintenance",
                "Investigate root cause of degradation",
                "Consider resource scaling or replacement"
            ]
        )
    
    async def _handle_exponential_degradation(self, resource_id: str, 
                                            time_series: List[Tuple[datetime, float]], 
                                            pattern: Dict[str, Any]) -> DegradationAlert:
        """Handle exponential degradation pattern"""
        return DegradationAlert(
            alert_id=str(uuid.uuid4()),
            resource_id=resource_id,
            degradation_pattern=DegradationPattern.EXPONENTIAL,
            severity=HealthStatus.CRITICAL,
            detected_at=datetime.now(),
            predicted_failure_time=datetime.now() + timedelta(hours=24),  # Urgent
            confidence=pattern["confidence"],
            description="Exponential degradation detected - rapid failure likely",
            recommended_actions=[
                "Immediate investigation required",
                "Prepare for emergency maintenance",
                "Consider immediate resource replacement"
            ]
        )
    
    async def _handle_sudden_degradation(self, resource_id: str, 
                                       time_series: List[Tuple[datetime, float]], 
                                       pattern: Dict[str, Any]) -> DegradationAlert:
        """Handle sudden degradation pattern"""
        return DegradationAlert(
            alert_id=str(uuid.uuid4()),
            resource_id=resource_id,
            degradation_pattern=DegradationPattern.SUDDEN,
            severity=HealthStatus.CRITICAL,
            detected_at=datetime.now(),
            predicted_failure_time=None,  # Already occurred
            confidence=pattern["confidence"],
            description=f"Sudden degradation detected: {pattern.get('drop_percentage', 0)*100:.1f}% drop",
            recommended_actions=[
                "Immediate investigation of recent changes",
                "Check for system failures or configuration changes",
                "Consider rollback if recent deployment caused issue"
            ]
        )

class MaintenanceScheduler:
    """Schedules optimal maintenance windows based on usage patterns and business impact"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.usage_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.business_calendars: Dict[str, List[datetime]] = {}
        self.maintenance_history: List[MaintenanceOutcome] = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for maintenance scheduler"""
        return {
            "preferred_maintenance_hours": [2, 3, 4, 5],  # 2-5 AM
            "avoid_business_hours": True,
            "min_maintenance_window_hours": 2,
            "max_maintenance_window_hours": 8,
            "advance_notice_days": 7,
            "emergency_threshold_hours": 4
        }
    
    async def schedule_maintenance(self, recommendation: MaintenanceRecommendation, 
                                 usage_pattern: Optional[Dict[str, Any]] = None) -> MaintenanceRecommendation:
        """Schedule optimal maintenance window for a recommendation"""
        logger.info(f"Scheduling maintenance for resource: {recommendation.resource_id}")
        
        # Analyze usage patterns
        if usage_pattern:
            optimal_windows = await self._find_optimal_windows(
                recommendation.resource_id, usage_pattern, recommendation.estimated_duration
            )
        else:
            optimal_windows = await self._get_default_windows(recommendation.estimated_duration)
        
        # Select best window based on priority and constraints
        selected_window = await self._select_best_window(
            optimal_windows, recommendation.priority, recommendation.maintenance_type
        )
        
        # Update recommendation with scheduled window
        recommendation.recommended_window = selected_window
        recommendation.confidence_score = await self._calculate_scheduling_confidence(
            recommendation, selected_window
        )
        
        logger.info(f"Maintenance scheduled for {recommendation.resource_id}: "
                   f"{selected_window[0]} - {selected_window[1]}")
        
        return recommendation
    
    async def _find_optimal_windows(self, resource_id: str, usage_pattern: Dict[str, Any], 
                                  duration: timedelta) -> List[Tuple[datetime, datetime]]:
        """Find optimal maintenance windows based on usage patterns"""
        windows = []
        
        # Get usage data
        hourly_usage = usage_pattern.get("hourly_usage", {})
        if not hourly_usage:
            return await self._get_default_windows(duration)
        
        # Find low-usage periods
        low_usage_hours = []
        avg_usage = statistics.mean(hourly_usage.values()) if hourly_usage else 50.0
        
        for hour, usage in hourly_usage.items():
            if usage < avg_usage * 0.3:  # Less than 30% of average usage
                low_usage_hours.append(int(hour))
        
        # Generate windows starting from low-usage hours
        now = datetime.now()
        for days_ahead in range(1, 15):  # Look up to 2 weeks ahead
            for start_hour in low_usage_hours:
                start_time = now.replace(
                    hour=start_hour, minute=0, second=0, microsecond=0
                ) + timedelta(days=days_ahead)
                end_time = start_time + duration
                
                # Check if window fits within low-usage period
                if self._is_valid_maintenance_window(start_time, end_time, hourly_usage):
                    windows.append((start_time, end_time))
        
        return windows[:10]  # Return top 10 windows
    
    async def _get_default_windows(self, duration: timedelta) -> List[Tuple[datetime, datetime]]:
        """Get default maintenance windows during preferred hours"""
        windows = []
        now = datetime.now()
        
        for days_ahead in range(1, 8):  # Next 7 days
            for start_hour in self.config["preferred_maintenance_hours"]:
                start_time = now.replace(
                    hour=start_hour, minute=0, second=0, microsecond=0
                ) + timedelta(days=days_ahead)
                end_time = start_time + duration
                
                # Skip weekends for non-emergency maintenance
                if start_time.weekday() < 5:  # Monday = 0, Friday = 4
                    windows.append((start_time, end_time))
        
        return windows
    
    def _is_valid_maintenance_window(self, start_time: datetime, end_time: datetime, 
                                   hourly_usage: Dict[str, float]) -> bool:
        """Check if a maintenance window is valid based on usage patterns"""
        # Check each hour in the window
        current_time = start_time
        while current_time < end_time:
            hour_key = str(current_time.hour)
            usage = hourly_usage.get(hour_key, 50.0)
            
            # If any hour has high usage, window is not valid
            if usage > 70.0:  # High usage threshold
                return False
            
            current_time += timedelta(hours=1)
        
        return True
    
    async def _select_best_window(self, windows: List[Tuple[datetime, datetime]], 
                                priority: MaintenancePriority, 
                                maintenance_type: MaintenanceType) -> Tuple[datetime, datetime]:
        """Select the best maintenance window from available options"""
        if not windows:
            # Emergency scheduling
            now = datetime.now()
            duration = timedelta(hours=2)  # Default duration
            return (now + timedelta(hours=1), now + timedelta(hours=1) + duration)
        
        # Score each window
        scored_windows = []
        for start_time, end_time in windows:
            score = await self._score_maintenance_window(
                start_time, end_time, priority, maintenance_type
            )
            scored_windows.append((score, start_time, end_time))
        
        # Sort by score (higher is better)
        scored_windows.sort(key=lambda x: x[0], reverse=True)
        
        return (scored_windows[0][1], scored_windows[0][2])
    
    async def _score_maintenance_window(self, start_time: datetime, end_time: datetime, 
                                      priority: MaintenancePriority, 
                                      maintenance_type: MaintenanceType) -> float:
        """Score a maintenance window based on various factors"""
        score = 0.0
        
        # Time-based scoring
        if start_time.hour in self.config["preferred_maintenance_hours"]:
            score += 30.0
        
        # Weekday preference (avoid weekends for routine maintenance)
        if maintenance_type != MaintenanceType.EMERGENCY:
            if start_time.weekday() < 5:  # Weekday
                score += 20.0
            else:
                score -= 10.0
        
        # Advance notice scoring
        days_ahead = (start_time - datetime.now()).days
        if days_ahead >= self.config["advance_notice_days"]:
            score += 15.0
        elif days_ahead >= 3:
            score += 10.0
        elif days_ahead >= 1:
            score += 5.0
        
        # Priority-based adjustments
        if priority == MaintenancePriority.CRITICAL:
            # Critical maintenance can happen anytime
            score += 25.0
        elif priority == MaintenancePriority.HIGH:
            score += 15.0
        
        # Duration preference (shorter windows are better for routine maintenance)
        duration_hours = (end_time - start_time).total_seconds() / 3600
        if duration_hours <= 4:
            score += 10.0
        elif duration_hours <= 8:
            score += 5.0
        
        return score
    
    async def _calculate_scheduling_confidence(self, recommendation: MaintenanceRecommendation, 
                                             window: Tuple[datetime, datetime]) -> float:
        """Calculate confidence score for the scheduled maintenance window"""
        confidence = 0.5  # Base confidence
        
        # Time-based confidence
        start_time = window[0]
        if start_time.hour in self.config["preferred_maintenance_hours"]:
            confidence += 0.2
        
        # Advance notice confidence
        days_ahead = (start_time - datetime.now()).days
        if days_ahead >= self.config["advance_notice_days"]:
            confidence += 0.2
        elif days_ahead >= 3:
            confidence += 0.1
        
        # Priority-based confidence
        if recommendation.priority == MaintenancePriority.CRITICAL:
            confidence += 0.1
        
        return min(confidence, 1.0)

class PredictiveMaintenanceSystem:
    """
    Main predictive maintenance system that coordinates all components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.health_analyzer = HealthMetricsAnalyzer(
            self.config.get("health_analyzer", {})
        )
        self.degradation_detector = DegradationDetector(
            self.config.get("degradation_detector", {})
        )
        self.maintenance_scheduler = MaintenanceScheduler(
            self.config.get("maintenance_scheduler", {})
        )
        
        # System state
        self.active_assessments: Dict[str, HealthAssessment] = {}
        self.active_alerts: List[DegradationAlert] = []
        self.maintenance_recommendations: List[MaintenanceRecommendation] = []
        self.maintenance_outcomes: List[MaintenanceOutcome] = []
        
        # Effectiveness tracking
        self.effectiveness_metrics: Dict[str, float] = {}
        self.prediction_accuracy: Dict[str, List[float]] = defaultdict(list)
        
        # Storage
        self.storage_path = Path("predictive_maintenance_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Predictive Maintenance System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            "assessment_interval_hours": 6,
            "alert_retention_days": 30,
            "recommendation_retention_days": 90,
            "effectiveness_tracking_enabled": True,
            "auto_scheduling_enabled": True,
            "notification_channels": ["email", "webhook"]
        }
    
    async def analyze_infrastructure_health(self, resources: Dict[str, List[HealthMetric]]) -> Dict[str, HealthAssessment]:
        """Analyze health of multiple infrastructure resources"""
        logger.info(f"Analyzing health for {len(resources)} resources")
        
        assessments = {}
        
        for resource_id, metrics in resources.items():
            try:
                assessment = await self.health_analyzer.analyze_resource_health(
                    resource_id, metrics
                )
                assessments[resource_id] = assessment
                self.active_assessments[resource_id] = assessment
                
                # Check for degradation
                await self._check_for_degradation(resource_id, assessment)
                
            except Exception as e:
                logger.error(f"Failed to analyze health for {resource_id}: {e}")
                continue
        
        logger.info(f"Completed health analysis for {len(assessments)} resources")
        return assessments
    
    async def _check_for_degradation(self, resource_id: str, current_assessment: HealthAssessment):
        """Check for degradation patterns and generate alerts"""
        # Get historical assessments for this resource
        health_history = await self._get_health_history(resource_id)
        health_history.append(current_assessment)
        
        # Detect degradation
        degradation_alert = await self.degradation_detector.detect_degradation(
            resource_id, health_history
        )
        
        if degradation_alert:
            self.active_alerts.append(degradation_alert)
            logger.warning(f"Degradation detected for {resource_id}: "
                          f"{degradation_alert.degradation_pattern.value}")
            
            # Generate maintenance recommendation
            await self._generate_maintenance_recommendation(degradation_alert)
    
    async def _get_health_history(self, resource_id: str) -> List[HealthAssessment]:
        """Get historical health assessments for a resource"""
        # In a real implementation, this would query a database
        # For now, return empty list (first assessment)
        return []
    
    async def _generate_maintenance_recommendation(self, alert: DegradationAlert):
        """Generate maintenance recommendation based on degradation alert"""
        # Determine maintenance type and priority
        if alert.severity == HealthStatus.CRITICAL:
            maintenance_type = MaintenanceType.EMERGENCY
            priority = MaintenancePriority.CRITICAL
            estimated_duration = timedelta(hours=2)
        elif alert.degradation_pattern == DegradationPattern.SUDDEN:
            maintenance_type = MaintenanceType.CORRECTIVE
            priority = MaintenancePriority.HIGH
            estimated_duration = timedelta(hours=4)
        else:
            maintenance_type = MaintenanceType.PREDICTIVE
            priority = MaintenancePriority.MEDIUM
            estimated_duration = timedelta(hours=6)
        
        # Create recommendation
        recommendation = MaintenanceRecommendation(
            recommendation_id=str(uuid.uuid4()),
            resource_id=alert.resource_id,
            maintenance_type=maintenance_type,
            priority=priority,
            recommended_window=(datetime.now(), datetime.now() + estimated_duration),
            estimated_duration=estimated_duration,
            description=f"Maintenance required due to {alert.degradation_pattern.value} degradation",
            expected_benefits=[
                "Prevent potential service disruption",
                "Restore optimal performance",
                "Reduce long-term maintenance costs"
            ],
            risks_if_delayed=[
                "Service degradation may continue",
                "Potential service outage",
                "Higher repair costs if failure occurs"
            ]
        )
        
        # Schedule maintenance if auto-scheduling is enabled
        if self.config["auto_scheduling_enabled"]:
            recommendation = await self.maintenance_scheduler.schedule_maintenance(
                recommendation
            )
        
        self.maintenance_recommendations.append(recommendation)
        
        logger.info(f"Generated maintenance recommendation for {alert.resource_id}: "
                   f"{maintenance_type.value} maintenance scheduled")
    
    async def track_maintenance_effectiveness(self, outcome: MaintenanceOutcome):
        """Track the effectiveness of maintenance actions"""
        if not self.config["effectiveness_tracking_enabled"]:
            return
        
        self.maintenance_outcomes.append(outcome)
        
        # Calculate effectiveness metrics
        effectiveness_score = await self._calculate_effectiveness_score(outcome)
        outcome.effectiveness_score = effectiveness_score
        
        # Update system-wide effectiveness metrics
        resource_type = self._get_resource_type(outcome.resource_id)
        if resource_type not in self.effectiveness_metrics:
            self.effectiveness_metrics[resource_type] = []
        
        self.effectiveness_metrics[resource_type] = effectiveness_score
        
        # Update prediction accuracy if applicable
        await self._update_prediction_accuracy(outcome)
        
        logger.info(f"Tracked maintenance effectiveness for {outcome.resource_id}: "
                   f"Score={effectiveness_score:.2f}")
    
    async def _calculate_effectiveness_score(self, outcome: MaintenanceOutcome) -> float:
        """Calculate effectiveness score for a maintenance outcome"""
        score = 0.0
        
        # Success factor (40% of score)
        if outcome.success:
            score += 40.0
        
        # Issues resolved factor (30% of score)
        if outcome.issues_resolved:
            score += 30.0 * min(len(outcome.issues_resolved) / 3, 1.0)
        
        # Duration efficiency factor (20% of score)
        # Compare actual vs estimated duration
        original_recommendation = next(
            (r for r in self.maintenance_recommendations 
             if r.recommendation_id == outcome.recommendation_id), None
        )
        
        if original_recommendation:
            estimated_hours = original_recommendation.estimated_duration.total_seconds() / 3600
            actual_hours = outcome.duration.total_seconds() / 3600
            
            if actual_hours <= estimated_hours:
                score += 20.0
            elif actual_hours <= estimated_hours * 1.5:
                score += 10.0
        
        # Cost efficiency factor (10% of score)
        if original_recommendation and original_recommendation.cost_estimate and outcome.cost_actual:
            if outcome.cost_actual <= original_recommendation.cost_estimate:
                score += 10.0
            elif outcome.cost_actual <= original_recommendation.cost_estimate * 1.2:
                score += 5.0
        
        return min(score, 100.0)
    
    async def _update_prediction_accuracy(self, outcome: MaintenanceOutcome):
        """Update prediction accuracy metrics"""
        # Find related degradation alert
        related_alert = next(
            (alert for alert in self.active_alerts 
             if alert.resource_id == outcome.resource_id), None
        )
        
        if related_alert and related_alert.predicted_failure_time:
            # Calculate prediction accuracy
            actual_maintenance_time = outcome.executed_at
            predicted_failure_time = related_alert.predicted_failure_time
            
            # If maintenance was successful and done before predicted failure, it's accurate
            if outcome.success and actual_maintenance_time < predicted_failure_time:
                accuracy = 1.0
            else:
                # Calculate time-based accuracy
                time_diff = abs((predicted_failure_time - actual_maintenance_time).total_seconds())
                max_acceptable_diff = 24 * 3600  # 24 hours
                accuracy = max(0.0, 1.0 - (time_diff / max_acceptable_diff))
            
            resource_type = self._get_resource_type(outcome.resource_id)
            self.prediction_accuracy[resource_type].append(accuracy)
    
    def _get_resource_type(self, resource_id: str) -> str:
        """Get resource type from resource ID"""
        # Simple heuristic - in real implementation, this would query metadata
        if "ec2" in resource_id.lower():
            return "compute"
        elif "rds" in resource_id.lower():
            return "database"
        elif "s3" in resource_id.lower():
            return "storage"
        else:
            return "unknown"
    
    async def get_system_effectiveness_report(self) -> Dict[str, Any]:
        """Generate system effectiveness report"""
        report = {
            "overall_effectiveness": 0.0,
            "resource_type_effectiveness": {},
            "prediction_accuracy": {},
            "maintenance_statistics": {},
            "recommendations": []
        }
        
        # Calculate overall effectiveness
        if self.maintenance_outcomes:
            overall_effectiveness = statistics.mean([
                outcome.effectiveness_score for outcome in self.maintenance_outcomes
                if outcome.effectiveness_score > 0
            ])
            report["overall_effectiveness"] = overall_effectiveness
        
        # Resource type effectiveness
        for resource_type, scores in self.effectiveness_metrics.items():
            if isinstance(scores, list) and scores:
                report["resource_type_effectiveness"][resource_type] = statistics.mean(scores)
            elif isinstance(scores, (int, float)):
                report["resource_type_effectiveness"][resource_type] = scores
        
        # Prediction accuracy
        for resource_type, accuracies in self.prediction_accuracy.items():
            if accuracies:
                report["prediction_accuracy"][resource_type] = {
                    "mean_accuracy": statistics.mean(accuracies),
                    "sample_count": len(accuracies)
                }
        
        # Maintenance statistics
        if self.maintenance_outcomes:
            successful_maintenance = sum(1 for outcome in self.maintenance_outcomes if outcome.success)
            report["maintenance_statistics"] = {
                "total_maintenance_actions": len(self.maintenance_outcomes),
                "successful_actions": successful_maintenance,
                "success_rate": successful_maintenance / len(self.maintenance_outcomes),
                "average_effectiveness": statistics.mean([
                    outcome.effectiveness_score for outcome in self.maintenance_outcomes
                    if outcome.effectiveness_score > 0
                ])
            }
        
        # Generate improvement recommendations
        report["recommendations"] = await self._generate_system_improvement_recommendations(report)
        
        return report
    
    async def _generate_system_improvement_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations for system improvement"""
        recommendations = []
        
        # Check overall effectiveness
        overall_effectiveness = report.get("overall_effectiveness", 0)
        if overall_effectiveness < 70:
            recommendations.append("Overall system effectiveness is below target - review maintenance procedures")
        
        # Check prediction accuracy
        prediction_accuracy = report.get("prediction_accuracy", {})
        for resource_type, accuracy_data in prediction_accuracy.items():
            if accuracy_data["mean_accuracy"] < 0.7:
                recommendations.append(f"Improve prediction accuracy for {resource_type} resources")
        
        # Check success rate
        maintenance_stats = report.get("maintenance_statistics", {})
        success_rate = maintenance_stats.get("success_rate", 1.0)
        if success_rate < 0.8:
            recommendations.append("Maintenance success rate is low - review procedures and training")
        
        if not recommendations:
            recommendations.append("System is performing well - continue current practices")
        
        return recommendations