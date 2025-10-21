# anomaly_detector.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import statistics

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from performance_monitor import MetricType, MetricsData
from enhanced_models import SeverityLevel


@dataclass
class Anomaly:
    """Represents a detected performance anomaly"""
    anomaly_id: str
    resource_id: str
    metric_type: MetricType
    severity: SeverityLevel
    detected_at: datetime
    anomaly_score: float  # 0-1, higher means more anomalous
    current_value: float
    expected_value: float
    description: str
    suggested_actions: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """ML-based anomaly detection for performance metrics"""
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity  # Lower values = more sensitive
        self.logger = logging.getLogger(f"{__name__}.AnomalyDetector")
        self.models: Dict[str, Any] = {}  # Trained models per resource/metric
        self.baseline_data: Dict[str, Dict[MetricType, List[float]]] = {}
        self.detection_history: List[Anomaly] = []
    
    def train_baseline(self, metrics_data: Dict[str, MetricsData], 
                      training_period_days: int = 14):
        """Train baseline models using historical data"""
        self.logger.info(f"Training anomaly detection baselines for {len(metrics_data)} resources")
        
        for resource_id, data in metrics_data.items():
            if resource_id not in self.baseline_data:
                self.baseline_data[resource_id] = {}
            
            for metric_type, time_series in data.metrics.items():
                if not time_series:
                    continue
                
                # Extract values for baseline calculation
                values = [value for _, value in time_series]
                
                if len(values) >= 10:  # Minimum data points for baseline
                    self.baseline_data[resource_id][metric_type] = values
                    
                    # Train ML model if sklearn is available
                    if SKLEARN_AVAILABLE and len(values) >= 50:
                        self._train_isolation_forest(resource_id, metric_type, values)
                    else:
                        self._calculate_statistical_baseline(resource_id, metric_type, values)
        
        self.logger.info(f"Trained baselines for {len(self.baseline_data)} resources")
    
    def _train_isolation_forest(self, resource_id: str, metric_type: MetricType, values: List[float]):
        """Train Isolation Forest model for anomaly detection"""
        try:
            # Prepare data for training
            X = np.array(values).reshape(-1, 1)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=self.sensitivity,
                random_state=42,
                n_estimators=100
            )
            model.fit(X)
            
            # Store model
            model_key = f"{resource_id}_{metric_type.value}"
            self.models[model_key] = {
                'type': 'isolation_forest',
                'model': model,
                'scaler': StandardScaler().fit(X),
                'baseline_mean': np.mean(values),
                'baseline_std': np.std(values)
            }
            
            self.logger.debug(f"Trained Isolation Forest for {resource_id}:{metric_type.value}")
            
        except Exception as e:
            self.logger.error(f"Error training Isolation Forest: {e}")
            # Fallback to statistical method
            self._calculate_statistical_baseline(resource_id, metric_type, values)
    
    def _calculate_statistical_baseline(self, resource_id: str, metric_type: MetricType, values: List[float]):
        """Calculate statistical baseline for anomaly detection"""
        if len(values) < 2:
            return
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        median_val = statistics.median(values)
        
        # Calculate percentiles for threshold-based detection
        sorted_values = sorted(values)
        p95 = sorted_values[int(0.95 * len(sorted_values))]
        p5 = sorted_values[int(0.05 * len(sorted_values))]
        
        model_key = f"{resource_id}_{metric_type.value}"
        self.models[model_key] = {
            'type': 'statistical',
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'p95': p95,
            'p5': p5,
            'upper_threshold': mean_val + (3 * std_val),  # 3-sigma rule
            'lower_threshold': max(0, mean_val - (3 * std_val))
        }
        
        self.logger.debug(f"Calculated statistical baseline for {resource_id}:{metric_type.value}")
    
    def detect_anomalies(self, metrics_data: Dict[str, MetricsData]) -> List[Anomaly]:
        """Detect anomalies in current metrics data"""
        anomalies = []
        
        for resource_id, data in metrics_data.items():
            for metric_type, time_series in data.metrics.items():
                if not time_series:
                    continue
                
                # Get the latest values for anomaly detection
                recent_values = [value for _, value in time_series[-10:]]  # Last 10 data points
                
                if not recent_values:
                    continue
                
                resource_anomalies = self._detect_metric_anomalies(
                    resource_id, metric_type, recent_values, time_series[-1][0]
                )
                anomalies.extend(resource_anomalies)
        
        # Store detection history
        self.detection_history.extend(anomalies)
        
        # Keep only recent history (last 1000 anomalies)
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        self.logger.info(f"Detected {len(anomalies)} anomalies across all resources")
        return anomalies
    
    def _detect_metric_anomalies(self, resource_id: str, metric_type: MetricType, 
                                values: List[float], timestamp: datetime) -> List[Anomaly]:
        """Detect anomalies for a specific metric"""
        anomalies = []
        model_key = f"{resource_id}_{metric_type.value}"
        
        if model_key not in self.models:
            return anomalies
        
        model_info = self.models[model_key]
        
        for i, value in enumerate(values):
            anomaly_score = 0.0
            is_anomaly = False
            expected_value = value
            
            if model_info['type'] == 'isolation_forest' and SKLEARN_AVAILABLE:
                # Use ML model for detection
                try:
                    X = np.array([[value]])
                    X_scaled = model_info['scaler'].transform(X)
                    
                    # Get anomaly score (-1 for anomaly, 1 for normal)
                    prediction = model_info['model'].predict(X_scaled)[0]
                    anomaly_score_raw = model_info['model'].decision_function(X_scaled)[0]
                    
                    # Convert to 0-1 scale (higher = more anomalous)
                    anomaly_score = max(0, min(1, (0.5 - anomaly_score_raw) * 2))
                    
                    is_anomaly = prediction == -1
                    expected_value = model_info['baseline_mean']
                    
                except Exception as e:
                    self.logger.error(f"Error in ML anomaly detection: {e}")
                    continue
            
            elif model_info['type'] == 'statistical':
                # Use statistical thresholds
                mean_val = model_info['mean']
                std_val = model_info['std']
                upper_threshold = model_info['upper_threshold']
                lower_threshold = model_info['lower_threshold']
                
                expected_value = mean_val
                
                # Calculate how many standard deviations away from mean
                if std_val > 0:
                    z_score = abs(value - mean_val) / std_val
                    anomaly_score = min(1.0, z_score / 3.0)  # Normalize to 0-1
                    
                    # Consider anomaly if beyond thresholds
                    is_anomaly = value > upper_threshold or value < lower_threshold
                else:
                    # No variation in baseline data
                    is_anomaly = abs(value - mean_val) > (mean_val * 0.1)  # 10% deviation
                    anomaly_score = 0.5 if is_anomaly else 0.0
            
            # Create anomaly record if detected
            if is_anomaly and anomaly_score > 0.3:  # Minimum threshold for reporting
                severity = self._calculate_severity(anomaly_score, metric_type)
                
                anomaly = Anomaly(
                    anomaly_id=f"anomaly_{resource_id}_{metric_type.value}_{timestamp.timestamp()}",
                    resource_id=resource_id,
                    metric_type=metric_type,
                    severity=severity,
                    detected_at=timestamp,
                    anomaly_score=anomaly_score,
                    current_value=value,
                    expected_value=expected_value,
                    description=self._generate_anomaly_description(metric_type, value, expected_value),
                    suggested_actions=self._generate_suggested_actions(metric_type, value, expected_value),
                    context={
                        'detection_method': model_info['type'],
                        'baseline_period': '14_days',
                        'confidence': anomaly_score
                    }
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_severity(self, anomaly_score: float, metric_type: MetricType) -> SeverityLevel:
        """Calculate severity level based on anomaly score and metric type"""
        # Critical metrics have lower thresholds for high severity
        critical_metrics = [
            MetricType.CPU_UTILIZATION, 
            MetricType.MEMORY_UTILIZATION,
            MetricType.ERROR_RATE,
            MetricType.AVAILABILITY
        ]
        
        if metric_type in critical_metrics:
            if anomaly_score >= 0.8:
                return SeverityLevel.CRITICAL
            elif anomaly_score >= 0.6:
                return SeverityLevel.HIGH
            elif anomaly_score >= 0.4:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.LOW
        else:
            # Less critical metrics
            if anomaly_score >= 0.9:
                return SeverityLevel.CRITICAL
            elif anomaly_score >= 0.7:
                return SeverityLevel.HIGH
            elif anomaly_score >= 0.5:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.LOW
    
    def _generate_anomaly_description(self, metric_type: MetricType, 
                                    current_value: float, expected_value: float) -> str:
        """Generate human-readable anomaly description"""
        deviation = abs(current_value - expected_value)
        percentage_change = (deviation / expected_value * 100) if expected_value > 0 else 0
        
        direction = "higher" if current_value > expected_value else "lower"
        
        descriptions = {
            MetricType.CPU_UTILIZATION: f"CPU utilization is {percentage_change:.1f}% {direction} than expected ({current_value:.1f}% vs {expected_value:.1f}%)",
            MetricType.MEMORY_UTILIZATION: f"Memory utilization is {percentage_change:.1f}% {direction} than expected ({current_value:.1f}% vs {expected_value:.1f}%)",
            MetricType.RESPONSE_TIME: f"Response time is {percentage_change:.1f}% {direction} than expected ({current_value:.1f}ms vs {expected_value:.1f}ms)",
            MetricType.THROUGHPUT: f"Throughput is {percentage_change:.1f}% {direction} than expected ({current_value:.1f} vs {expected_value:.1f})",
            MetricType.ERROR_RATE: f"Error rate is {percentage_change:.1f}% {direction} than expected ({current_value:.2f}% vs {expected_value:.2f}%)",
            MetricType.AVAILABILITY: f"Availability is {percentage_change:.1f}% {direction} than expected ({current_value:.2f}% vs {expected_value:.2f}%)"
        }
        
        return descriptions.get(metric_type, f"{metric_type.value} is {percentage_change:.1f}% {direction} than expected")
    
    def _generate_suggested_actions(self, metric_type: MetricType, 
                                  current_value: float, expected_value: float) -> List[str]:
        """Generate suggested remediation actions"""
        actions = []
        
        if current_value > expected_value:
            # Value is higher than expected
            if metric_type == MetricType.CPU_UTILIZATION:
                actions = [
                    "Check for runaway processes or high CPU workloads",
                    "Consider scaling up the instance or adding more instances",
                    "Review recent deployments for performance regressions",
                    "Monitor for CPU-intensive applications"
                ]
            elif metric_type == MetricType.MEMORY_UTILIZATION:
                actions = [
                    "Check for memory leaks in applications",
                    "Consider increasing instance memory or scaling horizontally",
                    "Review memory-intensive processes",
                    "Check for memory cache issues"
                ]
            elif metric_type == MetricType.RESPONSE_TIME:
                actions = [
                    "Check database query performance",
                    "Review application performance and bottlenecks",
                    "Consider adding caching layers",
                    "Check network latency and connectivity"
                ]
            elif metric_type == MetricType.ERROR_RATE:
                actions = [
                    "Check application logs for error patterns",
                    "Review recent deployments for bugs",
                    "Monitor external service dependencies",
                    "Check system resource availability"
                ]
        else:
            # Value is lower than expected
            if metric_type == MetricType.THROUGHPUT:
                actions = [
                    "Check for reduced traffic or load",
                    "Verify upstream services are functioning",
                    "Review load balancer configuration",
                    "Check for network connectivity issues"
                ]
            elif metric_type == MetricType.AVAILABILITY:
                actions = [
                    "Check system health and uptime",
                    "Review monitoring and health check configurations",
                    "Verify service dependencies",
                    "Check for infrastructure issues"
                ]
        
        if not actions:
            actions = [
                f"Investigate {metric_type.value} anomaly",
                "Review system logs and metrics",
                "Check for recent configuration changes",
                "Monitor trend continuation"
            ]
        
        return actions
    
    def get_anomaly_summary(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """Get summary of anomalies detected in the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
        recent_anomalies = [
            a for a in self.detection_history 
            if a.detected_at >= cutoff_time
        ]
        
        if not recent_anomalies:
            return {
                'total_anomalies': 0,
                'by_severity': {},
                'by_metric_type': {},
                'by_resource': {},
                'most_affected_resources': []
            }
        
        # Count by severity
        by_severity = {}
        for severity in SeverityLevel:
            count = len([a for a in recent_anomalies if a.severity == severity])
            if count > 0:
                by_severity[severity.value] = count
        
        # Count by metric type
        by_metric_type = {}
        for metric_type in MetricType:
            count = len([a for a in recent_anomalies if a.metric_type == metric_type])
            if count > 0:
                by_metric_type[metric_type.value] = count
        
        # Count by resource
        by_resource = {}
        for anomaly in recent_anomalies:
            if anomaly.resource_id not in by_resource:
                by_resource[anomaly.resource_id] = 0
            by_resource[anomaly.resource_id] += 1
        
        # Most affected resources
        most_affected = sorted(
            by_resource.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'total_anomalies': len(recent_anomalies),
            'by_severity': by_severity,
            'by_metric_type': by_metric_type,
            'by_resource': by_resource,
            'most_affected_resources': most_affected,
            'time_period_hours': time_period_hours
        }