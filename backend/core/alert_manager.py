# alert_manager.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import statistics

from .enums import MetricType, SeverityLevel
from .performance_monitor import MetricsData
from .anomaly_detector import Anomaly


@dataclass
class AlertRule:
    """Defines conditions for triggering alerts"""
    rule_id: str
    name: str
    metric_type: MetricType
    condition: str  # e.g., "greater_than", "less_than", "anomaly_detected"
    threshold: Optional[float] = None
    severity: SeverityLevel = SeverityLevel.MEDIUM
    enabled: bool = True
    resource_filters: Dict[str, List[str]] = field(default_factory=dict)  # e.g., {"tags": ["production"]}
    notification_channels: List[str] = field(default_factory=list)
    cooldown_minutes: int = 15  # Minimum time between alerts for same condition
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Represents a triggered alert"""
    alert_id: str
    rule_id: str
    resource_id: str
    metric_type: MetricType
    severity: SeverityLevel
    message: str
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    status: str = "active"  # active, acknowledged, resolved
    context: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AlertManager")
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, Any] = {}
        self.cooldown_tracker: Dict[str, datetime] = {}
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
    
    def register_notification_channel(self, channel_id: str, channel_config: Dict[str, Any]):
        """Register a notification channel (email, Slack, webhook, etc.)"""
        self.notification_channels[channel_id] = channel_config
        self.logger.info(f"Registered notification channel: {channel_id}")
    
    def evaluate_alerts(self, metrics_data: Dict[str, MetricsData], 
                       anomalies: List[Anomaly]) -> List[Alert]:
        """Evaluate alert rules against current metrics and anomalies"""
        triggered_alerts = []
        
        # Process anomaly-based alerts
        for anomaly in anomalies:
            anomaly_alerts = self._evaluate_anomaly_alerts(anomaly)
            triggered_alerts.extend(anomaly_alerts)
        
        # Process threshold-based alerts
        for resource_id, data in metrics_data.items():
            threshold_alerts = self._evaluate_threshold_alerts(resource_id, data)
            triggered_alerts.extend(threshold_alerts)
        
        # Update active alerts and send notifications
        for alert in triggered_alerts:
            self._process_alert(alert)
        
        self.logger.info(f"Evaluated alerts: {len(triggered_alerts)} new alerts triggered")
        return triggered_alerts
    
    def _evaluate_anomaly_alerts(self, anomaly: Anomaly) -> List[Alert]:
        """Evaluate alert rules for anomalies"""
        alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this anomaly
            if rule.condition != "anomaly_detected":
                continue
            
            if rule.metric_type != anomaly.metric_type:
                continue
            
            # Check resource filters
            if not self._resource_matches_filters(anomaly.resource_id, rule.resource_filters):
                continue
            
            # Check cooldown
            cooldown_key = f"{rule_id}_{anomaly.resource_id}_{anomaly.metric_type.value}"
            if self._is_in_cooldown(cooldown_key, rule.cooldown_minutes):
                continue
            
            # Create alert
            alert = Alert(
                alert_id=f"alert_{rule_id}_{anomaly.resource_id}_{datetime.now().timestamp()}",
                rule_id=rule_id,
                resource_id=anomaly.resource_id,
                metric_type=anomaly.metric_type,
                severity=max(rule.severity, anomaly.severity, key=lambda x: x.value),
                message=f"Anomaly detected: {anomaly.description}",
                current_value=anomaly.current_value,
                threshold_value=anomaly.expected_value,
                context={
                    'anomaly_id': anomaly.anomaly_id,
                    'anomaly_score': anomaly.anomaly_score,
                    'suggested_actions': anomaly.suggested_actions
                }
            )
            
            alerts.append(alert)
            
            # Update cooldown tracker
            self.cooldown_tracker[cooldown_key] = datetime.now()
        
        return alerts
    
    def _evaluate_threshold_alerts(self, resource_id: str, data: MetricsData) -> List[Alert]:
        """Evaluate threshold-based alert rules"""
        alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            if rule.condition == "anomaly_detected":
                continue  # Skip anomaly rules
            
            # Check resource filters
            if not self._resource_matches_filters(resource_id, rule.resource_filters):
                continue
            
            # Get metric data for this rule
            if rule.metric_type not in data.metrics:
                continue
            
            time_series = data.metrics[rule.metric_type]
            if not time_series:
                continue
            
            # Get latest value
            latest_timestamp, latest_value = time_series[-1]
            
            # Check threshold condition
            threshold_exceeded = False
            if rule.condition == "greater_than" and rule.threshold is not None:
                threshold_exceeded = latest_value > rule.threshold
            elif rule.condition == "less_than" and rule.threshold is not None:
                threshold_exceeded = latest_value < rule.threshold
            elif rule.condition == "equals" and rule.threshold is not None:
                threshold_exceeded = abs(latest_value - rule.threshold) < 0.01
            
            if not threshold_exceeded:
                continue
            
            # Check cooldown
            cooldown_key = f"{rule_id}_{resource_id}_{rule.metric_type.value}"
            if self._is_in_cooldown(cooldown_key, rule.cooldown_minutes):
                continue
            
            # Create alert
            alert = Alert(
                alert_id=f"alert_{rule_id}_{resource_id}_{datetime.now().timestamp()}",
                rule_id=rule_id,
                resource_id=resource_id,
                metric_type=rule.metric_type,
                severity=rule.severity,
                message=f"{rule.metric_type.value} {rule.condition} {rule.threshold}: current value {latest_value}",
                current_value=latest_value,
                threshold_value=rule.threshold,
                context={'rule_name': rule.name}
            )
            
            alerts.append(alert)
            
            # Update cooldown tracker
            self.cooldown_tracker[cooldown_key] = datetime.now()
        
        return alerts
    
    def _resource_matches_filters(self, resource_id: str, filters: Dict[str, List[str]]) -> bool:
        """Check if resource matches the filter criteria"""
        if not filters:
            return True  # No filters means match all
        
        # In a real implementation, this would check resource tags, types, etc.
        # For now, we'll assume all resources match
        return True
    
    def _is_in_cooldown(self, cooldown_key: str, cooldown_minutes: int) -> bool:
        """Check if alert is in cooldown period"""
        if cooldown_key not in self.cooldown_tracker:
            return False
        
        last_alert_time = self.cooldown_tracker[cooldown_key]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - last_alert_time < cooldown_period
    
    def _process_alert(self, alert: Alert):
        """Process a triggered alert"""
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send notifications
        rule = self.alert_rules.get(alert.rule_id)
        if rule and rule.notification_channels:
            self._send_notifications(alert, rule.notification_channels)
        
        self.logger.info(f"Processed alert: {alert.alert_id} - {alert.message}")
    
    def _send_notifications(self, alert: Alert, channels: List[str]):
        """Send alert notifications to configured channels"""
        for channel_id in channels:
            if channel_id in self.notification_channels:
                try:
                    channel_config = self.notification_channels[channel_id]
                    self._send_notification(alert, channel_config)
                except Exception as e:
                    self.logger.error(f"Error sending notification to {channel_id}: {e}")
    
    def _send_notification(self, alert: Alert, channel_config: Dict[str, Any]):
        """Send notification to a specific channel"""
        channel_type = channel_config.get('type', 'unknown')
        
        if channel_type == 'email':
            self._send_email_notification(alert, channel_config)
        elif channel_type == 'slack':
            self._send_slack_notification(alert, channel_config)
        elif channel_type == 'webhook':
            self._send_webhook_notification(alert, channel_config)
        else:
            self.logger.warning(f"Unknown notification channel type: {channel_type}")
    
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification (placeholder implementation)"""
        # In a real implementation, this would use SMTP or email service API
        self.logger.info(f"Email notification sent for alert {alert.alert_id}")
    
    def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack notification (placeholder implementation)"""
        # In a real implementation, this would use Slack API
        self.logger.info(f"Slack notification sent for alert {alert.alert_id}")
    
    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification (placeholder implementation)"""
        # In a real implementation, this would make HTTP POST request
        self.logger.info(f"Webhook notification sent for alert {alert.alert_id}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_at = datetime.now()
            alert.status = "acknowledged"
            alert.context['acknowledged_by'] = acknowledged_by
            
            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_note: str = "") -> bool:
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.now()
            alert.status = "resolved"
            alert.context['resolved_by'] = resolved_by
            alert.context['resolution_note'] = resolution_note
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
            return True
        
        return False
    
    def get_active_alerts(self, severity_filter: Optional[SeverityLevel] = None) -> List[Alert]:
        """Get list of active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        # Sort by severity and timestamp
        severity_order = {
            SeverityLevel.CRITICAL: 4,
            SeverityLevel.HIGH: 3,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 1
        }
        
        alerts.sort(key=lambda x: (severity_order.get(x.severity, 0), x.triggered_at), reverse=True)
        
        return alerts
    
    def get_alert_statistics(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
        recent_alerts = [
            a for a in self.alert_history 
            if a.triggered_at >= cutoff_time
        ]
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_metric_type': {},
                'by_resource': {},
                'resolution_stats': {}
            }
        
        # Count by severity
        by_severity = {}
        for severity in SeverityLevel:
            count = len([a for a in recent_alerts if a.severity == severity])
            if count > 0:
                by_severity[severity.value] = count
        
        # Count by metric type
        by_metric_type = {}
        for metric_type in MetricType:
            count = len([a for a in recent_alerts if a.metric_type == metric_type])
            if count > 0:
                by_metric_type[metric_type.value] = count
        
        # Count by resource
        by_resource = {}
        for alert in recent_alerts:
            if alert.resource_id not in by_resource:
                by_resource[alert.resource_id] = 0
            by_resource[alert.resource_id] += 1
        
        # Resolution statistics
        resolved_alerts = [a for a in recent_alerts if a.status == "resolved"]
        acknowledged_alerts = [a for a in recent_alerts if a.status == "acknowledged"]
        
        resolution_stats = {
            'total_resolved': len(resolved_alerts),
            'total_acknowledged': len(acknowledged_alerts),
            'resolution_rate': len(resolved_alerts) / len(recent_alerts) * 100 if recent_alerts else 0
        }
        
        # Calculate average resolution time
        if resolved_alerts:
            resolution_times = []
            for alert in resolved_alerts:
                if alert.resolved_at and alert.triggered_at:
                    resolution_time = (alert.resolved_at - alert.triggered_at).total_seconds() / 60  # minutes
                    resolution_times.append(resolution_time)
            
            if resolution_times:
                resolution_stats['avg_resolution_time_minutes'] = statistics.mean(resolution_times)
        
        return {
            'total_alerts': len(recent_alerts),
            'by_severity': by_severity,
            'by_metric_type': by_metric_type,
            'by_resource': by_resource,
            'resolution_stats': resolution_stats,
            'time_period_hours': time_period_hours
        }
    
    def create_default_alert_rules(self) -> List[AlertRule]:
        """Create a set of default alert rules for common scenarios"""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_utilization",
                name="High CPU Utilization",
                metric_type=MetricType.CPU_UTILIZATION,
                condition="greater_than",
                threshold=85.0,
                severity=SeverityLevel.HIGH,
                cooldown_minutes=10
            ),
            AlertRule(
                rule_id="high_memory_utilization",
                name="High Memory Utilization",
                metric_type=MetricType.MEMORY_UTILIZATION,
                condition="greater_than",
                threshold=90.0,
                severity=SeverityLevel.HIGH,
                cooldown_minutes=10
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                metric_type=MetricType.ERROR_RATE,
                condition="greater_than",
                threshold=5.0,
                severity=SeverityLevel.CRITICAL,
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="low_availability",
                name="Low Availability",
                metric_type=MetricType.AVAILABILITY,
                condition="less_than",
                threshold=99.0,
                severity=SeverityLevel.CRITICAL,
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="high_response_time",
                name="High Response Time",
                metric_type=MetricType.RESPONSE_TIME,
                condition="greater_than",
                threshold=1000.0,  # 1 second
                severity=SeverityLevel.MEDIUM,
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="cpu_anomaly_detection",
                name="CPU Anomaly Detection",
                metric_type=MetricType.CPU_UTILIZATION,
                condition="anomaly_detected",
                severity=SeverityLevel.MEDIUM,
                cooldown_minutes=20
            ),
            AlertRule(
                rule_id="memory_anomaly_detection",
                name="Memory Anomaly Detection",
                metric_type=MetricType.MEMORY_UTILIZATION,
                condition="anomaly_detected",
                severity=SeverityLevel.MEDIUM,
                cooldown_minutes=20
            )
        ]
        
        # Add all default rules
        for rule in default_rules:
            self.add_alert_rule(rule)
        
        self.logger.info(f"Created {len(default_rules)} default alert rules")
        return default_rules