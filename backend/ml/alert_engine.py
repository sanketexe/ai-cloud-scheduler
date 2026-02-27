"""
Intelligent Alerting and Notification System

Provides multi-channel alert delivery with intelligent grouping, consolidation,
escalation rules, and alert management capabilities for anomaly detection.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
import hashlib
from pathlib import Path
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
try:
    import aiohttp
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
import uuid

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status states"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    SNOOZED = "snoozed"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AlertChannel(Enum):
    """Supported alert channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"

class EscalationLevel(Enum):
    """Escalation levels"""
    LEVEL_1 = 1  # Initial alert
    LEVEL_2 = 2  # First escalation
    LEVEL_3 = 3  # Second escalation
    LEVEL_4 = 4  # Final escalation

@dataclass
class AlertContext:
    """Context information for alerts"""
    account_id: str
    service: str
    resource_id: Optional[str] = None
    region: Optional[str] = None
    cost_amount: Optional[float] = None
    baseline_amount: Optional[float] = None
    deviation_percentage: Optional[float] = None
    confidence_score: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Core alert object"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    context: AlertContext
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    # Status management
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    snoozed_until: Optional[datetime] = None
    
    # Grouping and correlation
    group_id: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_alert_id: Optional[str] = None
    related_alert_ids: List[str] = field(default_factory=list)
    
    # Escalation
    escalation_level: EscalationLevel = EscalationLevel.LEVEL_1
    escalated_at: Optional[datetime] = None
    escalation_count: int = 0
    
    # Delivery tracking
    delivery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    successful_deliveries: List[str] = field(default_factory=list)
    failed_deliveries: List[str] = field(default_factory=list)
    
    def get_fingerprint(self) -> str:
        """Generate unique fingerprint for alert grouping"""
        fingerprint_data = f"{self.context.account_id}:{self.context.service}:{self.context.resource_id}:{self.title}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()
    
    def is_similar_to(self, other: 'Alert') -> bool:
        """Check if this alert is similar to another for grouping"""
        return (
            self.context.account_id == other.context.account_id and
            self.context.service == other.context.service and
            self.severity == other.severity and
            abs((self.created_at - other.created_at).total_seconds()) < 3600  # Within 1 hour
        )
    
    def should_escalate(self, escalation_timeout_minutes: int = 30) -> bool:
        """Check if alert should be escalated"""
        if self.status != AlertStatus.ACTIVE:
            return False
        
        if self.escalation_level == EscalationLevel.LEVEL_4:
            return False  # Already at max escalation
        
        time_since_created = (datetime.now() - self.created_at).total_seconds() / 60
        time_since_escalated = (datetime.now() - (self.escalated_at or self.created_at)).total_seconds() / 60
        
        return time_since_escalated >= escalation_timeout_minutes
    
    def update_status(self, status: AlertStatus, user: Optional[str] = None):
        """Update alert status with tracking"""
        self.status = status
        self.updated_at = datetime.now()
        
        if status == AlertStatus.ACKNOWLEDGED:
            self.acknowledged_by = user
            self.acknowledged_at = datetime.now()
        elif status == AlertStatus.RESOLVED:
            self.resolved_at = datetime.now()

@dataclass
class AlertGroup:
    """Group of related alerts"""
    group_id: str
    title: str
    alerts: List[Alert] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.ACTIVE
    
    def add_alert(self, alert: Alert):
        """Add alert to group"""
        alert.group_id = self.group_id
        self.alerts.append(alert)
        self.updated_at = datetime.now()
        
        # Update group status based on alerts
        if any(a.status == AlertStatus.ACTIVE for a in self.alerts):
            self.status = AlertStatus.ACTIVE
        elif all(a.status == AlertStatus.RESOLVED for a in self.alerts):
            self.status = AlertStatus.RESOLVED
    
    def get_highest_severity(self) -> AlertSeverity:
        """Get highest severity among grouped alerts"""
        severities = [alert.severity for alert in self.alerts]
        severity_order = [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM, AlertSeverity.LOW]
        
        for severity in severity_order:
            if severity in severities:
                return severity
        
        return AlertSeverity.LOW
    
    def get_summary(self) -> str:
        """Get summary of grouped alerts"""
        if len(self.alerts) == 1:
            return self.alerts[0].title
        
        service_counts = {}
        for alert in self.alerts:
            service = alert.context.service
            service_counts[service] = service_counts.get(service, 0) + 1
        
        services_summary = ", ".join([f"{count} {service}" for service, count in service_counts.items()])
        return f"{len(self.alerts)} alerts: {services_summary}"

@dataclass
class NotificationChannel:
    """Configuration for notification channels"""
    channel_type: AlertChannel
    name: str
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=lambda: list(AlertSeverity))
    rate_limit_per_hour: Optional[int] = None
    
    def should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent through this channel"""
        if not self.enabled:
            return False
        
        if self.severity_filter and alert.severity not in self.severity_filter:
            return False
        
        return True

@dataclass
class EscalationRule:
    """Escalation rule configuration"""
    name: str
    severity_levels: List[AlertSeverity]
    timeout_minutes: int
    channels: List[str]  # Channel names
    max_escalations: int = 3
    enabled: bool = True
    
    def applies_to_alert(self, alert: Alert) -> bool:
        """Check if rule applies to alert"""
        return self.enabled and alert.severity in self.severity_levels

class AlertEngine:
    """
    Intelligent alerting and notification engine.
    
    Handles alert creation, grouping, escalation, and multi-channel delivery
    with intelligent consolidation and rate limiting.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "alert_data"
        
        # Alert storage
        self.alerts: Dict[str, Alert] = {}
        self.alert_groups: Dict[str, AlertGroup] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.escalation_rules: List[EscalationRule] = []
        
        # Rate limiting
        self.channel_rate_limits: Dict[str, List[datetime]] = {}
        
        # Configuration
        self.grouping_window_minutes = 15
        self.max_group_size = 10
        self.default_escalation_timeout = 30
        self.alert_retention_days = 30
        
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_alerts()
        self._load_configuration()
        
        # Setup default channels and rules
        self._setup_default_configuration()
    
    async def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        context: AlertContext,
        correlation_id: Optional[str] = None
    ) -> Alert:
        """Create new alert with intelligent grouping"""
        
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            context=context,
            correlation_id=correlation_id
        )
        
        # Store alert
        self.alerts[alert_id] = alert
        
        # Attempt to group with existing alerts
        group = await self._find_or_create_group(alert)
        if group:
            group.add_alert(alert)
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        # Save alert
        self._save_alert(alert)
        
        logger.info(f"Created alert {alert_id}: {title}")
        return alert
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        user: str,
        comment: Optional[str] = None
    ) -> bool:
        """Acknowledge alert"""
        
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.update_status(AlertStatus.ACKNOWLEDGED, user)
        
        # Add acknowledgment to delivery tracking
        alert.delivery_attempts.append({
            'action': 'acknowledged',
            'user': user,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        })
        
        self._save_alert(alert)
        
        logger.info(f"Alert {alert_id} acknowledged by {user}")
        return True
    
    async def snooze_alert(
        self,
        alert_id: str,
        snooze_minutes: int,
        user: str
    ) -> bool:
        """Snooze alert for specified duration"""
        
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.update_status(AlertStatus.SNOOZED, user)
        alert.snoozed_until = datetime.now() + timedelta(minutes=snooze_minutes)
        
        self._save_alert(alert)
        
        logger.info(f"Alert {alert_id} snoozed for {snooze_minutes} minutes by {user}")
        return True
    
    async def resolve_alert(
        self,
        alert_id: str,
        user: str,
        resolution_comment: Optional[str] = None
    ) -> bool:
        """Resolve alert"""
        
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.update_status(AlertStatus.RESOLVED, user)
        
        # Add resolution to delivery tracking
        alert.delivery_attempts.append({
            'action': 'resolved',
            'user': user,
            'comment': resolution_comment,
            'timestamp': datetime.now().isoformat()
        })
        
        self._save_alert(alert)
        
        logger.info(f"Alert {alert_id} resolved by {user}")
        return True
    
    async def process_escalations(self):
        """Process alert escalations"""
        
        escalated_count = 0
        
        for alert in self.alerts.values():
            if alert.should_escalate(self.default_escalation_timeout):
                # Find applicable escalation rules
                applicable_rules = [
                    rule for rule in self.escalation_rules
                    if rule.applies_to_alert(alert)
                ]
                
                if applicable_rules:
                    # Use first applicable rule
                    rule = applicable_rules[0]
                    
                    if alert.escalation_count < rule.max_escalations:
                        await self._escalate_alert(alert, rule)
                        escalated_count += 1
        
        if escalated_count > 0:
            logger.info(f"Escalated {escalated_count} alerts")
    
    async def _escalate_alert(self, alert: Alert, rule: EscalationRule):
        """Escalate individual alert"""
        
        # Update escalation level
        if alert.escalation_level == EscalationLevel.LEVEL_1:
            alert.escalation_level = EscalationLevel.LEVEL_2
        elif alert.escalation_level == EscalationLevel.LEVEL_2:
            alert.escalation_level = EscalationLevel.LEVEL_3
        elif alert.escalation_level == EscalationLevel.LEVEL_3:
            alert.escalation_level = EscalationLevel.LEVEL_4
        
        alert.escalated_at = datetime.now()
        alert.escalation_count += 1
        
        # Send escalation notifications
        escalation_channels = [
            self.notification_channels[channel_name]
            for channel_name in rule.channels
            if channel_name in self.notification_channels
        ]
        
        for channel in escalation_channels:
            await self._send_channel_notification(alert, channel, is_escalation=True)
        
        self._save_alert(alert)
        
        logger.info(f"Escalated alert {alert.alert_id} to level {alert.escalation_level.value}")
    
    async def _find_or_create_group(self, alert: Alert) -> Optional[AlertGroup]:
        """Find existing group or create new one for alert"""
        
        # Look for existing groups within time window
        cutoff_time = datetime.now() - timedelta(minutes=self.grouping_window_minutes)
        
        for group in self.alert_groups.values():
            if group.updated_at < cutoff_time:
                continue
            
            if len(group.alerts) >= self.max_group_size:
                continue
            
            # Check if alert is similar to any in the group
            for existing_alert in group.alerts:
                if alert.is_similar_to(existing_alert):
                    return group
        
        # Create new group if no suitable group found
        group_id = f"group_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        group = AlertGroup(
            group_id=group_id,
            title=alert.title
        )
        
        self.alert_groups[group_id] = group
        return group
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send notifications for new alert"""
        
        # Check if alert is in snooze period
        if alert.snoozed_until and datetime.now() < alert.snoozed_until:
            return
        
        # Send through all applicable channels
        for channel in self.notification_channels.values():
            if channel.should_send_alert(alert):
                await self._send_channel_notification(alert, channel)
    
    async def _send_channel_notification(
        self,
        alert: Alert,
        channel: NotificationChannel,
        is_escalation: bool = False
    ):
        """Send notification through specific channel"""
        
        # Check rate limits
        if not self._check_rate_limit(channel):
            logger.warning(f"Rate limit exceeded for channel {channel.name}")
            return
        
        try:
            success = False
            
            if channel.channel_type == AlertChannel.EMAIL:
                success = await self._send_email_notification(alert, channel, is_escalation)
            elif channel.channel_type == AlertChannel.SLACK:
                success = await self._send_slack_notification(alert, channel, is_escalation)
            elif channel.channel_type == AlertChannel.WEBHOOK:
                success = await self._send_webhook_notification(alert, channel, is_escalation)
            elif channel.channel_type == AlertChannel.TEAMS:
                success = await self._send_teams_notification(alert, channel, is_escalation)
            
            # Track delivery attempt
            delivery_attempt = {
                'channel': channel.name,
                'channel_type': channel.channel_type.value,
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'is_escalation': is_escalation
            }
            
            alert.delivery_attempts.append(delivery_attempt)
            
            if success:
                alert.successful_deliveries.append(channel.name)
                self._update_rate_limit(channel)
            else:
                alert.failed_deliveries.append(channel.name)
            
            self._save_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to send notification through {channel.name}: {e}")
            alert.failed_deliveries.append(channel.name)
    
    async def _send_email_notification(
        self,
        alert: Alert,
        channel: NotificationChannel,
        is_escalation: bool = False
    ) -> bool:
        """Send email notification"""
        
        try:
            if not EMAIL_AVAILABLE:
                logger.info(f"Email notification simulated for alert {alert.alert_id} (email libraries not available)")
                return True
            
            config = channel.config
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = config.get('from_email', 'alerts@finops.com')
            msg['To'] = config.get('to_email')
            msg['Subject'] = f"{'ESCALATION: ' if is_escalation else ''}{alert.severity.value.upper()}: {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert, is_escalation)
            msg.attach(MimeText(body, 'html'))
            
            # Send email (mock implementation)
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False
    
    async def _send_slack_notification(
        self,
        alert: Alert,
        channel: NotificationChannel,
        is_escalation: bool = False
    ) -> bool:
        """Send Slack notification"""
        
        try:
            config = channel.config
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False
            
            # Create Slack message
            color = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "warning")
            
            message = {
                "text": f"{'🚨 ESCALATION: ' if is_escalation else ''}Cost Anomaly Alert",
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Account",
                                "value": alert.context.account_id,
                                "short": True
                            },
                            {
                                "title": "Service",
                                "value": alert.context.service,
                                "short": True
                            },
                            {
                                "title": "Cost Impact",
                                "value": f"${alert.context.cost_amount:.2f}" if alert.context.cost_amount else "N/A",
                                "short": True
                            }
                        ],
                        "footer": "FinOps Anomaly Detection",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            # Send to Slack (mock implementation)
            logger.info(f"Slack notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False
    
    async def _send_webhook_notification(
        self,
        alert: Alert,
        channel: NotificationChannel,
        is_escalation: bool = False
    ) -> bool:
        """Send webhook notification"""
        
        try:
            config = channel.config
            webhook_url = config.get('url')
            
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "is_escalation": is_escalation,
                "escalation_level": alert.escalation_level.value,
                "created_at": alert.created_at.isoformat(),
                "context": {
                    "account_id": alert.context.account_id,
                    "service": alert.context.service,
                    "resource_id": alert.context.resource_id,
                    "region": alert.context.region,
                    "cost_amount": alert.context.cost_amount,
                    "baseline_amount": alert.context.baseline_amount,
                    "deviation_percentage": alert.context.deviation_percentage,
                    "confidence_score": alert.context.confidence_score,
                    "tags": alert.context.tags,
                    "metadata": alert.context.metadata
                }
            }
            
            # Send webhook (mock implementation)
            logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False
    
    async def _send_teams_notification(
        self,
        alert: Alert,
        channel: NotificationChannel,
        is_escalation: bool = False
    ) -> bool:
        """Send Microsoft Teams notification"""
        
        try:
            config = channel.config
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                logger.error("Teams webhook URL not configured")
                return False
            
            # Create Teams message
            color = {
                AlertSeverity.LOW: "0078D4",
                AlertSeverity.MEDIUM: "FF8C00",
                AlertSeverity.HIGH: "FF4B4B",
                AlertSeverity.CRITICAL: "DC143C"
            }.get(alert.severity, "FF8C00")
            
            message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color,
                "summary": f"Cost Anomaly Alert: {alert.title}",
                "sections": [
                    {
                        "activityTitle": f"{'🚨 ESCALATION: ' if is_escalation else ''}Cost Anomaly Detected",
                        "activitySubtitle": alert.title,
                        "facts": [
                            {"name": "Severity", "value": alert.severity.value.upper()},
                            {"name": "Account", "value": alert.context.account_id},
                            {"name": "Service", "value": alert.context.service},
                            {"name": "Cost Impact", "value": f"${alert.context.cost_amount:.2f}" if alert.context.cost_amount else "N/A"},
                            {"name": "Created", "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S")}
                        ],
                        "text": alert.description
                    }
                ]
            }
            
            # Send to Teams (mock implementation)
            logger.info(f"Teams notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Teams notification failed: {e}")
            return False
    
    def _create_email_body(self, alert: Alert, is_escalation: bool = False) -> str:
        """Create HTML email body"""
        
        escalation_header = "<h2 style='color: red;'>🚨 ESCALATION ALERT</h2>" if is_escalation else ""
        
        severity_color = {
            AlertSeverity.LOW: "#28a745",
            AlertSeverity.MEDIUM: "#ffc107",
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545"
        }.get(alert.severity, "#ffc107")
        
        return f"""
        <html>
        <body>
            {escalation_header}
            <h1>Cost Anomaly Alert</h1>
            
            <div style="border-left: 4px solid {severity_color}; padding-left: 20px; margin: 20px 0;">
                <h2>{alert.title}</h2>
                <p><strong>Severity:</strong> <span style="color: {severity_color};">{alert.severity.value.upper()}</span></p>
                <p><strong>Description:</strong> {alert.description}</p>
            </div>
            
            <h3>Alert Details</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Alert ID:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.alert_id}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Account:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.context.account_id}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Service:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.context.service}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Resource:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.context.resource_id or 'N/A'}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Region:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.context.region or 'N/A'}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Cost Amount:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">${alert.context.cost_amount:.2f if alert.context.cost_amount else 'N/A'}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Baseline:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">${alert.context.baseline_amount:.2f if alert.context.baseline_amount else 'N/A'}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Deviation:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.context.deviation_percentage:.1f}% if alert.context.deviation_percentage else 'N/A'</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Confidence:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.context.confidence_score:.2f if alert.context.confidence_score else 'N/A'}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Created:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>
            
            <p style="margin-top: 20px; font-size: 12px; color: #666;">
                This alert was generated by the FinOps Anomaly Detection System.
            </p>
        </body>
        </html>
        """
    
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel is within rate limits"""
        
        if not channel.rate_limit_per_hour:
            return True
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Get recent notifications for this channel
        recent_notifications = self.channel_rate_limits.get(channel.name, [])
        recent_notifications = [ts for ts in recent_notifications if ts > hour_ago]
        
        return len(recent_notifications) < channel.rate_limit_per_hour
    
    def _update_rate_limit(self, channel: NotificationChannel):
        """Update rate limit tracking for channel"""
        
        if channel.name not in self.channel_rate_limits:
            self.channel_rate_limits[channel.name] = []
        
        self.channel_rate_limits[channel.name].append(datetime.now())
        
        # Clean up old entries
        hour_ago = datetime.now() - timedelta(hours=1)
        self.channel_rate_limits[channel.name] = [
            ts for ts in self.channel_rate_limits[channel.name] if ts > hour_ago
        ]
    
    def add_notification_channel(
        self,
        name: str,
        channel_type: AlertChannel,
        config: Dict[str, Any],
        severity_filter: List[AlertSeverity] = None,
        rate_limit_per_hour: int = None
    ) -> NotificationChannel:
        """Add new notification channel"""
        
        channel = NotificationChannel(
            channel_type=channel_type,
            name=name,
            config=config,
            severity_filter=severity_filter or list(AlertSeverity),
            rate_limit_per_hour=rate_limit_per_hour
        )
        
        self.notification_channels[name] = channel
        self._save_configuration()
        
        logger.info(f"Added notification channel: {name} ({channel_type.value})")
        return channel
    
    def add_escalation_rule(
        self,
        name: str,
        severity_levels: List[AlertSeverity],
        timeout_minutes: int,
        channels: List[str],
        max_escalations: int = 3
    ) -> EscalationRule:
        """Add new escalation rule"""
        
        rule = EscalationRule(
            name=name,
            severity_levels=severity_levels,
            timeout_minutes=timeout_minutes,
            channels=channels,
            max_escalations=max_escalations
        )
        
        self.escalation_rules.append(rule)
        self._save_configuration()
        
        logger.info(f"Added escalation rule: {name}")
        return rule
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alerts.values()
            if alert.created_at > cutoff_time
        ]
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in recent_alerts if a.severity == severity
            ])
        
        # Count by status
        status_counts = {}
        for status in AlertStatus:
            status_counts[status.value] = len([
                a for a in recent_alerts if a.status == status
            ])
        
        # Channel delivery statistics
        channel_stats = {}
        for channel_name in self.notification_channels.keys():
            successful = len([
                a for a in recent_alerts if channel_name in a.successful_deliveries
            ])
            failed = len([
                a for a in recent_alerts if channel_name in a.failed_deliveries
            ])
            channel_stats[channel_name] = {
                'successful': successful,
                'failed': failed,
                'success_rate': successful / (successful + failed) if (successful + failed) > 0 else 0
            }
        
        return {
            'total_alerts': len(recent_alerts),
            'severity_breakdown': severity_counts,
            'status_breakdown': status_counts,
            'channel_statistics': channel_stats,
            'escalated_alerts': len([a for a in recent_alerts if a.escalation_count > 0]),
            'grouped_alerts': len([a for a in recent_alerts if a.group_id]),
            'time_period_hours': hours
        }
    
    def _load_alerts(self):
        """Load alerts from storage"""
        
        try:
            alerts_file = Path(self.storage_path) / "alerts.json"
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    alerts_data = json.load(f)
                
                for alert_data in alerts_data:
                    alert = self._deserialize_alert(alert_data)
                    self.alerts[alert.alert_id] = alert
            
            logger.info(f"Loaded {len(self.alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")
    
    def _save_alert(self, alert: Alert):
        """Save individual alert"""
        
        try:
            # For demo purposes, we'll just log the save operation
            # In production, this would save to persistent storage
            logger.debug(f"Saved alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to save alert {alert.alert_id}: {e}")
    
    def _load_configuration(self):
        """Load configuration from storage"""
        
        try:
            config_file = Path(self.storage_path) / "alert_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Load notification channels
                for channel_data in config_data.get('channels', []):
                    channel = NotificationChannel(
                        channel_type=AlertChannel(channel_data['channel_type']),
                        name=channel_data['name'],
                        config=channel_data['config'],
                        enabled=channel_data.get('enabled', True),
                        severity_filter=[AlertSeverity(s) for s in channel_data.get('severity_filter', [])],
                        rate_limit_per_hour=channel_data.get('rate_limit_per_hour')
                    )
                    self.notification_channels[channel.name] = channel
                
                # Load escalation rules
                for rule_data in config_data.get('escalation_rules', []):
                    rule = EscalationRule(
                        name=rule_data['name'],
                        severity_levels=[AlertSeverity(s) for s in rule_data['severity_levels']],
                        timeout_minutes=rule_data['timeout_minutes'],
                        channels=rule_data['channels'],
                        max_escalations=rule_data.get('max_escalations', 3),
                        enabled=rule_data.get('enabled', True)
                    )
                    self.escalation_rules.append(rule)
            
            logger.info(f"Loaded {len(self.notification_channels)} channels and {len(self.escalation_rules)} escalation rules")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _save_configuration(self):
        """Save configuration to storage"""
        
        try:
            config_data = {
                'channels': [
                    {
                        'channel_type': channel.channel_type.value,
                        'name': channel.name,
                        'config': channel.config,
                        'enabled': channel.enabled,
                        'severity_filter': [s.value for s in channel.severity_filter],
                        'rate_limit_per_hour': channel.rate_limit_per_hour
                    }
                    for channel in self.notification_channels.values()
                ],
                'escalation_rules': [
                    {
                        'name': rule.name,
                        'severity_levels': [s.value for s in rule.severity_levels],
                        'timeout_minutes': rule.timeout_minutes,
                        'channels': rule.channels,
                        'max_escalations': rule.max_escalations,
                        'enabled': rule.enabled
                    }
                    for rule in self.escalation_rules
                ]
            }
            
            config_file = Path(self.storage_path) / "alert_config.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _setup_default_configuration(self):
        """Setup default notification channels and escalation rules"""
        
        # Only setup defaults if no configuration exists
        if self.notification_channels or self.escalation_rules:
            return
        
        # Default email channel
        self.add_notification_channel(
            name="default_email",
            channel_type=AlertChannel.EMAIL,
            config={
                'from_email': 'alerts@finops.com',
                'to_email': 'admin@company.com',
                'smtp_server': 'localhost',
                'smtp_port': 587
            },
            rate_limit_per_hour=10
        )
        
        # Default Slack channel for critical alerts
        self.add_notification_channel(
            name="critical_slack",
            channel_type=AlertChannel.SLACK,
            config={
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            },
            severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL],
            rate_limit_per_hour=20
        )
        
        # Default escalation rule
        self.add_escalation_rule(
            name="critical_escalation",
            severity_levels=[AlertSeverity.CRITICAL],
            timeout_minutes=15,
            channels=["default_email", "critical_slack"],
            max_escalations=3
        )
        
        logger.info("Setup default alert configuration")
    
    def _deserialize_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Deserialize alert from dictionary"""
        
        context = AlertContext(
            account_id=alert_data['context']['account_id'],
            service=alert_data['context']['service'],
            resource_id=alert_data['context'].get('resource_id'),
            region=alert_data['context'].get('region'),
            cost_amount=alert_data['context'].get('cost_amount'),
            baseline_amount=alert_data['context'].get('baseline_amount'),
            deviation_percentage=alert_data['context'].get('deviation_percentage'),
            confidence_score=alert_data['context'].get('confidence_score'),
            tags=alert_data['context'].get('tags', {}),
            metadata=alert_data['context'].get('metadata', {})
        )
        
        alert = Alert(
            alert_id=alert_data['alert_id'],
            title=alert_data['title'],
            description=alert_data['description'],
            severity=AlertSeverity(alert_data['severity']),
            context=context,
            created_at=datetime.fromisoformat(alert_data['created_at']),
            updated_at=datetime.fromisoformat(alert_data['updated_at']),
            resolved_at=datetime.fromisoformat(alert_data['resolved_at']) if alert_data.get('resolved_at') else None,
            status=AlertStatus(alert_data['status']),
            acknowledged_by=alert_data.get('acknowledged_by'),
            acknowledged_at=datetime.fromisoformat(alert_data['acknowledged_at']) if alert_data.get('acknowledged_at') else None,
            snoozed_until=datetime.fromisoformat(alert_data['snoozed_until']) if alert_data.get('snoozed_until') else None,
            group_id=alert_data.get('group_id'),
            correlation_id=alert_data.get('correlation_id'),
            parent_alert_id=alert_data.get('parent_alert_id'),
            related_alert_ids=alert_data.get('related_alert_ids', []),
            escalation_level=EscalationLevel(alert_data.get('escalation_level', 1)),
            escalated_at=datetime.fromisoformat(alert_data['escalated_at']) if alert_data.get('escalated_at') else None,
            escalation_count=alert_data.get('escalation_count', 0),
            delivery_attempts=alert_data.get('delivery_attempts', []),
            successful_deliveries=alert_data.get('successful_deliveries', []),
            failed_deliveries=alert_data.get('failed_deliveries', [])
        )
        
        return alert

# Global alert engine instance
alert_engine = AlertEngine()