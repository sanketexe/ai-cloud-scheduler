#!/usr/bin/env python3
"""
Intelligent Budget Management and Alerting System

This module implements comprehensive budget management capabilities including:
- Flexible budget creation and configuration
- Real-time spending tracking and monitoring
- Predictive spending forecasting
- Proactive alert and notification system
- Budget variance analysis and reporting
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics
import uuid
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BudgetPeriod(Enum):
    """Budget period types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class BudgetDimension(Enum):
    """Budget dimension types for flexible categorization"""
    TEAM = "team"
    PROJECT = "project"
    DEPARTMENT = "department"
    BUSINESS_UNIT = "business_unit"
    SERVICE = "service"
    ENVIRONMENT = "environment"
    REGION = "region"
    COST_CENTER = "cost_center"
    CUSTOM = "custom"


class BudgetStatus(Enum):
    """Budget status types"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    DRAFT = "draft"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class NotificationChannel(Enum):
    """Notification channel types"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class BudgetDimensionFilter:
    """Filter criteria for budget dimensions"""
    dimension: BudgetDimension
    values: List[str]
    include: bool = True  # True for include, False for exclude
    
    def matches(self, resource_attributes: Dict[str, str]) -> bool:
        """Check if resource attributes match this filter"""
        dimension_key = self.dimension.value
        resource_value = resource_attributes.get(dimension_key)
        
        if resource_value is None:
            return not self.include  # If no value, exclude if include=True
        
        value_matches = resource_value in self.values
        return value_matches if self.include else not value_matches


@dataclass
class Budget:
    """Comprehensive budget definition with flexible dimensions"""
    budget_id: str
    name: str
    description: str
    amount: float
    currency: str
    period: BudgetPeriod
    start_date: datetime
    end_date: datetime
    dimension_filters: List[BudgetDimensionFilter]
    alert_thresholds: List[float] = field(default_factory=lambda: [50.0, 75.0, 90.0, 100.0])
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    notification_recipients: Dict[NotificationChannel, List[str]] = field(default_factory=dict)
    status: BudgetStatus = BudgetStatus.ACTIVE
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    last_modified: datetime = field(default_factory=datetime.now)
    last_modified_by: Optional[str] = None
    
    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError("Budget amount must be positive")
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if not self.alert_thresholds:
            self.alert_thresholds = [50.0, 75.0, 90.0, 100.0]
        # Sort alert thresholds
        self.alert_thresholds.sort()
    
    def is_applicable_to_resource(self, resource_attributes: Dict[str, str]) -> bool:
        """Check if this budget applies to a resource based on dimension filters"""
        if not self.dimension_filters:
            return True  # No filters means applies to all resources
        
        # All filters must match (AND logic)
        for filter_criteria in self.dimension_filters:
            if not filter_criteria.matches(resource_attributes):
                return False
        
        return True


@dataclass
class SpendingSnapshot:
    """Point-in-time spending snapshot for a budget"""
    budget_id: str
    snapshot_time: datetime
    current_spend: float
    projected_spend: float
    utilization_percentage: float
    spending_velocity: float  # spend per day
    days_remaining: int
    is_on_track: bool
    variance_from_expected: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)  # by service, team, etc.
    
    def get_burn_rate(self) -> float:
        """Calculate current burn rate (spend per day)"""
        return self.spending_velocity
    
    def get_projected_overage(self, budget_amount: float) -> float:
        """Calculate projected budget overage"""
        return max(0, self.projected_spend - budget_amount)


class SpendingTracker:
    """Real-time spending tracking with continuous cost monitoring"""
    
    def __init__(self, cost_collector=None):
        self.logger = logging.getLogger(f"{__name__}.SpendingTracker")
        self.cost_collector = cost_collector
        self.spending_snapshots: Dict[str, List[SpendingSnapshot]] = defaultdict(list)
        self.tracking_active = False
        self.tracking_interval = 3600  # 1 hour in seconds
    
    def get_current_spending_status(self, budget: Budget) -> Optional[SpendingSnapshot]:
        """Get current spending status for a budget"""
        snapshots = self.spending_snapshots.get(budget.budget_id, [])
        if snapshots:
            return snapshots[-1]  # Return most recent snapshot
        return None
    
    async def _update_budget_spending(self, budget: Budget):
        """Update spending information for a single budget"""
        try:
            # Generate mock spending data for testing
            current_spend = self._generate_mock_spending(budget)
            spending_velocity = self._calculate_spending_velocity(budget)
            projected_spend = self._calculate_projected_spend(budget, current_spend, spending_velocity)
            
            # Calculate utilization and tracking status
            utilization_percentage = (current_spend / budget.amount) * 100
            days_remaining = self._calculate_days_remaining(budget)
            is_on_track = self._is_spending_on_track(budget, current_spend, projected_spend)
            variance_from_expected = self._calculate_variance_from_expected(budget, current_spend)
            
            # Create spending snapshot
            snapshot = SpendingSnapshot(
                budget_id=budget.budget_id,
                snapshot_time=datetime.now(),
                current_spend=current_spend,
                projected_spend=projected_spend,
                utilization_percentage=utilization_percentage,
                spending_velocity=spending_velocity,
                days_remaining=days_remaining,
                is_on_track=is_on_track,
                variance_from_expected=variance_from_expected,
                cost_breakdown={}
            )
            
            # Store snapshot
            self.spending_snapshots[budget.budget_id].append(snapshot)
            
            # Keep only last 30 days of snapshots
            cutoff_time = datetime.now() - timedelta(days=30)
            self.spending_snapshots[budget.budget_id] = [
                s for s in self.spending_snapshots[budget.budget_id]
                if s.snapshot_time >= cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Error updating spending for budget {budget.name}: {e}")    

    def _generate_mock_spending(self, budget: Budget) -> float:
        """Generate mock spending data for testing"""
        # Calculate days elapsed since budget start
        days_elapsed = max(1, (datetime.now() - budget.start_date).days)
        total_budget_days = (budget.end_date - budget.start_date).days
        
        if total_budget_days <= 0:
            return 0.0
        
        # Simulate spending with some randomness
        expected_daily_spend = budget.amount / total_budget_days
        
        # Add some variance (±20%)
        import random
        variance_factor = random.uniform(0.8, 1.2)
        actual_daily_spend = expected_daily_spend * variance_factor
        
        # Calculate total spend with some acceleration/deceleration
        progress_factor = days_elapsed / total_budget_days
        if progress_factor > 0.7:  # Accelerate spending near end of period
            acceleration = 1.1
        else:
            acceleration = 1.0
        
        total_spend = actual_daily_spend * days_elapsed * acceleration
        
        # Ensure we don't exceed reasonable bounds
        return min(total_spend, budget.amount * 1.5)  # Cap at 150% of budget
    
    def _calculate_spending_velocity(self, budget: Budget) -> float:
        """Calculate current spending velocity (spend per day)"""
        snapshots = self.spending_snapshots.get(budget.budget_id, [])
        
        if len(snapshots) < 2:
            # Not enough data, estimate from current spend
            days_elapsed = max(1, (datetime.now() - budget.start_date).days)
            if snapshots:
                return snapshots[-1].current_spend / days_elapsed
            return 0.0
        
        # Calculate velocity from recent snapshots
        recent_snapshots = snapshots[-7:]  # Last 7 snapshots
        if len(recent_snapshots) >= 2:
            time_diff = (recent_snapshots[-1].snapshot_time - recent_snapshots[0].snapshot_time).total_seconds() / 86400  # days
            spend_diff = recent_snapshots[-1].current_spend - recent_snapshots[0].current_spend
            
            if time_diff > 0:
                return spend_diff / time_diff
        
        return 0.0
    
    def _calculate_projected_spend(self, budget: Budget, current_spend: float, spending_velocity: float) -> float:
        """Calculate projected spending for the budget period"""
        days_remaining = self._calculate_days_remaining(budget)
        if days_remaining <= 0:
            return current_spend
        
        # Project spending based on current velocity
        projected_additional_spend = spending_velocity * days_remaining
        return current_spend + projected_additional_spend
    
    def _calculate_days_remaining(self, budget: Budget) -> int:
        """Calculate days remaining in budget period"""
        now = datetime.now()
        if now >= budget.end_date:
            return 0
        return (budget.end_date - now).days
    
    def _is_spending_on_track(self, budget: Budget, current_spend: float, projected_spend: float) -> bool:
        """Check if spending is on track with budget"""
        # Allow 10% variance
        return projected_spend <= budget.amount * 1.1
    
    def _calculate_variance_from_expected(self, budget: Budget, current_spend: float) -> float:
        """Calculate variance from expected spending"""
        days_elapsed = max(1, (datetime.now() - budget.start_date).days)
        total_days = (budget.end_date - budget.start_date).days
        
        if total_days <= 0:
            return 0.0
        
        expected_spend = budget.amount * (days_elapsed / total_days)
        return current_spend - expected_spend


@dataclass
class BudgetAlert:
    """Budget alert definition and status"""
    alert_id: str
    budget_id: str
    alert_type: str  # "threshold", "forecast", "velocity", "anomaly"
    severity: AlertSeverity
    threshold_percentage: Optional[float] = None
    current_value: float = 0.0
    threshold_value: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    notification_sent: bool = False
    escalation_level: int = 0
    
    @property
    def is_active(self) -> bool:
        """Check if alert is still active"""
        return self.resolved_at is None
    
    @property
    def is_acknowledged(self) -> bool:
        """Check if alert has been acknowledged"""
        return self.acknowledged_at is not None
    
    @property
    def duration_minutes(self) -> int:
        """Get alert duration in minutes"""
        end_time = self.resolved_at or datetime.now()
        return int((end_time - self.triggered_at).total_seconds() / 60)
    
    def acknowledge(self, acknowledged_by: str):
        """Acknowledge the alert"""
        if not self.is_acknowledged:
            self.acknowledged_at = datetime.now()
            self.acknowledged_by = acknowledged_by
    
    def resolve(self, resolved_by: str):
        """Resolve the alert"""
        if self.is_active:
            self.resolved_at = datetime.now()
            self.resolved_by = resolved_by


@dataclass
class NotificationTemplate:
    """Template for alert notifications"""
    template_id: str
    name: str
    subject_template: str
    body_template: str
    channel: NotificationChannel
    severity_levels: List[AlertSeverity] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    
    def render_subject(self, alert: BudgetAlert, budget: Budget) -> str:
        """Render notification subject with alert data"""
        return self.subject_template.format(
            budget_name=budget.name,
            alert_type=alert.alert_type,
            severity=alert.severity.value,
            current_value=alert.current_value,
            threshold_value=alert.threshold_value,
            **alert.details
        )
    
    def render_body(self, alert: BudgetAlert, budget: Budget) -> str:
        """Render notification body with alert data"""
        return self.body_template.format(
            budget_name=budget.name,
            budget_amount=budget.amount,
            alert_type=alert.alert_type,
            severity=alert.severity.value,
            current_value=alert.current_value,
            threshold_value=alert.threshold_value,
            message=alert.message,
            triggered_at=alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S"),
            **alert.details
        )


@dataclass
class EscalationRule:
    """Alert escalation rule definition"""
    rule_id: str
    name: str
    trigger_conditions: Dict[str, Any]  # conditions that trigger escalation
    escalation_delay_minutes: int
    target_channels: List[NotificationChannel]
    target_recipients: Dict[NotificationChannel, List[str]]
    max_escalation_level: int = 3
    
    def should_escalate(self, alert: BudgetAlert) -> bool:
        """Check if alert should be escalated"""
        if alert.escalation_level >= self.max_escalation_level:
            return False
        
        if alert.is_acknowledged:
            return False
        
        # Check if enough time has passed
        minutes_since_triggered = (datetime.now() - alert.triggered_at).total_seconds() / 60
        required_delay = self.escalation_delay_minutes * (alert.escalation_level + 1)
        
        if minutes_since_triggered < required_delay:
            return False
        
        # Check severity conditions
        if 'min_severity' in self.trigger_conditions:
            min_severity = AlertSeverity(self.trigger_conditions['min_severity'])
            severity_order = [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            if severity_order.index(alert.severity) < severity_order.index(min_severity):
                return False
        
        return True


class NotificationService:
    """Service for sending notifications through various channels"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.NotificationService")
        self.templates: Dict[str, NotificationTemplate] = {}
        self.channel_configs: Dict[NotificationChannel, Dict[str, Any]] = {}
        self._setup_default_templates()
    
    def configure_channel(self, channel: NotificationChannel, config: Dict[str, Any]):
        """Configure notification channel settings"""
        self.channel_configs[channel] = config
        self.logger.info(f"Configured {channel.value} notification channel")
    
    def _setup_default_templates(self):
        """Setup default notification templates"""
        
        # Email template for budget threshold alerts
        email_threshold_template = NotificationTemplate(
            template_id="email_threshold",
            name="Email Budget Threshold Alert",
            subject_template="🚨 Budget Alert: {budget_name} - {threshold_percentage}% threshold exceeded",
            body_template="""Budget Alert Notification

Budget: {budget_name}
Alert Type: {alert_type}
Severity: {severity}

Current Spending: ${current_value:,.2f}
Budget Amount: ${budget_amount:,.2f}
Utilization: {utilization_percentage:.1f}%
Threshold: {threshold_percentage}%

Message: {message}

Triggered: {triggered_at}

Please review your budget and take appropriate action.""",
            channel=NotificationChannel.EMAIL,
            severity_levels=[AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )
        self.register_template(email_threshold_template)  
      
        # Slack template for budget alerts
        slack_template = NotificationTemplate(
            template_id="slack_budget",
            name="Slack Budget Alert",
            subject_template="Budget Alert: {budget_name}",
            body_template="""🚨 *Budget Alert: {budget_name}*

• *Current Spending:* ${current_value:,.2f}
• *Budget Amount:* ${budget_amount:,.2f}
• *Utilization:* {utilization_percentage:.1f}%
• *Severity:* {severity}

_{message}_

Triggered at {triggered_at}""",
            channel=NotificationChannel.SLACK,
            severity_levels=[AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )
        self.register_template(slack_template)
    
    def register_template(self, template: NotificationTemplate):
        """Register a notification template"""
        self.templates[template.template_id] = template
        self.logger.info(f"Registered notification template: {template.name}")
    
    async def send_notification(self, 
                              alert: BudgetAlert, 
                              budget: Budget,
                              channel: NotificationChannel,
                              recipients: List[str],
                              template_id: Optional[str] = None) -> bool:
        """Send notification for an alert"""
        try:
            # Get appropriate template
            if template_id and template_id in self.templates:
                template = self.templates[template_id]
            else:
                # Find default template for channel
                template = self._get_default_template(channel, alert.severity)
            
            if not template:
                self.logger.error(f"No template found for channel {channel.value}")
                return False
            
            # Add utilization percentage to alert details for template rendering
            if 'utilization_percentage' not in alert.details:
                alert.details['utilization_percentage'] = (alert.current_value / budget.amount) * 100 if budget.amount > 0 else 0
                alert.details['threshold_percentage'] = alert.threshold_percentage or 0
            
            # Render notification content
            subject = template.render_subject(alert, budget)
            body = template.render_body(alert, budget)
            
            # Send through appropriate channel
            success = await self._send_through_channel(channel, recipients, subject, body, alert)
            
            if success:
                self.logger.info(f"Sent {channel.value} notification for alert {alert.alert_id}")
            else:
                self.logger.error(f"Failed to send {channel.value} notification for alert {alert.alert_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False
    
    def _get_default_template(self, channel: NotificationChannel, severity: AlertSeverity) -> Optional[NotificationTemplate]:
        """Get default template for channel and severity"""
        for template in self.templates.values():
            if (template.channel == channel and 
                (not template.severity_levels or severity in template.severity_levels)):
                return template
        return None   
 
    async def _send_through_channel(self, 
                                  channel: NotificationChannel,
                                  recipients: List[str],
                                  subject: str,
                                  body: str,
                                  alert: BudgetAlert) -> bool:
        """Send notification through specific channel"""
        
        if channel == NotificationChannel.EMAIL:
            return await self._send_email(recipients, subject, body)
        elif channel == NotificationChannel.SLACK:
            return await self._send_slack(recipients, subject, body)
        elif channel == NotificationChannel.TEAMS:
            return await self._send_teams(recipients, subject, body)
        elif channel == NotificationChannel.WEBHOOK:
            return await self._send_webhook(recipients, subject, body, alert)
        elif channel == NotificationChannel.SMS:
            return await self._send_sms(recipients, subject, body)
        else:
            self.logger.error(f"Unsupported notification channel: {channel.value}")
            return False
    
    async def _send_email(self, recipients: List[str], subject: str, body: str) -> bool:
        """Send email notification"""
        try:
            # In production, this would use actual email service (SMTP, SES, etc.)
            self.logger.info(f"EMAIL: To: {', '.join(recipients)}")
            self.logger.info(f"EMAIL: Subject: {subject}")
            self.logger.info(f"EMAIL: Body: {body[:200]}...")
            
            # Mock successful email sending
            await asyncio.sleep(0.1)  # Simulate network delay
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_slack(self, recipients: List[str], subject: str, body: str) -> bool:
        """Send Slack notification"""
        try:
            # In production, this would use Slack API
            slack_config = self.channel_configs.get(NotificationChannel.SLACK, {})
            webhook_url = slack_config.get('webhook_url')
            
            self.logger.info(f"SLACK: Channels: {', '.join(recipients)}")
            self.logger.info(f"SLACK: Message: {body[:200]}...")
            
            # Mock successful Slack sending
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_teams(self, recipients: List[str], subject: str, body: str) -> bool:
        """Send Microsoft Teams notification"""
        try:
            # In production, this would use Teams API
            teams_config = self.channel_configs.get(NotificationChannel.TEAMS, {})
            webhook_url = teams_config.get('webhook_url')
            
            self.logger.info(f"TEAMS: Channels: {', '.join(recipients)}")
            self.logger.info(f"TEAMS: Message: {body[:200]}...")
            
            # Mock successful Teams sending
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Teams notification: {e}")
            return False   
 
    async def _send_webhook(self, recipients: List[str], subject: str, body: str, alert: BudgetAlert) -> bool:
        """Send webhook notification"""
        try:
            # In production, this would make HTTP POST requests
            webhook_config = self.channel_configs.get(NotificationChannel.WEBHOOK, {})
            
            payload = {
                'alert_id': alert.alert_id,
                'budget_id': alert.budget_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity.value,
                'subject': subject,
                'body': body,
                'triggered_at': alert.triggered_at.isoformat(),
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value
            }
            
            self.logger.info(f"WEBHOOK: URLs: {', '.join(recipients)}")
            self.logger.info(f"WEBHOOK: Payload: {payload}")
            
            # Mock successful webhook sending
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending webhook notification: {e}")
            return False
    
    async def _send_sms(self, recipients: List[str], subject: str, body: str) -> bool:
        """Send SMS notification"""
        try:
            # In production, this would use SMS service (Twilio, SNS, etc.)
            sms_config = self.channel_configs.get(NotificationChannel.SMS, {})
            
            # Truncate body for SMS
            sms_body = body[:160] + "..." if len(body) > 160 else body
            
            self.logger.info(f"SMS: Numbers: {', '.join(recipients)}")
            self.logger.info(f"SMS: Message: {sms_body}")
            
            # Mock successful SMS sending
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending SMS: {e}")
            return False


class AlertManager:
    """Comprehensive alert management system with configurable thresholds and notifications"""
    
    def __init__(self, notification_service: Optional[NotificationService] = None):
        self.logger = logging.getLogger(f"{__name__}.AlertManager")
        self.notification_service = notification_service or NotificationService()
        self.active_alerts: Dict[str, BudgetAlert] = {}
        self.alert_history: List[BudgetAlert] = []
        self.escalation_rules: Dict[str, EscalationRule] = {}
        self.alert_suppression: Dict[str, datetime] = {}  # budget_id -> suppression_end_time
        self.monitoring_active = False
        self.check_interval = 300  # 5 minutes
        self._setup_default_escalation_rules()
    
    def _setup_default_escalation_rules(self):
        """Setup default escalation rules"""
        
        # Critical alert escalation
        critical_escalation = EscalationRule(
            rule_id="critical_budget_escalation",
            name="Critical Budget Alert Escalation",
            trigger_conditions={
                'min_severity': 'critical',
                'unacknowledged_minutes': 15
            },
            escalation_delay_minutes=15,
            target_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            target_recipients={
                NotificationChannel.EMAIL: ['finance-team@company.com', 'executives@company.com'],
                NotificationChannel.SLACK: ['#finance-alerts', '#executive-alerts']
            },
            max_escalation_level=3
        )
        self.register_escalation_rule(critical_escalation)        

        # Emergency alert escalation
        emergency_escalation = EscalationRule(
            rule_id="emergency_budget_escalation",
            name="Emergency Budget Alert Escalation",
            trigger_conditions={
                'min_severity': 'emergency',
                'unacknowledged_minutes': 5
            },
            escalation_delay_minutes=5,
            target_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.SLACK],
            target_recipients={
                NotificationChannel.EMAIL: ['cfo@company.com', 'ceo@company.com'],
                NotificationChannel.SMS: ['+1234567890'],  # Executive phone numbers
                NotificationChannel.SLACK: ['#emergency-alerts']
            },
            max_escalation_level=2
        )
        self.register_escalation_rule(emergency_escalation)
    
    def register_escalation_rule(self, rule: EscalationRule):
        """Register an escalation rule"""
        self.escalation_rules[rule.rule_id] = rule
        self.logger.info(f"Registered escalation rule: {rule.name}")
    
    async def start_alert_monitoring(self, budgets: List[Budget], spending_tracker: SpendingTracker):
        """Start continuous alert monitoring"""
        self.monitoring_active = True
        self.logger.info(f"Starting alert monitoring for {len(budgets)} budgets")
        
        while self.monitoring_active:
            try:
                # Check all budgets for alert conditions
                for budget in budgets:
                    if budget.status == BudgetStatus.ACTIVE:
                        await self._check_budget_alerts(budget, spending_tracker)
                
                # Process escalations
                await self._process_escalations()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Wait for next check interval
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop_alert_monitoring(self):
        """Stop alert monitoring"""
        self.monitoring_active = False
        self.logger.info("Stopped alert monitoring")
    
    async def _check_budget_alerts(self, budget: Budget, spending_tracker: SpendingTracker):
        """Check for alert conditions on a budget"""
        try:
            # Get current spending status
            current_snapshot = spending_tracker.get_current_spending_status(budget)
            if not current_snapshot:
                return
            
            # Check if budget is suppressed
            if self._is_alert_suppressed(budget.budget_id):
                return
            
            # Check threshold alerts
            await self._check_threshold_alerts(budget, current_snapshot)
            
            # Check forecast alerts
            await self._check_forecast_alerts(budget, current_snapshot)
            
            # Check velocity alerts
            await self._check_velocity_alerts(budget, current_snapshot)
            
            # Check anomaly alerts
            await self._check_anomaly_alerts(budget, current_snapshot)
            
        except Exception as e:
            self.logger.error(f"Error checking alerts for budget {budget.name}: {e}") 
   
    async def _check_threshold_alerts(self, budget: Budget, snapshot: SpendingSnapshot):
        """Check for budget threshold alerts"""
        utilization_percentage = snapshot.utilization_percentage
        
        for threshold in budget.alert_thresholds:
            if utilization_percentage >= threshold:
                # Check if we already have an active alert for this threshold
                alert_key = f"{budget.budget_id}_threshold_{threshold}"
                
                if alert_key not in self.active_alerts:
                    # Determine severity based on threshold
                    if threshold >= 100:
                        severity = AlertSeverity.EMERGENCY
                    elif threshold >= 90:
                        severity = AlertSeverity.CRITICAL
                    elif threshold >= 75:
                        severity = AlertSeverity.WARNING
                    else:
                        severity = AlertSeverity.INFO
                    
                    # Create alert
                    alert = BudgetAlert(
                        alert_id=alert_key,
                        budget_id=budget.budget_id,
                        alert_type="threshold",
                        severity=severity,
                        threshold_percentage=threshold,
                        current_value=snapshot.current_spend,
                        threshold_value=budget.amount * (threshold / 100),
                        message=f"Budget {budget.name} has reached {utilization_percentage:.1f}% utilization (threshold: {threshold}%)",
                        details={
                            'utilization_percentage': utilization_percentage,
                            'budget_amount': budget.amount,
                            'projected_spend': snapshot.projected_spend,
                            'days_remaining': snapshot.days_remaining
                        }
                    )
                    
                    await self._trigger_alert(alert, budget)
    
    async def _check_forecast_alerts(self, budget: Budget, snapshot: SpendingSnapshot):
        """Check for forecast-based alerts"""
        if snapshot.projected_spend > budget.amount * 1.1:  # 10% over budget projection
            alert_key = f"{budget.budget_id}_forecast_overage"
            
            if alert_key not in self.active_alerts:
                projected_overage = snapshot.projected_spend - budget.amount
                overage_percentage = (projected_overage / budget.amount) * 100
                
                # Determine severity based on projected overage
                if overage_percentage >= 50:
                    severity = AlertSeverity.EMERGENCY
                elif overage_percentage >= 25:
                    severity = AlertSeverity.CRITICAL
                elif overage_percentage >= 10:
                    severity = AlertSeverity.WARNING
                else:
                    severity = AlertSeverity.INFO
                
                alert = BudgetAlert(
                    alert_id=alert_key,
                    budget_id=budget.budget_id,
                    alert_type="forecast",
                    severity=severity,
                    current_value=snapshot.projected_spend,
                    threshold_value=budget.amount,
                    message=f"Budget {budget.name} is projected to exceed budget by ${projected_overage:,.2f} ({overage_percentage:.1f}%)",
                    details={
                        'projected_overage': projected_overage,
                        'overage_percentage': overage_percentage,
                        'current_spend': snapshot.current_spend,
                        'days_remaining': snapshot.days_remaining
                    }
                )
                
                await self._trigger_alert(alert, budget)
    
    async def _check_velocity_alerts(self, budget: Budget, snapshot: SpendingSnapshot):
        """Check for spending velocity alerts"""
        if snapshot.spending_velocity > 0:
            # Calculate expected daily spend
            budget_days = (budget.end_date - budget.start_date).days
            expected_daily_spend = budget.amount / budget_days if budget_days > 0 else 0
            
            # Alert if spending velocity is significantly higher than expected
            if snapshot.spending_velocity > expected_daily_spend * 2:  # 2x expected rate
                alert_key = f"{budget.budget_id}_velocity_high"
                
                if alert_key not in self.active_alerts:
                    velocity_ratio = snapshot.spending_velocity / expected_daily_spend if expected_daily_spend > 0 else 0
                    
                    severity = AlertSeverity.CRITICAL if velocity_ratio > 3 else AlertSeverity.WARNING
                    
                    alert = BudgetAlert(
                        alert_id=alert_key,
                        budget_id=budget.budget_id,
                        alert_type="velocity",
                        severity=severity,
                        current_value=snapshot.spending_velocity,
                        threshold_value=expected_daily_spend * 2,
                        message=f"Budget {budget.name} spending velocity is {velocity_ratio:.1f}x higher than expected",
                        details={
                            'velocity_ratio': velocity_ratio,
                            'expected_daily_spend': expected_daily_spend,
                            'current_velocity': snapshot.spending_velocity
                        }
                    )
                    
                    await self._trigger_alert(alert, budget)
    
    async def _check_anomaly_alerts(self, budget: Budget, snapshot: SpendingSnapshot):
        """Check for spending anomalies"""
        # This is a simplified anomaly detection
        # In production, this would use more sophisticated ML-based anomaly detection
        
        if not snapshot.is_on_track and abs(snapshot.variance_from_expected) > budget.amount * 0.2:  # 20% variance
            alert_key = f"{budget.budget_id}_anomaly"
            
            if alert_key not in self.active_alerts:
                variance_percentage = (abs(snapshot.variance_from_expected) / budget.amount) * 100
                
                severity = AlertSeverity.WARNING if variance_percentage < 50 else AlertSeverity.CRITICAL
                
                alert = BudgetAlert(
                    alert_id=alert_key,
                    budget_id=budget.budget_id,
                    alert_type="anomaly",
                    severity=severity,
                    current_value=snapshot.current_spend,
                    threshold_value=snapshot.current_spend - snapshot.variance_from_expected,
                    message=f"Budget {budget.name} shows unusual spending pattern with {variance_percentage:.1f}% variance from expected",
                    details={
                        'variance_from_expected': snapshot.variance_from_expected,
                        'variance_percentage': variance_percentage,
                        'is_on_track': snapshot.is_on_track
                    }
                )
                
                await self._trigger_alert(alert, budget)
    
    async def _trigger_alert(self, alert: BudgetAlert, budget: Budget):
        """Trigger a new alert"""
        try:
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            self.logger.info(f"Triggered {alert.severity.value} alert: {alert.message}")
            
            # Send notifications
            await self._send_alert_notifications(alert, budget)
            
            # Mark notification as sent
            alert.notification_sent = True
            
        except Exception as e:
            self.logger.error(f"Error triggering alert {alert.alert_id}: {e}")  
  
    async def _send_alert_notifications(self, alert: BudgetAlert, budget: Budget):
        """Send notifications for an alert"""
        try:
            # Send to configured notification channels
            for channel in budget.notification_channels:
                recipients = budget.notification_recipients.get(channel, [])
                if recipients:
                    success = await self.notification_service.send_notification(
                        alert, budget, channel, recipients
                    )
                    if not success:
                        self.logger.error(f"Failed to send {channel.value} notification for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending alert notifications: {e}")
    
    async def _process_escalations(self):
        """Process alert escalations"""
        try:
            for alert in list(self.active_alerts.values()):
                for rule in self.escalation_rules.values():
                    if rule.should_escalate(alert):
                        await self._escalate_alert(alert, rule)
                        
        except Exception as e:
            self.logger.error(f"Error processing escalations: {e}")
    
    async def _escalate_alert(self, alert: BudgetAlert, rule: EscalationRule):
        """Escalate an alert"""
        try:
            alert.escalation_level += 1
            
            self.logger.warning(f"Escalating alert {alert.alert_id} to level {alert.escalation_level}")
            
            # Send escalation notifications
            for channel in rule.target_channels:
                recipients = rule.target_recipients.get(channel, [])
                if recipients:
                    # Create escalation message
                    escalation_alert = BudgetAlert(
                        alert_id=f"{alert.alert_id}_escalation_{alert.escalation_level}",
                        budget_id=alert.budget_id,
                        alert_type=f"{alert.alert_type}_escalation",
                        severity=alert.severity,
                        current_value=alert.current_value,
                        threshold_value=alert.threshold_value,
                        message=f"ESCALATION LEVEL {alert.escalation_level}: {alert.message}",
                        details=alert.details.copy()
                    )
                    
                    # Get budget for notification
                    # In production, this would fetch from budget store
                    budget = Budget(
                        budget_id=alert.budget_id,
                        name=f"Budget {alert.budget_id}",
                        description="",
                        amount=alert.details.get('budget_amount', 0),
                        currency="USD",
                        period=BudgetPeriod.MONTHLY,
                        start_date=datetime.now(),
                        end_date=datetime.now() + timedelta(days=30),
                        dimension_filters=[]
                    )
                    
                    await self.notification_service.send_notification(
                        escalation_alert, budget, channel, recipients
                    )
                    
        except Exception as e:
            self.logger.error(f"Error escalating alert {alert.alert_id}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledge(acknowledged_by)
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False    

    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve(resolved_by)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
        return False
    
    def suppress_alerts(self, budget_id: str, duration_minutes: int):
        """Suppress alerts for a budget for a specified duration"""
        suppression_end = datetime.now() + timedelta(minutes=duration_minutes)
        self.alert_suppression[budget_id] = suppression_end
        self.logger.info(f"Suppressed alerts for budget {budget_id} until {suppression_end}")
    
    def _is_alert_suppressed(self, budget_id: str) -> bool:
        """Check if alerts are suppressed for a budget"""
        if budget_id in self.alert_suppression:
            if datetime.now() < self.alert_suppression[budget_id]:
                return True
            else:
                # Suppression expired, remove it
                del self.alert_suppression[budget_id]
        return False
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts from history"""
        # Keep only last 1000 alerts in history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def get_active_alerts(self, budget_id: Optional[str] = None) -> List[BudgetAlert]:
        """Get active alerts, optionally filtered by budget"""
        alerts = list(self.active_alerts.values())
        if budget_id:
            alerts = [alert for alert in alerts if alert.budget_id == budget_id]
        return alerts
    
    def get_alert_history(self, 
                         budget_id: Optional[str] = None,
                         hours: int = 24) -> List[BudgetAlert]:
        """Get alert history, optionally filtered by budget and time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        alerts = [alert for alert in self.alert_history if alert.triggered_at >= cutoff_time]
        
        if budget_id:
            alerts = [alert for alert in alerts if alert.budget_id == budget_id]
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        active_count = len(self.active_alerts)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            type_counts[alert.alert_type] += 1
        
        # Calculate average resolution time for resolved alerts
        resolved_alerts = [alert for alert in self.alert_history if alert.resolved_at]
        if resolved_alerts:
            resolution_times = [alert.duration_minutes for alert in resolved_alerts]
            avg_resolution_time = statistics.mean(resolution_times)
        else:
            avg_resolution_time = 0
        
        return {
            'active_alerts': active_count,
            'total_alerts_24h': len(self.get_alert_history(hours=24)),
            'severity_breakdown': dict(severity_counts),
            'type_breakdown': dict(type_counts),
            'average_resolution_time_minutes': avg_resolution_time,
            'escalated_alerts': len([a for a in self.active_alerts.values() if a.escalation_level > 0])
        }


# Simple budget creator for testing
class BudgetCreator:
    """Simple budget creator for testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BudgetCreator")
        self.budgets: Dict[str, Budget] = {}
    
    def create_budget_from_template(self, 
                                  template_id: str,
                                  budget_name: str,
                                  dimension_values: Dict[BudgetDimension, List[str]],
                                  amount: Optional[float] = None,
                                  created_by: Optional[str] = None) -> Budget:
        """Create a budget from a template"""
        
        # Create dimension filters from provided values
        dimension_filters = []
        for dimension, values in dimension_values.items():
            dimension_filters.append(BudgetDimensionFilter(
                dimension=dimension,
                values=values,
                include=True
            ))
        
        # Set dates
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=30)  # Default to monthly
        
        budget = Budget(
            budget_id=str(uuid.uuid4()),
            name=budget_name,
            description=f"Budget created from template: {template_id}",
            amount=amount or 10000.0,
            currency="USD",
            period=BudgetPeriod.MONTHLY,
            start_date=start_date,
            end_date=end_date,
            dimension_filters=dimension_filters,
            alert_thresholds=[50.0, 75.0, 90.0, 100.0],
            created_by=created_by
        )
        
        self.budgets[budget.budget_id] = budget
        self.logger.info(f"Created budget {budget.name} from template {template_id}")
        
        return budget
    
    def create_default_templates(self):
        """Create default templates (placeholder for compatibility)"""
        self.logger.info("Default templates created")


if __name__ == "__main__":
    print("Budget Management System with AlertManager loaded successfully!")


@dataclass
class BudgetVarianceReport:
    """Budget variance analysis report"""
    budget_id: str
    budget_name: str
    report_period_start: datetime
    report_period_end: datetime
    budget_amount: float
    actual_spend: float
    variance_amount: float
    variance_percentage: float
    variance_type: str  # "over", "under", "on_target"
    trend_analysis: Dict[str, Any]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_over_budget(self) -> bool:
        """Check if spending is over budget"""
        return self.variance_amount > 0
    
    @property
    def utilization_percentage(self) -> float:
        """Calculate budget utilization percentage"""
        return (self.actual_spend / self.budget_amount) * 100 if self.budget_amount > 0 else 0


class BudgetVarianceAnalyzer:
    """Budget variance analysis and reporting system"""
    
    def __init__(self, spending_tracker: SpendingTracker):
        self.logger = logging.getLogger(f"{__name__}.BudgetVarianceAnalyzer")
        self.spending_tracker = spending_tracker
        self.variance_reports: Dict[str, List[BudgetVarianceReport]] = defaultdict(list)
    
    def generate_variance_report(self, budget: Budget, 
                               report_period_start: Optional[datetime] = None,
                               report_period_end: Optional[datetime] = None) -> BudgetVarianceReport:
        """Generate comprehensive variance analysis report for a budget"""
        
        # Set default report period if not provided
        if report_period_start is None:
            report_period_start = budget.start_date
        if report_period_end is None:
            report_period_end = min(datetime.now(), budget.end_date)
        
        # Get current spending data
        current_snapshot = self.spending_tracker.get_current_spending_status(budget)
        actual_spend = current_snapshot.current_spend if current_snapshot else 0.0
        
        # Calculate expected spend for the period
        total_budget_days = (budget.end_date - budget.start_date).days
        elapsed_days = (report_period_end - report_period_start).days
        expected_spend = budget.amount * (elapsed_days / total_budget_days) if total_budget_days > 0 else 0
        
        # Calculate variance
        variance_amount = actual_spend - expected_spend
        variance_percentage = (variance_amount / expected_spend) * 100 if expected_spend > 0 else 0
        
        # Determine variance type
        if abs(variance_percentage) <= 5:  # Within 5% tolerance
            variance_type = "on_target"
        elif variance_amount > 0:
            variance_type = "over"
        else:
            variance_type = "under"
        
        # Perform trend analysis
        trend_analysis = self._analyze_spending_trends(budget, report_period_start, report_period_end)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(budget, actual_spend, expected_spend)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(budget, variance_amount, variance_percentage, trend_analysis)
        
        # Create variance report
        report = BudgetVarianceReport(
            budget_id=budget.budget_id,
            budget_name=budget.name,
            report_period_start=report_period_start,
            report_period_end=report_period_end,
            budget_amount=expected_spend,
            actual_spend=actual_spend,
            variance_amount=variance_amount,
            variance_percentage=variance_percentage,
            variance_type=variance_type,
            trend_analysis=trend_analysis,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
        
        # Store report
        self.variance_reports[budget.budget_id].append(report)
        
        # Keep only last 12 reports per budget
        if len(self.variance_reports[budget.budget_id]) > 12:
            self.variance_reports[budget.budget_id] = self.variance_reports[budget.budget_id][-12:]
        
        self.logger.info(f"Generated variance report for budget {budget.name}: {variance_type} budget by {variance_percentage:.1f}%")
        
        return report 
   
    def _analyze_spending_trends(self, budget: Budget, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze spending trends over the report period"""
        
        snapshots = self.spending_tracker.spending_snapshots.get(budget.budget_id, [])
        
        # Filter snapshots to report period
        period_snapshots = [
            s for s in snapshots 
            if start_date <= s.snapshot_time <= end_date
        ]
        
        if len(period_snapshots) < 2:
            return {
                'trend_direction': 'insufficient_data',
                'trend_strength': 0.0,
                'average_daily_spend': 0.0,
                'spending_acceleration': 0.0,
                'volatility': 0.0
            }
        
        # Calculate daily spending amounts
        daily_spends = []
        for i in range(1, len(period_snapshots)):
            days_diff = (period_snapshots[i].snapshot_time - period_snapshots[i-1].snapshot_time).days
            spend_diff = period_snapshots[i].current_spend - period_snapshots[i-1].current_spend
            daily_spend = spend_diff / max(days_diff, 1)
            daily_spends.append(max(0, daily_spend))
        
        if not daily_spends:
            return {
                'trend_direction': 'insufficient_data',
                'trend_strength': 0.0,
                'average_daily_spend': 0.0,
                'spending_acceleration': 0.0,
                'volatility': 0.0
            }
        
        # Calculate trend metrics
        average_daily_spend = statistics.mean(daily_spends)
        
        # Simple trend analysis using first and last third
        if len(daily_spends) >= 6:
            first_third = daily_spends[:len(daily_spends)//3]
            last_third = daily_spends[-len(daily_spends)//3:]
            
            avg_first = statistics.mean(first_third)
            avg_last = statistics.mean(last_third)
            
            if avg_last > avg_first * 1.1:
                trend_direction = "increasing"
                trend_strength = (avg_last - avg_first) / avg_first
            elif avg_last < avg_first * 0.9:
                trend_direction = "decreasing"
                trend_strength = (avg_first - avg_last) / avg_first
            else:
                trend_direction = "stable"
                trend_strength = 0.1
        else:
            trend_direction = "stable"
            trend_strength = 0.1
        
        # Calculate spending acceleration (change in velocity)
        spending_acceleration = 0.0
        if len(daily_spends) >= 4:
            first_half = statistics.mean(daily_spends[:len(daily_spends)//2])
            second_half = statistics.mean(daily_spends[len(daily_spends)//2:])
            spending_acceleration = (second_half - first_half) / max(first_half, 1)
        
        # Calculate volatility (coefficient of variation)
        volatility = statistics.stdev(daily_spends) / average_daily_spend if average_daily_spend > 0 else 0
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'average_daily_spend': average_daily_spend,
            'spending_acceleration': spending_acceleration,
            'volatility': volatility,
            'data_points': len(daily_spends)
        }
    
    def _calculate_performance_metrics(self, budget: Budget, actual_spend: float, expected_spend: float) -> Dict[str, float]:
        """Calculate budget performance metrics"""
        
        # Basic performance metrics
        utilization_rate = (actual_spend / budget.amount) * 100 if budget.amount > 0 else 0
        efficiency_score = (expected_spend / actual_spend) * 100 if actual_spend > 0 else 100
        
        # Time-based metrics
        days_elapsed = (datetime.now() - budget.start_date).days
        total_days = (budget.end_date - budget.start_date).days
        time_progress = (days_elapsed / total_days) * 100 if total_days > 0 else 0
        
        # Spending velocity metrics
        current_snapshot = self.spending_tracker.get_current_spending_status(budget)
        spending_velocity = current_snapshot.spending_velocity if current_snapshot else 0
        expected_velocity = budget.amount / total_days if total_days > 0 else 0
        velocity_ratio = spending_velocity / expected_velocity if expected_velocity > 0 else 0
        
        # Forecast accuracy (if we have projections)
        forecast_accuracy = 100.0  # Default to perfect if no data
        if current_snapshot and current_snapshot.projected_spend > 0:
            forecast_error = abs(current_snapshot.projected_spend - budget.amount) / budget.amount
            forecast_accuracy = max(0, (1 - forecast_error) * 100)
        
        return {
            'utilization_rate': utilization_rate,
            'efficiency_score': min(efficiency_score, 200),  # Cap at 200%
            'time_progress': time_progress,
            'velocity_ratio': velocity_ratio,
            'forecast_accuracy': forecast_accuracy,
            'budget_health_score': self._calculate_budget_health_score(utilization_rate, efficiency_score, velocity_ratio)
        }
    
    def _calculate_budget_health_score(self, utilization_rate: float, efficiency_score: float, velocity_ratio: float) -> float:
        """Calculate overall budget health score (0-100)"""
        
        # Utilization score (optimal around 80-95%)
        if 80 <= utilization_rate <= 95:
            utilization_score = 100
        elif utilization_rate < 80:
            utilization_score = utilization_rate / 80 * 100
        else:  # Over 95%
            utilization_score = max(0, 100 - (utilization_rate - 95) * 5)
        
        # Efficiency score (higher is better, but cap at 100)
        efficiency_score = min(efficiency_score, 100)
        
        # Velocity score (optimal around 0.8-1.2)
        if 0.8 <= velocity_ratio <= 1.2:
            velocity_score = 100
        elif velocity_ratio < 0.8:
            velocity_score = velocity_ratio / 0.8 * 100
        else:  # Over 1.2
            velocity_score = max(0, 100 - (velocity_ratio - 1.2) * 50)
        
        # Weighted average
        health_score = (utilization_score * 0.4 + efficiency_score * 0.3 + velocity_score * 0.3)
        return round(health_score, 1)    

    def _generate_recommendations(self, budget: Budget, variance_amount: float, 
                                variance_percentage: float, trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on variance analysis"""
        
        recommendations = []
        
        # Variance-based recommendations
        if variance_percentage > 20:  # Significantly over budget
            recommendations.append("🚨 URGENT: Spending is significantly over budget. Implement immediate cost controls.")
            recommendations.append("Review and pause non-essential expenses until spending is back on track.")
            recommendations.append("Consider reallocating budget from other areas or requesting additional funding.")
        elif variance_percentage > 10:  # Moderately over budget
            recommendations.append("⚠️ WARNING: Spending is above expected levels. Monitor closely and implement cost controls.")
            recommendations.append("Review recent expenses to identify areas for optimization.")
        elif variance_percentage < -20:  # Significantly under budget
            recommendations.append("💡 OPPORTUNITY: Spending is well below budget. Consider accelerating planned initiatives.")
            recommendations.append("Evaluate if budget allocation is appropriate or if funds can be reallocated.")
        elif variance_percentage < -10:  # Moderately under budget
            recommendations.append("✅ GOOD: Spending is below budget. Maintain current cost discipline.")
        else:  # On target
            recommendations.append("✅ EXCELLENT: Spending is on target. Continue current budget management practices.")
        
        # Trend-based recommendations
        trend_direction = trend_analysis.get('trend_direction', 'stable')
        trend_strength = trend_analysis.get('trend_strength', 0)
        
        if trend_direction == "increasing" and trend_strength > 0.3:
            recommendations.append("📈 TREND ALERT: Spending is accelerating. Investigate causes and implement controls.")
            recommendations.append("Review spending patterns to identify drivers of increased costs.")
        elif trend_direction == "decreasing" and trend_strength > 0.3:
            recommendations.append("📉 TREND: Spending is decelerating. Ensure this aligns with business objectives.")
        
        # Volatility-based recommendations
        volatility = trend_analysis.get('volatility', 0)
        if volatility > 0.5:
            recommendations.append("🎯 STABILITY: High spending volatility detected. Work to stabilize spending patterns.")
            recommendations.append("Implement more predictable spending schedules and better expense planning.")
        
        # Velocity-based recommendations
        current_snapshot = self.spending_tracker.get_current_spending_status(budget)
        if current_snapshot:
            if current_snapshot.spending_velocity > 0:
                days_remaining = current_snapshot.days_remaining
                projected_overage = current_snapshot.get_projected_overage(budget.amount)
                
                if projected_overage > budget.amount * 0.1:  # Projected to exceed by 10%
                    recommendations.append(f"🔮 FORECAST: Current spending pace will exceed budget by ${projected_overage:,.2f}.")
                    recommendations.append("Reduce spending velocity or request budget increase.")
        
        # Seasonal recommendations
        current_month = datetime.now().month
        if current_month in [11, 12]:  # End of year
            recommendations.append("📅 YEAR-END: Review budget utilization for year-end planning and next year's budget.")
        elif current_month in [3, 6, 9]:  # Quarter ends
            recommendations.append("📊 QUARTER-END: Prepare quarterly budget review and variance analysis.")
        
        # Ensure we have at least one recommendation
        if not recommendations:
            recommendations.append("📋 Continue monitoring budget performance and maintain current spending discipline.")
        
        return recommendations
    
    def generate_automated_budget_report(self, budget: Budget) -> Dict[str, Any]:
        """Generate automated budget performance report"""
        
        # Generate variance report
        variance_report = self.generate_variance_report(budget)
        
        # Get historical reports for comparison
        historical_reports = self.variance_reports.get(budget.budget_id, [])
        
        # Calculate month-over-month changes if available
        mom_changes = {}
        if len(historical_reports) >= 2:
            current_report = historical_reports[-1]
            previous_report = historical_reports[-2]
            
            mom_changes = {
                'variance_change': current_report.variance_percentage - previous_report.variance_percentage,
                'spend_change': current_report.actual_spend - previous_report.actual_spend,
                'utilization_change': current_report.utilization_percentage - previous_report.utilization_percentage
            }
        
        # Create comprehensive report
        automated_report = {
            'budget_info': {
                'budget_id': budget.budget_id,
                'budget_name': budget.name,
                'budget_amount': budget.amount,
                'period': budget.period.value,
                'start_date': budget.start_date.isoformat(),
                'end_date': budget.end_date.isoformat()
            },
            'current_performance': {
                'actual_spend': variance_report.actual_spend,
                'budget_utilization': variance_report.utilization_percentage,
                'variance_amount': variance_report.variance_amount,
                'variance_percentage': variance_report.variance_percentage,
                'variance_type': variance_report.variance_type,
                'performance_metrics': variance_report.performance_metrics
            },
            'trend_analysis': variance_report.trend_analysis,
            'month_over_month': mom_changes,
            'recommendations': variance_report.recommendations,
            'report_metadata': {
                'generated_at': variance_report.generated_at.isoformat(),
                'report_period': f"{variance_report.report_period_start.date()} to {variance_report.report_period_end.date()}",
                'data_quality': 'high' if variance_report.trend_analysis.get('data_points', 0) > 5 else 'medium'
            }
        }
        
        self.logger.info(f"Generated automated budget report for {budget.name}")
        
        return automated_report
    
    def get_budget_optimization_recommendations(self, budget: Budget) -> List[Dict[str, Any]]:
        """Generate budget optimization recommendations based on historical performance"""
        
        historical_reports = self.variance_reports.get(budget.budget_id, [])
        
        if len(historical_reports) < 3:
            return [{
                'type': 'data_collection',
                'priority': 'low',
                'title': 'Collect More Data',
                'description': 'Need more historical data to provide meaningful optimization recommendations.',
                'action': 'Continue monitoring budget performance for at least 3 reporting periods.'
            }]
        
        recommendations = []
        
        # Analyze historical variance patterns
        variances = [report.variance_percentage for report in historical_reports[-6:]]  # Last 6 reports
        avg_variance = statistics.mean(variances)
        variance_consistency = statistics.stdev(variances) if len(variances) > 1 else 0
        
        # Budget sizing recommendations
        if avg_variance > 15:  # Consistently over budget
            recommendations.append({
                'type': 'budget_increase',
                'priority': 'high',
                'title': 'Consider Budget Increase',
                'description': f'Budget has been consistently over by {avg_variance:.1f}% on average.',
                'action': f'Consider increasing budget by {max(10, avg_variance):.0f}% for next period.',
                'estimated_impact': f'Reduce variance by {avg_variance:.1f}%'
            })
        elif avg_variance < -15:  # Consistently under budget
            recommendations.append({
                'type': 'budget_reallocation',
                'priority': 'medium',
                'title': 'Budget Reallocation Opportunity',
                'description': f'Budget has been consistently under by {abs(avg_variance):.1f}% on average.',
                'action': f'Consider reallocating {abs(avg_variance):.0f}% of budget to other areas.',
                'estimated_impact': f'Free up ${budget.amount * abs(avg_variance) / 100:,.2f} for other initiatives'
            })
        
        # Spending pattern recommendations
        if variance_consistency > 20:  # High volatility
            recommendations.append({
                'type': 'spending_stabilization',
                'priority': 'medium',
                'title': 'Stabilize Spending Patterns',
                'description': f'Spending variance is highly volatile (σ={variance_consistency:.1f}%).',
                'action': 'Implement more predictable spending schedules and better expense planning.',
                'estimated_impact': 'Reduce spending volatility by 30-50%'
            })
        
        # Performance-based recommendations
        latest_report = historical_reports[-1]
        health_score = latest_report.performance_metrics.get('budget_health_score', 50)
        
        if health_score < 60:
            recommendations.append({
                'type': 'performance_improvement',
                'priority': 'high',
                'title': 'Improve Budget Performance',
                'description': f'Budget health score is low ({health_score:.1f}/100).',
                'action': 'Focus on improving spending efficiency and velocity management.',
                'estimated_impact': f'Potential to improve health score to 75+'
            })
        
        return recommendations