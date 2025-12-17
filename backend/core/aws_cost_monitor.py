"""
AWS Cost Monitoring & Alerting Engine

Real-time cost monitoring, anomaly detection, and proactive alerting
to prevent cost surprises and enable immediate action.
"""

import boto3
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for systems with email import issues
    MimeText = None
    MimeMultipart = None
import logging
import statistics
import requests

logger = logging.getLogger(__name__)

@dataclass
class CostAlert:
    """Represents a cost alert"""
    alert_id: str
    alert_type: str  # 'budget_threshold', 'anomaly', 'spike', 'daily_summary'
    severity: str  # 'low', 'medium', 'high', 'critical'
    title: str
    description: str
    current_cost: float
    threshold_cost: Optional[float]
    percentage_change: Optional[float]
    service_affected: Optional[str]
    recommended_actions: List[str]
    created_at: datetime
    resolved: bool = False

@dataclass
class BudgetThreshold:
    """Budget threshold configuration"""
    name: str
    monthly_budget: float
    warning_threshold: float  # Percentage (e.g., 80.0 for 80%)
    critical_threshold: float  # Percentage (e.g., 95.0 for 95%)
    services: List[str]  # Empty list means all services
    enabled: bool = True

@dataclass
class CostAnomaly:
    """Detected cost anomaly"""
    service: str
    current_cost: float
    expected_cost: float
    deviation_percentage: float
    confidence_score: float
    detection_date: datetime

@dataclass
class DailyCostSummary:
    """Daily cost summary"""
    date: str
    total_cost: float
    cost_change: float
    cost_change_percentage: float
    top_services: List[Dict[str, Any]]
    alerts_count: int
    optimization_opportunities: int

class AWSCostMonitor:
    """
    AWS Cost Monitoring Engine
    
    Provides real-time cost monitoring, anomaly detection, and proactive alerting
    to help users stay on top of their AWS spending.
    """
    
    def __init__(self, aws_access_key_id: str = None, aws_secret_access_key: str = None, region: str = 'us-east-1'):
        """Initialize AWS Cost Monitor"""
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region
                )
            else:
                self.session = boto3.Session(region_name=region)
            
            self.cost_explorer = self.session.client('ce')
            self.cloudwatch = self.session.client('cloudwatch')
            self.budgets = self.session.client('budgets')
            
            # Alert storage (in production, use database)
            self.active_alerts: List[CostAlert] = []
            self.budget_thresholds: List[BudgetThreshold] = []
            
            logger.info("AWS Cost Monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS Cost Monitor: {str(e)}")
            raise
    
    def add_budget_threshold(self, threshold: BudgetThreshold) -> None:
        """Add a budget threshold for monitoring"""
        self.budget_thresholds.append(threshold)
        logger.info(f"Added budget threshold: {threshold.name} - ${threshold.monthly_budget}")
    
    def check_budget_thresholds(self) -> List[CostAlert]:
        """Check all budget thresholds and generate alerts"""
        alerts = []
        
        try:
            # Get current month costs
            current_month_costs = self._get_current_month_costs()
            
            for threshold in self.budget_thresholds:
                if not threshold.enabled:
                    continue
                
                # Calculate relevant costs
                if threshold.services:
                    # Filter by specific services
                    relevant_cost = sum(
                        cost for service, cost in current_month_costs.items()
                        if service in threshold.services
                    )
                else:
                    # All services
                    relevant_cost = sum(current_month_costs.values())
                
                # Check thresholds
                usage_percentage = (relevant_cost / threshold.monthly_budget) * 100
                
                if usage_percentage >= threshold.critical_threshold:
                    alert = CostAlert(
                        alert_id=f"budget_critical_{threshold.name}_{datetime.now().strftime('%Y%m%d')}",
                        alert_type='budget_threshold',
                        severity='critical',
                        title=f"🚨 Critical Budget Alert: {threshold.name}",
                        description=f"Budget usage at {usage_percentage:.1f}% (${relevant_cost:.2f} of ${threshold.monthly_budget:.2f})",
                        current_cost=relevant_cost,
                        threshold_cost=threshold.monthly_budget,
                        percentage_change=usage_percentage,
                        service_affected=', '.join(threshold.services) if threshold.services else 'All Services',
                        recommended_actions=[
                            "Review and pause non-essential resources immediately",
                            "Check for cost optimization opportunities",
                            "Consider increasing budget or implementing cost controls",
                            "Review recent cost spikes in AWS Cost Explorer"
                        ],
                        created_at=datetime.now()
                    )
                    alerts.append(alert)
                    
                elif usage_percentage >= threshold.warning_threshold:
                    alert = CostAlert(
                        alert_id=f"budget_warning_{threshold.name}_{datetime.now().strftime('%Y%m%d')}",
                        alert_type='budget_threshold',
                        severity='high',
                        title=f"⚠️ Budget Warning: {threshold.name}",
                        description=f"Budget usage at {usage_percentage:.1f}% (${relevant_cost:.2f} of ${threshold.monthly_budget:.2f})",
                        current_cost=relevant_cost,
                        threshold_cost=threshold.monthly_budget,
                        percentage_change=usage_percentage,
                        service_affected=', '.join(threshold.services) if threshold.services else 'All Services',
                        recommended_actions=[
                            "Monitor spending closely for remainder of month",
                            "Review cost optimization recommendations",
                            "Consider implementing cost controls",
                            "Plan for potential budget adjustments"
                        ],
                        created_at=datetime.now()
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to check budget thresholds: {str(e)}")
            return []
    
    def detect_cost_anomalies(self, days_back: int = 14) -> List[CostAnomaly]:
        """Detect cost anomalies using statistical analysis"""
        anomalies = []
        
        try:
            # Get historical cost data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            # Get daily costs by service
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            # Analyze each service for anomalies
            service_costs = {}
            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    
                    if service not in service_costs:
                        service_costs[service] = []
                    service_costs[service].append(cost)
            
            # Detect anomalies using statistical methods
            for service, costs in service_costs.items():
                if len(costs) < 7:  # Need at least a week of data
                    continue
                
                # Calculate baseline statistics (exclude last 2 days)
                baseline_costs = costs[:-2]
                recent_costs = costs[-2:]
                
                if len(baseline_costs) < 5:
                    continue
                
                baseline_mean = statistics.mean(baseline_costs)
                baseline_stdev = statistics.stdev(baseline_costs) if len(baseline_costs) > 1 else 0
                
                # Check recent costs for anomalies
                for recent_cost in recent_costs:
                    if baseline_stdev > 0:
                        z_score = abs(recent_cost - baseline_mean) / baseline_stdev
                        
                        # Anomaly if z-score > 2 (95% confidence) and cost increase > 20%
                        if z_score > 2 and recent_cost > baseline_mean * 1.2:
                            deviation_percentage = ((recent_cost - baseline_mean) / baseline_mean) * 100
                            confidence_score = min(z_score / 3.0, 1.0)  # Normalize to 0-1
                            
                            anomaly = CostAnomaly(
                                service=service,
                                current_cost=recent_cost,
                                expected_cost=baseline_mean,
                                deviation_percentage=deviation_percentage,
                                confidence_score=confidence_score,
                                detection_date=datetime.now()
                            )
                            anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Failed to detect cost anomalies: {str(e)}")
            return []
    
    def generate_anomaly_alerts(self, anomalies: List[CostAnomaly]) -> List[CostAlert]:
        """Generate alerts from detected anomalies"""
        alerts = []
        
        for anomaly in anomalies:
            severity = 'medium'
            if anomaly.deviation_percentage > 100:  # 100%+ increase
                severity = 'high'
            elif anomaly.deviation_percentage > 200:  # 200%+ increase
                severity = 'critical'
            
            alert = CostAlert(
                alert_id=f"anomaly_{anomaly.service}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type='anomaly',
                severity=severity,
                title=f"🔍 Cost Anomaly Detected: {anomaly.service}",
                description=f"Unusual spending detected - {anomaly.deviation_percentage:.1f}% above normal (${anomaly.current_cost:.2f} vs expected ${anomaly.expected_cost:.2f})",
                current_cost=anomaly.current_cost,
                threshold_cost=anomaly.expected_cost,
                percentage_change=anomaly.deviation_percentage,
                service_affected=anomaly.service,
                recommended_actions=[
                    f"Investigate recent changes in {anomaly.service}",
                    "Check for new resources or increased usage",
                    "Review CloudTrail logs for unusual activity",
                    "Consider cost optimization for this service"
                ],
                created_at=datetime.now()
            )
            alerts.append(alert)
        
        return alerts
    
    def detect_cost_spikes(self, spike_threshold: float = 50.0) -> List[CostAlert]:
        """Detect sudden cost spikes (day-over-day increases)"""
        alerts = []
        
        try:
            # Get last 3 days of costs
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=3)
            
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            # Analyze day-over-day changes
            if len(response['ResultsByTime']) >= 2:
                yesterday_costs = {}
                today_costs = {}
                
                # Parse yesterday's costs
                if len(response['ResultsByTime']) > 1:
                    for group in response['ResultsByTime'][-2]['Groups']:
                        service = group['Keys'][0]
                        cost = float(group['Metrics']['BlendedCost']['Amount'])
                        yesterday_costs[service] = cost
                
                # Parse today's costs
                for group in response['ResultsByTime'][-1]['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    today_costs[service] = cost
                
                # Check for spikes
                for service, today_cost in today_costs.items():
                    yesterday_cost = yesterday_costs.get(service, 0)
                    
                    if yesterday_cost > 0:
                        change_percentage = ((today_cost - yesterday_cost) / yesterday_cost) * 100
                        
                        if change_percentage >= spike_threshold and today_cost > 10:  # Minimum $10 to avoid noise
                            severity = 'medium'
                            if change_percentage > 100:
                                severity = 'high'
                            elif change_percentage > 200:
                                severity = 'critical'
                            
                            alert = CostAlert(
                                alert_id=f"spike_{service}_{datetime.now().strftime('%Y%m%d')}",
                                alert_type='spike',
                                severity=severity,
                                title=f"📈 Cost Spike Alert: {service}",
                                description=f"Daily cost increased by {change_percentage:.1f}% (${yesterday_cost:.2f} → ${today_cost:.2f})",
                                current_cost=today_cost,
                                threshold_cost=yesterday_cost,
                                percentage_change=change_percentage,
                                service_affected=service,
                                recommended_actions=[
                                    f"Investigate what changed in {service} today",
                                    "Check for new resource launches",
                                    "Review usage patterns and scaling events",
                                    "Consider immediate cost controls if unplanned"
                                ],
                                created_at=datetime.now()
                            )
                            alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to detect cost spikes: {str(e)}")
            return []
    
    def generate_daily_summary(self) -> DailyCostSummary:
        """Generate daily cost summary"""
        try:
            # Get yesterday and today costs
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            # Get costs for both days
            today_costs = self._get_daily_costs(today)
            yesterday_costs = self._get_daily_costs(yesterday)
            
            total_today = sum(today_costs.values())
            total_yesterday = sum(yesterday_costs.values())
            
            cost_change = total_today - total_yesterday
            cost_change_percentage = (cost_change / total_yesterday * 100) if total_yesterday > 0 else 0
            
            # Get top services
            top_services = [
                {'service': service, 'cost': cost, 'percentage': (cost / total_today * 100) if total_today > 0 else 0}
                for service, cost in sorted(today_costs.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
            
            # Count active alerts
            alerts_count = len([alert for alert in self.active_alerts if not alert.resolved])
            
            return DailyCostSummary(
                date=today.strftime('%Y-%m-%d'),
                total_cost=total_today,
                cost_change=cost_change,
                cost_change_percentage=cost_change_percentage,
                top_services=top_services,
                alerts_count=alerts_count,
                optimization_opportunities=0  # Would integrate with cost analyzer
            )
            
        except Exception as e:
            logger.error(f"Failed to generate daily summary: {str(e)}")
            return DailyCostSummary(
                date=datetime.now().date().strftime('%Y-%m-%d'),
                total_cost=0,
                cost_change=0,
                cost_change_percentage=0,
                top_services=[],
                alerts_count=0,
                optimization_opportunities=0
            )
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run complete monitoring cycle and return results"""
        try:
            logger.info("Starting cost monitoring cycle")
            
            # Clear resolved alerts older than 24 hours
            self._cleanup_old_alerts()
            
            # Check budget thresholds
            budget_alerts = self.check_budget_thresholds()
            
            # Detect anomalies
            anomalies = self.detect_cost_anomalies()
            anomaly_alerts = self.generate_anomaly_alerts(anomalies)
            
            # Detect cost spikes
            spike_alerts = self.detect_cost_spikes()
            
            # Combine all alerts
            new_alerts = budget_alerts + anomaly_alerts + spike_alerts
            
            # Add to active alerts (avoid duplicates)
            for alert in new_alerts:
                if not any(existing.alert_id == alert.alert_id for existing in self.active_alerts):
                    self.active_alerts.append(alert)
            
            # Generate daily summary
            daily_summary = self.generate_daily_summary()
            
            # Prepare results
            results = {
                'monitoring_date': datetime.now().isoformat(),
                'new_alerts_count': len(new_alerts),
                'total_active_alerts': len([a for a in self.active_alerts if not a.resolved]),
                'budget_alerts': len(budget_alerts),
                'anomaly_alerts': len(anomaly_alerts),
                'spike_alerts': len(spike_alerts),
                'daily_summary': daily_summary,
                'new_alerts': new_alerts,
                'active_alerts': [a for a in self.active_alerts if not a.resolved]
            }
            
            logger.info(f"Monitoring cycle completed: {len(new_alerts)} new alerts, {len(self.active_alerts)} total active")
            return results
            
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {str(e)}")
            return {
                'monitoring_date': datetime.now().isoformat(),
                'error': str(e),
                'new_alerts_count': 0,
                'total_active_alerts': 0
            }
    
    def send_alert_notification(self, alert: CostAlert, notification_config: Dict[str, Any]) -> bool:
        """Send alert notification via email or Slack"""
        try:
            if notification_config.get('email_enabled'):
                return self._send_email_alert(alert, notification_config['email'])
            
            if notification_config.get('slack_enabled'):
                return self._send_slack_alert(alert, notification_config['slack'])
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {str(e)}")
            return False
    
    def _get_current_month_costs(self) -> Dict[str, float]:
        """Get current month costs by service"""
        try:
            # Get first day of current month
            today = datetime.now().date()
            first_day = today.replace(day=1)
            
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': first_day.strftime('%Y-%m-%d'),
                    'End': today.strftime('%Y-%m-%d')
                },
                Granularity='MONTHLY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            costs = {}
            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    costs[service] = cost
            
            return costs
            
        except Exception as e:
            logger.error(f"Failed to get current month costs: {str(e)}")
            return {}
    
    def _get_daily_costs(self, date: datetime.date) -> Dict[str, float]:
        """Get costs for a specific day by service"""
        try:
            next_day = date + timedelta(days=1)
            
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': date.strftime('%Y-%m-%d'),
                    'End': next_day.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            costs = {}
            if response['ResultsByTime']:
                for group in response['ResultsByTime'][0]['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    costs[service] = cost
            
            return costs
            
        except Exception as e:
            logger.error(f"Failed to get daily costs for {date}: {str(e)}")
            return {}
    
    def _cleanup_old_alerts(self):
        """Remove resolved alerts older than 24 hours"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if not alert.resolved or alert.created_at > cutoff_time
        ]
    
    def _send_email_alert(self, alert: CostAlert, email_config: Dict[str, Any]) -> bool:
        """Send email alert notification"""
        try:
            if MimeMultipart is None or MimeText is None:
                logger.warning("Email functionality not available - missing email libraries")
                return False
                
            # Create email message
            msg = MimeMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"AWS Cost Alert: {alert.title}"
            
            # Create email body
            body = f"""
            {alert.title}
            
            {alert.description}
            
            Details:
            - Alert Type: {alert.alert_type}
            - Severity: {alert.severity.upper()}
            - Current Cost: ${alert.current_cost:.2f}
            - Service: {alert.service_affected or 'Multiple'}
            - Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            
            Recommended Actions:
            {chr(10).join(f'• {action}' for action in alert.recommended_actions)}
            
            ---
            This alert was generated by your FinOps Platform cost monitoring system.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('use_tls'):
                server.starttls()
            if email_config.get('username'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent successfully: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def _send_slack_alert(self, alert: CostAlert, slack_config: Dict[str, Any]) -> bool:
        """Send Slack alert notification"""
        try:
            # Determine emoji based on severity
            emoji_map = {
                'low': '💡',
                'medium': '⚠️',
                'high': '🚨',
                'critical': '🔥'
            }
            emoji = emoji_map.get(alert.severity, '📊')
            
            # Create Slack message
            message = {
                "text": f"{emoji} AWS Cost Alert",
                "attachments": [
                    {
                        "color": "danger" if alert.severity in ['high', 'critical'] else "warning",
                        "title": alert.title,
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Current Cost",
                                "value": f"${alert.current_cost:.2f}",
                                "short": True
                            },
                            {
                                "title": "Service",
                                "value": alert.service_affected or "Multiple",
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "FinOps Platform Cost Monitor"
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(
                slack_config['webhook_url'],
                json=message,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent successfully: {alert.alert_id}")
                return True
            else:
                logger.error(f"Slack alert failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

    def get_active_alerts(self) -> List[CostAlert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.active_alerts if not alert.resolved]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert statistics"""
        active_alerts = self.get_active_alerts()
        
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        type_counts = {}
        
        for alert in active_alerts:
            severity_counts[alert.severity] += 1
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
        
        return {
            'total_active': len(active_alerts),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'most_recent': active_alerts[0] if active_alerts else None
        }