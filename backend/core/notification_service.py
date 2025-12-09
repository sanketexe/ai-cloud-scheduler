"""
Notification Service for FinOps Alerts

Supports multiple notification channels:
- Email (SMTP)
- Slack (Webhooks)
- Microsoft Teams (Webhooks)
- Custom Webhooks
- SMS (via AWS SNS)
"""

import smtplib
import json
import logging
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import structlog

logger = structlog.get_logger(__name__)


class NotificationChannel(Enum):
    """Supported notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SNS = "sns"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NotificationMessage:
    """Notification message structure"""
    title: str
    message: str
    priority: NotificationPriority
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    alert_id: Optional[str] = None
    resource_id: Optional[str] = None
    cost_amount: Optional[float] = None


@dataclass
class EmailConfig:
    """Email notification configuration"""
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    from_address: str
    use_tls: bool = True


@dataclass
class SlackConfig:
    """Slack webhook configuration"""
    webhook_url: str
    channel: Optional[str] = None
    username: str = "FinOps Bot"
    icon_emoji: str = ":moneybag:"


@dataclass
class TeamsConfig:
    """Microsoft Teams webhook configuration"""
    webhook_url: str


@dataclass
class WebhookConfig:
    """Custom webhook configuration"""
    url: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    auth_token: Optional[str] = None


class NotificationService:
    """
    Centralized notification service for FinOps alerts.
    Handles multiple channels and formats messages appropriately.
    """
    
    def __init__(self):
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.notification_history: List[Dict[str, Any]] = []
    
    def register_email_channel(self, channel_id: str, config: EmailConfig):
        """Register an email notification channel"""
        self.channels[channel_id] = {
            'type': NotificationChannel.EMAIL,
            'config': config
        }
        logger.info("Registered email channel", channel_id=channel_id)
    
    def register_slack_channel(self, channel_id: str, config: SlackConfig):
        """Register a Slack notification channel"""
        self.channels[channel_id] = {
            'type': NotificationChannel.SLACK,
            'config': config
        }
        logger.info("Registered Slack channel", channel_id=channel_id)
    
    def register_teams_channel(self, channel_id: str, config: TeamsConfig):
        """Register a Microsoft Teams notification channel"""
        self.channels[channel_id] = {
            'type': NotificationChannel.TEAMS,
            'config': config
        }
        logger.info("Registered Teams channel", channel_id=channel_id)
    
    def register_webhook_channel(self, channel_id: str, config: WebhookConfig):
        """Register a custom webhook channel"""
        self.channels[channel_id] = {
            'type': NotificationChannel.WEBHOOK,
            'config': config
        }
        logger.info("Registered webhook channel", channel_id=channel_id)
    
    async def send_notification(self, 
                               channel_ids: List[str],
                               message: NotificationMessage) -> Dict[str, bool]:
        """
        Send notification to multiple channels.
        
        Args:
            channel_ids: List of channel IDs to send to
            message: Notification message
            
        Returns:
            Dict mapping channel_id to success status
        """
        results = {}
        tasks = []
        
        for channel_id in channel_ids:
            if channel_id not in self.channels:
                logger.warning("Channel not found", channel_id=channel_id)
                results[channel_id] = False
                continue
            
            channel = self.channels[channel_id]
            task = self._send_to_channel(channel_id, channel, message)
            tasks.append((channel_id, task))
        
        # Send to all channels in parallel
        for channel_id, task in tasks:
            try:
                success = await task
                results[channel_id] = success
            except Exception as e:
                logger.error("Failed to send notification",
                           channel_id=channel_id,
                           error=str(e))
                results[channel_id] = False
        
        # Log to history
        self.notification_history.append({
            'timestamp': message.timestamp,
            'title': message.title,
            'priority': message.priority.value,
            'channels': channel_ids,
            'results': results
        })
        
        return results
    
    async def _send_to_channel(self,
                               channel_id: str,
                               channel: Dict[str, Any],
                               message: NotificationMessage) -> bool:
        """Send notification to a specific channel"""
        channel_type = channel['type']
        config = channel['config']
        
        try:
            if channel_type == NotificationChannel.EMAIL:
                return await self._send_email(config, message)
            elif channel_type == NotificationChannel.SLACK:
                return await self._send_slack(config, message)
            elif channel_type == NotificationChannel.TEAMS:
                return await self._send_teams(config, message)
            elif channel_type == NotificationChannel.WEBHOOK:
                return await self._send_webhook(config, message)
            else:
                logger.warning("Unsupported channel type", type=channel_type)
                return False
        except Exception as e:
            logger.error("Error sending to channel",
                        channel_id=channel_id,
                        error=str(e))
            return False
    
    async def _send_email(self, config: EmailConfig, message: NotificationMessage) -> bool:
        """Send email notification"""
        def _send():
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{message.priority.value.upper()}] {message.title}"
            msg['From'] = config.from_address
            msg['To'] = message.metadata.get('to_address', config.from_address)
            
            # Create HTML body
            html_body = self._format_email_html(message)
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            # Send email
            if config.use_tls:
                server = smtplib.SMTP(config.smtp_host, config.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP(config.smtp_host, config.smtp_port)
            
            server.login(config.smtp_username, config.smtp_password)
            server.send_message(msg)
            server.quit()
            
            return True
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _send)
    
    async def _send_slack(self, config: SlackConfig, message: NotificationMessage) -> bool:
        """Send Slack notification"""
        def _send():
            # Format Slack message
            slack_message = {
                "username": config.username,
                "icon_emoji": config.icon_emoji,
                "attachments": [
                    {
                        "color": self._get_priority_color(message.priority),
                        "title": message.title,
                        "text": message.message,
                        "fields": self._format_slack_fields(message),
                        "footer": "FinOps Platform",
                        "ts": int(message.timestamp.timestamp())
                    }
                ]
            }
            
            if config.channel:
                slack_message["channel"] = config.channel
            
            # Send to Slack
            response = requests.post(
                config.webhook_url,
                json=slack_message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            return response.status_code == 200
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _send)
    
    async def _send_teams(self, config: TeamsConfig, message: NotificationMessage) -> bool:
        """Send Microsoft Teams notification"""
        def _send():
            # Format Teams message (Adaptive Card)
            teams_message = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": message.title,
                "themeColor": self._get_priority_color(message.priority),
                "title": message.title,
                "sections": [
                    {
                        "activityTitle": "FinOps Alert",
                        "activitySubtitle": message.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "facts": self._format_teams_facts(message),
                        "text": message.message
                    }
                ]
            }
            
            # Send to Teams
            response = requests.post(
                config.webhook_url,
                json=teams_message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            return response.status_code == 200
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _send)
    
    async def _send_webhook(self, config: WebhookConfig, message: NotificationMessage) -> bool:
        """Send custom webhook notification"""
        def _send():
            # Format webhook payload
            payload = {
                "title": message.title,
                "message": message.message,
                "priority": message.priority.value,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata
            }
            
            if message.alert_id:
                payload["alert_id"] = message.alert_id
            if message.resource_id:
                payload["resource_id"] = message.resource_id
            if message.cost_amount:
                payload["cost_amount"] = message.cost_amount
            
            # Prepare headers
            headers = config.headers.copy()
            headers['Content-Type'] = 'application/json'
            
            if config.auth_token:
                headers['Authorization'] = f"Bearer {config.auth_token}"
            
            # Send webhook
            response = requests.request(
                method=config.method,
                url=config.url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            return response.status_code in [200, 201, 202, 204]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _send)
    
    def _format_email_html(self, message: NotificationMessage) -> str:
        """Format email as HTML"""
        priority_colors = {
            NotificationPriority.LOW: "#36a64f",
            NotificationPriority.MEDIUM: "#ff9900",
            NotificationPriority.HIGH: "#ff6600",
            NotificationPriority.CRITICAL: "#ff0000"
        }
        
        color = priority_colors.get(message.priority, "#cccccc")
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: {color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .metadata {{ background-color: #f5f5f5; padding: 15px; margin-top: 20px; }}
                .metadata-item {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{message.title}</h2>
                <p>Priority: {message.priority.value.upper()}</p>
            </div>
            <div class="content">
                <p>{message.message}</p>
                <div class="metadata">
                    <h3>Details</h3>
        """
        
        if message.alert_id:
            html += f'<div class="metadata-item"><strong>Alert ID:</strong> {message.alert_id}</div>'
        if message.resource_id:
            html += f'<div class="metadata-item"><strong>Resource:</strong> {message.resource_id}</div>'
        if message.cost_amount:
            html += f'<div class="metadata-item"><strong>Cost:</strong> ${message.cost_amount:,.2f}</div>'
        
        for key, value in message.metadata.items():
            html += f'<div class="metadata-item"><strong>{key}:</strong> {value}</div>'
        
        html += f"""
                    <div class="metadata-item"><strong>Timestamp:</strong> {message.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}</div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_slack_fields(self, message: NotificationMessage) -> List[Dict[str, Any]]:
        """Format Slack message fields"""
        fields = []
        
        if message.alert_id:
            fields.append({"title": "Alert ID", "value": message.alert_id, "short": True})
        if message.resource_id:
            fields.append({"title": "Resource", "value": message.resource_id, "short": True})
        if message.cost_amount:
            fields.append({"title": "Cost", "value": f"${message.cost_amount:,.2f}", "short": True})
        
        fields.append({"title": "Priority", "value": message.priority.value.upper(), "short": True})
        
        return fields
    
    def _format_teams_facts(self, message: NotificationMessage) -> List[Dict[str, str]]:
        """Format Teams message facts"""
        facts = []
        
        if message.alert_id:
            facts.append({"name": "Alert ID", "value": message.alert_id})
        if message.resource_id:
            facts.append({"name": "Resource", "value": message.resource_id})
        if message.cost_amount:
            facts.append({"name": "Cost", "value": f"${message.cost_amount:,.2f}"})
        
        facts.append({"name": "Priority", "value": message.priority.value.upper()})
        
        return facts
    
    def _get_priority_color(self, priority: NotificationPriority) -> str:
        """Get color code for priority level"""
        colors = {
            NotificationPriority.LOW: "#36a64f",
            NotificationPriority.MEDIUM: "#ff9900",
            NotificationPriority.HIGH: "#ff6600",
            NotificationPriority.CRITICAL: "#ff0000"
        }
        return colors.get(priority, "#cccccc")
    
    def get_notification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent notification history"""
        return self.notification_history[-limit:]
    
    def clear_history(self):
        """Clear notification history"""
        self.notification_history.clear()


# Global notification service instance
_notification_service = None

def get_notification_service() -> NotificationService:
    """Get global notification service instance"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
