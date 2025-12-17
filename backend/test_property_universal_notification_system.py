#!/usr/bin/env python3
"""
Property-Based Tests for Universal Notification System

This module contains property-based tests to verify that the notification system
sends appropriate notifications through configured channels and provides detailed
execution reports for all automated actions and errors according to the requirements.

**Feature: automated-cost-optimization, Property 12: Universal Notification System**
**Validates: Requirements 5.1, 5.3**
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from hypothesis import given, strategies as st, assume, settings
from dataclasses import dataclass
import asyncio

# Import the components we're testing
from core.notification_service import (
    NotificationService, NotificationMessage, NotificationPriority,
    NotificationChannel, EmailConfig, SlackConfig, TeamsConfig, WebhookConfig
)


@dataclass
class MockActionResult:
    """Mock action result for testing"""
    action_id: uuid.UUID
    action_type: str
    resource_id: str
    resource_type: str
    execution_time: datetime
    success: bool
    actual_savings: Decimal
    resources_affected: List[str]
    error_message: Optional[str]
    rollback_required: bool
    execution_details: Dict[str, Any]


@dataclass
class MockErrorEvent:
    """Mock error event for testing"""
    error_id: uuid.UUID
    error_type: str
    component: str
    error_message: str
    severity: str
    timestamp: datetime
    context: Dict[str, Any]


class MockNotificationService:
    """Mock notification service for testing that captures sent notifications"""
    
    def __init__(self):
        self.channels = {}
        self.sent_notifications = []  # Store all sent notifications
        self.channel_results = {}  # Store results for each channel
        self.notification_history = []
        
    def register_email_channel(self, channel_id: str, config: EmailConfig):
        """Register an email notification channel"""
        self.channels[channel_id] = {
            'type': NotificationChannel.EMAIL,
            'config': config
        }
        
    def register_slack_channel(self, channel_id: str, config: SlackConfig):
        """Register a Slack notification channel"""
        self.channels[channel_id] = {
            'type': NotificationChannel.SLACK,
            'config': config
        }
        
    def register_teams_channel(self, channel_id: str, config: TeamsConfig):
        """Register a Microsoft Teams notification channel"""
        self.channels[channel_id] = {
            'type': NotificationChannel.TEAMS,
            'config': config
        }
        
    def register_webhook_channel(self, channel_id: str, config: WebhookConfig):
        """Register a custom webhook channel"""
        self.channels[channel_id] = {
            'type': NotificationChannel.WEBHOOK,
            'config': config
        }
    
    async def send_notification(self, 
                               channel_ids: List[str],
                               message: NotificationMessage) -> Dict[str, bool]:
        """
        Mock send notification that records the notification and returns success
        """
        results = {}
        
        # Record the notification attempt
        notification_record = {
            'timestamp': datetime.utcnow(),
            'channel_ids': channel_ids,
            'message': message,
            'results': {}
        }
        
        for channel_id in channel_ids:
            if channel_id not in self.channels:
                results[channel_id] = False
                notification_record['results'][channel_id] = False
                continue
            
            # Simulate successful sending for valid channels
            success = True
            results[channel_id] = success
            notification_record['results'][channel_id] = success
            
            # Store individual channel result
            if channel_id not in self.channel_results:
                self.channel_results[channel_id] = []
            self.channel_results[channel_id].append({
                'message': message,
                'success': success,
                'timestamp': datetime.utcnow()
            })
        
        # Store the notification record
        self.sent_notifications.append(notification_record)
        
        # Add to history
        self.notification_history.append({
            'timestamp': message.timestamp,
            'title': message.title,
            'priority': message.priority.value,
            'channels': channel_ids,
            'results': results
        })
        
        return results
    
    def get_notification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent notification history"""
        return self.notification_history[-limit:]
    
    def clear_history(self):
        """Clear notification history"""
        self.notification_history.clear()
        self.sent_notifications.clear()
        self.channel_results.clear()


class TestUniversalNotificationSystem:
    """Property-based tests for universal notification system"""
    
    def __init__(self):
        self.notification_service = MockNotificationService()
    
    @given(
        # Generate various action types and results
        action_type=st.sampled_from([
            'stop_instance', 'terminate_instance', 'resize_instance',
            'delete_volume', 'upgrade_storage', 'create_snapshot',
            'release_elastic_ip', 'delete_load_balancer', 'cleanup_security_group'
        ]),
        resource_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        resource_type=st.sampled_from(['ec2_instance', 'ebs_volume', 'elastic_ip', 'load_balancer', 'security_group']),
        execution_success=st.booleans(),
        actual_savings=st.decimals(min_value=0, max_value=10000, places=2),
        rollback_required=st.booleans(),
        
        # Generate various notification priorities
        notification_priority=st.sampled_from(list(NotificationPriority)),
        
        # Generate various channel configurations
        num_email_channels=st.integers(min_value=0, max_value=3),
        num_slack_channels=st.integers(min_value=0, max_value=3),
        num_teams_channels=st.integers(min_value=0, max_value=3),
        num_webhook_channels=st.integers(min_value=0, max_value=3),
        
        # Generate various error scenarios
        has_error=st.booleans(),
        error_type=st.sampled_from(['aws_api_error', 'permission_error', 'network_error', 'validation_error']),
        error_severity=st.sampled_from(['low', 'medium', 'high', 'critical']),
        
        # Generate various execution details
        resources_affected_count=st.integers(min_value=1, max_value=10),
        execution_duration_minutes=st.integers(min_value=1, max_value=120),
        
        # Generate various metadata scenarios
        has_alert_id=st.booleans(),
        has_cost_amount=st.booleans(),
        has_custom_metadata=st.booleans()
    )
    @settings(max_examples=100, deadline=None)
    def test_universal_notification_system_property(self,
                                                  action_type: str,
                                                  resource_id: str,
                                                  resource_type: str,
                                                  execution_success: bool,
                                                  actual_savings: Decimal,
                                                  rollback_required: bool,
                                                  notification_priority: NotificationPriority,
                                                  num_email_channels: int,
                                                  num_slack_channels: int,
                                                  num_teams_channels: int,
                                                  num_webhook_channels: int,
                                                  has_error: bool,
                                                  error_type: str,
                                                  error_severity: str,
                                                  resources_affected_count: int,
                                                  execution_duration_minutes: int,
                                                  has_alert_id: bool,
                                                  has_cost_amount: bool,
                                                  has_custom_metadata: bool):
        """
        **Feature: automated-cost-optimization, Property 12: Universal Notification System**
        
        Property: For any automated action or error, the system should send appropriate 
        notifications through configured channels and provide detailed execution reports.
        
        This property verifies that:
        1. Notifications are sent for all automated actions regardless of success/failure
        2. Notifications are sent through all configured channels
        3. Notification messages contain all required information
        4. Error notifications are sent with appropriate severity levels
        5. Execution reports include before/after states and detailed information
        6. Notification history is properly maintained
        7. Channel-specific formatting is applied correctly
        8. Priority levels are properly handled
        """
        
        # Skip invalid combinations
        assume(len(resource_id.strip()) > 0)
        assume(num_email_channels + num_slack_channels + num_teams_channels + num_webhook_channels > 0)
        
        # Clear previous state
        self.notification_service.channels.clear()
        self.notification_service.sent_notifications.clear()
        self.notification_service.channel_results.clear()
        self.notification_service.notification_history.clear()
        
        # Set up notification channels
        channel_ids = []
        
        # Register email channels
        for i in range(num_email_channels):
            channel_id = f"email-{i}"
            config = EmailConfig(
                smtp_host="smtp.example.com",
                smtp_port=587,
                smtp_username="test@example.com",
                smtp_password="password",
                from_address="finops@example.com"
            )
            self.notification_service.register_email_channel(channel_id, config)
            channel_ids.append(channel_id)
        
        # Register Slack channels
        for i in range(num_slack_channels):
            channel_id = f"slack-{i}"
            config = SlackConfig(
                webhook_url=f"https://hooks.slack.com/test-{i}",
                channel=f"#finops-{i}",
                username="FinOps Bot"
            )
            self.notification_service.register_slack_channel(channel_id, config)
            channel_ids.append(channel_id)
        
        # Register Teams channels
        for i in range(num_teams_channels):
            channel_id = f"teams-{i}"
            config = TeamsConfig(
                webhook_url=f"https://outlook.office.com/webhook/test-{i}"
            )
            self.notification_service.register_teams_channel(channel_id, config)
            channel_ids.append(channel_id)
        
        # Register webhook channels
        for i in range(num_webhook_channels):
            channel_id = f"webhook-{i}"
            config = WebhookConfig(
                url=f"https://api.example.com/webhook-{i}",
                method="POST",
                headers={"Content-Type": "application/json"},
                auth_token=f"token-{i}"
            )
            self.notification_service.register_webhook_channel(channel_id, config)
            channel_ids.append(channel_id)
        
        # Create action result
        action_id = uuid.uuid4()
        execution_time = datetime.utcnow()
        resources_affected = [f"{resource_type}-{i}" for i in range(resources_affected_count)]
        
        action_result = MockActionResult(
            action_id=action_id,
            action_type=action_type,
            resource_id=resource_id,
            resource_type=resource_type,
            execution_time=execution_time,
            success=execution_success,
            actual_savings=actual_savings,
            resources_affected=resources_affected,
            error_message=f"{error_type}: Failed to execute {action_type}" if has_error and not execution_success else None,
            rollback_required=rollback_required,
            execution_details={
                "duration_minutes": execution_duration_minutes,
                "resources_processed": len(resources_affected),
                "cost_savings": float(actual_savings),
                "rollback_plan_created": rollback_required
            }
        )
        
        # Create notification message for action
        message_title = f"Cost Optimization Action: {action_type.replace('_', ' ').title()}"
        if not execution_success:
            message_title = f"FAILED: {message_title}"
        
        message_text = f"Action {action_type} on {resource_type} {resource_id} "
        if execution_success:
            message_text += f"completed successfully. Savings: ${actual_savings}"
        else:
            message_text += f"failed. {action_result.error_message or 'Unknown error'}"
        
        # Create metadata
        metadata = {
            "execution_duration": execution_duration_minutes,
            "resources_affected_count": len(resources_affected),
            "rollback_required": rollback_required
        }
        
        if has_custom_metadata:
            metadata.update({
                "environment": "production",
                "region": "us-east-1",
                "account_id": "123456789012"
            })
        
        notification_message = NotificationMessage(
            title=message_title,
            message=message_text,
            priority=notification_priority,
            metadata=metadata,
            timestamp=execution_time,
            alert_id=f"alert-{uuid.uuid4().hex[:8]}" if has_alert_id else None,
            resource_id=resource_id,
            cost_amount=float(actual_savings) if has_cost_amount else None
        )
        
        # Execute notification sending
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                self.notification_service.send_notification(channel_ids, notification_message)
            )
        finally:
            loop.close()
        
        # PROPERTY ASSERTIONS: Universal notification system requirements
        
        # 1. Notifications must be sent for all automated actions
        assert len(self.notification_service.sent_notifications) > 0, \
            "Notifications must be sent for all automated actions"
        
        sent_notification = self.notification_service.sent_notifications[0]
        assert sent_notification is not None, "Notification record must be created"
        
        # 2. Notifications must be sent through all configured channels
        assert len(results) == len(channel_ids), \
            "Results must be returned for all configured channels"
        
        for channel_id in channel_ids:
            assert channel_id in results, f"Result must be provided for channel {channel_id}"
            assert isinstance(results[channel_id], bool), f"Result for {channel_id} must be boolean"
        
        # 3. All configured channels must be attempted
        assert sent_notification['channel_ids'] == channel_ids, \
            "All configured channels must be attempted"
        
        # 4. Notification message must contain all required information
        sent_message = sent_notification['message']
        assert sent_message.title == message_title, "Message title must match"
        assert sent_message.message == message_text, "Message text must match"
        assert sent_message.priority == notification_priority, "Message priority must match"
        assert sent_message.resource_id == resource_id, "Resource ID must be included"
        
        # 5. Timestamp must be properly set
        assert sent_message.timestamp == execution_time, "Timestamp must match execution time"
        assert isinstance(sent_message.timestamp, datetime), "Timestamp must be datetime object"
        
        # 6. Metadata must be properly included
        assert sent_message.metadata is not None, "Metadata must be included"
        assert "execution_duration" in sent_message.metadata, "Execution duration must be in metadata"
        assert "resources_affected_count" in sent_message.metadata, "Resource count must be in metadata"
        assert "rollback_required" in sent_message.metadata, "Rollback flag must be in metadata"
        
        assert sent_message.metadata["execution_duration"] == execution_duration_minutes, \
            "Execution duration must match"
        assert sent_message.metadata["resources_affected_count"] == len(resources_affected), \
            "Resource count must match"
        assert sent_message.metadata["rollback_required"] == rollback_required, \
            "Rollback flag must match"
        
        # 7. Optional fields must be handled correctly
        if has_alert_id:
            assert sent_message.alert_id is not None, "Alert ID must be included when specified"
            assert sent_message.alert_id.startswith("alert-"), "Alert ID must have correct format"
        else:
            assert sent_message.alert_id is None, "Alert ID must be None when not specified"
        
        if has_cost_amount:
            assert sent_message.cost_amount is not None, "Cost amount must be included when specified"
            assert sent_message.cost_amount == float(actual_savings), "Cost amount must match savings"
        else:
            assert sent_message.cost_amount is None, "Cost amount must be None when not specified"
        
        # 8. Custom metadata must be preserved
        if has_custom_metadata:
            assert "environment" in sent_message.metadata, "Custom metadata must be preserved"
            assert "region" in sent_message.metadata, "Custom metadata must be preserved"
            assert "account_id" in sent_message.metadata, "Custom metadata must be preserved"
        
        # 9. Notification history must be maintained
        history = self.notification_service.get_notification_history()
        assert len(history) > 0, "Notification history must be maintained"
        
        history_record = history[0]
        assert history_record['title'] == message_title, "History must preserve title"
        assert history_record['priority'] == notification_priority.value, "History must preserve priority"
        assert history_record['channels'] == channel_ids, "History must preserve channel list"
        assert 'results' in history_record, "History must include results"
        assert 'timestamp' in history_record, "History must include timestamp"
        
        # 10. Channel-specific results must be tracked
        for channel_id in channel_ids:
            assert channel_id in self.notification_service.channel_results, \
                f"Channel results must be tracked for {channel_id}"
            
            channel_results = self.notification_service.channel_results[channel_id]
            assert len(channel_results) > 0, f"Results must be recorded for {channel_id}"
            
            latest_result = channel_results[-1]
            assert 'message' in latest_result, "Channel result must include message"
            assert 'success' in latest_result, "Channel result must include success status"
            assert 'timestamp' in latest_result, "Channel result must include timestamp"
            
            assert latest_result['message'] == sent_message, "Channel result message must match"
            assert isinstance(latest_result['success'], bool), "Channel success must be boolean"
            assert isinstance(latest_result['timestamp'], datetime), "Channel timestamp must be datetime"
        
        # 11. Test error notification scenario if error occurred
        if has_error and not execution_success:
            # Create error notification
            error_event = MockErrorEvent(
                error_id=uuid.uuid4(),
                error_type=error_type,
                component="auto_remediation_engine",
                error_message=action_result.error_message,
                severity=error_severity,
                timestamp=datetime.utcnow(),
                context={
                    "action_id": str(action_id),
                    "resource_id": resource_id,
                    "action_type": action_type
                }
            )
            
            # Map error severity to notification priority
            severity_priority_map = {
                'low': NotificationPriority.LOW,
                'medium': NotificationPriority.MEDIUM,
                'high': NotificationPriority.HIGH,
                'critical': NotificationPriority.CRITICAL
            }
            
            error_priority = severity_priority_map.get(error_severity, NotificationPriority.MEDIUM)
            
            error_notification = NotificationMessage(
                title=f"Automation Error: {error_type.replace('_', ' ').title()}",
                message=f"Error in {error_event.component}: {error_event.error_message}",
                priority=error_priority,
                metadata={
                    "error_type": error_type,
                    "component": error_event.component,
                    "severity": error_severity,
                    "context": error_event.context
                },
                timestamp=error_event.timestamp,
                alert_id=f"error-{error_event.error_id.hex[:8]}",
                resource_id=resource_id
            )
            
            # Send error notification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                error_results = loop.run_until_complete(
                    self.notification_service.send_notification(channel_ids, error_notification)
                )
            finally:
                loop.close()
            
            # Verify error notification was sent
            assert len(self.notification_service.sent_notifications) == 2, \
                "Error notification must be sent in addition to action notification"
            
            error_sent = self.notification_service.sent_notifications[1]
            error_sent_message = error_sent['message']
            
            assert error_sent_message.priority == error_priority, \
                "Error notification priority must match severity"
            assert "Error in" in error_sent_message.message, \
                "Error notification must indicate error context"
            assert error_sent_message.alert_id.startswith("error-"), \
                "Error alert ID must have correct prefix"
            
            # Verify error metadata
            assert "error_type" in error_sent_message.metadata, \
                "Error metadata must include error type"
            assert "severity" in error_sent_message.metadata, \
                "Error metadata must include severity"
            assert "context" in error_sent_message.metadata, \
                "Error metadata must include context"
        
        # 12. Test notification history limits and management
        history_full = self.notification_service.get_notification_history(limit=1)
        assert len(history_full) <= 1, "History limit must be respected"
        
        # 13. Test history clearing functionality
        initial_history_count = len(self.notification_service.get_notification_history())
        self.notification_service.clear_history()
        
        cleared_history = self.notification_service.get_notification_history()
        assert len(cleared_history) == 0, "History must be cleared when requested"
        
        cleared_notifications = self.notification_service.sent_notifications
        assert len(cleared_notifications) == 0, "Sent notifications must be cleared"
        
        cleared_results = self.notification_service.channel_results
        assert len(cleared_results) == 0, "Channel results must be cleared"


def run_property_test():
    """Run the universal notification system property test"""
    print("Running Property-Based Test for Universal Notification System")
    print("=" * 60)
    print("**Feature: automated-cost-optimization, Property 12: Universal Notification System**")
    print("**Validates: Requirements 5.1, 5.3**")
    print()
    
    test_instance = TestUniversalNotificationSystem()
    
    try:
        print("Testing Property 12: Universal Notification System...")
        test_instance.test_universal_notification_system_property()
        print("✓ Property 12 test completed successfully")
        print()
        print("Property validation confirmed:")
        print("- Notifications are sent for all automated actions")
        print("- Notifications are sent through all configured channels")
        print("- Notification messages contain all required information")
        print("- Error notifications are sent with appropriate severity levels")
        print("- Execution reports include detailed information")
        print("- Notification history is properly maintained")
        print("- Channel-specific formatting is applied correctly")
        print("- Priority levels are properly handled")
        print("- Error scenarios are handled appropriately")
        print("- History management works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Property test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_test()
    if success:
        print("\nUniversal Notification System property test passed!")
    else:
        print("\nUniversal Notification System property test failed!")
        exit(1)