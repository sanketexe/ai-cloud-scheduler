"""
Monitoring Service for Automated Cost Optimization

Provides comprehensive monitoring and error detection for automation activities:
- Real-time action monitoring with severity levels
- Error detection and administrator alerting
- Detailed execution reporting with before/after states
- System health monitoring and automation state management
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import structlog

from .automation_models import OptimizationAction, ActionStatus, AutomationPolicy
from .notification_service import (
    get_notification_service, NotificationMessage, NotificationPriority
)
from .automation_audit_logger import AutomationAuditLogger
from .database import get_db_session

logger = structlog.get_logger(__name__)


class MonitoringSeverity(Enum):
    """Monitoring severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AutomationState(Enum):
    """Automation system states"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class ExecutionReport:
    """Detailed execution report with before/after states"""
    
    def __init__(self, action: OptimizationAction):
        self.action_id = action.id
        self.action_type = action.action_type
        self.resource_id = action.resource_id
        self.resource_type = action.resource_type
        self.execution_started_at = action.execution_started_at
        self.execution_completed_at = action.execution_completed_at
        self.execution_status = action.execution_status
        self.estimated_savings = action.estimated_monthly_savings
        self.actual_savings = action.actual_savings
        self.error_message = action.error_message
        
        # Before/after states
        self.before_state = {}
        self.after_state = {}
        self.state_changes = {}
        
        # Execution metrics
        self.execution_duration = None
        if action.execution_started_at and action.execution_completed_at:
            self.execution_duration = (
                action.execution_completed_at - action.execution_started_at
            ).total_seconds()
    
    def set_before_state(self, state: Dict[str, Any]):
        """Set the resource state before action execution"""
        self.before_state = state
    
    def set_after_state(self, state: Dict[str, Any]):
        """Set the resource state after action execution"""
        self.after_state = state
        self._calculate_state_changes()
    
    def _calculate_state_changes(self):
        """Calculate what changed between before and after states"""
        self.state_changes = {}
        
        # Find changed values
        for key in set(self.before_state.keys()) | set(self.after_state.keys()):
            before_value = self.before_state.get(key)
            after_value = self.after_state.get(key)
            
            if before_value != after_value:
                self.state_changes[key] = {
                    "before": before_value,
                    "after": after_value
                }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format"""
        return {
            "action_id": str(self.action_id),
            "action_type": self.action_type.value,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "execution_started_at": self.execution_started_at.isoformat() if self.execution_started_at else None,
            "execution_completed_at": self.execution_completed_at.isoformat() if self.execution_completed_at else None,
            "execution_status": self.execution_status.value,
            "execution_duration_seconds": self.execution_duration,
            "estimated_savings": float(self.estimated_savings) if self.estimated_savings else 0,
            "actual_savings": float(self.actual_savings) if self.actual_savings else 0,
            "error_message": self.error_message,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "state_changes": self.state_changes
        }


class MonitoringService:
    """
    Comprehensive monitoring service for automated cost optimization.
    
    Handles real-time monitoring, error detection, and detailed reporting
    of all automation activities with appropriate alerting.
    """
    
    def __init__(self):
        self.notification_service = get_notification_service()
        self.audit_logger = AutomationAuditLogger()
        self.automation_state = AutomationState.ENABLED
        self.error_count = 0
        self.last_error_time = None
        self.monitoring_active = True
        
        # Configuration
        self.error_threshold = 5  # Number of errors before halting
        self.error_window_minutes = 15  # Time window for error counting
        self.administrator_channels = []  # Will be configured
        
        # Execution reports cache
        self.execution_reports = {}
    
    def set_automation_state(self, state: AutomationState, reason: str = None):
        """
        Set the automation system state.
        
        Args:
            state: New automation state
            reason: Reason for state change
        """
        previous_state = self.automation_state
        self.automation_state = state
        
        logger.info("Automation state changed",
                   previous_state=previous_state.value,
                   new_state=state.value,
                   reason=reason)
        
        # Log state change event
        self.audit_logger.log_system_event(
            "automation_state_changed",
            {
                "previous_state": previous_state.value,
                "new_state": state.value,
                "reason": reason,
                "changed_at": datetime.utcnow().isoformat()
            },
            severity="info" if state == AutomationState.ENABLED else "warning"
        )
        
        # Notify administrators of state changes
        try:
            asyncio.create_task(self._notify_state_change(previous_state, state, reason))
        except RuntimeError:
            # No event loop running, create a new one for the notification
            asyncio.run(self._notify_state_change(previous_state, state, reason))
    
    def get_automation_state(self) -> AutomationState:
        """Get current automation state"""
        return self.automation_state
    
    def is_automation_enabled(self) -> bool:
        """Check if automation is currently enabled"""
        return self.automation_state == AutomationState.ENABLED
    
    def configure_administrator_channels(self, channel_ids: List[str]):
        """Configure notification channels for administrator alerts"""
        self.administrator_channels = channel_ids
        logger.info("Administrator channels configured", 
                   channel_count=len(channel_ids))
    
    async def monitor_action_execution(self, action: OptimizationAction) -> ExecutionReport:
        """
        Monitor the execution of an optimization action.
        
        Args:
            action: The optimization action being executed
            
        Returns:
            Detailed execution report
        """
        logger.info("Starting action execution monitoring",
                   action_id=str(action.id),
                   action_type=action.action_type.value)
        
        # Create execution report
        report = ExecutionReport(action)
        self.execution_reports[str(action.id)] = report
        
        # Capture before state
        before_state = await self._capture_resource_state(action)
        report.set_before_state(before_state)
        
        # Send action start notification
        await self._send_action_notification(
            action,
            "Action execution started",
            MonitoringSeverity.INFO,
            {"before_state": before_state}
        )
        
        return report
    
    async def report_action_completion(self, 
                                     action: OptimizationAction,
                                     success: bool,
                                     execution_details: Dict[str, Any] = None):
        """
        Report completion of an action execution.
        
        Args:
            action: The completed optimization action
            success: Whether the action succeeded
            execution_details: Additional execution details
        """
        logger.info("Reporting action completion",
                   action_id=str(action.id),
                   success=success)
        
        # Get execution report
        report = self.execution_reports.get(str(action.id))
        if not report:
            report = ExecutionReport(action)
            self.execution_reports[str(action.id)] = report
        
        # Capture after state
        after_state = await self._capture_resource_state(action)
        report.set_after_state(after_state)
        
        # Determine severity and message
        if success:
            severity = MonitoringSeverity.INFO
            message = f"Action completed successfully: {action.action_type.value}"
        else:
            severity = MonitoringSeverity.ERROR
            message = f"Action failed: {action.action_type.value}"
            await self._handle_action_error(action, execution_details)
        
        # Send completion notification
        await self._send_action_notification(
            action,
            message,
            severity,
            {
                "execution_details": execution_details or {},
                "before_state": report.before_state,
                "after_state": report.after_state,
                "state_changes": report.state_changes,
                "execution_duration": report.execution_duration
            }
        )
        
        # Log execution report
        self.audit_logger.log_execution_event(
            action.id,
            "completed" if success else "failed",
            {
                "execution_report": report.to_dict(),
                "execution_details": execution_details or {}
            }
        )
    
    async def detect_and_handle_errors(self, 
                                     error: Exception,
                                     context: Dict[str, Any] = None):
        """
        Detect and handle system errors with appropriate alerting.
        
        Args:
            error: The error that occurred
            context: Additional context about the error
        """
        current_time = datetime.utcnow()
        
        # Update error tracking
        if (self.last_error_time is None or 
            (current_time - self.last_error_time).total_seconds() > self.error_window_minutes * 60):
            # Reset error count if outside window
            self.error_count = 1
        else:
            self.error_count += 1
        
        self.last_error_time = current_time
        
        logger.error("System error detected",
                    error=str(error),
                    error_count=self.error_count,
                    context=context or {})
        
        # Check if we should halt automation
        should_halt = self.error_count >= self.error_threshold
        
        if should_halt and self.automation_state == AutomationState.ENABLED:
            self.set_automation_state(
                AutomationState.ERROR,
                f"Too many errors ({self.error_count}) in {self.error_window_minutes} minutes"
            )
        
        # Send error notification to administrators
        await self._send_administrator_alert(
            "System Error Detected",
            MonitoringSeverity.CRITICAL if should_halt else MonitoringSeverity.ERROR,
            {
                "error_message": str(error),
                "error_count": self.error_count,
                "time_window_minutes": self.error_window_minutes,
                "automation_halted": should_halt,
                "context": context or {}
            }
        )
        
        # Log system error event
        self.audit_logger.log_system_event(
            "system_error_detected",
            {
                "error_message": str(error),
                "error_type": type(error).__name__,
                "error_count": self.error_count,
                "automation_halted": should_halt,
                "context": context or {}
            },
            severity="critical" if should_halt else "error"
        )
    
    async def generate_execution_summary(self, 
                                       time_period_hours: int = 24) -> Dict[str, Any]:
        """
        Generate execution summary for a time period.
        
        Args:
            time_period_hours: Hours to look back for summary
            
        Returns:
            Execution summary report
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_period_hours)
        
        try:
            with get_db_session() as session:
                # Query actions in time period
                actions = session.query(OptimizationAction).filter(
                    OptimizationAction.execution_started_at >= start_time,
                    OptimizationAction.execution_started_at <= end_time
                ).all()
                
                # Calculate summary statistics
                total_actions = len(actions)
                successful_actions = len([a for a in actions if a.execution_status == ActionStatus.COMPLETED])
                failed_actions = len([a for a in actions if a.execution_status == ActionStatus.FAILED])
                
                total_estimated_savings = sum(
                    float(a.estimated_monthly_savings) for a in actions
                    if a.estimated_monthly_savings
                )
                
                total_actual_savings = sum(
                    float(a.actual_savings) for a in actions
                    if a.actual_savings
                )
                
                # Group by action type
                actions_by_type = {}
                for action in actions:
                    action_type = action.action_type.value
                    if action_type not in actions_by_type:
                        actions_by_type[action_type] = {
                            "count": 0,
                            "successful": 0,
                            "failed": 0,
                            "estimated_savings": 0,
                            "actual_savings": 0
                        }
                    
                    actions_by_type[action_type]["count"] += 1
                    if action.execution_status == ActionStatus.COMPLETED:
                        actions_by_type[action_type]["successful"] += 1
                    elif action.execution_status == ActionStatus.FAILED:
                        actions_by_type[action_type]["failed"] += 1
                    
                    if action.estimated_monthly_savings:
                        actions_by_type[action_type]["estimated_savings"] += float(action.estimated_monthly_savings)
                    if action.actual_savings:
                        actions_by_type[action_type]["actual_savings"] += float(action.actual_savings)
                
                summary = {
                    "time_period": {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "hours": time_period_hours
                    },
                    "overall_statistics": {
                        "total_actions": total_actions,
                        "successful_actions": successful_actions,
                        "failed_actions": failed_actions,
                        "success_rate": (successful_actions / total_actions * 100) if total_actions > 0 else 0,
                        "total_estimated_savings": total_estimated_savings,
                        "total_actual_savings": total_actual_savings,
                        "savings_accuracy": (total_actual_savings / total_estimated_savings * 100) if total_estimated_savings > 0 else 0
                    },
                    "actions_by_type": actions_by_type,
                    "automation_state": self.automation_state.value,
                    "error_count": self.error_count,
                    "generated_at": datetime.utcnow().isoformat()
                }
                
                logger.info("Execution summary generated",
                           time_period_hours=time_period_hours,
                           total_actions=total_actions,
                           success_rate=summary["overall_statistics"]["success_rate"])
                
                return summary
                
        except Exception as e:
            logger.error("Failed to generate execution summary", error=str(e))
            return {
                "error": "Failed to generate summary",
                "error_message": str(e),
                "automation_state": self.automation_state.value,
                "generated_at": datetime.utcnow().isoformat()
            }
    
    async def _capture_resource_state(self, action: OptimizationAction) -> Dict[str, Any]:
        """Capture the current state of a resource"""
        # In a real implementation, this would query AWS APIs
        # For now, simulate based on action type and metadata
        
        resource_state = {
            "resource_id": action.resource_id,
            "resource_type": action.resource_type,
            "captured_at": datetime.utcnow().isoformat()
        }
        
        # Add action-specific state information
        if action.action_type.value.startswith("stop_") or action.action_type.value.startswith("terminate_"):
            resource_state.update({
                "instance_state": "running" if action.execution_status == ActionStatus.PENDING else "stopped",
                "instance_type": action.resource_metadata.get("instance_type", "unknown"),
                "availability_zone": action.resource_metadata.get("availability_zone", "unknown")
            })
        elif "volume" in action.action_type.value:
            resource_state.update({
                "volume_state": "available" if action.execution_status == ActionStatus.PENDING else "deleted",
                "volume_type": action.resource_metadata.get("volume_type", "gp2"),
                "size_gb": action.resource_metadata.get("size_gb", 0),
                "attached_to": action.resource_metadata.get("attached_to")
            })
        elif "elastic_ip" in action.action_type.value:
            resource_state.update({
                "allocation_state": "allocated" if action.execution_status == ActionStatus.PENDING else "released",
                "public_ip": action.resource_metadata.get("public_ip"),
                "associated_with": action.resource_metadata.get("associated_with")
            })
        
        return resource_state
    
    async def _send_action_notification(self,
                                      action: OptimizationAction,
                                      message: str,
                                      severity: MonitoringSeverity,
                                      additional_data: Dict[str, Any] = None):
        """Send notification about an action"""
        
        # Map severity to notification priority
        priority_map = {
            MonitoringSeverity.INFO: NotificationPriority.LOW,
            MonitoringSeverity.WARNING: NotificationPriority.MEDIUM,
            MonitoringSeverity.ERROR: NotificationPriority.HIGH,
            MonitoringSeverity.CRITICAL: NotificationPriority.CRITICAL
        }
        
        notification = NotificationMessage(
            title=f"Automation Action: {action.action_type.value}",
            message=message,
            priority=priority_map[severity],
            metadata={
                "action_id": str(action.id),
                "resource_id": action.resource_id,
                "resource_type": action.resource_type,
                "estimated_savings": float(action.estimated_monthly_savings) if action.estimated_monthly_savings else 0,
                "severity": severity.value,
                **(additional_data or {})
            },
            alert_id=str(action.id),
            resource_id=action.resource_id,
            cost_amount=float(action.estimated_monthly_savings) if action.estimated_monthly_savings else None
        )
        
        # Send to configured channels (would be configured based on severity)
        channels = self._get_notification_channels(severity)
        if channels:
            await self.notification_service.send_notification(channels, notification)
    
    async def _send_administrator_alert(self,
                                      title: str,
                                      severity: MonitoringSeverity,
                                      alert_data: Dict[str, Any]):
        """Send alert to administrators"""
        
        if not self.administrator_channels:
            logger.warning("No administrator channels configured for alerts")
            return
        
        priority_map = {
            MonitoringSeverity.INFO: NotificationPriority.LOW,
            MonitoringSeverity.WARNING: NotificationPriority.MEDIUM,
            MonitoringSeverity.ERROR: NotificationPriority.HIGH,
            MonitoringSeverity.CRITICAL: NotificationPriority.CRITICAL
        }
        
        notification = NotificationMessage(
            title=f"[ADMIN ALERT] {title}",
            message=f"Automation system alert: {alert_data.get('error_message', 'System event occurred')}",
            priority=priority_map[severity],
            metadata={
                "alert_type": "administrator_alert",
                "severity": severity.value,
                "automation_state": self.automation_state.value,
                **alert_data
            }
        )
        
        await self.notification_service.send_notification(
            self.administrator_channels,
            notification
        )
    
    async def _notify_state_change(self,
                                 previous_state: AutomationState,
                                 new_state: AutomationState,
                                 reason: str = None):
        """Notify about automation state changes"""
        
        severity = MonitoringSeverity.INFO
        if new_state in [AutomationState.ERROR, AutomationState.DISABLED]:
            severity = MonitoringSeverity.CRITICAL
        elif new_state == AutomationState.PAUSED:
            severity = MonitoringSeverity.WARNING
        
        await self._send_administrator_alert(
            f"Automation State Changed: {new_state.value.upper()}",
            severity,
            {
                "previous_state": previous_state.value,
                "new_state": new_state.value,
                "reason": reason,
                "state_changed_at": datetime.utcnow().isoformat()
            }
        )
    
    async def _handle_action_error(self,
                                 action: OptimizationAction,
                                 execution_details: Dict[str, Any] = None):
        """Handle errors during action execution"""
        
        error_message = execution_details.get("error_message", "Unknown error") if execution_details else "Unknown error"
        
        # Log error details
        self.audit_logger.log_execution_event(
            action.id,
            "error_occurred",
            {
                "error_message": error_message,
                "execution_details": execution_details or {},
                "error_time": datetime.utcnow().isoformat()
            }
        )
        
        # Check if this is a critical error that should halt automation
        critical_errors = [
            "permission_denied",
            "resource_not_found",
            "invalid_state",
            "safety_check_failed"
        ]
        
        is_critical = any(critical_error in error_message.lower() for critical_error in critical_errors)
        
        if is_critical:
            await self.detect_and_handle_errors(
                Exception(f"Critical action error: {error_message}"),
                {
                    "action_id": str(action.id),
                    "action_type": action.action_type.value,
                    "resource_id": action.resource_id,
                    "execution_details": execution_details or {}
                }
            )
    
    def _get_notification_channels(self, severity: MonitoringSeverity) -> List[str]:
        """Get notification channels based on severity"""
        # In a real implementation, this would be configurable
        # For now, return administrator channels for errors and critical
        if severity in [MonitoringSeverity.ERROR, MonitoringSeverity.CRITICAL]:
            return self.administrator_channels
        else:
            # Return general notification channels for info/warning
            return []
    
    def get_execution_report(self, action_id: str) -> Optional[ExecutionReport]:
        """Get execution report for an action"""
        return self.execution_reports.get(action_id)
    
    def clear_execution_reports(self, older_than_hours: int = 24):
        """Clear old execution reports"""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        reports_to_remove = []
        for action_id, report in self.execution_reports.items():
            if (report.execution_completed_at and 
                report.execution_completed_at < cutoff_time):
                reports_to_remove.append(action_id)
        
        for action_id in reports_to_remove:
            del self.execution_reports[action_id]
        
        logger.info("Cleared old execution reports",
                   removed_count=len(reports_to_remove),
                   cutoff_hours=older_than_hours)


# Global monitoring service instance
_monitoring_service = None

def get_monitoring_service() -> MonitoringService:
    """Get global monitoring service instance"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service