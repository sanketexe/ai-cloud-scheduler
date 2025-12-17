"""
Scheduling Engine for Automated Cost Optimization

Provides intelligent scheduling and timing for optimization actions:
- Business hours awareness and time zone handling
- Maintenance window and blackout period support
- Resource usage pattern analysis for optimal timing
- Emergency override capabilities with proper authorization
"""

import uuid
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum as PyEnum
import pytz
import structlog

from .automation_models import (
    OptimizationAction, AutomationPolicy, ActionType, RiskLevel, 
    ActionStatus, ApprovalStatus
)
from .database import get_db_session
from .automation_audit_logger import AutomationAuditLogger

logger = structlog.get_logger(__name__)


class SchedulePriority(PyEnum):
    """Priority levels for action scheduling"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EMERGENCY = "emergency"


class MaintenanceWindowType(PyEnum):
    """Types of maintenance windows"""
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class BusinessHours:
    """Business hours configuration"""
    timezone: str
    start_time: time
    end_time: time
    business_days: List[str]  # ['monday', 'tuesday', etc.]
    enabled: bool = True


@dataclass
class MaintenanceWindow:
    """Maintenance window configuration"""
    name: str
    window_type: MaintenanceWindowType
    start_time: datetime
    end_time: datetime
    timezone: str
    recurrence_pattern: Optional[Dict[str, Any]] = None
    enabled: bool = True


@dataclass
class BlackoutPeriod:
    """Blackout period where no actions should be executed"""
    name: str
    start_time: datetime
    end_time: datetime
    timezone: str
    reason: str
    enabled: bool = True


@dataclass
class ResourceUsagePattern:
    """Resource usage pattern for optimal timing"""
    resource_id: str
    resource_type: str
    peak_hours: List[int]  # Hours of day (0-23) when resource is heavily used
    low_usage_hours: List[int]  # Hours when resource usage is minimal
    timezone: str
    confidence_score: float  # 0.0 to 1.0


@dataclass
class SchedulingContext:
    """Context information for scheduling decisions"""
    current_time: datetime
    business_hours: Optional[BusinessHours]
    maintenance_windows: List[MaintenanceWindow]
    blackout_periods: List[BlackoutPeriod]
    resource_patterns: Dict[str, ResourceUsagePattern]
    emergency_override: bool = False
    override_reason: Optional[str] = None
    override_authorized_by: Optional[str] = None


class SchedulingEngine:
    """
    Intelligent scheduling engine for optimization actions.
    
    Determines optimal execution timing based on:
    - Business hours and organizational policies
    - Maintenance windows and blackout periods
    - Resource usage patterns and historical data
    - Risk levels and action priorities
    - Emergency override capabilities
    """
    
    def __init__(self):
        self.audit_logger = AutomationAuditLogger()
        self.default_timezone = "UTC"
        
    def calculate_optimal_execution_time(self,
                                       action: OptimizationAction,
                                       policy: AutomationPolicy,
                                       scheduling_context: Optional[SchedulingContext] = None) -> datetime:
        """
        Calculate the optimal execution time for an optimization action.
        
        Args:
            action: The optimization action to schedule
            policy: Automation policy with scheduling rules
            scheduling_context: Current scheduling context (optional)
            
        Returns:
            Optimal execution datetime
        """
        logger.info("Calculating optimal execution time",
                   action_id=str(action.id),
                   action_type=action.action_type.value,
                   risk_level=action.risk_level.value)
        
        if scheduling_context is None:
            scheduling_context = self._build_scheduling_context(policy)
        
        # Check for emergency override
        if scheduling_context.emergency_override:
            execution_time = self._schedule_emergency_action(action, scheduling_context)
            logger.info("Emergency override applied",
                       action_id=str(action.id),
                       execution_time=execution_time.isoformat(),
                       override_reason=scheduling_context.override_reason)
            return execution_time
        
        # Get base scheduling preferences from policy
        time_restrictions = policy.time_restrictions
        
        # Calculate execution time based on action characteristics
        execution_time = self._calculate_base_execution_time(
            action, time_restrictions, scheduling_context
        )
        
        # Apply business hours constraints
        execution_time = self._apply_business_hours_constraints(
            execution_time, action, scheduling_context
        )
        
        # Check and avoid blackout periods
        execution_time = self._avoid_blackout_periods(
            execution_time, action, scheduling_context
        )
        
        # Optimize based on resource usage patterns
        execution_time = self._optimize_for_resource_patterns(
            execution_time, action, scheduling_context
        )
        
        # Apply maintenance window preferences
        execution_time = self._apply_maintenance_window_preferences(
            execution_time, action, scheduling_context
        )
        
        # Final validation and adjustment
        execution_time = self._validate_and_adjust_time(
            execution_time, action, scheduling_context
        )
        
        # Log scheduling decision
        self._log_scheduling_decision(action, execution_time, scheduling_context)
        
        logger.info("Optimal execution time calculated",
                   action_id=str(action.id),
                   execution_time=execution_time.isoformat())
        
        return execution_time
    
    def _build_scheduling_context(self, policy: AutomationPolicy) -> SchedulingContext:
        """Build scheduling context from policy and current state"""
        
        current_time = datetime.utcnow()
        
        # Extract business hours from policy
        business_hours = None
        time_restrictions = policy.time_restrictions
        if "business_hours" in time_restrictions:
            bh_config = time_restrictions["business_hours"]
            business_hours = BusinessHours(
                timezone=bh_config.get("timezone", self.default_timezone),
                start_time=time.fromisoformat(bh_config.get("start", "09:00")),
                end_time=time.fromisoformat(bh_config.get("end", "17:00")),
                business_days=bh_config.get("days", ["monday", "tuesday", "wednesday", "thursday", "friday"]),
                enabled=bh_config.get("enabled", True)
            )
        
        # Extract maintenance windows
        maintenance_windows = []
        if "maintenance_windows" in time_restrictions:
            for mw_config in time_restrictions["maintenance_windows"]:
                maintenance_windows.append(MaintenanceWindow(
                    name=mw_config["name"],
                    window_type=MaintenanceWindowType(mw_config.get("type", "weekly")),
                    start_time=datetime.fromisoformat(mw_config["start_time"]),
                    end_time=datetime.fromisoformat(mw_config["end_time"]),
                    timezone=mw_config.get("timezone", self.default_timezone),
                    recurrence_pattern=mw_config.get("recurrence_pattern"),
                    enabled=mw_config.get("enabled", True)
                ))
        
        # Extract blackout periods
        blackout_periods = []
        if "blackout_periods" in time_restrictions:
            for bp_config in time_restrictions["blackout_periods"]:
                blackout_periods.append(BlackoutPeriod(
                    name=bp_config["name"],
                    start_time=datetime.fromisoformat(bp_config["start_time"]),
                    end_time=datetime.fromisoformat(bp_config["end_time"]),
                    timezone=bp_config.get("timezone", self.default_timezone),
                    reason=bp_config.get("reason", "Scheduled blackout"),
                    enabled=bp_config.get("enabled", True)
                ))
        
        # Load resource usage patterns (would come from analytics in real system)
        resource_patterns = self._load_resource_usage_patterns()
        
        return SchedulingContext(
            current_time=current_time,
            business_hours=business_hours,
            maintenance_windows=maintenance_windows,
            blackout_periods=blackout_periods,
            resource_patterns=resource_patterns
        )
    
    def _calculate_base_execution_time(self,
                                     action: OptimizationAction,
                                     time_restrictions: Dict[str, Any],
                                     context: SchedulingContext) -> datetime:
        """Calculate base execution time based on action characteristics"""
        
        now = context.current_time
        
        # Base delay based on risk level (this takes priority)
        if action.risk_level == RiskLevel.LOW:
            # Low risk actions can be executed soon
            base_delay = timedelta(minutes=15)
        elif action.risk_level == RiskLevel.MEDIUM:
            # Medium risk actions get more delay
            base_delay = timedelta(hours=1)
        else:  # HIGH risk
            # High risk actions get significant delay but not excessive
            base_delay = timedelta(hours=4)
        
        # Action type modifiers (small adjustments to base delay)
        action_modifiers = {
            ActionType.RELEASE_ELASTIC_IP: timedelta(minutes=-10),  # Reduce delay
            ActionType.UPGRADE_STORAGE: timedelta(minutes=15),     # Small increase
            ActionType.DELETE_VOLUME: timedelta(minutes=30),       # Moderate increase
            ActionType.STOP_INSTANCE: timedelta(minutes=0),        # No change
            ActionType.RESIZE_INSTANCE: timedelta(hours=1),        # Increase delay
            ActionType.TERMINATE_INSTANCE: timedelta(hours=2),     # Significant increase
            ActionType.DELETE_LOAD_BALANCER: timedelta(hours=1),   # Increase delay
            ActionType.CLEANUP_SECURITY_GROUP: timedelta(minutes=30) # Moderate increase
        }
        
        action_modifier = action_modifiers.get(action.action_type, timedelta(0))
        
        # Apply modifier but ensure minimum delay
        final_delay = base_delay + action_modifier
        
        # Ensure minimum delay of 1 minute
        if final_delay < timedelta(minutes=1):
            final_delay = timedelta(minutes=1)
        
        return now + final_delay
    
    def _apply_business_hours_constraints(self,
                                        execution_time: datetime,
                                        action: OptimizationAction,
                                        context: SchedulingContext) -> datetime:
        """Apply business hours constraints to execution time"""
        
        if not context.business_hours or not context.business_hours.enabled:
            return execution_time
        
        bh = context.business_hours
        
        # Convert to business hours timezone
        tz = pytz.timezone(bh.timezone)
        
        # Ensure execution_time is timezone-aware (assume UTC if naive)
        if execution_time.tzinfo is None:
            execution_time = pytz.UTC.localize(execution_time)
        
        local_time = execution_time.astimezone(tz)
        
        # For high-risk actions, avoid business hours
        if action.risk_level == RiskLevel.HIGH:
            # If scheduled during business hours, move to after hours
            if self._is_business_hours(local_time, bh):
                # Move to end of business day
                end_of_business = local_time.replace(
                    hour=bh.end_time.hour,
                    minute=bh.end_time.minute,
                    second=0,
                    microsecond=0
                )
                # Add buffer time after business hours
                execution_time = end_of_business + timedelta(hours=1)
                # Convert back to UTC and make naive
                execution_time = execution_time.astimezone(pytz.UTC).replace(tzinfo=None)
        
        return execution_time
    
    def _avoid_blackout_periods(self,
                              execution_time: datetime,
                              action: OptimizationAction,
                              context: SchedulingContext) -> datetime:
        """Ensure execution time avoids blackout periods"""
        
        for blackout in context.blackout_periods:
            if not blackout.enabled:
                continue
            
            # Convert times to same timezone for comparison
            tz = pytz.timezone(blackout.timezone)
            blackout_start = blackout.start_time.astimezone(tz)
            blackout_end = blackout.end_time.astimezone(tz)
            exec_time_local = execution_time.astimezone(tz)
            
            # Check if execution time falls within blackout period
            if blackout_start <= exec_time_local <= blackout_end:
                logger.info("Execution time conflicts with blackout period",
                           action_id=str(action.id),
                           blackout_name=blackout.name,
                           original_time=execution_time.isoformat())
                
                # Move execution to after blackout period
                execution_time = blackout_end + timedelta(minutes=30)
                # Convert back to UTC
                execution_time = execution_time.astimezone(pytz.UTC).replace(tzinfo=None)
                
                logger.info("Execution time moved to avoid blackout",
                           action_id=str(action.id),
                           new_time=execution_time.isoformat())
        
        return execution_time
    
    def _optimize_for_resource_patterns(self,
                                      execution_time: datetime,
                                      action: OptimizationAction,
                                      context: SchedulingContext) -> datetime:
        """Optimize execution time based on resource usage patterns"""
        
        resource_pattern = context.resource_patterns.get(action.resource_id)
        if not resource_pattern or resource_pattern.confidence_score < 0.5:
            return execution_time
        
        # Convert to resource timezone
        tz = pytz.timezone(resource_pattern.timezone)
        local_time = execution_time.astimezone(tz)
        current_hour = local_time.hour
        
        # If currently scheduled during peak hours, try to move to low usage hours
        if current_hour in resource_pattern.peak_hours:
            # Find next low usage hour
            next_low_hour = None
            for hour in resource_pattern.low_usage_hours:
                if hour > current_hour:
                    next_low_hour = hour
                    break
            
            if next_low_hour is None and resource_pattern.low_usage_hours:
                # Wrap to next day
                next_low_hour = resource_pattern.low_usage_hours[0]
                local_time = local_time + timedelta(days=1)
            
            if next_low_hour is not None:
                # Move to the low usage hour
                optimized_time = local_time.replace(
                    hour=next_low_hour,
                    minute=0,
                    second=0,
                    microsecond=0
                )
                
                # Convert back to UTC
                execution_time = optimized_time.astimezone(pytz.UTC).replace(tzinfo=None)
                
                logger.info("Execution time optimized for resource usage pattern",
                           action_id=str(action.id),
                           resource_id=action.resource_id,
                           optimized_hour=next_low_hour)
        
        return execution_time
    
    def _apply_maintenance_window_preferences(self,
                                            execution_time: datetime,
                                            action: OptimizationAction,
                                            context: SchedulingContext) -> datetime:
        """Apply maintenance window preferences for high-risk actions"""
        
        # Only apply to high-risk actions
        if action.risk_level != RiskLevel.HIGH:
            return execution_time
        
        # Find next suitable maintenance window
        suitable_window = self._find_next_maintenance_window(execution_time, context)
        
        if suitable_window:
            # Only use maintenance window if it's within reasonable time (24 hours)
            window_start = suitable_window.start_time
            
            # Ensure consistent timezone handling
            current_time = context.current_time
            if window_start.tzinfo is not None and current_time.tzinfo is None:
                current_time = pytz.UTC.localize(current_time)
            elif window_start.tzinfo is None and current_time.tzinfo is not None:
                window_start = pytz.UTC.localize(window_start)
            
            time_to_window = window_start - current_time
            
            # If maintenance window is more than 24 hours away, don't use it
            if time_to_window <= timedelta(hours=24):
                # Schedule within the maintenance window
                execution_time = window_start + timedelta(minutes=15)
                
                logger.info("High-risk action scheduled in maintenance window",
                           action_id=str(action.id),
                           window_name=suitable_window.name,
                           execution_time=execution_time.isoformat())
            else:
                logger.info("Maintenance window too far in future, using original schedule",
                           action_id=str(action.id),
                           window_name=suitable_window.name,
                           time_to_window=str(time_to_window))
        
        return execution_time
    
    def _find_next_maintenance_window(self,
                                    after_time: datetime,
                                    context: SchedulingContext) -> Optional[MaintenanceWindow]:
        """Find the next suitable maintenance window after the given time"""
        
        suitable_windows = []
        
        for window in context.maintenance_windows:
            if not window.enabled:
                continue
            
            # For weekly windows, calculate next occurrence
            if window.window_type == MaintenanceWindowType.WEEKLY:
                next_occurrence = self._calculate_next_weekly_occurrence(window, after_time)
                if next_occurrence and next_occurrence.start_time > after_time:
                    suitable_windows.append(next_occurrence)
            
            # For custom windows, check if they're in the future
            elif window.window_type == MaintenanceWindowType.CUSTOM:
                if window.start_time > after_time:
                    suitable_windows.append(window)
        
        # Return the earliest suitable window
        if suitable_windows:
            return min(suitable_windows, key=lambda w: w.start_time)
        
        return None
    
    def _calculate_next_weekly_occurrence(self,
                                        window: MaintenanceWindow,
                                        after_time: datetime) -> Optional[MaintenanceWindow]:
        """Calculate next weekly occurrence of a maintenance window"""
        
        # This is a simplified implementation
        # Real implementation would handle complex recurrence patterns
        
        if not window.recurrence_pattern:
            return None
        
        # Find next occurrence based on day of week
        target_weekday = window.recurrence_pattern.get("weekday", 6)  # Default to Sunday
        days_ahead = target_weekday - after_time.weekday()
        
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        next_date = after_time + timedelta(days=days_ahead)
        next_start = next_date.replace(
            hour=window.start_time.hour,
            minute=window.start_time.minute,
            second=0,
            microsecond=0
        )
        next_end = next_date.replace(
            hour=window.end_time.hour,
            minute=window.end_time.minute,
            second=0,
            microsecond=0
        )
        
        return MaintenanceWindow(
            name=window.name,
            window_type=window.window_type,
            start_time=next_start,
            end_time=next_end,
            timezone=window.timezone,
            recurrence_pattern=window.recurrence_pattern,
            enabled=window.enabled
        )
    
    def _validate_and_adjust_time(self,
                                execution_time: datetime,
                                action: OptimizationAction,
                                context: SchedulingContext) -> datetime:
        """Final validation and adjustment of execution time"""
        
        now = context.current_time
        
        # Ensure both times have consistent timezone info
        if execution_time.tzinfo is not None and now.tzinfo is None:
            now = pytz.UTC.localize(now)
        elif execution_time.tzinfo is None and now.tzinfo is not None:
            execution_time = pytz.UTC.localize(execution_time)
        
        # Ensure execution time is not in the past
        if execution_time <= now:
            execution_time = now + timedelta(minutes=5)
        
        # Ensure minimum delay for safety
        min_delay = timedelta(minutes=1)
        if execution_time - now < min_delay:
            execution_time = now + min_delay
        
        # Round to nearest minute for cleaner scheduling
        execution_time = execution_time.replace(second=0, microsecond=0)
        
        # Ensure we return a naive datetime (consistent with input expectations)
        if execution_time.tzinfo is not None:
            execution_time = execution_time.replace(tzinfo=None)
        
        return execution_time
    
    def _schedule_emergency_action(self,
                                 action: OptimizationAction,
                                 context: SchedulingContext) -> datetime:
        """Schedule action with emergency override"""
        
        # Emergency actions are scheduled immediately with minimal delay
        emergency_delay = timedelta(minutes=2)
        execution_time = context.current_time + emergency_delay
        
        # Log emergency override
        self.audit_logger.log_action_event(
            action.id,
            "emergency_override_applied",
            {
                "override_reason": context.override_reason,
                "authorized_by": context.override_authorized_by,
                "original_risk_level": action.risk_level.value,
                "emergency_execution_time": execution_time.isoformat()
            }
        )
        
        return execution_time
    
    def _is_business_hours(self, local_time: datetime, business_hours: BusinessHours) -> bool:
        """Check if given time is within business hours"""
        
        # Check day of week
        weekday_name = local_time.strftime("%A").lower()
        if weekday_name not in business_hours.business_days:
            return False
        
        # Check time of day
        current_time = local_time.time()
        return business_hours.start_time <= current_time <= business_hours.end_time
    
    def _load_resource_usage_patterns(self) -> Dict[str, ResourceUsagePattern]:
        """Load resource usage patterns from analytics data"""
        
        # In a real implementation, this would query analytics data
        # For now, return empty patterns
        return {}
    
    def _log_scheduling_decision(self,
                               action: OptimizationAction,
                               execution_time: datetime,
                               context: SchedulingContext):
        """Log the scheduling decision for audit purposes"""
        
        self.audit_logger.log_action_event(
            action.id,
            "scheduling_decision",
            {
                "scheduled_execution_time": execution_time.isoformat(),
                "scheduling_factors": {
                    "risk_level": action.risk_level.value,
                    "action_type": action.action_type.value,
                    "business_hours_enabled": context.business_hours.enabled if context.business_hours else False,
                    "maintenance_windows_count": len(context.maintenance_windows),
                    "blackout_periods_count": len(context.blackout_periods),
                    "resource_pattern_available": action.resource_id in context.resource_patterns,
                    "emergency_override": context.emergency_override
                }
            }
        )
    
    def schedule_actions_batch(self,
                             actions: List[OptimizationAction],
                             policy: AutomationPolicy,
                             emergency_override: bool = False,
                             override_reason: Optional[str] = None,
                             authorized_by: Optional[str] = None) -> List[OptimizationAction]:
        """
        Schedule multiple actions as a batch with intelligent timing distribution.
        
        Args:
            actions: List of actions to schedule
            policy: Automation policy
            emergency_override: Whether to apply emergency override
            override_reason: Reason for emergency override
            authorized_by: Who authorized the emergency override
            
        Returns:
            List of actions with updated execution times
        """
        logger.info("Starting batch action scheduling",
                   action_count=len(actions),
                   emergency_override=emergency_override)
        
        # Build scheduling context
        context = self._build_scheduling_context(policy)
        
        # Apply emergency override if specified
        if emergency_override:
            context.emergency_override = True
            context.override_reason = override_reason
            context.override_authorized_by = authorized_by
        
        scheduled_actions = []
        
        # Group actions by risk level for better distribution
        actions_by_risk = {
            RiskLevel.LOW: [],
            RiskLevel.MEDIUM: [],
            RiskLevel.HIGH: []
        }
        
        for action in actions:
            actions_by_risk[action.risk_level].append(action)
        
        # Schedule actions with staggered timing to avoid resource conflicts
        time_offset = timedelta(minutes=0)
        
        # For emergency override, don't add staggered delays
        if emergency_override:
            # Schedule all actions immediately with minimal stagger
            for action in actions:
                execution_time = self.calculate_optimal_execution_time(action, policy, context)
                action.scheduled_execution_time = execution_time
                action.execution_status = ActionStatus.SCHEDULED
                scheduled_actions.append(action)
        else:
            # Normal scheduling with staggered timing
            # Schedule low-risk actions first
            for action in actions_by_risk[RiskLevel.LOW]:
                execution_time = self.calculate_optimal_execution_time(action, policy, context)
                execution_time += time_offset
                action.scheduled_execution_time = execution_time
                action.execution_status = ActionStatus.SCHEDULED
                scheduled_actions.append(action)
                time_offset += timedelta(minutes=5)  # 5-minute stagger
            
            # Schedule medium-risk actions with more spacing
            time_offset = timedelta(minutes=15)
            for action in actions_by_risk[RiskLevel.MEDIUM]:
                execution_time = self.calculate_optimal_execution_time(action, policy, context)
                execution_time += time_offset
                action.scheduled_execution_time = execution_time
                action.execution_status = ActionStatus.SCHEDULED
                scheduled_actions.append(action)
                time_offset += timedelta(minutes=15)  # 15-minute stagger
            
            # Schedule high-risk actions with maximum spacing
            time_offset = timedelta(hours=1)
            for action in actions_by_risk[RiskLevel.HIGH]:
                execution_time = self.calculate_optimal_execution_time(action, policy, context)
                execution_time += time_offset
                action.scheduled_execution_time = execution_time
                action.execution_status = ActionStatus.SCHEDULED
                scheduled_actions.append(action)
                time_offset += timedelta(hours=2)  # 2-hour stagger
        
        logger.info("Batch scheduling completed",
                   scheduled_count=len(scheduled_actions))
        
        return scheduled_actions
    
    def check_maintenance_window_availability(self,
                                            window_name: str,
                                            policy: AutomationPolicy) -> Dict[str, Any]:
        """
        Check availability and capacity of a maintenance window.
        
        Args:
            window_name: Name of the maintenance window
            policy: Automation policy
            
        Returns:
            Dictionary with availability information
        """
        context = self._build_scheduling_context(policy)
        
        # Find the specified maintenance window
        target_window = None
        for window in context.maintenance_windows:
            if window.name == window_name:
                target_window = window
                break
        
        if not target_window:
            return {
                "available": False,
                "reason": "Maintenance window not found"
            }
        
        # Calculate next occurrence
        next_occurrence = self._find_next_maintenance_window(
            context.current_time, 
            SchedulingContext(
                current_time=context.current_time,
                business_hours=None,
                maintenance_windows=[target_window],
                blackout_periods=[],
                resource_patterns={}
            )
        )
        
        if not next_occurrence:
            return {
                "available": False,
                "reason": "No future occurrences found"
            }
        
        # Check for conflicts with blackout periods
        conflicts = []
        for blackout in context.blackout_periods:
            if (blackout.start_time <= next_occurrence.end_time and 
                blackout.end_time >= next_occurrence.start_time):
                conflicts.append(blackout.name)
        
        # Calculate available capacity (simplified)
        window_duration = next_occurrence.end_time - next_occurrence.start_time
        available_slots = int(window_duration.total_seconds() / 3600)  # Assume 1 action per hour
        
        return {
            "available": True,
            "next_occurrence": {
                "start_time": next_occurrence.start_time.isoformat(),
                "end_time": next_occurrence.end_time.isoformat(),
                "duration_hours": window_duration.total_seconds() / 3600
            },
            "available_slots": available_slots,
            "conflicts": conflicts,
            "recommended": len(conflicts) == 0
        }
    
    def create_emergency_override(self,
                                action_ids: List[uuid.UUID],
                                reason: str,
                                authorized_by: str,
                                policy: AutomationPolicy) -> Dict[str, Any]:
        """
        Create emergency override for immediate action execution.
        
        Args:
            action_ids: List of action IDs to override
            reason: Reason for emergency override
            authorized_by: Who authorized the override
            policy: Automation policy
            
        Returns:
            Dictionary with override results
        """
        logger.info("Creating emergency override",
                   action_count=len(action_ids),
                   reason=reason,
                   authorized_by=authorized_by)
        
        results = {
            "override_id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "authorized_by": authorized_by,
            "actions": []
        }
        
        with get_db_session() as session:
            for action_id in action_ids:
                action = session.query(OptimizationAction).filter_by(id=action_id).first()
                
                if not action:
                    results["actions"].append({
                        "action_id": str(action_id),
                        "status": "not_found",
                        "message": "Action not found"
                    })
                    continue
                
                # Check if action can be overridden
                if action.execution_status not in [ActionStatus.PENDING, ActionStatus.SCHEDULED]:
                    results["actions"].append({
                        "action_id": str(action_id),
                        "status": "invalid_status",
                        "message": f"Action status is {action.execution_status.value}, cannot override"
                    })
                    continue
                
                # Apply emergency scheduling
                context = self._build_scheduling_context(policy)
                context.emergency_override = True
                context.override_reason = reason
                context.override_authorized_by = authorized_by
                
                emergency_time = self._schedule_emergency_action(action, context)
                
                # Update action
                action.scheduled_execution_time = emergency_time
                action.execution_status = ActionStatus.SCHEDULED
                
                # Log override
                self.audit_logger.log_action_event(
                    action.id,
                    "emergency_override_created",
                    {
                        "override_id": results["override_id"],
                        "reason": reason,
                        "authorized_by": authorized_by,
                        "original_execution_time": action.scheduled_execution_time.isoformat() if action.scheduled_execution_time else None,
                        "emergency_execution_time": emergency_time.isoformat()
                    }
                )
                
                results["actions"].append({
                    "action_id": str(action_id),
                    "status": "overridden",
                    "emergency_execution_time": emergency_time.isoformat(),
                    "message": "Emergency override applied successfully"
                })
            
            session.commit()
        
        logger.info("Emergency override created",
                   override_id=results["override_id"],
                   successful_overrides=len([a for a in results["actions"] if a["status"] == "overridden"]))
        
        return results