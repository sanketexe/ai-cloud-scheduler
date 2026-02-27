"""
Decision Tracking and Notification System for Approval Workflows

This module implements comprehensive decision tracking and intelligent notification
routing for the approval workflow engine, providing immediate notifications to
session participants and role-based notification routing.

Requirements addressed:
- 2.3: Intelligent notification routing based on roles and urgency
- 2.4: Immediate notification to all session participants
- Approval status tracking and progress monitoring
"""

import uuid
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from sqlalchemy.exc import IntegrityError

from .database import get_db_session
from .models import User, UserRole
from .collaboration_models import CollaborativeSession, SessionParticipant
from .redis_config import redis_manager

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Types of notifications"""
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_UPDATE = "approval_update"
    APPROVAL_ESCALATION = "approval_escalation"
    DEADLINE_WARNING = "deadline_warning"
    SESSION_UPDATE = "session_update"
    URGENT_DECISION = "urgent_decision"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class NotificationRule:
    """Notification routing rule"""
    rule_id: str
    user_role: str
    notification_types: List[NotificationType]
    channels: List[NotificationChannel]
    priority_threshold: NotificationPriority
    quiet_hours: Optional[Dict[str, str]] = None
    escalation_delay_minutes: int = 30

@dataclass
class NotificationPreference:
    """User notification preferences"""
    user_id: str
    enabled_channels: List[NotificationChannel]
    quiet_hours: Optional[Dict[str, str]] = None
    priority_threshold: NotificationPriority = NotificationPriority.LOW
    aggregation_enabled: bool = True
    aggregation_window_minutes: int = 15

@dataclass
class NotificationMessage:
    """Notification message structure"""
    message_id: str
    recipient_id: str
    notification_type: NotificationType
    priority: NotificationPriority
    title: str
    content: str
    data: Dict[str, Any]
    channels: List[NotificationChannel]
    created_at: datetime
    expires_at: Optional[datetime] = None
    session_id: Optional[str] = None

@dataclass
class DecisionTrackingEntry:
    """Decision tracking entry"""
    entry_id: str
    decision_id: str
    session_id: str
    participant_id: str
    action_type: str
    timestamp: datetime
    data: Dict[str, Any]
    context: Dict[str, Any]

@dataclass
class NotificationDeliveryResult:
    """Result of notification delivery"""
    success: bool
    channel: NotificationChannel
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

class DecisionTrackingSystem:
    """
    Tracks all decisions and their progress through approval workflows
    """
    
    def __init__(self):
        self.active_decisions: Dict[str, Dict[str, Any]] = {}
        self.decision_history: Dict[str, List[DecisionTrackingEntry]] = {}
    
    async def track_decision_created(self, decision_id: str, session_id: str, 
                                   participant_id: str, decision_data: Dict[str, Any]) -> bool:
        """
        Track when a new decision is created
        
        Args:
            decision_id: Unique identifier for the decision
            session_id: ID of the collaborative session
            participant_id: ID of the participant who created the decision
            decision_data: Complete decision data
            
        Returns:
            bool: True if tracking was successful
        """
        try:
            entry = DecisionTrackingEntry(
                entry_id=str(uuid.uuid4()),
                decision_id=decision_id,
                session_id=session_id,
                participant_id=participant_id,
                action_type="decision_created",
                timestamp=datetime.utcnow(),
                data=decision_data,
                context={"status": "created"}
            )
            
            # Store in memory
            if decision_id not in self.decision_history:
                self.decision_history[decision_id] = []
            self.decision_history[decision_id].append(entry)
            
            # Store in Redis for persistence
            await self._persist_decision_entry(entry)
            
            # Initialize active decision tracking
            self.active_decisions[decision_id] = {
                "status": "created",
                "session_id": session_id,
                "created_by": participant_id,
                "created_at": entry.timestamp,
                "last_updated": entry.timestamp,
                "approval_progress": 0,
                "current_approver": None
            }
            
            logger.info(f"Started tracking decision {decision_id} from session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking decision creation: {e}")
            return False
    
    async def track_approval_submitted(self, decision_id: str, approval_request_id: str, 
                                     approver_id: str, approval_data: Dict[str, Any]) -> bool:
        """
        Track when a decision is submitted for approval
        
        Args:
            decision_id: ID of the decision
            approval_request_id: ID of the approval request
            approver_id: ID of the current approver
            approval_data: Approval request data
            
        Returns:
            bool: True if tracking was successful
        """
        try:
            entry = DecisionTrackingEntry(
                entry_id=str(uuid.uuid4()),
                decision_id=decision_id,
                session_id=self.active_decisions.get(decision_id, {}).get("session_id", ""),
                participant_id=approver_id,
                action_type="approval_submitted",
                timestamp=datetime.utcnow(),
                data=approval_data,
                context={
                    "approval_request_id": approval_request_id,
                    "status": "pending_approval"
                }
            )
            
            # Add to history
            if decision_id not in self.decision_history:
                self.decision_history[decision_id] = []
            self.decision_history[decision_id].append(entry)
            
            # Update active decision
            if decision_id in self.active_decisions:
                self.active_decisions[decision_id].update({
                    "status": "pending_approval",
                    "approval_request_id": approval_request_id,
                    "current_approver": approver_id,
                    "last_updated": entry.timestamp
                })
            
            await self._persist_decision_entry(entry)
            
            logger.info(f"Tracked approval submission for decision {decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking approval submission: {e}")
            return False
    
    async def track_approval_decision(self, decision_id: str, approver_id: str, 
                                    approved: bool, comments: str, next_step: Optional[str] = None) -> bool:
        """
        Track an approval decision
        
        Args:
            decision_id: ID of the decision
            approver_id: ID of the approver
            approved: Whether the decision was approved
            comments: Approver comments
            next_step: Next step in the approval process
            
        Returns:
            bool: True if tracking was successful
        """
        try:
            entry = DecisionTrackingEntry(
                entry_id=str(uuid.uuid4()),
                decision_id=decision_id,
                session_id=self.active_decisions.get(decision_id, {}).get("session_id", ""),
                participant_id=approver_id,
                action_type="approval_decision",
                timestamp=datetime.utcnow(),
                data={
                    "approved": approved,
                    "comments": comments,
                    "next_step": next_step
                },
                context={
                    "status": "approved" if approved else "rejected"
                }
            )
            
            # Add to history
            if decision_id not in self.decision_history:
                self.decision_history[decision_id] = []
            self.decision_history[decision_id].append(entry)
            
            # Update active decision
            if decision_id in self.active_decisions:
                status = "approved" if approved else "rejected"
                if next_step and approved:
                    status = "pending_approval"  # Still pending if there's a next step
                
                self.active_decisions[decision_id].update({
                    "status": status,
                    "last_updated": entry.timestamp,
                    "current_approver": next_step if approved and next_step else None
                })
            
            await self._persist_decision_entry(entry)
            
            logger.info(f"Tracked approval decision for {decision_id}: {'approved' if approved else 'rejected'}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking approval decision: {e}")
            return False
    
    async def track_decision_implemented(self, decision_id: str, implementation_data: Dict[str, Any]) -> bool:
        """
        Track when a decision is implemented
        
        Args:
            decision_id: ID of the decision
            implementation_data: Implementation details
            
        Returns:
            bool: True if tracking was successful
        """
        try:
            entry = DecisionTrackingEntry(
                entry_id=str(uuid.uuid4()),
                decision_id=decision_id,
                session_id=self.active_decisions.get(decision_id, {}).get("session_id", ""),
                participant_id="system",  # Implementation is typically automated
                action_type="decision_implemented",
                timestamp=datetime.utcnow(),
                data=implementation_data,
                context={"status": "implemented"}
            )
            
            # Add to history
            if decision_id not in self.decision_history:
                self.decision_history[decision_id] = []
            self.decision_history[decision_id].append(entry)
            
            # Update active decision
            if decision_id in self.active_decisions:
                self.active_decisions[decision_id].update({
                    "status": "implemented",
                    "last_updated": entry.timestamp,
                    "implemented_at": entry.timestamp
                })
            
            await self._persist_decision_entry(entry)
            
            logger.info(f"Tracked implementation of decision {decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking decision implementation: {e}")
            return False
    
    async def get_decision_status(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a decision
        
        Args:
            decision_id: ID of the decision
            
        Returns:
            Dict containing decision status and progress
        """
        try:
            if decision_id not in self.active_decisions:
                # Try to load from cache
                await self._load_decision_from_cache(decision_id)
            
            if decision_id in self.active_decisions:
                decision_data = self.active_decisions[decision_id]
                history = self.decision_history.get(decision_id, [])
                
                return {
                    "decision_id": decision_id,
                    "status": decision_data["status"],
                    "session_id": decision_data["session_id"],
                    "created_by": decision_data["created_by"],
                    "created_at": decision_data["created_at"].isoformat(),
                    "last_updated": decision_data["last_updated"].isoformat(),
                    "current_approver": decision_data.get("current_approver"),
                    "approval_progress": decision_data.get("approval_progress", 0),
                    "history_count": len(history),
                    "timeline": [
                        {
                            "action": entry.action_type,
                            "timestamp": entry.timestamp.isoformat(),
                            "participant": entry.participant_id,
                            "data": entry.data
                        }
                        for entry in history[-10:]  # Last 10 entries
                    ]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting decision status: {e}")
            return None
    
    async def get_session_decisions(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all decisions for a session
        
        Args:
            session_id: ID of the collaborative session
            
        Returns:
            List of decision summaries
        """
        try:
            session_decisions = []
            
            for decision_id, decision_data in self.active_decisions.items():
                if decision_data.get("session_id") == session_id:
                    status = await self.get_decision_status(decision_id)
                    if status:
                        session_decisions.append(status)
            
            return session_decisions
            
        except Exception as e:
            logger.error(f"Error getting session decisions: {e}")
            return []
    
    async def _persist_decision_entry(self, entry: DecisionTrackingEntry):
        """Persist decision entry to Redis"""
        try:
            cache_key = f"decision_history:{entry.decision_id}"
            
            # Get Redis client
            redis_client = await redis_manager.get_async_client()
            
            # Get existing history
            existing_data = await redis_client.get(cache_key)
            existing_history = json.loads(existing_data) if existing_data else []
            
            # Add new entry
            entry_data = {
                "entry_id": entry.entry_id,
                "decision_id": entry.decision_id,
                "session_id": entry.session_id,
                "participant_id": entry.participant_id,
                "action_type": entry.action_type,
                "timestamp": entry.timestamp.isoformat(),
                "data": entry.data,
                "context": entry.context
            }
            
            existing_history.append(entry_data)
            
            # Store updated history
            await redis_client.set(cache_key, json.dumps(existing_history), ex=86400 * 30)  # 30 days
            
        except Exception as e:
            logger.error(f"Error persisting decision entry: {e}")
    
    async def _load_decision_from_cache(self, decision_id: str):
        """Load decision data from Redis cache"""
        try:
            cache_key = f"decision_history:{decision_id}"
            redis_client = await redis_manager.get_async_client()
            
            history_data_str = await redis_client.get(cache_key)
            
            if history_data_str:
                history_data = json.loads(history_data_str)
            
            if history_data:
                # Reconstruct decision history
                history = []
                for entry_data in history_data:
                    entry = DecisionTrackingEntry(
                        entry_id=entry_data["entry_id"],
                        decision_id=entry_data["decision_id"],
                        session_id=entry_data["session_id"],
                        participant_id=entry_data["participant_id"],
                        action_type=entry_data["action_type"],
                        timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                        data=entry_data["data"],
                        context=entry_data["context"]
                    )
                    history.append(entry)
                
                self.decision_history[decision_id] = history
                
                # Reconstruct active decision data from latest entries
                if history:
                    latest_entry = history[-1]
                    self.active_decisions[decision_id] = {
                        "status": latest_entry.context.get("status", "unknown"),
                        "session_id": latest_entry.session_id,
                        "created_by": history[0].participant_id if history else "unknown",
                        "created_at": history[0].timestamp if history else datetime.utcnow(),
                        "last_updated": latest_entry.timestamp,
                        "approval_progress": 0,  # Would need to calculate from history
                        "current_approver": None  # Would need to determine from latest state
                    }
            
        except Exception as e:
            logger.error(f"Error loading decision from cache: {e}")

class NotificationRoutingSystem:
    """
    Intelligent notification routing system that delivers notifications based on
    roles, urgency, and user preferences
    """
    
    def __init__(self):
        self.notification_rules: Dict[str, NotificationRule] = {}
        self.user_preferences: Dict[str, NotificationPreference] = {}
        self.pending_notifications: Dict[str, List[NotificationMessage]] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default notification routing rules"""
        # Admin rules
        admin_rule = NotificationRule(
            rule_id="admin_notifications",
            user_role="admin",
            notification_types=[nt for nt in NotificationType],
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            priority_threshold=NotificationPriority.LOW
        )
        
        # Manager rules
        manager_rule = NotificationRule(
            rule_id="manager_notifications",
            user_role="finance_manager",
            notification_types=[
                NotificationType.APPROVAL_REQUEST,
                NotificationType.APPROVAL_ESCALATION,
                NotificationType.URGENT_DECISION
            ],
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL, NotificationChannel.SMS],
            priority_threshold=NotificationPriority.MEDIUM,
            escalation_delay_minutes=15
        )
        
        # Analyst rules
        analyst_rule = NotificationRule(
            rule_id="analyst_notifications",
            user_role="analyst",
            notification_types=[
                NotificationType.APPROVAL_UPDATE,
                NotificationType.SESSION_UPDATE
            ],
            channels=[NotificationChannel.IN_APP],
            priority_threshold=NotificationPriority.MEDIUM
        )
        
        # Viewer rules
        viewer_rule = NotificationRule(
            rule_id="viewer_notifications",
            user_role="viewer",
            notification_types=[NotificationType.SESSION_UPDATE],
            channels=[NotificationChannel.IN_APP],
            priority_threshold=NotificationPriority.HIGH
        )
        
        self.notification_rules = {
            "admin": admin_rule,
            "finance_manager": manager_rule,
            "analyst": analyst_rule,
            "viewer": viewer_rule
        }
    
    async def send_notification(self, recipient_id: str, notification_type: NotificationType,
                              data: Dict[str, Any], priority: str = "medium",
                              session_id: Optional[str] = None) -> bool:
        """
        Send a notification with intelligent routing
        
        Args:
            recipient_id: ID of the notification recipient
            notification_type: Type of notification
            data: Notification data
            priority: Priority level
            session_id: Optional session ID for context
            
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            # Get user role and preferences
            user_role = await self._get_user_role(recipient_id)
            user_prefs = await self._get_user_preferences(recipient_id)
            
            # Determine notification channels based on rules and preferences
            channels = await self._determine_channels(user_role, notification_type, priority, user_prefs)
            
            if not channels:
                logger.info(f"No channels available for notification to {recipient_id}")
                return True  # Not an error, just filtered out
            
            # Create notification message
            message = NotificationMessage(
                message_id=str(uuid.uuid4()),
                recipient_id=recipient_id,
                notification_type=notification_type,
                priority=NotificationPriority(priority),
                title=self._generate_title(notification_type, data),
                content=self._generate_content(notification_type, data),
                data=data,
                channels=channels,
                created_at=datetime.utcnow(),
                session_id=session_id
            )
            
            # Check for aggregation
            if user_prefs and user_prefs.aggregation_enabled:
                if await self._should_aggregate(message, user_prefs):
                    await self._add_to_aggregation(message)
                    return True
            
            # Send immediately
            success = await self._deliver_notification(message)
            
            # Store for audit
            await self._store_notification_audit(message, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
    
    async def send_session_notification(self, session_id: str, notification_type: NotificationType,
                                      data: Dict[str, Any], priority: str = "medium",
                                      exclude_participants: List[str] = None) -> Dict[str, bool]:
        """
        Send notification to all session participants
        
        Args:
            session_id: ID of the collaborative session
            notification_type: Type of notification
            data: Notification data
            priority: Priority level
            exclude_participants: List of participant IDs to exclude
            
        Returns:
            Dict mapping participant IDs to delivery success status
        """
        try:
            # Get session participants
            participants = await self._get_session_participants(session_id)
            
            if exclude_participants:
                participants = [p for p in participants if p not in exclude_participants]
            
            results = {}
            
            # Send to each participant
            for participant_id in participants:
                success = await self.send_notification(
                    recipient_id=participant_id,
                    notification_type=notification_type,
                    data=data,
                    priority=priority,
                    session_id=session_id
                )
                results[participant_id] = success
            
            logger.info(f"Sent session notification to {len(participants)} participants")
            return results
            
        except Exception as e:
            logger.error(f"Error sending session notification: {e}")
            return {}
    
    async def escalate_notification(self, original_message_id: str, escalation_targets: List[str],
                                  reason: str) -> bool:
        """
        Escalate a notification to higher-level recipients
        
        Args:
            original_message_id: ID of the original notification
            escalation_targets: List of escalation target IDs
            reason: Reason for escalation
            
        Returns:
            bool: True if escalation was successful
        """
        try:
            # Get original message data (would be stored in audit)
            original_data = await self._get_notification_audit(original_message_id)
            
            if not original_data:
                logger.error(f"Original notification {original_message_id} not found for escalation")
                return False
            
            # Create escalation notification
            escalation_data = {
                **original_data["data"],
                "escalation_reason": reason,
                "original_message_id": original_message_id,
                "escalated_at": datetime.utcnow().isoformat()
            }
            
            # Send to escalation targets
            success_count = 0
            for target_id in escalation_targets:
                success = await self.send_notification(
                    recipient_id=target_id,
                    notification_type=NotificationType.APPROVAL_ESCALATION,
                    data=escalation_data,
                    priority="high"
                )
                if success:
                    success_count += 1
            
            logger.info(f"Escalated notification to {success_count}/{len(escalation_targets)} targets")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error escalating notification: {e}")
            return False
    
    async def _get_user_role(self, user_id: str) -> str:
        """Get user role from database"""
        try:
            async with get_db_session() as db:
                user = db.query(User).filter(User.id == user_id).first()
                return user.role.value if user else "viewer"
        except Exception as e:
            logger.error(f"Error getting user role: {e}")
            return "viewer"
    
    async def _get_user_preferences(self, user_id: str) -> Optional[NotificationPreference]:
        """Get user notification preferences"""
        try:
            # Check cache first
            cache_key = f"user_prefs:{user_id}"
            redis_client = await redis_manager.get_async_client()
            prefs_data_str = await redis_client.get(cache_key)
            
            if prefs_data_str:
                prefs_data = json.loads(prefs_data_str)
                return NotificationPreference(
                    user_id=prefs_data["user_id"],
                    enabled_channels=[NotificationChannel(ch) for ch in prefs_data["enabled_channels"]],
                    quiet_hours=prefs_data.get("quiet_hours"),
                    priority_threshold=NotificationPriority(prefs_data["priority_threshold"]),
                    aggregation_enabled=prefs_data.get("aggregation_enabled", True),
                    aggregation_window_minutes=prefs_data.get("aggregation_window_minutes", 15)
                )
            
            # Return default preferences
            return NotificationPreference(
                user_id=user_id,
                enabled_channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
                priority_threshold=NotificationPriority.LOW
            )
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return None
    
    async def _determine_channels(self, user_role: str, notification_type: NotificationType,
                                priority: str, user_prefs: Optional[NotificationPreference]) -> List[NotificationChannel]:
        """Determine appropriate notification channels"""
        try:
            # Get role-based rules
            rule = self.notification_rules.get(user_role)
            if not rule:
                return [NotificationChannel.IN_APP]  # Default fallback
            
            # Check if notification type is allowed for this role
            if notification_type not in rule.notification_types:
                return []
            
            # Check priority threshold
            priority_enum = NotificationPriority(priority)
            if priority_enum.value < rule.priority_threshold.value:
                return []
            
            # Start with rule channels
            channels = rule.channels.copy()
            
            # Apply user preferences if available
            if user_prefs:
                # Filter by user's enabled channels
                channels = [ch for ch in channels if ch in user_prefs.enabled_channels]
                
                # Check user's priority threshold
                if priority_enum.value < user_prefs.priority_threshold.value:
                    # Only use in-app for low priority
                    channels = [NotificationChannel.IN_APP] if NotificationChannel.IN_APP in channels else []
            
            return channels
            
        except Exception as e:
            logger.error(f"Error determining channels: {e}")
            return [NotificationChannel.IN_APP]
    
    def _generate_title(self, notification_type: NotificationType, data: Dict[str, Any]) -> str:
        """Generate notification title"""
        titles = {
            NotificationType.APPROVAL_REQUEST: f"Approval Required: {data.get('decision_type', 'Decision')}",
            NotificationType.APPROVAL_UPDATE: f"Approval Update: {data.get('status', 'Status Changed')}",
            NotificationType.APPROVAL_ESCALATION: f"Escalated Approval: {data.get('decision_type', 'Decision')}",
            NotificationType.DEADLINE_WARNING: f"Deadline Warning: {data.get('decision_type', 'Decision')}",
            NotificationType.SESSION_UPDATE: f"Session Update: {data.get('update_type', 'Activity')}",
            NotificationType.URGENT_DECISION: f"Urgent Decision Required: {data.get('decision_type', 'Action')}"
        }
        
        return titles.get(notification_type, "FinOps Notification")
    
    def _generate_content(self, notification_type: NotificationType, data: Dict[str, Any]) -> str:
        """Generate notification content"""
        if notification_type == NotificationType.APPROVAL_REQUEST:
            return f"A {data.get('decision_type', 'decision')} requiring approval has been submitted. " \
                   f"Cost impact: ${data.get('cost_impact', 0):,.2f}. " \
                   f"Priority: {data.get('priority', 'medium')}."
        
        elif notification_type == NotificationType.APPROVAL_UPDATE:
            return f"Approval status has been updated to: {data.get('status', 'unknown')}. " \
                   f"Comments: {data.get('comments', 'No comments provided')}."
        
        elif notification_type == NotificationType.APPROVAL_ESCALATION:
            return f"An approval has been escalated due to: {data.get('reason', 'timeout')}. " \
                   f"Your immediate attention is required."
        
        elif notification_type == NotificationType.SESSION_UPDATE:
            return f"Session activity: {data.get('update_type', 'update')}. " \
                   f"Participants: {len(data.get('participants', []))}."
        
        else:
            return f"FinOps notification: {data.get('description', 'No description available')}."
    
    async def _should_aggregate(self, message: NotificationMessage, 
                              user_prefs: NotificationPreference) -> bool:
        """Check if notification should be aggregated"""
        try:
            # Don't aggregate critical notifications
            if message.priority == NotificationPriority.CRITICAL:
                return False
            
            # Check if there are pending notifications for this user
            pending = self.pending_notifications.get(message.recipient_id, [])
            
            if not pending:
                return False
            
            # Check if any pending notifications are of the same type
            same_type_pending = [n for n in pending if n.notification_type == message.notification_type]
            
            # Aggregate if there are similar notifications within the window
            if same_type_pending:
                latest_pending = max(same_type_pending, key=lambda n: n.created_at)
                time_diff = (message.created_at - latest_pending.created_at).total_seconds() / 60
                
                return time_diff <= user_prefs.aggregation_window_minutes
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking aggregation: {e}")
            return False
    
    async def _add_to_aggregation(self, message: NotificationMessage):
        """Add notification to aggregation queue"""
        try:
            if message.recipient_id not in self.pending_notifications:
                self.pending_notifications[message.recipient_id] = []
            
            self.pending_notifications[message.recipient_id].append(message)
            
            # Schedule aggregated delivery
            await self._schedule_aggregated_delivery(message.recipient_id)
            
        except Exception as e:
            logger.error(f"Error adding to aggregation: {e}")
    
    async def _deliver_notification(self, message: NotificationMessage) -> bool:
        """Deliver notification through specified channels"""
        try:
            delivery_results = []
            
            for channel in message.channels:
                result = await self._deliver_via_channel(message, channel)
                delivery_results.append(result)
            
            # Consider successful if at least one channel succeeded
            return any(result.success for result in delivery_results)
            
        except Exception as e:
            logger.error(f"Error delivering notification: {e}")
            return False
    
    async def _deliver_via_channel(self, message: NotificationMessage, 
                                 channel: NotificationChannel) -> NotificationDeliveryResult:
        """Deliver notification via specific channel"""
        try:
            if channel == NotificationChannel.IN_APP:
                # Store in Redis for in-app retrieval
                await self._store_in_app_notification(message)
                return NotificationDeliveryResult(
                    success=True,
                    channel=channel,
                    delivered_at=datetime.utcnow()
                )
            
            elif channel == NotificationChannel.EMAIL:
                # Would integrate with email service
                logger.info(f"Email notification sent to {message.recipient_id}: {message.title}")
                return NotificationDeliveryResult(
                    success=True,
                    channel=channel,
                    delivered_at=datetime.utcnow()
                )
            
            elif channel == NotificationChannel.SMS:
                # Would integrate with SMS service
                logger.info(f"SMS notification sent to {message.recipient_id}: {message.title}")
                return NotificationDeliveryResult(
                    success=True,
                    channel=channel,
                    delivered_at=datetime.utcnow()
                )
            
            else:
                # Other channels would be implemented similarly
                logger.info(f"Notification sent via {channel.value} to {message.recipient_id}")
                return NotificationDeliveryResult(
                    success=True,
                    channel=channel,
                    delivered_at=datetime.utcnow()
                )
            
        except Exception as e:
            logger.error(f"Error delivering via {channel.value}: {e}")
            return NotificationDeliveryResult(
                success=False,
                channel=channel,
                error_message=str(e)
            )
    
    async def _store_in_app_notification(self, message: NotificationMessage):
        """Store notification for in-app retrieval"""
        try:
            cache_key = f"notifications:{message.recipient_id}"
            redis_client = await redis_manager.get_async_client()
            
            # Get existing notifications
            existing_data = await redis_client.get(cache_key)
            existing = json.loads(existing_data) if existing_data else []
            
            # Add new notification
            notification_data = {
                "message_id": message.message_id,
                "type": message.notification_type.value,
                "priority": message.priority.value,
                "title": message.title,
                "content": message.content,
                "data": message.data,
                "created_at": message.created_at.isoformat(),
                "read": False,
                "session_id": message.session_id
            }
            
            existing.insert(0, notification_data)  # Add to beginning
            
            # Keep only last 100 notifications
            existing = existing[:100]
            
            # Store updated notifications
            await redis_client.set(cache_key, json.dumps(existing), ex=86400 * 7)  # 7 days
            
        except Exception as e:
            logger.error(f"Error storing in-app notification: {e}")
    
    async def _store_notification_audit(self, message: NotificationMessage, success: bool):
        """Store notification audit trail"""
        try:
            audit_key = f"notification_audit:{message.message_id}"
            redis_client = await redis_manager.get_async_client()
            
            audit_data = {
                "message_id": message.message_id,
                "recipient_id": message.recipient_id,
                "notification_type": message.notification_type.value,
                "priority": message.priority.value,
                "channels": [ch.value for ch in message.channels],
                "success": success,
                "created_at": message.created_at.isoformat(),
                "data": message.data
            }
            
            await redis_client.set(audit_key, json.dumps(audit_data), ex=86400 * 30)  # 30 days
            
        except Exception as e:
            logger.error(f"Error storing notification audit: {e}")
    
    async def _get_notification_audit(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get notification audit data"""
        try:
            audit_key = f"notification_audit:{message_id}"
            redis_client = await redis_manager.get_async_client()
            
            audit_data_str = await redis_client.get(audit_key)
            return json.loads(audit_data_str) if audit_data_str else None
        except Exception as e:
            logger.error(f"Error getting notification audit: {e}")
            return None
    
    async def _get_session_participants(self, session_id: str) -> List[str]:
        """Get list of session participant IDs"""
        try:
            async with get_db_session() as db:
                participants = db.query(SessionParticipant).filter(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.left_at.is_(None)
                ).all()
                
                return [str(p.user_id) for p in participants]
                
        except Exception as e:
            logger.error(f"Error getting session participants: {e}")
            return []
    
    async def _schedule_aggregated_delivery(self, recipient_id: str):
        """Schedule delivery of aggregated notifications"""
        # This would integrate with a task scheduler
        # For now, we'll just log the scheduling
        logger.info(f"Scheduled aggregated notification delivery for {recipient_id}")

# Global instances
decision_tracker = DecisionTrackingSystem()
notification_service = NotificationRoutingSystem()