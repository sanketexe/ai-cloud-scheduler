"""
Policy Manager for Automated Cost Optimization

Manages user-defined automation policies and approval workflows:
- Policy configuration with validation
- Approval workflow management
- Policy violation detection
- Dry run mode support
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import structlog

from .automation_models import (
    AutomationPolicy, AutomationLevel, ActionType, ActionApproval, 
    ApprovalStatus, OptimizationAction
)
from .database import get_db_session
from .automation_audit_logger import AutomationAuditLogger
from .notification_service import get_notification_service, NotificationMessage, NotificationPriority

logger = structlog.get_logger(__name__)


@dataclass
class PolicyValidationResult:
    """Result of policy validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class PolicyManager:
    """
    Manages automation policies and approval workflows.
    
    Provides capabilities to:
    - Create and validate automation policies
    - Manage approval workflows
    - Detect policy violations
    - Support dry run mode operations
    """
    
    def __init__(self):
        self.audit_logger = AutomationAuditLogger()
        self.notification_service = get_notification_service()
    
    def create_policy(self,
                     name: str,
                     automation_level: AutomationLevel,
                     enabled_actions: List[str],
                     approval_required_actions: List[str],
                     blocked_actions: List[str],
                     resource_filters: Dict[str, Any],
                     time_restrictions: Dict[str, Any],
                     safety_overrides: Dict[str, Any],
                     created_by: uuid.UUID) -> Tuple[Optional[AutomationPolicy], PolicyValidationResult]:
        """
        Create a new automation policy with validation.
        
        Args:
            name: Policy name
            automation_level: Level of automation (conservative, balanced, aggressive)
            enabled_actions: List of enabled action types
            approval_required_actions: List of actions requiring approval
            blocked_actions: List of blocked action types
            resource_filters: Resource filtering rules
            time_restrictions: Time-based restrictions
            safety_overrides: Safety rule overrides
            created_by: User ID creating the policy
            
        Returns:
            Tuple of (created_policy, validation_result)
        """
        
        logger.info("Creating automation policy",
                   name=name,
                   automation_level=automation_level.value)
        
        # Validate policy configuration
        validation_result = self.validate_policy_configuration(
            automation_level=automation_level,
            enabled_actions=enabled_actions,
            approval_required_actions=approval_required_actions,
            blocked_actions=blocked_actions,
            resource_filters=resource_filters,
            time_restrictions=time_restrictions,
            safety_overrides=safety_overrides
        )
        
        if not validation_result.is_valid:
            logger.warning("Policy validation failed",
                          name=name,
                          errors=validation_result.errors)
            return None, validation_result
        
        try:
            with get_db_session() as session:
                policy = AutomationPolicy(
                    name=name,
                    automation_level=automation_level,
                    enabled_actions=enabled_actions,
                    approval_required_actions=approval_required_actions,
                    blocked_actions=blocked_actions,
                    resource_filters=resource_filters,
                    time_restrictions=time_restrictions,
                    safety_overrides=safety_overrides,
                    created_by=created_by
                )
                
                session.add(policy)
                session.commit()
                
                # Log policy creation
                self.audit_logger.log_policy_event(
                    policy.id,
                    "policy_created",
                    {
                        "policy_name": name,
                        "automation_level": automation_level.value,
                        "enabled_actions": enabled_actions,
                        "approval_required_actions": approval_required_actions,
                        "blocked_actions": blocked_actions
                    },
                    {"user_id": str(created_by)}
                )
                
                logger.info("Automation policy created successfully",
                           policy_id=str(policy.id),
                           name=name)
                
                return policy, validation_result
                
        except Exception as e:
            logger.error("Failed to create automation policy",
                        name=name,
                        error=str(e))
            validation_result.errors.append(f"Database error: {str(e)}")
            return None, validation_result
    
    def validate_policy_configuration(self,
                                    automation_level: AutomationLevel,
                                    enabled_actions: List[str],
                                    approval_required_actions: List[str],
                                    blocked_actions: List[str],
                                    resource_filters: Dict[str, Any],
                                    time_restrictions: Dict[str, Any],
                                    safety_overrides: Dict[str, Any]) -> PolicyValidationResult:
        """
        Validate automation policy configuration.
        
        Returns:
            PolicyValidationResult with validation details
        """
        
        errors = []
        warnings = []
        
        # Validate action types
        valid_action_types = [action.value for action in ActionType]
        
        for action in enabled_actions:
            if action not in valid_action_types:
                errors.append(f"Invalid enabled action type: {action}")
        
        for action in approval_required_actions:
            if action not in valid_action_types:
                errors.append(f"Invalid approval required action type: {action}")
        
        for action in blocked_actions:
            if action not in valid_action_types:
                errors.append(f"Invalid blocked action type: {action}")
        
        # Check for conflicting action configurations
        enabled_set = set(enabled_actions)
        blocked_set = set(blocked_actions)
        approval_set = set(approval_required_actions)
        
        # Actions cannot be both enabled and blocked
        conflicting_enabled_blocked = enabled_set.intersection(blocked_set)
        if conflicting_enabled_blocked:
            errors.append(f"Actions cannot be both enabled and blocked: {list(conflicting_enabled_blocked)}")
        
        # Approval required actions should be in enabled actions
        approval_not_enabled = approval_set - enabled_set
        if approval_not_enabled:
            warnings.append(f"Approval required actions not in enabled actions: {list(approval_not_enabled)}")
        
        # Validate automation level consistency
        if automation_level == AutomationLevel.CONSERVATIVE:
            high_risk_actions = [ActionType.TERMINATE_INSTANCE.value, ActionType.DELETE_LOAD_BALANCER.value]
            enabled_high_risk = enabled_set.intersection(set(high_risk_actions))
            if enabled_high_risk:
                warnings.append(f"Conservative policy enables high-risk actions: {list(enabled_high_risk)}")
        
        elif automation_level == AutomationLevel.AGGRESSIVE:
            if len(approval_required_actions) > len(enabled_actions) * 0.5:
                warnings.append("Aggressive policy requires approval for many actions")
        
        # Validate resource filters
        if resource_filters:
            self._validate_resource_filters(resource_filters, errors, warnings)
        
        # Validate time restrictions
        if time_restrictions:
            self._validate_time_restrictions(time_restrictions, errors, warnings)
        
        # Validate safety overrides
        if safety_overrides:
            self._validate_safety_overrides(safety_overrides, errors, warnings)
        
        is_valid = len(errors) == 0
        
        return PolicyValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_resource_filters(self, 
                                 resource_filters: Dict[str, Any],
                                 errors: List[str],
                                 warnings: List[str]):
        """Validate resource filter configuration"""
        
        # Check for required filter fields
        if "exclude_tags" in resource_filters:
            exclude_tags = resource_filters["exclude_tags"]
            if not isinstance(exclude_tags, list):
                errors.append("exclude_tags must be a list")
        
        if "include_services" in resource_filters:
            include_services = resource_filters["include_services"]
            if not isinstance(include_services, list):
                errors.append("include_services must be a list")
            else:
                valid_services = ["EC2", "EBS", "EIP", "ELB", "VPC"]
                for service in include_services:
                    if service not in valid_services:
                        warnings.append(f"Unknown service in include_services: {service}")
        
        if "min_cost_threshold" in resource_filters:
            threshold = resource_filters["min_cost_threshold"]
            if not isinstance(threshold, (int, float)) or threshold < 0:
                errors.append("min_cost_threshold must be a non-negative number")
    
    def _validate_time_restrictions(self,
                                  time_restrictions: Dict[str, Any],
                                  errors: List[str],
                                  warnings: List[str]):
        """Validate time restriction configuration"""
        
        if "business_hours" in time_restrictions:
            business_hours = time_restrictions["business_hours"]
            
            if not isinstance(business_hours, dict):
                errors.append("business_hours must be a dictionary")
                return
            
            # Validate timezone
            if "timezone" in business_hours:
                # In a real implementation, would validate against pytz timezones
                pass
            
            # Validate time format
            for time_field in ["start", "end"]:
                if time_field in business_hours:
                    time_str = business_hours[time_field]
                    try:
                        # Validate HH:MM format
                        hour, minute = time_str.split(":")
                        hour_int = int(hour)
                        minute_int = int(minute)
                        if not (0 <= hour_int <= 23 and 0 <= minute_int <= 59):
                            errors.append(f"Invalid time format for {time_field}: {time_str}")
                    except (ValueError, AttributeError):
                        errors.append(f"Invalid time format for {time_field}: {time_str}")
            
            # Validate days
            if "days" in business_hours:
                days = business_hours["days"]
                if not isinstance(days, list):
                    errors.append("business_hours.days must be a list")
                else:
                    valid_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                    for day in days:
                        if day.lower() not in valid_days:
                            errors.append(f"Invalid day: {day}")
    
    def _validate_safety_overrides(self,
                                 safety_overrides: Dict[str, Any],
                                 errors: List[str],
                                 warnings: List[str]):
        """Validate safety override configuration"""
        
        valid_safety_rules = [
            "production_tag_protection",
            "business_hours_protection",
            "auto_scaling_group_protection",
            "load_balancer_target_protection",
            "database_dependency_protection",
            "recent_activity_protection"
        ]
        
        for rule_name, override_config in safety_overrides.items():
            if rule_name not in valid_safety_rules:
                warnings.append(f"Unknown safety rule in overrides: {rule_name}")
            
            if not isinstance(override_config, dict):
                errors.append(f"Safety override for {rule_name} must be a dictionary")
                continue
            
            # Validate override structure
            if "enabled" in override_config:
                if not isinstance(override_config["enabled"], bool):
                    errors.append(f"Safety override enabled flag for {rule_name} must be boolean")
            
            if "parameters" in override_config:
                if not isinstance(override_config["parameters"], dict):
                    errors.append(f"Safety override parameters for {rule_name} must be a dictionary")
    
    def validate_action_against_policy(self,
                                     action_type: ActionType,
                                     resource_metadata: Dict[str, Any],
                                     policy: AutomationPolicy) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate if an action is allowed by the policy.
        
        Args:
            action_type: Type of action to validate
            resource_metadata: Metadata about the resource
            policy: Automation policy to validate against
            
        Returns:
            Tuple of (is_allowed, validation_details)
        """
        
        validation_details = {
            "policy_id": str(policy.id),
            "action_type": action_type.value,
            "checks": {},
            "violations": []
        }
        
        # Check if action is blocked
        if action_type.value in policy.blocked_actions:
            validation_details["violations"].append(f"Action {action_type.value} is blocked by policy")
            validation_details["checks"]["blocked_action"] = False
            return False, validation_details
        
        validation_details["checks"]["blocked_action"] = True
        
        # Check if action is enabled
        if action_type.value not in policy.enabled_actions:
            validation_details["violations"].append(f"Action {action_type.value} is not enabled in policy")
            validation_details["checks"]["enabled_action"] = False
            return False, validation_details
        
        validation_details["checks"]["enabled_action"] = True
        
        # Check resource filters
        resource_filter_passed, filter_details = self._check_resource_filters(
            resource_metadata, policy.resource_filters
        )
        
        validation_details["checks"]["resource_filters"] = resource_filter_passed
        validation_details["filter_details"] = filter_details
        
        if not resource_filter_passed:
            validation_details["violations"].extend(filter_details.get("violations", []))
            return False, validation_details
        
        # All checks passed
        return True, validation_details
    
    def _check_resource_filters(self,
                              resource_metadata: Dict[str, Any],
                              resource_filters: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if resource passes policy filters"""
        
        filter_details = {
            "checks": {},
            "violations": []
        }
        
        # Check exclude tags
        if "exclude_tags" in resource_filters:
            exclude_tags = resource_filters["exclude_tags"]
            resource_tags = resource_metadata.get("tags", {})
            
            # Convert resource tags to key=value format for comparison
            resource_tag_strings = [f"{k}={v}" for k, v in resource_tags.items()]
            
            for exclude_tag in exclude_tags:
                if exclude_tag in resource_tag_strings:
                    filter_details["violations"].append(f"Resource has excluded tag: {exclude_tag}")
                    filter_details["checks"]["exclude_tags"] = False
                    return False, filter_details
            
            filter_details["checks"]["exclude_tags"] = True
        
        # Check include services
        if "include_services" in resource_filters:
            include_services = resource_filters["include_services"]
            resource_service = resource_metadata.get("service", "")
            
            if resource_service not in include_services:
                filter_details["violations"].append(f"Resource service {resource_service} not in included services")
                filter_details["checks"]["include_services"] = False
                return False, filter_details
            
            filter_details["checks"]["include_services"] = True
        
        # Check minimum cost threshold
        if "min_cost_threshold" in resource_filters:
            min_threshold = resource_filters["min_cost_threshold"]
            resource_cost = resource_metadata.get("monthly_cost", 0)
            
            if resource_cost < min_threshold:
                filter_details["violations"].append(f"Resource cost ${resource_cost} below threshold ${min_threshold}")
                filter_details["checks"]["min_cost_threshold"] = False
                return False, filter_details
            
            filter_details["checks"]["min_cost_threshold"] = True
        
        return True, filter_details
    
    def create_approval_request(self,
                               action: OptimizationAction,
                               requested_by: str = "automation_system",
                               notification_channels: List[str] = None) -> ActionApproval:
        """
        Create an approval request for an action with notification integration.
        
        Args:
            action: Optimization action requiring approval
            requested_by: Who/what is requesting approval
            notification_channels: List of notification channel IDs to send approval request
            
        Returns:
            Created approval request
        """
        
        logger.info("Creating approval request",
                   action_id=str(action.id),
                   resource_id=action.resource_id,
                   action_type=action.action_type.value)
        
        # Calculate expiration time (24 hours from now)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        with get_db_session() as session:
            approval = ActionApproval(
                action_id=action.id,
                requested_by=requested_by,
                expires_at=expires_at
            )
            
            session.add(approval)
            session.commit()
            
            # Log approval request creation
            self.audit_logger.log_action_event(
                action.id,
                "approval_requested",
                {
                    "approval_id": str(approval.id),
                    "requested_by": requested_by,
                    "expires_at": expires_at.isoformat()
                }
            )
            
            # Send notification for approval request
            if notification_channels:
                self._send_approval_request_notification(approval, action, notification_channels)
            
            logger.info("Approval request created",
                       approval_id=str(approval.id),
                       action_id=str(action.id))
            
            return approval
    
    def process_approval_decision(self,
                                 approval_id: uuid.UUID,
                                 approved: bool,
                                 approved_by: uuid.UUID,
                                 rejection_reason: Optional[str] = None,
                                 notification_channels: List[str] = None) -> bool:
        """
        Process an approval decision with notification integration.
        
        Args:
            approval_id: ID of the approval request
            approved: Whether the action was approved
            approved_by: User ID who made the decision
            rejection_reason: Reason for rejection (if applicable)
            notification_channels: List of notification channel IDs to send decision notification
            
        Returns:
            True if decision was processed successfully
        """
        
        logger.info("Processing approval decision",
                   approval_id=str(approval_id),
                   approved=approved)
        
        try:
            with get_db_session() as session:
                approval = session.query(ActionApproval).filter_by(id=approval_id).first()
                
                if not approval:
                    logger.error("Approval request not found", approval_id=str(approval_id))
                    return False
                
                if approval.approval_status != ApprovalStatus.PENDING:
                    logger.warning("Approval request already processed",
                                 approval_id=str(approval_id),
                                 current_status=approval.approval_status.value)
                    return False
                
                # Check if approval has expired
                if approval.expires_at and datetime.utcnow() > approval.expires_at:
                    logger.warning("Approval request has expired",
                                 approval_id=str(approval_id))
                    approval.approval_status = ApprovalStatus.REJECTED
                    approval.rejection_reason = "Approval request expired"
                    session.commit()
                    return False
                
                # Get the associated action for notification
                action = session.query(OptimizationAction).filter_by(id=approval.action_id).first()
                
                # Update approval status
                approval.approved_by = approved_by
                approval.approved_at = datetime.utcnow()
                
                if approved:
                    approval.approval_status = ApprovalStatus.APPROVED
                else:
                    approval.approval_status = ApprovalStatus.REJECTED
                    approval.rejection_reason = rejection_reason or "No reason provided"
                
                # Update the associated action
                if action:
                    action.approval_status = approval.approval_status
                
                session.commit()
                
                # Log approval decision
                self.audit_logger.log_action_event(
                    approval.action_id,
                    "approval_decided",
                    {
                        "approval_id": str(approval.id),
                        "approved": approved,
                        "approved_by": str(approved_by),
                        "rejection_reason": rejection_reason
                    },
                    {"user_id": str(approved_by)}
                )
                
                # Send notification for approval decision
                if notification_channels and action:
                    self._send_approval_decision_notification(
                        approval, action, approved, rejection_reason, notification_channels
                    )
                
                logger.info("Approval decision processed successfully",
                           approval_id=str(approval_id),
                           approved=approved)
                
                return True
                
        except Exception as e:
            logger.error("Failed to process approval decision",
                        approval_id=str(approval_id),
                        error=str(e))
            return False
    
    def simulate_dry_run(self,
                        opportunities: List,  # List[OptimizationOpportunity]
                        policy: AutomationPolicy) -> Dict[str, Any]:
        """
        Simulate what actions would be taken without executing them.
        
        Args:
            opportunities: List of optimization opportunities
            policy: Automation policy to apply
            
        Returns:
            Detailed simulation results
        """
        
        logger.info("Starting dry run simulation",
                   opportunity_count=len(opportunities),
                   policy_id=str(policy.id))
        
        simulation_results = {
            "policy_id": str(policy.id),
            "policy_name": policy.name,
            "simulation_timestamp": datetime.utcnow().isoformat(),
            "total_opportunities": len(opportunities),
            "actions_would_execute": [],
            "actions_would_require_approval": [],
            "actions_would_be_blocked": [],
            "total_estimated_savings": 0.0,
            "safety_violations": [],
            "policy_violations": []
        }
        
        for opportunity in opportunities:
            # Check policy validation
            policy_allowed, policy_details = self.validate_action_against_policy(
                opportunity.action_type, opportunity.resource_metadata, policy
            )
            
            action_summary = {
                "resource_id": opportunity.resource_id,
                "resource_type": opportunity.resource_type,
                "action_type": opportunity.action_type.value,
                "estimated_savings": float(opportunity.estimated_monthly_savings),
                "risk_level": opportunity.risk_level.value
            }
            
            if not policy_allowed:
                action_summary["policy_violations"] = policy_details["violations"]
                simulation_results["actions_would_be_blocked"].append(action_summary)
                simulation_results["policy_violations"].extend(policy_details["violations"])
                continue
            
            # Check if approval would be required
            requires_approval = (
                opportunity.action_type.value in policy.approval_required_actions or
                opportunity.risk_level.value == "high"
            )
            
            if requires_approval:
                simulation_results["actions_would_require_approval"].append(action_summary)
            else:
                simulation_results["actions_would_execute"].append(action_summary)
                simulation_results["total_estimated_savings"] += float(opportunity.estimated_monthly_savings)
        
        # Log dry run simulation
        self.audit_logger.log_policy_event(
            policy.id,
            "dry_run_simulation",
            simulation_results
        )
        
        logger.info("Dry run simulation completed",
                   policy_id=str(policy.id),
                   would_execute=len(simulation_results["actions_would_execute"]),
                   would_require_approval=len(simulation_results["actions_would_require_approval"]),
                   would_be_blocked=len(simulation_results["actions_would_be_blocked"]))
        
        return simulation_results
    
    def get_policy_by_id(self, policy_id: uuid.UUID) -> Optional[AutomationPolicy]:
        """Get automation policy by ID"""
        
        with get_db_session() as session:
            return session.query(AutomationPolicy).filter_by(id=policy_id).first()
    
    def get_active_policies(self) -> List[AutomationPolicy]:
        """Get all active automation policies"""
        
        with get_db_session() as session:
            return session.query(AutomationPolicy).filter_by(is_active=True).all()
    
    def update_policy(self,
                     policy_id: uuid.UUID,
                     updates: Dict[str, Any],
                     updated_by: uuid.UUID) -> Tuple[bool, PolicyValidationResult]:
        """
        Update an existing automation policy.
        
        Args:
            policy_id: ID of policy to update
            updates: Dictionary of fields to update
            updated_by: User ID making the update
            
        Returns:
            Tuple of (success, validation_result)
        """
        
        logger.info("Updating automation policy",
                   policy_id=str(policy_id))
        
        try:
            with get_db_session() as session:
                policy = session.query(AutomationPolicy).filter_by(id=policy_id).first()
                
                if not policy:
                    validation_result = PolicyValidationResult(
                        is_valid=False,
                        errors=["Policy not found"],
                        warnings=[]
                    )
                    return False, validation_result
                
                # Store old values for audit
                old_values = {
                    "name": policy.name,
                    "automation_level": policy.automation_level.value,
                    "enabled_actions": policy.enabled_actions,
                    "approval_required_actions": policy.approval_required_actions,
                    "blocked_actions": policy.blocked_actions
                }
                
                # Apply updates
                for field, value in updates.items():
                    if hasattr(policy, field):
                        setattr(policy, field, value)
                
                # Validate updated policy
                validation_result = self.validate_policy_configuration(
                    automation_level=policy.automation_level,
                    enabled_actions=policy.enabled_actions,
                    approval_required_actions=policy.approval_required_actions,
                    blocked_actions=policy.blocked_actions,
                    resource_filters=policy.resource_filters,
                    time_restrictions=policy.time_restrictions,
                    safety_overrides=policy.safety_overrides
                )
                
                if not validation_result.is_valid:
                    session.rollback()
                    return False, validation_result
                
                policy.updated_at = datetime.utcnow()
                session.commit()
                
                # Log policy update
                self.audit_logger.log_policy_event(
                    policy.id,
                    "policy_updated",
                    {
                        "old_values": old_values,
                        "new_values": updates
                    },
                    {"user_id": str(updated_by)}
                )
                
                logger.info("Automation policy updated successfully",
                           policy_id=str(policy_id))
                
                return True, validation_result
                
        except Exception as e:
            logger.error("Failed to update automation policy",
                        policy_id=str(policy_id),
                        error=str(e))
            validation_result = PolicyValidationResult(
                is_valid=False,
                errors=[f"Database error: {str(e)}"],
                warnings=[]
            )
            return False, validation_result
    
    def _send_approval_request_notification(self,
                                          approval: ActionApproval,
                                          action: OptimizationAction,
                                          notification_channels: List[str]):
        """Send notification for new approval request"""
        
        try:
            message = NotificationMessage(
                title=f"Approval Required: {action.action_type.value.replace('_', ' ').title()}",
                message=f"""
A cost optimization action requires your approval:

Action: {action.action_type.value.replace('_', ' ').title()}
Resource: {action.resource_id} ({action.resource_type})
Estimated Monthly Savings: ${action.estimated_monthly_savings:.2f}
Risk Level: {action.risk_level.value.title()}

Approval ID: {approval.id}
Expires: {approval.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC') if approval.expires_at else 'No expiration'}

Please review and approve or reject this action through the FinOps dashboard.
                """.strip(),
                priority=NotificationPriority.HIGH,
                metadata={
                    "approval_id": str(approval.id),
                    "action_id": str(action.id),
                    "action_type": action.action_type.value,
                    "resource_id": action.resource_id,
                    "estimated_savings": float(action.estimated_monthly_savings),
                    "risk_level": action.risk_level.value
                },
                alert_id=f"approval_request_{approval.id}",
                resource_id=action.resource_id,
                cost_amount=float(action.estimated_monthly_savings)
            )
            
            # Send notification asynchronously
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(
                    self.notification_service.send_notification(notification_channels, message)
                )
            except RuntimeError:
                # No event loop running, create a new one
                asyncio.run(
                    self.notification_service.send_notification(notification_channels, message)
                )
            
            logger.info("Approval request notification sent",
                       approval_id=str(approval.id),
                       channels=notification_channels)
                       
        except Exception as e:
            logger.error("Failed to send approval request notification",
                        approval_id=str(approval.id),
                        error=str(e))
    
    def _send_approval_decision_notification(self,
                                           approval: ActionApproval,
                                           action: OptimizationAction,
                                           approved: bool,
                                           rejection_reason: Optional[str],
                                           notification_channels: List[str]):
        """Send notification for approval decision"""
        
        try:
            status = "Approved" if approved else "Rejected"
            priority = NotificationPriority.MEDIUM if approved else NotificationPriority.LOW
            
            message_text = f"""
Cost optimization action has been {status.lower()}:

Action: {action.action_type.value.replace('_', ' ').title()}
Resource: {action.resource_id} ({action.resource_type})
Estimated Monthly Savings: ${action.estimated_monthly_savings:.2f}
Decision: {status}
            """
            
            if not approved and rejection_reason:
                message_text += f"\nReason: {rejection_reason}"
            
            if approved:
                message_text += "\n\nThe action will be scheduled for execution according to the automation policy."
            
            message = NotificationMessage(
                title=f"Action {status}: {action.action_type.value.replace('_', ' ').title()}",
                message=message_text.strip(),
                priority=priority,
                metadata={
                    "approval_id": str(approval.id),
                    "action_id": str(action.id),
                    "action_type": action.action_type.value,
                    "resource_id": action.resource_id,
                    "approved": approved,
                    "rejection_reason": rejection_reason
                },
                alert_id=f"approval_decision_{approval.id}",
                resource_id=action.resource_id,
                cost_amount=float(action.estimated_monthly_savings) if approved else None
            )
            
            # Send notification asynchronously
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(
                    self.notification_service.send_notification(notification_channels, message)
                )
            except RuntimeError:
                # No event loop running, create a new one
                asyncio.run(
                    self.notification_service.send_notification(notification_channels, message)
                )
            
            logger.info("Approval decision notification sent",
                       approval_id=str(approval.id),
                       approved=approved,
                       channels=notification_channels)
                       
        except Exception as e:
            logger.error("Failed to send approval decision notification",
                        approval_id=str(approval.id),
                        error=str(e))
    
    def detect_policy_violations(self,
                               actions: List[OptimizationAction],
                               policy: AutomationPolicy) -> Dict[str, Any]:
        """
        Detect policy violations for a list of actions.
        
        Args:
            actions: List of optimization actions to check
            policy: Automation policy to validate against
            
        Returns:
            Dictionary containing violation details and blocked actions
        """
        
        logger.info("Detecting policy violations",
                   action_count=len(actions),
                   policy_id=str(policy.id))
        
        violation_report = {
            "policy_id": str(policy.id),
            "policy_name": policy.name,
            "total_actions_checked": len(actions),
            "violations_detected": [],
            "blocked_actions": [],
            "allowed_actions": [],
            "violation_summary": {
                "blocked_action_types": 0,
                "resource_filter_violations": 0,
                "time_restriction_violations": 0,
                "safety_override_violations": 0
            }
        }
        
        for action in actions:
            # Check policy validation
            is_allowed, validation_details = self.validate_action_against_policy(
                action.action_type, action.resource_metadata, policy
            )
            
            action_summary = {
                "action_id": str(action.id),
                "resource_id": action.resource_id,
                "resource_type": action.resource_type,
                "action_type": action.action_type.value,
                "estimated_savings": float(action.estimated_monthly_savings),
                "risk_level": action.risk_level.value
            }
            
            if not is_allowed:
                # Policy violation detected
                violation_details_copy = validation_details.copy()
                violation_details_copy.update(action_summary)
                
                violation_report["violations_detected"].append(violation_details_copy)
                violation_report["blocked_actions"].append(action_summary)
                
                # Categorize violation types
                for violation in validation_details.get("violations", []):
                    if "blocked by policy" in violation:
                        violation_report["violation_summary"]["blocked_action_types"] += 1
                    elif any(filter_term in violation for filter_term in ["tag", "service", "cost", "threshold"]):
                        violation_report["violation_summary"]["resource_filter_violations"] += 1
                    elif "time" in violation or "business hours" in violation:
                        violation_report["violation_summary"]["time_restriction_violations"] += 1
                    elif "safety" in violation or "override" in violation:
                        violation_report["violation_summary"]["safety_override_violations"] += 1
                
                # Log individual violation
                self.audit_logger.log_action_event(
                    action.id,
                    "policy_violation_detected",
                    {
                        "policy_id": str(policy.id),
                        "violations": validation_details.get("violations", []),
                        "validation_details": validation_details
                    }
                )
                
            else:
                violation_report["allowed_actions"].append(action_summary)
        
        # Log overall violation detection results
        self.audit_logger.log_policy_event(
            policy.id,
            "policy_violation_detection",
            {
                "total_actions": len(actions),
                "violations_detected": len(violation_report["violations_detected"]),
                "blocked_actions": len(violation_report["blocked_actions"]),
                "violation_summary": violation_report["violation_summary"]
            }
        )
        
        logger.info("Policy violation detection completed",
                   policy_id=str(policy.id),
                   total_actions=len(actions),
                   violations=len(violation_report["violations_detected"]),
                   blocked=len(violation_report["blocked_actions"]))
        
        return violation_report
    
    def block_policy_violating_actions(self,
                                     actions: List[OptimizationAction],
                                     policy: AutomationPolicy,
                                     notification_channels: List[str] = None) -> Dict[str, Any]:
        """
        Block actions that violate policy and send notifications.
        
        Args:
            actions: List of optimization actions to check and potentially block
            policy: Automation policy to validate against
            notification_channels: List of notification channel IDs for violation alerts
            
        Returns:
            Dictionary containing blocking results and statistics
        """
        
        logger.info("Blocking policy violating actions",
                   action_count=len(actions),
                   policy_id=str(policy.id))
        
        # Detect violations first
        violation_report = self.detect_policy_violations(actions, policy)
        
        blocking_results = {
            "policy_id": str(policy.id),
            "blocking_timestamp": datetime.utcnow().isoformat(),
            "actions_processed": len(actions),
            "actions_blocked": 0,
            "actions_allowed": 0,
            "blocked_action_ids": [],
            "violation_notifications_sent": False
        }
        
        try:
            with get_db_session() as session:
                # Block violating actions
                for violation in violation_report["violations_detected"]:
                    action_id = uuid.UUID(violation["action_id"])
                    action = session.query(OptimizationAction).filter_by(id=action_id).first()
                    
                    if action and action.execution_status == ActionStatus.PENDING:
                        # Block the action
                        action.execution_status = ActionStatus.CANCELLED
                        action.error_message = f"Blocked due to policy violations: {', '.join(violation.get('violations', []))}"
                        
                        blocking_results["actions_blocked"] += 1
                        blocking_results["blocked_action_ids"].append(str(action_id))
                        
                        # Log blocking action
                        self.audit_logger.log_action_event(
                            action.id,
                            "action_blocked_policy_violation",
                            {
                                "policy_id": str(policy.id),
                                "violations": violation.get("violations", []),
                                "blocking_timestamp": datetime.utcnow().isoformat()
                            }
                        )
                
                blocking_results["actions_allowed"] = len(actions) - blocking_results["actions_blocked"]
                session.commit()
                
                # Send violation notification if there were blocked actions
                if blocking_results["actions_blocked"] > 0 and notification_channels:
                    self._send_policy_violation_notification(
                        violation_report, blocking_results, notification_channels
                    )
                    blocking_results["violation_notifications_sent"] = True
                
        except Exception as e:
            logger.error("Failed to block policy violating actions",
                        policy_id=str(policy.id),
                        error=str(e))
            blocking_results["error"] = str(e)
        
        logger.info("Policy violation blocking completed",
                   policy_id=str(policy.id),
                   actions_blocked=blocking_results["actions_blocked"],
                   actions_allowed=blocking_results["actions_allowed"])
        
        return blocking_results
    
    def _send_policy_violation_notification(self,
                                          violation_report: Dict[str, Any],
                                          blocking_results: Dict[str, Any],
                                          notification_channels: List[str]):
        """Send notification for policy violations and blocked actions"""
        
        try:
            blocked_count = blocking_results["actions_blocked"]
            total_count = blocking_results["actions_processed"]
            policy_name = violation_report["policy_name"]
            
            message_text = f"""
Policy violations detected and actions blocked:

Policy: {policy_name}
Total Actions Processed: {total_count}
Actions Blocked: {blocked_count}
Actions Allowed: {blocking_results["actions_allowed"]}

Violation Summary:
- Blocked Action Types: {violation_report["violation_summary"]["blocked_action_types"]}
- Resource Filter Violations: {violation_report["violation_summary"]["resource_filter_violations"]}
- Time Restriction Violations: {violation_report["violation_summary"]["time_restriction_violations"]}
- Safety Override Violations: {violation_report["violation_summary"]["safety_override_violations"]}

Please review the policy configuration and blocked actions in the FinOps dashboard.
            """
            
            message = NotificationMessage(
                title=f"Policy Violations Detected: {blocked_count} Actions Blocked",
                message=message_text.strip(),
                priority=NotificationPriority.HIGH,
                metadata={
                    "policy_id": violation_report["policy_id"],
                    "policy_name": policy_name,
                    "actions_blocked": blocked_count,
                    "actions_allowed": blocking_results["actions_allowed"],
                    "violation_summary": violation_report["violation_summary"],
                    "blocked_action_ids": blocking_results["blocked_action_ids"]
                },
                alert_id=f"policy_violations_{violation_report['policy_id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Send notification asynchronously
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(
                    self.notification_service.send_notification(notification_channels, message)
                )
            except RuntimeError:
                # No event loop running, create a new one
                asyncio.run(
                    self.notification_service.send_notification(notification_channels, message)
                )
            
            logger.info("Policy violation notification sent",
                       policy_id=violation_report["policy_id"],
                       blocked_count=blocked_count,
                       channels=notification_channels)
                       
        except Exception as e:
            logger.error("Failed to send policy violation notification",
                        policy_id=violation_report["policy_id"],
                        error=str(e))