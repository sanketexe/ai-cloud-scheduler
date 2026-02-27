"""
Approval Workflow Integration Module

This module provides enhanced integration between the approval workflow engine
and the decision tracking system, ensuring proper notification routing and
status tracking for collaborative FinOps workspace.

Requirements addressed:
- 2.3: Intelligent notification routing based on roles and urgency
- 2.4: Immediate notification to all session participants
"""

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal

from .approval_workflow_engine import (
    ApprovalWorkflowEngine, OptimizationAction, DecisionContext,
    ApprovalDecision, ApprovalRequest, ApprovalResult, EscalationResult,
    Priority, ApprovalStatus, EscalationReason
)
from .decision_tracking_system import (
    decision_tracker, notification_service, NotificationType, NotificationPriority
)

logger = logging.getLogger(__name__)


class EnhancedApprovalWorkflowEngine(ApprovalWorkflowEngine):
    """
    Enhanced approval workflow engine with integrated decision tracking
    and intelligent notification routing
    """
    
    def __init__(self):
        super().__init__()
        self.decision_tracker = decision_tracker
        self.notification_service = notification_service
    
    async def submit_for_approval(self, decision: OptimizationAction, context: DecisionContext) -> ApprovalRequest:
        """
        Submit an optimization decision for approval with enhanced tracking and notifications
        
        Args:
            decision: The optimization action requiring approval
            context: Complete decision context including session and participants
            
        Returns:
            ApprovalRequest: The created approval request
        """
        try:
            # Create approval request using parent method
            approval_request = await super().submit_for_approval(decision, context)
            
            # Track decision creation
            decision_id = approval_request.request_id
            await self.decision_tracker.track_decision_created(
                decision_id=decision_id,
                session_id=context.session_id,
                participant_id=approval_request.created_by,
                decision_data={
                    "action_type": decision.action_type,
                    "resource_id": decision.resource_id,
                    "resource_type": decision.resource_type,
                    "cost_impact": float(decision.cost_impact),
                    "risk_level": decision.risk_level,
                    "description": decision.description,
                    "estimated_savings": float(decision.estimated_savings),
                    "implementation_timeline": decision.implementation_timeline
                }
            )
            
            # Track approval submission
            await self.decision_tracker.track_approval_submitted(
                decision_id=decision_id,
                approval_request_id=approval_request.request_id,
                approver_id=approval_request.approval_chain.steps[0].approver_id,
                approval_data={
                    "chain_id": approval_request.approval_chain.chain_id,
                    "priority": approval_request.priority.value,
                    "deadline": approval_request.deadline.isoformat(),
                    "threshold_conditions": approval_request.approval_chain.threshold_conditions
                }
            )
            
            # Send intelligent notifications
            await self._send_approval_notifications(approval_request, context)
            
            # Notify session participants
            await self._notify_session_participants_approval_submitted(approval_request, context)
            
            logger.info(f"Enhanced approval submission completed for decision {decision_id}")
            return approval_request
            
        except Exception as e:
            logger.error(f"Error in enhanced approval submission: {e}")
            raise
    
    async def process_approval(self, approval_id: str, approver_id: str, decision: ApprovalDecision) -> ApprovalResult:
        """
        Process an approval decision with enhanced tracking and notifications
        
        Args:
            approval_id: ID of the approval request
            approver_id: ID of the approver
            decision: The approval decision
            
        Returns:
            ApprovalResult: Result of processing the approval
        """
        try:
            # Get approval request
            request = await self._get_approval_request(approval_id)
            if not request:
                return ApprovalResult(success=False, error_message="Approval request not found")
            
            # Process approval using parent method
            result = await super().process_approval(approval_id, approver_id, decision)
            
            if result.success:
                # Track approval decision
                next_step = None
                if result.next_step is not None and result.next_step < len(request.approval_chain.steps):
                    next_step = request.approval_chain.steps[result.next_step].approver_id
                
                await self.decision_tracker.track_approval_decision(
                    decision_id=approval_id,
                    approver_id=approver_id,
                    approved=decision.approved,
                    comments=decision.comments,
                    next_step=next_step
                )
                
                # Send notifications based on decision
                await self._send_decision_notifications(request, decision, result)
                
                # Notify session participants immediately
                await self._notify_session_participants_decision_made(request, decision, result)
                
                # If approved and complete, track implementation
                if result.status == ApprovalStatus.APPROVED and result.next_step is None:
                    await self._handle_final_approval(request, decision)
                elif result.status == ApprovalStatus.REJECTED:
                    await self._handle_rejection(request, decision)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced approval processing: {e}")
            return ApprovalResult(success=False, error_message=str(e))
    
    async def escalate_approval(self, approval_id: str, reason: EscalationReason) -> EscalationResult:
        """
        Escalate an approval request with enhanced notifications
        
        Args:
            approval_id: ID of the approval request
            reason: Reason for escalation
            
        Returns:
            EscalationResult: Result of the escalation
        """
        try:
            # Get approval request
            request = await self._get_approval_request(approval_id)
            if not request:
                return EscalationResult(success=False, error_message="Approval request not found")
            
            # Escalate using parent method
            result = await super().escalate_approval(approval_id, reason)
            
            if result.success:
                # Send escalation notifications with high priority
                await self._send_escalation_notifications(request, reason, result.escalated_to)
                
                # Notify session participants about escalation
                await self._notify_session_participants_escalation(request, reason, result.escalated_to)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced approval escalation: {e}")
            return EscalationResult(success=False, error_message=str(e))
    
    async def _send_approval_notifications(self, request: ApprovalRequest, context: DecisionContext):
        """Send intelligent notifications for approval request"""
        try:
            # Determine priority based on cost impact and risk
            priority = self._map_priority_to_notification_priority(request.priority)
            
            # Get current approver
            current_step = request.approval_chain.steps[request.current_step]
            
            # Send notification to current approver
            notification_data = {
                "decision_id": request.request_id,
                "decision_type": request.decision.action_type,
                "cost_impact": float(request.decision.cost_impact),
                "risk_level": request.decision.risk_level,
                "priority": request.priority.value,
                "deadline": request.deadline.isoformat(),
                "session_id": context.session_id,
                "description": request.decision.description,
                "estimated_savings": float(request.decision.estimated_savings),
                "participants": context.participants,
                "discussion_summary": context.discussion_summary
            }
            
            success = await self.notification_service.send_notification(
                recipient_id=current_step.approver_id,
                notification_type=NotificationType.APPROVAL_REQUEST,
                data=notification_data,
                priority=priority.value,
                session_id=context.session_id
            )
            
            if success:
                logger.info(f"Approval notification sent to {current_step.approver_id}")
            else:
                logger.warning(f"Failed to send approval notification to {current_step.approver_id}")
                
        except Exception as e:
            logger.error(f"Error sending approval notifications: {e}")
    
    async def _notify_session_participants_approval_submitted(self, request: ApprovalRequest, context: DecisionContext):
        """Notify all session participants that approval was submitted"""
        try:
            notification_data = {
                "decision_id": request.request_id,
                "decision_type": request.decision.action_type,
                "cost_impact": float(request.decision.cost_impact),
                "status": "submitted_for_approval",
                "approver": request.approval_chain.steps[0].approver_id,
                "priority": request.priority.value,
                "deadline": request.deadline.isoformat()
            }
            
            # Exclude the current approver from session notifications to avoid duplicate
            exclude_participants = [request.approval_chain.steps[0].approver_id]
            
            results = await self.notification_service.send_session_notification(
                session_id=context.session_id,
                notification_type=NotificationType.SESSION_UPDATE,
                data=notification_data,
                priority="medium",
                exclude_participants=exclude_participants
            )
            
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Notified {success_count}/{len(results)} session participants about approval submission")
            
        except Exception as e:
            logger.error(f"Error notifying session participants about approval submission: {e}")
    
    async def _send_decision_notifications(self, request: ApprovalRequest, decision: ApprovalDecision, result: ApprovalResult):
        """Send notifications based on approval decision"""
        try:
            notification_data = {
                "decision_id": request.request_id,
                "decision_type": request.decision.action_type,
                "approver_id": decision.approver_id,
                "approved": decision.approved,
                "comments": decision.comments,
                "status": result.status.value if result.status else "unknown",
                "cost_impact": float(request.decision.cost_impact),
                "next_step": result.next_step
            }
            
            # Determine notification type and priority
            if decision.approved:
                if result.next_step is not None:
                    # More approvals needed
                    notification_type = NotificationType.APPROVAL_UPDATE
                    priority = "medium"
                    
                    # Notify next approver
                    if result.next_step < len(request.approval_chain.steps):
                        next_approver = request.approval_chain.steps[result.next_step].approver_id
                        await self.notification_service.send_notification(
                            recipient_id=next_approver,
                            notification_type=NotificationType.APPROVAL_REQUEST,
                            data=notification_data,
                            priority=priority,
                            session_id=request.context.session_id
                        )
                else:
                    # Final approval
                    notification_type = NotificationType.APPROVAL_UPDATE
                    priority = "high"
            else:
                # Rejected
                notification_type = NotificationType.APPROVAL_UPDATE
                priority = "high"
            
            # Send update notification to original requester
            await self.notification_service.send_notification(
                recipient_id=request.created_by,
                notification_type=notification_type,
                data=notification_data,
                priority=priority,
                session_id=request.context.session_id
            )
            
        except Exception as e:
            logger.error(f"Error sending decision notifications: {e}")
    
    async def _notify_session_participants_decision_made(self, request: ApprovalRequest, decision: ApprovalDecision, result: ApprovalResult):
        """Immediately notify all session participants about approval decision"""
        try:
            notification_data = {
                "decision_id": request.request_id,
                "decision_type": request.decision.action_type,
                "approver_id": decision.approver_id,
                "approved": decision.approved,
                "comments": decision.comments,
                "status": result.status.value if result.status else "unknown",
                "cost_impact": float(request.decision.cost_impact),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to all session participants
            results = await self.notification_service.send_session_notification(
                session_id=request.context.session_id,
                notification_type=NotificationType.APPROVAL_UPDATE,
                data=notification_data,
                priority="high"  # High priority for immediate decision updates
            )
            
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Immediately notified {success_count}/{len(results)} session participants about approval decision")
            
        except Exception as e:
            logger.error(f"Error immediately notifying session participants about decision: {e}")
    
    async def _send_escalation_notifications(self, request: ApprovalRequest, reason: EscalationReason, escalated_to: List[str]):
        """Send escalation notifications with high priority"""
        try:
            notification_data = {
                "decision_id": request.request_id,
                "decision_type": request.decision.action_type,
                "escalation_reason": reason.value,
                "original_approver": request.approval_chain.steps[request.current_step].approver_id,
                "cost_impact": float(request.decision.cost_impact),
                "priority": request.priority.value,
                "deadline": request.deadline.isoformat(),
                "session_id": request.context.session_id,
                "urgency": "high"
            }
            
            # Send to escalation targets
            for target_id in escalated_to:
                await self.notification_service.send_notification(
                    recipient_id=target_id,
                    notification_type=NotificationType.APPROVAL_ESCALATION,
                    data=notification_data,
                    priority="critical",  # Escalations are always critical
                    session_id=request.context.session_id
                )
            
            logger.info(f"Sent escalation notifications to {len(escalated_to)} targets")
            
        except Exception as e:
            logger.error(f"Error sending escalation notifications: {e}")
    
    async def _notify_session_participants_escalation(self, request: ApprovalRequest, reason: EscalationReason, escalated_to: List[str]):
        """Notify session participants about escalation"""
        try:
            notification_data = {
                "decision_id": request.request_id,
                "decision_type": request.decision.action_type,
                "escalation_reason": reason.value,
                "escalated_to": escalated_to,
                "cost_impact": float(request.decision.cost_impact),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            results = await self.notification_service.send_session_notification(
                session_id=request.context.session_id,
                notification_type=NotificationType.APPROVAL_ESCALATION,
                data=notification_data,
                priority="high"
            )
            
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Notified {success_count}/{len(results)} session participants about escalation")
            
        except Exception as e:
            logger.error(f"Error notifying session participants about escalation: {e}")
    
    async def _handle_final_approval(self, request: ApprovalRequest, decision: ApprovalDecision):
        """Handle final approval completion"""
        try:
            # Track implementation
            await self.decision_tracker.track_decision_implemented(
                decision_id=request.request_id,
                implementation_data={
                    "approved_by": decision.approver_id,
                    "final_comments": decision.comments,
                    "implementation_status": "ready_for_execution",
                    "estimated_savings": float(request.decision.estimated_savings),
                    "implementation_timeline": request.decision.implementation_timeline
                }
            )
            
            logger.info(f"Final approval completed for decision {request.request_id}")
            
        except Exception as e:
            logger.error(f"Error handling final approval: {e}")
    
    async def _handle_rejection(self, request: ApprovalRequest, decision: ApprovalDecision):
        """Handle approval rejection"""
        try:
            # Update decision tracking
            await self.decision_tracker.track_approval_decision(
                decision_id=request.request_id,
                approver_id=decision.approver_id,
                approved=False,
                comments=f"REJECTED: {decision.comments}"
            )
            
            logger.info(f"Approval rejected for decision {request.request_id}")
            
        except Exception as e:
            logger.error(f"Error handling rejection: {e}")
    
    def _map_priority_to_notification_priority(self, priority: Priority) -> NotificationPriority:
        """Map approval priority to notification priority"""
        mapping = {
            Priority.LOW: NotificationPriority.LOW,
            Priority.MEDIUM: NotificationPriority.MEDIUM,
            Priority.HIGH: NotificationPriority.HIGH,
            Priority.CRITICAL: NotificationPriority.CRITICAL
        }
        return mapping.get(priority, NotificationPriority.MEDIUM)


# Global enhanced workflow engine instance
enhanced_workflow_engine = EnhancedApprovalWorkflowEngine()