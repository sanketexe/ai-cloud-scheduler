"""
Approval Workflow Engine for Real-Time Collaborative FinOps Workspace

This module implements the approval workflow engine that manages organizational
approval processes for cost optimization decisions with threshold-based routing,
escalation procedures, and complete decision context tracking.

Requirements addressed:
- 2.1: Threshold-based approval routing with configurable chains
- 2.2: Approval context provision with complete decision history
- 2.3: Intelligent notification routing based on roles and urgency
- 2.4: Immediate notification to all session participants
- 2.5: Escalation procedures with deadline management
"""

import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from sqlalchemy.exc import IntegrityError

from .database import get_db_session
from .models import User
from .collaboration_models import CollaborativeSession, SessionParticipant
from .redis_config import redis_manager
from .decision_tracking_system import notification_service

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions requiring approval"""
    COST_OPTIMIZATION = "cost_optimization"
    BUDGET_CHANGE = "budget_change"
    RESOURCE_TERMINATION = "resource_termination"
    POLICY_CHANGE = "policy_change"
    EMERGENCY_ACTION = "emergency_action"

class Priority(Enum):
    """Priority levels for approval requests"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ApprovalStatus(Enum):
    """Status of approval requests"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class EscalationReason(Enum):
    """Reasons for escalation"""
    DEADLINE_APPROACHING = "deadline_approaching"
    APPROVER_UNAVAILABLE = "approver_unavailable"
    COMPLEXITY_THRESHOLD = "complexity_threshold"
    MANUAL_ESCALATION = "manual_escalation"

@dataclass
class OptimizationAction:
    """Optimization action requiring approval"""
    action_id: str
    action_type: str
    resource_id: str
    resource_type: str
    cost_impact: Decimal
    risk_level: str
    description: str
    proposed_changes: Dict[str, Any]
    estimated_savings: Decimal
    implementation_timeline: str

@dataclass
class DecisionContext:
    """Complete context for approval decisions"""
    session_id: str
    participants: List[str]
    discussion_summary: str
    data_analyzed: List[Dict[str, Any]]
    alternatives_considered: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    cost_benefit_analysis: Dict[str, Any]
    timeline_constraints: Dict[str, Any]

@dataclass
class ApprovalStep:
    """Individual step in approval chain"""
    step_id: str
    approver_id: str
    approver_role: str
    required: bool
    parallel: bool = False
    timeout_hours: int = 24
    escalation_chain: List[str] = None

@dataclass
class ApprovalChain:
    """Complete approval chain configuration"""
    chain_id: str
    name: str
    description: str
    steps: List[ApprovalStep]
    threshold_conditions: Dict[str, Any]
    escalation_rules: Dict[str, Any]

@dataclass
class ApprovalRequest:
    """Approval request with complete context"""
    request_id: str
    decision: OptimizationAction
    context: DecisionContext
    approval_chain: ApprovalChain
    current_step: int
    priority: Priority
    deadline: datetime
    created_at: datetime
    created_by: str

@dataclass
class ApprovalDecision:
    """Individual approval decision"""
    decision_id: str
    approver_id: str
    approved: bool
    comments: str
    conditions: List[str] = None
    decided_at: datetime = None

@dataclass
class ApprovalResult:
    """Result of approval processing"""
    success: bool
    request_id: Optional[str] = None
    status: Optional[ApprovalStatus] = None
    next_step: Optional[int] = None
    error_message: Optional[str] = None

@dataclass
class EscalationResult:
    """Result of escalation processing"""
    success: bool
    escalated_to: Optional[List[str]] = None
    reason: Optional[EscalationReason] = None
    error_message: Optional[str] = None

class ApprovalWorkflowEngine:
    """
    Manages approval workflows for cost optimization decisions with threshold-based
    routing, escalation procedures, and complete decision context tracking.
    """
    
    def __init__(self):
        self.active_requests: Dict[str, ApprovalRequest] = {}
        self.approval_chains: Dict[str, ApprovalChain] = {}
        self.threshold_configs: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_chains()
    
    def _initialize_default_chains(self):
        """Initialize default approval chains"""
        # Low-impact approval chain
        low_impact_chain = ApprovalChain(
            chain_id="low_impact",
            name="Low Impact Approvals",
            description="For low-risk, low-cost optimization decisions",
            steps=[
                ApprovalStep(
                    step_id="team_lead",
                    approver_id="team_lead",
                    approver_role="team_lead",
                    required=True,
                    timeout_hours=4,
                    escalation_chain=["manager"]
                )
            ],
            threshold_conditions={
                "max_cost_impact": 1000,
                "max_risk_level": "low"
            },
            escalation_rules={
                "timeout_escalation": True,
                "complexity_escalation": False
            }
        )
        
        # Medium-impact approval chain
        medium_impact_chain = ApprovalChain(
            chain_id="medium_impact",
            name="Medium Impact Approvals",
            description="For medium-risk optimization decisions",
            steps=[
                ApprovalStep(
                    step_id="team_lead",
                    approver_id="team_lead",
                    approver_role="team_lead",
                    required=True,
                    timeout_hours=8,
                    escalation_chain=["manager"]
                ),
                ApprovalStep(
                    step_id="manager",
                    approver_id="manager",
                    approver_role="manager",
                    required=True,
                    timeout_hours=12,
                    escalation_chain=["director"]
                )
            ],
            threshold_conditions={
                "max_cost_impact": 10000,
                "max_risk_level": "medium"
            },
            escalation_rules={
                "timeout_escalation": True,
                "complexity_escalation": True
            }
        )
        
        # High-impact approval chain
        high_impact_chain = ApprovalChain(
            chain_id="high_impact",
            name="High Impact Approvals",
            description="For high-risk, high-cost optimization decisions",
            steps=[
                ApprovalStep(
                    step_id="team_lead",
                    approver_id="team_lead",
                    approver_role="team_lead",
                    required=True,
                    timeout_hours=4,
                    escalation_chain=["manager"]
                ),
                ApprovalStep(
                    step_id="manager",
                    approver_id="manager",
                    approver_role="manager",
                    required=True,
                    timeout_hours=8,
                    escalation_chain=["director"]
                ),
                ApprovalStep(
                    step_id="director",
                    approver_id="director",
                    approver_role="director",
                    required=True,
                    timeout_hours=24,
                    escalation_chain=["cfo"]
                )
            ],
            threshold_conditions={
                "min_cost_impact": 10000,
                "min_risk_level": "high"
            },
            escalation_rules={
                "timeout_escalation": True,
                "complexity_escalation": True,
                "executive_review": True
            }
        )
        
        self.approval_chains = {
            "low_impact": low_impact_chain,
            "medium_impact": medium_impact_chain,
            "high_impact": high_impact_chain
        }
    
    async def submit_for_approval(self, decision: OptimizationAction, context: DecisionContext) -> ApprovalRequest:
        """
        Submit an optimization decision for approval routing
        
        Args:
            decision: The optimization decision requiring approval
            context: Complete decision context including session data
            
        Returns:
            ApprovalRequest: The created approval request
            
        Raises:
            ValueError: If decision or context is invalid
            RuntimeError: If approval request creation fails
        """
        try:
            # Determine appropriate approval chain based on thresholds
            approval_chain = self._determine_approval_chain(decision)
            
            # Calculate priority based on cost impact and risk
            priority = self._calculate_priority(decision)
            
            # Set deadline based on priority and chain configuration
            deadline = self._calculate_deadline(priority, approval_chain)
            
            # Create approval request
            request = ApprovalRequest(
                request_id=str(uuid.uuid4()),
                decision=decision,
                context=context,
                approval_chain=approval_chain,
                current_step=0,
                priority=priority,
                deadline=deadline,
                created_at=datetime.utcnow(),
                created_by=context.session_id  # Using session_id as creator for now
            )
            
            # Store request
            self.active_requests[request.request_id] = request
            
            # Cache request in Redis for persistence
            await self._cache_approval_request(request)
            
            # Start approval process
            await self._start_approval_process(request)
            
            logger.info(f"Created approval request {request.request_id} for decision {decision.action_id}")
            return request
            
        except Exception as e:
            logger.error(f"Error creating approval request: {e}")
            raise RuntimeError(f"Failed to create approval request: {e}")
    
    async def process_approval(self, approval_id: str, approver_id: str, decision: ApprovalDecision) -> ApprovalResult:
        """
        Process an approval decision
        
        Args:
            approval_id: ID of the approval request
            approver_id: ID of the approver making the decision
            decision: The approval decision
            
        Returns:
            ApprovalResult: Result of processing the approval
        """
        try:
            # Get approval request
            request = await self._get_approval_request(approval_id)
            if not request:
                return ApprovalResult(success=False, error_message="Approval request not found")
            
            # Validate approver is authorized for current step
            current_step = request.approval_chain.steps[request.current_step]
            if not await self._validate_approver(approver_id, current_step):
                return ApprovalResult(success=False, error_message="Approver not authorized for this step")
            
            # Record the decision
            await self._record_approval_decision(request, decision)
            
            # Process the decision
            if decision.approved:
                # Move to next step or complete approval
                if request.current_step + 1 < len(request.approval_chain.steps):
                    request.current_step += 1
                    await self._notify_next_approver(request)
                    status = ApprovalStatus.PENDING
                else:
                    # All approvals complete
                    status = ApprovalStatus.APPROVED
                    await self._complete_approval(request)
            else:
                # Rejection - complete the process
                status = ApprovalStatus.REJECTED
                await self._complete_rejection(request, decision.comments)
            
            # Update request status
            await self._update_request_status(request, status)
            
            # Notify session participants
            await self._notify_session_participants(request, decision)
            
            return ApprovalResult(
                success=True,
                request_id=approval_id,
                status=status,
                next_step=request.current_step if status == ApprovalStatus.PENDING else None
            )
            
        except Exception as e:
            logger.error(f"Error processing approval {approval_id}: {e}")
            return ApprovalResult(success=False, error_message=str(e))
    
    async def escalate_approval(self, approval_id: str, reason: EscalationReason) -> EscalationResult:
        """
        Escalate an approval request
        
        Args:
            approval_id: ID of the approval request to escalate
            reason: Reason for escalation
            
        Returns:
            EscalationResult: Result of the escalation
        """
        try:
            # Get approval request
            request = await self._get_approval_request(approval_id)
            if not request:
                return EscalationResult(success=False, error_message="Approval request not found")
            
            # Get current step
            current_step = request.approval_chain.steps[request.current_step]
            
            # Determine escalation targets
            escalation_targets = []
            if current_step.escalation_chain:
                escalation_targets = current_step.escalation_chain
            else:
                # Default escalation logic
                escalation_targets = await self._get_default_escalation_targets(request)
            
            # Record escalation
            await self._record_escalation(request, reason, escalation_targets)
            
            # Notify escalation targets
            await self._notify_escalation_targets(request, escalation_targets, reason)
            
            # Update request status
            await self._update_request_status(request, ApprovalStatus.ESCALATED)
            
            logger.info(f"Escalated approval request {approval_id} to {escalation_targets}")
            return EscalationResult(
                success=True,
                escalated_to=escalation_targets,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"Error escalating approval {approval_id}: {e}")
            return EscalationResult(success=False, error_message=str(e))
    
    async def get_approval_status(self, approval_id: str) -> Dict[str, Any]:
        """
        Get current status of an approval request
        
        Args:
            approval_id: ID of the approval request
            
        Returns:
            Dict containing approval status and details
        """
        try:
            request = await self._get_approval_request(approval_id)
            if not request:
                return {"error": "Approval request not found"}
            
            # Get approval history
            history = await self._get_approval_history(approval_id)
            
            # Calculate progress
            total_steps = len(request.approval_chain.steps)
            completed_steps = request.current_step
            progress_percentage = (completed_steps / total_steps) * 100
            
            return {
                "request_id": approval_id,
                "status": "pending",  # This would be determined from the request
                "decision_type": request.decision.action_type,
                "cost_impact": float(request.decision.cost_impact),
                "priority": request.priority.value,
                "deadline": request.deadline.isoformat(),
                "current_step": request.current_step,
                "total_steps": total_steps,
                "progress_percentage": progress_percentage,
                "approval_chain": request.approval_chain.name,
                "history": history,
                "context": {
                    "session_id": request.context.session_id,
                    "participants": request.context.participants,
                    "discussion_summary": request.context.discussion_summary
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting approval status {approval_id}: {e}")
            return {"error": str(e)}
    
    def _determine_approval_chain(self, decision: OptimizationAction) -> ApprovalChain:
        """Determine appropriate approval chain based on decision thresholds"""
        cost_impact = float(decision.cost_impact)
        risk_level = decision.risk_level
        
        # High impact thresholds
        if cost_impact >= 10000 or risk_level == "high":
            return self.approval_chains["high_impact"]
        
        # Medium impact thresholds
        elif cost_impact >= 1000 or risk_level == "medium":
            return self.approval_chains["medium_impact"]
        
        # Low impact (default)
        else:
            return self.approval_chains["low_impact"]
    
    def _calculate_priority(self, decision: OptimizationAction) -> Priority:
        """Calculate priority based on cost impact and risk level"""
        cost_impact = float(decision.cost_impact)
        risk_level = decision.risk_level
        
        if cost_impact >= 50000 or risk_level == "high":
            return Priority.CRITICAL
        elif cost_impact >= 10000 or risk_level == "medium":
            return Priority.HIGH
        elif cost_impact >= 1000:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _calculate_deadline(self, priority: Priority, chain: ApprovalChain) -> datetime:
        """Calculate approval deadline based on priority and chain configuration"""
        base_hours = sum(step.timeout_hours for step in chain.steps)
        
        # Adjust based on priority
        if priority == Priority.CRITICAL:
            hours = base_hours * 0.5  # 50% of normal time
        elif priority == Priority.HIGH:
            hours = base_hours * 0.75  # 75% of normal time
        elif priority == Priority.MEDIUM:
            hours = base_hours  # Normal time
        else:
            hours = base_hours * 1.5  # 150% of normal time
        
        return datetime.utcnow() + timedelta(hours=hours)
    
    async def _cache_approval_request(self, request: ApprovalRequest):
        """Cache approval request in Redis"""
        try:
            cache_key = f"approval_request:{request.request_id}"
            request_data = {
                "request_id": request.request_id,
                "decision": {
                    "action_id": request.decision.action_id,
                    "action_type": request.decision.action_type,
                    "resource_id": request.decision.resource_id,
                    "cost_impact": float(request.decision.cost_impact),
                    "risk_level": request.decision.risk_level,
                    "description": request.decision.description
                },
                "context": {
                    "session_id": request.context.session_id,
                    "participants": request.context.participants,
                    "discussion_summary": request.context.discussion_summary
                },
                "approval_chain_id": request.approval_chain.chain_id,
                "current_step": request.current_step,
                "priority": request.priority.value,
                "deadline": request.deadline.isoformat(),
                "created_at": request.created_at.isoformat(),
                "created_by": request.created_by
            }
            
            await redis_manager.set_json(cache_key, request_data, expire=86400 * 7)  # 7 days
            
        except Exception as e:
            logger.error(f"Error caching approval request: {e}")
    
    async def _get_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get approval request from cache or memory"""
        # First check memory
        if request_id in self.active_requests:
            return self.active_requests[request_id]
        
        # Then check Redis cache
        try:
            cache_key = f"approval_request:{request_id}"
            request_data = await redis_manager.get_json(cache_key)
            if request_data:
                # Reconstruct request object (simplified for this implementation)
                return self._reconstruct_approval_request(request_data)
        except Exception as e:
            logger.error(f"Error getting approval request from cache: {e}")
        
        return None
    
    def _reconstruct_approval_request(self, data: Dict[str, Any]) -> ApprovalRequest:
        """Reconstruct ApprovalRequest from cached data"""
        # This is a simplified reconstruction - in production you'd want more robust serialization
        decision = OptimizationAction(
            action_id=data["decision"]["action_id"],
            action_type=data["decision"]["action_type"],
            resource_id=data["decision"]["resource_id"],
            resource_type="",  # Would need to be stored
            cost_impact=Decimal(str(data["decision"]["cost_impact"])),
            risk_level=data["decision"]["risk_level"],
            description=data["decision"]["description"],
            proposed_changes={},
            estimated_savings=Decimal("0"),
            implementation_timeline=""
        )
        
        context = DecisionContext(
            session_id=data["context"]["session_id"],
            participants=data["context"]["participants"],
            discussion_summary=data["context"]["discussion_summary"],
            data_analyzed=[],
            alternatives_considered=[],
            risk_assessment={},
            cost_benefit_analysis={},
            timeline_constraints={}
        )
        
        approval_chain = self.approval_chains.get(data["approval_chain_id"])
        
        return ApprovalRequest(
            request_id=data["request_id"],
            decision=decision,
            context=context,
            approval_chain=approval_chain,
            current_step=data["current_step"],
            priority=Priority(data["priority"]),
            deadline=datetime.fromisoformat(data["deadline"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"]
        )
    
    async def _start_approval_process(self, request: ApprovalRequest):
        """Start the approval process by notifying first approver"""
        try:
            first_step = request.approval_chain.steps[0]
            await self._notify_approver(request, first_step)
            
            # Schedule deadline check
            await self._schedule_deadline_check(request)
            
        except Exception as e:
            logger.error(f"Error starting approval process: {e}")
    
    async def _notify_approver(self, request: ApprovalRequest, step: ApprovalStep):
        """Notify approver about pending approval"""
        try:
            notification_data = {
                "type": "approval_request",
                "request_id": request.request_id,
                "decision_type": request.decision.action_type,
                "cost_impact": float(request.decision.cost_impact),
                "priority": request.priority.value,
                "deadline": request.deadline.isoformat(),
                "description": request.decision.description,
                "session_id": request.context.session_id
            }
            
            # Send notification (implementation would depend on notification service)
            await notification_service.send_notification(
                recipient_id=step.approver_id,
                notification_type="approval_request",
                data=notification_data,
                priority=request.priority.value
            )
            
        except Exception as e:
            logger.error(f"Error notifying approver: {e}")
    
    async def _validate_approver(self, approver_id: str, step: ApprovalStep) -> bool:
        """Validate that approver is authorized for the step"""
        # In a real implementation, this would check user roles and permissions
        # For now, we'll do a simple check
        return step.approver_id == approver_id or step.approver_role == "any"
    
    async def _record_approval_decision(self, request: ApprovalRequest, decision: ApprovalDecision):
        """Record approval decision in audit trail"""
        try:
            audit_key = f"approval_audit:{request.request_id}"
            audit_entry = {
                "decision_id": decision.decision_id,
                "approver_id": decision.approver_id,
                "approved": decision.approved,
                "comments": decision.comments,
                "conditions": decision.conditions or [],
                "decided_at": decision.decided_at.isoformat() if decision.decided_at else datetime.utcnow().isoformat(),
                "step": request.current_step
            }
            
            # Append to audit trail
            existing_audit = await redis_manager.get_json(audit_key) or []
            existing_audit.append(audit_entry)
            await redis_manager.set_json(audit_key, existing_audit, expire=86400 * 30)  # 30 days
            
        except Exception as e:
            logger.error(f"Error recording approval decision: {e}")
    
    async def _notify_next_approver(self, request: ApprovalRequest):
        """Notify next approver in the chain"""
        if request.current_step < len(request.approval_chain.steps):
            next_step = request.approval_chain.steps[request.current_step]
            await self._notify_approver(request, next_step)
    
    async def _complete_approval(self, request: ApprovalRequest):
        """Complete the approval process"""
        try:
            # Notify session participants of approval
            await self._notify_session_participants(request, None, "approved")
            
            # Execute the approved action (would integrate with action execution system)
            logger.info(f"Approval request {request.request_id} completed - action approved")
            
        except Exception as e:
            logger.error(f"Error completing approval: {e}")
    
    async def _complete_rejection(self, request: ApprovalRequest, reason: str):
        """Complete the rejection process"""
        try:
            # Notify session participants of rejection
            await self._notify_session_participants(request, None, "rejected", reason)
            
            logger.info(f"Approval request {request.request_id} rejected: {reason}")
            
        except Exception as e:
            logger.error(f"Error completing rejection: {e}")
    
    async def _update_request_status(self, request: ApprovalRequest, status: ApprovalStatus):
        """Update request status in cache"""
        try:
            cache_key = f"approval_request:{request.request_id}"
            request_data = await redis_manager.get_json(cache_key)
            if request_data:
                request_data["status"] = status.value
                request_data["updated_at"] = datetime.utcnow().isoformat()
                await redis_manager.set_json(cache_key, request_data, expire=86400 * 7)
            
        except Exception as e:
            logger.error(f"Error updating request status: {e}")
    
    async def _notify_session_participants(self, request: ApprovalRequest, decision: Optional[ApprovalDecision], 
                                         status: str = None, reason: str = None):
        """Notify all session participants about approval status changes"""
        try:
            notification_data = {
                "type": "approval_status_update",
                "request_id": request.request_id,
                "decision_type": request.decision.action_type,
                "status": status or ("approved" if decision and decision.approved else "rejected"),
                "approver_id": decision.approver_id if decision else None,
                "comments": decision.comments if decision else reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Notify all session participants
            for participant_id in request.context.participants:
                await notification_service.send_notification(
                    recipient_id=participant_id,
                    notification_type="approval_update",
                    data=notification_data,
                    priority="medium"
                )
            
        except Exception as e:
            logger.error(f"Error notifying session participants: {e}")
    
    async def _schedule_deadline_check(self, request: ApprovalRequest):
        """Schedule deadline monitoring for the request"""
        # This would integrate with a task scheduler like Celery
        # For now, we'll just log the scheduling
        logger.info(f"Scheduled deadline check for approval request {request.request_id} at {request.deadline}")
    
    async def _record_escalation(self, request: ApprovalRequest, reason: EscalationReason, targets: List[str]):
        """Record escalation in audit trail"""
        try:
            escalation_key = f"approval_escalation:{request.request_id}"
            escalation_entry = {
                "reason": reason.value,
                "escalated_to": targets,
                "escalated_at": datetime.utcnow().isoformat(),
                "step": request.current_step
            }
            
            existing_escalations = await redis_manager.get_json(escalation_key) or []
            existing_escalations.append(escalation_entry)
            await redis_manager.set_json(escalation_key, existing_escalations, expire=86400 * 30)
            
        except Exception as e:
            logger.error(f"Error recording escalation: {e}")
    
    async def _notify_escalation_targets(self, request: ApprovalRequest, targets: List[str], reason: EscalationReason):
        """Notify escalation targets"""
        try:
            notification_data = {
                "type": "approval_escalation",
                "request_id": request.request_id,
                "reason": reason.value,
                "original_approver": request.approval_chain.steps[request.current_step].approver_id,
                "decision_type": request.decision.action_type,
                "cost_impact": float(request.decision.cost_impact),
                "priority": request.priority.value,
                "deadline": request.deadline.isoformat()
            }
            
            for target_id in targets:
                await notification_service.send_notification(
                    recipient_id=target_id,
                    notification_type="approval_escalation",
                    data=notification_data,
                    priority="high"
                )
                
        except Exception as e:
            logger.error(f"Error notifying escalation targets: {e}")
    
    async def _get_default_escalation_targets(self, request: ApprovalRequest) -> List[str]:
        """Get default escalation targets when none are configured"""
        # This would query user management system for hierarchy
        # For now, return a default escalation path
        return ["manager", "director"]
    
    async def _get_approval_history(self, request_id: str) -> List[Dict[str, Any]]:
        """Get approval history for a request"""
        try:
            audit_key = f"approval_audit:{request_id}"
            history = await redis_manager.get_json(audit_key) or []
            return history
        except Exception as e:
            logger.error(f"Error getting approval history: {e}")
            return []

# Global workflow engine instance
workflow_engine = ApprovalWorkflowEngine()