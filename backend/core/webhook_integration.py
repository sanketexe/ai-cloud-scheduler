"""
Integration layer for webhook system with automation components.
Provides event emission helpers and integration points.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .webhook_manager import (
    WebhookEventType, emit_action_event, emit_system_event,
    get_webhook_manager
)
from .automation_models import OptimizationAction, ActionStatus
from .safety_checker import SafetyCheckResult


class WebhookIntegration:
    """
    Integration layer that connects the webhook system with automation components.
    Provides standardized event emission for all automation activities.
    """
    
    def __init__(self):
        self.webhook_manager = get_webhook_manager()
    
    async def on_action_created(self, action: OptimizationAction):
        """Emit event when optimization action is created"""
        await emit_action_event(action, WebhookEventType.ACTION_CREATED, {
            "created_at": action.created_at.isoformat(),
            "policy_name": getattr(action.policy, 'name', 'Unknown') if action.policy else 'Unknown'
        })
    
    async def on_action_scheduled(self, action: OptimizationAction):
        """Emit event when optimization action is scheduled"""
        await emit_action_event(action, WebhookEventType.ACTION_SCHEDULED, {
            "scheduled_for": action.scheduled_execution_time.isoformat() if action.scheduled_execution_time else None,
            "delay_reason": "business_hours" if action.scheduled_execution_time else None
        })
    
    async def on_action_started(self, action: OptimizationAction):
        """Emit event when optimization action starts executing"""
        await emit_action_event(action, WebhookEventType.ACTION_STARTED, {
            "started_at": action.execution_started_at.isoformat() if action.execution_started_at else datetime.now().isoformat(),
            "rollback_plan_ready": bool(action.rollback_plan)
        })
    
    async def on_action_completed(self, action: OptimizationAction, execution_details: Optional[Dict[str, Any]] = None):
        """Emit event when optimization action completes successfully"""
        event_data = {
            "completed_at": action.execution_completed_at.isoformat() if action.execution_completed_at else datetime.now().isoformat(),
            "execution_duration": self._calculate_duration(action),
            "cost_savings_realized": float(action.actual_savings) if action.actual_savings else 0.0
        }
        
        if execution_details:
            event_data.update(execution_details)
        
        await emit_action_event(action, WebhookEventType.ACTION_COMPLETED, event_data)
    
    async def on_action_failed(self, action: OptimizationAction, error_details: Dict[str, Any]):
        """Emit event when optimization action fails"""
        event_data = {
            "failed_at": datetime.now().isoformat(),
            "error_message": action.error_message,
            "execution_duration": self._calculate_duration(action),
            **error_details
        }
        
        await emit_action_event(action, WebhookEventType.ACTION_FAILED, event_data)
    
    async def on_action_rolled_back(self, action: OptimizationAction, rollback_details: Dict[str, Any]):
        """Emit event when optimization action is rolled back"""
        event_data = {
            "rolled_back_at": datetime.now().isoformat(),
            "rollback_reason": rollback_details.get("reason", "Unknown"),
            "rollback_success": rollback_details.get("success", False),
            "resources_restored": rollback_details.get("resources_restored", [])
        }
        
        await emit_action_event(action, WebhookEventType.ACTION_ROLLED_BACK, event_data)
    
    async def on_safety_check_failed(self, action: OptimizationAction, safety_results: list):
        """Emit event when safety checks fail for an action"""
        failed_checks = [
            {
                "check_name": result.check_name,
                "check_details": result.check_details,
                "checked_at": result.checked_at.isoformat()
            }
            for result in safety_results if not result.check_result
        ]
        
        event_data = {
            "failed_checks": failed_checks,
            "total_checks": len(safety_results),
            "failed_count": len(failed_checks)
        }
        
        await emit_action_event(action, WebhookEventType.SAFETY_CHECK_FAILED, event_data)
    
    async def on_policy_violation(self, action: OptimizationAction, violation_details: Dict[str, Any]):
        """Emit event when policy violation is detected"""
        event_data = {
            "violation_type": violation_details.get("type", "Unknown"),
            "policy_rule": violation_details.get("rule", "Unknown"),
            "violation_reason": violation_details.get("reason", "Unknown"),
            "detected_at": datetime.now().isoformat()
        }
        
        await emit_action_event(action, WebhookEventType.POLICY_VIOLATION, event_data)
    
    async def on_approval_required(self, action: OptimizationAction, approval_details: Dict[str, Any]):
        """Emit event when action requires approval"""
        event_data = {
            "approval_reason": approval_details.get("reason", "Policy requirement"),
            "requested_at": datetime.now().isoformat(),
            "expires_at": approval_details.get("expires_at"),
            "approver_roles": approval_details.get("approver_roles", [])
        }
        
        await emit_action_event(action, WebhookEventType.APPROVAL_REQUIRED, event_data)
    
    async def on_approval_granted(self, action: OptimizationAction, approval_details: Dict[str, Any]):
        """Emit event when action approval is granted"""
        event_data = {
            "approved_by": approval_details.get("approved_by", "Unknown"),
            "approved_at": approval_details.get("approved_at", datetime.now().isoformat()),
            "approval_comments": approval_details.get("comments", "")
        }
        
        await emit_action_event(action, WebhookEventType.APPROVAL_GRANTED, event_data)
    
    async def on_approval_denied(self, action: OptimizationAction, approval_details: Dict[str, Any]):
        """Emit event when action approval is denied"""
        event_data = {
            "denied_by": approval_details.get("denied_by", "Unknown"),
            "denied_at": approval_details.get("denied_at", datetime.now().isoformat()),
            "denial_reason": approval_details.get("reason", "No reason provided")
        }
        
        await emit_action_event(action, WebhookEventType.APPROVAL_DENIED, event_data)
    
    async def on_system_error(self, error_details: Dict[str, Any], 
                            resource_id: Optional[str] = None, 
                            action_id: Optional[str] = None):
        """Emit event for system-level errors"""
        event_data = {
            "error_type": error_details.get("type", "Unknown"),
            "error_message": error_details.get("message", "Unknown error"),
            "error_code": error_details.get("code"),
            "component": error_details.get("component", "Unknown"),
            "occurred_at": datetime.now().isoformat(),
            "stack_trace": error_details.get("stack_trace"),
            "context": error_details.get("context", {})
        }
        
        await emit_system_event(
            WebhookEventType.SYSTEM_ERROR, 
            event_data, 
            resource_id=resource_id, 
            action_id=action_id
        )
    
    async def on_savings_calculated(self, calculation_details: Dict[str, Any]):
        """Emit event when cost savings are calculated"""
        event_data = {
            "calculation_type": calculation_details.get("type", "monthly"),
            "total_savings": calculation_details.get("total_savings", 0.0),
            "savings_by_service": calculation_details.get("by_service", {}),
            "savings_by_action": calculation_details.get("by_action", {}),
            "calculation_period": calculation_details.get("period", "unknown"),
            "calculated_at": datetime.now().isoformat()
        }
        
        await emit_system_event(WebhookEventType.SAVINGS_CALCULATED, event_data)
    
    def _calculate_duration(self, action: OptimizationAction) -> Optional[float]:
        """Calculate execution duration in seconds"""
        if action.execution_started_at and action.execution_completed_at:
            delta = action.execution_completed_at - action.execution_started_at
            return delta.total_seconds()
        elif action.execution_started_at:
            delta = datetime.now() - action.execution_started_at.replace(tzinfo=None)
            return delta.total_seconds()
        return None


# Global integration instance
_webhook_integration = None

def get_webhook_integration() -> WebhookIntegration:
    """Get global webhook integration instance"""
    global _webhook_integration
    if _webhook_integration is None:
        _webhook_integration = WebhookIntegration()
    return _webhook_integration


# Decorator for automatic event emission
def emit_webhook_event(event_type: WebhookEventType):
    """Decorator to automatically emit webhook events for method calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            integration = get_webhook_integration()
            
            try:
                result = await func(*args, **kwargs)
                
                # Extract action from result or args if available
                action = None
                if hasattr(result, 'id') and hasattr(result, 'action_type'):
                    action = result
                elif args and hasattr(args[0], 'id') and hasattr(args[0], 'action_type'):
                    action = args[0]
                
                # Emit appropriate event
                if action:
                    if event_type == WebhookEventType.ACTION_CREATED:
                        await integration.on_action_created(action)
                    elif event_type == WebhookEventType.ACTION_SCHEDULED:
                        await integration.on_action_scheduled(action)
                    elif event_type == WebhookEventType.ACTION_STARTED:
                        await integration.on_action_started(action)
                    elif event_type == WebhookEventType.ACTION_COMPLETED:
                        await integration.on_action_completed(action)
                
                return result
                
            except Exception as e:
                # Emit error event
                await integration.on_system_error({
                    "type": "method_execution_error",
                    "message": str(e),
                    "component": func.__name__,
                    "context": {"args": str(args), "kwargs": str(kwargs)}
                })
                raise
        
        return wrapper
    return decorator


# Context manager for webhook event batching
class WebhookEventBatch:
    """Context manager for batching webhook events"""
    
    def __init__(self):
        self.events = []
        self.integration = get_webhook_integration()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Emit all batched events
        for event_func, args, kwargs in self.events:
            try:
                await event_func(*args, **kwargs)
            except Exception as e:
                # Log error but don't fail the batch
                import structlog
                logger = structlog.get_logger(__name__)
                logger.error("Failed to emit batched webhook event", error=str(e))
    
    def add_action_event(self, action: OptimizationAction, event_type: WebhookEventType, **kwargs):
        """Add action event to batch"""
        if event_type == WebhookEventType.ACTION_CREATED:
            self.events.append((self.integration.on_action_created, (action,), {}))
        elif event_type == WebhookEventType.ACTION_SCHEDULED:
            self.events.append((self.integration.on_action_scheduled, (action,), {}))
        elif event_type == WebhookEventType.ACTION_STARTED:
            self.events.append((self.integration.on_action_started, (action,), {}))
        elif event_type == WebhookEventType.ACTION_COMPLETED:
            self.events.append((self.integration.on_action_completed, (action, kwargs.get('execution_details')), {}))
        elif event_type == WebhookEventType.ACTION_FAILED:
            self.events.append((self.integration.on_action_failed, (action, kwargs.get('error_details', {})), {}))
    
    def add_system_event(self, event_type: WebhookEventType, data: Dict[str, Any], **kwargs):
        """Add system event to batch"""
        if event_type == WebhookEventType.SYSTEM_ERROR:
            self.events.append((self.integration.on_system_error, (data,), kwargs))
        elif event_type == WebhookEventType.SAVINGS_CALCULATED:
            self.events.append((self.integration.on_savings_calculated, (data,), {}))


# Utility functions for common webhook patterns
async def notify_action_lifecycle(action: OptimizationAction, status: ActionStatus, **kwargs):
    """Notify about action lifecycle changes"""
    integration = get_webhook_integration()
    
    if status == ActionStatus.PENDING:
        await integration.on_action_created(action)
    elif status == ActionStatus.SCHEDULED:
        await integration.on_action_scheduled(action)
    elif status == ActionStatus.EXECUTING:
        await integration.on_action_started(action)
    elif status == ActionStatus.COMPLETED:
        await integration.on_action_completed(action, kwargs.get('execution_details'))
    elif status == ActionStatus.FAILED:
        await integration.on_action_failed(action, kwargs.get('error_details', {}))
    elif status == ActionStatus.ROLLED_BACK:
        await integration.on_action_rolled_back(action, kwargs.get('rollback_details', {}))


async def notify_bulk_actions(actions: list, event_type: WebhookEventType, **kwargs):
    """Efficiently notify about multiple actions"""
    async with WebhookEventBatch() as batch:
        for action in actions:
            batch.add_action_event(action, event_type, **kwargs)


async def start_webhook_system():
    """Initialize and start the webhook system"""
    webhook_manager = get_webhook_manager()
    await webhook_manager.start()


async def stop_webhook_system():
    """Stop the webhook system"""
    webhook_manager = get_webhook_manager()
    await webhook_manager.stop()