"""
Rollback Manager for Automated Cost Optimization

Handles action failures and provides rollback capabilities:
- Create rollback plans before action execution
- Monitor post-action health with automatic triggers
- Execute rollbacks when issues are detected
- Calculate rollback costs and impact
- Provide manual rollback capabilities with impact assessment
- Validate rollback success with retry logic
"""

import uuid
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from decimal import Decimal
from enum import Enum
import structlog

from .automation_models import OptimizationAction, ActionType, ActionStatus
from .automation_audit_logger import AutomationAuditLogger

logger = structlog.get_logger(__name__)


class RollbackTrigger(Enum):
    """Types of rollback triggers"""
    MANUAL = "manual"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_THRESHOLD_EXCEEDED = "error_threshold_exceeded"
    DEPENDENCY_FAILURE = "dependency_failure"
    TIMEOUT = "timeout"


class RollbackStatus(Enum):
    """Status of rollback operations"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


class HealthCheckResult(Enum):
    """Health check results"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RollbackManager:
    """
    Manages rollback operations for automated cost optimization actions.
    
    Provides capabilities to:
    - Create rollback plans before executing actions
    - Monitor system health after actions with automatic triggers
    - Execute rollbacks when issues are detected
    - Calculate the cost impact of rollbacks
    - Provide manual rollback capabilities with impact assessment
    - Validate rollback success with retry logic
    """
    
    def __init__(self):
        self.audit_logger = AutomationAuditLogger()
        # In a real implementation, this would initialize AWS clients
        self.aws_clients = {}
        
        # Health monitoring configuration
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitored_actions: Dict[str, OptimizationAction] = {}
        self.health_check_interval = 60  # seconds
        self.max_retry_attempts = 3
        self.retry_delay = 30  # seconds
        
        # Rollback triggers configuration
        self.rollback_triggers = {
            RollbackTrigger.HEALTH_CHECK_FAILURE: True,
            RollbackTrigger.PERFORMANCE_DEGRADATION: True,
            RollbackTrigger.ERROR_THRESHOLD_EXCEEDED: True,
            RollbackTrigger.DEPENDENCY_FAILURE: True,
            RollbackTrigger.TIMEOUT: True
        }
        
        # Health check thresholds
        self.health_thresholds = {
            "error_rate_threshold": 0.05,  # 5% error rate
            "response_time_threshold": 5000,  # 5 seconds
            "cpu_threshold": 90,  # 90% CPU
            "memory_threshold": 90,  # 90% memory
            "disk_threshold": 95  # 95% disk usage
        }
    
    def create_rollback_plan(self, opportunity) -> Dict[str, Any]:
        """
        Create a rollback plan for an optimization opportunity.
        
        Args:
            opportunity: OptimizationOpportunity to create rollback plan for
            
        Returns:
            Dictionary containing rollback plan details
        """
        logger.info("Creating rollback plan",
                   resource_id=opportunity.resource_id,
                   action_type=opportunity.action_type.value)
        
        rollback_plan = {
            "resource_id": opportunity.resource_id,
            "resource_type": opportunity.resource_type,
            "action_type": opportunity.action_type.value,
            "created_at": datetime.utcnow().isoformat(),
            "rollback_steps": [],
            "prerequisites": [],
            "estimated_rollback_cost": 0.0,
            "rollback_time_estimate_minutes": 5
        }
        
        # Create action-specific rollback steps
        if opportunity.action_type == ActionType.STOP_INSTANCE:
            rollback_plan.update(self._create_instance_stop_rollback_plan(opportunity))
        elif opportunity.action_type == ActionType.TERMINATE_INSTANCE:
            rollback_plan.update(self._create_instance_terminate_rollback_plan(opportunity))
        elif opportunity.action_type == ActionType.RESIZE_INSTANCE:
            rollback_plan.update(self._create_instance_resize_rollback_plan(opportunity))
        elif opportunity.action_type == ActionType.DELETE_VOLUME:
            rollback_plan.update(self._create_volume_delete_rollback_plan(opportunity))
        elif opportunity.action_type == ActionType.UPGRADE_STORAGE:
            rollback_plan.update(self._create_storage_upgrade_rollback_plan(opportunity))
        elif opportunity.action_type == ActionType.RELEASE_ELASTIC_IP:
            rollback_plan.update(self._create_elastic_ip_rollback_plan(opportunity))
        elif opportunity.action_type == ActionType.DELETE_LOAD_BALANCER:
            rollback_plan.update(self._create_load_balancer_rollback_plan(opportunity))
        elif opportunity.action_type == ActionType.CLEANUP_SECURITY_GROUP:
            rollback_plan.update(self._create_security_group_rollback_plan(opportunity))
        
        logger.info("Rollback plan created",
                   resource_id=opportunity.resource_id,
                   steps_count=len(rollback_plan["rollback_steps"]))
        
        return rollback_plan
    
    def _create_instance_stop_rollback_plan(self, opportunity) -> Dict[str, Any]:
        """Create rollback plan for stopping an instance"""
        return {
            "rollback_steps": [
                {
                    "step": 1,
                    "action": "start_instance",
                    "parameters": {"instance_id": opportunity.resource_id},
                    "description": "Start the stopped instance"
                }
            ],
            "prerequisites": [
                "Instance must be in 'stopped' state",
                "No instance modifications during downtime"
            ],
            "estimated_rollback_cost": float(opportunity.estimated_monthly_savings),
            "rollback_time_estimate_minutes": 3
        }
    
    def _create_instance_terminate_rollback_plan(self, opportunity) -> Dict[str, Any]:
        """Create rollback plan for terminating an instance"""
        # Termination is generally not reversible, but we can document recovery steps
        return {
            "rollback_steps": [
                {
                    "step": 1,
                    "action": "launch_replacement_instance",
                    "parameters": {
                        "instance_type": opportunity.resource_metadata.get("instance_type"),
                        "ami_id": opportunity.resource_metadata.get("ami_id"),
                        "subnet_id": opportunity.resource_metadata.get("subnet_id"),
                        "security_groups": opportunity.resource_metadata.get("security_groups", [])
                    },
                    "description": "Launch replacement instance with same configuration"
                },
                {
                    "step": 2,
                    "action": "restore_from_backup",
                    "parameters": {"backup_source": "latest_snapshot"},
                    "description": "Restore data from latest backup/snapshot"
                }
            ],
            "prerequisites": [
                "Recent AMI or snapshot available",
                "Instance configuration documented",
                "Data backup exists"
            ],
            "estimated_rollback_cost": float(opportunity.estimated_monthly_savings * 2),  # Higher cost due to complexity
            "rollback_time_estimate_minutes": 30,
            "rollback_complexity": "high",
            "data_loss_risk": "medium"
        }
    
    def _create_instance_resize_rollback_plan(self, opportunity) -> Dict[str, Any]:
        """Create rollback plan for resizing an instance"""
        current_type = opportunity.resource_metadata.get("instance_type")
        return {
            "rollback_steps": [
                {
                    "step": 1,
                    "action": "stop_instance",
                    "parameters": {"instance_id": opportunity.resource_id},
                    "description": "Stop instance for type change"
                },
                {
                    "step": 2,
                    "action": "modify_instance_type",
                    "parameters": {
                        "instance_id": opportunity.resource_id,
                        "instance_type": current_type
                    },
                    "description": f"Restore original instance type: {current_type}"
                },
                {
                    "step": 3,
                    "action": "start_instance",
                    "parameters": {"instance_id": opportunity.resource_id},
                    "description": "Start instance with original type"
                }
            ],
            "prerequisites": [
                "Instance can be stopped safely",
                "Original instance type is compatible"
            ],
            "estimated_rollback_cost": float(opportunity.estimated_monthly_savings * 0.1),
            "rollback_time_estimate_minutes": 10
        }
    
    def _create_volume_delete_rollback_plan(self, opportunity) -> Dict[str, Any]:
        """Create rollback plan for deleting a volume"""
        return {
            "rollback_steps": [
                {
                    "step": 1,
                    "action": "create_volume_from_snapshot",
                    "parameters": {
                        "snapshot_id": "will_be_created_during_action",
                        "availability_zone": opportunity.resource_metadata.get("availability_zone"),
                        "volume_type": opportunity.resource_metadata.get("volume_type", "gp3")
                    },
                    "description": "Recreate volume from snapshot taken before deletion"
                }
            ],
            "prerequisites": [
                "Snapshot created successfully before deletion",
                "Sufficient storage quota available"
            ],
            "estimated_rollback_cost": float(opportunity.estimated_monthly_savings),
            "rollback_time_estimate_minutes": 15
        }
    
    def _create_storage_upgrade_rollback_plan(self, opportunity) -> Dict[str, Any]:
        """Create rollback plan for storage type upgrade"""
        original_type = opportunity.resource_metadata.get("volume_type", "gp2")
        return {
            "rollback_steps": [
                {
                    "step": 1,
                    "action": "modify_volume_type",
                    "parameters": {
                        "volume_id": opportunity.resource_id,
                        "volume_type": original_type
                    },
                    "description": f"Revert volume type to {original_type}"
                }
            ],
            "prerequisites": [
                "Volume modification not in progress",
                "Original volume type supported"
            ],
            "estimated_rollback_cost": float(opportunity.estimated_monthly_savings * -1),  # Negative savings
            "rollback_time_estimate_minutes": 5
        }
    
    def _create_elastic_ip_rollback_plan(self, opportunity) -> Dict[str, Any]:
        """Create rollback plan for releasing Elastic IP"""
        return {
            "rollback_steps": [
                {
                    "step": 1,
                    "action": "allocate_elastic_ip",
                    "parameters": {"domain": "vpc"},
                    "description": "Allocate new Elastic IP (original IP cannot be recovered)"
                }
            ],
            "prerequisites": [
                "Elastic IP quota available",
                "Application can handle IP address change"
            ],
            "estimated_rollback_cost": float(opportunity.estimated_monthly_savings),
            "rollback_time_estimate_minutes": 2,
            "rollback_limitations": ["Original IP address cannot be recovered"]
        }
    
    def _create_load_balancer_rollback_plan(self, opportunity) -> Dict[str, Any]:
        """Create rollback plan for deleting load balancer"""
        lb_config = opportunity.resource_metadata
        return {
            "rollback_steps": [
                {
                    "step": 1,
                    "action": "create_load_balancer",
                    "parameters": {
                        "name": lb_config.get("name", "restored-lb"),
                        "scheme": lb_config.get("scheme", "internet-facing"),
                        "type": lb_config.get("type", "application"),
                        "subnets": lb_config.get("subnets", []),
                        "security_groups": lb_config.get("security_groups", [])
                    },
                    "description": "Recreate load balancer with original configuration"
                },
                {
                    "step": 2,
                    "action": "restore_target_groups",
                    "parameters": {"target_groups": lb_config.get("target_groups", [])},
                    "description": "Recreate and attach target groups"
                }
            ],
            "prerequisites": [
                "Load balancer configuration documented",
                "Target instances still available",
                "Security groups exist"
            ],
            "estimated_rollback_cost": float(opportunity.estimated_monthly_savings),
            "rollback_time_estimate_minutes": 20
        }
    
    def _create_security_group_rollback_plan(self, opportunity) -> Dict[str, Any]:
        """Create rollback plan for cleaning up security group"""
        sg_config = opportunity.resource_metadata
        return {
            "rollback_steps": [
                {
                    "step": 1,
                    "action": "create_security_group",
                    "parameters": {
                        "group_name": sg_config.get("group_name", "restored-sg"),
                        "description": sg_config.get("description", "Restored security group"),
                        "vpc_id": sg_config.get("vpc_id")
                    },
                    "description": "Recreate security group"
                },
                {
                    "step": 2,
                    "action": "restore_security_group_rules",
                    "parameters": {
                        "ingress_rules": sg_config.get("ingress_rules", []),
                        "egress_rules": sg_config.get("egress_rules", [])
                    },
                    "description": "Restore security group rules"
                }
            ],
            "prerequisites": [
                "Security group rules documented",
                "VPC still exists",
                "Referenced security groups exist"
            ],
            "estimated_rollback_cost": 0.0,  # No direct cost for security groups
            "rollback_time_estimate_minutes": 5
        }
    
    def execute_rollback(self, action: OptimizationAction) -> bool:
        """
        Execute rollback for a failed or problematic action.
        
        Args:
            action: The optimization action to rollback
            
        Returns:
            True if rollback was successful, False otherwise
        """
        logger.info("Executing rollback",
                   action_id=str(action.id),
                   resource_id=action.resource_id,
                   action_type=action.action_type.value)
        
        rollback_plan = action.rollback_plan
        
        if not rollback_plan or not rollback_plan.get("rollback_steps"):
            logger.error("No rollback plan available",
                        action_id=str(action.id))
            return False
        
        # Log rollback start
        self.audit_logger.log_action_event(
            action.id,
            "rollback_started",
            {
                "rollback_plan": rollback_plan,
                "reason": "Action monitoring detected issues"
            }
        )
        
        try:
            # Execute rollback steps in order
            for step in rollback_plan["rollback_steps"]:
                step_success = self._execute_rollback_step(action, step)
                
                if not step_success:
                    logger.error("Rollback step failed",
                               action_id=str(action.id),
                               step=step["step"],
                               step_action=step["action"])
                    
                    # Log failed rollback
                    self.audit_logger.log_action_event(
                        action.id,
                        "rollback_failed",
                        {
                            "failed_step": step,
                            "step_number": step["step"]
                        }
                    )
                    return False
                
                logger.info("Rollback step completed",
                           action_id=str(action.id),
                           step=step["step"],
                           step_action=step["action"])
            
            # Log successful rollback
            self.audit_logger.log_action_event(
                action.id,
                "rollback_completed",
                {
                    "rollback_plan": rollback_plan,
                    "steps_executed": len(rollback_plan["rollback_steps"])
                }
            )
            
            logger.info("Rollback completed successfully",
                       action_id=str(action.id))
            
            return True
            
        except Exception as e:
            logger.error("Exception during rollback execution",
                        action_id=str(action.id),
                        error=str(e))
            
            # Log rollback exception
            self.audit_logger.log_action_event(
                action.id,
                "rollback_exception",
                {"exception": str(e)}
            )
            
            return False
    
    def _execute_rollback_step(self, action: OptimizationAction, step: Dict[str, Any]) -> bool:
        """Execute a single rollback step"""
        
        step_action = step["action"]
        parameters = step["parameters"]
        
        try:
            # Route to appropriate rollback action handler
            if step_action == "start_instance":
                return self._rollback_start_instance(parameters)
            elif step_action == "launch_replacement_instance":
                return self._rollback_launch_replacement_instance(parameters)
            elif step_action == "modify_instance_type":
                return self._rollback_modify_instance_type(parameters)
            elif step_action == "create_volume_from_snapshot":
                return self._rollback_create_volume_from_snapshot(parameters)
            elif step_action == "modify_volume_type":
                return self._rollback_modify_volume_type(parameters)
            elif step_action == "allocate_elastic_ip":
                return self._rollback_allocate_elastic_ip(parameters)
            elif step_action == "create_load_balancer":
                return self._rollback_create_load_balancer(parameters)
            elif step_action == "create_security_group":
                return self._rollback_create_security_group(parameters)
            else:
                logger.warning("Unknown rollback action", action=step_action)
                return False
                
        except Exception as e:
            logger.error("Exception in rollback step",
                        step_action=step_action,
                        error=str(e))
            return False
    
    def _rollback_start_instance(self, parameters: Dict[str, Any]) -> bool:
        """Rollback step: Start an instance"""
        instance_id = parameters["instance_id"]
        logger.info("Rollback: Starting instance", instance_id=instance_id)
        # Simulate AWS API call
        return True
    
    def _rollback_launch_replacement_instance(self, parameters: Dict[str, Any]) -> bool:
        """Rollback step: Launch replacement instance"""
        logger.info("Rollback: Launching replacement instance", parameters=parameters)
        # Simulate AWS API call
        return True
    
    def _rollback_modify_instance_type(self, parameters: Dict[str, Any]) -> bool:
        """Rollback step: Modify instance type"""
        instance_id = parameters["instance_id"]
        instance_type = parameters["instance_type"]
        logger.info("Rollback: Modifying instance type",
                   instance_id=instance_id,
                   instance_type=instance_type)
        # Simulate AWS API call
        return True
    
    def _rollback_create_volume_from_snapshot(self, parameters: Dict[str, Any]) -> bool:
        """Rollback step: Create volume from snapshot"""
        logger.info("Rollback: Creating volume from snapshot", parameters=parameters)
        # Simulate AWS API call
        return True
    
    def _rollback_modify_volume_type(self, parameters: Dict[str, Any]) -> bool:
        """Rollback step: Modify volume type"""
        volume_id = parameters["volume_id"]
        volume_type = parameters["volume_type"]
        logger.info("Rollback: Modifying volume type",
                   volume_id=volume_id,
                   volume_type=volume_type)
        # Simulate AWS API call
        return True
    
    def _rollback_allocate_elastic_ip(self, parameters: Dict[str, Any]) -> bool:
        """Rollback step: Allocate Elastic IP"""
        logger.info("Rollback: Allocating Elastic IP", parameters=parameters)
        # Simulate AWS API call
        return True
    
    def _rollback_create_load_balancer(self, parameters: Dict[str, Any]) -> bool:
        """Rollback step: Create load balancer"""
        logger.info("Rollback: Creating load balancer", parameters=parameters)
        # Simulate AWS API call
        return True
    
    def _rollback_create_security_group(self, parameters: Dict[str, Any]) -> bool:
        """Rollback step: Create security group"""
        logger.info("Rollback: Creating security group", parameters=parameters)
        # Simulate AWS API call
        return True
    
    def monitor_post_action_health(self, action: OptimizationAction) -> Dict[str, Any]:
        """
        Monitor system health after an action is executed.
        
        Args:
            action: The executed optimization action to monitor
            
        Returns:
            Dictionary with monitoring results
        """
        logger.info("Monitoring post-action health",
                   action_id=str(action.id),
                   resource_id=action.resource_id)
        
        monitoring_results = {
            "action_id": str(action.id),
            "resource_id": action.resource_id,
            "monitoring_started_at": datetime.utcnow().isoformat(),
            "issues_detected": False,
            "health_checks": {},
            "warnings": [],
            "errors": []
        }
        
        # Perform action-specific health checks
        if action.action_type == ActionType.STOP_INSTANCE:
            monitoring_results.update(self._monitor_stopped_instance_health(action))
        elif action.action_type == ActionType.RESIZE_INSTANCE:
            monitoring_results.update(self._monitor_resized_instance_health(action))
        elif action.action_type == ActionType.DELETE_VOLUME:
            monitoring_results.update(self._monitor_volume_deletion_health(action))
        elif action.action_type == ActionType.UPGRADE_STORAGE:
            monitoring_results.update(self._monitor_storage_upgrade_health(action))
        
        # Log monitoring results
        self.audit_logger.log_action_event(
            action.id,
            "health_monitoring_completed",
            monitoring_results
        )
        
        logger.info("Post-action health monitoring completed",
                   action_id=str(action.id),
                   issues_detected=monitoring_results["issues_detected"])
        
        return monitoring_results
    
    def _monitor_stopped_instance_health(self, action: OptimizationAction) -> Dict[str, Any]:
        """Monitor health after stopping an instance"""
        
        # In a real implementation, this would check:
        # - Application health checks
        # - Load balancer target health
        # - Dependent service status
        
        # Simulate health checks
        health_checks = {
            "instance_state": {"status": "healthy", "details": "Instance stopped successfully"},
            "dependent_services": {"status": "healthy", "details": "No dependent services affected"},
            "application_health": {"status": "healthy", "details": "Application not running (expected)"}
        }
        
        return {
            "health_checks": health_checks,
            "issues_detected": False
        }
    
    def _monitor_resized_instance_health(self, action: OptimizationAction) -> Dict[str, Any]:
        """Monitor health after resizing an instance"""
        
        # Simulate health checks for resized instance
        health_checks = {
            "instance_state": {"status": "healthy", "details": "Instance running with new type"},
            "performance_metrics": {"status": "healthy", "details": "CPU and memory within normal ranges"},
            "application_health": {"status": "healthy", "details": "Application responding normally"}
        }
        
        return {
            "health_checks": health_checks,
            "issues_detected": False
        }
    
    def _monitor_volume_deletion_health(self, action: OptimizationAction) -> Dict[str, Any]:
        """Monitor health after deleting a volume"""
        
        # Simulate health checks for volume deletion
        health_checks = {
            "snapshot_status": {"status": "healthy", "details": "Snapshot created successfully"},
            "volume_deletion": {"status": "healthy", "details": "Volume deleted successfully"},
            "dependent_instances": {"status": "healthy", "details": "No instances affected"}
        }
        
        return {
            "health_checks": health_checks,
            "issues_detected": False
        }
    
    def _monitor_storage_upgrade_health(self, action: OptimizationAction) -> Dict[str, Any]:
        """Monitor health after upgrading storage"""
        
        # Simulate health checks for storage upgrade
        health_checks = {
            "volume_modification": {"status": "healthy", "details": "Volume type upgraded successfully"},
            "performance_impact": {"status": "healthy", "details": "No performance degradation detected"},
            "data_integrity": {"status": "healthy", "details": "Data integrity verified"}
        }
        
        return {
            "health_checks": health_checks,
            "issues_detected": False
        }
    
    def calculate_rollback_cost(self, action: OptimizationAction) -> Decimal:
        """
        Calculate the estimated cost of rolling back an action.
        
        Args:
            action: The optimization action to calculate rollback cost for
            
        Returns:
            Estimated rollback cost
        """
        rollback_plan = action.rollback_plan
        
        if not rollback_plan:
            return Decimal('0.0')
        
        estimated_cost = rollback_plan.get("estimated_rollback_cost", 0.0)
        
        logger.info("Calculated rollback cost",
                   action_id=str(action.id),
                   estimated_cost=estimated_cost)
        
        return Decimal(str(estimated_cost))
    
    def start_health_monitoring(self):
        """Start continuous health monitoring for executed actions"""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Health monitoring started")
        
        # Log monitoring start
        self.audit_logger.log_system_event(
            "health_monitoring_started",
            {
                "monitoring_interval": self.health_check_interval,
                "enabled_triggers": [trigger.value for trigger, enabled in self.rollback_triggers.items() if enabled]
            }
        )
    
    def stop_health_monitoring(self):
        """Stop continuous health monitoring"""
        if not self.monitoring_active:
            logger.warning("Health monitoring not active")
            return
        
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Health monitoring stopped")
        
        # Log monitoring stop
        self.audit_logger.log_system_event(
            "health_monitoring_stopped",
            {"monitored_actions_count": len(self.monitored_actions)}
        )
    
    def add_action_to_monitoring(self, action: OptimizationAction):
        """Add an action to continuous health monitoring"""
        action_key = str(action.id)
        self.monitored_actions[action_key] = action
        
        logger.info("Action added to health monitoring",
                   action_id=action_key,
                   resource_id=action.resource_id)
        
        # Log action monitoring start
        self.audit_logger.log_action_event(
            action.id,
            "monitoring_started",
            {
                "monitoring_type": "continuous_health_monitoring",
                "monitoring_interval": self.health_check_interval
            }
        )
    
    def remove_action_from_monitoring(self, action_id: uuid.UUID):
        """Remove an action from continuous health monitoring"""
        action_key = str(action_id)
        
        if action_key in self.monitored_actions:
            del self.monitored_actions[action_key]
            
            logger.info("Action removed from health monitoring",
                       action_id=action_key)
            
            # Log action monitoring stop
            self.audit_logger.log_action_event(
                action_id,
                "monitoring_stopped",
                {"reason": "manual_removal"}
            )
    
    def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        logger.info("Health monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Check health of all monitored actions
                for action_key, action in list(self.monitored_actions.items()):
                    if not self.monitoring_active:
                        break
                    
                    try:
                        self._check_action_health(action)
                    except Exception as e:
                        logger.error("Error checking action health",
                                   action_id=action_key,
                                   error=str(e))
                
                # Sleep for the monitoring interval
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
                time.sleep(self.health_check_interval)
        
        logger.info("Health monitoring loop stopped")
    
    def _check_action_health(self, action: OptimizationAction):
        """Check health of a specific action and trigger rollback if needed"""
        
        # Perform comprehensive health check
        health_result = self._perform_comprehensive_health_check(action)
        
        # Log health check result
        self.audit_logger.log_action_event(
            action.id,
            "health_check_completed",
            {
                "health_result": health_result,
                "check_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Check if rollback should be triggered
        rollback_trigger = self._evaluate_rollback_triggers(action, health_result)
        
        if rollback_trigger:
            logger.warning("Automatic rollback triggered",
                          action_id=str(action.id),
                          trigger=rollback_trigger.value)
            
            # Execute automatic rollback
            self._trigger_automatic_rollback(action, rollback_trigger, health_result)
    
    def _perform_comprehensive_health_check(self, action: OptimizationAction) -> Dict[str, Any]:
        """Perform comprehensive health check for an action"""
        
        health_result = {
            "action_id": str(action.id),
            "resource_id": action.resource_id,
            "check_timestamp": datetime.utcnow().isoformat(),
            "overall_status": HealthCheckResult.HEALTHY.value,
            "checks": {},
            "metrics": {},
            "issues": []
        }
        
        try:
            # Resource-specific health checks
            if action.action_type == ActionType.STOP_INSTANCE:
                health_result.update(self._check_stopped_instance_health(action))
            elif action.action_type == ActionType.RESIZE_INSTANCE:
                health_result.update(self._check_resized_instance_health(action))
            elif action.action_type == ActionType.DELETE_VOLUME:
                health_result.update(self._check_volume_deletion_health(action))
            elif action.action_type == ActionType.UPGRADE_STORAGE:
                health_result.update(self._check_storage_upgrade_health(action))
            
            # Common health checks
            health_result["checks"]["dependency_check"] = self._check_dependencies(action)
            health_result["checks"]["performance_check"] = self._check_performance_impact(action)
            health_result["checks"]["error_rate_check"] = self._check_error_rates(action)
            
            # Determine overall status
            health_result["overall_status"] = self._determine_overall_health_status(health_result["checks"])
            
        except Exception as e:
            logger.error("Error performing health check",
                        action_id=str(action.id),
                        error=str(e))
            
            health_result["overall_status"] = HealthCheckResult.UNKNOWN.value
            health_result["issues"].append(f"Health check error: {str(e)}")
        
        return health_result
    
    def _check_dependencies(self, action: OptimizationAction) -> Dict[str, Any]:
        """Check if action has affected dependent resources"""
        
        # Simulate dependency checking
        dependency_check = {
            "status": HealthCheckResult.HEALTHY.value,
            "dependent_resources": [],
            "issues": []
        }
        
        # In a real implementation, this would check:
        # - Load balancer targets
        # - Auto Scaling Group health
        # - Database connections
        # - Application dependencies
        
        return dependency_check
    
    def _check_performance_impact(self, action: OptimizationAction) -> Dict[str, Any]:
        """Check performance impact of the action"""
        
        # Simulate performance checking
        performance_check = {
            "status": HealthCheckResult.HEALTHY.value,
            "metrics": {
                "response_time_ms": 150,
                "cpu_utilization": 45,
                "memory_utilization": 60,
                "error_rate": 0.01
            },
            "issues": []
        }
        
        # Check against thresholds
        if performance_check["metrics"]["response_time_ms"] > self.health_thresholds["response_time_threshold"]:
            performance_check["status"] = HealthCheckResult.WARNING.value
            performance_check["issues"].append("Response time above threshold")
        
        if performance_check["metrics"]["cpu_utilization"] > self.health_thresholds["cpu_threshold"]:
            performance_check["status"] = HealthCheckResult.CRITICAL.value
            performance_check["issues"].append("CPU utilization above threshold")
        
        return performance_check
    
    def _check_error_rates(self, action: OptimizationAction) -> Dict[str, Any]:
        """Check error rates after action execution"""
        
        # Simulate error rate checking
        error_check = {
            "status": HealthCheckResult.HEALTHY.value,
            "error_rate": 0.02,
            "error_count": 5,
            "total_requests": 250,
            "issues": []
        }
        
        if error_check["error_rate"] > self.health_thresholds["error_rate_threshold"]:
            error_check["status"] = HealthCheckResult.WARNING.value
            error_check["issues"].append("Error rate above threshold")
        
        return error_check
    
    def _determine_overall_health_status(self, checks: Dict[str, Any]) -> str:
        """Determine overall health status from individual checks"""
        
        statuses = [check.get("status", HealthCheckResult.UNKNOWN.value) for check in checks.values()]
        
        if HealthCheckResult.CRITICAL.value in statuses:
            return HealthCheckResult.CRITICAL.value
        elif HealthCheckResult.WARNING.value in statuses:
            return HealthCheckResult.WARNING.value
        elif HealthCheckResult.UNKNOWN.value in statuses:
            return HealthCheckResult.UNKNOWN.value
        else:
            return HealthCheckResult.HEALTHY.value
    
    def _evaluate_rollback_triggers(self, action: OptimizationAction, health_result: Dict[str, Any]) -> Optional[RollbackTrigger]:
        """Evaluate if any rollback triggers should be activated"""
        
        overall_status = health_result.get("overall_status")
        
        # Check for critical health status
        if (overall_status == HealthCheckResult.CRITICAL.value and 
            self.rollback_triggers.get(RollbackTrigger.HEALTH_CHECK_FAILURE, False)):
            return RollbackTrigger.HEALTH_CHECK_FAILURE
        
        # Check for performance degradation
        performance_check = health_result.get("checks", {}).get("performance_check", {})
        if (performance_check.get("status") == HealthCheckResult.CRITICAL.value and
            self.rollback_triggers.get(RollbackTrigger.PERFORMANCE_DEGRADATION, False)):
            return RollbackTrigger.PERFORMANCE_DEGRADATION
        
        # Check for error threshold exceeded
        error_check = health_result.get("checks", {}).get("error_rate_check", {})
        if (error_check.get("status") == HealthCheckResult.WARNING.value and
            self.rollback_triggers.get(RollbackTrigger.ERROR_THRESHOLD_EXCEEDED, False)):
            return RollbackTrigger.ERROR_THRESHOLD_EXCEEDED
        
        # Check for dependency failures
        dependency_check = health_result.get("checks", {}).get("dependency_check", {})
        if (dependency_check.get("status") == HealthCheckResult.CRITICAL.value and
            self.rollback_triggers.get(RollbackTrigger.DEPENDENCY_FAILURE, False)):
            return RollbackTrigger.DEPENDENCY_FAILURE
        
        return None
    
    def _trigger_automatic_rollback(self, action: OptimizationAction, trigger: RollbackTrigger, health_result: Dict[str, Any]):
        """Trigger automatic rollback for an action"""
        
        logger.warning("Triggering automatic rollback",
                      action_id=str(action.id),
                      trigger=trigger.value)
        
        # Log rollback trigger
        self.audit_logger.log_rollback_event(
            action.id,
            "automatic_trigger",
            {
                "trigger_type": trigger.value,
                "health_result": health_result,
                "trigger_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Execute rollback with retry logic
        rollback_success = self.execute_rollback_with_retry(action, trigger)
        
        if rollback_success:
            # Remove from monitoring after successful rollback
            self.remove_action_from_monitoring(action.id)
        else:
            # Log failed automatic rollback
            self.audit_logger.log_rollback_event(
                action.id,
                "automatic_rollback_failed",
                {
                    "trigger_type": trigger.value,
                    "retry_attempts": self.max_retry_attempts
                }
            )
    
    def execute_manual_rollback(self, action: OptimizationAction, reason: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute manual rollback with impact assessment.
        
        Args:
            action: The optimization action to rollback
            reason: Reason for manual rollback
            user_context: User context information
            
        Returns:
            Dictionary with rollback results and impact assessment
        """
        
        logger.info("Executing manual rollback",
                   action_id=str(action.id),
                   reason=reason)
        
        # Perform impact assessment before rollback
        impact_assessment = self.assess_rollback_impact(action)
        
        # Log manual rollback initiation
        self.audit_logger.log_rollback_event(
            action.id,
            "manual_rollback_initiated",
            {
                "reason": reason,
                "user_context": user_context or {},
                "impact_assessment": impact_assessment,
                "initiation_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Execute rollback with retry logic
        rollback_success = self.execute_rollback_with_retry(action, RollbackTrigger.MANUAL)
        
        rollback_result = {
            "action_id": str(action.id),
            "rollback_success": rollback_success,
            "rollback_trigger": RollbackTrigger.MANUAL.value,
            "reason": reason,
            "impact_assessment": impact_assessment,
            "execution_timestamp": datetime.utcnow().isoformat()
        }
        
        # Log manual rollback completion
        self.audit_logger.log_rollback_event(
            action.id,
            "manual_rollback_completed",
            rollback_result
        )
        
        return rollback_result
    
    def assess_rollback_impact(self, action: OptimizationAction) -> Dict[str, Any]:
        """
        Assess the impact of rolling back an action.
        
        Args:
            action: The optimization action to assess
            
        Returns:
            Dictionary with impact assessment details
        """
        
        logger.info("Assessing rollback impact",
                   action_id=str(action.id))
        
        impact_assessment = {
            "action_id": str(action.id),
            "resource_id": action.resource_id,
            "action_type": action.action_type.value,
            "assessment_timestamp": datetime.utcnow().isoformat(),
            "financial_impact": {},
            "operational_impact": {},
            "risk_assessment": {},
            "recommendations": []
        }
        
        # Calculate financial impact
        rollback_cost = self.calculate_rollback_cost(action)
        lost_savings = action.actual_savings or action.estimated_monthly_savings
        
        impact_assessment["financial_impact"] = {
            "rollback_cost": float(rollback_cost),
            "lost_monthly_savings": float(lost_savings),
            "net_impact": float(rollback_cost + lost_savings),
            "currency": "USD"
        }
        
        # Assess operational impact
        impact_assessment["operational_impact"] = self._assess_operational_impact(action)
        
        # Assess risks
        impact_assessment["risk_assessment"] = self._assess_rollback_risks(action)
        
        # Generate recommendations
        impact_assessment["recommendations"] = self._generate_rollback_recommendations(action, impact_assessment)
        
        logger.info("Rollback impact assessment completed",
                   action_id=str(action.id),
                   net_financial_impact=impact_assessment["financial_impact"]["net_impact"])
        
        return impact_assessment
    
    def _assess_operational_impact(self, action: OptimizationAction) -> Dict[str, Any]:
        """Assess operational impact of rollback"""
        
        operational_impact = {
            "downtime_required": False,
            "estimated_downtime_minutes": 0,
            "affected_services": [],
            "user_impact": "none",
            "data_loss_risk": "none"
        }
        
        # Action-specific operational impact
        if action.action_type == ActionType.STOP_INSTANCE:
            operational_impact.update({
                "downtime_required": False,
                "estimated_downtime_minutes": 3,
                "user_impact": "minimal",
                "affected_services": ["compute"]
            })
        elif action.action_type == ActionType.TERMINATE_INSTANCE:
            operational_impact.update({
                "downtime_required": True,
                "estimated_downtime_minutes": 30,
                "user_impact": "high",
                "data_loss_risk": "medium",
                "affected_services": ["compute", "storage"]
            })
        elif action.action_type == ActionType.DELETE_VOLUME:
            operational_impact.update({
                "downtime_required": True,
                "estimated_downtime_minutes": 15,
                "user_impact": "medium",
                "data_loss_risk": "low",  # Assuming snapshot exists
                "affected_services": ["storage"]
            })
        
        return operational_impact
    
    def _assess_rollback_risks(self, action: OptimizationAction) -> Dict[str, Any]:
        """Assess risks associated with rollback"""
        
        risk_assessment = {
            "overall_risk": "low",
            "technical_risks": [],
            "business_risks": [],
            "mitigation_strategies": []
        }
        
        # Action-specific risk assessment
        if action.action_type == ActionType.TERMINATE_INSTANCE:
            risk_assessment.update({
                "overall_risk": "high",
                "technical_risks": [
                    "Data loss if backup is incomplete",
                    "Configuration drift from original instance",
                    "Network connectivity issues"
                ],
                "business_risks": [
                    "Service disruption during recovery",
                    "Potential data inconsistency"
                ],
                "mitigation_strategies": [
                    "Verify backup integrity before rollback",
                    "Test network connectivity",
                    "Implement gradual traffic restoration"
                ]
            })
        elif action.action_type == ActionType.DELETE_VOLUME:
            risk_assessment.update({
                "overall_risk": "medium",
                "technical_risks": [
                    "Snapshot restoration time",
                    "Potential data loss since snapshot"
                ],
                "business_risks": [
                    "Temporary service unavailability"
                ],
                "mitigation_strategies": [
                    "Verify snapshot completeness",
                    "Schedule rollback during low-usage period"
                ]
            })
        
        return risk_assessment
    
    def _generate_rollback_recommendations(self, action: OptimizationAction, impact_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on impact assessment"""
        
        recommendations = []
        
        financial_impact = impact_assessment["financial_impact"]["net_impact"]
        operational_impact = impact_assessment["operational_impact"]
        risk_level = impact_assessment["risk_assessment"]["overall_risk"]
        
        # Financial recommendations
        if financial_impact > 100:
            recommendations.append("Consider if rollback cost justifies the action reversal")
        
        # Operational recommendations
        if operational_impact["downtime_required"]:
            recommendations.append("Schedule rollback during maintenance window to minimize user impact")
        
        if operational_impact["data_loss_risk"] != "none":
            recommendations.append("Verify all backups and snapshots before proceeding with rollback")
        
        # Risk-based recommendations
        if risk_level == "high":
            recommendations.append("Consider alternative solutions before proceeding with high-risk rollback")
            recommendations.append("Ensure incident response team is available during rollback")
        
        # Action-specific recommendations
        if action.action_type == ActionType.TERMINATE_INSTANCE:
            recommendations.append("Test instance launch and configuration in staging environment first")
        
        return recommendations
    
    def execute_rollback_with_retry(self, action: OptimizationAction, trigger: RollbackTrigger) -> bool:
        """
        Execute rollback with retry logic and success validation.
        
        Args:
            action: The optimization action to rollback
            trigger: The trigger that initiated the rollback
            
        Returns:
            True if rollback was successful, False otherwise
        """
        
        logger.info("Executing rollback with retry logic",
                   action_id=str(action.id),
                   trigger=trigger.value,
                   max_attempts=self.max_retry_attempts)
        
        for attempt in range(1, self.max_retry_attempts + 1):
            logger.info("Rollback attempt",
                       action_id=str(action.id),
                       attempt=attempt,
                       max_attempts=self.max_retry_attempts)
            
            # Log retry attempt
            self.audit_logger.log_rollback_event(
                action.id,
                "rollback_attempt",
                {
                    "attempt_number": attempt,
                    "max_attempts": self.max_retry_attempts,
                    "trigger": trigger.value
                }
            )
            
            # Execute rollback
            rollback_success = self.execute_rollback(action)
            
            if rollback_success:
                # Validate rollback success
                validation_success = self.validate_rollback_success(action)
                
                if validation_success:
                    logger.info("Rollback completed successfully",
                               action_id=str(action.id),
                               attempt=attempt)
                    
                    # Log successful rollback
                    self.audit_logger.log_rollback_event(
                        action.id,
                        "rollback_success_validated",
                        {
                            "attempt_number": attempt,
                            "validation_passed": True,
                            "trigger": trigger.value
                        }
                    )
                    
                    return True
                else:
                    logger.warning("Rollback validation failed",
                                 action_id=str(action.id),
                                 attempt=attempt)
                    
                    # Log validation failure
                    self.audit_logger.log_rollback_event(
                        action.id,
                        "rollback_validation_failed",
                        {
                            "attempt_number": attempt,
                            "validation_passed": False
                        }
                    )
            else:
                logger.warning("Rollback execution failed",
                             action_id=str(action.id),
                             attempt=attempt)
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retry_attempts:
                logger.info("Waiting before retry",
                           action_id=str(action.id),
                           retry_delay=self.retry_delay)
                time.sleep(self.retry_delay)
        
        # All attempts failed
        logger.error("Rollback failed after all retry attempts",
                    action_id=str(action.id),
                    attempts=self.max_retry_attempts)
        
        # Log final failure
        self.audit_logger.log_rollback_event(
            action.id,
            "rollback_failed_final",
            {
                "total_attempts": self.max_retry_attempts,
                "trigger": trigger.value,
                "final_status": "failed"
            }
        )
        
        return False
    
    def validate_rollback_success(self, action: OptimizationAction) -> bool:
        """
        Validate that rollback was successful.
        
        Args:
            action: The optimization action that was rolled back
            
        Returns:
            True if rollback validation passed, False otherwise
        """
        
        logger.info("Validating rollback success",
                   action_id=str(action.id))
        
        validation_results = {
            "action_id": str(action.id),
            "validation_timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "overall_success": False
        }
        
        try:
            # Perform action-specific validation
            if action.action_type == ActionType.STOP_INSTANCE:
                validation_results["checks"]["instance_running"] = self._validate_instance_started(action)
            elif action.action_type == ActionType.RESIZE_INSTANCE:
                validation_results["checks"]["instance_type_restored"] = self._validate_instance_type_restored(action)
            elif action.action_type == ActionType.DELETE_VOLUME:
                validation_results["checks"]["volume_restored"] = self._validate_volume_restored(action)
            elif action.action_type == ActionType.UPGRADE_STORAGE:
                validation_results["checks"]["storage_type_reverted"] = self._validate_storage_type_reverted(action)
            
            # Common validation checks
            validation_results["checks"]["resource_accessible"] = self._validate_resource_accessible(action)
            validation_results["checks"]["health_check_passed"] = self._validate_post_rollback_health(action)
            
            # Determine overall success
            validation_results["overall_success"] = all(validation_results["checks"].values())
            
            logger.info("Rollback validation completed",
                       action_id=str(action.id),
                       success=validation_results["overall_success"])
            
            # Log validation results
            self.audit_logger.log_rollback_event(
                action.id,
                "rollback_validation_completed",
                validation_results
            )
            
            return validation_results["overall_success"]
            
        except Exception as e:
            logger.error("Error during rollback validation",
                        action_id=str(action.id),
                        error=str(e))
            
            # Log validation error
            self.audit_logger.log_rollback_event(
                action.id,
                "rollback_validation_error",
                {"error": str(e)}
            )
            
            return False
    
    def _validate_instance_started(self, action: OptimizationAction) -> bool:
        """Validate that instance was started successfully"""
        # Simulate AWS API call to check instance state
        logger.info("Validating instance started", instance_id=action.resource_id)
        return True  # Simulate success
    
    def _validate_instance_type_restored(self, action: OptimizationAction) -> bool:
        """Validate that instance type was restored"""
        # Simulate AWS API call to check instance type
        logger.info("Validating instance type restored", instance_id=action.resource_id)
        return True  # Simulate success
    
    def _validate_volume_restored(self, action: OptimizationAction) -> bool:
        """Validate that volume was restored from snapshot"""
        # Simulate AWS API call to check volume state
        logger.info("Validating volume restored", volume_id=action.resource_id)
        return True  # Simulate success
    
    def _validate_storage_type_reverted(self, action: OptimizationAction) -> bool:
        """Validate that storage type was reverted"""
        # Simulate AWS API call to check volume type
        logger.info("Validating storage type reverted", volume_id=action.resource_id)
        return True  # Simulate success
    
    def _validate_resource_accessible(self, action: OptimizationAction) -> bool:
        """Validate that resource is accessible after rollback"""
        # Simulate connectivity/accessibility check
        logger.info("Validating resource accessible", resource_id=action.resource_id)
        return True  # Simulate success
    
    def _validate_post_rollback_health(self, action: OptimizationAction) -> bool:
        """Validate system health after rollback"""
        # Perform quick health check
        health_result = self._perform_comprehensive_health_check(action)
        overall_status = health_result.get("overall_status")
        
        return overall_status in [HealthCheckResult.HEALTHY.value, HealthCheckResult.WARNING.value]
    
    def _check_stopped_instance_health(self, action: OptimizationAction) -> Dict[str, Any]:
        """Check health after stopping an instance"""
        
        # Simulate health checks for stopped instance
        health_checks = {
            "instance_state": {"status": "healthy", "details": "Instance stopped successfully"},
            "dependent_services": {"status": "healthy", "details": "No dependent services affected"},
            "application_health": {"status": "healthy", "details": "Application not running (expected)"}
        }
        
        return {
            "checks": health_checks,
            "issues_detected": False
        }
    
    def _check_resized_instance_health(self, action: OptimizationAction) -> Dict[str, Any]:
        """Check health after resizing an instance"""
        
        # Simulate health checks for resized instance
        health_checks = {
            "instance_state": {"status": "healthy", "details": "Instance running with new type"},
            "performance_metrics": {"status": "healthy", "details": "CPU and memory within normal ranges"},
            "application_health": {"status": "healthy", "details": "Application responding normally"}
        }
        
        return {
            "checks": health_checks,
            "issues_detected": False
        }
    
    def _check_volume_deletion_health(self, action: OptimizationAction) -> Dict[str, Any]:
        """Check health after deleting a volume"""
        
        # Simulate health checks for volume deletion
        health_checks = {
            "snapshot_status": {"status": "healthy", "details": "Snapshot created successfully"},
            "volume_deletion": {"status": "healthy", "details": "Volume deleted successfully"},
            "dependent_instances": {"status": "healthy", "details": "No instances affected"}
        }
        
        return {
            "checks": health_checks,
            "issues_detected": False
        }
    
    def _check_storage_upgrade_health(self, action: OptimizationAction) -> Dict[str, Any]:
        """Check health after upgrading storage"""
        
        # Simulate health checks for storage upgrade
        health_checks = {
            "volume_modification": {"status": "healthy", "details": "Volume type upgraded successfully"},
            "performance_impact": {"status": "healthy", "details": "No performance degradation detected"},
            "data_integrity": {"status": "healthy", "details": "Data integrity verified"}
        }
        
        return {
            "checks": health_checks,
            "issues_detected": False
        }
    
    def get_rollback_status(self, action_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get current rollback status for an action.
        
        Args:
            action_id: ID of the optimization action
            
        Returns:
            Dictionary with rollback status information
        """
        
        # Get audit trail for rollback events
        audit_trail = self.audit_logger.get_action_audit_trail(action_id)
        
        rollback_events = [
            event for event in audit_trail 
            if event["event_type"].startswith("rollback_")
        ]
        
        if not rollback_events:
            return {
                "action_id": str(action_id),
                "rollback_status": RollbackStatus.NOT_STARTED.value,
                "rollback_events": []
            }
        
        # Determine current status from latest events
        latest_event = rollback_events[-1]
        
        if latest_event["event_type"] == "rollback_success_validated":
            status = RollbackStatus.COMPLETED.value
        elif latest_event["event_type"] == "rollback_failed_final":
            status = RollbackStatus.FAILED.value
        elif latest_event["event_type"] in ["rollback_started", "rollback_attempt"]:
            status = RollbackStatus.IN_PROGRESS.value
        else:
            status = RollbackStatus.NOT_STARTED.value
        
        return {
            "action_id": str(action_id),
            "rollback_status": status,
            "rollback_events": rollback_events,
            "latest_event": latest_event
        }
    
    def configure_rollback_triggers(self, trigger_config: Dict[RollbackTrigger, bool]):
        """
        Configure which rollback triggers are enabled.
        
        Args:
            trigger_config: Dictionary mapping triggers to enabled status
        """
        
        self.rollback_triggers.update(trigger_config)
        
        logger.info("Rollback triggers configured",
                   enabled_triggers=[trigger.value for trigger, enabled in self.rollback_triggers.items() if enabled])
        
        # Log configuration change
        self.audit_logger.log_system_event(
            "rollback_triggers_configured",
            {
                "trigger_configuration": {trigger.value: enabled for trigger, enabled in self.rollback_triggers.items()}
            }
        )
    
    def configure_health_thresholds(self, threshold_config: Dict[str, float]):
        """
        Configure health check thresholds.
        
        Args:
            threshold_config: Dictionary with threshold values
        """
        
        self.health_thresholds.update(threshold_config)
        
        logger.info("Health thresholds configured",
                   thresholds=self.health_thresholds)
        
        # Log configuration change
        self.audit_logger.log_system_event(
            "health_thresholds_configured",
            {"threshold_configuration": self.health_thresholds}
        )