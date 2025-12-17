"""
Safety Checker for Automated Cost Optimization

Validates all actions against safety rules and production protection policies:
- Production tag protection
- Business hours restrictions
- Resource dependency checks
- Risk assessment
"""

from datetime import datetime, time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import structlog

from .automation_models import (
    AutomationPolicy, ActionType, RiskLevel, SafetyCheckResult
)
from .database import get_db_session

logger = structlog.get_logger(__name__)


@dataclass
class SafetyRule:
    """Represents a safety rule configuration"""
    name: str
    enabled: bool
    parameters: Dict[str, Any]


class SafetyChecker:
    """
    Validates all optimization actions against safety rules and policies.
    
    Ensures that automated actions never impact production workloads or
    violate organizational safety requirements.
    """
    
    def __init__(self):
        self.default_safety_rules = self._get_default_safety_rules()
    
    def _get_default_safety_rules(self) -> List[SafetyRule]:
        """Get default safety rules for all actions"""
        return [
            SafetyRule(
                name="production_tag_protection",
                enabled=True,
                parameters={
                    "protected_tags": [
                        {"key": "Environment", "values": ["production", "prod", "live"]},
                        {"key": "Critical", "values": ["true", "yes", "1"]},
                        {"key": "Tier", "values": ["production", "prod", "critical", "live"]},
                        {"key": "Stage", "values": ["production", "prod", "live"]}
                    ]
                }
            ),
            SafetyRule(
                name="business_hours_protection",
                enabled=True,
                parameters={
                    "timezone": "UTC",
                    "business_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "business_start": "09:00",
                    "business_end": "17:00"
                }
            ),
            SafetyRule(
                name="auto_scaling_group_protection",
                enabled=True,
                parameters={
                    "check_asg_membership": True,
                    "min_instances_threshold": 1
                }
            ),
            SafetyRule(
                name="load_balancer_target_protection",
                enabled=True,
                parameters={
                    "check_lb_targets": True,
                    "allow_unhealthy_targets": False
                }
            ),
            SafetyRule(
                name="database_dependency_protection",
                enabled=True,
                parameters={
                    "check_db_connections": True,
                    "protected_db_types": ["rds", "aurora", "dynamodb"]
                }
            ),
            SafetyRule(
                name="recent_activity_protection",
                enabled=True,
                parameters={
                    "min_idle_hours": 24,
                    "check_cloudtrail": True
                }
            )
        ]
    
    def validate_action_safety(self, 
                             opportunity,  # OptimizationOpportunity
                             policy: AutomationPolicy) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that an optimization opportunity meets all safety requirements.
        
        Args:
            opportunity: The optimization opportunity to validate
            policy: Automation policy with safety overrides
            
        Returns:
            Tuple of (all_checks_passed, detailed_results)
        """
        logger.info("Starting safety validation",
                   resource_id=opportunity.resource_id,
                   action_type=opportunity.action_type.value)
        
        safety_results = {
            "overall_passed": True,
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        # Get safety rules (default + policy overrides)
        safety_rules = self._get_effective_safety_rules(policy)
        
        # Run each safety check
        for rule in safety_rules:
            if not rule.enabled:
                continue
                
            check_passed, check_details = self._run_safety_check(
                rule, opportunity, policy
            )
            
            safety_results["checks"][rule.name] = {
                "passed": check_passed,
                "details": check_details
            }
            
            if not check_passed:
                safety_results["overall_passed"] = False
                safety_results["errors"].append(f"{rule.name}: {check_details.get('reason', 'Check failed')}")
            
            logger.debug("Safety check completed",
                        rule_name=rule.name,
                        passed=check_passed,
                        resource_id=opportunity.resource_id)
        
        # Store safety check results in database
        self._store_safety_check_results(opportunity, safety_results)
        
        logger.info("Safety validation completed",
                   resource_id=opportunity.resource_id,
                   overall_passed=safety_results["overall_passed"])
        
        return safety_results["overall_passed"], safety_results
    
    def _get_effective_safety_rules(self, policy: AutomationPolicy) -> List[SafetyRule]:
        """Get effective safety rules combining defaults with policy overrides"""
        
        rules = self.default_safety_rules.copy()
        
        # Apply policy safety overrides
        safety_overrides = policy.safety_overrides
        
        for rule in rules:
            if rule.name in safety_overrides:
                override = safety_overrides[rule.name]
                
                if "enabled" in override:
                    rule.enabled = override["enabled"]
                
                if "parameters" in override:
                    rule.parameters.update(override["parameters"])
        
        return rules
    
    def _run_safety_check(self, 
                         rule: SafetyRule,
                         opportunity,  # OptimizationOpportunity
                         policy: AutomationPolicy) -> Tuple[bool, Dict[str, Any]]:
        """Run a specific safety check"""
        
        if rule.name == "production_tag_protection":
            return self._check_production_tags(rule, opportunity)
        elif rule.name == "business_hours_protection":
            return self._check_business_hours(rule, opportunity, policy)
        elif rule.name == "auto_scaling_group_protection":
            return self._check_auto_scaling_group(rule, opportunity)
        elif rule.name == "load_balancer_target_protection":
            return self._check_load_balancer_targets(rule, opportunity)
        elif rule.name == "database_dependency_protection":
            return self._check_database_dependencies(rule, opportunity)
        elif rule.name == "recent_activity_protection":
            return self._check_recent_activity(rule, opportunity)
        else:
            logger.warning("Unknown safety rule", rule_name=rule.name)
            return True, {"reason": "Unknown rule, defaulting to pass"}
    
    def _check_production_tags(self, 
                              rule: SafetyRule,
                              opportunity) -> Tuple[bool, Dict[str, Any]]:
        """Check if resource has production tags that should protect it"""
        
        resource_tags = opportunity.resource_metadata.get("tags", {})
        protected_tags = rule.parameters.get("protected_tags", [])
        
        for protected_tag in protected_tags:
            tag_key = protected_tag["key"]
            protected_values = [v.lower() for v in protected_tag["values"]]
            
            if tag_key in resource_tags:
                tag_value = str(resource_tags[tag_key]).lower()
                if tag_value in protected_values:
                    return False, {
                        "reason": f"Resource has protected tag {tag_key}={resource_tags[tag_key]}",
                        "protected_tag": tag_key,
                        "tag_value": resource_tags[tag_key]
                    }
        
        return True, {"reason": "No protected tags found"}
    
    def _check_business_hours(self, 
                             rule: SafetyRule,
                             opportunity,
                             policy: AutomationPolicy) -> Tuple[bool, Dict[str, Any]]:
        """Check if action should be deferred due to business hours"""
        
        # Get time restrictions from policy
        time_restrictions = policy.time_restrictions
        business_hours = time_restrictions.get("business_hours", {})
        
        if not business_hours:
            return True, {"reason": "No business hours restrictions configured"}
        
        # For high-risk actions, check if we're in business hours
        if opportunity.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
            now = datetime.utcnow()
            
            # Simple business hours check (would be more sophisticated in real implementation)
            current_hour = now.hour
            business_start = int(business_hours.get("start", "09:00").split(":")[0])
            business_end = int(business_hours.get("end", "17:00").split(":")[0])
            
            if business_start <= current_hour < business_end:
                # We're in business hours and this is a risky action
                return False, {
                    "reason": f"High-risk action during business hours ({current_hour}:00 UTC)",
                    "current_time": now.isoformat(),
                    "business_hours": f"{business_start}:00-{business_end}:00 UTC"
                }
        
        return True, {"reason": "Business hours check passed"}
    
    def _check_auto_scaling_group(self, 
                                 rule: SafetyRule,
                                 opportunity) -> Tuple[bool, Dict[str, Any]]:
        """Check if resource is part of an Auto Scaling Group"""
        
        # In a real implementation, this would check AWS APIs
        # For now, check metadata for ASG information
        resource_metadata = opportunity.resource_metadata
        
        if "auto_scaling_group" in resource_metadata:
            asg_name = resource_metadata["auto_scaling_group"]
            return False, {
                "reason": f"Resource is part of Auto Scaling Group: {asg_name}",
                "asg_name": asg_name
            }
        
        # Check for ASG-related tags
        tags = resource_metadata.get("tags", {})
        asg_tags = [tag for tag in tags.keys() if "autoscaling" in tag.lower() or "asg" in tag.lower()]
        
        if asg_tags:
            return False, {
                "reason": f"Resource has Auto Scaling Group tags: {asg_tags}",
                "asg_tags": asg_tags
            }
        
        return True, {"reason": "No Auto Scaling Group membership detected"}
    
    def _check_load_balancer_targets(self, 
                                   rule: SafetyRule,
                                   opportunity) -> Tuple[bool, Dict[str, Any]]:
        """Check if resource is a target of a load balancer"""
        
        # In a real implementation, this would check ELB/ALB target groups
        resource_metadata = opportunity.resource_metadata
        
        if "load_balancer_targets" in resource_metadata:
            lb_targets = resource_metadata["load_balancer_targets"]
            if lb_targets:
                return False, {
                    "reason": f"Resource is target of load balancers: {lb_targets}",
                    "load_balancers": lb_targets
                }
        
        return True, {"reason": "No load balancer target relationships found"}
    
    def _check_database_dependencies(self, 
                                   rule: SafetyRule,
                                   opportunity) -> Tuple[bool, Dict[str, Any]]:
        """Check if resource has database dependencies"""
        
        # In a real implementation, this would check for database connections,
        # security group rules, etc.
        resource_metadata = opportunity.resource_metadata
        
        if "database_connections" in resource_metadata:
            db_connections = resource_metadata["database_connections"]
            if db_connections:
                return False, {
                    "reason": f"Resource has database connections: {db_connections}",
                    "databases": db_connections
                }
        
        # Check for database-related security group rules
        security_groups = resource_metadata.get("security_groups", [])
        for sg in security_groups:
            if any(port in str(sg) for port in ["3306", "5432", "1433", "27017"]):
                return False, {
                    "reason": f"Resource has database-related security group rules",
                    "security_groups": security_groups
                }
        
        return True, {"reason": "No database dependencies detected"}
    
    def _check_recent_activity(self, 
                              rule: SafetyRule,
                              opportunity) -> Tuple[bool, Dict[str, Any]]:
        """Check if resource has had recent activity"""
        
        min_idle_hours = rule.parameters.get("min_idle_hours", 24)
        resource_metadata = opportunity.resource_metadata
        
        # Check last activity timestamp
        if "last_activity" in resource_metadata:
            last_activity = datetime.fromisoformat(resource_metadata["last_activity"])
            hours_since_activity = (datetime.utcnow() - last_activity).total_seconds() / 3600
            
            if hours_since_activity < min_idle_hours:
                return False, {
                    "reason": f"Resource had recent activity {hours_since_activity:.1f} hours ago (minimum: {min_idle_hours})",
                    "last_activity": last_activity.isoformat(),
                    "hours_since_activity": hours_since_activity
                }
        
        # For EC2 instances, check CPU utilization
        if opportunity.resource_type == "ec2_instance":
            cpu_utilization = resource_metadata.get("cpu_utilization_24h", 0)
            if cpu_utilization > 5.0:  # More than 5% average CPU
                return False, {
                    "reason": f"Instance has {cpu_utilization}% CPU utilization (threshold: 5%)",
                    "cpu_utilization": cpu_utilization
                }
        
        return True, {"reason": f"No recent activity detected (idle for sufficient time)"}
    
    def _store_safety_check_results(self, opportunity, safety_results: Dict[str, Any]):
        """Store safety check results in database for audit purposes"""
        
        # This would create SafetyCheckResult records
        # For now, we'll just log the results
        logger.info("Safety check results stored",
                   resource_id=opportunity.resource_id,
                   overall_passed=safety_results["overall_passed"],
                   check_count=len(safety_results["checks"]))
    
    def check_production_tags(self, resource_tags: Dict[str, str]) -> bool:
        """
        Public method to check if resource has production tags.
        
        Args:
            resource_tags: Dictionary of resource tags
            
        Returns:
            True if resource has production tags (should be protected)
        """
        production_indicators = [
            ("Environment", ["production", "prod", "live"]),
            ("Critical", ["true", "yes", "1"]),
            ("Tier", ["production", "prod", "critical", "live"]),
            ("Stage", ["production", "prod", "live"])
        ]
        
        for tag_key, protected_values in production_indicators:
            if tag_key in resource_tags:
                tag_value = str(resource_tags[tag_key]).lower()
                if tag_value in [v.lower() for v in protected_values]:
                    return True
        
        return False
    
    def verify_business_hours(self, 
                            business_hours_config: Dict[str, Any],
                            current_time: Optional[datetime] = None) -> bool:
        """
        Public method to verify if current time is within business hours.
        
        Args:
            business_hours_config: Business hours configuration
            current_time: Time to check (defaults to now)
            
        Returns:
            True if within business hours
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        if not business_hours_config:
            return False  # No business hours configured
        
        # Simple implementation - would be more sophisticated in real system
        current_hour = current_time.hour
        start_hour = int(business_hours_config.get("start", "09:00").split(":")[0])
        end_hour = int(business_hours_config.get("end", "17:00").split(":")[0])
        
        return start_hour <= current_hour < end_hour
    
    def validate_resource_dependencies(self, 
                                     resource_metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Public method to validate resource dependencies.
        
        Args:
            resource_metadata: Resource metadata to check
            
        Returns:
            Tuple of (is_safe, list_of_dependencies)
        """
        dependencies = []
        
        # Check for various dependency types
        if "auto_scaling_group" in resource_metadata:
            dependencies.append(f"Auto Scaling Group: {resource_metadata['auto_scaling_group']}")
        
        if "load_balancer_targets" in resource_metadata:
            lb_targets = resource_metadata["load_balancer_targets"]
            if lb_targets:
                dependencies.extend([f"Load Balancer: {lb}" for lb in lb_targets])
        
        if "database_connections" in resource_metadata:
            db_connections = resource_metadata["database_connections"]
            if db_connections:
                dependencies.extend([f"Database: {db}" for db in db_connections])
        
        # Resource is safe if it has no critical dependencies
        is_safe = len(dependencies) == 0
        
        return is_safe, dependencies
    
    def assess_action_risk(self, 
                          action_type: ActionType,
                          resource_metadata: Dict[str, Any]) -> RiskLevel:
        """
        Public method to assess the risk level of an action.
        
        Args:
            action_type: Type of action to perform
            resource_metadata: Metadata about the resource
            
        Returns:
            Risk level for the action
        """
        # Base risk levels by action type
        base_risk = {
            ActionType.RELEASE_ELASTIC_IP: RiskLevel.LOW,
            ActionType.UPGRADE_STORAGE: RiskLevel.LOW,
            ActionType.DELETE_VOLUME: RiskLevel.MEDIUM,
            ActionType.STOP_INSTANCE: RiskLevel.MEDIUM,
            ActionType.RESIZE_INSTANCE: RiskLevel.HIGH,
            ActionType.TERMINATE_INSTANCE: RiskLevel.HIGH,
            ActionType.DELETE_LOAD_BALANCER: RiskLevel.HIGH,
            ActionType.CLEANUP_SECURITY_GROUP: RiskLevel.MEDIUM
        }
        
        risk = base_risk.get(action_type, RiskLevel.MEDIUM)
        
        # Increase risk based on resource characteristics
        if self.check_production_tags(resource_metadata.get("tags", {})):
            # Production resources are always high risk
            risk = RiskLevel.HIGH
        
        is_safe, dependencies = self.validate_resource_dependencies(resource_metadata)
        if not is_safe:
            # Resources with dependencies are higher risk
            if risk == RiskLevel.LOW:
                risk = RiskLevel.MEDIUM
            elif risk == RiskLevel.MEDIUM:
                risk = RiskLevel.HIGH
        
        return risk