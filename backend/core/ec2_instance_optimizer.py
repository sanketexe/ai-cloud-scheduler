"""
EC2 Instance Optimizer for Automated Cost Optimization

Handles EC2 instance optimization actions:
- Detection of unused instances based on CPU utilization
- Stopping unused instances with safety checks
- Resizing underutilized instances for better cost/performance
- Terminating zombie instances with proper validation
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from dataclasses import dataclass
import structlog

from .automation_models import (
    OptimizationAction, ActionType, RiskLevel, AutomationPolicy
)
from .safety_checker import SafetyChecker
from .automation_audit_logger import AutomationAuditLogger

logger = structlog.get_logger(__name__)


@dataclass
class EC2Instance:
    """Represents an EC2 instance with optimization metadata"""
    instance_id: str
    instance_type: str
    state: str
    launch_time: datetime
    tags: Dict[str, str]
    cpu_utilization_avg: float  # Average CPU utilization over analysis period
    cpu_utilization_max: float  # Maximum CPU utilization over analysis period
    network_in_avg: float  # Average network in (bytes/hour)
    network_out_avg: float  # Average network out (bytes/hour)
    monthly_cost: Decimal
    availability_zone: str
    vpc_id: str
    subnet_id: str
    security_groups: List[str]
    auto_scaling_group: Optional[str] = None
    load_balancer_targets: List[str] = None


@dataclass
class InstanceOptimizationOpportunity:
    """Represents an EC2 instance optimization opportunity"""
    instance: EC2Instance
    optimization_type: ActionType
    estimated_monthly_savings: Decimal
    risk_level: RiskLevel
    recommendation_reason: str
    target_instance_type: Optional[str] = None  # For resize operations


class EC2InstanceOptimizer:
    """
    EC2 Instance optimization engine that identifies and executes
    cost optimization opportunities for EC2 instances.
    
    Handles:
    - Unused instance detection and stopping
    - Underutilized instance resizing
    - Zombie instance termination
    """
    
    def __init__(self):
        self.safety_checker = SafetyChecker()
        self.audit_logger = AutomationAuditLogger()
        
        # Thresholds for optimization decisions
        self.UNUSED_CPU_THRESHOLD = 5.0  # CPU utilization below 5% considered unused
        self.UNUSED_NETWORK_THRESHOLD = 1024 * 1024  # 1MB/hour network activity
        self.UNUSED_ANALYSIS_PERIOD_HOURS = 24  # Analyze last 24 hours
        self.UNDERUTILIZED_CPU_THRESHOLD = 20.0  # CPU below 20% considered underutilized
        self.ZOMBIE_TAG_REQUIREMENTS = ["Environment", "Owner", "Project"]  # Required tags
        
    def detect_unused_instances(self, 
                              instances: List[EC2Instance],
                              policy: AutomationPolicy) -> List[InstanceOptimizationOpportunity]:
        """
        Detect EC2 instances that are unused based on CPU utilization analysis.
        
        An instance is considered unused if:
        - Average CPU utilization < 5% over the last 24 hours
        - Network activity is minimal
        - Instance has been running for more than 24 hours
        
        Args:
            instances: List of EC2 instances to analyze
            policy: Automation policy with filtering rules
            
        Returns:
            List of optimization opportunities for unused instances
        """
        logger.info("Starting unused instance detection", 
                   instance_count=len(instances))
        
        opportunities = []
        current_time = datetime.utcnow()
        
        for instance in instances:
            # Skip instances not in running state
            if instance.state != "running":
                continue
                
            # Check if instance has been running long enough to analyze
            running_hours = (current_time - instance.launch_time).total_seconds() / 3600
            if running_hours < self.UNUSED_ANALYSIS_PERIOD_HOURS:
                logger.debug("Instance too new for analysis",
                           instance_id=instance.instance_id,
                           running_hours=running_hours)
                continue
            
            # Apply resource filters from policy
            if not self._matches_resource_filters(instance, policy.resource_filters):
                logger.debug("Instance filtered out by resource filters",
                           instance_id=instance.instance_id,
                           filters=policy.resource_filters)
                continue
            
            # Check if instance meets unused criteria
            is_unused = (
                instance.cpu_utilization_avg < self.UNUSED_CPU_THRESHOLD and
                instance.network_in_avg < self.UNUSED_NETWORK_THRESHOLD and
                instance.network_out_avg < self.UNUSED_NETWORK_THRESHOLD
            )
            
            if is_unused:
                # Calculate estimated savings (instance cost for remaining month)
                days_remaining = 30 - (current_time.day - 1)
                estimated_savings = (instance.monthly_cost / 30) * days_remaining
                
                # Determine risk level based on tags and configuration
                risk_level = self._assess_stop_risk_level(instance)
                
                opportunity = InstanceOptimizationOpportunity(
                    instance=instance,
                    optimization_type=ActionType.STOP_INSTANCE,
                    estimated_monthly_savings=estimated_savings,
                    risk_level=risk_level,
                    recommendation_reason=(
                        f"Instance has {instance.cpu_utilization_avg:.1f}% average CPU "
                        f"utilization over {self.UNUSED_ANALYSIS_PERIOD_HOURS} hours, "
                        f"indicating it is unused"
                    )
                )
                
                opportunities.append(opportunity)
                
                logger.info("Detected unused instance",
                           instance_id=instance.instance_id,
                           cpu_avg=instance.cpu_utilization_avg,
                           estimated_savings=float(estimated_savings))
        
        logger.info("Completed unused instance detection",
                   opportunities_found=len(opportunities))
        
        return opportunities
    
    def detect_underutilized_instances(self,
                                     instances: List[EC2Instance],
                                     policy: AutomationPolicy) -> List[InstanceOptimizationOpportunity]:
        """
        Detect EC2 instances that are underutilized and can be resized to smaller types.
        
        An instance is considered underutilized if:
        - Average CPU utilization < 20% over analysis period
        - A smaller instance type can handle the workload
        - Resizing would result in cost savings
        
        Args:
            instances: List of EC2 instances to analyze
            policy: Automation policy with filtering rules
            
        Returns:
            List of optimization opportunities for underutilized instances
        """
        logger.info("Starting underutilized instance detection",
                   instance_count=len(instances))
        
        opportunities = []
        
        for instance in instances:
            # Skip non-running instances
            if instance.state != "running":
                continue
            
            # Apply resource filters
            if not self._matches_resource_filters(instance, policy.resource_filters):
                continue
            
            # Check if instance is underutilized but not unused
            is_underutilized = (
                self.UNUSED_CPU_THRESHOLD < instance.cpu_utilization_avg < self.UNDERUTILIZED_CPU_THRESHOLD
            )
            
            if is_underutilized:
                # Find optimal smaller instance type
                target_type, savings = self._find_optimal_instance_type(instance)
                
                if target_type and savings > 0:
                    risk_level = self._assess_resize_risk_level(instance)
                    
                    opportunity = InstanceOptimizationOpportunity(
                        instance=instance,
                        optimization_type=ActionType.RESIZE_INSTANCE,
                        estimated_monthly_savings=savings,
                        risk_level=risk_level,
                        recommendation_reason=(
                            f"Instance has {instance.cpu_utilization_avg:.1f}% average CPU "
                            f"utilization and can be resized from {instance.instance_type} "
                            f"to {target_type} for cost savings"
                        ),
                        target_instance_type=target_type
                    )
                    
                    opportunities.append(opportunity)
                    
                    logger.info("Detected underutilized instance",
                               instance_id=instance.instance_id,
                               current_type=instance.instance_type,
                               target_type=target_type,
                               cpu_avg=instance.cpu_utilization_avg,
                               estimated_savings=float(savings))
        
        logger.info("Completed underutilized instance detection",
                   opportunities_found=len(opportunities))
        
        return opportunities
    
    def detect_zombie_instances(self,
                              instances: List[EC2Instance],
                              policy: AutomationPolicy) -> List[InstanceOptimizationOpportunity]:
        """
        Detect zombie instances that lack proper tags and may be orphaned.
        
        A zombie instance is one that:
        - Lacks required tags (Environment, Owner, Project)
        - Has been running for more than 7 days without proper tagging
        - Is not part of an Auto Scaling Group
        
        Args:
            instances: List of EC2 instances to analyze
            policy: Automation policy with filtering rules
            
        Returns:
            List of optimization opportunities for zombie instances
        """
        logger.info("Starting zombie instance detection",
                   instance_count=len(instances))
        
        opportunities = []
        current_time = datetime.utcnow()
        zombie_age_threshold_hours = 7 * 24  # 7 days
        
        for instance in instances:
            # Skip non-running instances
            if instance.state != "running":
                continue
            
            # Skip instances in Auto Scaling Groups (they're managed)
            if instance.auto_scaling_group:
                continue
            
            # Check if instance has been running long enough to be considered zombie
            running_hours = (current_time - instance.launch_time).total_seconds() / 3600
            if running_hours < zombie_age_threshold_hours:
                continue
            
            # Apply resource filters
            if not self._matches_resource_filters(instance, policy.resource_filters):
                continue
            
            # Check for missing required tags
            missing_tags = []
            for required_tag in self.ZOMBIE_TAG_REQUIREMENTS:
                if required_tag not in instance.tags or not instance.tags[required_tag].strip():
                    missing_tags.append(required_tag)
            
            if missing_tags:
                # This is a zombie instance
                estimated_savings = instance.monthly_cost
                risk_level = RiskLevel.HIGH  # Termination is always high risk
                
                opportunity = InstanceOptimizationOpportunity(
                    instance=instance,
                    optimization_type=ActionType.TERMINATE_INSTANCE,
                    estimated_monthly_savings=estimated_savings,
                    risk_level=risk_level,
                    recommendation_reason=(
                        f"Instance lacks required tags: {', '.join(missing_tags)}. "
                        f"Running for {running_hours/24:.1f} days without proper tagging "
                        f"indicates it may be orphaned."
                    )
                )
                
                opportunities.append(opportunity)
                
                logger.info("Detected zombie instance",
                           instance_id=instance.instance_id,
                           missing_tags=missing_tags,
                           running_days=running_hours/24,
                           estimated_savings=float(estimated_savings))
        
        logger.info("Completed zombie instance detection",
                   opportunities_found=len(opportunities))
        
        return opportunities
    
    def stop_unused_instances(self, 
                            instance_ids: List[str],
                            policy: AutomationPolicy) -> Dict[str, bool]:
        """
        Stop unused EC2 instances with CPU utilization analysis and safety checks.
        
        Args:
            instance_ids: List of instance IDs to stop
            policy: Automation policy for safety rules
            
        Returns:
            Dictionary mapping instance_id to success status
        """
        logger.info("Starting unused instance stopping",
                   instance_count=len(instance_ids))
        
        results = {}
        
        for instance_id in instance_ids:
            try:
                # Get instance details (in real implementation, this would use boto3)
                instance = self._get_instance_details(instance_id)
                
                if not instance:
                    logger.error("Instance not found", instance_id=instance_id)
                    results[instance_id] = False
                    continue
                
                # Perform safety checks
                has_production_tags = self.safety_checker.check_production_tags(instance.tags)
                
                if has_production_tags:
                    logger.warning("Safety check failed for instance stop - production tags detected",
                                 instance_id=instance_id)
                    results[instance_id] = False
                    continue
                
                # Check business hours restrictions
                business_hours_config = policy.time_restrictions.get('business_hours', {})
                if business_hours_config and self.safety_checker.verify_business_hours(business_hours_config):
                    # We're in business hours, defer high-risk actions
                    if instance.cpu_utilization_avg < 1.0:  # Very low utilization = higher risk
                        logger.warning("Business hours restriction prevents instance stop",
                                     instance_id=instance_id)
                        results[instance_id] = False
                        continue
                
                # Verify CPU utilization confirms unused status
                if instance.cpu_utilization_avg >= self.UNUSED_CPU_THRESHOLD:
                    logger.warning("Instance CPU utilization too high for stopping",
                                 instance_id=instance_id,
                                 cpu_avg=instance.cpu_utilization_avg)
                    results[instance_id] = False
                    continue
                
                # Execute stop action (simulated)
                logger.info("Stopping unused instance", instance_id=instance_id)
                
                # In real implementation, would use boto3:
                # ec2_client.stop_instances(InstanceIds=[instance_id])
                
                # Log the action
                self.audit_logger.log_action_event(
                    uuid.uuid4(),  # Would be actual action ID
                    "instance_stopped",
                    {
                        "instance_id": instance_id,
                        "cpu_utilization_avg": instance.cpu_utilization_avg,
                        "safety_checks": {"production_tags": has_production_tags},
                        "estimated_savings": float(instance.monthly_cost)
                    }
                )
                
                results[instance_id] = True
                
                logger.info("Successfully stopped unused instance",
                           instance_id=instance_id)
                
            except Exception as e:
                logger.error("Failed to stop unused instance",
                           instance_id=instance_id,
                           error=str(e))
                results[instance_id] = False
        
        logger.info("Completed unused instance stopping",
                   total_instances=len(instance_ids),
                   successful=sum(results.values()))
        
        return results
    
    def resize_underutilized_instances(self,
                                     resize_plans: List[Dict[str, str]],
                                     policy: AutomationPolicy) -> Dict[str, bool]:
        """
        Resize underutilized instances with performance monitoring and safety checks.
        
        Args:
            resize_plans: List of dicts with instance_id, current_type, target_type
            policy: Automation policy for safety rules
            
        Returns:
            Dictionary mapping instance_id to success status
        """
        logger.info("Starting instance resizing",
                   resize_count=len(resize_plans))
        
        results = {}
        
        for plan in resize_plans:
            instance_id = plan["instance_id"]
            current_type = plan["current_type"]
            target_type = plan["target_type"]
            
            try:
                # Get instance details
                instance = self._get_instance_details(instance_id)
                
                if not instance:
                    logger.error("Instance not found for resize", instance_id=instance_id)
                    results[instance_id] = False
                    continue
                
                # Perform safety checks
                has_production_tags = self.safety_checker.check_production_tags(instance.tags)
                
                if has_production_tags:
                    logger.warning("Safety check failed for instance resize - production tags detected",
                                 instance_id=instance_id)
                    results[instance_id] = False
                    continue
                
                # Verify instance is not in Auto Scaling Group
                if instance.auto_scaling_group:
                    logger.warning("Cannot resize instance in Auto Scaling Group",
                                 instance_id=instance_id,
                                 asg=instance.auto_scaling_group)
                    results[instance_id] = False
                    continue
                
                # Validate performance requirements
                if not self._validate_resize_performance(instance, target_type):
                    logger.warning("Target instance type insufficient for workload",
                                 instance_id=instance_id,
                                 current_type=current_type,
                                 target_type=target_type)
                    results[instance_id] = False
                    continue
                
                # Execute resize (simulated - real implementation would use boto3)
                logger.info("Resizing underutilized instance",
                           instance_id=instance_id,
                           current_type=current_type,
                           target_type=target_type)
                
                # Steps: Stop -> Modify -> Start
                # 1. Stop instance
                # 2. Modify instance attribute
                # 3. Start instance
                
                # Log the action
                self.audit_logger.log_action_event(
                    uuid.uuid4(),  # Would be actual action ID
                    "instance_resized",
                    {
                        "instance_id": instance_id,
                        "previous_type": current_type,
                        "new_type": target_type,
                        "cpu_utilization_avg": instance.cpu_utilization_avg,
                        "safety_checks": {"production_tags": has_production_tags}
                    }
                )
                
                results[instance_id] = True
                
                logger.info("Successfully resized underutilized instance",
                           instance_id=instance_id,
                           new_type=target_type)
                
            except Exception as e:
                logger.error("Failed to resize underutilized instance",
                           instance_id=instance_id,
                           error=str(e))
                results[instance_id] = False
        
        logger.info("Completed instance resizing",
                   total_instances=len(resize_plans),
                   successful=sum(results.values()))
        
        return results
    
    def terminate_zombie_instances(self,
                                 instance_ids: List[str],
                                 policy: AutomationPolicy) -> Dict[str, bool]:
        """
        Terminate zombie instances with proper safety checks.
        
        This is a high-risk operation that requires careful validation.
        
        Args:
            instance_ids: List of instance IDs to terminate
            policy: Automation policy for safety rules
            
        Returns:
            Dictionary mapping instance_id to success status
        """
        logger.info("Starting zombie instance termination",
                   instance_count=len(instance_ids))
        
        results = {}
        
        for instance_id in instance_ids:
            try:
                # Get instance details
                instance = self._get_instance_details(instance_id)
                
                if not instance:
                    logger.error("Instance not found for termination", instance_id=instance_id)
                    results[instance_id] = False
                    continue
                
                # Perform comprehensive safety checks for termination
                safety_checks = self._perform_termination_safety_checks(instance, policy)
                
                if not all(safety_checks.values()):
                    logger.warning("Safety checks failed for zombie termination",
                                 instance_id=instance_id,
                                 failed_checks=[k for k, v in safety_checks.items() if not v])
                    results[instance_id] = False
                    continue
                
                # Verify zombie status
                missing_tags = []
                for required_tag in self.ZOMBIE_TAG_REQUIREMENTS:
                    if required_tag not in instance.tags or not instance.tags[required_tag].strip():
                        missing_tags.append(required_tag)
                
                if not missing_tags:
                    logger.warning("Instance no longer qualifies as zombie",
                                 instance_id=instance_id)
                    results[instance_id] = False
                    continue
                
                # Execute termination (simulated)
                logger.info("Terminating zombie instance",
                           instance_id=instance_id,
                           missing_tags=missing_tags)
                
                # In real implementation, would use boto3:
                # ec2_client.terminate_instances(InstanceIds=[instance_id])
                
                # Log the action
                self.audit_logger.log_action_event(
                    uuid.uuid4(),  # Would be actual action ID
                    "zombie_instance_terminated",
                    {
                        "instance_id": instance_id,
                        "missing_tags": missing_tags,
                        "safety_checks": safety_checks,
                        "estimated_savings": float(instance.monthly_cost)
                    }
                )
                
                results[instance_id] = True
                
                logger.info("Successfully terminated zombie instance",
                           instance_id=instance_id)
                
            except Exception as e:
                logger.error("Failed to terminate zombie instance",
                           instance_id=instance_id,
                           error=str(e))
                results[instance_id] = False
        
        logger.info("Completed zombie instance termination",
                   total_instances=len(instance_ids),
                   successful=sum(results.values()))
        
        return results
    
    def _matches_resource_filters(self, 
                                instance: EC2Instance, 
                                resource_filters: Dict[str, Any]) -> bool:
        """Check if instance matches policy resource filters"""
        
        # Check exclude tags - if any exclude tag matches, filter out the instance
        exclude_tags = resource_filters.get("exclude_tags", [])
        for exclude_tag in exclude_tags:
            if "=" in exclude_tag:
                key, value = exclude_tag.split("=", 1)
                if instance.tags.get(key) == value:
                    logger.debug("Instance excluded by tag filter",
                               instance_id=instance.instance_id,
                               exclude_tag=exclude_tag,
                               instance_tag_value=instance.tags.get(key))
                    return False
            else:
                if exclude_tag in instance.tags:
                    logger.debug("Instance excluded by tag key filter",
                               instance_id=instance.instance_id,
                               exclude_tag=exclude_tag)
                    return False
        
        # Additional safety check: exclude instances with production tags
        # This provides an extra layer of protection beyond policy filters
        has_production_tags = self.safety_checker.check_production_tags(instance.tags)
        if has_production_tags:
            logger.debug("Instance excluded due to production tags",
                       instance_id=instance.instance_id,
                       tags=instance.tags)
            return False
        
        # Check minimum cost threshold
        min_cost = resource_filters.get("min_cost_threshold", 0)
        if instance.monthly_cost < min_cost:
            return False
        
        # Check included services (EC2 should be included)
        include_services = resource_filters.get("include_services", ["EC2"])
        if "EC2" not in include_services:
            return False
        
        return True
    
    def _assess_stop_risk_level(self, instance: EC2Instance) -> RiskLevel:
        """Assess risk level for stopping an instance"""
        
        # High risk if production tags
        production_indicators = ["production", "prod", "live"]
        for tag_value in instance.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return RiskLevel.HIGH
        
        # Medium risk if part of load balancer targets
        if instance.load_balancer_targets:
            return RiskLevel.MEDIUM
        
        # Low risk otherwise
        return RiskLevel.LOW
    
    def _assess_resize_risk_level(self, instance: EC2Instance) -> RiskLevel:
        """Assess risk level for resizing an instance"""
        
        # High risk if production or part of ASG
        if instance.auto_scaling_group:
            return RiskLevel.HIGH
        
        production_indicators = ["production", "prod", "live"]
        for tag_value in instance.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return RiskLevel.HIGH
        
        # Medium risk if high CPU utilization (close to threshold)
        if instance.cpu_utilization_max > 50:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _find_optimal_instance_type(self, 
                                  instance: EC2Instance) -> Tuple[Optional[str], Decimal]:
        """Find optimal smaller instance type and calculate savings"""
        
        # Simplified instance type mapping for cost optimization
        # In real implementation, this would use AWS pricing APIs
        instance_type_hierarchy = {
            "t3.large": ("t3.medium", Decimal("20.00")),
            "t3.xlarge": ("t3.large", Decimal("40.00")),
            "t3.2xlarge": ("t3.xlarge", Decimal("80.00")),
            "m5.large": ("m5.medium", Decimal("25.00")),
            "m5.xlarge": ("m5.large", Decimal("50.00")),
            "m5.2xlarge": ("m5.xlarge", Decimal("100.00")),
            "c5.large": ("c5.medium", Decimal("22.00")),
            "c5.xlarge": ("c5.large", Decimal("44.00")),
            "c5.2xlarge": ("c5.xlarge", Decimal("88.00")),
        }
        
        if instance.instance_type in instance_type_hierarchy:
            target_type, monthly_savings = instance_type_hierarchy[instance.instance_type]
            return target_type, monthly_savings
        
        return None, Decimal("0")
    
    def _validate_resize_performance(self, 
                                   instance: EC2Instance, 
                                   target_type: str) -> bool:
        """Validate that target instance type can handle the workload"""
        
        # Simplified validation - in real implementation would check:
        # - CPU requirements vs target CPU capacity
        # - Memory requirements vs target memory
        # - Network requirements vs target network performance
        # - Storage IOPS requirements
        
        # For now, ensure CPU utilization won't exceed 80% on target
        # This is a simplified calculation
        current_capacity_factor = self._get_instance_capacity_factor(instance.instance_type)
        target_capacity_factor = self._get_instance_capacity_factor(target_type)
        
        if target_capacity_factor == 0:
            return False
        
        projected_cpu = instance.cpu_utilization_max * (current_capacity_factor / target_capacity_factor)
        
        return projected_cpu <= 80.0
    
    def _get_instance_capacity_factor(self, instance_type: str) -> float:
        """Get relative capacity factor for instance type"""
        
        # Simplified capacity factors (in real implementation, would be more comprehensive)
        capacity_factors = {
            "t3.nano": 0.5,
            "t3.micro": 1.0,
            "t3.small": 2.0,
            "t3.medium": 4.0,
            "t3.large": 8.0,
            "t3.xlarge": 16.0,
            "t3.2xlarge": 32.0,
            "m5.medium": 4.0,
            "m5.large": 8.0,
            "m5.xlarge": 16.0,
            "m5.2xlarge": 32.0,
            "c5.medium": 4.0,
            "c5.large": 8.0,
            "c5.xlarge": 16.0,
            "c5.2xlarge": 32.0,
        }
        
        return capacity_factors.get(instance_type, 0)
    
    def _perform_termination_safety_checks(self, 
                                         instance: EC2Instance,
                                         policy: AutomationPolicy) -> Dict[str, bool]:
        """Perform comprehensive safety checks before termination"""
        
        checks = {}
        
        # Check 1: Not in Auto Scaling Group
        checks["not_in_asg"] = instance.auto_scaling_group is None
        
        # Check 2: Not attached to load balancers
        checks["not_lb_target"] = not instance.load_balancer_targets
        
        # Check 3: No production tags
        production_indicators = ["production", "prod", "live", "critical"]
        has_production_tags = any(
            any(indicator in tag_value.lower() for indicator in production_indicators)
            for tag_value in instance.tags.values()
        )
        checks["no_production_tags"] = not has_production_tags
        
        # Check 4: Business hours compliance
        business_hours_config = policy.time_restrictions.get('business_hours', {})
        if business_hours_config:
            # For termination, we want to avoid business hours
            checks["business_hours_ok"] = not self.safety_checker.verify_business_hours(business_hours_config)
        else:
            checks["business_hours_ok"] = True
        
        # Check 5: Instance age (must be old enough to be considered zombie)
        instance_age_days = (datetime.utcnow() - instance.launch_time).days
        checks["sufficient_age"] = instance_age_days >= 7
        
        # Check 6: Missing required tags (confirms zombie status)
        missing_tags = []
        for required_tag in self.ZOMBIE_TAG_REQUIREMENTS:
            if required_tag not in instance.tags or not instance.tags[required_tag].strip():
                missing_tags.append(required_tag)
        checks["missing_required_tags"] = len(missing_tags) > 0
        
        return checks
    
    def _get_instance_details(self, instance_id: str) -> Optional[EC2Instance]:
        """Get instance details (simulated - would use boto3 in real implementation)"""
        
        # This is a placeholder implementation
        # In real system, would use boto3 to get actual instance details
        
        # Simulate instance data
        return EC2Instance(
            instance_id=instance_id,
            instance_type="t3.medium",
            state="running",
            launch_time=datetime.utcnow() - timedelta(days=2),
            tags={"Name": f"test-instance-{instance_id[-4:]}", "Environment": "dev"},
            cpu_utilization_avg=3.5,
            cpu_utilization_max=8.2,
            network_in_avg=512 * 1024,  # 512 KB/hour
            network_out_avg=256 * 1024,  # 256 KB/hour
            monthly_cost=Decimal("45.60"),
            availability_zone="us-east-1a",
            vpc_id="vpc-12345678",
            subnet_id="subnet-12345678",
            security_groups=["sg-12345678"],
            auto_scaling_group=None,
            load_balancer_targets=[]
        )