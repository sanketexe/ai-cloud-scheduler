"""
Network Optimizer for Automated Cost Optimization

Handles network resource optimization actions:
- Detection of unused Elastic IP addresses for release
- Identification of unused load balancers for deletion
- Cleanup of unused security groups with dependency analysis
- Network resource management with safety checks
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
class ElasticIP:
    """Represents an Elastic IP address with optimization metadata"""
    allocation_id: str
    public_ip: str
    association_id: Optional[str]
    associated_instance_id: Optional[str]
    associated_network_interface_id: Optional[str]
    domain: str  # vpc or classic
    allocation_time: datetime
    last_association_time: Optional[datetime]
    last_disassociation_time: Optional[datetime]
    tags: Dict[str, str]
    monthly_cost: Decimal
    region: str


@dataclass
class LoadBalancer:
    """Represents a load balancer with optimization metadata"""
    load_balancer_arn: str
    load_balancer_name: str
    load_balancer_type: str  # application, network, gateway, classic
    scheme: str  # internet-facing, internal
    state: str  # active, provisioning, failed
    vpc_id: Optional[str]
    availability_zones: List[str]
    security_groups: List[str]
    target_groups: List[str]
    healthy_targets_count: int
    unhealthy_targets_count: int
    total_targets_count: int
    creation_time: datetime
    last_request_time: Optional[datetime]
    request_count_24h: int
    data_processed_gb_24h: float
    tags: Dict[str, str]
    monthly_cost: Decimal
    region: str


@dataclass
class SecurityGroup:
    """Represents a security group with optimization metadata"""
    group_id: str
    group_name: str
    description: str
    vpc_id: Optional[str]
    owner_id: str
    inbound_rules: List[Dict[str, Any]]
    outbound_rules: List[Dict[str, Any]]
    referenced_by_groups: List[str]  # Other security groups that reference this one
    attached_resources: List[str]  # EC2 instances, RDS, etc. using this group
    creation_time: datetime
    last_modified_time: Optional[datetime]
    tags: Dict[str, str]
    region: str


@dataclass
class NetworkOptimizationOpportunity:
    """Represents a network resource optimization opportunity"""
    resource_id: str
    resource_type: str  # elastic_ip, load_balancer, security_group
    optimization_type: ActionType
    estimated_monthly_savings: Decimal
    risk_level: RiskLevel
    recommendation_reason: str
    resource_metadata: Dict[str, Any]


class NetworkOptimizer:
    """
    Network resource optimization engine that identifies and executes
    cost optimization opportunities for network resources.
    
    Handles:
    - Unused Elastic IP detection and release
    - Unused load balancer detection and deletion
    - Unused security group cleanup with dependency analysis
    """
    
    def __init__(self):
        self.safety_checker = SafetyChecker()
        self.audit_logger = AutomationAuditLogger()
        
        # Thresholds for optimization decisions
        self.UNUSED_EIP_THRESHOLD_DAYS = 7  # Release EIPs unused for 7+ days
        self.UNUSED_LB_THRESHOLD_DAYS = 7   # Delete LBs with no targets for 7+ days
        self.UNUSED_SG_THRESHOLD_DAYS = 30  # Delete SGs unused for 30+ days
        self.MIN_REQUEST_COUNT_24H = 10     # Minimum requests to consider LB active
        
        # Cost calculations (simplified - real implementation would use AWS pricing API)
        self.COST_PER_EIP_MONTHLY = Decimal("3.65")  # $0.005 per hour
        self.COST_PER_ALB_MONTHLY = Decimal("16.43")  # $0.0225 per hour
        self.COST_PER_NLB_MONTHLY = Decimal("16.43")  # $0.0225 per hour
        self.COST_PER_CLB_MONTHLY = Decimal("18.25")  # $0.025 per hour
        
    def detect_unused_elastic_ips(self, 
                                 elastic_ips: List[ElasticIP],
                                 policy: AutomationPolicy) -> List[NetworkOptimizationOpportunity]:
        """
        Detect Elastic IP addresses that are unused and can be released.
        
        An EIP is considered unused if:
        - It's not associated with any instance or network interface
        - It has been unassociated for more than 7 days
        - It passes safety checks
        
        Args:
            elastic_ips: List of Elastic IP addresses to analyze
            policy: Automation policy with filtering rules
            
        Returns:
            List of optimization opportunities for unused EIPs
        """
        logger.info("Starting unused Elastic IP detection", 
                   eip_count=len(elastic_ips))
        
        opportunities = []
        current_time = datetime.utcnow()
        
        for eip in elastic_ips:
            # Skip EIPs that are currently associated
            if eip.association_id or eip.associated_instance_id or eip.associated_network_interface_id:
                continue
            
            # Apply resource filters from policy
            if not self._matches_resource_filters(eip, policy.resource_filters, "elastic_ip"):
                logger.debug("EIP filtered out by resource filters",
                           allocation_id=eip.allocation_id,
                           filters=policy.resource_filters)
                continue
            
            # Check how long the EIP has been unassociated
            unassociated_time = None
            if eip.last_disassociation_time:
                unassociated_time = eip.last_disassociation_time
            else:
                # If no disassociation time, use allocation time as fallback
                unassociated_time = eip.allocation_time
            
            days_unassociated = (current_time - unassociated_time).days
            
            if days_unassociated >= self.UNUSED_EIP_THRESHOLD_DAYS:
                # Check if this EIP is protected
                if self._is_protected_eip(eip):
                    logger.debug("EIP protected from release",
                               allocation_id=eip.allocation_id,
                               protection_reason="Protected tags")
                    continue
                
                # Calculate estimated savings (full monthly cost)
                estimated_savings = self.COST_PER_EIP_MONTHLY
                
                # Determine risk level
                risk_level = self._assess_eip_release_risk_level(eip)
                
                opportunity = NetworkOptimizationOpportunity(
                    resource_id=eip.allocation_id,
                    resource_type="elastic_ip",
                    optimization_type=ActionType.RELEASE_ELASTIC_IP,
                    estimated_monthly_savings=estimated_savings,
                    risk_level=risk_level,
                    recommendation_reason=(
                        f"Elastic IP {eip.public_ip} has been unassociated for {days_unassociated} days "
                        f"(threshold: {self.UNUSED_EIP_THRESHOLD_DAYS} days). "
                        f"Releasing will save ${estimated_savings}/month."
                    ),
                    resource_metadata={
                        "public_ip": eip.public_ip,
                        "domain": eip.domain,
                        "tags": eip.tags,
                        "allocation_time": eip.allocation_time.isoformat(),
                        "days_unassociated": days_unassociated
                    }
                )
                
                opportunities.append(opportunity)
                
                logger.info("Detected unused Elastic IP",
                           allocation_id=eip.allocation_id,
                           public_ip=eip.public_ip,
                           days_unassociated=days_unassociated,
                           estimated_savings=float(estimated_savings))
        
        logger.info("Completed unused Elastic IP detection",
                   opportunities_found=len(opportunities))
        
        return opportunities
    
    def detect_unused_load_balancers(self,
                                   load_balancers: List[LoadBalancer],
                                   policy: AutomationPolicy) -> List[NetworkOptimizationOpportunity]:
        """
        Detect load balancers that are unused and can be deleted.
        
        A load balancer is considered unused if:
        - It has no healthy targets for more than 7 days
        - It has minimal request traffic (< 10 requests in 24h)
        - It passes safety checks
        
        Args:
            load_balancers: List of load balancers to analyze
            policy: Automation policy with filtering rules
            
        Returns:
            List of optimization opportunities for unused load balancers
        """
        logger.info("Starting unused load balancer detection",
                   lb_count=len(load_balancers))
        
        opportunities = []
        current_time = datetime.utcnow()
        
        for lb in load_balancers:
            # Skip load balancers not in active state
            if lb.state != "active":
                continue
            
            # Apply resource filters from policy
            if not self._matches_resource_filters(lb, policy.resource_filters, "load_balancer"):
                logger.debug("Load balancer filtered out by resource filters",
                           lb_name=lb.load_balancer_name,
                           filters=policy.resource_filters)
                continue
            
            # Check if load balancer has no healthy targets
            has_healthy_targets = lb.healthy_targets_count > 0
            has_minimal_traffic = lb.request_count_24h < self.MIN_REQUEST_COUNT_24H
            
            # Calculate how long it's been without healthy targets
            days_without_targets = 0
            if not has_healthy_targets and lb.last_request_time:
                days_without_targets = (current_time - lb.last_request_time).days
            elif not has_healthy_targets:
                # If no last request time, use creation time
                days_without_targets = (current_time - lb.creation_time).days
            
            is_unused = (
                not has_healthy_targets and 
                has_minimal_traffic and 
                days_without_targets >= self.UNUSED_LB_THRESHOLD_DAYS
            )
            
            if is_unused:
                # Check if this load balancer is protected
                if self._is_protected_load_balancer(lb):
                    logger.debug("Load balancer protected from deletion",
                               lb_name=lb.load_balancer_name,
                               protection_reason="Protected tags or critical")
                    continue
                
                # Calculate estimated savings based on load balancer type
                estimated_savings = self._calculate_lb_monthly_cost(lb)
                
                # Determine risk level
                risk_level = self._assess_lb_deletion_risk_level(lb)
                
                opportunity = NetworkOptimizationOpportunity(
                    resource_id=lb.load_balancer_arn,
                    resource_type="load_balancer",
                    optimization_type=ActionType.DELETE_LOAD_BALANCER,
                    estimated_monthly_savings=estimated_savings,
                    risk_level=risk_level,
                    recommendation_reason=(
                        f"Load balancer {lb.load_balancer_name} has no healthy targets for {days_without_targets} days "
                        f"and only {lb.request_count_24h} requests in the last 24 hours "
                        f"(threshold: {self.MIN_REQUEST_COUNT_24H}). "
                        f"Deleting will save ${estimated_savings}/month."
                    ),
                    resource_metadata={
                        "load_balancer_name": lb.load_balancer_name,
                        "load_balancer_type": lb.load_balancer_type,
                        "scheme": lb.scheme,
                        "healthy_targets_count": lb.healthy_targets_count,
                        "total_targets_count": lb.total_targets_count,
                        "request_count_24h": lb.request_count_24h,
                        "tags": lb.tags,
                        "vpc_id": lb.vpc_id,
                        "days_without_targets": days_without_targets
                    }
                )
                
                opportunities.append(opportunity)
                
                logger.info("Detected unused load balancer",
                           lb_name=lb.load_balancer_name,
                           lb_type=lb.load_balancer_type,
                           days_without_targets=days_without_targets,
                           request_count=lb.request_count_24h,
                           estimated_savings=float(estimated_savings))
        
        logger.info("Completed unused load balancer detection",
                   opportunities_found=len(opportunities))
        
        return opportunities
    
    def detect_unused_security_groups(self,
                                    security_groups: List[SecurityGroup],
                                    policy: AutomationPolicy) -> List[NetworkOptimizationOpportunity]:
        """
        Detect security groups that are unused and can be deleted.
        
        A security group is considered unused if:
        - It's not attached to any resources (EC2, RDS, etc.)
        - It's not referenced by other security groups
        - It has been unused for more than 30 days
        - It's not a default security group
        
        Args:
            security_groups: List of security groups to analyze
            policy: Automation policy with filtering rules
            
        Returns:
            List of optimization opportunities for unused security groups
        """
        logger.info("Starting unused security group detection",
                   sg_count=len(security_groups))
        
        opportunities = []
        current_time = datetime.utcnow()
        
        for sg in security_groups:
            # Skip default security groups (they can't be deleted)
            if sg.group_name == "default":
                continue
            
            # Apply resource filters from policy
            if not self._matches_resource_filters(sg, policy.resource_filters, "security_group"):
                logger.debug("Security group filtered out by resource filters",
                           group_id=sg.group_id,
                           filters=policy.resource_filters)
                continue
            
            # Check if security group is unused
            has_attached_resources = len(sg.attached_resources) > 0
            is_referenced_by_others = len(sg.referenced_by_groups) > 0
            
            # Calculate how long it's been unused
            days_unused = 0
            if sg.last_modified_time:
                days_unused = (current_time - sg.last_modified_time).days
            else:
                # If no last modified time, use creation time
                days_unused = (current_time - sg.creation_time).days
            
            is_unused = (
                not has_attached_resources and 
                not is_referenced_by_others and 
                days_unused >= self.UNUSED_SG_THRESHOLD_DAYS
            )
            
            if is_unused:
                # Check if this security group is protected
                if self._is_protected_security_group(sg):
                    logger.debug("Security group protected from deletion",
                               group_id=sg.group_id,
                               protection_reason="Protected tags or critical rules")
                    continue
                
                # Security groups don't have direct costs, but cleanup provides operational benefits
                estimated_savings = Decimal("0.00")  # No direct cost savings
                
                # Determine risk level
                risk_level = self._assess_sg_deletion_risk_level(sg)
                
                opportunity = NetworkOptimizationOpportunity(
                    resource_id=sg.group_id,
                    resource_type="security_group",
                    optimization_type=ActionType.CLEANUP_SECURITY_GROUP,
                    estimated_monthly_savings=estimated_savings,
                    risk_level=risk_level,
                    recommendation_reason=(
                        f"Security group {sg.group_name} ({sg.group_id}) has no attached resources "
                        f"and is not referenced by other security groups for {days_unused} days "
                        f"(threshold: {self.UNUSED_SG_THRESHOLD_DAYS} days). "
                        f"Cleanup improves security posture and reduces management overhead."
                    ),
                    resource_metadata={
                        "group_name": sg.group_name,
                        "description": sg.description,
                        "vpc_id": sg.vpc_id,
                        "inbound_rules_count": len(sg.inbound_rules),
                        "outbound_rules_count": len(sg.outbound_rules),
                        "tags": sg.tags,
                        "days_unused": days_unused
                    }
                )
                
                opportunities.append(opportunity)
                
                logger.info("Detected unused security group",
                           group_id=sg.group_id,
                           group_name=sg.group_name,
                           days_unused=days_unused)
        
        logger.info("Completed unused security group detection",
                   opportunities_found=len(opportunities))
        
        return opportunities
    
    def release_unused_elastic_ips(self,
                                 allocation_ids: List[str],
                                 policy: AutomationPolicy) -> Dict[str, bool]:
        """
        Release unused Elastic IP addresses with association checks.
        
        Args:
            allocation_ids: List of EIP allocation IDs to release
            policy: Automation policy for safety rules
            
        Returns:
            Dictionary mapping allocation_id to success status
        """
        logger.info("Starting Elastic IP release",
                   eip_count=len(allocation_ids))
        
        results = {}
        
        for allocation_id in allocation_ids:
            try:
                # Get EIP details (in real implementation, this would use boto3)
                eip = self._get_eip_details(allocation_id)
                
                if not eip:
                    logger.error("Elastic IP not found", allocation_id=allocation_id)
                    results[allocation_id] = False
                    continue
                
                # Perform safety checks
                if not self._validate_eip_release_safety(eip, policy):
                    logger.warning("Safety check failed for EIP release",
                                 allocation_id=allocation_id)
                    results[allocation_id] = False
                    continue
                
                # Verify EIP is not associated
                if eip.association_id or eip.associated_instance_id:
                    logger.warning("Cannot release associated Elastic IP",
                                 allocation_id=allocation_id,
                                 association_id=eip.association_id)
                    results[allocation_id] = False
                    continue
                
                # Execute EIP release (simulated)
                logger.info("Releasing unused Elastic IP",
                           allocation_id=allocation_id,
                           public_ip=eip.public_ip)
                
                # In real implementation, would use boto3:
                # ec2_client.release_address(AllocationId=allocation_id)
                
                # Log the action
                self.audit_logger.log_action_event(
                    uuid.uuid4(),  # Would be actual action ID
                    "elastic_ip_released",
                    {
                        "allocation_id": allocation_id,
                        "public_ip": eip.public_ip,
                        "domain": eip.domain,
                        "days_unassociated": (datetime.utcnow() - (eip.last_disassociation_time or eip.allocation_time)).days,
                        "estimated_savings": float(self.COST_PER_EIP_MONTHLY)
                    }
                )
                
                results[allocation_id] = True
                
                logger.info("Successfully released Elastic IP",
                           allocation_id=allocation_id,
                           public_ip=eip.public_ip)
                
            except Exception as e:
                logger.error("Failed to release Elastic IP",
                           allocation_id=allocation_id,
                           error=str(e))
                results[allocation_id] = False
        
        logger.info("Completed Elastic IP release",
                   total_eips=len(allocation_ids),
                   successful=sum(results.values()))
        
        return results
    
    def delete_unused_load_balancers(self,
                                   load_balancer_arns: List[str],
                                   policy: AutomationPolicy) -> Dict[str, bool]:
        """
        Delete unused load balancers with target validation.
        
        Args:
            load_balancer_arns: List of load balancer ARNs to delete
            policy: Automation policy for safety rules
            
        Returns:
            Dictionary mapping load_balancer_arn to success status
        """
        logger.info("Starting load balancer deletion",
                   lb_count=len(load_balancer_arns))
        
        results = {}
        
        for lb_arn in load_balancer_arns:
            try:
                # Get load balancer details
                lb = self._get_load_balancer_details(lb_arn)
                
                if not lb:
                    logger.error("Load balancer not found", lb_arn=lb_arn)
                    results[lb_arn] = False
                    continue
                
                # Perform safety checks
                if not self._validate_lb_deletion_safety(lb, policy):
                    logger.warning("Safety check failed for load balancer deletion",
                                 lb_arn=lb_arn)
                    results[lb_arn] = False
                    continue
                
                # Verify load balancer has no healthy targets
                if lb.healthy_targets_count > 0:
                    logger.warning("Cannot delete load balancer with healthy targets",
                                 lb_arn=lb_arn,
                                 healthy_targets=lb.healthy_targets_count)
                    results[lb_arn] = False
                    continue
                
                # Execute load balancer deletion (simulated)
                logger.info("Deleting unused load balancer",
                           lb_arn=lb_arn,
                           lb_name=lb.load_balancer_name)
                
                # In real implementation, would use boto3:
                # if lb.load_balancer_type in ['application', 'network', 'gateway']:
                #     elbv2_client.delete_load_balancer(LoadBalancerArn=lb_arn)
                # else:  # classic load balancer
                #     elb_client.delete_load_balancer(LoadBalancerName=lb.load_balancer_name)
                
                # Calculate savings
                estimated_savings = self._calculate_lb_monthly_cost(lb)
                
                # Log the action
                self.audit_logger.log_action_event(
                    uuid.uuid4(),  # Would be actual action ID
                    "load_balancer_deleted",
                    {
                        "load_balancer_arn": lb_arn,
                        "load_balancer_name": lb.load_balancer_name,
                        "load_balancer_type": lb.load_balancer_type,
                        "healthy_targets_count": lb.healthy_targets_count,
                        "total_targets_count": lb.total_targets_count,
                        "request_count_24h": lb.request_count_24h,
                        "estimated_savings": float(estimated_savings)
                    }
                )
                
                results[lb_arn] = True
                
                logger.info("Successfully deleted load balancer",
                           lb_arn=lb_arn,
                           lb_name=lb.load_balancer_name)
                
            except Exception as e:
                logger.error("Failed to delete load balancer",
                           lb_arn=lb_arn,
                           error=str(e))
                results[lb_arn] = False
        
        logger.info("Completed load balancer deletion",
                   total_lbs=len(load_balancer_arns),
                   successful=sum(results.values()))
        
        return results
    
    def cleanup_unused_security_groups(self,
                                     group_ids: List[str],
                                     policy: AutomationPolicy) -> Dict[str, bool]:
        """
        Cleanup unused security groups with dependency analysis.
        
        Args:
            group_ids: List of security group IDs to delete
            policy: Automation policy for safety rules
            
        Returns:
            Dictionary mapping group_id to success status
        """
        logger.info("Starting security group cleanup",
                   sg_count=len(group_ids))
        
        results = {}
        
        for group_id in group_ids:
            try:
                # Get security group details
                sg = self._get_security_group_details(group_id)
                
                if not sg:
                    logger.error("Security group not found", group_id=group_id)
                    results[group_id] = False
                    continue
                
                # Perform safety checks
                if not self._validate_sg_deletion_safety(sg, policy):
                    logger.warning("Safety check failed for security group deletion",
                                 group_id=group_id)
                    results[group_id] = False
                    continue
                
                # Verify security group has no dependencies
                if sg.attached_resources or sg.referenced_by_groups:
                    logger.warning("Cannot delete security group with dependencies",
                                 group_id=group_id,
                                 attached_resources=len(sg.attached_resources),
                                 referenced_by=len(sg.referenced_by_groups))
                    results[group_id] = False
                    continue
                
                # Execute security group deletion (simulated)
                logger.info("Deleting unused security group",
                           group_id=group_id,
                           group_name=sg.group_name)
                
                # In real implementation, would use boto3:
                # ec2_client.delete_security_group(GroupId=group_id)
                
                # Log the action
                self.audit_logger.log_action_event(
                    uuid.uuid4(),  # Would be actual action ID
                    "security_group_deleted",
                    {
                        "group_id": group_id,
                        "group_name": sg.group_name,
                        "vpc_id": sg.vpc_id,
                        "inbound_rules_count": len(sg.inbound_rules),
                        "outbound_rules_count": len(sg.outbound_rules),
                        "days_unused": (datetime.utcnow() - (sg.last_modified_time or sg.creation_time)).days
                    }
                )
                
                results[group_id] = True
                
                logger.info("Successfully deleted security group",
                           group_id=group_id,
                           group_name=sg.group_name)
                
            except Exception as e:
                logger.error("Failed to delete security group",
                           group_id=group_id,
                           error=str(e))
                results[group_id] = False
        
        logger.info("Completed security group cleanup",
                   total_sgs=len(group_ids),
                   successful=sum(results.values()))
        
        return results
    
    def _matches_resource_filters(self, 
                                resource,  # ElasticIP, LoadBalancer, or SecurityGroup
                                resource_filters: Dict[str, Any],
                                resource_type: str) -> bool:
        """Check if resource matches policy resource filters"""
        
        # Get resource tags
        resource_tags = resource.tags
        
        # Check exclude tags - if any exclude tag matches, filter out the resource
        exclude_tags = resource_filters.get("exclude_tags", [])
        for exclude_tag in exclude_tags:
            if "=" in exclude_tag:
                key, value = exclude_tag.split("=", 1)
                if resource_tags.get(key) == value:
                    logger.debug("Resource excluded by tag filter",
                               resource_id=getattr(resource, 'allocation_id', getattr(resource, 'load_balancer_arn', getattr(resource, 'group_id', 'unknown'))),
                               exclude_tag=exclude_tag,
                               resource_tag_value=resource_tags.get(key))
                    return False
            else:
                if exclude_tag in resource_tags:
                    logger.debug("Resource excluded by tag key filter",
                               resource_id=getattr(resource, 'allocation_id', getattr(resource, 'load_balancer_arn', getattr(resource, 'group_id', 'unknown'))),
                               exclude_tag=exclude_tag)
                    return False
        
        # Additional safety check: exclude resources with production tags
        has_production_tags = self.safety_checker.check_production_tags(resource_tags)
        if has_production_tags:
            logger.debug("Resource excluded due to production tags",
                       resource_id=getattr(resource, 'allocation_id', getattr(resource, 'load_balancer_arn', getattr(resource, 'group_id', 'unknown'))),
                       tags=resource_tags)
            return False
        
        # Check minimum cost threshold (only applies to resources with costs)
        min_cost = resource_filters.get("min_cost_threshold", 0)
        if hasattr(resource, 'monthly_cost') and resource.monthly_cost < min_cost:
            return False
        
        # Check included services
        include_services = resource_filters.get("include_services", [])
        if include_services:
            service_mapping = {
                "elastic_ip": "EIP",
                "load_balancer": "ELB",
                "security_group": "EC2"
            }
            required_service = service_mapping.get(resource_type)
            if required_service and required_service not in include_services:
                return False
        
        return True
    
    def _is_protected_eip(self, eip: ElasticIP) -> bool:
        """Check if Elastic IP is protected from release"""
        
        # Check for protection tags
        protection_tags = ["DoNotDelete", "Protected", "Critical", "Reserved"]
        for tag_key in protection_tags:
            if tag_key in eip.tags:
                tag_value = eip.tags[tag_key].lower()
                if tag_value in ["true", "yes", "1", "enabled"]:
                    return True
        
        # Check for production indicators in tags
        production_indicators = ["production", "prod", "live"]
        for tag_value in eip.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return True
        
        return False
    
    def _is_protected_load_balancer(self, lb: LoadBalancer) -> bool:
        """Check if load balancer is protected from deletion"""
        
        # Check for protection tags
        protection_tags = ["DoNotDelete", "Protected", "Critical"]
        for tag_key in protection_tags:
            if tag_key in lb.tags:
                tag_value = lb.tags[tag_key].lower()
                if tag_value in ["true", "yes", "1", "enabled"]:
                    return True
        
        # Check for production indicators
        production_indicators = ["production", "prod", "live"]
        for tag_value in lb.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return True
        
        # Internet-facing load balancers are higher risk
        if lb.scheme == "internet-facing":
            return True
        
        return False
    
    def _is_protected_security_group(self, sg: SecurityGroup) -> bool:
        """Check if security group is protected from deletion"""
        
        # Default security groups can't be deleted
        if sg.group_name == "default":
            return True
        
        # Check for protection tags
        protection_tags = ["DoNotDelete", "Protected", "Critical"]
        for tag_key in protection_tags:
            if tag_key in sg.tags:
                tag_value = sg.tags[tag_key].lower()
                if tag_value in ["true", "yes", "1", "enabled"]:
                    return True
        
        # Check for production indicators
        production_indicators = ["production", "prod", "live"]
        for tag_value in sg.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return True
        
        # Check for critical ports in rules (database, SSH, etc.)
        critical_ports = [22, 3389, 3306, 5432, 1433, 27017, 6379, 443, 80]
        all_rules = sg.inbound_rules + sg.outbound_rules
        
        for rule in all_rules:
            if "port" in rule:
                port = rule["port"]
                if isinstance(port, int) and port in critical_ports:
                    return True
                elif isinstance(port, str) and any(str(cp) in port for cp in critical_ports):
                    return True
        
        return False
    
    def _assess_eip_release_risk_level(self, eip: ElasticIP) -> RiskLevel:
        """Assess risk level for releasing an Elastic IP"""
        
        # High risk if production tags
        production_indicators = ["production", "prod", "live"]
        for tag_value in eip.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return RiskLevel.HIGH
        
        # Medium risk if recently allocated (< 30 days)
        days_since_allocation = (datetime.utcnow() - eip.allocation_time).days
        if days_since_allocation < 30:
            return RiskLevel.MEDIUM
        
        # Low risk otherwise
        return RiskLevel.LOW
    
    def _assess_lb_deletion_risk_level(self, lb: LoadBalancer) -> RiskLevel:
        """Assess risk level for deleting a load balancer"""
        
        # High risk if production tags or internet-facing
        production_indicators = ["production", "prod", "live"]
        for tag_value in lb.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return RiskLevel.HIGH
        
        if lb.scheme == "internet-facing":
            return RiskLevel.HIGH
        
        # Medium risk if has any targets (even unhealthy)
        if lb.total_targets_count > 0:
            return RiskLevel.MEDIUM
        
        # Medium risk for Application Load Balancers (more complex)
        if lb.load_balancer_type == "application":
            return RiskLevel.MEDIUM
        
        # Low risk otherwise
        return RiskLevel.LOW
    
    def _assess_sg_deletion_risk_level(self, sg: SecurityGroup) -> RiskLevel:
        """Assess risk level for deleting a security group"""
        
        # High risk if production tags
        production_indicators = ["production", "prod", "live"]
        for tag_value in sg.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return RiskLevel.HIGH
        
        # Medium risk if has many rules (complex configuration)
        total_rules = len(sg.inbound_rules) + len(sg.outbound_rules)
        if total_rules > 10:
            return RiskLevel.MEDIUM
        
        # Low risk otherwise
        return RiskLevel.LOW
    
    def _calculate_lb_monthly_cost(self, lb: LoadBalancer) -> Decimal:
        """Calculate monthly cost for a load balancer"""
        
        cost_mapping = {
            "application": self.COST_PER_ALB_MONTHLY,
            "network": self.COST_PER_NLB_MONTHLY,
            "gateway": self.COST_PER_ALB_MONTHLY,  # Similar to ALB
            "classic": self.COST_PER_CLB_MONTHLY
        }
        
        return cost_mapping.get(lb.load_balancer_type, self.COST_PER_ALB_MONTHLY)
    
    def _validate_eip_release_safety(self, 
                                   eip: ElasticIP,
                                   policy: AutomationPolicy) -> bool:
        """Validate safety requirements for EIP release"""
        
        # Check if EIP is actually unassociated
        if eip.association_id or eip.associated_instance_id or eip.associated_network_interface_id:
            logger.warning("Cannot release associated Elastic IP",
                         allocation_id=eip.allocation_id,
                         association_id=eip.association_id)
            return False
        
        # Check production tags
        has_production_tags = self.safety_checker.check_production_tags(eip.tags)
        if has_production_tags:
            logger.warning("Cannot release EIP with production tags",
                         allocation_id=eip.allocation_id,
                         tags=eip.tags)
            return False
        
        # Check if EIP is protected
        if self._is_protected_eip(eip):
            logger.warning("Cannot release protected EIP",
                         allocation_id=eip.allocation_id)
            return False
        
        return True
    
    def _validate_lb_deletion_safety(self, 
                                   lb: LoadBalancer,
                                   policy: AutomationPolicy) -> bool:
        """Validate safety requirements for load balancer deletion"""
        
        # Check if load balancer has healthy targets
        if lb.healthy_targets_count > 0:
            logger.warning("Cannot delete load balancer with healthy targets",
                         lb_name=lb.load_balancer_name,
                         healthy_targets=lb.healthy_targets_count)
            return False
        
        # Check production tags
        has_production_tags = self.safety_checker.check_production_tags(lb.tags)
        if has_production_tags:
            logger.warning("Cannot delete load balancer with production tags",
                         lb_name=lb.load_balancer_name,
                         tags=lb.tags)
            return False
        
        # Check if load balancer is protected
        if self._is_protected_load_balancer(lb):
            logger.warning("Cannot delete protected load balancer",
                         lb_name=lb.load_balancer_name)
            return False
        
        # Check business hours for internet-facing load balancers
        business_hours_config = policy.time_restrictions.get('business_hours', {})
        if (business_hours_config and 
            self.safety_checker.verify_business_hours(business_hours_config) and
            lb.scheme == "internet-facing"):
            logger.warning("Business hours restriction prevents internet-facing LB deletion",
                         lb_name=lb.load_balancer_name)
            return False
        
        return True
    
    def _validate_sg_deletion_safety(self, 
                                   sg: SecurityGroup,
                                   policy: AutomationPolicy) -> bool:
        """Validate safety requirements for security group deletion"""
        
        # Check if security group has dependencies
        if sg.attached_resources or sg.referenced_by_groups:
            logger.warning("Cannot delete security group with dependencies",
                         group_id=sg.group_id,
                         attached_resources=len(sg.attached_resources),
                         referenced_by=len(sg.referenced_by_groups))
            return False
        
        # Cannot delete default security groups
        if sg.group_name == "default":
            logger.warning("Cannot delete default security group",
                         group_id=sg.group_id)
            return False
        
        # Check production tags
        has_production_tags = self.safety_checker.check_production_tags(sg.tags)
        if has_production_tags:
            logger.warning("Cannot delete security group with production tags",
                         group_id=sg.group_id,
                         tags=sg.tags)
            return False
        
        # Check if security group is protected
        if self._is_protected_security_group(sg):
            logger.warning("Cannot delete protected security group",
                         group_id=sg.group_id)
            return False
        
        return True
    
    def _get_eip_details(self, allocation_id: str) -> Optional[ElasticIP]:
        """Get EIP details (simulated - would use boto3 in real implementation)"""
        
        # This is a placeholder implementation
        # In real system, would use boto3 to get actual EIP details
        
        # Simulate EIP data
        return ElasticIP(
            allocation_id=allocation_id,
            public_ip=f"203.0.113.{allocation_id[-3:]}",
            association_id=None,
            associated_instance_id=None,
            associated_network_interface_id=None,
            domain="vpc",
            allocation_time=datetime.utcnow() - timedelta(days=10),
            last_association_time=datetime.utcnow() - timedelta(days=8),
            last_disassociation_time=datetime.utcnow() - timedelta(days=8),
            tags={"Name": f"test-eip-{allocation_id[-4:]}", "Environment": "dev"},
            monthly_cost=self.COST_PER_EIP_MONTHLY,
            region="us-east-1"
        )
    
    def _get_load_balancer_details(self, lb_arn: str) -> Optional[LoadBalancer]:
        """Get load balancer details (simulated - would use boto3 in real implementation)"""
        
        # This is a placeholder implementation
        # In real system, would use boto3 to get actual load balancer details
        
        # Extract LB name from ARN (simplified)
        lb_name = lb_arn.split("/")[-1] if "/" in lb_arn else lb_arn
        
        # Simulate load balancer data
        return LoadBalancer(
            load_balancer_arn=lb_arn,
            load_balancer_name=lb_name,
            load_balancer_type="application",
            scheme="internal",
            state="active",
            vpc_id="vpc-12345678",
            availability_zones=["us-east-1a", "us-east-1b"],
            security_groups=["sg-12345678"],
            target_groups=["tg-12345678"],
            healthy_targets_count=0,
            unhealthy_targets_count=0,
            total_targets_count=0,
            creation_time=datetime.utcnow() - timedelta(days=15),
            last_request_time=datetime.utcnow() - timedelta(days=10),
            request_count_24h=5,
            data_processed_gb_24h=0.1,
            tags={"Name": f"test-lb-{lb_name[-4:]}", "Environment": "dev"},
            monthly_cost=self.COST_PER_ALB_MONTHLY,
            region="us-east-1"
        )
    
    def _get_security_group_details(self, group_id: str) -> Optional[SecurityGroup]:
        """Get security group details (simulated - would use boto3 in real implementation)"""
        
        # This is a placeholder implementation
        # In real system, would use boto3 to get actual security group details
        
        # Simulate security group data
        return SecurityGroup(
            group_id=group_id,
            group_name=f"test-sg-{group_id[-4:]}",
            description="Test security group for optimization",
            vpc_id="vpc-12345678",
            owner_id="123456789012",
            inbound_rules=[
                {"protocol": "tcp", "port": 80, "source": "0.0.0.0/0"},
                {"protocol": "tcp", "port": 443, "source": "0.0.0.0/0"}
            ],
            outbound_rules=[
                {"protocol": "-1", "port": "all", "destination": "0.0.0.0/0"}
            ],
            referenced_by_groups=[],
            attached_resources=[],
            creation_time=datetime.utcnow() - timedelta(days=45),
            last_modified_time=datetime.utcnow() - timedelta(days=35),
            tags={"Name": f"test-sg-{group_id[-4:]}", "Environment": "dev"},
            region="us-east-1"
        )