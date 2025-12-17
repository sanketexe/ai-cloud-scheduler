"""
Storage Optimizer for Automated Cost Optimization

Handles EBS volume optimization actions:
- Detection of unattached volumes for deletion
- Upgrading gp2 volumes to gp3 for cost savings
- Volume cleanup with safety validation
- Snapshot creation before deletion for safety
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
class EBSVolume:
    """Represents an EBS volume with optimization metadata"""
    volume_id: str
    volume_type: str  # gp2, gp3, io1, io2, st1, sc1
    size_gb: int
    state: str  # available, in-use, creating, deleting
    attachment_state: Optional[str]  # attached, detached, attaching, detaching
    attached_instance_id: Optional[str]
    creation_time: datetime
    last_attachment_time: Optional[datetime]
    last_detachment_time: Optional[datetime]
    tags: Dict[str, str]
    monthly_cost: Decimal
    iops: Optional[int]  # Provisioned IOPS for io1/io2
    throughput: Optional[int]  # Throughput for gp3
    availability_zone: str
    encrypted: bool
    snapshot_id: Optional[str]  # Source snapshot if created from snapshot


@dataclass
class VolumeOptimizationOpportunity:
    """Represents an EBS volume optimization opportunity"""
    volume: EBSVolume
    optimization_type: ActionType
    estimated_monthly_savings: Decimal
    risk_level: RiskLevel
    recommendation_reason: str
    target_volume_type: Optional[str] = None  # For upgrade operations
    target_iops: Optional[int] = None  # For gp3 upgrades
    target_throughput: Optional[int] = None  # For gp3 upgrades


class StorageOptimizer:
    """
    EBS volume optimization engine that identifies and executes
    cost optimization opportunities for storage resources.
    
    Handles:
    - Unattached volume detection and deletion
    - gp2 to gp3 upgrades for cost savings
    - Volume cleanup with safety validation
    """
    
    def __init__(self):
        self.safety_checker = SafetyChecker()
        self.audit_logger = AutomationAuditLogger()
        
        # Thresholds for optimization decisions
        self.UNATTACHED_THRESHOLD_DAYS = 7  # Delete volumes unattached for 7+ days
        self.GP2_TO_GP3_MIN_SIZE_GB = 100  # Only upgrade volumes >= 100GB
        self.GP3_DEFAULT_IOPS = 3000  # Default IOPS for gp3
        self.GP3_DEFAULT_THROUGHPUT = 125  # Default throughput (MB/s) for gp3
        
        # Cost calculations (simplified - real implementation would use AWS pricing API)
        self.COST_PER_GB_GP2 = Decimal("0.10")  # $0.10 per GB per month
        self.COST_PER_GB_GP3 = Decimal("0.08")  # $0.08 per GB per month (20% savings)
        
    def detect_unattached_volumes(self, 
                                volumes: List[EBSVolume],
                                policy: AutomationPolicy) -> List[VolumeOptimizationOpportunity]:
        """
        Detect EBS volumes that are unattached and can be deleted.
        
        A volume is considered for deletion if:
        - It has been unattached for more than 7 days
        - It's not a root volume (based on tags/metadata)
        - It passes safety checks
        
        Args:
            volumes: List of EBS volumes to analyze
            policy: Automation policy with filtering rules
            
        Returns:
            List of optimization opportunities for unattached volumes
        """
        logger.info("Starting unattached volume detection", 
                   volume_count=len(volumes))
        
        opportunities = []
        current_time = datetime.utcnow()
        
        for volume in volumes:
            # Skip volumes that are currently attached
            if volume.attachment_state == "attached" or volume.attached_instance_id:
                continue
            
            # Skip volumes that are not in available state
            if volume.state != "available":
                continue
            
            # Apply resource filters from policy
            if not self._matches_resource_filters(volume, policy.resource_filters):
                logger.debug("Volume filtered out by resource filters",
                           volume_id=volume.volume_id,
                           filters=policy.resource_filters)
                continue
            
            # Check how long the volume has been unattached
            unattached_time = None
            if volume.last_detachment_time:
                unattached_time = volume.last_detachment_time
            else:
                # If no detachment time, use creation time as fallback
                unattached_time = volume.creation_time
            
            days_unattached = (current_time - unattached_time).days
            
            if days_unattached >= self.UNATTACHED_THRESHOLD_DAYS:
                # Check if this is a root volume or has protection tags
                if self._is_protected_volume(volume):
                    logger.debug("Volume protected from deletion",
                               volume_id=volume.volume_id,
                               protection_reason="Protected tags or root volume")
                    continue
                
                # Calculate estimated savings (full monthly cost)
                estimated_savings = volume.monthly_cost
                
                # Determine risk level
                risk_level = self._assess_deletion_risk_level(volume)
                
                opportunity = VolumeOptimizationOpportunity(
                    volume=volume,
                    optimization_type=ActionType.DELETE_VOLUME,
                    estimated_monthly_savings=estimated_savings,
                    risk_level=risk_level,
                    recommendation_reason=(
                        f"Volume has been unattached for {days_unattached} days "
                        f"(threshold: {self.UNATTACHED_THRESHOLD_DAYS} days). "
                        f"Size: {volume.size_gb}GB, Type: {volume.volume_type}"
                    )
                )
                
                opportunities.append(opportunity)
                
                logger.info("Detected unattached volume",
                           volume_id=volume.volume_id,
                           days_unattached=days_unattached,
                           size_gb=volume.size_gb,
                           estimated_savings=float(estimated_savings))
        
        logger.info("Completed unattached volume detection",
                   opportunities_found=len(opportunities))
        
        return opportunities
    
    def detect_gp2_upgrade_opportunities(self,
                                       volumes: List[EBSVolume],
                                       policy: AutomationPolicy) -> List[VolumeOptimizationOpportunity]:
        """
        Detect gp2 volumes that can be upgraded to gp3 for cost savings.
        
        A gp2 volume is considered for upgrade if:
        - It's larger than 100GB (where gp3 savings are significant)
        - It's currently in-use or available
        - It passes safety checks
        
        Args:
            volumes: List of EBS volumes to analyze
            policy: Automation policy with filtering rules
            
        Returns:
            List of optimization opportunities for gp2 to gp3 upgrades
        """
        logger.info("Starting gp2 to gp3 upgrade detection",
                   volume_count=len(volumes))
        
        opportunities = []
        
        for volume in volumes:
            # Only consider gp2 volumes
            if volume.volume_type != "gp2":
                continue
            
            # Only upgrade volumes that meet minimum size threshold
            if volume.size_gb < self.GP2_TO_GP3_MIN_SIZE_GB:
                continue
            
            # Skip volumes not in stable states
            if volume.state not in ["available", "in-use"]:
                continue
            
            # Apply resource filters from policy
            if not self._matches_resource_filters(volume, policy.resource_filters):
                logger.debug("Volume filtered out by resource filters",
                           volume_id=volume.volume_id,
                           filters=policy.resource_filters)
                continue
            
            # Calculate cost savings from gp2 to gp3 upgrade
            current_cost = volume.size_gb * self.COST_PER_GB_GP2
            gp3_cost = volume.size_gb * self.COST_PER_GB_GP3
            monthly_savings = current_cost - gp3_cost
            
            if monthly_savings > 0:
                # Determine optimal gp3 configuration
                # gp3 baseline is 3,000 IOPS, but we can use 3 IOPS per GB as baseline
                baseline_iops = min(3000, volume.size_gb * 3)
                target_iops = max(baseline_iops, self.GP3_DEFAULT_IOPS)
                target_throughput = self.GP3_DEFAULT_THROUGHPUT
                
                # Determine risk level
                risk_level = self._assess_upgrade_risk_level(volume)
                
                opportunity = VolumeOptimizationOpportunity(
                    volume=volume,
                    optimization_type=ActionType.UPGRADE_STORAGE,
                    estimated_monthly_savings=monthly_savings,
                    risk_level=risk_level,
                    recommendation_reason=(
                        f"Upgrading {volume.size_gb}GB gp2 volume to gp3 will save "
                        f"${monthly_savings:.2f}/month (20% cost reduction). "
                        f"Performance will be maintained or improved."
                    ),
                    target_volume_type="gp3",
                    target_iops=target_iops,
                    target_throughput=target_throughput
                )
                
                opportunities.append(opportunity)
                
                logger.info("Detected gp2 upgrade opportunity",
                           volume_id=volume.volume_id,
                           size_gb=volume.size_gb,
                           monthly_savings=float(monthly_savings),
                           target_iops=target_iops)
        
        logger.info("Completed gp2 to gp3 upgrade detection",
                   opportunities_found=len(opportunities))
        
        return opportunities
    
    def delete_unattached_volumes(self,
                                volume_ids: List[str],
                                policy: AutomationPolicy) -> Dict[str, bool]:
        """
        Delete unattached EBS volumes after creating snapshots for safety.
        
        Args:
            volume_ids: List of volume IDs to delete
            policy: Automation policy for safety rules
            
        Returns:
            Dictionary mapping volume_id to success status
        """
        logger.info("Starting unattached volume deletion",
                   volume_count=len(volume_ids))
        
        results = {}
        
        for volume_id in volume_ids:
            try:
                # Get volume details (in real implementation, this would use boto3)
                volume = self._get_volume_details(volume_id)
                
                if not volume:
                    logger.error("Volume not found", volume_id=volume_id)
                    results[volume_id] = False
                    continue
                
                # Perform safety checks
                if not self._validate_volume_deletion_safety(volume, policy):
                    logger.warning("Safety check failed for volume deletion",
                                 volume_id=volume_id)
                    results[volume_id] = False
                    continue
                
                # Create snapshot before deletion for safety
                snapshot_id = self._create_volume_snapshot(volume)
                
                if not snapshot_id:
                    logger.error("Failed to create safety snapshot",
                               volume_id=volume_id)
                    results[volume_id] = False
                    continue
                
                # Execute volume deletion (simulated)
                logger.info("Deleting unattached volume",
                           volume_id=volume_id,
                           snapshot_id=snapshot_id)
                
                # In real implementation, would use boto3:
                # ec2_client.delete_volume(VolumeId=volume_id)
                
                # Log the action
                self.audit_logger.log_action_event(
                    uuid.uuid4(),  # Would be actual action ID
                    "volume_deleted",
                    {
                        "volume_id": volume_id,
                        "size_gb": volume.size_gb,
                        "volume_type": volume.volume_type,
                        "snapshot_id": snapshot_id,
                        "days_unattached": (datetime.utcnow() - (volume.last_detachment_time or volume.creation_time)).days,
                        "estimated_savings": float(volume.monthly_cost)
                    }
                )
                
                results[volume_id] = True
                
                logger.info("Successfully deleted unattached volume",
                           volume_id=volume_id,
                           snapshot_id=snapshot_id)
                
            except Exception as e:
                logger.error("Failed to delete unattached volume",
                           volume_id=volume_id,
                           error=str(e))
                results[volume_id] = False
        
        logger.info("Completed unattached volume deletion",
                   total_volumes=len(volume_ids),
                   successful=sum(results.values()))
        
        return results
    
    def upgrade_gp2_to_gp3(self,
                          upgrade_plans: List[Dict[str, Any]],
                          policy: AutomationPolicy) -> Dict[str, bool]:
        """
        Upgrade gp2 volumes to gp3 with cost calculation and safety checks.
        
        Args:
            upgrade_plans: List of dicts with volume_id, target_iops, target_throughput
            policy: Automation policy for safety rules
            
        Returns:
            Dictionary mapping volume_id to success status
        """
        logger.info("Starting gp2 to gp3 upgrades",
                   upgrade_count=len(upgrade_plans))
        
        results = {}
        
        for plan in upgrade_plans:
            volume_id = plan["volume_id"]
            target_iops = plan.get("target_iops", self.GP3_DEFAULT_IOPS)
            target_throughput = plan.get("target_throughput", self.GP3_DEFAULT_THROUGHPUT)
            
            try:
                # Get volume details
                volume = self._get_volume_details(volume_id)
                
                if not volume:
                    logger.error("Volume not found for upgrade", volume_id=volume_id)
                    results[volume_id] = False
                    continue
                
                # Verify volume is gp2
                if volume.volume_type != "gp2":
                    logger.warning("Volume is not gp2, skipping upgrade",
                                 volume_id=volume_id,
                                 current_type=volume.volume_type)
                    results[volume_id] = False
                    continue
                
                # Perform safety checks
                if not self._validate_volume_upgrade_safety(volume, policy):
                    logger.warning("Safety check failed for volume upgrade",
                                 volume_id=volume_id)
                    results[volume_id] = False
                    continue
                
                # Validate gp3 configuration
                if not self._validate_gp3_configuration(volume, target_iops, target_throughput):
                    logger.warning("Invalid gp3 configuration",
                                 volume_id=volume_id,
                                 target_iops=target_iops,
                                 target_throughput=target_throughput)
                    results[volume_id] = False
                    continue
                
                # Execute upgrade (simulated)
                logger.info("Upgrading volume from gp2 to gp3",
                           volume_id=volume_id,
                           size_gb=volume.size_gb,
                           target_iops=target_iops,
                           target_throughput=target_throughput)
                
                # In real implementation, would use boto3:
                # ec2_client.modify_volume(
                #     VolumeId=volume_id,
                #     VolumeType='gp3',
                #     Iops=target_iops,
                #     Throughput=target_throughput
                # )
                
                # Calculate actual savings
                current_cost = volume.size_gb * self.COST_PER_GB_GP2
                new_cost = volume.size_gb * self.COST_PER_GB_GP3
                monthly_savings = current_cost - new_cost
                
                # Log the action
                self.audit_logger.log_action_event(
                    uuid.uuid4(),  # Would be actual action ID
                    "volume_upgraded",
                    {
                        "volume_id": volume_id,
                        "size_gb": volume.size_gb,
                        "previous_type": "gp2",
                        "new_type": "gp3",
                        "target_iops": target_iops,
                        "target_throughput": target_throughput,
                        "monthly_savings": float(monthly_savings)
                    }
                )
                
                results[volume_id] = True
                
                logger.info("Successfully upgraded volume to gp3",
                           volume_id=volume_id,
                           monthly_savings=float(monthly_savings))
                
            except Exception as e:
                logger.error("Failed to upgrade volume to gp3",
                           volume_id=volume_id,
                           error=str(e))
                results[volume_id] = False
        
        logger.info("Completed gp2 to gp3 upgrades",
                   total_volumes=len(upgrade_plans),
                   successful=sum(results.values()))
        
        return results
    
    def _matches_resource_filters(self, 
                                volume: EBSVolume, 
                                resource_filters: Dict[str, Any]) -> bool:
        """Check if volume matches policy resource filters"""
        
        # Check exclude tags - if any exclude tag matches, filter out the volume
        exclude_tags = resource_filters.get("exclude_tags", [])
        for exclude_tag in exclude_tags:
            if "=" in exclude_tag:
                key, value = exclude_tag.split("=", 1)
                if volume.tags.get(key) == value:
                    logger.debug("Volume excluded by tag filter",
                               volume_id=volume.volume_id,
                               exclude_tag=exclude_tag,
                               volume_tag_value=volume.tags.get(key))
                    return False
            else:
                if exclude_tag in volume.tags:
                    logger.debug("Volume excluded by tag key filter",
                               volume_id=volume.volume_id,
                               exclude_tag=exclude_tag)
                    return False
        
        # Additional safety check: exclude volumes with production tags
        has_production_tags = self.safety_checker.check_production_tags(volume.tags)
        if has_production_tags:
            logger.debug("Volume excluded due to production tags",
                       volume_id=volume.volume_id,
                       tags=volume.tags)
            return False
        
        # Check minimum cost threshold
        min_cost = resource_filters.get("min_cost_threshold", 0)
        if volume.monthly_cost < min_cost:
            return False
        
        # Check included services (EBS should be included)
        include_services = resource_filters.get("include_services", ["EBS"])
        if "EBS" not in include_services:
            return False
        
        return True
    
    def _is_protected_volume(self, volume: EBSVolume) -> bool:
        """Check if volume is protected from deletion"""
        
        # Check for root volume indicators
        root_indicators = ["root", "boot", "system", "/dev/sda1", "/dev/xvda"]
        for tag_value in volume.tags.values():
            if any(indicator in tag_value.lower() for indicator in root_indicators):
                return True
        
        # Check for protection tags
        protection_tags = ["DoNotDelete", "Protected", "Backup", "Critical"]
        for tag_key in protection_tags:
            if tag_key in volume.tags:
                tag_value = volume.tags[tag_key].lower()
                if tag_value in ["true", "yes", "1", "enabled"]:
                    return True
        
        # Check if volume was created from a snapshot (might be important)
        if volume.snapshot_id and "ami-" in volume.snapshot_id:
            return True  # Created from AMI snapshot, likely important
        
        return False
    
    def _assess_deletion_risk_level(self, volume: EBSVolume) -> RiskLevel:
        """Assess risk level for deleting a volume"""
        
        # High risk if production tags or large size
        production_indicators = ["production", "prod", "live"]
        for tag_value in volume.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return RiskLevel.HIGH
        
        # High risk for very large volumes (>1TB)
        if volume.size_gb > 1000:
            return RiskLevel.HIGH
        
        # Medium risk for encrypted volumes or volumes with snapshots
        if volume.encrypted or volume.snapshot_id:
            return RiskLevel.MEDIUM
        
        # Medium risk for volumes larger than 100GB
        if volume.size_gb > 100:
            return RiskLevel.MEDIUM
        
        # Low risk otherwise
        return RiskLevel.LOW
    
    def _assess_upgrade_risk_level(self, volume: EBSVolume) -> RiskLevel:
        """Assess risk level for upgrading a volume"""
        
        # High risk if production tags
        production_indicators = ["production", "prod", "live"]
        for tag_value in volume.tags.values():
            if any(indicator in tag_value.lower() for indicator in production_indicators):
                return RiskLevel.HIGH
        
        # Medium risk if volume is currently attached to running instance
        if volume.attachment_state == "attached":
            return RiskLevel.MEDIUM
        
        # Low risk for unattached volumes or development environments
        return RiskLevel.LOW
    
    def _validate_volume_deletion_safety(self, 
                                       volume: EBSVolume,
                                       policy: AutomationPolicy) -> bool:
        """Validate safety requirements for volume deletion"""
        
        # Check if volume is actually unattached
        if volume.attachment_state == "attached" or volume.attached_instance_id:
            logger.warning("Cannot delete attached volume",
                         volume_id=volume.volume_id,
                         attachment_state=volume.attachment_state)
            return False
        
        # Check production tags
        has_production_tags = self.safety_checker.check_production_tags(volume.tags)
        if has_production_tags:
            logger.warning("Cannot delete volume with production tags",
                         volume_id=volume.volume_id,
                         tags=volume.tags)
            return False
        
        # Check if volume is protected
        if self._is_protected_volume(volume):
            logger.warning("Cannot delete protected volume",
                         volume_id=volume.volume_id)
            return False
        
        # Check business hours for high-risk deletions
        business_hours_config = policy.time_restrictions.get('business_hours', {})
        if business_hours_config and self.safety_checker.verify_business_hours(business_hours_config):
            # We're in business hours, check if this is a high-risk deletion
            if volume.size_gb > 500 or volume.encrypted:  # Large or encrypted volumes
                logger.warning("Business hours restriction prevents high-risk volume deletion",
                             volume_id=volume.volume_id,
                             size_gb=volume.size_gb,
                             encrypted=volume.encrypted)
                return False
        
        return True
    
    def _validate_volume_upgrade_safety(self, 
                                      volume: EBSVolume,
                                      policy: AutomationPolicy) -> bool:
        """Validate safety requirements for volume upgrade"""
        
        # Check production tags for high-risk upgrades
        has_production_tags = self.safety_checker.check_production_tags(volume.tags)
        if has_production_tags:
            # Production volumes can be upgraded but need extra caution
            logger.info("Production volume upgrade requires extra validation",
                       volume_id=volume.volume_id)
            
            # Check business hours for production upgrades
            business_hours_config = policy.time_restrictions.get('business_hours', {})
            if business_hours_config and self.safety_checker.verify_business_hours(business_hours_config):
                logger.warning("Business hours restriction prevents production volume upgrade",
                             volume_id=volume.volume_id)
                return False
        
        return True
    
    def _validate_gp3_configuration(self, 
                                  volume: EBSVolume,
                                  target_iops: int,
                                  target_throughput: int) -> bool:
        """Validate gp3 configuration parameters"""
        
        # gp3 IOPS limits: 3,000 to 16,000 IOPS (but baseline is 3,000)
        # For volumes smaller than 1TB, we can use baseline IOPS
        baseline_iops = min(3000, volume.size_gb * 3)  # 3 IOPS per GB baseline
        
        if target_iops < baseline_iops:
            target_iops = baseline_iops
        
        if target_iops > 16000:
            return False
        
        # gp3 throughput limits: 125 to 1,000 MB/s
        if target_throughput < 125 or target_throughput > 1000:
            return False
        
        # For gp3, baseline throughput is 125 MB/s
        # Additional throughput can be provisioned independently of IOPS
        # No strict ratio requirement between IOPS and throughput for gp3
        
        return True
    
    def _create_volume_snapshot(self, volume: EBSVolume) -> Optional[str]:
        """Create a snapshot of the volume before deletion"""
        
        snapshot_id = f"snap-{uuid.uuid4().hex[:8]}"
        
        logger.info("Creating safety snapshot before volume deletion",
                   volume_id=volume.volume_id,
                   snapshot_id=snapshot_id)
        
        # In real implementation, would use boto3:
        # response = ec2_client.create_snapshot(
        #     VolumeId=volume.volume_id,
        #     Description=f"Automated backup before deletion of {volume.volume_id}"
        # )
        # return response['SnapshotId']
        
        # Simulate snapshot creation
        return snapshot_id
    
    def _get_volume_details(self, volume_id: str) -> Optional[EBSVolume]:
        """Get volume details (simulated - would use boto3 in real implementation)"""
        
        # This is a placeholder implementation
        # In real system, would use boto3 to get actual volume details
        
        # Simulate volume data
        return EBSVolume(
            volume_id=volume_id,
            volume_type="gp2",
            size_gb=150,
            state="available",
            attachment_state="detached",
            attached_instance_id=None,
            creation_time=datetime.utcnow() - timedelta(days=10),
            last_attachment_time=datetime.utcnow() - timedelta(days=8),
            last_detachment_time=datetime.utcnow() - timedelta(days=8),
            tags={"Name": f"test-volume-{volume_id[-4:]}", "Environment": "dev"},
            monthly_cost=Decimal("15.00"),  # $0.10 per GB for gp2
            iops=None,
            throughput=None,
            availability_zone="us-east-1a",
            encrypted=False,
            snapshot_id=None
        )