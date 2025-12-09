"""
Tagging Engine for Cloud Migration Advisor

This module generates and applies tags to cloud resources based on categorization.
It handles tag generation, validation, conflict resolution, and provider-specific
tag application.

Requirements: 5.3
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .resource_discovery_engine import CloudResource, CloudProvider
from .auto_categorization_engine import ResourceCategorization, CategorizedResources


logger = logging.getLogger(__name__)


class TagConflictResolution(Enum):
    """Strategies for resolving tag conflicts"""
    OVERWRITE = "overwrite"  # Overwrite existing tags
    PRESERVE = "preserve"  # Keep existing tags
    MERGE = "merge"  # Merge new and existing tags
    FAIL = "fail"  # Fail on conflict


@dataclass
class TaggingPolicy:
    """Policy for tag generation and application"""
    policy_id: str
    name: str
    required_tags: List[str] = field(default_factory=list)  # Tags that must be present
    tag_prefix: Optional[str] = None  # Prefix for all generated tags
    conflict_resolution: TagConflictResolution = TagConflictResolution.MERGE
    max_tags_per_resource: int = 50  # Cloud provider limits
    tag_key_max_length: int = 128
    tag_value_max_length: int = 256
    
    def __post_init__(self):
        """Ensure conflict_resolution is enum"""
        if isinstance(self.conflict_resolution, str):
            self.conflict_resolution = TagConflictResolution(self.conflict_resolution)


@dataclass
class TagOperation:
    """Represents a tag operation to be performed"""
    resource_id: str
    provider: CloudProvider
    operation_type: str  # "add", "update", "delete"
    tag_key: str
    tag_value: Optional[str] = None
    
    def __post_init__(self):
        """Ensure provider is enum"""
        if isinstance(self.provider, str):
            self.provider = CloudProvider(self.provider)


@dataclass
class TaggingResult:
    """Result of tagging operation"""
    resource_id: str
    success: bool
    tags_applied: Dict[str, str] = field(default_factory=dict)
    tags_skipped: Dict[str, str] = field(default_factory=dict)
    conflicts_resolved: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class BulkTaggingResult:
    """Result of bulk tagging operation"""
    total_resources: int
    successful: int = 0
    failed: int = 0
    results: List[TaggingResult] = field(default_factory=list)
    
    def add_result(self, result: TaggingResult):
        """Add a tagging result"""
        self.results.append(result)
        if result.success:
            self.successful += 1
        else:
            self.failed += 1


class TagGenerator:
    """Generates tags from resource categorization"""
    
    def __init__(self, policy: TaggingPolicy):
        """
        Initialize tag generator with policy
        
        Args:
            policy: Tagging policy to use
        """
        self.policy = policy
        logger.info(f"Tag Generator initialized with policy: {policy.name}")
    
    def generate_tags(
        self,
        categorization: ResourceCategorization
    ) -> Dict[str, str]:
        """
        Generate tags from resource categorization
        
        Args:
            categorization: Resource categorization
            
        Returns:
            Dictionary of tags to apply
        """
        tags = {}
        
        # Add team tag
        if categorization.team:
            key = self._format_tag_key("team")
            tags[key] = self._format_tag_value(categorization.team)
        
        # Add project tag
        if categorization.project:
            key = self._format_tag_key("project")
            tags[key] = self._format_tag_value(categorization.project)
        
        # Add environment tag
        if categorization.environment:
            key = self._format_tag_key("environment")
            tags[key] = self._format_tag_value(categorization.environment)
        
        # Add region tag
        if categorization.region:
            key = self._format_tag_key("region")
            tags[key] = self._format_tag_value(categorization.region)
        
        # Add cost center tag
        if categorization.cost_center:
            key = self._format_tag_key("cost_center")
            tags[key] = self._format_tag_value(categorization.cost_center)
        
        # Add custom attributes
        for key, value in categorization.custom_attributes.items():
            formatted_key = self._format_tag_key(key)
            tags[formatted_key] = self._format_tag_value(value)
        
        # Add metadata tags
        tags[self._format_tag_key("managed_by")] = "migration_advisor"
        tags[self._format_tag_key("categorization_method")] = categorization.categorization_method
        tags[self._format_tag_key("categorization_confidence")] = str(
            round(categorization.confidence_score, 2)
        )
        
        # Validate tags
        validated_tags = self._validate_tags(tags)
        
        return validated_tags
    
    def _format_tag_key(self, key: str) -> str:
        """
        Format tag key according to policy
        
        Args:
            key: Raw tag key
            
        Returns:
            Formatted tag key
        """
        # Apply prefix if configured
        if self.policy.tag_prefix:
            key = f"{self.policy.tag_prefix}:{key}"
        
        # Ensure key length is within limits
        if len(key) > self.policy.tag_key_max_length:
            key = key[:self.policy.tag_key_max_length]
        
        return key
    
    def _format_tag_value(self, value: str) -> str:
        """
        Format tag value according to policy
        
        Args:
            value: Raw tag value
            
        Returns:
            Formatted tag value
        """
        # Ensure value length is within limits
        if len(value) > self.policy.tag_value_max_length:
            value = value[:self.policy.tag_value_max_length]
        
        return value
    
    def _validate_tags(self, tags: Dict[str, str]) -> Dict[str, str]:
        """
        Validate tags according to policy
        
        Args:
            tags: Tags to validate
            
        Returns:
            Validated tags
        """
        validated = {}
        
        # Check tag count limit
        if len(tags) > self.policy.max_tags_per_resource:
            logger.warning(
                f"Tag count ({len(tags)}) exceeds limit ({self.policy.max_tags_per_resource})"
            )
            # Keep only the first N tags
            tag_items = list(tags.items())[:self.policy.max_tags_per_resource]
            tags = dict(tag_items)
        
        # Validate each tag
        for key, value in tags.items():
            # Ensure key and value are strings
            key = str(key)
            value = str(value)
            
            # Remove invalid characters (provider-specific rules would go here)
            # For now, just ensure they're not empty
            if key and value:
                validated[key] = value
        
        return validated


class TagApplicator:
    """Applies tags to cloud resources"""
    
    def __init__(self, policy: TaggingPolicy):
        """
        Initialize tag applicator with policy
        
        Args:
            policy: Tagging policy to use
        """
        self.policy = policy
        logger.info(f"Tag Applicator initialized with policy: {policy.name}")
    
    def apply_tags(
        self,
        resource: CloudResource,
        new_tags: Dict[str, str]
    ) -> TaggingResult:
        """
        Apply tags to a resource
        
        Args:
            resource: Resource to tag
            new_tags: Tags to apply
            
        Returns:
            TaggingResult with operation results
        """
        result = TaggingResult(
            resource_id=resource.resource_id,
            success=True
        )
        
        # Resolve conflicts with existing tags
        resolved_tags, conflicts = self._resolve_conflicts(
            resource.tags,
            new_tags
        )
        
        result.conflicts_resolved = conflicts
        
        # Apply tags based on provider
        try:
            if resource.provider == CloudProvider.AWS:
                applied, skipped = self._apply_aws_tags(resource, resolved_tags)
            elif resource.provider == CloudProvider.GCP:
                applied, skipped = self._apply_gcp_tags(resource, resolved_tags)
            elif resource.provider == CloudProvider.AZURE:
                applied, skipped = self._apply_azure_tags(resource, resolved_tags)
            else:
                raise ValueError(f"Unsupported provider: {resource.provider}")
            
            result.tags_applied = applied
            result.tags_skipped = skipped
            
            # Update resource tags
            resource.tags.update(applied)
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Failed to apply tags to {resource.resource_id}: {e}")
        
        return result
    
    def _resolve_conflicts(
        self,
        existing_tags: Dict[str, str],
        new_tags: Dict[str, str]
    ) -> tuple[Dict[str, str], List[str]]:
        """
        Resolve conflicts between existing and new tags
        
        Args:
            existing_tags: Existing tags on resource
            new_tags: New tags to apply
            
        Returns:
            Tuple of (resolved_tags, conflicts)
        """
        resolved = {}
        conflicts = []
        
        if self.policy.conflict_resolution == TagConflictResolution.OVERWRITE:
            # New tags overwrite existing
            resolved = {**existing_tags, **new_tags}
            conflicts = [k for k in new_tags if k in existing_tags]
        
        elif self.policy.conflict_resolution == TagConflictResolution.PRESERVE:
            # Keep existing tags, only add new ones
            resolved = existing_tags.copy()
            for key, value in new_tags.items():
                if key not in resolved:
                    resolved[key] = value
                else:
                    conflicts.append(key)
        
        elif self.policy.conflict_resolution == TagConflictResolution.MERGE:
            # Merge tags, preferring new values for conflicts
            resolved = {**existing_tags, **new_tags}
            conflicts = [k for k in new_tags if k in existing_tags and existing_tags[k] != new_tags[k]]
        
        elif self.policy.conflict_resolution == TagConflictResolution.FAIL:
            # Fail if any conflicts exist
            conflicts = [k for k in new_tags if k in existing_tags]
            if conflicts:
                raise ValueError(f"Tag conflicts detected: {conflicts}")
            resolved = {**existing_tags, **new_tags}
        
        return resolved, conflicts
    
    def _apply_aws_tags(
        self,
        resource: CloudResource,
        tags: Dict[str, str]
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        """
        Apply tags to AWS resource
        
        Args:
            resource: AWS resource
            tags: Tags to apply
            
        Returns:
            Tuple of (applied_tags, skipped_tags)
        """
        # In a real implementation, this would use boto3
        # Example:
        # client = boto3.client('resourcegroupstaggingapi')
        # client.tag_resources(
        #     ResourceARNList=[resource.resource_id],
        #     Tags=tags
        # )
        
        logger.debug(f"Applying {len(tags)} tags to AWS resource {resource.resource_id}")
        
        # For now, simulate successful application
        applied = tags.copy()
        skipped = {}
        
        return applied, skipped
    
    def _apply_gcp_tags(
        self,
        resource: CloudResource,
        tags: Dict[str, str]
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        """
        Apply tags (labels) to GCP resource
        
        Args:
            resource: GCP resource
            tags: Tags to apply
            
        Returns:
            Tuple of (applied_tags, skipped_tags)
        """
        # In a real implementation, this would use google-cloud libraries
        # Note: GCP calls them "labels" not "tags"
        
        logger.debug(f"Applying {len(tags)} labels to GCP resource {resource.resource_id}")
        
        # For now, simulate successful application
        applied = tags.copy()
        skipped = {}
        
        return applied, skipped
    
    def _apply_azure_tags(
        self,
        resource: CloudResource,
        tags: Dict[str, str]
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        """
        Apply tags to Azure resource
        
        Args:
            resource: Azure resource
            tags: Tags to apply
            
        Returns:
            Tuple of (applied_tags, skipped_tags)
        """
        # In a real implementation, this would use azure-mgmt libraries
        # Example:
        # from azure.mgmt.resource import ResourceManagementClient
        # client = ResourceManagementClient(credentials, subscription_id)
        # client.tags.create_or_update_at_scope(
        #     scope=resource.resource_id,
        #     parameters={'properties': {'tags': tags}}
        # )
        
        logger.debug(f"Applying {len(tags)} tags to Azure resource {resource.resource_id}")
        
        # For now, simulate successful application
        applied = tags.copy()
        skipped = {}
        
        return applied, skipped


class TaggingEngine:
    """
    Main tagging engine that coordinates tag generation and application
    
    Requirements: 5.3
    """
    
    def __init__(self, policy: TaggingPolicy):
        """
        Initialize tagging engine with policy
        
        Args:
            policy: Tagging policy to use
        """
        self.policy = policy
        self.tag_generator = TagGenerator(policy)
        self.tag_applicator = TagApplicator(policy)
        logger.info("Tagging Engine initialized")
    
    def tag_resources(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources
    ) -> BulkTaggingResult:
        """
        Tag multiple resources based on their categorizations
        
        Args:
            resources: List of resources to tag
            categorizations: Categorizations for the resources
            
        Returns:
            BulkTaggingResult with results for all resources
        """
        logger.info(f"Starting bulk tagging for {len(resources)} resources")
        
        bulk_result = BulkTaggingResult(total_resources=len(resources))
        
        for resource in resources:
            # Get categorization for this resource
            categorization = categorizations.get_categorization(resource.resource_id)
            
            if not categorization:
                logger.warning(f"No categorization found for resource {resource.resource_id}")
                result = TaggingResult(
                    resource_id=resource.resource_id,
                    success=False,
                    errors=["No categorization found"]
                )
                bulk_result.add_result(result)
                continue
            
            # Generate tags from categorization
            tags = self.tag_generator.generate_tags(categorization)
            
            # Apply tags to resource
            result = self.tag_applicator.apply_tags(resource, tags)
            bulk_result.add_result(result)
        
        logger.info(
            f"Bulk tagging complete: {bulk_result.successful} successful, "
            f"{bulk_result.failed} failed"
        )
        
        return bulk_result
    
    def validate_tags(
        self,
        resource: CloudResource
    ) -> tuple[bool, List[str]]:
        """
        Validate that a resource has all required tags
        
        Args:
            resource: Resource to validate
            
        Returns:
            Tuple of (is_valid, missing_tags)
        """
        missing_tags = []
        
        for required_tag in self.policy.required_tags:
            formatted_key = self.tag_generator._format_tag_key(required_tag)
            if formatted_key not in resource.tags:
                missing_tags.append(required_tag)
        
        is_valid = len(missing_tags) == 0
        
        return is_valid, missing_tags
    
    def get_tagging_summary(
        self,
        result: BulkTaggingResult
    ) -> Dict[str, Any]:
        """
        Get summary of tagging operation
        
        Args:
            result: Bulk tagging result
            
        Returns:
            Dictionary with summary statistics
        """
        total_tags_applied = sum(len(r.tags_applied) for r in result.results)
        total_conflicts = sum(len(r.conflicts_resolved) for r in result.results)
        
        return {
            "total_resources": result.total_resources,
            "successful": result.successful,
            "failed": result.failed,
            "success_rate": result.successful / result.total_resources if result.total_resources > 0 else 0,
            "total_tags_applied": total_tags_applied,
            "total_conflicts_resolved": total_conflicts,
            "average_tags_per_resource": total_tags_applied / result.successful if result.successful > 0 else 0
        }
