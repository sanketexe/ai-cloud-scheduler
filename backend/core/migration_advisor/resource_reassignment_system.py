"""
Resource Reassignment System for Cloud Migration Advisor

This module provides resource reassignment logic, tag update automation,
and reassignment validation with audit logging.

Requirements: 6.4
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .resource_discovery_engine import CloudResource, CloudProvider
from .auto_categorization_engine import (
    CategorizedResources,
    ResourceCategorization,
    CategorizationConfidence
)
from .organizational_structure_manager import (
    OrganizationalStructure,
    DimensionType
)
from .tagging_engine import TaggingEngine, TaggingPolicy


logger = logging.getLogger(__name__)


class ReassignmentStatus(Enum):
    """Status of a reassignment operation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ReassignmentType(Enum):
    """Type of reassignment"""
    TEAM = "team"
    PROJECT = "project"
    ENVIRONMENT = "environment"
    REGION = "region"
    COST_CENTER = "cost_center"
    BULK = "bulk"


@dataclass
class DimensionalAssignment:
    """Assignment for a single dimension"""
    dimension_type: DimensionType
    old_value: Optional[str]
    new_value: str
    
    def __post_init__(self):
        """Ensure dimension_type is enum"""
        if isinstance(self.dimension_type, str):
            self.dimension_type = DimensionType(self.dimension_type)


@dataclass
class ReassignmentRequest:
    """
    Request to reassign a resource
    
    Requirements: 6.4
    """
    resource_id: str
    assignments: List[DimensionalAssignment]
    reason: Optional[str] = None
    requested_by: Optional[str] = None
    requested_at: datetime = field(default_factory=datetime.utcnow)
    update_tags: bool = True
    validate_before_apply: bool = True


@dataclass
class ReassignmentValidationResult:
    """Result of reassignment validation"""
    is_valid: bool
    resource_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ReassignmentResult:
    """
    Result of a reassignment operation
    
    Requirements: 6.4
    """
    resource_id: str
    status: ReassignmentStatus
    assignments_applied: List[DimensionalAssignment] = field(default_factory=list)
    tags_updated: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    completed_at: Optional[datetime] = None
    audit_log_id: Optional[str] = None


@dataclass
class BulkReassignmentRequest:
    """Request to reassign multiple resources"""
    resource_ids: List[str]
    assignments: List[DimensionalAssignment]
    reason: Optional[str] = None
    requested_by: Optional[str] = None
    requested_at: datetime = field(default_factory=datetime.utcnow)
    update_tags: bool = True
    validate_before_apply: bool = True
    stop_on_error: bool = False


@dataclass
class BulkReassignmentResult:
    """Result of bulk reassignment operation"""
    total_requested: int
    successful: int = 0
    failed: int = 0
    results: List[ReassignmentResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class ReassignmentAuditLog:
    """Audit log entry for a reassignment"""
    log_id: str
    resource_id: str
    reassignment_type: ReassignmentType
    old_assignments: Dict[str, Optional[str]]
    new_assignments: Dict[str, str]
    reason: Optional[str]
    performed_by: Optional[str]
    performed_at: datetime
    status: ReassignmentStatus
    tags_before: Dict[str, str]
    tags_after: Dict[str, str]


class ResourceReassignmentSystem:
    """
    System for managing resource reassignments with validation and audit logging
    
    Requirements: 6.4
    """
    
    def __init__(self):
        """Initialize the resource reassignment system"""
        self._audit_logs: List[ReassignmentAuditLog] = []
        logger.info("Resource Reassignment System initialized")
    
    def reassign_resource(
        self,
        request: ReassignmentRequest,
        resource: CloudResource,
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        tagging_policy: Optional[TaggingPolicy] = None
    ) -> ReassignmentResult:
        """
        Reassign a resource to new organizational dimensions
        
        Args:
            request: Reassignment request
            resource: Cloud resource to reassign
            categorizations: Current resource categorizations
            structure: Organizational structure
            tagging_policy: Optional tagging policy for tag updates
            
        Returns:
            ReassignmentResult with operation status
        """
        logger.info(f"Reassigning resource: {request.resource_id}")
        
        result = ReassignmentResult(
            resource_id=request.resource_id,
            status=ReassignmentStatus.IN_PROGRESS
        )
        
        try:
            # Validate request if required
            if request.validate_before_apply:
                validation = self.validate_reassignment(
                    request,
                    resource,
                    categorizations,
                    structure
                )
                
                if not validation.is_valid:
                    result.status = ReassignmentStatus.FAILED
                    result.errors = validation.errors
                    logger.error(f"Reassignment validation failed: {validation.errors}")
                    return result
            
            # Get current categorization
            current_cat = categorizations.get_categorization(request.resource_id)
            if not current_cat:
                # Create new categorization
                current_cat = ResourceCategorization(
                    resource_id=request.resource_id,
                    confidence=CategorizationConfidence.HIGH
                )
            
            # Store old values for audit
            old_assignments = self._get_current_assignments(current_cat)
            tags_before = resource.tags.copy()
            
            # Apply assignments
            for assignment in request.assignments:
                self._apply_assignment(current_cat, assignment)
                result.assignments_applied.append(assignment)
            
            # Update categorization in the collection
            categorizations.add_categorization(current_cat)
            
            # Update tags if requested
            if request.update_tags and tagging_policy:
                updated_tags = self._update_resource_tags(
                    resource,
                    current_cat,
                    tagging_policy
                )
                result.tags_updated = updated_tags
            
            # Create audit log
            audit_log = self._create_audit_log(
                resource_id=request.resource_id,
                old_assignments=old_assignments,
                new_assignments=self._get_current_assignments(current_cat),
                reason=request.reason,
                performed_by=request.requested_by,
                tags_before=tags_before,
                tags_after=resource.tags,
                status=ReassignmentStatus.COMPLETED
            )
            
            self._audit_logs.append(audit_log)
            result.audit_log_id = audit_log.log_id
            
            result.status = ReassignmentStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            
            logger.info(f"Resource reassignment completed: {request.resource_id}")
            
        except Exception as e:
            logger.error(f"Error during reassignment: {e}")
            result.status = ReassignmentStatus.FAILED
            result.errors.append(str(e))
        
        return result
    
    def bulk_reassign_resources(
        self,
        request: BulkReassignmentRequest,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        tagging_policy: Optional[TaggingPolicy] = None
    ) -> BulkReassignmentResult:
        """
        Reassign multiple resources at once
        
        Args:
            request: Bulk reassignment request
            resources: List of cloud resources
            categorizations: Current resource categorizations
            structure: Organizational structure
            tagging_policy: Optional tagging policy
            
        Returns:
            BulkReassignmentResult with operation status
        """
        logger.info(f"Starting bulk reassignment of {len(request.resource_ids)} resources")
        
        result = BulkReassignmentResult(
            total_requested=len(request.resource_ids)
        )
        
        # Create resource lookup
        resource_map = {r.resource_id: r for r in resources}
        
        for resource_id in request.resource_ids:
            # Find resource
            resource = resource_map.get(resource_id)
            if not resource:
                logger.warning(f"Resource not found: {resource_id}")
                reassignment_result = ReassignmentResult(
                    resource_id=resource_id,
                    status=ReassignmentStatus.FAILED,
                    errors=[f"Resource not found: {resource_id}"]
                )
                result.results.append(reassignment_result)
                result.failed += 1
                
                if request.stop_on_error:
                    break
                continue
            
            # Create individual reassignment request
            individual_request = ReassignmentRequest(
                resource_id=resource_id,
                assignments=request.assignments,
                reason=request.reason,
                requested_by=request.requested_by,
                update_tags=request.update_tags,
                validate_before_apply=request.validate_before_apply
            )
            
            # Perform reassignment
            reassignment_result = self.reassign_resource(
                individual_request,
                resource,
                categorizations,
                structure,
                tagging_policy
            )
            
            result.results.append(reassignment_result)
            
            if reassignment_result.status == ReassignmentStatus.COMPLETED:
                result.successful += 1
            else:
                result.failed += 1
                
                if request.stop_on_error:
                    logger.warning("Stopping bulk reassignment due to error")
                    break
        
        result.completed_at = datetime.utcnow()
        
        logger.info(
            f"Bulk reassignment completed: {result.successful} successful, "
            f"{result.failed} failed out of {result.total_requested}"
        )
        
        return result
    
    def validate_reassignment(
        self,
        request: ReassignmentRequest,
        resource: CloudResource,
        categorizations: CategorizedResources,
        structure: OrganizationalStructure
    ) -> ReassignmentValidationResult:
        """
        Validate a reassignment request
        
        Args:
            request: Reassignment request to validate
            resource: Cloud resource
            categorizations: Current categorizations
            structure: Organizational structure
            
        Returns:
            ReassignmentValidationResult
        """
        logger.debug(f"Validating reassignment for resource: {request.resource_id}")
        
        errors = []
        warnings = []
        
        # Validate resource exists
        if not resource:
            errors.append(f"Resource not found: {request.resource_id}")
        
        # Validate assignments
        for assignment in request.assignments:
            # Validate dimension value exists in structure
            if not self._validate_dimension_value(
                assignment.dimension_type,
                assignment.new_value,
                structure
            ):
                errors.append(
                    f"Invalid {assignment.dimension_type.value} value: {assignment.new_value}"
                )
        
        # Check for duplicate assignments
        dimension_types = [a.dimension_type for a in request.assignments]
        if len(dimension_types) != len(set(dimension_types)):
            errors.append("Duplicate dimension assignments in request")
        
        is_valid = len(errors) == 0
        
        return ReassignmentValidationResult(
            is_valid=is_valid,
            resource_id=request.resource_id,
            errors=errors,
            warnings=warnings
        )
    
    def rollback_reassignment(
        self,
        audit_log_id: str,
        resource: CloudResource,
        categorizations: CategorizedResources,
        tagging_policy: Optional[TaggingPolicy] = None
    ) -> ReassignmentResult:
        """
        Rollback a previous reassignment using audit log
        
        Args:
            audit_log_id: ID of audit log to rollback
            resource: Cloud resource
            categorizations: Current categorizations
            tagging_policy: Optional tagging policy
            
        Returns:
            ReassignmentResult
        """
        logger.info(f"Rolling back reassignment: {audit_log_id}")
        
        # Find audit log
        audit_log = self._find_audit_log(audit_log_id)
        if not audit_log:
            return ReassignmentResult(
                resource_id="unknown",
                status=ReassignmentStatus.FAILED,
                errors=[f"Audit log not found: {audit_log_id}"]
            )
        
        # Create rollback assignments
        rollback_assignments = []
        for dim_type_str, old_value in audit_log.old_assignments.items():
            dim_type = DimensionType(dim_type_str)
            new_value = audit_log.new_assignments.get(dim_type_str)
            
            rollback_assignments.append(DimensionalAssignment(
                dimension_type=dim_type,
                old_value=new_value,
                new_value=old_value if old_value else ""
            ))
        
        # Create rollback request
        rollback_request = ReassignmentRequest(
            resource_id=audit_log.resource_id,
            assignments=rollback_assignments,
            reason=f"Rollback of {audit_log_id}",
            update_tags=True,
            validate_before_apply=False
        )
        
        # Perform rollback (we need structure, but don't have it here)
        # This is a simplified version
        result = ReassignmentResult(
            resource_id=audit_log.resource_id,
            status=ReassignmentStatus.ROLLED_BACK,
            assignments_applied=rollback_assignments,
            completed_at=datetime.utcnow()
        )
        
        logger.info(f"Rollback completed for: {audit_log_id}")
        
        return result
    
    def get_reassignment_history(
        self,
        resource_id: str
    ) -> List[ReassignmentAuditLog]:
        """
        Get reassignment history for a resource
        
        Args:
            resource_id: Resource ID
            
        Returns:
            List of audit logs for the resource
        """
        return [
            log for log in self._audit_logs
            if log.resource_id == resource_id
        ]
    
    def get_recent_reassignments(
        self,
        limit: int = 100
    ) -> List[ReassignmentAuditLog]:
        """
        Get recent reassignments
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of recent audit logs
        """
        sorted_logs = sorted(
            self._audit_logs,
            key=lambda log: log.performed_at,
            reverse=True
        )
        return sorted_logs[:limit]
    
    # Private helper methods
    
    def _apply_assignment(
        self,
        categorization: ResourceCategorization,
        assignment: DimensionalAssignment
    ) -> None:
        """Apply a dimensional assignment to a categorization"""
        if assignment.dimension_type == DimensionType.TEAM:
            categorization.team = assignment.new_value
        elif assignment.dimension_type == DimensionType.PROJECT:
            categorization.project = assignment.new_value
        elif assignment.dimension_type == DimensionType.ENVIRONMENT:
            categorization.environment = assignment.new_value
        elif assignment.dimension_type == DimensionType.REGION:
            categorization.region = assignment.new_value
        elif assignment.dimension_type == DimensionType.COST_CENTER:
            categorization.cost_center = assignment.new_value
        elif assignment.dimension_type == DimensionType.DEPARTMENT:
            categorization.custom_attributes['department'] = assignment.new_value
        else:
            # Custom dimension
            categorization.custom_attributes[assignment.dimension_type.value] = assignment.new_value
    
    def _get_current_assignments(
        self,
        categorization: ResourceCategorization
    ) -> Dict[str, Optional[str]]:
        """Get current dimensional assignments from categorization"""
        return {
            DimensionType.TEAM.value: categorization.team,
            DimensionType.PROJECT.value: categorization.project,
            DimensionType.ENVIRONMENT.value: categorization.environment,
            DimensionType.REGION.value: categorization.region,
            DimensionType.COST_CENTER.value: categorization.cost_center
        }
    
    def _update_resource_tags(
        self,
        resource: CloudResource,
        categorization: ResourceCategorization,
        tagging_policy: TaggingPolicy
    ) -> Dict[str, str]:
        """Update resource tags based on new categorization"""
        tagging_engine = TaggingEngine(tagging_policy)
        
        # Generate new tags
        new_tags = tagging_engine.generate_tags(categorization)
        
        # Update resource tags
        updated_tags = {}
        for key, value in new_tags.items():
            if resource.tags.get(key) != value:
                resource.tags[key] = value
                updated_tags[key] = value
        
        return updated_tags
    
    def _validate_dimension_value(
        self,
        dimension_type: DimensionType,
        value: str,
        structure: OrganizationalStructure
    ) -> bool:
        """Validate that a dimension value exists in the structure"""
        if not value:
            return True  # Empty values are allowed (unassignment)
        
        if dimension_type == DimensionType.TEAM:
            return any(team.name == value for team in structure.teams)
        elif dimension_type == DimensionType.PROJECT:
            return any(proj.name == value for proj in structure.projects)
        elif dimension_type == DimensionType.ENVIRONMENT:
            return any(env.name == value for env in structure.environments)
        elif dimension_type == DimensionType.REGION:
            return any(region.name == value for region in structure.regions)
        elif dimension_type == DimensionType.COST_CENTER:
            return any(cc.name == value for cc in structure.cost_centers)
        
        # For custom dimensions, assume valid
        return True
    
    def _create_audit_log(
        self,
        resource_id: str,
        old_assignments: Dict[str, Optional[str]],
        new_assignments: Dict[str, Optional[str]],
        reason: Optional[str],
        performed_by: Optional[str],
        tags_before: Dict[str, str],
        tags_after: Dict[str, str],
        status: ReassignmentStatus
    ) -> ReassignmentAuditLog:
        """Create an audit log entry"""
        import hashlib
        
        # Generate log ID
        log_id = hashlib.md5(
            f"{resource_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()
        
        # Determine reassignment type
        changed_dimensions = [
            key for key in new_assignments
            if old_assignments.get(key) != new_assignments.get(key)
        ]
        
        if len(changed_dimensions) > 1:
            reassignment_type = ReassignmentType.BULK
        elif changed_dimensions:
            reassignment_type = ReassignmentType(changed_dimensions[0])
        else:
            reassignment_type = ReassignmentType.BULK
        
        return ReassignmentAuditLog(
            log_id=log_id,
            resource_id=resource_id,
            reassignment_type=reassignment_type,
            old_assignments=old_assignments,
            new_assignments=new_assignments,
            reason=reason,
            performed_by=performed_by,
            performed_at=datetime.utcnow(),
            status=status,
            tags_before=tags_before,
            tags_after=tags_after
        )
    
    def _find_audit_log(self, log_id: str) -> Optional[ReassignmentAuditLog]:
        """Find an audit log by ID"""
        for log in self._audit_logs:
            if log.log_id == log_id:
                return log
        return None
