"""
Dimension Management System for Cloud Migration Advisor

This module provides comprehensive dimension management including CRUD operations,
validation, and dimension lifecycle management.

Requirements: 6.3
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .organizational_structure_manager import (
    Dimension,
    DimensionType,
    OrganizationalStructure,
    Team,
    Project,
    Environment,
    Region,
    CostCenter
)


logger = logging.getLogger(__name__)


class DimensionValidationError(Exception):
    """Exception raised for dimension validation errors"""
    pass


@dataclass
class DimensionCreateRequest:
    """Request to create a new dimension"""
    dimension_type: DimensionType
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionUpdateRequest:
    """Request to update an existing dimension"""
    name: Optional[str] = None
    description: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DimensionDeleteResult:
    """Result of dimension deletion"""
    success: bool
    dimension_id: str
    message: str
    affected_resources: int = 0


@dataclass
class DimensionValidationResult:
    """Result of dimension validation"""
    is_valid: bool
    dimension_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DimensionHierarchy:
    """Represents a dimension hierarchy"""
    root_dimension: Dimension
    children: List['DimensionHierarchy'] = field(default_factory=list)
    depth: int = 0
    
    def get_all_descendants(self) -> List[Dimension]:
        """Get all descendant dimensions"""
        descendants = []
        for child in self.children:
            descendants.append(child.root_dimension)
            descendants.extend(child.get_all_descendants())
        return descendants


class DimensionManagementSystem:
    """
    Comprehensive dimension management system with CRUD operations and validation
    
    Requirements: 6.3
    """
    
    def __init__(self):
        """Initialize the dimension management system"""
        self._dimension_cache: Dict[str, Dimension] = {}
        logger.info("Dimension Management System initialized")
    
    def create_dimension(
        self,
        structure: OrganizationalStructure,
        request: DimensionCreateRequest
    ) -> Dimension:
        """
        Create a new dimension in the organizational structure
        
        Args:
            structure: Organizational structure to add dimension to
            request: Dimension creation request
            
        Returns:
            Created Dimension
            
        Raises:
            DimensionValidationError: If validation fails
        """
        logger.info(f"Creating dimension: {request.name} of type {request.dimension_type.value}")
        
        # Generate unique dimension ID
        dimension_id = self._generate_dimension_id(request.dimension_type, request.name)
        
        # Validate parent reference if provided
        if request.parent_id:
            self._validate_parent_reference(structure, request.parent_id, request.dimension_type)
        
        # Validate name uniqueness within dimension type
        self._validate_name_uniqueness(structure, request.name, request.dimension_type)
        
        # Create dimension based on type
        dimension = self._create_typed_dimension(
            dimension_id=dimension_id,
            request=request
        )
        
        # Add to structure
        self._add_dimension_to_structure(structure, dimension, request.dimension_type)
        
        # Cache dimension
        self._dimension_cache[dimension_id] = dimension
        
        logger.info(f"Dimension created successfully: {dimension_id}")
        return dimension
    
    def get_dimension(
        self,
        structure: OrganizationalStructure,
        dimension_id: str
    ) -> Optional[Dimension]:
        """
        Get a dimension by ID
        
        Args:
            structure: Organizational structure to search
            dimension_id: ID of dimension to retrieve
            
        Returns:
            Dimension if found, None otherwise
        """
        # Check cache first
        if dimension_id in self._dimension_cache:
            return self._dimension_cache[dimension_id]
        
        # Search in structure
        dimension = self._find_dimension_in_structure(structure, dimension_id)
        
        # Cache if found
        if dimension:
            self._dimension_cache[dimension_id] = dimension
        
        return dimension
    
    def update_dimension(
        self,
        structure: OrganizationalStructure,
        dimension_id: str,
        request: DimensionUpdateRequest
    ) -> Dimension:
        """
        Update an existing dimension
        
        Args:
            structure: Organizational structure containing the dimension
            dimension_id: ID of dimension to update
            request: Update request with new values
            
        Returns:
            Updated Dimension
            
        Raises:
            DimensionValidationError: If dimension not found or validation fails
        """
        logger.info(f"Updating dimension: {dimension_id}")
        
        # Find dimension
        dimension = self.get_dimension(structure, dimension_id)
        if not dimension:
            raise DimensionValidationError(f"Dimension not found: {dimension_id}")
        
        # Validate updates
        if request.name:
            self._validate_name_uniqueness(
                structure, 
                request.name, 
                dimension.dimension_type,
                exclude_id=dimension_id
            )
        
        if request.parent_id:
            self._validate_parent_reference(
                structure, 
                request.parent_id, 
                dimension.dimension_type
            )
            # Prevent circular references
            self._validate_no_circular_reference(
                structure, 
                dimension_id, 
                request.parent_id
            )
        
        # Apply updates
        if request.name:
            dimension.name = request.name
        if request.description is not None:
            dimension.description = request.description
        if request.parent_id is not None:
            dimension.parent_id = request.parent_id
        if request.metadata is not None:
            dimension.metadata.update(request.metadata)
        
        # Update structure timestamp
        structure.updated_at = datetime.utcnow()
        
        # Update cache
        self._dimension_cache[dimension_id] = dimension
        
        logger.info(f"Dimension updated successfully: {dimension_id}")
        return dimension
    
    def delete_dimension(
        self,
        structure: OrganizationalStructure,
        dimension_id: str,
        cascade: bool = False
    ) -> DimensionDeleteResult:
        """
        Delete a dimension from the organizational structure
        
        Args:
            structure: Organizational structure containing the dimension
            dimension_id: ID of dimension to delete
            cascade: If True, delete child dimensions as well
            
        Returns:
            DimensionDeleteResult with deletion status
            
        Raises:
            DimensionValidationError: If dimension has children and cascade is False
        """
        logger.info(f"Deleting dimension: {dimension_id} (cascade={cascade})")
        
        # Find dimension
        dimension = self.get_dimension(structure, dimension_id)
        if not dimension:
            return DimensionDeleteResult(
                success=False,
                dimension_id=dimension_id,
                message=f"Dimension not found: {dimension_id}"
            )
        
        # Check for children
        children = self._find_child_dimensions(structure, dimension_id)
        if children and not cascade:
            raise DimensionValidationError(
                f"Dimension {dimension_id} has {len(children)} child dimensions. "
                "Use cascade=True to delete children as well."
            )
        
        # Delete children if cascade
        affected_count = 0
        if cascade and children:
            for child in children:
                child_result = self.delete_dimension(structure, child.dimension_id, cascade=True)
                affected_count += child_result.affected_resources
        
        # Remove from structure
        self._remove_dimension_from_structure(structure, dimension_id, dimension.dimension_type)
        
        # Remove from cache
        if dimension_id in self._dimension_cache:
            del self._dimension_cache[dimension_id]
        
        # Update structure timestamp
        structure.updated_at = datetime.utcnow()
        
        affected_count += 1
        
        logger.info(f"Dimension deleted successfully: {dimension_id}")
        return DimensionDeleteResult(
            success=True,
            dimension_id=dimension_id,
            message="Dimension deleted successfully",
            affected_resources=affected_count
        )
    
    def list_dimensions(
        self,
        structure: OrganizationalStructure,
        dimension_type: Optional[DimensionType] = None,
        parent_id: Optional[str] = None
    ) -> List[Dimension]:
        """
        List dimensions with optional filtering
        
        Args:
            structure: Organizational structure to search
            dimension_type: Optional filter by dimension type
            parent_id: Optional filter by parent ID
            
        Returns:
            List of matching dimensions
        """
        dimensions = self._get_all_dimensions_from_structure(structure)
        
        # Apply filters
        if dimension_type:
            dimensions = [d for d in dimensions if d.dimension_type == dimension_type]
        
        if parent_id is not None:
            dimensions = [d for d in dimensions if d.parent_id == parent_id]
        
        return dimensions
    
    def validate_dimension(
        self,
        structure: OrganizationalStructure,
        dimension_id: str
    ) -> DimensionValidationResult:
        """
        Validate a dimension for consistency and correctness
        
        Args:
            structure: Organizational structure containing the dimension
            dimension_id: ID of dimension to validate
            
        Returns:
            DimensionValidationResult with validation status
        """
        logger.debug(f"Validating dimension: {dimension_id}")
        
        errors = []
        warnings = []
        
        # Check if dimension exists
        dimension = self.get_dimension(structure, dimension_id)
        if not dimension:
            return DimensionValidationResult(
                is_valid=False,
                dimension_id=dimension_id,
                errors=[f"Dimension not found: {dimension_id}"]
            )
        
        # Validate name
        if not dimension.name or not dimension.name.strip():
            errors.append("Dimension name cannot be empty")
        
        # Validate parent reference
        if dimension.parent_id:
            parent = self.get_dimension(structure, dimension.parent_id)
            if not parent:
                errors.append(f"Parent dimension not found: {dimension.parent_id}")
            elif parent.dimension_type != dimension.dimension_type:
                errors.append(
                    f"Parent dimension type mismatch: expected {dimension.dimension_type.value}, "
                    f"got {parent.dimension_type.value}"
                )
        
        # Check for circular references
        if dimension.parent_id:
            try:
                self._validate_no_circular_reference(structure, dimension_id, dimension.parent_id)
            except DimensionValidationError as e:
                errors.append(str(e))
        
        # Check for orphaned children if this is being deleted
        children = self._find_child_dimensions(structure, dimension_id)
        if children:
            warnings.append(f"Dimension has {len(children)} child dimensions")
        
        is_valid = len(errors) == 0
        
        return DimensionValidationResult(
            is_valid=is_valid,
            dimension_id=dimension_id,
            errors=errors,
            warnings=warnings
        )
    
    def build_dimension_hierarchy(
        self,
        structure: OrganizationalStructure,
        dimension_type: DimensionType,
        root_id: Optional[str] = None
    ) -> List[DimensionHierarchy]:
        """
        Build hierarchical view of dimensions
        
        Args:
            structure: Organizational structure
            dimension_type: Type of dimensions to include
            root_id: Optional root dimension ID (if None, returns all root dimensions)
            
        Returns:
            List of DimensionHierarchy trees
        """
        logger.debug(f"Building dimension hierarchy for type: {dimension_type.value}")
        
        # Get all dimensions of this type
        dimensions = self.list_dimensions(structure, dimension_type=dimension_type)
        
        # Build hierarchy
        if root_id:
            root = self.get_dimension(structure, root_id)
            if not root:
                return []
            return [self._build_hierarchy_recursive(structure, root, dimensions)]
        else:
            # Find all root dimensions (no parent)
            roots = [d for d in dimensions if not d.parent_id]
            return [self._build_hierarchy_recursive(structure, root, dimensions) for root in roots]
    
    def get_dimension_path(
        self,
        structure: OrganizationalStructure,
        dimension_id: str
    ) -> List[Dimension]:
        """
        Get the path from root to the specified dimension
        
        Args:
            structure: Organizational structure
            dimension_id: ID of dimension
            
        Returns:
            List of dimensions from root to target (inclusive)
        """
        path = []
        current = self.get_dimension(structure, dimension_id)
        
        while current:
            path.insert(0, current)
            if current.parent_id:
                current = self.get_dimension(structure, current.parent_id)
            else:
                break
        
        return path
    
    # Private helper methods
    
    def _generate_dimension_id(self, dimension_type: DimensionType, name: str) -> str:
        """Generate unique dimension ID"""
        import hashlib
        timestamp = datetime.utcnow().isoformat()
        unique_string = f"{dimension_type.value}_{name}_{timestamp}"
        hash_suffix = hashlib.md5(unique_string.encode()).hexdigest()[:8]
        return f"{dimension_type.value}_{hash_suffix}"
    
    def _validate_parent_reference(
        self,
        structure: OrganizationalStructure,
        parent_id: str,
        dimension_type: DimensionType
    ) -> None:
        """Validate parent dimension exists and has correct type"""
        parent = self.get_dimension(structure, parent_id)
        if not parent:
            raise DimensionValidationError(f"Parent dimension not found: {parent_id}")
        
        if parent.dimension_type != dimension_type:
            raise DimensionValidationError(
                f"Parent dimension type mismatch: expected {dimension_type.value}, "
                f"got {parent.dimension_type.value}"
            )
    
    def _validate_name_uniqueness(
        self,
        structure: OrganizationalStructure,
        name: str,
        dimension_type: DimensionType,
        exclude_id: Optional[str] = None
    ) -> None:
        """Validate dimension name is unique within its type"""
        dimensions = self.list_dimensions(structure, dimension_type=dimension_type)
        for dim in dimensions:
            if dim.name == name and dim.dimension_id != exclude_id:
                raise DimensionValidationError(
                    f"Dimension name '{name}' already exists for type {dimension_type.value}"
                )
    
    def _validate_no_circular_reference(
        self,
        structure: OrganizationalStructure,
        dimension_id: str,
        parent_id: str
    ) -> None:
        """Validate no circular reference in parent chain"""
        visited = set()
        current_id = parent_id
        
        while current_id:
            if current_id == dimension_id:
                raise DimensionValidationError(
                    f"Circular reference detected: dimension {dimension_id} cannot be its own ancestor"
                )
            
            if current_id in visited:
                raise DimensionValidationError(
                    f"Circular reference detected in parent chain"
                )
            
            visited.add(current_id)
            current = self.get_dimension(structure, current_id)
            if not current:
                break
            current_id = current.parent_id
    
    def _create_typed_dimension(
        self,
        dimension_id: str,
        request: DimensionCreateRequest
    ) -> Dimension:
        """Create a dimension object based on type"""
        return Dimension(
            dimension_id=dimension_id,
            dimension_type=request.dimension_type,
            name=request.name,
            description=request.description,
            parent_id=request.parent_id,
            metadata=request.metadata
        )
    
    def _add_dimension_to_structure(
        self,
        structure: OrganizationalStructure,
        dimension: Dimension,
        dimension_type: DimensionType
    ) -> None:
        """Add dimension to appropriate structure collection"""
        # For standard dimension types, we don't add to custom_dimensions
        # They are managed through their specific collections (teams, projects, etc.)
        # This method is primarily for custom dimensions
        if dimension_type == DimensionType.CUSTOM:
            category = dimension.metadata.get('category', 'general')
            if category not in structure.custom_dimensions:
                structure.custom_dimensions[category] = []
            structure.custom_dimensions[category].append(dimension)
        
        structure.updated_at = datetime.utcnow()
    
    def _remove_dimension_from_structure(
        self,
        structure: OrganizationalStructure,
        dimension_id: str,
        dimension_type: DimensionType
    ) -> None:
        """Remove dimension from structure"""
        if dimension_type == DimensionType.CUSTOM:
            for category, dimensions in structure.custom_dimensions.items():
                structure.custom_dimensions[category] = [
                    d for d in dimensions if d.dimension_id != dimension_id
                ]
        
        structure.updated_at = datetime.utcnow()
    
    def _find_dimension_in_structure(
        self,
        structure: OrganizationalStructure,
        dimension_id: str
    ) -> Optional[Dimension]:
        """Find dimension in structure by ID"""
        # Search in custom dimensions
        for dimensions in structure.custom_dimensions.values():
            for dim in dimensions:
                if dim.dimension_id == dimension_id:
                    return dim
        
        return None
    
    def _get_all_dimensions_from_structure(
        self,
        structure: OrganizationalStructure
    ) -> List[Dimension]:
        """Get all dimensions from structure"""
        dimensions = []
        
        # Get custom dimensions
        for dims in structure.custom_dimensions.values():
            dimensions.extend(dims)
        
        return dimensions
    
    def _find_child_dimensions(
        self,
        structure: OrganizationalStructure,
        parent_id: str
    ) -> List[Dimension]:
        """Find all child dimensions of a parent"""
        return self.list_dimensions(structure, parent_id=parent_id)
    
    def _build_hierarchy_recursive(
        self,
        structure: OrganizationalStructure,
        root: Dimension,
        all_dimensions: List[Dimension],
        depth: int = 0
    ) -> DimensionHierarchy:
        """Recursively build dimension hierarchy"""
        hierarchy = DimensionHierarchy(
            root_dimension=root,
            depth=depth
        )
        
        # Find children
        children = [d for d in all_dimensions if d.parent_id == root.dimension_id]
        
        # Recursively build child hierarchies
        for child in children:
            child_hierarchy = self._build_hierarchy_recursive(
                structure, child, all_dimensions, depth + 1
            )
            hierarchy.children.append(child_hierarchy)
        
        return hierarchy
