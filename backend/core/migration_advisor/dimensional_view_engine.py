"""
Dimensional View Engine for Cloud Migration Advisor

This module provides dimensional view generation, resource grouping by dimension,
and aggregated view calculations.

Requirements: 6.1
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

from .organizational_structure_manager import (
    DimensionType,
    OrganizationalStructure
)
from .resource_discovery_engine import CloudResource
from .auto_categorization_engine import CategorizedResources, ResourceCategorization


logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Types of aggregations for dimensional views"""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"


@dataclass
class DimensionalViewNode:
    """Represents a node in a dimensional view"""
    dimension_value: str
    dimension_type: DimensionType
    resources: List[CloudResource] = field(default_factory=list)
    resource_count: int = 0
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)
    children: List['DimensionalViewNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_resource(self, resource: CloudResource) -> None:
        """Add a resource to this node"""
        self.resources.append(resource)
        self.resource_count += 1
    
    def get_all_resources(self) -> List[CloudResource]:
        """Get all resources including from children"""
        all_resources = list(self.resources)
        for child in self.children:
            all_resources.extend(child.get_all_resources())
        return all_resources
    
    def get_total_count(self) -> int:
        """Get total resource count including children"""
        total = self.resource_count
        for child in self.children:
            total += child.get_total_count()
        return total


@dataclass
class DimensionalView:
    """
    Complete dimensional view of resources
    
    Requirements: 6.1
    """
    dimension_type: DimensionType
    root_nodes: List[DimensionalViewNode] = field(default_factory=list)
    total_resources: int = 0
    generated_at: datetime = field(default_factory=datetime.utcnow)
    aggregations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_nodes(self) -> List[DimensionalViewNode]:
        """Get all nodes in the view"""
        nodes = []
        for root in self.root_nodes:
            nodes.append(root)
            nodes.extend(self._get_descendants(root))
        return nodes
    
    def _get_descendants(self, node: DimensionalViewNode) -> List[DimensionalViewNode]:
        """Recursively get all descendant nodes"""
        descendants = []
        for child in node.children:
            descendants.append(child)
            descendants.extend(self._get_descendants(child))
        return descendants
    
    def find_node(self, dimension_value: str) -> Optional[DimensionalViewNode]:
        """Find a node by dimension value"""
        for node in self.get_all_nodes():
            if node.dimension_value == dimension_value:
                return node
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the dimensional view"""
        return {
            "dimension_type": self.dimension_type.value,
            "total_resources": self.total_resources,
            "root_nodes_count": len(self.root_nodes),
            "total_nodes_count": len(self.get_all_nodes()),
            "generated_at": self.generated_at.isoformat(),
            "aggregations": self.aggregations
        }


@dataclass
class ViewGenerationOptions:
    """Options for view generation"""
    include_uncategorized: bool = True
    include_empty_nodes: bool = False
    max_depth: Optional[int] = None
    aggregations: List[AggregationType] = field(default_factory=list)
    custom_aggregations: Dict[str, Callable] = field(default_factory=dict)


class DimensionalViewEngine:
    """
    Engine for generating dimensional views of resources
    
    Requirements: 6.1
    """
    
    def __init__(self):
        """Initialize the dimensional view engine"""
        logger.info("Dimensional View Engine initialized")
    
    def generate_view(
        self,
        dimension_type: DimensionType,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        options: Optional[ViewGenerationOptions] = None
    ) -> DimensionalView:
        """
        Generate a dimensional view for the specified dimension type
        
        Args:
            dimension_type: Type of dimension to view by
            resources: List of cloud resources
            categorizations: Resource categorizations
            structure: Organizational structure
            options: Optional view generation options
            
        Returns:
            DimensionalView organized by the specified dimension
        """
        logger.info(f"Generating dimensional view for: {dimension_type.value}")
        
        if options is None:
            options = ViewGenerationOptions()
        
        # Group resources by dimension value
        grouped_resources = self._group_resources_by_dimension(
            dimension_type,
            resources,
            categorizations,
            options.include_uncategorized
        )
        
        # Create view nodes
        root_nodes = []
        total_resources = 0
        
        for dimension_value, resource_list in grouped_resources.items():
            if not resource_list and not options.include_empty_nodes:
                continue
            
            node = DimensionalViewNode(
                dimension_value=dimension_value,
                dimension_type=dimension_type,
                resources=resource_list,
                resource_count=len(resource_list)
            )
            
            # Calculate aggregations for this node
            if options.aggregations or options.custom_aggregations:
                node.aggregated_metrics = self._calculate_aggregations(
                    resource_list,
                    options.aggregations,
                    options.custom_aggregations
                )
            
            root_nodes.append(node)
            total_resources += len(resource_list)
        
        # Sort nodes by resource count (descending)
        root_nodes.sort(key=lambda n: n.resource_count, reverse=True)
        
        # Create view
        view = DimensionalView(
            dimension_type=dimension_type,
            root_nodes=root_nodes,
            total_resources=total_resources
        )
        
        # Calculate global aggregations
        if options.aggregations or options.custom_aggregations:
            view.aggregations = self._calculate_aggregations(
                resources,
                options.aggregations,
                options.custom_aggregations
            )
        
        logger.info(
            f"Generated view with {len(root_nodes)} nodes and {total_resources} resources"
        )
        
        return view
    
    def generate_multi_dimensional_view(
        self,
        dimensions: List[DimensionType],
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        options: Optional[ViewGenerationOptions] = None
    ) -> DimensionalView:
        """
        Generate a multi-dimensional hierarchical view
        
        Args:
            dimensions: List of dimensions in hierarchy order (top to bottom)
            resources: List of cloud resources
            categorizations: Resource categorizations
            structure: Organizational structure
            options: Optional view generation options
            
        Returns:
            DimensionalView with nested hierarchy
        """
        logger.info(f"Generating multi-dimensional view with {len(dimensions)} levels")
        
        if not dimensions:
            raise ValueError("At least one dimension must be specified")
        
        if options is None:
            options = ViewGenerationOptions()
        
        # Start with the first dimension
        primary_dimension = dimensions[0]
        view = self.generate_view(
            primary_dimension,
            resources,
            categorizations,
            structure,
            options
        )
        
        # If there are more dimensions, build nested hierarchy
        if len(dimensions) > 1:
            for node in view.root_nodes:
                self._build_nested_hierarchy(
                    node,
                    dimensions[1:],
                    categorizations,
                    structure,
                    options
                )
        
        return view
    
    def generate_view_by_team(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        options: Optional[ViewGenerationOptions] = None
    ) -> DimensionalView:
        """Generate view grouped by team"""
        return self.generate_view(
            DimensionType.TEAM,
            resources,
            categorizations,
            structure,
            options
        )
    
    def generate_view_by_project(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        options: Optional[ViewGenerationOptions] = None
    ) -> DimensionalView:
        """Generate view grouped by project"""
        return self.generate_view(
            DimensionType.PROJECT,
            resources,
            categorizations,
            structure,
            options
        )
    
    def generate_view_by_environment(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        options: Optional[ViewGenerationOptions] = None
    ) -> DimensionalView:
        """Generate view grouped by environment"""
        return self.generate_view(
            DimensionType.ENVIRONMENT,
            resources,
            categorizations,
            structure,
            options
        )
    
    def generate_view_by_region(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        options: Optional[ViewGenerationOptions] = None
    ) -> DimensionalView:
        """Generate view grouped by region"""
        return self.generate_view(
            DimensionType.REGION,
            resources,
            categorizations,
            structure,
            options
        )
    
    def generate_view_by_cost_center(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        options: Optional[ViewGenerationOptions] = None
    ) -> DimensionalView:
        """Generate view grouped by cost center"""
        return self.generate_view(
            DimensionType.COST_CENTER,
            resources,
            categorizations,
            structure,
            options
        )
    
    def calculate_view_aggregations(
        self,
        view: DimensionalView,
        aggregation_functions: Dict[str, Callable[[List[CloudResource]], Any]]
    ) -> DimensionalView:
        """
        Calculate custom aggregations for an existing view
        
        Args:
            view: Dimensional view to add aggregations to
            aggregation_functions: Dictionary of aggregation name to function
            
        Returns:
            Updated view with aggregations
        """
        logger.debug(f"Calculating {len(aggregation_functions)} aggregations for view")
        
        # Calculate for each node
        for node in view.get_all_nodes():
            for agg_name, agg_func in aggregation_functions.items():
                try:
                    node.aggregated_metrics[agg_name] = agg_func(node.resources)
                except Exception as e:
                    logger.error(f"Error calculating aggregation {agg_name}: {e}")
                    node.aggregated_metrics[agg_name] = None
        
        # Calculate global aggregations
        all_resources = []
        for node in view.root_nodes:
            all_resources.extend(node.get_all_resources())
        
        for agg_name, agg_func in aggregation_functions.items():
            try:
                view.aggregations[agg_name] = agg_func(all_resources)
            except Exception as e:
                logger.error(f"Error calculating global aggregation {agg_name}: {e}")
                view.aggregations[agg_name] = None
        
        return view
    
    def filter_view(
        self,
        view: DimensionalView,
        filter_func: Callable[[DimensionalViewNode], bool]
    ) -> DimensionalView:
        """
        Filter a view to include only nodes matching the filter function
        
        Args:
            view: Dimensional view to filter
            filter_func: Function that returns True for nodes to include
            
        Returns:
            New filtered view
        """
        logger.debug("Filtering dimensional view")
        
        filtered_nodes = []
        total_resources = 0
        
        for node in view.root_nodes:
            if filter_func(node):
                filtered_nodes.append(node)
                total_resources += node.get_total_count()
        
        filtered_view = DimensionalView(
            dimension_type=view.dimension_type,
            root_nodes=filtered_nodes,
            total_resources=total_resources,
            aggregations=view.aggregations.copy(),
            metadata=view.metadata.copy()
        )
        
        return filtered_view
    
    # Private helper methods
    
    def _group_resources_by_dimension(
        self,
        dimension_type: DimensionType,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        include_uncategorized: bool
    ) -> Dict[str, List[CloudResource]]:
        """Group resources by dimension value"""
        grouped: Dict[str, List[CloudResource]] = defaultdict(list)
        
        for resource in resources:
            # Get categorization for this resource
            categorization = categorizations.get_categorization(resource.resource_id)
            
            if not categorization:
                if include_uncategorized:
                    grouped["Uncategorized"].append(resource)
                continue
            
            # Get dimension value
            dimension_value = self._get_dimension_value(categorization, dimension_type)
            
            if dimension_value:
                grouped[dimension_value].append(resource)
            elif include_uncategorized:
                grouped["Uncategorized"].append(resource)
        
        return dict(grouped)
    
    def _get_dimension_value(
        self,
        categorization: ResourceCategorization,
        dimension_type: DimensionType
    ) -> Optional[str]:
        """Extract dimension value from categorization"""
        dimension_map = {
            DimensionType.TEAM: categorization.team,
            DimensionType.PROJECT: categorization.project,
            DimensionType.ENVIRONMENT: categorization.environment,
            DimensionType.REGION: categorization.region,
            DimensionType.COST_CENTER: categorization.cost_center,
            DimensionType.DEPARTMENT: categorization.custom_attributes.get('department')
        }
        
        return dimension_map.get(dimension_type)
    
    def _calculate_aggregations(
        self,
        resources: List[CloudResource],
        aggregation_types: List[AggregationType],
        custom_aggregations: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Calculate aggregations for a list of resources"""
        aggregations = {}
        
        # Standard aggregations
        for agg_type in aggregation_types:
            if agg_type == AggregationType.COUNT:
                aggregations['count'] = len(resources)
            elif agg_type == AggregationType.SUM:
                # Sum of estimated costs if available
                total_cost = sum(
                    getattr(r, 'estimated_cost', 0) for r in resources
                )
                aggregations['total_cost'] = total_cost
            elif agg_type == AggregationType.AVERAGE:
                if resources:
                    avg_cost = sum(
                        getattr(r, 'estimated_cost', 0) for r in resources
                    ) / len(resources)
                    aggregations['average_cost'] = avg_cost
        
        # Custom aggregations
        for agg_name, agg_func in custom_aggregations.items():
            try:
                aggregations[agg_name] = agg_func(resources)
            except Exception as e:
                logger.error(f"Error calculating custom aggregation {agg_name}: {e}")
                aggregations[agg_name] = None
        
        return aggregations
    
    def _build_nested_hierarchy(
        self,
        parent_node: DimensionalViewNode,
        remaining_dimensions: List[DimensionType],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        options: ViewGenerationOptions
    ) -> None:
        """Recursively build nested hierarchy for remaining dimensions"""
        if not remaining_dimensions:
            return
        
        next_dimension = remaining_dimensions[0]
        
        # Group parent's resources by next dimension
        grouped = self._group_resources_by_dimension(
            next_dimension,
            parent_node.resources,
            categorizations,
            options.include_uncategorized
        )
        
        # Create child nodes
        for dimension_value, resource_list in grouped.items():
            if not resource_list and not options.include_empty_nodes:
                continue
            
            child_node = DimensionalViewNode(
                dimension_value=dimension_value,
                dimension_type=next_dimension,
                resources=resource_list,
                resource_count=len(resource_list)
            )
            
            # Calculate aggregations
            if options.aggregations or options.custom_aggregations:
                child_node.aggregated_metrics = self._calculate_aggregations(
                    resource_list,
                    options.aggregations,
                    options.custom_aggregations
                )
            
            parent_node.children.append(child_node)
            
            # Recurse for remaining dimensions
            if len(remaining_dimensions) > 1:
                self._build_nested_hierarchy(
                    child_node,
                    remaining_dimensions[1:],
                    categorizations,
                    structure,
                    options
                )
