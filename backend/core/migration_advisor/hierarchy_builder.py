"""
Hierarchy Builder for Cloud Migration Advisor

This module builds hierarchical views of resources grouped by organizational dimensions
and calculates aggregated metrics.

Requirements: 5.6
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .resource_discovery_engine import CloudResource, ResourceType
from .auto_categorization_engine import CategorizedResources, ResourceCategorization
from .organizational_structure_manager import DimensionType


logger = logging.getLogger(__name__)


@dataclass
class HierarchyNode:
    """Represents a node in the resource hierarchy"""
    node_id: str
    node_name: str
    node_type: str  # dimension type or "resource"
    level: int
    parent_id: Optional[str] = None
    children: List['HierarchyNode'] = field(default_factory=list)
    resources: List[CloudResource] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'HierarchyNode'):
        """Add a child node"""
        child.parent_id = self.node_id
        child.level = self.level + 1
        self.children.append(child)
    
    def add_resource(self, resource: CloudResource):
        """Add a resource to this node"""
        self.resources.append(resource)
    
    def get_total_resources(self) -> int:
        """Get total number of resources in this node and all children"""
        total = len(self.resources)
        for child in self.children:
            total += child.get_total_resources()
        return total
    
    def get_resources_by_type(self) -> Dict[ResourceType, int]:
        """Get count of resources by type"""
        counts: Dict[ResourceType, int] = {}
        
        for resource in self.resources:
            if resource.resource_type not in counts:
                counts[resource.resource_type] = 0
            counts[resource.resource_type] += 1
        
        # Aggregate from children
        for child in self.children:
            child_counts = child.get_resources_by_type()
            for resource_type, count in child_counts.items():
                if resource_type not in counts:
                    counts[resource_type] = 0
                counts[resource_type] += count
        
        return counts


@dataclass
class HierarchyView:
    """
    Complete hierarchical view of resources
    
    Requirements: 5.6
    """
    view_id: str
    root_dimension: DimensionType
    root_node: HierarchyNode
    total_resources: int = 0
    total_cost: float = 0.0
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Ensure root_dimension is enum"""
        if isinstance(self.root_dimension, str):
            self.root_dimension = DimensionType(self.root_dimension)
    
    def get_node_by_id(self, node_id: str) -> Optional[HierarchyNode]:
        """Find a node by ID in the hierarchy"""
        return self._search_node(self.root_node, node_id)
    
    def _search_node(self, node: HierarchyNode, node_id: str) -> Optional[HierarchyNode]:
        """Recursively search for a node"""
        if node.node_id == node_id:
            return node
        
        for child in node.children:
            result = self._search_node(child, node_id)
            if result:
                return result
        
        return None
    
    def get_all_nodes(self) -> List[HierarchyNode]:
        """Get all nodes in the hierarchy"""
        nodes = []
        self._collect_nodes(self.root_node, nodes)
        return nodes
    
    def _collect_nodes(self, node: HierarchyNode, nodes: List[HierarchyNode]):
        """Recursively collect all nodes"""
        nodes.append(node)
        for child in node.children:
            self._collect_nodes(child, nodes)


class HierarchyBuilder:
    """
    Builds hierarchical views of resources
    
    Requirements: 5.6
    """
    
    def __init__(self):
        """Initialize the hierarchy builder"""
        logger.info("Hierarchy Builder initialized")
    
    def build_hierarchy(
        self,
        dimension: DimensionType,
        resources: List[CloudResource],
        categorizations: CategorizedResources
    ) -> HierarchyView:
        """
        Build a hierarchical view of resources grouped by dimension
        
        Args:
            dimension: Root dimension for the hierarchy
            resources: List of resources to organize
            categorizations: Categorizations for the resources
            
        Returns:
            HierarchyView with hierarchical organization
        """
        logger.info(f"Building hierarchy view for dimension: {dimension.value}")
        
        # Create root node
        root_node = HierarchyNode(
            node_id="root",
            node_name=f"All Resources by {dimension.value}",
            node_type="root",
            level=0
        )
        
        # Group resources by the specified dimension
        grouped_resources = self._group_by_dimension(
            dimension,
            resources,
            categorizations
        )
        
        # Build hierarchy structure
        for group_name, group_resources in grouped_resources.items():
            group_node = HierarchyNode(
                node_id=f"{dimension.value}_{group_name}",
                node_name=group_name,
                node_type=dimension.value,
                level=1
            )
            
            # Add resources to group node
            for resource in group_resources:
                group_node.add_resource(resource)
            
            # Calculate metrics for this group
            group_node.metrics = self._calculate_node_metrics(group_node)
            
            root_node.add_child(group_node)
        
        # Calculate root metrics
        root_node.metrics = self._calculate_node_metrics(root_node)
        
        # Create hierarchy view
        view = HierarchyView(
            view_id=f"hierarchy_{dimension.value}_{datetime.utcnow().timestamp()}",
            root_dimension=dimension,
            root_node=root_node,
            total_resources=root_node.get_total_resources()
        )
        
        # Calculate aggregated metrics
        view.aggregated_metrics = self._calculate_aggregated_metrics(view)
        
        logger.info(
            f"Hierarchy built: {len(root_node.children)} groups, "
            f"{view.total_resources} total resources"
        )
        
        return view
    
    def build_multi_level_hierarchy(
        self,
        dimensions: List[DimensionType],
        resources: List[CloudResource],
        categorizations: CategorizedResources
    ) -> HierarchyView:
        """
        Build a multi-level hierarchical view
        
        Args:
            dimensions: List of dimensions in hierarchy order (top to bottom)
            resources: List of resources to organize
            categorizations: Categorizations for the resources
            
        Returns:
            HierarchyView with multi-level hierarchy
        """
        if not dimensions:
            raise ValueError("At least one dimension is required")
        
        logger.info(f"Building multi-level hierarchy with {len(dimensions)} levels")
        
        # Create root node
        root_dimension = dimensions[0]
        root_node = HierarchyNode(
            node_id="root",
            node_name=f"All Resources",
            node_type="root",
            level=0
        )
        
        # Build hierarchy recursively
        self._build_hierarchy_level(
            root_node,
            dimensions,
            0,
            resources,
            categorizations
        )
        
        # Calculate metrics for all nodes
        self._calculate_all_metrics(root_node)
        
        # Create hierarchy view
        view = HierarchyView(
            view_id=f"hierarchy_multi_{datetime.utcnow().timestamp()}",
            root_dimension=root_dimension,
            root_node=root_node,
            total_resources=root_node.get_total_resources()
        )
        
        # Calculate aggregated metrics
        view.aggregated_metrics = self._calculate_aggregated_metrics(view)
        
        logger.info(f"Multi-level hierarchy built: {view.total_resources} total resources")
        
        return view
    
    def _build_hierarchy_level(
        self,
        parent_node: HierarchyNode,
        dimensions: List[DimensionType],
        level_index: int,
        resources: List[CloudResource],
        categorizations: CategorizedResources
    ):
        """
        Recursively build hierarchy levels
        
        Args:
            parent_node: Parent node to add children to
            dimensions: List of dimensions
            level_index: Current level index
            resources: Resources to organize
            categorizations: Categorizations for resources
        """
        if level_index >= len(dimensions):
            # Leaf level - add resources directly
            for resource in resources:
                parent_node.add_resource(resource)
            return
        
        current_dimension = dimensions[level_index]
        
        # Group resources by current dimension
        grouped_resources = self._group_by_dimension(
            current_dimension,
            resources,
            categorizations
        )
        
        # Create child nodes for each group
        for group_name, group_resources in grouped_resources.items():
            child_node = HierarchyNode(
                node_id=f"{parent_node.node_id}_{current_dimension.value}_{group_name}",
                node_name=group_name,
                node_type=current_dimension.value,
                level=parent_node.level + 1
            )
            
            # Recursively build next level
            self._build_hierarchy_level(
                child_node,
                dimensions,
                level_index + 1,
                group_resources,
                categorizations
            )
            
            parent_node.add_child(child_node)
    
    def _group_by_dimension(
        self,
        dimension: DimensionType,
        resources: List[CloudResource],
        categorizations: CategorizedResources
    ) -> Dict[str, List[CloudResource]]:
        """
        Group resources by a specific dimension
        
        Args:
            dimension: Dimension to group by
            resources: Resources to group
            categorizations: Categorizations for resources
            
        Returns:
            Dictionary mapping dimension values to resources
        """
        grouped: Dict[str, List[CloudResource]] = {}
        
        for resource in resources:
            categorization = categorizations.get_categorization(resource.resource_id)
            
            if not categorization:
                # Uncategorized resources go to "Unassigned" group
                group_name = "Unassigned"
            else:
                # Get dimension value from categorization
                if dimension == DimensionType.TEAM:
                    group_name = categorization.team or "Unassigned"
                elif dimension == DimensionType.PROJECT:
                    group_name = categorization.project or "Unassigned"
                elif dimension == DimensionType.ENVIRONMENT:
                    group_name = categorization.environment or "Unassigned"
                elif dimension == DimensionType.REGION:
                    group_name = categorization.region or "Unassigned"
                elif dimension == DimensionType.COST_CENTER:
                    group_name = categorization.cost_center or "Unassigned"
                else:
                    group_name = "Unassigned"
            
            if group_name not in grouped:
                grouped[group_name] = []
            
            grouped[group_name].append(resource)
        
        return grouped
    
    def _calculate_node_metrics(self, node: HierarchyNode) -> Dict[str, Any]:
        """
        Calculate metrics for a node
        
        Args:
            node: Node to calculate metrics for
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "resource_count": node.get_total_resources(),
            "direct_resource_count": len(node.resources),
            "child_count": len(node.children),
            "resources_by_type": node.get_resources_by_type()
        }
        
        # Calculate estimated cost (placeholder - would use actual cost data)
        # In a real implementation, this would aggregate cost from resource metadata
        metrics["estimated_monthly_cost"] = 0.0
        
        return metrics
    
    def _calculate_all_metrics(self, node: HierarchyNode):
        """
        Recursively calculate metrics for all nodes
        
        Args:
            node: Root node to start from
        """
        # Calculate metrics for children first
        for child in node.children:
            self._calculate_all_metrics(child)
        
        # Calculate metrics for this node
        node.metrics = self._calculate_node_metrics(node)
    
    def _calculate_aggregated_metrics(self, view: HierarchyView) -> Dict[str, Any]:
        """
        Calculate aggregated metrics for the entire hierarchy
        
        Args:
            view: Hierarchy view
            
        Returns:
            Dictionary of aggregated metrics
        """
        all_nodes = view.get_all_nodes()
        
        metrics = {
            "total_nodes": len(all_nodes),
            "total_resources": view.total_resources,
            "max_depth": max(node.level for node in all_nodes) if all_nodes else 0,
            "resources_by_type": view.root_node.get_resources_by_type(),
            "average_resources_per_node": (
                view.total_resources / len(all_nodes) if all_nodes else 0
            )
        }
        
        return metrics
    
    def flatten_hierarchy(self, view: HierarchyView) -> List[Dict[str, Any]]:
        """
        Flatten hierarchy into a list of dictionaries for reporting
        
        Args:
            view: Hierarchy view to flatten
            
        Returns:
            List of node dictionaries
        """
        flattened = []
        self._flatten_node(view.root_node, flattened, "")
        return flattened
    
    def _flatten_node(
        self,
        node: HierarchyNode,
        flattened: List[Dict[str, Any]],
        path: str
    ):
        """
        Recursively flatten a node
        
        Args:
            node: Node to flatten
            flattened: List to append to
            path: Current path in hierarchy
        """
        current_path = f"{path}/{node.node_name}" if path else node.node_name
        
        flattened.append({
            "node_id": node.node_id,
            "node_name": node.node_name,
            "node_type": node.node_type,
            "level": node.level,
            "path": current_path,
            "resource_count": len(node.resources),
            "total_resource_count": node.get_total_resources(),
            "child_count": len(node.children),
            "metrics": node.metrics
        })
        
        for child in node.children:
            self._flatten_node(child, flattened, current_path)
    
    def export_hierarchy_json(self, view: HierarchyView) -> Dict[str, Any]:
        """
        Export hierarchy as JSON-serializable dictionary
        
        Args:
            view: Hierarchy view to export
            
        Returns:
            Dictionary representation of hierarchy
        """
        return {
            "view_id": view.view_id,
            "root_dimension": view.root_dimension.value,
            "total_resources": view.total_resources,
            "aggregated_metrics": view.aggregated_metrics,
            "created_at": view.created_at.isoformat(),
            "hierarchy": self._export_node(view.root_node)
        }
    
    def _export_node(self, node: HierarchyNode) -> Dict[str, Any]:
        """
        Export a node as dictionary
        
        Args:
            node: Node to export
            
        Returns:
            Dictionary representation of node
        """
        return {
            "node_id": node.node_id,
            "node_name": node.node_name,
            "node_type": node.node_type,
            "level": node.level,
            "resource_count": len(node.resources),
            "total_resource_count": node.get_total_resources(),
            "metrics": node.metrics,
            "children": [self._export_node(child) for child in node.children],
            "resources": [
                {
                    "resource_id": r.resource_id,
                    "resource_name": r.resource_name,
                    "resource_type": r.resource_type.value,
                    "provider": r.provider.value,
                    "region": r.region
                }
                for r in node.resources
            ]
        }
