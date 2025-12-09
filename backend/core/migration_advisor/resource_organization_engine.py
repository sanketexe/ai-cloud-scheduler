"""
Resource Organization Engine for Cloud Migration Advisor

This is the main engine that coordinates resource discovery, categorization,
tagging, hierarchy building, and ownership resolution.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .resource_discovery_engine import (
    ResourceDiscoveryEngine,
    CloudProvider,
    ProviderCredentials,
    ResourceInventory,
    CloudResource
)
from .organizational_structure_manager import (
    OrganizationalStructureManager,
    OrganizationalStructure,
    DimensionType
)
from .auto_categorization_engine import (
    AutoCategorizationEngine,
    CategorizedResources,
    CategorizationRule
)
from .tagging_engine import (
    TaggingEngine,
    TaggingPolicy,
    BulkTaggingResult
)
from .hierarchy_builder import (
    HierarchyBuilder,
    HierarchyView
)
from .ownership_resolver import (
    OwnershipResolver,
    OwnershipResolutionResult
)


logger = logging.getLogger(__name__)


@dataclass
class OrganizationResult:
    """Complete result of resource organization process"""
    project_id: str
    inventory: ResourceInventory
    categorizations: CategorizedResources
    tagging_result: BulkTaggingResult
    hierarchy_views: Dict[str, HierarchyView] = field(default_factory=dict)
    ownership_result: Optional[OwnershipResolutionResult] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)


class ResourceOrganizationEngine:
    """
    Main resource organization engine that coordinates all organization activities
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    
    def __init__(self):
        """Initialize the resource organization engine"""
        self.discovery_engine = ResourceDiscoveryEngine()
        self.structure_manager = OrganizationalStructureManager()
        self.categorization_engine = AutoCategorizationEngine()
        self.hierarchy_builder = HierarchyBuilder()
        self.ownership_resolver = OwnershipResolver()
        
        logger.info("Resource Organization Engine initialized")
    
    def discover_and_organize_resources(
        self,
        project_id: str,
        provider: CloudProvider,
        credentials: ProviderCredentials,
        structure: OrganizationalStructure,
        tagging_policy: TaggingPolicy,
        categorization_rules: Optional[List[CategorizationRule]] = None,
        auto_assign_ownership: bool = False
    ) -> OrganizationResult:
        """
        Complete workflow: discover, categorize, tag, and organize resources
        
        Args:
            project_id: Migration project ID
            provider: Cloud provider to discover from
            credentials: Provider credentials
            structure: Organizational structure
            tagging_policy: Tagging policy to use
            categorization_rules: Optional categorization rules
            auto_assign_ownership: Whether to auto-assign ownership
            
        Returns:
            OrganizationResult with complete organization results
        """
        logger.info(f"Starting complete resource organization for project {project_id}")
        
        # Step 1: Discover resources
        logger.info("Step 1: Discovering resources")
        inventory = self.discovery_engine.discover_resources(provider, credentials)
        
        # Step 2: Add categorization rules if provided
        if categorization_rules:
            logger.info(f"Adding {len(categorization_rules)} categorization rules")
            for rule in categorization_rules:
                self.categorization_engine.add_rule(rule)
        
        # Step 3: Categorize resources
        logger.info("Step 2: Categorizing resources")
        categorizations = self.categorization_engine.categorize_resources(
            inventory,
            structure
        )
        
        # Step 4: Tag resources
        logger.info("Step 3: Tagging resources")
        tagging_engine = TaggingEngine(tagging_policy)
        tagging_result = tagging_engine.tag_resources(
            inventory.resources,
            categorizations
        )
        
        # Step 5: Build hierarchy views
        logger.info("Step 4: Building hierarchy views")
        hierarchy_views = {}
        
        # Build view for each dimension
        for dimension in [DimensionType.TEAM, DimensionType.PROJECT, 
                         DimensionType.ENVIRONMENT, DimensionType.REGION]:
            try:
                view = self.hierarchy_builder.build_hierarchy(
                    dimension,
                    inventory.resources,
                    categorizations
                )
                hierarchy_views[dimension.value] = view
            except Exception as e:
                logger.error(f"Failed to build hierarchy for {dimension.value}: {e}")
        
        # Step 6: Resolve ownership
        logger.info("Step 5: Resolving ownership")
        ownership_result = self.ownership_resolver.resolve_ownership(
            inventory,
            categorizations,
            structure,
            auto_assign_high_confidence=auto_assign_ownership,
            confidence_threshold=0.8
        )
        
        # Create result
        result = OrganizationResult(
            project_id=project_id,
            inventory=inventory,
            categorizations=categorizations,
            tagging_result=tagging_result,
            hierarchy_views=hierarchy_views,
            ownership_result=ownership_result
        )
        
        logger.info(
            f"Resource organization complete: {inventory.total_count} resources discovered, "
            f"{categorizations.fully_categorized_count} fully categorized, "
            f"{tagging_result.successful} tagged successfully"
        )
        
        return result
    
    def define_organizational_structure(
        self,
        project_id: str,
        structure_name: str
    ) -> OrganizationalStructure:
        """
        Create a new organizational structure
        
        Args:
            project_id: Migration project ID
            structure_name: Name for the structure
            
        Returns:
            New OrganizationalStructure
        """
        logger.info(f"Creating organizational structure for project {project_id}")
        
        structure = self.structure_manager.create_structure(
            structure_id=f"struct_{project_id}",
            name=structure_name
        )
        
        return structure
    
    def categorize_resources(
        self,
        inventory: ResourceInventory,
        structure: OrganizationalStructure,
        rules: Optional[List[CategorizationRule]] = None
    ) -> CategorizedResources:
        """
        Categorize resources using the auto-categorization engine
        
        Args:
            inventory: Resource inventory
            structure: Organizational structure
            rules: Optional categorization rules
            
        Returns:
            CategorizedResources
        """
        logger.info("Categorizing resources")
        
        # Add rules if provided
        if rules:
            for rule in rules:
                self.categorization_engine.add_rule(rule)
        
        categorizations = self.categorization_engine.categorize_resources(
            inventory,
            structure
        )
        
        return categorizations
    
    def apply_tags(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        policy: TaggingPolicy
    ) -> BulkTaggingResult:
        """
        Apply tags to resources based on categorizations
        
        Args:
            resources: Resources to tag
            categorizations: Resource categorizations
            policy: Tagging policy
            
        Returns:
            BulkTaggingResult
        """
        logger.info("Applying tags to resources")
        
        tagging_engine = TaggingEngine(policy)
        result = tagging_engine.tag_resources(resources, categorizations)
        
        return result
    
    def build_hierarchy_view(
        self,
        dimension: DimensionType,
        resources: List[CloudResource],
        categorizations: CategorizedResources
    ) -> HierarchyView:
        """
        Build a hierarchical view of resources
        
        Args:
            dimension: Dimension to organize by
            resources: Resources to organize
            categorizations: Resource categorizations
            
        Returns:
            HierarchyView
        """
        logger.info(f"Building hierarchy view for dimension: {dimension.value}")
        
        view = self.hierarchy_builder.build_hierarchy(
            dimension,
            resources,
            categorizations
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
            dimensions: Dimensions in hierarchy order
            resources: Resources to organize
            categorizations: Resource categorizations
            
        Returns:
            HierarchyView
        """
        logger.info(f"Building multi-level hierarchy with {len(dimensions)} levels")
        
        view = self.hierarchy_builder.build_multi_level_hierarchy(
            dimensions,
            resources,
            categorizations
        )
        
        return view
    
    def resolve_ownership(
        self,
        inventory: ResourceInventory,
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        auto_assign: bool = False
    ) -> OwnershipResolutionResult:
        """
        Resolve resource ownership
        
        Args:
            inventory: Resource inventory
            categorizations: Resource categorizations
            structure: Organizational structure
            auto_assign: Whether to auto-assign high confidence suggestions
            
        Returns:
            OwnershipResolutionResult
        """
        logger.info("Resolving resource ownership")
        
        result = self.ownership_resolver.resolve_ownership(
            inventory,
            categorizations,
            structure,
            auto_assign_high_confidence=auto_assign
        )
        
        return result
    
    def get_organization_summary(
        self,
        result: OrganizationResult
    ) -> Dict[str, Any]:
        """
        Get summary of organization results
        
        Args:
            result: Organization result
            
        Returns:
            Dictionary with summary statistics
        """
        tagging_summary = {
            "total_resources": result.tagging_result.total_resources,
            "successful": result.tagging_result.successful,
            "failed": result.tagging_result.failed
        }
        
        ownership_summary = None
        if result.ownership_result:
            ownership_summary = self.ownership_resolver.get_ownership_summary(
                result.ownership_result
            )
        
        return {
            "project_id": result.project_id,
            "completed_at": result.completed_at.isoformat(),
            "discovery": {
                "total_resources": result.inventory.total_count,
                "resources_by_type": {
                    rt.value: count 
                    for rt, count in result.inventory.resources_by_type.items()
                },
                "resources_by_region": result.inventory.resources_by_region
            },
            "categorization": {
                "fully_categorized": result.categorizations.fully_categorized_count,
                "partially_categorized": result.categorizations.partially_categorized_count,
                "uncategorized": result.categorizations.uncategorized_count
            },
            "tagging": tagging_summary,
            "hierarchy_views": {
                name: {
                    "total_resources": view.total_resources,
                    "total_nodes": len(view.get_all_nodes())
                }
                for name, view in result.hierarchy_views.items()
            },
            "ownership": ownership_summary
        }
