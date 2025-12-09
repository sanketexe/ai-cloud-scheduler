"""
Auto-Categorization Engine for Cloud Migration Advisor

This module provides automatic resource categorization based on naming patterns,
tags, and relationships between resources.

Requirements: 5.2
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .resource_discovery_engine import CloudResource, ResourceInventory
from .organizational_structure_manager import OrganizationalStructure


logger = logging.getLogger(__name__)


@dataclass
class CategorizationRule:
    """Rule for automatic categorization"""
    rule_id: str
    rule_type: str  # "naming_pattern", "tag_based", "relationship_based"
    pattern: Optional[str] = None  # Regex pattern for naming rules
    tag_key: Optional[str] = None  # Tag key for tag-based rules
    tag_value: Optional[str] = None  # Tag value for tag-based rules
    dimension_type: str = ""  # team, project, environment, region, cost_center
    dimension_value: str = ""  # The value to assign
    priority: int = 0  # Higher priority rules are applied first
    
    def matches(self, resource: CloudResource) -> bool:
        """Check if this rule matches the given resource"""
        if self.rule_type == "naming_pattern" and self.pattern:
            return bool(re.search(self.pattern, resource.resource_name, re.IGNORECASE))
        
        elif self.rule_type == "tag_based" and self.tag_key:
            if self.tag_key in resource.tags:
                if self.tag_value:
                    return resource.tags[self.tag_key] == self.tag_value
                return True
        
        return False


@dataclass
class ResourceCategorization:
    """Categorization result for a resource"""
    resource_id: str
    team: Optional[str] = None
    project: Optional[str] = None
    environment: Optional[str] = None
    region: Optional[str] = None
    cost_center: Optional[str] = None
    custom_attributes: Dict[str, str] = field(default_factory=dict)
    categorization_method: str = "auto"  # auto, manual, suggested
    confidence_score: float = 0.0  # 0.0 to 1.0
    applied_rules: List[str] = field(default_factory=list)
    
    def is_complete(self) -> bool:
        """Check if categorization has all required dimensions"""
        return all([
            self.team is not None,
            self.project is not None,
            self.environment is not None,
            self.region is not None
        ])


@dataclass
class CategorizedResources:
    """Collection of categorized resources"""
    categorizations: List[ResourceCategorization] = field(default_factory=list)
    fully_categorized_count: int = 0
    partially_categorized_count: int = 0
    uncategorized_count: int = 0
    
    def add_categorization(self, categorization: ResourceCategorization):
        """Add a categorization to the collection"""
        self.categorizations.append(categorization)
        
        if categorization.is_complete():
            self.fully_categorized_count += 1
        elif any([
            categorization.team,
            categorization.project,
            categorization.environment,
            categorization.region
        ]):
            self.partially_categorized_count += 1
        else:
            self.uncategorized_count += 1
    
    def get_categorization(self, resource_id: str) -> Optional[ResourceCategorization]:
        """Get categorization for a specific resource"""
        for cat in self.categorizations:
            if cat.resource_id == resource_id:
                return cat
        return None


class AutoCategorizationEngine:
    """
    Automatic resource categorization engine
    
    Requirements: 5.2
    """
    
    def __init__(self):
        """Initialize the auto-categorization engine"""
        self.categorization_rules: List[CategorizationRule] = []
        logger.info("Auto-Categorization Engine initialized")
    
    def add_rule(self, rule: CategorizationRule):
        """
        Add a categorization rule
        
        Args:
            rule: Categorization rule to add
        """
        logger.debug(f"Adding categorization rule: {rule.rule_id}")
        self.categorization_rules.append(rule)
        # Sort rules by priority (higher priority first)
        self.categorization_rules.sort(key=lambda r: r.priority, reverse=True)
    
    def categorize_resources(
        self,
        inventory: ResourceInventory,
        structure: OrganizationalStructure
    ) -> CategorizedResources:
        """
        Automatically categorize resources based on naming patterns and tags
        
        Args:
            inventory: Resource inventory to categorize
            structure: Organizational structure for validation
            
        Returns:
            CategorizedResources with categorization results
        """
        logger.info(f"Starting auto-categorization for {inventory.total_count} resources")
        
        categorized = CategorizedResources()
        
        for resource in inventory.resources:
            categorization = self._categorize_single_resource(resource, structure)
            categorized.add_categorization(categorization)
        
        logger.info(
            f"Categorization complete: {categorized.fully_categorized_count} fully categorized, "
            f"{categorized.partially_categorized_count} partially categorized, "
            f"{categorized.uncategorized_count} uncategorized"
        )
        
        return categorized
    
    def _categorize_single_resource(
        self,
        resource: CloudResource,
        structure: OrganizationalStructure
    ) -> ResourceCategorization:
        """
        Categorize a single resource
        
        Args:
            resource: Resource to categorize
            structure: Organizational structure for validation
            
        Returns:
            ResourceCategorization for the resource
        """
        categorization = ResourceCategorization(resource_id=resource.resource_id)
        
        # Apply naming pattern rules
        self._apply_naming_patterns(resource, categorization, structure)
        
        # Apply tag-based rules
        self._apply_tag_based_rules(resource, categorization, structure)
        
        # Apply relationship-based categorization
        self._apply_relationship_based_categorization(resource, categorization, structure)
        
        # Calculate confidence score
        categorization.confidence_score = self._calculate_confidence_score(categorization)
        
        return categorization
    
    def _apply_naming_patterns(
        self,
        resource: CloudResource,
        categorization: ResourceCategorization,
        structure: OrganizationalStructure
    ):
        """
        Apply naming pattern rules to categorize resource
        
        Args:
            resource: Resource to categorize
            categorization: Categorization to update
            structure: Organizational structure for validation
        """
        for rule in self.categorization_rules:
            if rule.rule_type != "naming_pattern":
                continue
            
            if rule.matches(resource):
                self._apply_rule_to_categorization(rule, categorization, structure)
                categorization.applied_rules.append(rule.rule_id)
    
    def _apply_tag_based_rules(
        self,
        resource: CloudResource,
        categorization: ResourceCategorization,
        structure: OrganizationalStructure
    ):
        """
        Apply tag-based rules to categorize resource
        
        Args:
            resource: Resource to categorize
            categorization: Categorization to update
            structure: Organizational structure for validation
        """
        # First, check existing tags on the resource
        if "team" in resource.tags and not categorization.team:
            team_name = resource.tags["team"]
            if self._validate_team(team_name, structure):
                categorization.team = team_name
                categorization.applied_rules.append("existing_tag:team")
        
        if "project" in resource.tags and not categorization.project:
            project_name = resource.tags["project"]
            if self._validate_project(project_name, structure):
                categorization.project = project_name
                categorization.applied_rules.append("existing_tag:project")
        
        if "environment" in resource.tags and not categorization.environment:
            env_name = resource.tags["environment"]
            if self._validate_environment(env_name, structure):
                categorization.environment = env_name
                categorization.applied_rules.append("existing_tag:environment")
        
        if "cost_center" in resource.tags and not categorization.cost_center:
            cc_name = resource.tags["cost_center"]
            if self._validate_cost_center(cc_name, structure):
                categorization.cost_center = cc_name
                categorization.applied_rules.append("existing_tag:cost_center")
        
        # Apply custom tag-based rules
        for rule in self.categorization_rules:
            if rule.rule_type != "tag_based":
                continue
            
            if rule.matches(resource):
                self._apply_rule_to_categorization(rule, categorization, structure)
                categorization.applied_rules.append(rule.rule_id)
    
    def _apply_relationship_based_categorization(
        self,
        resource: CloudResource,
        categorization: ResourceCategorization,
        structure: OrganizationalStructure
    ):
        """
        Apply relationship-based categorization
        
        This looks at resource relationships (e.g., VPC, subnet) to infer categorization
        
        Args:
            resource: Resource to categorize
            categorization: Categorization to update
            structure: Organizational structure for validation
        """
        # Check if resource has parent/related resources in metadata
        if "parent_resource_id" in resource.metadata:
            # In a real implementation, we would look up the parent resource
            # and inherit its categorization
            pass
        
        # Check if resource is in a specific VPC or network
        if "vpc_id" in resource.metadata:
            # Could infer team/project from VPC naming or tags
            pass
        
        # Use region from resource to set region dimension
        if resource.region and not categorization.region:
            # Validate region exists in structure
            for region in structure.regions:
                if region.cloud_provider_region == resource.region:
                    categorization.region = region.name
                    categorization.applied_rules.append("relationship:region")
                    break
    
    def _apply_rule_to_categorization(
        self,
        rule: CategorizationRule,
        categorization: ResourceCategorization,
        structure: OrganizationalStructure
    ):
        """
        Apply a rule to update categorization
        
        Args:
            rule: Rule to apply
            categorization: Categorization to update
            structure: Organizational structure for validation
        """
        dimension_type = rule.dimension_type
        dimension_value = rule.dimension_value
        
        # Only apply if dimension not already set
        if dimension_type == "team" and not categorization.team:
            if self._validate_team(dimension_value, structure):
                categorization.team = dimension_value
        
        elif dimension_type == "project" and not categorization.project:
            if self._validate_project(dimension_value, structure):
                categorization.project = dimension_value
        
        elif dimension_type == "environment" and not categorization.environment:
            if self._validate_environment(dimension_value, structure):
                categorization.environment = dimension_value
        
        elif dimension_type == "region" and not categorization.region:
            if self._validate_region(dimension_value, structure):
                categorization.region = dimension_value
        
        elif dimension_type == "cost_center" and not categorization.cost_center:
            if self._validate_cost_center(dimension_value, structure):
                categorization.cost_center = dimension_value
    
    def _validate_team(self, team_name: str, structure: OrganizationalStructure) -> bool:
        """Validate team exists in structure"""
        for team in structure.teams:
            if team.name == team_name or team.team_id == team_name:
                return True
        return False
    
    def _validate_project(self, project_name: str, structure: OrganizationalStructure) -> bool:
        """Validate project exists in structure"""
        for project in structure.projects:
            if project.name == project_name or project.project_id == project_name:
                return True
        return False
    
    def _validate_environment(self, env_name: str, structure: OrganizationalStructure) -> bool:
        """Validate environment exists in structure"""
        for env in structure.environments:
            if env.name == env_name or env.environment_id == env_name:
                return True
        return False
    
    def _validate_region(self, region_name: str, structure: OrganizationalStructure) -> bool:
        """Validate region exists in structure"""
        for region in structure.regions:
            if region.name == region_name or region.region_id == region_name:
                return True
        return False
    
    def _validate_cost_center(self, cc_name: str, structure: OrganizationalStructure) -> bool:
        """Validate cost center exists in structure"""
        for cc in structure.cost_centers:
            if cc.name == cc_name or cc.cost_center_id == cc_name or cc.code == cc_name:
                return True
        return False
    
    def _calculate_confidence_score(self, categorization: ResourceCategorization) -> float:
        """
        Calculate confidence score for categorization
        
        Args:
            categorization: Categorization to score
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base score on number of dimensions filled
        dimensions_filled = sum([
            1 if categorization.team else 0,
            1 if categorization.project else 0,
            1 if categorization.environment else 0,
            1 if categorization.region else 0,
            1 if categorization.cost_center else 0
        ])
        
        base_score = dimensions_filled / 5.0
        
        # Boost score if multiple rules were applied
        rule_boost = min(len(categorization.applied_rules) * 0.1, 0.3)
        
        confidence = min(base_score + rule_boost, 1.0)
        
        return confidence
    
    def create_naming_pattern_rule(
        self,
        rule_id: str,
        pattern: str,
        dimension_type: str,
        dimension_value: str,
        priority: int = 0
    ) -> CategorizationRule:
        """
        Create a naming pattern rule
        
        Args:
            rule_id: Unique rule identifier
            pattern: Regex pattern to match resource names
            dimension_type: Type of dimension (team, project, etc.)
            dimension_value: Value to assign
            priority: Rule priority
            
        Returns:
            CategorizationRule
        """
        return CategorizationRule(
            rule_id=rule_id,
            rule_type="naming_pattern",
            pattern=pattern,
            dimension_type=dimension_type,
            dimension_value=dimension_value,
            priority=priority
        )
    
    def create_tag_based_rule(
        self,
        rule_id: str,
        tag_key: str,
        tag_value: Optional[str],
        dimension_type: str,
        dimension_value: str,
        priority: int = 0
    ) -> CategorizationRule:
        """
        Create a tag-based rule
        
        Args:
            rule_id: Unique rule identifier
            tag_key: Tag key to match
            tag_value: Tag value to match (optional)
            dimension_type: Type of dimension (team, project, etc.)
            dimension_value: Value to assign
            priority: Rule priority
            
        Returns:
            CategorizationRule
        """
        return CategorizationRule(
            rule_id=rule_id,
            rule_type="tag_based",
            tag_key=tag_key,
            tag_value=tag_value,
            dimension_type=dimension_type,
            dimension_value=dimension_value,
            priority=priority
        )
    
    def suggest_categorization_rules(
        self,
        inventory: ResourceInventory,
        structure: OrganizationalStructure
    ) -> List[CategorizationRule]:
        """
        Analyze resources and suggest categorization rules
        
        Args:
            inventory: Resource inventory to analyze
            structure: Organizational structure
            
        Returns:
            List of suggested categorization rules
        """
        logger.info("Analyzing resources to suggest categorization rules")
        
        suggested_rules = []
        
        # Analyze naming patterns
        naming_patterns = self._analyze_naming_patterns(inventory, structure)
        suggested_rules.extend(naming_patterns)
        
        # Analyze common tags
        tag_patterns = self._analyze_tag_patterns(inventory, structure)
        suggested_rules.extend(tag_patterns)
        
        logger.info(f"Generated {len(suggested_rules)} suggested categorization rules")
        
        return suggested_rules
    
    def _analyze_naming_patterns(
        self,
        inventory: ResourceInventory,
        structure: OrganizationalStructure
    ) -> List[CategorizationRule]:
        """Analyze resource names to suggest naming pattern rules"""
        suggested_rules = []
        
        # Look for common prefixes/suffixes that match team names
        for team in structure.teams:
            team_pattern = f".*{re.escape(team.name.lower())}.*"
            matching_count = sum(
                1 for r in inventory.resources
                if re.search(team_pattern, r.resource_name.lower())
            )
            
            if matching_count > 0:
                suggested_rules.append(
                    CategorizationRule(
                        rule_id=f"auto_team_{team.team_id}",
                        rule_type="naming_pattern",
                        pattern=team_pattern,
                        dimension_type="team",
                        dimension_value=team.name,
                        priority=5
                    )
                )
        
        # Look for environment indicators (dev, staging, prod)
        env_keywords = ["dev", "development", "staging", "stage", "prod", "production", "test"]
        for env in structure.environments:
            for keyword in env_keywords:
                if keyword in env.name.lower():
                    env_pattern = f".*{keyword}.*"
                    matching_count = sum(
                        1 for r in inventory.resources
                        if re.search(env_pattern, r.resource_name.lower())
                    )
                    
                    if matching_count > 0:
                        suggested_rules.append(
                            CategorizationRule(
                                rule_id=f"auto_env_{env.environment_id}_{keyword}",
                                rule_type="naming_pattern",
                                pattern=env_pattern,
                                dimension_type="environment",
                                dimension_value=env.name,
                                priority=5
                            )
                        )
        
        return suggested_rules
    
    def _analyze_tag_patterns(
        self,
        inventory: ResourceInventory,
        structure: OrganizationalStructure
    ) -> List[CategorizationRule]:
        """Analyze resource tags to suggest tag-based rules"""
        suggested_rules = []
        
        # Collect all tag keys used
        tag_keys = set()
        for resource in inventory.resources:
            tag_keys.update(resource.tags.keys())
        
        # Look for common organizational tag keys
        org_tag_keys = ["team", "project", "environment", "cost-center", "owner", "department"]
        
        for tag_key in tag_keys:
            if tag_key.lower() in org_tag_keys:
                # This is a potentially useful organizational tag
                # Count how many resources have this tag
                tagged_count = sum(1 for r in inventory.resources if tag_key in r.tags)
                
                if tagged_count > 0:
                    logger.debug(f"Found organizational tag: {tag_key} on {tagged_count} resources")
        
        return suggested_rules
